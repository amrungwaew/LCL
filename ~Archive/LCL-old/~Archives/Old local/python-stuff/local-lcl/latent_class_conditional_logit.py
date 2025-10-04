"""Estimation and prediction for latent-class conditional logit."""

from functools import partial
from time import time

import jax.numpy as jnp
import numpy as onp
import polars as pl
from jax import jit, lax
from jax.ops import segment_sum
from jax.random import key, permutation
from lcl._choice_model import ChoiceModel
from lcl._demographics import (
    _compute_grouped_data_loglik_and_grad,
    _predict_class_membership_probs,
)
from lcl._em_alg_steps import (
    _compute_conditional_class_probs,
    _compute_probs_and_exp_utility,
    _em_alg,
)
from lcl._optimize import _minimize_bfgs
from lcl.conditional_logit import _loglik_gradient
from lcl.utils import _ensure_sequential

"""
Notation
---------
    N : Number of choice situations
    J : Number of alternatives
    K : Number of variables
"""


class LatentClassConditionalLogit(ChoiceModel):
    """Class for estimation of latent-class conditional logit models.

    Attributes
    ----------
        coeff_ : numpy array, shape (n_variables)
            Estimated coefficients

        coeff_names : numpy array, shape (n_variables)
            Names of the estimated coefficients

        stderr : numpy array, shape (n_variables)
            Standard errors of the estimated coefficients

        zvalues : numpy array, shape (n_variables)
            Z-values for t-distribution of the estimated coefficients

        pvalues : numpy array, shape (n_variables)
            P-values of the estimated coefficients

        loglikelihood : float
            Log-likelihood at the end of the estimation

        convergence : bool
            Whether convergence was reached during estimation

        total_iter : int
            Total number of iterations executed during estimation

        estim_time_sec : float
            Estimation time in seconds

        sample_size : int
            Number of samples used for estimation

        aic : float
            Akaike information criteria of the estimated model

        bic : float
            Bayesian information criteria of the estimated model
    """

    def fit(
        self,
        X,
        y,
        varnames,
        alts,
        ids,
        panels,
        num_classes,
        dems=None,
        dem_varnames=None,
        jax_prng_seed=0,
        maxiter=2000,
        em_loglik_tol=1e-4,
        em_maxiter=2000,
    ):
        """Fit multinomial and/or conditional logit models.

        Parameters
        ----------
        X : array-like, shape (n_samples*n_alts, n_variables)
            Input data for explanatory variables in long format
        y : array-like, shape (n_samples*n_alts,)
            Chosen alternatives or one-hot encoded representation
            of the choices
        varnames : list, shape (n_variables,)
            Names of explanatory variables that must match the number and
            order of columns in `X`
        alts : array-like, shape (n_samples*n_alts,)
            Alternative values in long format
        ids : array-like, shape (n_samples*n_alts,)
            Identifiers for the samples in long format.
        num_classes : int
            Number of latent classes.
        dems : array, optional
            (Np, D) matrix of panel demographic characteristics.
        dem_varnames : list, shape (num_dem_vars,)
        jax_prng_seed : int
            Seed for pseud-random number generator.
        maxiter : int, default=200
            Maximum number of iterations for conditional logit estimation
        tol_opts : dict, default=None
            Options for tolerance of optimization routine. The dictionary accepts the following options (keys):
                ftol : float, default=1e-10
                    Tolerance for objective function (log-likelihood)

        Returns
        -------
        None.

        """
        varnames, X, y, alts, ids, panels, dems = self._as_array(
            varnames, X, y, alts, ids, panels, dems
        )
        self._validate_inputs(X, y, alts, varnames)
        self._pre_fit(alts, varnames, maxiter)

        self.num_classes, self.num_vars = num_classes, len(varnames)
        self.num_dem_vars = len(dem_varnames) if dem_varnames is not None else 0
        num_panels = jnp.unique(panels).shape[0]

        print(f"dem_varnames : {dem_varnames}")
        print(f"dems.shape : {dems.shape}")

        y, Xnames, ids, panels, dems, num_ids = self._setup_input_data(
            y, ids, panels, dems, predict_mode=False
        )

        # Define Xd as Xij - Xi*
        Xd, alts, ids, panels, panels_of_ids = self._diff_nonchosen_chosen(
            X, y, alts, ids, panels
        )
        num_choices_per_panel = self._count_choices_per_panel(panels, ids)

        # Set up demographics grouped-data regression
        if dems is not None:
            _class_membership_probs_fn = jit(
                partial(
                    _predict_class_membership_probs,
                    num_dem_vars=self.num_dem_vars,
                    num_classes=self.num_classes,
                ),
                static_argnames=["return_grad_components"],
            )
            _grouped_data_loglik_fn = jit(
                partial(
                    _compute_grouped_data_loglik_and_grad,
                    _class_membership_probs_fn=_class_membership_probs_fn,
                    num_dem_vars=self.num_dem_vars,
                    num_classes=self.num_classes,
                ),
                static_argnames=["return_gradient"],
            )

        # Obtain starting class shares and coefficients
        self.betas, self.thetas, self.shares = self._get_starting_vals(
            Xd,
            alts,
            ids,
            panels,
            panels_of_ids,
            num_choices_per_panel,
            num_ids,
            num_panels,
            jax_prng_seed,
        )

        logliks_list, em_recursion = [], 0
        while em_recursion < em_maxiter:
            print(f"EM recursion: {em_recursion}")
            (
                self.betas,
                self.thetas,
                self.shares,
                self.loglik,
                class_probs_by_panel,
            ) = _em_alg(
                self.betas,
                Xd,
                ids,
                panels_of_ids,
                num_choices_per_panel,
                self.num_vars,
                num_classes,
                num_ids,
                num_panels,
                self.thetas,
                dems,
                self.num_dem_vars,
                _grouped_data_loglik_fn,
                _class_membership_probs_fn,
                self.shares,
            )
            print("\n".join(["Shares:", str(self.shares)]))
            logliks_list.append(self.loglik)
            em_recursion += 1
            if em_recursion >= 5:
                if (self.loglik - logliks_list[-5]) / logliks_list[-5] <= em_loglik_tol:
                    break

        self._em_post_fit(
            dems,
            dem_varnames,
            Xnames,
            em_recursion,
            em_maxiter,
            class_probs_by_panel,
            num_ids,
        )

    def predict(
        self,
        X,
        varnames,
        alts,
        ids,
        panels,
        num_classes,
        dems=None,
        X_past=None,
        y_past=None,
        alts_past=None,
        ids_past=None,
        panels_past=None,
        dems_past=None,
        dem_varnames=None,
    ):
        """Predict chosen alternatives.

        Parameters
        ----------
        X : array-like, shape (n_samples*n_alts, n_variables)
            Input data for explanatory variables in long format

        varnames : list, shape (n_variables,)
            Names of explanatory variables that must match the number and
            order of columns in ``X``

        alts : array-like, shape (n_samples*n_alts,)
            Alternative values in long format

        ids : array-like, shape (n_samples*n_alts,)
            Identifiers for the samples in long format.

        weights : array-like, shape (n_variables,), default=None
            Sample weights in long format.

        return_proba : bool, default=False
            If True, also return the choice probabilities

        return_freq : bool, default=False
            If True, also return the choice frequency for the alternatives


        Returns
        -------
        array-like, shape (n_samples, n_alts), optional
            Choice probabilities for each sample in the dataset. The
            alternatives are ordered (in the columns) as they appear
            in ``self.alternatives``. Only provided if
            `return_proba` is True.

        """
        # Format prediction data
        varnames, X, alts, ids, panels, dems = self._as_array(
            varnames, X, alts, ids, panels, dems
        )
        num_panels = jnp.unique(panels).shape[0]
        _, Xnames, ids, panels, dems, num_ids = self._setup_input_data(
            None, ids, panels, dems, predict_mode=True
        )

        # Compute panel-specific class membership probabilities
        if X_past is not None:
            varnames, X_past, y_past, alts_past, ids_past, panels_past, dems_past = (
                self._as_array(
                    varnames,
                    X_past,
                    y_past,
                    alts_past,
                    ids_past,
                    panels_past,
                    dems_past,
                )
            )
            self._validate_inputs(X, y_past, alts, varnames)

            self.num_classes, self.num_vars = num_classes, len(varnames)
            self.num_dem_vars = len(dem_varnames) if dem_varnames is not None else 0
            num_panels_past = jnp.unique(panels_past).shape[0]

            y_past, Xnames, ids_past, panels_past, dems_past, num_ids_past = (
                self._setup_input_data(
                    y_past, ids_past, panels_past, dems_past, predict_mode=False
                )
            )
            if not onp.array_equal(Xnames, self.coeff_names):
                raise ValueError(
                    "The provided `varnames` yield coefficient names that are inconsistent with those stored "
                    "in `self.coeff_names`"
                )

            # Define Xd as Xij - Xi*
            X_past, alts_past, ids_past, panels_past, panels_of_ids_past = (
                self._diff_nonchosen_chosen(
                    X_past, y_past, alts_past, ids_past, panels_past
                )
            )
            num_choices_per_panel_past = self._count_choices_per_panel(
                panels_past, ids_past
            )

            # Set up demographics grouped-data regression
            if dems is not None:
                _class_membership_probs_fn = jit(
                    partial(
                        _predict_class_membership_probs,
                        num_dem_vars=self.num_dem_vars,
                        num_classes=self.num_classes,
                    ),
                    static_argnames=["return_grad_components"],
                )
                _grouped_data_loglik_fn = jit(
                    partial(
                        _compute_grouped_data_loglik_and_grad,
                        _class_membership_probs_fn=_class_membership_probs_fn,
                        num_dem_vars=self.num_dem_vars,
                        num_classes=self.num_classes,
                    ),
                    static_argnames=["return_gradient"],
                )

            *_, class_probs_by_panel = _em_alg(
                self.betas,
                X_past,
                ids_past,
                panels_of_ids_past,
                num_choices_per_panel_past,
                self.num_vars,
                num_classes,
                num_ids_past,
                num_panels_past,
                self.thetas,
                dems_past,
                self.num_dem_vars,
                _grouped_data_loglik_fn,
                _class_membership_probs_fn,
                self.shares,
            )  # (Np, C)
        else:
            class_probs_by_panel = jnp.repeat(
                self.shares[None, :], num_panels, axis=0
            )  # (Np, C)

        _probs_and_exp_utility_fn = partial(
            _compute_probs_and_exp_utility, X=X, ids=ids, num_ids=num_ids
        )

        # self.betas: (K, C) matrix of coefficients
        choice_probs_by_class, exp_utility_by_class = lax.map(
            _probs_and_exp_utility_fn, self.betas.T
        )  # Both are (C, N x J)

        # Predicted choice probabilities, conditional on observed choices and demographics
        conditional_choice_probs = (
            class_probs_by_panel[panels] * choice_probs_by_class.T
        )  # (N x J, C)

        print(
            f"jnp.isfinite(conditional_choice_probs).mean() : {jnp.isfinite(conditional_choice_probs).mean()}"
        )

        log_sum_exp_utility = jnp.log(
            segment_sum(
                exp_utility_by_class.T,  # (N x J, C)
                ids,  # (N x J,)
                num_segments=num_ids,
            )
        )  # (N, C)

        price_coeff_by_class = self.betas[
            jnp.array([coeff_name == "neg_price" for coeff_name in Xnames])
        ]  # (C,)

        surplus_dollars_by_class = (
            log_sum_exp_utility / price_coeff_by_class[None, :]
        ).squeeze()  # (N, C)

        num_choices_per_panel = self._count_choices_per_panel(panels, ids)

        conditional_surplus_dollars = jnp.einsum(
            "np,np->n",
            jnp.repeat(class_probs_by_panel, num_choices_per_panel, axis=0),
            surplus_dollars_by_class,
        )  # (N, C)

        predicted_probs_df = pl.DataFrame(
            {
                "panels": onp.array(panels),
                "ids": onp.array(ids),
                "alts": onp.array(alts),
                "choice_probs": onp.array(conditional_choice_probs, dtype=onp.float64),
            }
        )

        surplus_df = pl.DataFrame(
            {
                "panels": onp.array(
                    panels[ids != jnp.roll(ids, shift=1)]
                ),  # One per transaction
                "ids": onp.array(ids[ids != jnp.roll(ids, shift=1)]),  # Ditto
                "surplus_dollars": onp.array(
                    conditional_surplus_dollars, dtype=onp.float64
                ),
            }
        )

        print("Strange subset:")
        print(surplus_df["surplus_dollars"].mean())
        print(onp.isfinite(surplus_df["surplus_dollars"].to_numpy()).mean())

        return predicted_probs_df, surplus_df

    def _get_starting_vals(
        self,
        Xd,
        alts,
        ids,
        panels,
        panels_of_ids,
        num_choices_per_panel,
        num_ids,
        num_panels,
        jax_prng_seed,
    ):
        data_dicts_by_class = self._random_class_partition(
            Xd, alts, ids, panels, num_panels, jax_prng_seed
        )
        betas = jnp.empty((self.num_vars, self.num_classes))
        for class_idx, data_dict in enumerate(data_dicts_by_class):
            class_starting_coeffs = _minimize_bfgs(
                _loglik_gradient,
                jnp.repeat(0.0, self.num_vars),
                args=(
                    data_dict["Xd"],
                    jnp.ones(shape=(data_dict["num_ids"],)),
                    data_dict["ids"],
                    data_dict["num_ids"],
                ),
            )["x"]
            betas = betas.at[:, class_idx].set(class_starting_coeffs)

        starting_class_probs_by_panel, _ = _compute_conditional_class_probs(
            betas,
            Xd,
            ids,
            panels_of_ids,
            num_choices_per_panel,
            num_ids,
            num_panels,
            shares=jnp.full((self.num_classes,), 1 / self.num_classes),
        )  # (Np, C)
        return betas, None, jnp.mean(starting_class_probs_by_panel, axis=0)

    def _random_class_partition(self, Xd, alts, ids, panels, num_panels, jax_prng_seed):
        panels_unique = onp.unique(panels)
        num_panels_per_class = -(num_panels // -self.num_classes)  # Ceiling division

        unshuffled_starting_classes = jnp.repeat(
            jnp.arange(self.num_classes), num_panels_per_class
        )[:num_panels]
        starting_classes = permutation(key(jax_prng_seed), unshuffled_starting_classes)
        starting_classes_by_panel_df = pl.DataFrame(
            {"panels": panels_unique, "starting_classes": onp.array(starting_classes)}
        )

        est_df = (
            pl.from_numpy(onp.array(Xd))
            .with_columns(
                pl.Series(name="panels", values=onp.asarray(panels)),
                pl.Series(name="ids", values=onp.array(ids)),
                pl.Series(name="alts", values=onp.array(alts)),
            )
            .join(starting_classes_by_panel_df, on="panels", how="left", coalesce=True)
        )
        est_dfs_by_class = est_df.partition_by("starting_classes", include_key=False)

        data_dicts_by_class = []
        for class_est_df in est_dfs_by_class:
            sorted_class_est_df = class_est_df.sort("panels", "ids", "alts")
            class_Xd = (
                sorted_class_est_df.drop("panels", "ids", "alts")
                .cast(pl.Float32)
                .to_jax()
            )
            class_ids = sorted_class_est_df["ids"].to_jax()
            data_dicts_by_class.append(
                {
                    "Xd": class_Xd,
                    "ids": class_ids,
                    "num_ids": jnp.unique(class_ids).shape[0],
                }
            )
        return data_dicts_by_class

    def _setup_input_data(self, y, ids, panels, dems, predict_mode=False):
        Xnames = onp.array(self._varnames.copy())

        # Ensure one set of demographic vars per panel
        if (dems is not None) and (y is not None):
            if dems.shape[0] == y.shape[0]:
                dems = dems[panels != jnp.roll(panels, shift=1)]

        if predict_mode:
            y = None
        else:
            y = y.astype(bool)
        return (
            y,
            Xnames,
            _ensure_sequential(ids),
            _ensure_sequential(panels),
            dems,
            jnp.unique(ids).shape[0],
        )

    def _count_choices_per_panel(self, panels, ids):
        panels_ids_df = pl.DataFrame(
            {"panels": onp.array(panels), "ids": onp.array(ids)}
        ).unique()
        return (
            panels_ids_df.group_by("panels")
            .agg(num_choices=pl.len())
            .sort("panels")["num_choices"]
            .to_jax()
        )

    def _diff_nonchosen_chosen(self, X, y, alts, ids, panels):
        """Construct Xd as Xij - Xi* (difference between non-chosen and chosen alternatives)"""
        _, num_unchosen_per_id = jnp.unique(ids[~y], return_counts=True)
        return (
            X[~y] - jnp.repeat(X[y], num_unchosen_per_id, axis=0),
            alts[~y],
            ids[~y],
            panels[~y],  # One per choice situation
            panels[y],  # One per choice situation!
        )

    def _em_post_fit(
        self,
        dems,
        dem_varnames,
        coeff_names,
        em_recursion,
        em_maxiter,
        class_probs_by_panel,
        sample_size,
    ):
        """Summarize results.

        Parameters
        ----------
        dems : array
            (Np, D) matrix of panel demographic characteristics.
        dem_varnames : list
            List of string names of all D demographic characteristics.
        coeff_names : list
            List of string names of all K alternative-specific variables.
        em_recursion : int
            Number of EM recursions.
        em_maxiter : int
            Maximum allowable EM recursions.
        class_probs_by_panel : array
            (Np, C) matrix of conditional class membership probabilities for each panel.
        sample_size : int
            Number of consumers.

        """
        # Summarize betas, which have dim (K, C)
        self.coeff_names = coeff_names

        mean_betas = self.betas.mean(axis=1)  # (K,)
        std_devs_betas = self.betas.std(axis=1)  # (K,)
        print("====\nMEANS AND STD DEVS\n")
        for coeff_idx, coeff_nm in enumerate(coeff_names):
            print(
                f"{coeff_nm} : {mean_betas[coeff_idx]:.3} ({std_devs_betas[coeff_idx]:.3})"
            )

        print("====\nVARIANCE/COVARIANCE MATRIX\n")
        print(jnp.cov(self.betas, aweights=self.shares, ddof=0))

        # Compute covariance between demographics and tastes
        dems_minus_mean = dems - dems.mean(axis=0)[None, :]  # (Np, D) - (D,) -> (Np, D)
        weighted_dems_minus_mean = jnp.einsum(
            "nc,nd->cd", class_probs_by_panel, dems_minus_mean
        )  # (Np, C) * (Np, D) -> (C, D)

        betas_minus_mean = self.betas - mean_betas[:, None]  # (K, C) - (K,) -> (K, C)

        dem_taste_cov = (
            jnp.einsum("cd,kc->dk", weighted_dems_minus_mean, betas_minus_mean)
            / sample_size
        )  # (C, D) * (K, C) ->  (D, K)

        print("====\nDEMOGRAPHIC COVARIANCES\n")
        for dem_idx, dem_nm in enumerate(dem_varnames):
            print(f"** Demographic variable: {dem_nm} **")
            for coeff_idx, coeff_nm in enumerate(coeff_names):
                print(f"{coeff_nm} : {dem_taste_cov[dem_idx, coeff_idx]:.3}")

        print(
            "\n".join(
                [
                    "====",
                    "RAW RESULTS",
                    "",
                    "Betas:",
                    str(self.betas),
                    "Thetas:",
                    str(self.thetas),
                    "Shares",
                    str(self.shares),
                ]
            )
        )

        self.total_recursions = em_recursion
        self.estim_time_sec = time() - self._fit_start_time

        self.num_params = self.betas.size + self.shares.size
        # Consistent Aikake information criterion (smaller is better!)
        self.caic = (jnp.log(sample_size) + 1) * self.num_params - 2 * self.loglik
        # Bayesian information criterion (smaller is better!)
        self.bic = jnp.log(sample_size) * self.num_params - 2 * self.loglik
        self.sample_size = sample_size

        self.convergence = em_recursion < em_maxiter - 1
        if not self.convergence:
            print(
                f"**** The optimization did not converge after {self.total_recursions} iterations. ****"
            )
