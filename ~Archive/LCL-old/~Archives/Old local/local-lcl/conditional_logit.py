"""Estimation and prediction for conditional logit."""

from functools import partial

import jax.numpy as jnp
import numpy as onp
from jax import jit
from jax.ops import segment_sum
from lcl._choice_model import ChoiceModel
from lcl._optimize import _minimize_bfgs
from lcl.utils import _ensure_sequential, _unpack_tuple

"""
Notation
---------
    N : Number of choice situations
    J : Number of alternatives
    K : Number of variables
"""


@partial(jit, static_argnames=["num_ids", "return_gradient"])
def _loglik_gradient(betas, Xd, weights, ids, num_ids, return_gradient=True):
    """Compute log-likelihood, gradient, and hessian."""
    Vd = Xd.dot(betas)  # (Nd, K) * (K,) -> (Nd,)
    eVd = jnp.exp(Vd)
    probs = 1 / (1 + segment_sum(eVd, ids, num_segments=num_ids))  # (N,)

    # Log likelihood
    loglik = jnp.log(probs) * weights
    output = (-jnp.sum(loglik),)

    # Individual decisions' contribution to the gradient
    if return_gradient:
        grad_n = -segment_sum(Xd * eVd[:, None], ids, num_segments=num_ids)  # (Nd, K)
        grad_n = grad_n * probs[:, None]
        grad_n = grad_n if weights is None else grad_n * weights[:, None]
        grad = jnp.sum(grad_n, axis=0)
        output += (-grad.ravel(),)
        output += (grad_n,)
    return _unpack_tuple(output)


class ConditionalLogit(ChoiceModel):
    """Class for estimation of Multinomial and Conditional Logit Models.

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
        avail=None,
        weights=None,
        init_coeff=None,
        maxiter=2000,
        tol_opts=None,
        verbose=1,
        robust=False,
        skip_std_errs=False,
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
            order of columns in ``X``

        alts : array-like, shape (n_samples*n_alts,)
            Alternative values in long format

        ids : array-like, shape (n_samples*n_alts,)
            Identifiers for the samples in long format.

        avail: array-like, shape (n_samples*n_alts,), default=None
            Availability of alternatives for the samples. True when
            available and False otherwise.

        weights : array-like, shape (n_variables,), default=None
            Weights for the choice situations in long format.

        init_coeff : numpy array, shape (n_variables,), default=None
            Initial coefficients for estimation.

        maxiter : int, default=200
            Maximum number of iterations

        robust: bool, default=False
            Whether robust standard errors should be computed

        num_hess: bool, default=False
            Whether numerical hessian should be used for estimation of standard errors

        skip_std_errs: bool, default=False
            Whether estimation of standard errors should be skipped

        tol_opts : dict, default=None
            Options for tolerance of optimization routine. The dictionary accepts the following options (keys):

                ftol : float, default=1e-10
                    Tolerance for objective function (log-likelihood)

        verbose : int, default=1
            Verbosity of messages to show during estimation. 0: No messages,
            1: Some messages, 2: All messages


        Returns
        -------
        None.
        """
        varnames, X, y, alts, ids, avail, weights = self._as_array(
            varnames, X, y, alts, ids, avail, weights
        )
        self._validate_inputs(X, y, alts, varnames)
        self._pre_fit(alts, varnames, maxiter)

        betas, X, y, weights, Xnames, ids, num_ids = self._setup_input_data(
            X, y, ids, weights, avail, init_coeff, predict_mode=False
        )

        # Define Xd as Xij - Xi*
        Xd, ids = self._diff_nonchosen_chosen(X, y, ids)

        # Set optimization tolerances
        tol = {"ftol": 1e-10, "gtol": 1e-6}
        if tol_opts is not None:
            tol.update(tol_opts)

        # Perform optimization
        fargs = (Xd, weights, ids, num_ids)
        optim_res = _minimize_bfgs(_loglik_gradient, betas, args=fargs, tol=tol["ftol"])
        if skip_std_errs:
            optim_res["hess_inv"] = jnp.eye(len(optim_res["x"]))

        self._post_fit(optim_res, Xnames, X.shape[0], verbose, robust)

    def predict(self, X, varnames, alts, ids, weights=None, avail=None, verbose=1):
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

        verbose : int, default=1
            Verbosity of messages to show during estimation. 0: No messages,
            1: Some messages, 2: All messages

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
        varnames, X, alts, ids, weights = self._as_array(
            varnames, X, alts, ids, weights
        )
        self._validate_inputs(X, None, alts, varnames)

        betas, X, _, weights, Xnames, ids, num_ids = self._setup_input_data(
            X, None, ids, weights, avail, init_coeff=self.coeff_, predict_mode=True
        )

        if not onp.array_equal(Xnames, self.coeff_names):
            raise ValueError(
                "The provided `varnames` yield coefficient names that are inconsistent with those stored "
                "in `self.coeff_names`"
            )

        # === 2. Compute choice probabilities
        eV = jnp.exp((X.dot(betas)))
        probs = (
            eV
            / segment_sum(eV, ids, num_segments=num_ids)[ids != jnp.roll(ids, shift=1)]
        )  # (N * J,)

        return probs, alts

    def _setup_input_data(
        self, X, y, ids, weights=None, avail=None, init_coeff=None, predict_mode=False
    ):
        Xnames = onp.array(self._varnames.copy())
        K = len(self._varnames)
        if predict_mode:
            y = None
        if init_coeff is None:
            betas = jnp.repeat(0.0, K)
        else:
            betas = init_coeff
            if len(init_coeff) != K:
                raise ValueError(f"The size of initial_coeff must be: {K}")

        if avail is not None:
            X, y, ids = X[avail], y[avail], ids[avail]
            if weights is not None:
                weights = weights[avail]

        num_ids = jnp.unique(ids).shape[0]

        # If not provided, weights equal unity
        if weights is None:
            weights = jnp.ones(shape=(num_ids,))

        return (
            betas,
            X,
            y.astype(bool),
            weights,
            Xnames,
            _ensure_sequential(ids),
            num_ids,
        )

    def _diff_nonchosen_chosen(self, X, y, ids):
        """Construct Xd as Xij - Xi* (difference between non-chosen and chosen alternatives)"""
        _, num_unchosen_per_id = jnp.unique(ids[~y], return_counts=True)
        return X[~y] - jnp.repeat(X[y], num_unchosen_per_id, axis=0), ids[~y]
