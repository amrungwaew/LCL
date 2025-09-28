from functools import partial

import jax.numpy as jnp
from jax import jit
from jax.ops import segment_prod, segment_sum
from lcl._demographics import _update_thetas
from lcl._optimize import _minimize_bfgs
from lcl.conditional_logit import _loglik_gradient


def _em_alg(
    betas,
    Xd,
    ids,
    panels_of_ids,
    num_choices_per_panel,
    num_vars,
    num_classes,
    num_ids,
    num_panels,
    thetas=None,
    dems=None,
    num_dem_vars=None,
    _grouped_data_loglik_fn=None,
    _class_membership_probs_fn=None,
    shares=None,
):
    """Run EM algorithm, which proceeds as follows:
        1. Update conditional class membership probabilities associated with each panel
        2. Update class coefficients
        3. Update aggregate class shares

    Parameters
    ----------
    betas : array
        (K, C) matrix of coefficients associated with each latent class.
    shares : array
        (C,) vector of class shares.
    Xd : array
        (N, K) tensor of differences in explanatory variables between unchosen and
        chosen alternatives.
    ids : array
        (N,) vector of choice situation IDs.
    panels_of_ids : array
        (Nc,) vector of consumer IDs, one per choice situation.
    num_vars : int
        Number of explanatory variables
    num_classes: int
        Number of latent classes
    num_ids : int
        Number of choices observed.
    num_panels : int
        Number of consumers in data.
    dems : array, optional
        (Np, D) matrix of panel demographic characteristics.
    thetas : array, optional
        (D, C) matrix of coefficients relating demographic variables to
        membership in specific latent classes.
    _grouped_data_loglik_fn : fun, optional
        Function that computes log-likelihood given class membership coefficients
        and explanatory variables.
    _class_membership_probs_fn : fun, optional
        Function that computes class membership probabilities given class membership
        coefficients and explanatory variables.

    Returns
    -------
    updated_betas : array
        (K, C) matrix of coefficients associated with each latent class.
    updated_thetas : array, optional
        (D, C) matrix of coefficients relating demographic variables to
        membership in specific latent classes. Only returned if demographic
        variables are provided.
    updated_shares : array, optional
        (C,) vector of class shares.
    _unconditional_loglik : float
        Unconditional log-likelihood of parameters.
    updated_class_probs_by_panel : array
        (Np, C) array of conditional class membership probabilities, given observed
        choices and demographics.

    """

    # 1. Compute conditional class membership probabilities given choices and demographics
    (
        updated_class_probs_by_panel,  # (Np, C)
        updated_class_probs_by_choice,  # (N, C)
    ) = _compute_conditional_class_probs(
        betas,
        Xd,
        ids,
        panels_of_ids,
        num_choices_per_panel,
        num_ids,
        num_panels,
        thetas,
        dems,
        _class_membership_probs_fn,
        shares,
    )

    # 2. Update classes' respective taste coefficients based on conditional
    # class membership probabilities
    updated_betas = _update_betas(
        betas,
        Xd,
        updated_class_probs_by_choice,
        ids,
        num_vars,
        num_classes,
        num_ids,
    )

    # 3. Update class membership model coefficients or class share vectors

    # 3.1 If demographics omitted, update class share vectors
    if dems is None:
        updated_shares = (
            updated_class_probs_by_panel.sum(axis=0)
            / updated_class_probs_by_panel.sum()
        )  # (C,)
        unconditional_class_probs_by_panel = jnp.repeat(
            updated_shares[None, :], repeats=num_panels, axis=0
        )  # (Np, C)
        updated_thetas = None  # Not applicable

    # 3.2 Otherwise, update class membership model coefficients
    else:
        # Initialize class membership model coefficients if not provided
        if thetas is None:
            thetas = jnp.zeros(((num_dem_vars + 1) * (num_classes - 1),))
            print(f"thetas.shape : {thetas.shape}")

        # Update coefficients and recover unconditional class membership probabilities
        updated_thetas, unconditional_class_probs_by_panel = _update_thetas(
            thetas,
            updated_class_probs_by_panel,
            dems,
            _grouped_data_loglik_fn,
            _class_membership_probs_fn,
        )
        updated_shares = unconditional_class_probs_by_panel.mean(axis=0)

    # Compute unconditional log likelihood given taste coefficients and class membership
    # coefficients
    unconditional_loglik = _compute_unconditional_loglik(
        betas,
        unconditional_class_probs_by_panel,
        Xd,
        ids,
        panels_of_ids,
        num_ids,
        num_panels,
    )

    return (
        updated_betas,
        updated_thetas,
        updated_shares,
        unconditional_loglik,
        updated_class_probs_by_panel,
    )


def _compute_conditional_class_probs(
    betas,
    Xd,
    ids,
    panels_of_ids,
    num_choices_per_panel,
    num_ids,
    num_panels,
    thetas=None,
    dems=None,
    _class_membership_probs_fn=None,
    shares=None,
):
    """Update conditional class membership probabilities of all classes.

    Parameters
    ----------
    betas : array
        (K, C) matrix of coefficients associated with each latent class.
    Xd : array
        (N, K) tensor of differences in explanatory variables between unchosen and
        chosen alternatives.
    ids : array
        (N,) vector of choice situation IDs.
    panels_of_ids : array
        (Nc,) vector of consumer IDs, one per choice situation.
    num_choices_per_panel : array
        (Np,) vector with each panel's number of observed choices.
    num_ids : int
        Number of choices observed.
    num_panels : int
        Number of consumers in data.
    num_classes: int
        Number of latent classes.
    thetas : array, optional
        (D, C) matrix of coefficients relating demographic variables to
        membership in specific latent classes.
    dems : array, optional
        (Np, D) matrix of demographic variables.
    _class_membership_probs_fn : fun, optional
        Function that computes predicted probabilities of class membership
        based on demographics
    shares : array, optional
        (C,) vector of class shares.

    Returns
    -------
    updated_class_probs_by_panel : array
        (Np, C) matrix of conditional class membership probabilities for each panel.
    updated_class_probs_by_choice : array
        (N, C) matrix of conditional class membership probabilities for each choice situation.

    """
    kernels = _compute_kernels(
        betas, Xd, ids, panels_of_ids, num_ids, num_panels
    )  # (Np, C)
    if thetas is None:
        weighted_kernels = kernels * shares[None, :]  # (Np, C), (C,) -> (Np, C)
        print(
            "\n".join(
                [
                    "====",
                    "`weighted_kernels`",
                    f"Frac. finite: {jnp.isfinite(weighted_kernels).mean()}",
                    f"Frac. nonzero: {(weighted_kernels>0).mean()}",
                    f"Mean: {weighted_kernels.mean()}",
                ]
            )
        )

    else:
        class_probs_given_dems = _class_membership_probs_fn(thetas, dems)  # (Np, C)
        weighted_kernels = kernels * class_probs_given_dems  # (Np, C)

    # Remove zero kernels from floating point errors
    weighted_kernels_plus_delta = weighted_kernels + 1e-200
    weighted_kernels = (
        weighted_kernels_plus_delta / weighted_kernels_plus_delta.sum(axis=1)[:, None]
    )  # (Np, C)

    conditional_class_probs = (
        weighted_kernels / jnp.sum(weighted_kernels, axis=1)[:, None]
    )  # (Np, C)
    print(
        "\n".join(
            [
                "====",
                "`conditional_class_probs`",
                f"Frac. finite: {jnp.isfinite(conditional_class_probs).mean()}",
                f"Frac. nonzero: {(conditional_class_probs>0).mean()}",
                f"Mean: {conditional_class_probs.mean()}",
            ]
        )
    )

    return conditional_class_probs, jnp.repeat(
        conditional_class_probs,
        num_choices_per_panel,
        axis=0,
        total_repeat_length=num_ids,
    )


def _update_betas(
    betas,
    Xd,
    class_probs_by_choice,
    ids,
    num_vars,
    num_classes,
    num_ids,
):
    """Update the coefficients of each class (contained in a matrix)

    Parameters
    ----------
    betas : array
        (K, C) matrix of coefficients associated with each latent class.
    Xd : array
        (N, K) tensor of differences in explanatory variables between unchosen and
        chosen alternatives.
    class_probs_by_choice : array
        (Np, C) matrix of conditional class membership probabilities for each panel.
    ids : array
        (N,) vector of choice situation IDs.
    num_vars : int
        Number of explanatory variables.
    num_classes: int
        Number of latent classes.
    num_ids : int
        Number of choices observed.

    Returns
    -------
    updated_betas : array
        (K, C) matrix of coefficients associated with each latent class.

    """
    updated_betas = jnp.empty((num_vars, num_classes))
    for latent_class in range(num_classes):
        class_betas = _minimize_bfgs(
            _loglik_gradient,
            betas[:, latent_class],
            args=(Xd, class_probs_by_choice[:, latent_class], ids, num_ids),
        )["x"]
        updated_betas = updated_betas.at[:, latent_class].set(class_betas)
    return updated_betas


@partial(jit, static_argnames=["num_ids", "num_panels"])
def _compute_unconditional_loglik(
    betas,
    unconditional_class_probs_by_panel,
    Xd,
    ids,
    panels_of_ids,
    num_ids,
    num_panels,
):
    """Compute log likelihood value, given present class coefficients and shares

    Parameters
    ----------
    betas : array
        (K, C) matrix of coefficients associated with each latent class.
    unconditional_class_probs_by_panel : array
        (Np, C) matrix of unconditional probabilities that each panel belongs to each
        class. May depend on panels_of_ids' demographic characteristics, but does NOT directly
        reflect their respective choices.
    Xd : array
        (N, K) tensor of differences in explanatory variables between unchosen and
        chosen alternatives.
    ids : array
        (N,) vector of choice situation IDs.
    panels_of_ids : array
        (Nc,) vector of consumer IDs, one per choice situation.
    num_ids : int
        Number of choices observed.
    num_panels : int
        Number of consumers in data.

    Returns
    -------
    loglik : float
        Unconditional log-likelihood of parameters.

    """
    kernels = _compute_kernels(betas, Xd, ids, panels_of_ids, num_ids, num_panels)
    weighted_kernels = jnp.einsum(
        "nc,nc->n", unconditional_class_probs_by_panel, kernels
    )
    return jnp.sum(jnp.log(weighted_kernels))


@partial(jit, static_argnames=["num_ids", "num_panels"])
def _compute_kernels(betas, Xd, ids, panels_of_ids, num_ids, num_panels):
    """Compute conditional probabilities of observed choice sequences.

    Parameters
    ----------
    betas : array
        (K, C) matrix of coefficients associated with each latent class.
    Xd : array
        (N, K) tensor of differences in explanatory variables between unchosen and
        chosen alternatives.
    ids : array
        (N,) vector of choice situation IDs.
    panels_of_ids : array
        (Nc,) vector of consumer IDs, one per choice situation.
    num_ids : int
        Number of choices observed.
    num_panels : int
        Number of consumers in data.

    Returns
    -------
    kernels_by_class : array
        (Np, C) matrix of logit kernels assocaited with each latent class.

    """
    Vd_by_class = jnp.einsum("nk,kc->nc", Xd, betas)
    eVd_by_class, Vd_by_class = jnp.exp(Vd_by_class), None

    # Compute chosen alts' conditional choice probabalities by latent class
    probs_by_class = 1 / (
        1 + segment_sum(eVd_by_class, ids, num_segments=num_ids)
    )  # (N, C)

    return segment_prod(probs_by_class, panels_of_ids, num_segments=num_panels)


# @partial(jit, static_argnames=["num_ids"])
def _compute_probs_and_exp_utility(betas, X, ids, num_ids):
    """Compute conditional choice probabilities for an individual class.

    Parameters
    ----------
    betas : array
        (K, C) matrix of coefficients (which are transposed from their usual shape!)
    ids : array
        (N,) vector of choice situation IDs.
    num_ids : int
        Number of choices observed.

    Returns
    -------
    probs : array
        (N * J,) vector of conditional choice probabilities.
    eV : array
        Exponentiated representative utility.

    """
    eV = jnp.exp((X.dot(betas.T)))
    probs = eV / segment_sum(eV, ids, num_segments=num_ids)[ids]  # (N * J,)
    return probs, eV


# Bug shield
