import jax.numpy as jnp
from lcl._optimize import _minimize_bfgs
from lcl.utils import _unpack_tuple


def _update_thetas(
    starting_thetas,
    class_probs_by_panel,
    dems,
    _grouped_data_loglik_fn,
    _class_membership_probs_fn,
):
    """Update class shares based on conditional choice probabilities and demographics.

    Parameters
    ----------
    starting_thetas : array
        (D, C) matrix of coefficients relating demographic variables to
        membership in specific latent classes.
    class_probs_by_panel : array
        (Np, C) matrix of class membership probabilities conditional on each
        panel's respective choices.
    dems : array
        (Np, D) matrix of demographic variables.
    _grouped_data_loglik_fn : fun
        Function that computes log-likelihood given class membership coefficients
        and explanatory variables.
    _class_membership_probs_fn : fun
        Function that computes class membership probabilities given class membership
        coefficients and explanatory variables.

    Returns
    -------
    updated_thetas : array
        (D, C) matrix of coefficients relating demographic variables to
        membership in specific latent classes.
    predicted_class_probs : array
        (Np, C) matrix of predicted class membership probabilities.

    """
    updated_thetas, convergence = _perform_frac_response_reg(
        starting_thetas, class_probs_by_panel, dems, _grouped_data_loglik_fn
    )  # (D * C,)
    print(f"Thetas: {updated_thetas}")
    if not convergence:
        # print("oops!")
        print("Oops! Nonconvergence")

        # raise Exception("Demographic regression failed to converge.")
    else:
        print("Yay! Convergence")
    predicted_class_probs = _class_membership_probs_fn(updated_thetas, dems)  # (Np, C)

    return updated_thetas, predicted_class_probs


def _perform_frac_response_reg(
    thetas, class_probs_by_panel, dems, _grouped_data_loglik_fn
):
    """Perform fractional response regression of class membership probabilities.

    Parameters
    ----------
    thetas : array
        (D, C) matrix of coefficients relating demographic variables to
        membership in specific latent classes.
    class_probs_by_panel : array
        (Np, C) matrix of class membership probabilities conditional on each
        panel's respective choices.
    dems : array
        (Np, D) matrix of demographic variables.
    _grouped_data_loglik_fn : fun
        Function that computes log-likelihood given class membership coefficients
        and explanatory variables.

    Returns
    -------
    updated_shares : array
        (C,) vector of class shares.

    """
    fargs = (class_probs_by_panel, dems)
    optim_res = _minimize_bfgs(_grouped_data_loglik_fn, thetas, args=fargs)
    return optim_res["x"], optim_res["success"]


def _compute_grouped_data_loglik_and_grad(
    thetas,
    class_probs_by_panel,
    dems,
    _class_membership_probs_fn,
    num_dem_vars,
    num_classes,
    return_gradient=True,
):
    """Compute grouped-data log likelihood.

    Parameters
    ----------
    thetas : array
        (D, C) matrix of coefficients relating demographic variables to
        membership in specific latent classes.
    class_probs_by_panel : array
        (Np, C) matrix of class membership probabilities conditional on each
        panel's respective choices.
    dems : array
        (Np, D) matrix of demographic variables.
    _class_membership_probs_fn : fun, optional
        Function that computes predicted probabilities of class membership
        based on demographics.
    num_dem_vars : int
        Number of demographic variables.
    num_classes: int
        Number of latent classes.

    Returns
    -------
    neg_grouped_data_loglik : float
        Grouped-data log likelihood given current coefficients.
    grad : array, optional
        (D * C,) vector representing the gradient, which is actually (D, C).
    grad_n: array, optional
        (Np, D * C) matrix representing individual panels' contributions to
        the gradient. (Conceptualize as [Np, D, C].)

    """
    predicted_class_probs, exp_latent_class_vars, sum_exp_latent_class_vars = (
        _class_membership_probs_fn(thetas, dems, return_grad_components=True)
    )
    loglik = jnp.sum(class_probs_by_panel * jnp.log(predicted_class_probs))
    output = (-loglik,)

    if return_gradient:
        probs_times_quotient = (
            class_probs_by_panel[:, 1:] * sum_exp_latent_class_vars[:, None]
            - exp_latent_class_vars
        ) / sum_exp_latent_class_vars[
            :, None
        ]  # (Np, C - 1)
        grad_n = jnp.concat(
            [
                probs_times_quotient[:, None, :],  # (Np, C - 1)
                probs_times_quotient[:, None, :] * dems[..., None],  # (Np, D, C - 1)
            ],
            axis=1,
        )  # (Np, D + 1, C - 1)

        grad = grad_n.sum(axis=0)  # (D, C - 1)
        output += (
            -grad.ravel(),
            grad_n.reshape(
                -1, (num_dem_vars + 1) * (num_classes - 1)
            ),  # (Np, (D + 1) * (C - 1))
        )

    return _unpack_tuple(output)


def _predict_class_membership_probs(
    thetas, dems, num_dem_vars, num_classes, return_grad_components=False
):
    """Compute predicted probabilities of class membership based on demographics.

    Parameters
    ----------
    thetas : array
        (D, C) matrix of coefficients relating demographic variables to
        membership in specific latent classes.
    dems : array
        (Np, D) matrix of demographic variables.
    num_dem_vars : int
        Number of demographic variables.
    num_classes: int
        Number of latent classes.
    return_grad_components : bool, default=False
        Return

    Returns
    -------
    predicted_class_probs : array
        (Np, C) matrix of conditional probabilities of class membership for
        each panel.
    exp_latent_class_vars : array, optional
        (Np, C - 1) matrix of exponentiated latent variables
    sum_exp_latent_class_vars : array, optional
        (Np,) vector of sum of exponentiated latent variables

    """
    thetas = thetas.reshape(num_dem_vars + 1, num_classes - 1)  # (D + 1, C - 1)
    exp_latent_class_vars = jnp.exp(
        thetas[None, 0] + dems @ thetas[1:]
    )  # (D + 1,) + (Np, D) (D, C - 1) -> (Np, C - 1)
    sum_exp_latent_class_vars = 1 + exp_latent_class_vars.sum(axis=1)  # (Np,)

    probs_identified_classes = (
        exp_latent_class_vars / sum_exp_latent_class_vars[:, None]
    )  # (Np, C - 1)

    output = (
        jnp.concat(
            [
                (1 / sum_exp_latent_class_vars)[:, None],  # (Np, 1)
                probs_identified_classes,  # (Np, C - 1)
            ],
            axis=1,
        ),  # (Np, C)
    )

    # If requested, also return intermediate results that appear in the gradient
    if return_grad_components:
        output += (exp_latent_class_vars, sum_exp_latent_class_vars)

    return _unpack_tuple(output)
