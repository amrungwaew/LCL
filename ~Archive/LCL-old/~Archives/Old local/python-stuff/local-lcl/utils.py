"""General utilities for xlogit library."""

import jax.numpy as jnp
from scipy.stats import chi2


def lrtest(general_model, restricted_model):
    """Conducts likelihood-ratio test.

    Parameters
    ----------
    general_model : xlogit Model
        Fitted model that contains all parameters (unrestricted)

    restricted_model : xlogit Model
        Fitted model with less parameters than ``general_model``.

    Returns
    -------
    lrtest_result : dict
        p-value result, chisq statistic, and degrees of freedom used in test
    """
    if len(general_model.coeff_) <= len(restricted_model.coeff_):
        raise ValueError(
            "The general_model is expected to have less estimates"
            "than the restricted_model"
        )
    genLL, resLL = general_model.loglikelihood, restricted_model.loglikelihood
    degfreedom = len(general_model.coeff_) - len(restricted_model.coeff_)
    stat = 2 * (resLL - genLL)
    return {"pval": chi2.sf(stat, df=degfreedom), "chisq": stat, "degfree": degfreedom}


def _ensure_sequential(vals):
    """Ensure ids can also serve as indices"""
    vals_change = vals != jnp.roll(vals, shift=1)
    return jnp.cumsum(vals_change) - 1


_unpack_tuple = lambda x: x if len(x) > 1 else x[0]
