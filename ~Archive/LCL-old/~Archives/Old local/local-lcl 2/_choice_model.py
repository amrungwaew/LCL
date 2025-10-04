import warnings
from abc import ABC
from time import time

import jax.numpy as jnp
import numpy as onp
from scipy.stats import t

"""
Notation
---------
    N : Number of choice situations
    P : Number of observations per panel
    J : Number of alternatives
    K : Number of variables (Kf: fixed, Kr: random)
"""


class ChoiceModel(ABC):
    """Base class for estimation of discrete choice models."""

    def __init__(self):
        self.coeff_names = None
        self.coeff_ = None
        self.stderr = None
        self.zvalues = None
        self.pvalues = None
        self.loglikelihood = None
        self.total_fun_eval = 0
        self.verbose = 1
        self.robust = False

    def _reset_attributes(self):
        self.coeff_names = None
        self.coeff_ = None
        self.stderr = None
        self.zvalues = None
        self.pvalues = None
        self.loglikelihood = None
        self.total_fun_eval = 0
        self.verbose = 1
        self.robust = False

    def _as_array(self, varnames, *arrays):
        output_tuple = (onp.asarray(varnames),) if varnames is not None else (None,)
        for array in arrays:
            if array is None:
                output_tuple += (None,)
            else:
                output_tuple += (jnp.asarray(array),)
        return output_tuple

    def _pre_fit(self, alts, _varnames, maxiter):
        self._reset_attributes()
        self._fit_start_time = time()
        self._varnames = list(_varnames)  # Easier to handle with lists
        self.alternatives = jnp.sort(jnp.unique(alts))
        self.maxiter = maxiter

    def _post_fit(self, optim_res, coeff_names, sample_size, verbose=1, robust=False):
        self.convergence = optim_res["success"]
        self.coeff_ = optim_res["x"]
        self.hess_inv = optim_res["hess_inv"]
        self.covariance = (
            self._robust_covariance(optim_res["hess_inv"], optim_res["grad_n"])
            if robust
            else optim_res["hess_inv"]
        )
        self.stderr = jnp.sqrt(jnp.diag(self.covariance))
        self.zvalues = self.coeff_ / self.stderr
        self.pvalues = 2 * t.cdf(-onp.abs(self.zvalues), df=sample_size)
        self.loglikelihood = -optim_res["fun"]
        self.estimation_message = optim_res["message"]
        self.coeff_names = coeff_names
        self.total_iter = optim_res["nit"]
        self.estim_time_sec = time() - self._fit_start_time
        self.sample_size = sample_size
        self.aic = 2 * len(self.coeff_) - 2 * self.loglikelihood
        self.bic = jnp.log(sample_size) * len(self.coeff_) - 2 * self.loglikelihood
        self.grad_n = optim_res["grad_n"]
        self.total_fun_eval = optim_res["nfev"]

        if not self.convergence and verbose > 0:
            print(
                "\n".join(
                    [
                        f"**** The optimization did not converge after {self.total_iter} iterations. ****",
                        f"Message: {optim_res['message']}",
                    ]
                )
            )

    def _robust_covariance(self, hess_inv, grad_n):
        """Employ the Huber/White correction for heteroskedasticity."""
        n = jnp.shape(grad_n)[0]
        grad_n_sub = grad_n - (
            jnp.sum(grad_n, axis=0) / n
        )  # subtract out mean gradient value
        inner = jnp.transpose(grad_n_sub) @ grad_n_sub
        correction = (n) / (n - 1)
        covariance = correction * (hess_inv @ inner @ hess_inv)
        return covariance

    def _format_choice_var(self, y, alts):
        """Format choice (y) variable as one-hot encoded."""
        uq_alts = jnp.unique(alts)
        J, N = len(uq_alts), len(y) // len(uq_alts)
        # When already one-hot encoded the sum by row is one
        if jnp.array_equal(
            y.reshape(N, J).sum(axis=1), jnp.ones(N)
        ):  # isinstance(y[0], (jnp.number, jnp.bool_)) and
            return y
        else:
            y1h = (y == alts).astype(bool)  # Apply one hot encoding
            if jnp.array_equal(y1h.reshape(N, J).sum(axis=1), jnp.ones(N)):
                return y1h
            else:
                print(y1h.reshape(N, J).sum(axis=1))
                raise ValueError(
                    "Inconsistent `y` values. Ensure the data have one choice per sample"
                )

    def _validate_inputs(self, X, y, alts, varnames):
        """Validate potential mistakes in the input data."""
        if varnames is None:
            raise ValueError("The parameter varnames is required")
        if alts is None:
            raise ValueError("The parameter alternatives is required")
        if X.ndim != 2:
            raise ValueError("X must be an array of two dimensions in long format")
        if y is not None and y.ndim != 1:
            raise ValueError("y must be an array of one dimension in long format")
        if len(varnames) != X.shape[1]:
            raise ValueError(
                "The length of varnames must match the number of columns in X"
            )

    def summary(self):
        """Return estimation results as string."""
        summary_as_list = []
        summary_as_list.extend(
            [f"self.coeff_: {self.coeff_}", f"self.coeff_names: {self.coeff_names}"]
        )
        if self.coeff_ is None:
            warning_str = "The current model has not been yet estimated"
            warnings.warn(warning_str, UserWarning)
            summary_as_list.append(warning_str)
            return summary_as_list
        if not self.convergence:
            warning_str = (
                "WARNING: Convergence not reached. The estimates may not be reliable."
            )
            warnings.warn(warning_str, UserWarning)
            summary_as_list.append(warning_str)
        if self.convergence:
            summary_as_list.append("Optimization terminated successfully.")
        summary_as_list.extend(
            [
                f"    Message: {self.estimation_message}",
                f"    Iterations: {self.total_iter}",
                f"    Function evaluations: {self.total_fun_eval}",
                f"Estimation time= {self.estim_time_sec:.1f} seconds",
                "-" * 75,
                "{:40} {:>13} {:>13} {:>13} {:>13}".format(
                    "Coefficient", "Estimate", "Std.Err.", "z-val", "P>|z|"
                ),
                "-" * 75,
            ]
        )
        fmt = "{:40} {:13.7f} {:13.7f} {:13.7f} {:13.3g} {:3}"
        for i in range(len(self.coeff_)):
            signif = ""
            if self.pvalues[i] < 0.001:
                signif = "***"
            elif self.pvalues[i] < 0.01:
                signif = "**"
            elif self.pvalues[i] < 0.05:
                signif = "*"
            elif self.pvalues[i] < 0.1:
                signif = "."
            summary_as_list.append(
                fmt.format(
                    self.coeff_names[i][:40],
                    self.coeff_[i],
                    self.stderr[i],
                    self.zvalues[i],
                    self.pvalues[i],
                    signif,
                )
            )
        summary_as_list.append("-" * 75)
        summary_as_list.append(
            "Significance:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1"
        )
        summary_as_list.append("")
        summary_as_list.append(f"Log-Likelihood= {self.loglikelihood:.3f}")
        summary_as_list.append(f"AIC= {self.aic:.3f}")
        summary_as_list.append(f"BIC= {self.bic:.3f}")
        return "\n".join(summary_as_list)
