import jax.numpy as jnp


def _minimize_bfgs(
    loglik_fn,
    params,
    args,
    maxiter=2000,
    tol=1e-10,
    gtol=1e-6,
    step_tol=1e-10,
    disp=False,
):
    """BFGS optimization routine."""
    neg_loglik, grad, grad_n = loglik_fn(params, *args, **{"return_gradient": True})
    Hinv = jnp.linalg.pinv(jnp.dot(grad_n.T, grad_n))
    convergence = False
    step_tol_failed = False
    nit, nfev, njev = 0, 1, 1

    while True:
        old_grad = grad

        direction = -Hinv.dot(grad)

        step = 2
        while True:
            step = step / 2
            s = step * direction
            resnew = loglik_fn(params + s, *args, **{"return_gradient": False})
            nfev += 1
            if step > step_tol:
                if resnew <= neg_loglik or step < step_tol:
                    params += s
                    resnew, gnew, grad_n = loglik_fn(
                        params, *args, **{"return_gradient": True}
                    )
                    njev += 1
                    break
            else:
                step_tol_failed = True
                break

        nit += 1

        if step_tol_failed:
            convergence = False
            message = "Local search could not find a higher log likelihood value"
            break

        old_res = neg_loglik
        neg_loglik = resnew
        grad = gnew
        gproj = jnp.abs(jnp.dot(direction, old_grad))

        if disp:
            print(
                f"Iteration: {nit} \t Log-Lik.= {resnew:.3f} \t |proj grad|= {gproj:e}"
            )

        if gproj < gtol:
            convergence = True
            message = "The gradients are close to zero"
            break

        if jnp.abs(neg_loglik - old_res) < tol:
            convergence = True
            message = "Succesive log-likelihood values within tolerance limits"
            break

        if nit > maxiter:
            convergence = False
            message = "Maximum number of iterations reached without convergence"
            break

        delta_g = grad - old_grad

        Hinv += (
            (
                (s.dot(delta_g) + (delta_g[None, :].dot(Hinv)).dot(delta_g))
                * jnp.outer(s, s)
            )
            / (s.dot(delta_g)) ** 2
        ) - (
            (jnp.outer(Hinv.dot(delta_g), s) + (jnp.outer(s, delta_g)).dot(Hinv))
            / (s.dot(delta_g))
        )

    Hinv = jnp.linalg.pinv(jnp.dot(grad_n.T, grad_n))
    return {
        "success": convergence,
        "x": params,
        "fun": neg_loglik,
        "message": message,
        "hess_inv": Hinv,
        "grad_n": grad_n,
        "grad": grad,
        "nit": nit,
        "nfev": nfev,
        "njev": njev,
    }
