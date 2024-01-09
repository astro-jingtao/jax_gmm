import jax
import jax.numpy as jnp
import jax.scipy as jsp
from jax import jit, vmap, grad

import tensorflow_probability.substrates.jax as jaxp

jaxd = jaxp.distributions

# @jax.jit
# def log_likelihood(X, pi, mu, sigma):

#     mixture_log_prob = jnp.log(pi) + jaxd.MultivariateNormalTriL(
#         loc=mu, scale_tril=jnp.linalg.cholesky(sigma)).log_prob(X[:, None,
#                                                                   ...])
#     return jsp.special.logsumexp(mixture_log_prob, axis=-1, keepdims=True).T[0]

# TODO: use analytical solution of derivative


@jax.jit
def _log_likelihood(X, pi, mu, sigma):
    mixture_log_prob = jnp.log(pi) + jaxd.MultivariateNormalTriL(
        loc=mu, scale_tril=jnp.linalg.cholesky(sigma)).log_prob(X[None, ...])
    return jsp.special.logsumexp(mixture_log_prob, axis=-1, keepdims=False)


log_likelihood = jit(vmap(_log_likelihood, in_axes=(0, None, None, None)))

d_log_likelihood = jit(
    vmap(grad(_log_likelihood, argnums=0), in_axes=(0, None, None, None)))

# @jax.jit
# def likelihood(X, pi, mu, sigma):
#     return jnp.exp(log_likelihood(X, pi, mu, sigma))


@jax.jit
def _likelihood(X, pi, mu, sigma):
    return jnp.exp(_log_likelihood(X, pi, mu, sigma))


likelihood = jit(vmap(_likelihood, in_axes=(0, None, None, None)))

d_likelihood = jit(
    vmap(grad(_likelihood, argnums=0), in_axes=(0, None, None, None)))


def cond_split(X, mu, sigma, cond_flag):

    is_cond = jnp.where(cond_flag)[0]
    not_cond = jnp.where(~cond_flag)[0]

    X1 = X[:, not_cond]
    X2 = X[:, is_cond]

    mu1 = mu[:, not_cond]
    mu2 = mu[:, is_cond]

    sigma_11 = sigma[:, not_cond][:, :, not_cond]
    sigma_12 = sigma[:, not_cond][:, :, is_cond]
    sigma_21 = sigma[:, is_cond][:, :, not_cond]
    sigma_22 = sigma[:, is_cond][:, :, is_cond]

    return X1, X2, mu1, mu2, sigma_11, sigma_12, sigma_21, sigma_22


def _log_clikelihood(X1, X2, pi, mu1, mu2, sigma_11, sigma_12, sigma_21,
                     sigma_22):
    '''
    X1: (nc,)
    X2: (c,)
    pi: (N,)
    mu1: (N, nc)
    mu2: (N, c)
    sigma_11: (N, nc, nc)
    sigma_12: (N, nc, c)
    sigma_21: (N, c, nc)
    sigma_22: (N, c, c)

    mu_cond: (N, nc)
    sigma_cond: (N, nc, nc)
    '''
    # https://en.wikipedia.org/wiki/Multivariate_normal_distribution#Conditional_distributions

    mu_cond = mu1 + jnp.einsum('xyz,xz->xy', sigma_12,
                               jnp.linalg.solve(sigma_22, X2 - mu2))

    sigma_cond = sigma_11 - jnp.einsum('xyz,xzt->xyt', sigma_12,
                                       jnp.linalg.solve(sigma_22, sigma_21))

    # print(mu_cond)
    # print(sigma_cond)

    return _log_likelihood(X1, pi, mu_cond, sigma_cond)


log_clikelihood = jit(
    vmap(_log_clikelihood,
         in_axes=(0, 0, None, None, None, None, None, None, None)))
d_log_clikelihood = jit(
    vmap(grad(_log_clikelihood, argnums=0),
         in_axes=(0, 0, None, None, None, None, None, None, None)))

# TODO: conditional likelihood
# TODO: sample
# TODO: OOP wrapper
