from typing import Callable

import jax.numpy as jnp
from jax.scipy import stats
from jaxtyping import Array


def make_distribution(logpdf_fn: Callable) -> Callable:
    def distribution(loc: Array, scale: Array) -> Callable[[Array], Array]:
        def log_prob(data: Array) -> Array:
            return jnp.sum(logpdf_fn(data, loc, scale))
        return log_prob
    return distribution


# Standard distributions with loc and scale
normal = make_distribution(stats.norm.logpdf)
laplace = make_distribution(stats.laplace.logpdf)
cauchy = make_distribution(stats.cauchy.logpdf)


# Custom parameter distributions
def beta(a: Array, b: Array) -> Callable[[Array], Array]:
    def log_prob(data: Array) -> Array:
        return jnp.sum(stats.beta.logpdf(data, a, b))
    return log_prob

def gamma(shape: Array, scale: Array) -> Callable[[Array], Array]:
    def log_prob(data: Array) -> Array:
        return jnp.sum(stats.gamma.logpdf(data, shape, scale=scale))
    return log_prob
