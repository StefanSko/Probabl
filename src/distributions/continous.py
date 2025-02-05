from dataclasses import dataclass
from typing import Callable, TypeAlias

import jax.numpy as jnp
from jax.scipy import stats
from jaxtyping import Array

# Type aliases for Bayesian probability functions
LogDensityFn: TypeAlias = Callable[[Array], Array]  # log p(x|θ)
ParametricDensityFn: TypeAlias = Callable[[Array | float, Array | float], LogDensityFn] # p(x|θ)
LogPDFFn: TypeAlias = Callable[[Array, Array, Array], Array]  # raw PDF computation

def make_distribution(
    logpdf_fn: LogPDFFn
) -> ParametricDensityFn:
    def distribution(loc: Array | float = 0.0, scale: Array | float = 1.0) -> LogDensityFn:
        def log_prob(data: Array) -> Array:
            return jnp.sum(logpdf_fn(data, jnp.array(loc), jnp.array(scale)))
        return log_prob
    return distribution


# Standard distributions with loc and scale
normal = make_distribution(stats.norm.logpdf)
laplace = make_distribution(stats.laplace.logpdf)
cauchy = make_distribution(stats.cauchy.logpdf)
exp = make_distribution(stats.expon.logpdf)


# Custom parameter distributions
def beta(a: Array, b: Array) -> LogDensityFn:
    def log_prob(data: Array) -> Array:
        return jnp.sum(stats.beta.logpdf(data, a, b))
    return log_prob

def gamma(shape: Array, scale: Array) -> LogDensityFn:
    def log_prob(data: Array) -> Array:
        return jnp.sum(stats.gamma.logpdf(data, shape, scale=scale))
    return log_prob
