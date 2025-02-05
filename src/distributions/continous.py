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



class Model:
    def __init__(self) -> None:
        self.components: list[LogDensityFn] = []
        self.param_sizes: list[int] = []
        self._param_slices: list[slice] = []

    def param(self, size: int = 1) -> slice:
        start = sum(self.param_sizes)
        self.param_sizes.append(size)
        param_slice = slice(start, start + size)
        self._param_slices.append(param_slice)
        return param_slice

    def add(self, log_prob_fn: LogDensityFn) -> 'Model':
        self.components.append(log_prob_fn)
        return self

    def __call__(self, params: Array) -> Array:
        return jnp.sum(jnp.array([component(params) for component in self.components]))

