import jax.numpy as jnp
from jax.scipy import stats
from typing import Any
from jax.Array import Array


class Distribution:
    """Base class for probability distributions."""
    parameters: dict[str, Any]

    def __init__(self, **params: Any) -> None:
        self.parameters = params

    def log_prob(self, value: Array, loc: Array, scale: Array) -> Array:
        raise NotImplementedError


class Normal(Distribution):
    """Normal distribution."""

    def __init__(self, loc: Array, scale: Array) -> None:
        super().__init__(loc=loc, scale=scale)

    def log_prob(self, value: Array, loc: Array, scale: Array) -> Array:
        return jnp.sum(stats.norm.logpdf(value, loc, scale))


class Beta(Distribution):
    """Beta distribution."""

    def __init__(self, a: Array, b: Array) -> None:
        super().__init__(a=a, b=b)

    def log_prob(self, value: Array, a: Array, b: Array) -> Array:
        return jnp.sum(stats.beta.logpdf(value, a, b))


class Gamma(Distribution):
    """Gamma distribution."""

    def __init__(self, shape: Array, scale: Array) -> None:
        super().__init__(shape=shape, scale=scale)

    def log_prob(self, value: Array, shape: Array, scale: Array) -> Array:
        return jnp.sum(stats.gamma.logpdf(value, shape, scale=scale))


class Exponential(Distribution):
    """Exponential distribution."""

    def __init__(self, scale: Array) -> None:
        super().__init__(scale=scale)

    def log_prob(self, value: Array, loc: Array, scale: Array) -> Array:
        return jnp.sum(stats.expon.logpdf(value, scale=scale))
