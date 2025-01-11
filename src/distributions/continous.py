import jax.numpy as jnp
from jax.scipy import stats


class Distribution:
    """Base class for probability distributions."""

    def __init__(self, **params):
        self.parameters = params

    def log_prob(self, value, **params):
        raise NotImplementedError


class Normal(Distribution):
    """Normal distribution."""

    def __init__(self, loc, scale):
        super().__init__(loc=loc, scale=scale)

    def log_prob(self, value, loc, scale):
        return jnp.sum(stats.norm.logpdf(value, loc, scale))


class HalfNormal(Distribution):
    """Half-Normal distribution."""

    def __init__(self, scale):
        super().__init__(scale=scale)

    def log_prob(self, value, scale):
        return jnp.sum(stats.halfnorm.logpdf(value, scale=scale))