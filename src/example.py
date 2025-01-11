from typing import Callable

import blackjax
import jax
import jax.numpy as jnp
from blackjax.base import State
from jax.scipy import stats
from jaxtyping import Array, Float


class RandomVariable:
    def __init__(
        self,
        logpdf: Callable[[Array], Float[Array, ""]],
        sample: Float[Array, "..."] | None = None
    ) -> None:
        self.logpdf = logpdf
        self.sample = sample if sample is not None else jnp.array(0.0)
        self._dependencies: list[RandomVariable] = []

    def __add__(self, other: "RandomVariable | int | float") -> "RandomVariable":
        if isinstance(other, (int, float)):
            other = RandomVariable(lambda _: jnp.array(0.0), jnp.array(float(other)))

        result = RandomVariable(
            lambda data: self.logpdf(data - other.sample) + other.logpdf(other.sample),
            self.sample + other.sample
        )
        result._dependencies = self._dependencies + other._dependencies
        return result

    def __mul__(self, other: "RandomVariable | int | float") -> "RandomVariable":
        if isinstance(other, (int, float)):
            other = RandomVariable(lambda _: jnp.array(0.0), jnp.array(float(other)))

        result = RandomVariable(
            lambda data: self.logpdf(data / other.sample) + other.logpdf(other.sample),
            self.sample * other.sample
        )
        result._dependencies = self._dependencies + other._dependencies
        return result

    def __radd__(self, other: "RandomVariable | int | float") -> "RandomVariable":
        return self.__add__(other)

    def __rmul__(self, other: "RandomVariable | int | float") -> "RandomVariable":
        return self.__mul__(other)

    def get_model_logpdf(self) -> Callable[[Float[Array, "..."]], Float[Array, ""]]:
        def model_logpdf(params: Float[Array, "..."]) -> Float[Array, ""]:
            total_logp = jnp.array(0.0)
            # Add this variable's logpdf
            total_logp += self.logpdf(params)
            # Add all dependencies' logpdfs
            for dep in self._dependencies:
                total_logp += dep.logpdf(dep.sample)
            return total_logp
        return model_logpdf

def normal(
    loc: "RandomVariable | Array | float",
    scale: "RandomVariable | Array | float"
) -> RandomVariable:
    if isinstance(loc, (float, Array)):
        loc = RandomVariable(lambda _: jnp.array(0.0), jnp.array(float(loc)))
    if isinstance(scale, (float, Array)):
        scale = RandomVariable(lambda _: jnp.array(0.0), jnp.array(float(scale)))

    rv = RandomVariable(
        lambda data: jnp.sum(stats.norm.logpdf(data, loc.sample, scale.sample))
    )
    # Include the parameters' dependencies as well
    rv._dependencies = loc._dependencies + scale._dependencies + [loc, scale]
    return rv

def binomial(
    n: int,
    p: "RandomVariable | Array | float"
) -> RandomVariable:
    if isinstance(p, (float, Array)):
        p = RandomVariable(lambda _: jnp.array(0.0), jnp.array(float(p)))

    rv = RandomVariable(
        lambda data: jnp.sum(stats.binom.logpmf(data, n, p.sample))
    )
    # Include the parameter's dependencies
    rv._dependencies = p._dependencies + [p]
    return rv

def invlogit(x: Array) -> Array:
    return 1.0 / (1.0 + jnp.exp(-x))

# Generate features
n_samples = 1000  # or whatever size you need
u = normal(0.0, 1.0)  # Primary feature
v = normal(0.0, 1.0)  # Secondary feature

# Build the model
mu = -3 + 0.6 * u + 0.4 * v
sigma = normal(0.0, 0.5)  # Make the standard deviation a random variable
scores = normal(mu, sigma)  # Both location and scale can be random variables

# Generate scores and probabilities
probs = RandomVariable(lambda _: jnp.array(0.0), invlogit(scores.sample))
conversions = binomial(n=1, p=probs)  # p can be a RandomVariable

# Get the combined logpdf for blackjax
model_logpdf = conversions.get_model_logpdf()

# Setup blackjax
step_size = 1e-3
inverse_mass_matrix = jnp.array([1.])
nuts = blackjax.nuts(model_logpdf, step_size, inverse_mass_matrix)

# Initialize and run the sampler
rng_key = jax.random.PRNGKey(0)
initial_position = jax.random.normal(rng_key, shape=(1,))
initial_state = nuts.init(initial_position)

def inference_loop(
    rng_key: jax.Array,
    initial_state: State
) -> Float[Array, "n_samples 1"]:
    @jax.jit
    def one_step(
        state: State,
        rng_key: jax.Array
    ) -> tuple[State, State]:
        state, _ = nuts.step(rng_key, state)
        return state, state

    keys = jax.random.split(rng_key, n_samples)
    final_state, states = jax.lax.scan(one_step, initial_state, keys)
    return states.position

# Use the existing inference_loop function
samples = inference_loop(jax.random.PRNGKey(1), initial_state)

print('stop')




