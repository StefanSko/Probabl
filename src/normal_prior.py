import blackjax
import jax
import jax.numpy as jnp
from blackjax.base import State
from jaxtyping import Array, Float

from distributions.continous import normal
from inference.samplers import nuts_with_warmup

# Build the kernel
step_size = 1e-3
inverse_mass_matrix = jnp.array([1.])
mean = jnp.array([0.])
var = jnp.array([1.])
nuts = blackjax.nuts(normal(mean, var), step_size, inverse_mass_matrix)

# Initialize a random position
rng_key = jax.random.PRNGKey(0)
initial_position = jax.random.normal(rng_key, shape=(1,))

# Run the sampler
n_samples = 1000
initial_state: State = nuts.init(initial_position)

samples = nuts_with_warmup(normal(0,1), initial_position, jax.random.PRNGKey(1000))

empirical_mean = jnp.mean(samples, axis=0)
empirical_std = jnp.std(samples, axis=0)

# add print statements
print(f"Empirical mean: {empirical_mean}")
print(f"Empirical std: {empirical_std}")




