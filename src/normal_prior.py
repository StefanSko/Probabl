import blackjax
import jax
import jax.numpy as jnp
from blackjax.base import State

from distributions import LocationScaleParams, normal_distribution
from inference.samplers import nuts_with_warmup

# Create log density function for the prior
log_density_fn = normal_distribution.log_prob(
    LocationScaleParams(loc=jnp.array([0.]), scale=jnp.array([1.]))
)

# Build the kernel
step_size = 1e-3
inverse_mass_matrix = jnp.array([1.])
nuts = blackjax.nuts(log_density_fn, step_size, inverse_mass_matrix)

# Initialize a random position
rng_key = jax.random.PRNGKey(0)
initial_position = jax.random.normal(rng_key, shape=(1,))

# Run the sampler
n_samples = 1000
initial_state: State = nuts.init(initial_position)

# Use our nuts_with_warmup function
samples = nuts_with_warmup(
    normal_distribution.log_prob(LocationScaleParams(loc=0, scale=1)),
    initial_position,
    jax.random.PRNGKey(1000)
)

empirical_mean = jnp.mean(samples, axis=0)
empirical_std = jnp.std(samples, axis=0)

# add print statements
print(f"Empirical mean: {empirical_mean}")
print(f"Empirical std: {empirical_std}")
