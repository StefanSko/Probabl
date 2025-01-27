import blackjax
import jax
import jax.numpy as jnp
from blackjax.base import State
from jaxtyping import Array, Float

from distributions.continous import normal
from inference.samplers import sample_nuts

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

# Generate samples
#samples = inference_loop(jax.random.PRNGKey(1), initial_state)


samples = sample_nuts(normal(0,1), initial_position, 1000, 42)

empirical_mean = jnp.mean(samples, axis=0)
empirical_std = jnp.std(samples, axis=0)

# add print statements
print(f"Empirical mean: {empirical_mean}")
print(f"Empirical std: {empirical_std}")




