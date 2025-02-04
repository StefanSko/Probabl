import jax
import jax.numpy as jnp
from jaxtyping import Array, Float

from distributions.continous import exp, normal
from inference.samplers import nuts_with_warmup

# Simulate some observed data (replace this with your actual data)
rng_key = jax.random.PRNGKey(0)
key_data, key_init = jax.random.split(rng_key)

true_mu, true_sigma = 1.0, 2.0
n_samples = 1000

observed_data = jax.random.normal(key_data, shape=(n_samples,)) * true_sigma + true_mu


def posterior_log_prob(params: Float[Array, "2"]) -> Float[Array, ""]:
    mean, std = params  # we sample log_std for numerical stability

    # Prior contributions (both N(0,1))
    prior_mean = normal(0., 1.)(mean)
    prior_std = exp()(std)

    # Likelihood contribution
    likelihood = normal(mean, std)(observed_data)

    return prior_mean + prior_std + likelihood


# Split the key for initialization
key_init, key_sampling = jax.random.split(key_init)
initial_position = jnp.array([0., 0.1])

# Run the sampler
samples = nuts_with_warmup(
    posterior_log_prob,
    initial_position,
    key_sampling,
    num_samples=2000
)

# Transform samples back (for std)
mean_samples = samples[:, 0]
std_samples = samples[:, 1]

# Compute posterior statistics
print(f"Posterior mean (mean): {jnp.mean(mean_samples):.3f} ± {jnp.std(mean_samples):.3f}")
print(f"Posterior mean (std): {jnp.mean(std_samples):.3f} ± {jnp.std(std_samples):.3f}")