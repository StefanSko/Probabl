import jax
import jax.numpy as jnp
from jaxtyping import Array, Float

from distributions.continous import normal, exp
from inference.samplers import nuts_with_warmup

# Simulate some observed data (replace this with your actual data)
rng_key = jax.random.PRNGKey(0)

true_mu, true_sigma = 1.0, 2.0
n_samples = 1000

observed_data = jax.random.normal(rng_key, shape=(n_samples,)) * true_sigma + true_mu


def posterior_log_prob(params: Float[Array, "2"]) -> Float[Array, ""]:
    mean, log_std = params  # we sample log_std for numerical stability
    std = jnp.exp(log_std)

    # Prior contributions (both N(0,1))
    prior_mean = normal(0., 1.)(mean)
    prior_std = exp(0., 1.)(log_std)

    # Likelihood contribution
    likelihood = normal(mean, std)(observed_data)

    return prior_mean + prior_std + likelihood


# Initialize from random position
rng_key = jax.random.PRNGKey(1000)
initial_position = jnp.array([0., 0.])  # starting guess for [mean, log_std]

# Run the sampler
samples = nuts_with_warmup(
    posterior_log_prob,
    initial_position,
    rng_key,
    num_samples=2000
)

# Transform samples back (for std)
mean_samples = samples[:, 0]
std_samples = jnp.exp(samples[:, 1])

# Compute posterior statistics
print(f"Posterior mean (mean): {jnp.mean(mean_samples):.3f} ± {jnp.std(mean_samples):.3f}")
print(f"Posterior mean (std): {jnp.mean(std_samples):.3f} ± {jnp.std(std_samples):.3f}")