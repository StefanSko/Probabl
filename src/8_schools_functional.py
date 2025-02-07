import jax
import jax.numpy as jnp
from jaxtyping import Array, Float

from distributions.continous import normal
from inference.samplers import nuts_with_warmup

# The 8 schools data
treatment_effects = jnp.array([28., 8., -3., 7., -1., 1., 18., 12.])
standard_errors = jnp.array([15., 10., 16., 11., 9., 11., 10., 18.])
n_schools = len(treatment_effects)

def posterior_log_prob(params: Float[Array, "10"]) -> Float[Array, ""]:
    # Unpack parameters
    mu = params[0]          # population mean
    log_tau = params[1]     # log population standard deviation
    theta = params[2:]      # school-specific effects

    # Transform tau
    tau = jnp.exp(log_tau)

    # Priors
    # Weakly informative priors for mu and tau
    prior_mu = normal(0., 1.)(mu)
    prior_tau = normal(5., 1.)(log_tau)  # prior on log_tau

    # Hierarchical prior for school effects
    prior_theta = normal(mu, tau)(theta)

    # Likelihood for observed effects
    likelihood = normal(theta, standard_errors)(treatment_effects)

    return prior_mu + prior_tau + prior_theta + likelihood

# Initialize from random position
rng_key = jax.random.PRNGKey(1000)
initial_position = jnp.zeros(10)  # [mu, log_tau, theta(8)]

# Run the sampler
samples = nuts_with_warmup(
    posterior_log_prob,
    initial_position,
    rng_key,
    num_samples=2000
)

# Extract parameters
mu_samples = samples[:, 0]
tau_samples = jnp.exp(samples[:, 1])
theta_samples = samples[:, 2:]

# Print results
print("Population parameters:")
print(f"mu: {jnp.mean(mu_samples):.1f} ± {jnp.std(mu_samples):.1f}")
print(f"tau: {jnp.mean(tau_samples):.1f} ± {jnp.std(tau_samples):.1f}")

print("\nSchool-specific effects:")
for i in range(n_schools):
    school_samples = theta_samples[:, i]
    print(f"School {i+1}: {jnp.mean(school_samples):.1f} ± {jnp.std(school_samples):.1f}")


# Visualize the posterior distributions
import seaborn as sns
from matplotlib import pyplot as plt

# Create swarm plot of posterior samples
plt.figure(figsize=(10, 6))
# Convert to numpy and thin samples for visualization
sns.swarmplot(data=theta_samples[::10,:], s=2, zorder=0)
plt.show()
