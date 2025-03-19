"""Example demonstrating the context-aware data framework."""
import jax
import jax.numpy as jnp
from jaxtyping import Array

from src.distributions import (
    LocationScaleParams,
    normal_distribution,
)
from src.distributions.distribution import data_from_distribution
from src.inference import (
    DataContext,
    context,
    make_context_aware_data,
    nuts_with_warmup,
)


def main() -> None:
    """Run a simple example demonstrating context-aware data functions."""
    # Set up random keys
    key = jax.random.PRNGKey(0)
    key_data, key_prior, key_posterior, key_init = jax.random.split(key, 4)

    # Create some observed data
    true_mu, true_sigma = 2.0, 1.5
    observed_data = jax.random.normal(key_data, (100,)) * true_sigma + true_mu

    # Create data functions for different contexts
    def observed_data_fn() -> Array:
        return observed_data

    # Create prior simulator using our new distribution class
    prior_data_fn = data_from_distribution(
        normal_distribution,
        LocationScaleParams(loc=0.0, scale=1.0),
        key_prior,
        (100,)
    )

    # Create a context-aware data function
    context_aware_data = make_context_aware_data(
        observed_data=observed_data_fn,
        prior_simulator=prior_data_fn,
    )

    # Demonstrate that it works in different contexts
    with context(DataContext.INFERENCE):
        inference_data = context_aware_data()
        print("Inference data (first 5 elements):", inference_data[:5])
        print("Inference data mean:", jnp.mean(inference_data))

    with context(DataContext.PRIOR_PREDICTIVE):
        prior_pred_data = context_aware_data()
        print("\nPrior predictive data (first 5 elements):", prior_pred_data[:5])
        print("Prior predictive data mean:", jnp.mean(prior_pred_data))

    # Define a model using the context-aware data
    def model_log_prob(params: Array) -> Array:
        mean, std = params

        # Prior contributions
        prior_mean = normal_distribution.log_prob(LocationScaleParams(loc=0.0, scale=10.0))(mean)
        prior_std = normal_distribution.log_prob(LocationScaleParams(loc=0.0, scale=10.0))(std)

        # Likelihood - uses context-aware data
        with context(DataContext.INFERENCE):
            data = context_aware_data()
            likelihood = normal_distribution.log_prob(LocationScaleParams(loc=mean, scale=std))(data)

        return prior_mean + prior_std + likelihood

    # Run inference
    initial_params = jnp.array([0.0, 1.0])
    samples = nuts_with_warmup(model_log_prob, initial_params, key_init, num_samples=1000)

    # Extract samples
    mean_samples = samples[:, 0]
    std_samples = samples[:, 1]

    # Print results
    print("\nPosterior estimates:")
    print(f"Mean: {jnp.mean(mean_samples):.2f} ± {jnp.std(mean_samples):.2f}")
    print(f"Std: {jnp.mean(std_samples):.2f} ± {jnp.std(std_samples):.2f}")
    print(f"True values - Mean: {true_mu}, Std: {true_sigma}")


if __name__ == "__main__":
    main()
