"""Linear regression example with the context-aware Bayesian workflow."""
from dataclasses import dataclass
from typing import Callable
import sys
import os

# Add the parent directory to path so we can import our modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
from jaxtyping import Array, Float

from distributions import (
    BaseParams,
    EnhancedModelBuilder,
    LocationScaleParams,
    normal_distribution,
)
from distributions.distribution import data_from_distribution
from inference import (
    BayesianWorkflow,
    ConstraintRegistry,
    DataContext,
    ParameterStructure,
    context,
    flatten_params,
    transform_params,
    unflatten_params,
)


@dataclass(frozen=True)
class LinearRegressionParams(BaseParams):
    """Parameters for linear regression model."""
    
    intercept: Float[Array, " "]
    slope: Float[Array, " "]
    log_noise: Float[Array, " "]
    
    @property
    def noise(self) -> Float[Array, " "]:
        """Get the noise parameter (standard deviation)."""
        return jnp.exp(self.log_noise)


def generate_data(
    true_intercept: float, 
    true_slope: float, 
    noise_scale: float,
    n_samples: int,
    x_min: float = -3.0,
    x_max: float = 3.0,
    rng_key: jax.Array = None,
) -> tuple[Array, Array]:
    """Generate synthetic data for linear regression."""
    if rng_key is None:
        rng_key = jax.random.PRNGKey(0)
    
    # Generate X values
    x = jnp.linspace(x_min, x_max, n_samples)
    
    # Generate noise
    noise = jax.random.normal(rng_key, shape=(n_samples,)) * noise_scale
    
    # Generate Y values
    y = true_intercept + true_slope * x + noise
    
    return x, y


def main() -> None:
    """Run a linear regression example with the Bayesian workflow."""
    # Set up random keys
    key = jax.random.PRNGKey(0)
    key_data, key_prior, key_workflow = jax.random.split(key, 3)
    
    # Generate synthetic data
    true_intercept = 2.0
    true_slope = 0.7
    true_noise = 0.5
    n_samples = 100
    
    x_data, y_data = generate_data(
        true_intercept, true_slope, true_noise, n_samples, rng_key=key_data
    )
    
    # Plot the data
    plt.figure(figsize=(10, 6))
    plt.scatter(x_data, y_data, alpha=0.7, label="Observed Data")
    
    # Add true regression line
    x_line = jnp.linspace(x_data.min(), x_data.max(), 100)
    y_line = true_intercept + true_slope * x_line
    plt.plot(x_line, y_line, 'r-', label=f"True: y = {true_intercept:.1f} + {true_slope:.1f}x")
    
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.legend()
    plt.title("Linear Regression Data")
    plt.show()
    
    # Create observed data function
    def observed_data_fn() -> Array:
        return y_data
    
    # Create prior simulator
    prior_intercept = normal_distribution.log_prob(
        LocationScaleParams(loc=0.0, scale=10.0)
    )
    prior_slope = normal_distribution.log_prob(
        LocationScaleParams(loc=0.0, scale=10.0)
    )
    prior_noise = normal_distribution.log_prob(
        LocationScaleParams(loc=0.0, scale=1.0)
    )
    
    # Function to generate simulated data from parameters
    def simulate_data(params: LinearRegressionParams) -> Array:
        # Generate predictions
        predictions = params.intercept + params.slope * x_data
        
        # Add noise
        key_sim = jax.random.PRNGKey(0)  # Fixed seed for reproducibility
        noise = jax.random.normal(key_sim, shape=(n_samples,)) * params.noise
        
        return predictions + noise
    
    # Create initial parameters for the model
    initial_params = LinearRegressionParams(
        intercept=jnp.array(0.0),
        slope=jnp.array(0.0),
        log_noise=jnp.array(0.0),
    )
    
    # Create model builder with auto-simulation capabilities
    model_builder = (EnhancedModelBuilder[LinearRegressionParams]()
        # Add observed data
        .with_observed_data(observed_data_fn)
        
        # Add the random key
        .with_rng_key(jax.random.PRNGKey(42))
        
        # Store data shape for auto-simulation
        .with_data_shape(y_data.shape)
        
        # Define parametric density function
        .with_parametric_density_fn(
            lambda params: (
                lambda data: (
                    # Prior contributions
                    normal_distribution.log_prob(
                        LocationScaleParams(loc=0.0, scale=10.0)
                    )(params.intercept) +
                    
                    normal_distribution.log_prob(
                        LocationScaleParams(loc=0.0, scale=10.0)
                    )(params.slope) +
                    
                    normal_distribution.log_prob(
                        LocationScaleParams(loc=0.0, scale=1.0)
                    )(params.log_noise) +
                    
                    # Likelihood contribution
                    normal_distribution.log_prob(
                        LocationScaleParams(
                            loc=params.intercept + params.slope * x_data, 
                            scale=params.noise
                        )
                    )(data)
                )
            )
        )
        
        # Use auto-prior simulator
        .with_auto_prior_simulator(
            default_params=initial_params,
            data_shape=y_data.shape
        )
        
        # Auto-generate the posterior simulator too
        .with_posterior_data_simulator(
            lambda params: (
                params.intercept + params.slope * x_data + 
                jax.random.normal(jax.random.PRNGKey(0), shape=y_data.shape) * params.noise
            )
        )
    )
    
    # Build the model
    model = model_builder.build()
    
    # Create the workflow manager
    workflow = BayesianWorkflow(model, rng_key=key_workflow)
    
    # Create initial parameters
    initial_params = LinearRegressionParams(
        intercept=jnp.array(0.0),
        slope=jnp.array(0.0),
        log_noise=jnp.array(0.0),
    )
    
    # Create parameter structure for flattening/unflattening
    param_structure = ParameterStructure.from_params(initial_params)
    
    # For demonstration purposes, let's just do the inference part
    # instead of the full workflow, to avoid circular dependency issues
    posterior_samples = workflow.run_inference(
        initial_params=initial_params,
        num_samples=2000,
        flatten_fn=param_structure.flatten,
        unflatten_fn=lambda x: param_structure.unflatten(x),
    )
    
    # Store and extract the posterior samples
    # Extract the posterior samples directly (no need for results object)
    print(f"Obtained {len(posterior_samples)} posterior samples")
    
    # Convert to parameter objects
    posterior_params = [param_structure.unflatten(sample) for sample in posterior_samples]
    
    # Extract parameters
    intercept_samples = jnp.array([p.intercept for p in posterior_params])
    slope_samples = jnp.array([p.slope for p in posterior_params])
    noise_samples = jnp.array([p.noise for p in posterior_params])
    
    # Print results
    print("\nPosterior Parameter Estimates:")
    print(f"Intercept: {jnp.mean(intercept_samples):.2f} ± {jnp.std(intercept_samples):.2f}")
    print(f"Slope: {jnp.mean(slope_samples):.2f} ± {jnp.std(slope_samples):.2f}")
    print(f"Noise: {jnp.mean(noise_samples):.2f} ± {jnp.std(noise_samples):.2f}")
    
    # Plot posterior regression lines
    plt.figure(figsize=(10, 6))
    plt.scatter(x_data, y_data, alpha=0.5, label="Observed Data")
    
    # Add true regression line
    plt.plot(
        x_line, y_line, 'r-', linewidth=2,
        label=f"True: y = {true_intercept:.1f} + {true_slope:.1f}x"
    )
    
    # Add posterior regression lines
    for i in range(0, min(2000, len(posterior_params)), 100):
        param = posterior_params[i]
        y_posterior = param.intercept + param.slope * x_line
        plt.plot(x_line, y_posterior, 'g-', alpha=0.1)
    
    # Add posterior mean
    mean_intercept = jnp.mean(intercept_samples)
    mean_slope = jnp.mean(slope_samples)
    y_posterior_mean = mean_intercept + mean_slope * x_line
    
    plt.plot(
        x_line, y_posterior_mean, 'b-', linewidth=2,
        label=f"Posterior Mean: y = {mean_intercept:.1f} + {mean_slope:.1f}x"
    )
    
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.legend()
    plt.title("Linear Regression with Posterior Samples")
    plt.show()
    
    # Plot parameter distributions
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Intercept
    axes[0].hist(intercept_samples, bins=30)
    axes[0].axvline(true_intercept, color='r', linestyle='dashed')
    axes[0].set_title(f"Intercept (True: {true_intercept})")
    
    # Slope
    axes[1].hist(slope_samples, bins=30)
    axes[1].axvline(true_slope, color='r', linestyle='dashed')
    axes[1].set_title(f"Slope (True: {true_slope})")
    
    # Noise
    axes[2].hist(noise_samples, bins=30)
    axes[2].axvline(true_noise, color='r', linestyle='dashed')
    axes[2].set_title(f"Noise (True: {true_noise})")
    
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()