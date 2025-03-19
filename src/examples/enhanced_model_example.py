"""Example demonstrating the enhanced model builder and parameter handling."""
from dataclasses import dataclass
import sys
import os

# Add the parent directory to path so we can import our modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import jax
import jax.numpy as jnp
from jaxtyping import Array, Float

from distributions import (
    BaseParams,
    EnhancedModelBuilder,
    LocationScaleParams,
    normal_distribution,
)
from inference import (
    ConstraintRegistry,
    DataContext,
    context,
    flatten_params,
    nuts_with_warmup,
    transform_params,
    unflatten_params,
)


@dataclass(frozen=True)
class LinearRegressionParams(BaseParams):
    """Parameters for linear regression model."""

    intercept: Float[Array, ""]
    slope: Float[Array, ""]
    log_scale: Float[Array, ""]

    @property
    def scale(self) -> Float[Array, ""]:
        """Get the scale parameter (standard deviation)."""
        return jnp.exp(self.log_scale)


def generate_data(
    true_intercept: float,
    true_slope: float,
    noise_scale: float,
    n_samples: int,
    rng_key: jax.Array
) -> tuple[Array, Array]:
    """Generate synthetic data for linear regression."""
    x = jnp.linspace(-5, 5, n_samples)
    key1, key2 = jax.random.split(rng_key)
    noise = jax.random.normal(key1, shape=(n_samples,)) * noise_scale
    y = true_intercept + true_slope * x + noise
    return x, y


def main() -> None:
    """Run a linear regression example with the enhanced model builder."""
    # Set up random keys
    key = jax.random.PRNGKey(0)
    key_data, key_prior, key_init = jax.random.split(key, 3)

    # Generate synthetic data
    true_intercept = 2.0
    true_slope = 0.5
    true_scale = 1.0
    n_samples = 100

    x_data, y_data = generate_data(true_intercept, true_slope, true_scale, n_samples, key_data)

    # Create observed data function
    def observed_data_fn() -> Array:
        return y_data

    # Create model builder
    model_builder = EnhancedModelBuilder[LinearRegressionParams]()

    # Add observed data
    model_builder.with_observed_data(observed_data_fn)

    # Define prior parameters for reference (not used in this example)
    # Removed unused variable to satisfy linter

    # Create parametric density function
    def parametric_density_fn(params: LinearRegressionParams):
        def log_prob(data: Array) -> Array:
            # Prior contributions
            prior_intercept = normal_distribution.log_prob(
                LocationScaleParams(loc=0.0, scale=10.0)
            )(params.intercept)

            prior_slope = normal_distribution.log_prob(
                LocationScaleParams(loc=0.0, scale=10.0)
            )(params.slope)

            prior_log_scale = normal_distribution.log_prob(
                LocationScaleParams(loc=0.0, scale=1.0)
            )(params.log_scale)

            # Likelihood contribution
            predicted = params.intercept + params.slope * x_data
            likelihood = normal_distribution.log_prob(
                LocationScaleParams(loc=predicted, scale=params.scale)
            )(data)

            return prior_intercept + prior_slope + prior_log_scale + likelihood

        return log_prob

    # Add parametric density function
    model_builder.with_parametric_density_fn(parametric_density_fn)

    # Build the model
    model = model_builder.build()

    # Create initial parameters
    initial_params = LinearRegressionParams(
        intercept=jnp.array(0.0),
        slope=jnp.array(0.0),
        log_scale=jnp.array(0.0),
    )

    # Flatten parameters and get structure
    flat_params, param_structure = flatten_params(initial_params)

    # Define constraints
    constraints = {
        "log_scale": ConstraintRegistry.positive(),
    }

    # Apply constraints
    transformed_params, log_det_fn = transform_params(initial_params, constraints)

    # Define the posterior log probability function
    def posterior_log_prob(flat_transformed_params: Array) -> Array:
        # Unflatten parameters
        transformed_params = unflatten_params(flat_transformed_params, param_structure)

        # Forward the model with parameters
        log_prob = model.forward(transformed_params)

        # Add log determinant of the Jacobian for the transform
        log_prob = log_prob + log_det_fn(transformed_params)

        return log_prob

    # Run inference
    flat_transformed_initial = param_structure.flatten(transformed_params)
    samples = nuts_with_warmup(
        posterior_log_prob,
        flat_transformed_initial,
        key_init,
        num_samples=1000,
    )

    # Process results
    transformed_samples = jnp.array([
        unflatten_params(flat_sample, param_structure)
        for flat_sample in samples
    ])

    # Extract parameters
    intercept_samples = jnp.array([sample.intercept for sample in transformed_samples])
    slope_samples = jnp.array([sample.slope for sample in transformed_samples])
    scale_samples = jnp.array([sample.scale for sample in transformed_samples])

    # Print results
    print("True parameters:")
    print(f"Intercept: {true_intercept:.2f}")
    print(f"Slope: {true_slope:.2f}")
    print(f"Scale: {true_scale:.2f}")

    print("\nPosterior estimates:")
    print(f"Intercept: {jnp.mean(intercept_samples):.2f} ± {jnp.std(intercept_samples):.2f}")
    print(f"Slope: {jnp.mean(slope_samples):.2f} ± {jnp.std(slope_samples):.2f}")
    print(f"Scale: {jnp.mean(scale_samples):.2f} ± {jnp.std(scale_samples):.2f}")

    # Demonstrate context switching
    with context(DataContext.INFERENCE):
        print("\nInference context data (first 5 elements):", model.data()[:5])


if __name__ == "__main__":
    main()
