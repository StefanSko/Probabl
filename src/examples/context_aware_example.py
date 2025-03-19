"""Example showing the enhanced model builder with auto-simulation capabilities."""
from dataclasses import dataclass
from typing import Dict, Any, List
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
from inference import DataContext, context

@dataclass(frozen=True)
class SimpleHierarchicalParams(BaseParams):
    """Parameters for a simple hierarchical model."""
    
    # Global parameters
    mu: Float[Array, " "]       # Global mean
    log_sigma: Float[Array, " "] # Log of observation noise
    
    # Group-specific parameters
    group_means: Float[Array, "n_groups"]
    
    @property
    def sigma(self) -> Float[Array, " "]:
        """Get the sigma parameter (observation noise)."""
        return jnp.exp(self.log_sigma)

def main() -> None:
    """Run a simple example with the enhanced model builder."""
    # Set up random key
    key = jax.random.PRNGKey(0)
    key_data, key_model = jax.random.split(key)
    
    # Generate synthetic data
    n_groups = 4
    n_per_group = 20
    
    # True parameters
    true_mu = 2.0
    true_sigma = 0.8
    
    # Generate group means
    key_means = jax.random.fold_in(key_data, 0)
    true_group_means = true_mu + jax.random.normal(key_means, (n_groups,)) * 1.0
    
    # Generate data for each group
    all_data = []
    group_indices = []
    
    for i in range(n_groups):
        # Generate data for this group
        group_key = jax.random.fold_in(key_data, i + 1)
        
        # Group data (just random noise around the group mean)
        # In a real model, this could be more complex with covariates, etc.
        group_mean = true_group_means[i]
        group_data = group_mean + jax.random.normal(group_key, (n_per_group,)) * true_sigma
        
        # Store data
        start_idx = i * n_per_group
        end_idx = start_idx + n_per_group
        all_data.append(group_data)
        group_indices.append(jnp.arange(start_idx, end_idx))
    
    # Combine all data
    y_data = jnp.concatenate(all_data)
    print(f"Generated data shape: {y_data.shape}")
    print(f"True group means: {true_group_means}")
    
    # Create data function for the observed data
    def observed_data_fn() -> Array:
        return y_data
    
    # Create parametric density function (model)
    def parametric_density_fn(params: SimpleHierarchicalParams):
        def log_prob(data: Array) -> Array:
            # Global prior
            prior_mu = normal_distribution.log_prob(
                LocationScaleParams(loc=0.0, scale=10.0)
            )(params.mu)
            
            prior_log_sigma = normal_distribution.log_prob(
                LocationScaleParams(loc=0.0, scale=1.0)
            )(params.log_sigma)
            
            # Group means prior
            prior_group_means = normal_distribution.log_prob(
                LocationScaleParams(loc=params.mu, scale=1.0)
            )(params.group_means)
            
            # Likelihood for each group
            log_likelihood = 0.0
            
            for i in range(n_groups):
                # Get indices for this group
                indices = group_indices[i]
                
                # Get group data
                group_data = data[indices]
                
                # Group likelihood
                group_likelihood = normal_distribution.log_prob(
                    LocationScaleParams(loc=params.group_means[i], scale=params.sigma)
                )(group_data)
                
                # Accumulate likelihood
                log_likelihood = log_likelihood + group_likelihood
            
            # Total log probability
            return prior_mu + prior_log_sigma + jnp.sum(prior_group_means) + log_likelihood
        
        return log_prob
    
    # Create initial parameters (just for structure, not actual values)
    initial_params = SimpleHierarchicalParams(
        mu=jnp.array(0.0),
        log_sigma=jnp.array(0.0),
        group_means=jnp.zeros(n_groups),
    )
    
    # Create the model builder with auto-prior and auto-posterior simulation
    model_builder = (EnhancedModelBuilder[SimpleHierarchicalParams]()
        .with_observed_data(observed_data_fn)
        .with_parametric_density_fn(parametric_density_fn)
        .with_rng_key(key_model)
        .with_data_shape(y_data.shape)  # Add data shape for posterior simulation
        .with_hierarchical_data(
            x_data=jnp.ones_like(y_data),  # We don't have x_data in this simple example
            group_indices=group_indices
        )
        .with_auto_prior_simulator(
            default_params=initial_params,
            n_groups=n_groups,
            x_data=jnp.ones_like(y_data),  # Not used in this model, but required
            group_indices=group_indices
        )
    )
    
    # Manually add the posterior simulator function for this example
    def manual_posterior_simulator(params: SimpleHierarchicalParams) -> Array:
        # Initialize simulated data
        simulated_data = jnp.zeros_like(y_data)
        
        # Generate group-specific data
        for i in range(n_groups):
            indices = group_indices[i]
            mean = params.group_means[i]
            noise = jax.random.normal(key_model, shape=(len(indices),)) * params.sigma
            simulated_data = simulated_data.at[indices].set(mean + noise)
        
        return simulated_data
    
    # Add the posterior simulator
    model_builder = model_builder.with_posterior_data_simulator(manual_posterior_simulator)
    
    # Build the model
    model = model_builder.build()
    
    # Demonstrate the auto-generated simulators
    
    # Prior predictive simulation
    with context(DataContext.PRIOR_PREDICTIVE):
        # This uses the auto-generated prior simulator
        prior_samples = []
        for _ in range(10):
            sample = model.data()
            prior_samples.append(sample)
    
    print(f"\nPrior predictive samples shape: {prior_samples[0].shape}")
    print(f"First few values of prior sample: {prior_samples[0][:5]}")
    
    # Plot the prior samples vs. observed data
    plt.figure(figsize=(10, 6))
    
    # Plot the observed data
    plt.hist(y_data, bins=15, alpha=0.5, label='Observed Data', color='blue')
    
    # Plot a few prior samples
    for i, sample in enumerate(prior_samples[:3]):
        plt.hist(sample, bins=15, alpha=0.3, label=f'Prior Sample {i+1}', color=f'C{i+1}')
    
    plt.title('Prior Predictive Check: Auto-Generated Prior Simulator')
    plt.xlabel('Value')
    plt.ylabel('Count')
    plt.legend()
    plt.tight_layout()
    plt.show()
    
    # Demonstrate posterior simulator (with mock posterior params)
    # In a real workflow, these would come from MCMC
    mock_posterior_params = SimpleHierarchicalParams(
        mu=jnp.array(1.8),  # Close to true value of 2.0
        log_sigma=jnp.log(jnp.array(0.75)),  # Close to true value of 0.8
        group_means=jnp.array([1.5, 2.5, 1.8, 2.2]),  # Similar to true group means
    )
    
    # We need to pass the model a way to get the posterior samples
    # This would be handled by the workflow in a real scenario
    with context(DataContext.POSTERIOR_PREDICTIVE):
        # This uses the auto-generated posterior simulator
        # We manually provide the parameter since we're not running through the workflow
        data_fn = model._posterior_data_generator(mock_posterior_params)
        posterior_samples = []
        for _ in range(10):
            sample = data_fn()
            posterior_samples.append(sample)
    
    print(f"\nPosterior predictive samples shape: {posterior_samples[0].shape}")
    print(f"First few values of posterior sample: {posterior_samples[0][:5]}")
    
    # Plot the posterior samples vs. observed data
    plt.figure(figsize=(10, 6))
    
    # Plot the observed data
    plt.hist(y_data, bins=15, alpha=0.5, label='Observed Data', color='blue')
    
    # Plot a few posterior samples
    for i, sample in enumerate(posterior_samples[:3]):
        plt.hist(sample, bins=15, alpha=0.3, label=f'Posterior Sample {i+1}', color=f'C{i+1}')
    
    plt.title('Posterior Predictive Check: Auto-Generated Posterior Simulator')
    plt.xlabel('Value')
    plt.ylabel('Count')
    plt.legend()
    plt.tight_layout()
    plt.show()
    
    print("\nCompleted example with auto-generated simulators")

if __name__ == "__main__":
    main()
