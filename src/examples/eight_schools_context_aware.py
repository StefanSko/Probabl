"""Eight schools example refactored to use the context-aware framework."""
from dataclasses import dataclass
import sys
import os

# Add the parent directory to path so we can import our modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import seaborn as sns
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
    DataContext,
    ParameterStructure,
    context,
    flatten_params,
    unflatten_params,
)


@dataclass(frozen=True)
class HierarchicalParams(BaseParams):
    """Parameters for hierarchical model."""
    
    mu: Float[Array, " "]
    log_tau: Float[Array, " "]
    theta: Float[Array, "..."]  # Use ellipsis for arbitrary dimensions
    
    @property
    def tau(self) -> Float[Array, " "]:
        """Get the tau parameter (standard deviation)."""
        return jnp.exp(self.log_tau)


def main() -> None:
    """Run the 8 schools example with the context-aware Bayesian workflow."""
    # The 8 schools data
    treatment_effects = jnp.array([28., 8., -3., 7., -1., 1., 18., 12.])
    standard_errors = jnp.array([15., 10., 16., 11., 9., 11., 10., 18.])
    n_schools = len(treatment_effects)
    
    # Set up random keys
    key = jax.random.PRNGKey(0)
    key_prior, key_workflow = jax.random.split(key)
    
    # Create observed data function
    def observed_data_fn() -> Array:
        return treatment_effects
    
    # Create function to generate simulated data from parameters
    def simulate_data(params: HierarchicalParams) -> Array:
        # Generate simulated treatment effects
        key_sim = jax.random.PRNGKey(0)  # Fixed seed for reproducibility
        noise = jax.random.normal(key_sim, shape=(n_schools,)) * standard_errors
        
        return params.theta + noise
    
    # Create model builder
    model_builder = EnhancedModelBuilder[HierarchicalParams]()
    
    # Add observed data
    model_builder.with_observed_data(observed_data_fn)
    
    # Create parametric density function
    def parametric_density_fn(params: HierarchicalParams):
        def log_prob(data: Array) -> Array:
            # Priors
            prior_mu = normal_distribution.log_prob(
                LocationScaleParams(loc=0., scale=5.)
            )(params.mu)
            
            prior_tau = normal_distribution.log_prob(
                LocationScaleParams(loc=0., scale=5.)
            )(params.log_tau)
            
            prior_theta = normal_distribution.log_prob(
                LocationScaleParams(loc=params.mu, scale=params.tau)
            )(params.theta)
            
            # Likelihood
            likelihood = normal_distribution.log_prob(
                LocationScaleParams(loc=params.theta, scale=standard_errors)
            )(data)
            
            return prior_mu + prior_tau + prior_theta + likelihood
        
        return log_prob
    
    # Add parametric density function
    model_builder.with_parametric_density_fn(parametric_density_fn)
    
    # Build the model
    model = model_builder.build()
    
    # Create the workflow manager
    workflow = BayesianWorkflow(model, rng_key=key_workflow)
    
    # Create initial parameters
    initial_params = HierarchicalParams(
        mu=jnp.array(0.0),
        log_tau=jnp.array(0.0),
        theta=jnp.zeros(n_schools),
    )
    
    # Define parameter flattening and unflattening functions
    def flatten_param_fn(params: HierarchicalParams) -> Array:
        return jnp.concatenate([
            jnp.array([params.mu]),
            jnp.array([params.log_tau]),
            params.theta,
        ])
    
    def unflatten_param_fn(flat_params: Array) -> HierarchicalParams:
        return HierarchicalParams(
            mu=flat_params[0],
            log_tau=flat_params[1],
            theta=flat_params[2:],
        )
    
    # Run the full workflow
    results = workflow.run_workflow(
        initial_params=initial_params,
        observed_data=treatment_effects,
        num_samples=2000,
        num_predictive_samples=100,
        flatten_fn=flatten_param_fn,
        unflatten_fn=unflatten_param_fn,
    )
    
    # Extract posterior samples
    posterior_samples = results.posterior_samples
    
    # Convert to parameter objects
    posterior_params = [unflatten_param_fn(sample) for sample in posterior_samples]
    
    # Extract parameters
    mu_samples = jnp.array([p.mu for p in posterior_params])
    tau_samples = jnp.array([p.tau for p in posterior_params])
    theta_samples = jnp.stack([p.theta for p in posterior_params])
    
    # Print results
    print("\nPopulation parameters:")
    print(f"mu: {jnp.mean(mu_samples):.1f} ± {jnp.std(mu_samples):.1f}")
    print(f"tau: {jnp.mean(tau_samples):.1f} ± {jnp.std(tau_samples):.1f}")
    
    print("\nSchool-specific effects:")
    for i in range(n_schools):
        school_samples = theta_samples[:, i]
        print(f"School {i+1}: {jnp.mean(school_samples):.1f} ± {jnp.std(school_samples):.1f}")
    
    # Plot the results
    plt.figure(figsize=(12, 6))
    
    # Create boxplots for each school
    plt.boxplot(
        theta_samples,
        label="School Effects",
        patch_artist=True,
    )
    
    # Add the observed data
    plt.scatter(
        range(1, n_schools + 1),
        treatment_effects,
        color='red',
        marker='o',
        label='Observed Effects',
    )
    
    # Add error bars for standard errors
    plt.errorbar(
        range(1, n_schools + 1),
        treatment_effects,
        yerr=standard_errors,
        fmt='none',
        color='red',
        capsize=5,
    )
    
    # Add the global mean
    plt.axhline(
        float(jnp.mean(mu_samples)),
        color='blue',
        linestyle='--',
        label=f'Population Mean: {float(jnp.mean(mu_samples)):.1f}',
    )
    
    plt.xlabel('School')
    plt.ylabel('Treatment Effect')
    plt.title('Eight Schools: Posterior Distributions of Treatment Effects')
    plt.legend()
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.show()
    
    # Plot the shrinkage effect
    plt.figure(figsize=(10, 6))
    
    # Add school means
    observed_means = treatment_effects
    posterior_means = jnp.mean(theta_samples, axis=0)
    
    # Plot the shrinkage
    for i in range(n_schools):
        plt.plot(
            [0, 1],
            [observed_means[i], posterior_means[i]],
            'k-',
            alpha=0.5,
        )
        
        plt.scatter(0, observed_means[i], color='red', s=50)
        plt.scatter(1, posterior_means[i], color='blue', s=50)
    
    # Add labels
    plt.xticks(
        [0, 1],
        ['Observed Effects', 'Posterior Means'],
    )
    
    plt.title('Eight Schools: Shrinkage of Treatment Effects')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.show()


if __name__ == "__main__":
    main()