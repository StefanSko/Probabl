"""Hierarchical model example with the context-aware framework."""
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple
import sys
import os

# Add the parent directory to path so we can import our modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
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
    """Parameters for multi-level hierarchical model."""
    
    # Hyperpriors (global parameters)
    mu_alpha: Float[Array, " "]  # Global intercept mean
    mu_beta: Float[Array, " "]   # Global slope mean
    log_tau_alpha: Float[Array, " "]  # Log of global intercept std
    log_tau_beta: Float[Array, " "]   # Log of global slope std
    log_sigma: Float[Array, " "]  # Log of observation noise
    
    # Group-level parameters
    alpha: Float[Array, "n_groups"]  # Group-specific intercepts
    beta: Float[Array, "n_groups"]   # Group-specific slopes
    
    @property
    def tau_alpha(self) -> Float[Array, " "]:
        """Get the tau_alpha parameter (global intercept std)."""
        return jnp.exp(self.log_tau_alpha)
    
    @property
    def tau_beta(self) -> Float[Array, " "]:
        """Get the tau_beta parameter (global slope std)."""
        return jnp.exp(self.log_tau_beta)
    
    @property
    def sigma(self) -> Float[Array, " "]:
        """Get the sigma parameter (observation noise)."""
        return jnp.exp(self.log_sigma)


def generate_hierarchical_data(
    n_groups: int,
    n_per_group: int,
    mu_alpha: float,
    mu_beta: float,
    tau_alpha: float,
    tau_beta: float,
    sigma: float,
    x_min: float = -3.0,
    x_max: float = 3.0,
    rng_key: jax.Array = None,
) -> Tuple[Dict[str, Any], Dict[str, Array]]:
    """Generate synthetic data for hierarchical linear regression.
    
    Args:
        n_groups: Number of groups
        n_per_group: Number of observations per group
        mu_alpha: Global intercept mean
        mu_beta: Global slope mean
        tau_alpha: Global intercept standard deviation
        tau_beta: Global slope standard deviation
        sigma: Observation noise
        x_min: Minimum x value
        x_max: Maximum x value
        rng_key: JAX random key for reproducibility
        
    Returns:
        Tuple of (parameters, data)
            parameters: Dictionary with true parameter values
            data: Dictionary with observed data (x, y, group)
    """
    if rng_key is None:
        rng_key = jax.random.PRNGKey(0)
    
    # Split the random key
    key_alpha, key_beta, key_x, key_noise = jax.random.split(rng_key, 4)
    
    # Generate group-level parameters
    true_alpha = mu_alpha + jax.random.normal(key_alpha, (n_groups,)) * tau_alpha
    true_beta = mu_beta + jax.random.normal(key_beta, (n_groups,)) * tau_beta
    
    # Generate X values for each group
    x_values = []
    y_values = []
    group_indices = []
    
    for i in range(n_groups):
        # Generate group-specific random X values
        group_key = jax.random.fold_in(key_x, i)
        x_group = jax.random.uniform(
            group_key, (n_per_group,), minval=x_min, maxval=x_max
        )
        
        # Generate noise
        noise_key = jax.random.fold_in(key_noise, i)
        noise = jax.random.normal(noise_key, (n_per_group,)) * sigma
        
        # Generate Y values
        y_group = true_alpha[i] + true_beta[i] * x_group + noise
        
        # Store values
        x_values.append(x_group)
        y_values.append(y_group)
        group_indices.extend([i] * n_per_group)
    
    # Combine data
    x_all = jnp.concatenate(x_values)
    y_all = jnp.concatenate(y_values)
    group_all = jnp.array(group_indices)
    
    # Create dictionaries
    parameters = {
        "mu_alpha": mu_alpha,
        "mu_beta": mu_beta,
        "tau_alpha": tau_alpha,
        "tau_beta": tau_beta,
        "sigma": sigma,
        "alpha": true_alpha,
        "beta": true_beta,
    }
    
    data = {
        "x": x_all,
        "y": y_all,
        "group": group_all,
    }
    
    return parameters, data


def main() -> None:
    """Run a hierarchical model example with the Bayesian workflow."""
    # Set up random keys
    key = jax.random.PRNGKey(0)
    key_data, key_workflow = jax.random.split(key)
    
    # Generate synthetic data
    n_groups = 8
    n_per_group = 20
    
    true_params, data = generate_hierarchical_data(
        n_groups=n_groups,
        n_per_group=n_per_group,
        mu_alpha=1.5,
        mu_beta=0.8,
        tau_alpha=0.5,
        tau_beta=0.3,
        sigma=0.7,
        rng_key=key_data,
    )
    
    # Extract data
    x_data = data["x"]
    y_data = data["y"]
    group_data = data["group"]
    
    # Plot the data by group
    plt.figure(figsize=(12, 8))
    
    for i in range(n_groups):
        # Get group-specific data
        group_mask = group_data == i
        x_group = x_data[group_mask]
        y_group = y_data[group_mask]
        
        # Plot data points
        plt.scatter(
            x_group, y_group, alpha=0.7, label=f"Group {i+1}"
        )
        
        # Plot true regression line
        x_line = jnp.linspace(x_data.min(), x_data.max(), 100)
        y_line = true_params["alpha"][i] + true_params["beta"][i] * x_line
        plt.plot(x_line, y_line, '-')
    
    # Plot global trend
    x_global = jnp.linspace(x_data.min(), x_data.max(), 100)
    y_global = true_params["mu_alpha"] + true_params["mu_beta"] * x_global
    plt.plot(
        x_global, y_global, 'k--', linewidth=2,
        label=f"Global: y = {true_params['mu_alpha']:.1f} + {true_params['mu_beta']:.1f}x"
    )
    
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
    plt.title("Hierarchical Model Data by Group")
    plt.tight_layout()
    plt.show()
    
    # Create observed data function
    def observed_data_fn() -> Array:
        return y_data
    
    # Function to generate simulated data from parameters
    def simulate_data(params: HierarchicalParams) -> Array:
        # Initialize simulated data
        simulated_data = jnp.zeros_like(y_data)
        
        # For each group, generate data
        for i in range(n_groups):
            # Get group-specific data
            group_mask = group_data == i
            x_group = x_data[group_mask]
            
            # Compute mean prediction
            mean_pred = params.alpha[i] + params.beta[i] * x_group
            
            # Generate random noise
            key_sim = jax.random.PRNGKey(i)  # Fixed seed per group for reproducibility
            noise = jax.random.normal(key_sim, shape=mean_pred.shape) * params.sigma
            
            # Set simulated data for this group
            simulated_data = simulated_data.at[group_mask].set(mean_pred + noise)
        
        return simulated_data
    
    # Create model builder
    model_builder = EnhancedModelBuilder[HierarchicalParams]()
    
    # Add observed data
    model_builder.with_observed_data(observed_data_fn)
    
    # Define parametric density function
    def parametric_density_fn(params: HierarchicalParams):
        def log_prob(data: Array) -> Array:
            # Hyperpriors
            prior_mu_alpha = normal_distribution.log_prob(
                LocationScaleParams(loc=0.0, scale=5.0)
            )(params.mu_alpha)
            
            prior_mu_beta = normal_distribution.log_prob(
                LocationScaleParams(loc=0.0, scale=5.0)
            )(params.mu_beta)
            
            prior_tau_alpha = normal_distribution.log_prob(
                LocationScaleParams(loc=0.0, scale=1.0)
            )(params.log_tau_alpha)
            
            prior_tau_beta = normal_distribution.log_prob(
                LocationScaleParams(loc=0.0, scale=1.0)
            )(params.log_tau_beta)
            
            prior_sigma = normal_distribution.log_prob(
                LocationScaleParams(loc=0.0, scale=1.0)
            )(params.log_sigma)
            
            # Group-level priors
            prior_alpha = normal_distribution.log_prob(
                LocationScaleParams(loc=params.mu_alpha, scale=params.tau_alpha)
            )(params.alpha)
            
            prior_beta = normal_distribution.log_prob(
                LocationScaleParams(loc=params.mu_beta, scale=params.tau_beta)
            )(params.beta)
            
            # Initialize log likelihood
            log_likelihood = 0.0
            
            # Compute likelihood for each group
            for i in range(n_groups):
                # Get group-specific data
                group_mask = group_data == i
                x_group = x_data[group_mask]
                y_group = data[group_mask]
                
                # Compute mean prediction
                mean_pred = params.alpha[i] + params.beta[i] * x_group
                
                # Compute log likelihood for this group
                group_likelihood = normal_distribution.log_prob(
                    LocationScaleParams(loc=mean_pred, scale=params.sigma)
                )(y_group)
                
                # Add to total log likelihood
                # Cast to the same type to avoid type error
                log_likelihood = float(log_likelihood) + float(group_likelihood)
            
            # Combine all log probabilities
            return (
                prior_mu_alpha + prior_mu_beta +
                prior_tau_alpha + prior_tau_beta + prior_sigma +
                prior_alpha + prior_beta +
                log_likelihood
            )
        
        return log_prob
    
    # Add parametric density function
    model_builder.with_parametric_density_fn(parametric_density_fn)
    
    # Build the model
    model = model_builder.build()
    
    # Create the workflow manager
    workflow = BayesianWorkflow(model, rng_key=key_workflow)
    
    # Create initial parameters
    initial_params = HierarchicalParams(
        mu_alpha=jnp.array(0.0),
        mu_beta=jnp.array(0.0),
        log_tau_alpha=jnp.array(0.0),
        log_tau_beta=jnp.array(0.0),
        log_sigma=jnp.array(0.0),
        alpha=jnp.zeros(n_groups),
        beta=jnp.zeros(n_groups),
    )
    
    # Define parameter flattening and unflattening functions
    def flatten_param_fn(params: HierarchicalParams) -> Array:
        return jnp.concatenate([
            jnp.array([params.mu_alpha]),
            jnp.array([params.mu_beta]),
            jnp.array([params.log_tau_alpha]),
            jnp.array([params.log_tau_beta]),
            jnp.array([params.log_sigma]),
            params.alpha,
            params.beta,
        ])
    
    def unflatten_param_fn(flat_params: Array) -> HierarchicalParams:
        return HierarchicalParams(
            mu_alpha=flat_params[0],
            mu_beta=flat_params[1],
            log_tau_alpha=flat_params[2],
            log_tau_beta=flat_params[3],
            log_sigma=flat_params[4],
            alpha=flat_params[5:5+n_groups],
            beta=flat_params[5+n_groups:5+2*n_groups],
        )
    
    # Run the full workflow
    results = workflow.run_workflow(
        initial_params=initial_params,
        observed_data=y_data,
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
    mu_alpha_samples = jnp.array([p.mu_alpha for p in posterior_params])
    mu_beta_samples = jnp.array([p.mu_beta for p in posterior_params])
    tau_alpha_samples = jnp.array([p.tau_alpha for p in posterior_params])
    tau_beta_samples = jnp.array([p.tau_beta for p in posterior_params])
    sigma_samples = jnp.array([p.sigma for p in posterior_params])
    alpha_samples = jnp.stack([p.alpha for p in posterior_params])
    beta_samples = jnp.stack([p.beta for p in posterior_params])
    
    # Print results
    print("\nGlobal Parameters:")
    print(f"mu_alpha: {jnp.mean(mu_alpha_samples):.2f} ± {jnp.std(mu_alpha_samples):.2f}")
    print(f"mu_beta: {jnp.mean(mu_beta_samples):.2f} ± {jnp.std(mu_beta_samples):.2f}")
    print(f"tau_alpha: {jnp.mean(tau_alpha_samples):.2f} ± {jnp.std(tau_alpha_samples):.2f}")
    print(f"tau_beta: {jnp.mean(tau_beta_samples):.2f} ± {jnp.std(tau_beta_samples):.2f}")
    print(f"sigma: {jnp.mean(sigma_samples):.2f} ± {jnp.std(sigma_samples):.2f}")
    
    print("\nGroup-Level Parameters:")
    for i in range(n_groups):
        print(f"Group {i+1}:")
        print(f"  alpha: {jnp.mean(alpha_samples[:, i]):.2f} ± {jnp.std(alpha_samples[:, i]):.2f}")
        print(f"  beta: {jnp.mean(beta_samples[:, i]):.2f} ± {jnp.std(beta_samples[:, i]):.2f}")
    
    # Plot posterior regression lines by group
    plt.figure(figsize=(12, 8))
    
    for i in range(n_groups):
        # Get group-specific data
        group_mask = group_data == i
        x_group = x_data[group_mask]
        y_group = y_data[group_mask]
        
        # Plot data points
        plt.scatter(
            x_group, y_group, alpha=0.4, s=30,
            label=f"Group {i+1}" if i == 0 else ""
        )
        
        # Plot true regression line
        x_line = jnp.linspace(x_data.min(), x_data.max(), 100)
        y_line = true_params["alpha"][i] + true_params["beta"][i] * x_line
        plt.plot(
            x_line, y_line, '-', color=f'C{i}',
            alpha=0.7, linewidth=1.5,
            label=f"True Model (Group {i+1})" if i == 0 else ""
        )
        
        # Plot posterior mean regression line
        alpha_mean = jnp.mean(alpha_samples[:, i])
        beta_mean = jnp.mean(beta_samples[:, i])
        y_posterior = alpha_mean + beta_mean * x_line
        plt.plot(
            x_line, y_posterior, '--', color=f'C{i}',
            alpha=0.7, linewidth=1.5,
            label=f"Posterior Mean (Group {i+1})" if i == 0 else ""
        )
    
    # Plot global trend
    x_global = jnp.linspace(x_data.min(), x_data.max(), 100)
    
    # True global trend
    y_global_true = true_params["mu_alpha"] + true_params["mu_beta"] * x_global
    plt.plot(
        x_global, y_global_true, 'k-', linewidth=2,
        label=f"True Global: y = {true_params['mu_alpha']:.1f} + {true_params['mu_beta']:.1f}x"
    )
    
    # Posterior global trend
    mu_alpha_mean = jnp.mean(mu_alpha_samples)
    mu_beta_mean = jnp.mean(mu_beta_samples)
    y_global_posterior = mu_alpha_mean + mu_beta_mean * x_global
    plt.plot(
        x_global, y_global_posterior, 'k--', linewidth=2,
        label=f"Posterior Global: y = {mu_alpha_mean:.1f} + {mu_beta_mean:.1f}x"
    )
    
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
    plt.title("Hierarchical Model: True vs Posterior")
    plt.tight_layout()
    plt.show()
    
    # Plot shrinkage effect for intercepts (alpha)
    plt.figure(figsize=(10, 6))
    
    # True vs raw vs posterior
    true_alpha = true_params["alpha"]
    
    # Compute raw alpha estimates (simple linear regression for each group)
    raw_alpha = np.zeros(n_groups)
    
    for i in range(n_groups):
        group_mask = group_data == i
        x_group = x_data[group_mask]
        y_group = y_data[group_mask]
        
        # Simple linear regression
        X = np.vstack([np.ones_like(x_group), x_group]).T
        coef = np.linalg.lstsq(X, y_group, rcond=None)[0]
        raw_alpha[i] = coef[0]
    
    # Posterior mean alpha
    posterior_alpha = jnp.mean(alpha_samples, axis=0)
    
    # Plot
    plt.scatter(true_alpha, raw_alpha, s=60, label="Raw Estimates", color='red')
    plt.scatter(true_alpha, posterior_alpha, s=60, label="Posterior Means", color='blue')
    
    # Add arrows to show shrinkage
    for i in range(n_groups):
        plt.arrow(
            float(true_alpha[i]), float(raw_alpha[i]),
            0, float(posterior_alpha[i] - raw_alpha[i]),
            color='gray', width=0.01, head_width=0.05, head_length=0.05,
            length_includes_head=True, alpha=0.5
        )
    
    # Add reference line
    min_val = min(np.min(true_alpha), np.min(raw_alpha), np.min(posterior_alpha))
    max_val = max(np.max(true_alpha), np.max(raw_alpha), np.max(posterior_alpha))
    plt.plot([min_val, max_val], [min_val, max_val], 'k--', alpha=0.5)
    
    # Add global mean
    plt.axhline(
        float(jnp.mean(mu_alpha_samples)),
        color='blue',
        linestyle=':',
        alpha=0.5,
        label=f"Global Mean: {float(jnp.mean(mu_alpha_samples)):.2f}"
    )
    
    plt.xlabel("True Values")
    plt.ylabel("Estimates")
    plt.title("Shrinkage Effect for Group Intercepts (Alpha)")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.show()
    
    # Plot shrinkage effect for slopes (beta)
    plt.figure(figsize=(10, 6))
    
    # True vs raw vs posterior
    true_beta = true_params["beta"]
    
    # Compute raw beta estimates (simple linear regression for each group)
    raw_beta = np.zeros(n_groups)
    
    for i in range(n_groups):
        group_mask = group_data == i
        x_group = x_data[group_mask]
        y_group = y_data[group_mask]
        
        # Simple linear regression
        X = np.vstack([np.ones_like(x_group), x_group]).T
        coef = np.linalg.lstsq(X, y_group, rcond=None)[0]
        raw_beta[i] = coef[1]
    
    # Posterior mean beta
    posterior_beta = jnp.mean(beta_samples, axis=0)
    
    # Plot
    plt.scatter(true_beta, raw_beta, s=60, label="Raw Estimates", color='red')
    plt.scatter(true_beta, posterior_beta, s=60, label="Posterior Means", color='blue')
    
    # Add arrows to show shrinkage
    for i in range(n_groups):
        plt.arrow(
            float(true_beta[i]), float(raw_beta[i]),
            0, float(posterior_beta[i] - raw_beta[i]),
            color='gray', width=0.01, head_width=0.05, head_length=0.05,
            length_includes_head=True, alpha=0.5
        )
    
    # Add reference line
    min_val = min(np.min(true_beta), np.min(raw_beta), np.min(posterior_beta))
    max_val = max(np.max(true_beta), np.max(raw_beta), np.max(posterior_beta))
    plt.plot([min_val, max_val], [min_val, max_val], 'k--', alpha=0.5)
    
    # Add global mean
    plt.axhline(
        float(jnp.mean(mu_beta_samples)),
        color='blue',
        linestyle=':',
        alpha=0.5,
        label=f"Global Mean: {float(jnp.mean(mu_beta_samples)):.2f}"
    )
    
    plt.xlabel("True Values")
    plt.ylabel("Estimates")
    plt.title("Shrinkage Effect for Group Slopes (Beta)")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.show()


if __name__ == "__main__":
    main()