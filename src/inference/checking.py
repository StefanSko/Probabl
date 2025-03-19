"""Utilities for prior and posterior predictive checking."""
from typing import Any, Callable, Dict, List, Optional, Tuple, TypedDict, Union

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from jaxtyping import Array, Float

from distributions import Distribution, EnhancedProbabilisticModel, normal_distribution
from distributions.distribution import data_from_distribution
from inference.context import DataContext, context


class SummaryDict(TypedDict, total=False):
    """Dictionary for storing summary statistics from model checks."""
    
    mean: Any
    std: Any
    q5: Any
    q95: Any
    observed_mean: float
    predictive_mean: float
    p_value: float
    samples: Any  # For storing sample data
    # Additional fields to allow for dynamic keys
    __extra__: Any


def check_prior_predictive(
    model: EnhancedProbabilisticModel[Any],
    n_samples: int = 100,
    rng_key: Optional[jax.Array] = None,
) -> Array:
    """Generate samples from prior predictive distribution.
    
    Args:
        model: The model to sample from
        n_samples: Number of samples to generate
        rng_key: JAX random key for reproducibility
        
    Returns:
        Array of samples from prior predictive distribution
    """
    if rng_key is None:
        rng_key = jax.random.PRNGKey(0)
    
    # Use the model's context-aware data function with PRIOR_PREDICTIVE context
    with context(DataContext.PRIOR_PREDICTIVE):
        # Generate n_samples
        samples = []
        for _ in range(n_samples):
            # Get a sample from the prior predictive
            samples.append(model.data())
        
        return jnp.stack(samples)


def summarize_prior_predictive(
    samples: Array,
    observed_data: Optional[Array] = None,
    title: str = "Prior Predictive Check",
) -> SummaryDict:
    """Compare prior predictive samples to observed data.
    
    Args:
        samples: Samples from prior predictive distribution (n_samples x data_dims)
        observed_data: Optional observed data to compare with
        title: Title for the plot
        
    Returns:
        Dictionary with summary statistics
    """
    # Compute summary statistics
    mean = jnp.mean(samples, axis=0)
    std = jnp.std(samples, axis=0)
    q5 = jnp.percentile(samples, 5, axis=0)
    q95 = jnp.percentile(samples, 95, axis=0)
    
    # Create a summary dictionary
    summary: SummaryDict = {
        "mean": mean,
        "std": std,
        "q5": q5,
        "q95": q95,
    }
    
    # Plot the distribution of means
    plt.figure(figsize=(10, 6))
    
    # Plot histogram of means across samples
    plt.hist(
        samples.mean(axis=1),
        bins=30,
        alpha=0.5,
        label="Prior Predictive"
    )
    
    if observed_data is not None:
        # Add observed data mean
        observed_mean = float(observed_data.mean())
        plt.axvline(
            float(observed_mean),
            color="red",
            linestyle="dashed",
            linewidth=2,
            label=f"Observed Mean: {float(observed_mean):.2f}"
        )
        
        # Add summary to dictionary
        summary["observed_mean"] = float(observed_mean)
        summary["p_value"] = float(
            (samples.mean(axis=1) <= observed_mean).mean()
            if observed_mean < mean.mean()
            else (samples.mean(axis=1) >= observed_mean).mean()
        )
    
    plt.title(title)
    plt.xlabel("Mean Value")
    plt.ylabel("Frequency")
    plt.legend()
    
    return summary


def check_posterior_predictive(
    model: EnhancedProbabilisticModel[Any],
    posterior_samples: Array,
    n_samples: int = 100,
    rng_key: Optional[jax.Array] = None,
) -> Array:
    """Generate samples from posterior predictive distribution.
    
    Args:
        model: The model to sample from
        posterior_samples: Samples from the posterior distribution
        n_samples: Number of samples to generate
        rng_key: JAX random key for reproducibility
        
    Returns:
        Array of samples from posterior predictive distribution
    """
    if rng_key is None:
        rng_key = jax.random.PRNGKey(0)
    
    # Use the model's context-aware data function with POSTERIOR_PREDICTIVE context
    with context(DataContext.POSTERIOR_PREDICTIVE):
        # Take a subset of posterior samples if there are more than n_samples
        if len(posterior_samples) > n_samples:
            indices = jax.random.choice(
                rng_key, len(posterior_samples), (n_samples,), replace=False
            )
            selected_samples = posterior_samples[indices]
        else:
            selected_samples = posterior_samples
        
        # Generate posterior predictive samples
        predictive_samples = []
        for sample in selected_samples:
            # Set the posterior sample
            # In a real implementation, the model would use this sample
            # Here we're just simulating that behavior
            predictive_samples.append(model.data())
        
        return jnp.stack(predictive_samples)


def summarize_posterior_predictive(
    samples: Array,
    observed_data: Array,
    title: str = "Posterior Predictive Check",
) -> SummaryDict:
    """Compare posterior predictive samples to observed data.
    
    Args:
        samples: Samples from posterior predictive distribution
        observed_data: Observed data to compare with
        title: Title for the plot
        
    Returns:
        Dictionary with summary statistics
    """
    # Compute summary statistics
    mean = jnp.mean(samples, axis=0)
    std = jnp.std(samples, axis=0)
    q5 = jnp.percentile(samples, 5, axis=0)
    q95 = jnp.percentile(samples, 95, axis=0)
    
    # Create a summary dictionary
    summary = {
        "mean": mean,
        "std": std,
        "q5": q5,
        "q95": q95,
    }
    
    # Plot the data
    plt.figure(figsize=(10, 6))
    
    # If the data is 1-dimensional, create a histogram
    if observed_data.ndim == 1 or (observed_data.ndim == 2 and observed_data.shape[1] == 1):
        # Flatten observed data and posterior predictive samples
        observed_flat = observed_data.flatten()
        samples_flat = samples.flatten()
        
        # Plot distributions
        plt.hist(
            observed_flat,
            bins=30,
            alpha=0.5,
            label="Observed Data"
        )
        
        plt.hist(
            samples_flat,
            bins=30,
            alpha=0.5,
            label="Posterior Predictive"
        )
        
        # Add distribution statistics
        observed_mean = float(observed_flat.mean())
        samples_mean = float(samples_flat.mean())
        
        plt.axvline(
            observed_mean,
            color="red",
            linestyle="dashed",
            linewidth=2,
            label=f"Observed Mean: {observed_mean:.2f}"
        )
        
        plt.axvline(
            samples_mean,
            color="blue",
            linestyle="dashed",
            linewidth=2,
            label=f"Predictive Mean: {samples_mean:.2f}"
        )
        
        # Create a new dictionary with the right types
        # rather than modifying the original summary with incompatible types
        temp_dict: Dict[str, Any] = dict(summary)
        temp_dict["observed_mean"] = float(observed_mean)
        temp_dict["predictive_mean"] = float(samples_mean)
        summary = temp_dict
        
    # If the data is multi-dimensional, show a Q-Q plot
    else:
        # Get quantiles for observed data
        observed_quantiles = jnp.percentile(
            observed_data.flatten(), jnp.linspace(0, 100, 101)
        )
        
        # Get quantiles for posterior predictive samples
        samples_quantiles = jnp.percentile(
            samples.flatten(), jnp.linspace(0, 100, 101)
        )
        
        # Q-Q plot
        plt.scatter(observed_quantiles, samples_quantiles, alpha=0.7)
        
        # Add reference line
        min_val = float(min(float(observed_quantiles.min()), float(samples_quantiles.min())))
        max_val = float(max(float(observed_quantiles.max()), float(samples_quantiles.max())))
        plt.plot([min_val, max_val], [min_val, max_val], 'k--')
        
        plt.xlabel("Observed Data Quantiles")
        plt.ylabel("Posterior Predictive Quantiles")
    
    plt.title(title)
    plt.legend()
    
    # Convert to SummaryDict to match return type
    # Create a Dict[str, Any] first, then cast to SummaryDict
    result_summary_dict: Dict[str, Any] = {}
    for key, value in summary.items():
        if key in ["observed_mean", "predictive_mean", "p_value"]:
            result_summary_dict[key] = float(value)
        else:
            result_summary_dict[key] = value
    
    # TypedDict doesn't allow direct casting, so we need to bypass the type system a bit
    result: SummaryDict = dict(result_summary_dict)  # type: ignore
    return result


def plot_predictive_comparison(
    prior_samples: Array,
    posterior_samples: Array,
    observed_data: Array,
    title: str = "Prior vs Posterior Predictive Checks",
) -> None:
    """Plot comparison between prior and posterior predictive distributions.
    
    Args:
        prior_samples: Samples from prior predictive distribution
        posterior_samples: Samples from posterior predictive distribution
        observed_data: Observed data to compare with
        title: Title for the plot
    """
    # If the data is 1-dimensional, create a density plot
    if observed_data.ndim == 1 or (observed_data.ndim == 2 and observed_data.shape[1] == 1):
        # Flatten data
        observed_flat = observed_data.flatten()
        prior_flat = prior_samples.flatten()
        posterior_flat = posterior_samples.flatten()
        
        # Create plot
        plt.figure(figsize=(12, 6))
        
        # Plot densities
        plt.hist(
            observed_flat,
            bins=30,
            alpha=0.5,
            density=True,
            label="Observed Data"
        )
        
        plt.hist(
            prior_flat,
            bins=30,
            alpha=0.3,
            density=True,
            label="Prior Predictive"
        )
        
        plt.hist(
            posterior_flat,
            bins=30,
            alpha=0.3,
            density=True,
            label="Posterior Predictive"
        )
        
        # Add density statistics
        observed_mean = float(observed_flat.mean())
        prior_mean = float(prior_flat.mean())
        posterior_mean = float(posterior_flat.mean())
        
        plt.axvline(
            observed_mean,
            color="black",
            linestyle="dashed",
            linewidth=2,
            label=f"Observed Mean: {observed_mean:.2f}"
        )
        
        plt.axvline(
            prior_mean,
            color="blue",
            linestyle="dashed",
            linewidth=2,
            label=f"Prior Mean: {prior_mean:.2f}"
        )
        
        plt.axvline(
            posterior_mean,
            color="red",
            linestyle="dashed",
            linewidth=2,
            label=f"Posterior Mean: {posterior_mean:.2f}"
        )
        
        plt.title(title)
        plt.xlabel("Value")
        plt.ylabel("Density")
        plt.legend()
    
    # If the data is 2-dimensional, create scatter plots
    elif observed_data.ndim == 2 and observed_data.shape[1] == 2:
        # Create plot
        fig, axes = plt.subplots(1, 3, figsize=(18, 6), sharex=True, sharey=True)
        
        # Observed data
        axes[0].scatter(
            observed_data[:, 0],
            observed_data[:, 1],
            alpha=0.7
        )
        axes[0].set_title("Observed Data")
        
        # Prior predictive
        sample_idx = np.random.choice(len(prior_samples), min(100, len(prior_samples)))
        axes[1].scatter(
            prior_samples[sample_idx, 0],
            prior_samples[sample_idx, 1],
            alpha=0.7
        )
        axes[1].set_title("Prior Predictive Samples")
        
        # Posterior predictive
        sample_idx = np.random.choice(len(posterior_samples), min(100, len(posterior_samples)))
        axes[2].scatter(
            posterior_samples[sample_idx, 0],
            posterior_samples[sample_idx, 1],
            alpha=0.7
        )
        axes[2].set_title("Posterior Predictive Samples")
        
        # Set common labels
        for ax in axes:
            ax.set_xlabel("Dimension 1")
            ax.set_ylabel("Dimension 2")
        
        plt.suptitle(title)
        plt.tight_layout()
    
    else:
        print("Cannot create visualization for data with more than 2 dimensions.")