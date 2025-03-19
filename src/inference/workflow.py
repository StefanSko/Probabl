"""Unified Bayesian workflow manager."""
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Generic, List, Optional, Tuple, TypeVar, Union

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
from jaxtyping import Array

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from distributions import (
        BaseParams,
        Distribution, 
        EnhancedModelBuilder,
        EnhancedProbabilisticModel,
    )
    from distributions.distribution import data_from_distribution

# Define P TypeVar here to avoid circular imports
from distributions.continous import BaseParams
from inference.checking import (
    SummaryDict,
    check_posterior_predictive,
    check_prior_predictive,
    plot_predictive_comparison,
    summarize_posterior_predictive,
    summarize_prior_predictive,
)
from inference.context import DataContext, context
from inference.samplers import nuts_with_warmup


P = TypeVar('P', bound=BaseParams)


@dataclass
class WorkflowResults(Generic[P]):
    """Results from a Bayesian workflow run."""
    
    # Prior predictive results
    prior_samples: Optional[Array] = None
    prior_summary: Optional[SummaryDict] = None
    
    # Inference results
    posterior_samples: Optional[Array] = None
    posterior_params: Optional[List[P]] = None
    
    # Posterior predictive results
    posterior_predictive_samples: Optional[Array] = None
    posterior_predictive_summary: Optional[SummaryDict] = None
    
    # Model comparison metrics
    model_metrics: Dict[str, Any] = field(default_factory=dict)


class BayesianWorkflow(Generic[P]):
    """Manager for the full Bayesian workflow.
    
    This class orchestrates the entire Bayesian workflow, including:
    - Prior predictive checks
    - MCMC inference
    - Posterior predictive checks
    
    It manages context transitions and maintains results from each phase.
    """
    
    def __init__(
        self,
        model: Any,  # EnhancedProbabilisticModel[P],
        rng_key: Optional[jax.Array] = None,
    ):
        """Initialize the workflow manager.
        
        Args:
            model: The enhanced probabilistic model
            rng_key: JAX random key for reproducibility
        """
        self.model = model
        # Fix JAX PRNGKey handling
        if rng_key is None:
            self.rng_key = jax.random.PRNGKey(0)
        else:
            self.rng_key = rng_key
        self.results: WorkflowResults[P] = WorkflowResults()
    
    def prior_check(
        self,
        n_samples: int = 100,
        plot: bool = True,
        observed_data: Optional[Array] = None,
    ) -> SummaryDict:
        """Run prior predictive checks.
        
        Args:
            n_samples: Number of prior predictive samples to generate
            plot: Whether to create plots
            observed_data: Optional observed data to compare with
            
        Returns:
            Dictionary with summary statistics
        """
        # Split the random key
        self.rng_key, subkey = jax.random.split(self.rng_key)
        
        # Generate prior predictive samples
        prior_samples = check_prior_predictive(
            self.model,
            n_samples=n_samples,
            rng_key=subkey,
        )
        
        # Store the samples
        self.results.prior_samples = prior_samples
        
        # If we have observed data or plot is requested, summarize the results
        if plot or observed_data is not None:
            # Summarize the prior predictive checks
            summary = summarize_prior_predictive(
                prior_samples,
                observed_data=observed_data,
            )
            
            # Store the summary
            self.results.prior_summary = summary
            
            return summary
        
        return {"samples": prior_samples}
    
    def run_inference(
        self,
        initial_params: P,
        num_samples: int = 1000,
        flatten_fn: Optional[Callable[[P], Array]] = None,
        unflatten_fn: Optional[Callable[[Array], P]] = None,
    ) -> Array:
        """Run MCMC inference.
        
        Args:
            initial_params: Initial parameter values
            num_samples: Number of posterior samples to generate
            flatten_fn: Function to flatten structured parameters (if needed)
            unflatten_fn: Function to unflatten parameters (if needed)
            
        Returns:
            Array of posterior samples
        """
        # Store the unflatten function for later use
        if unflatten_fn is not None:
            self._unflatten_fn = unflatten_fn
            
        # Split the random key
        self.rng_key, subkey = jax.random.split(self.rng_key)
        
        # Create a log probability function
        def log_prob_fn(params_flat: Array) -> Array:
            # Unflatten parameters if needed
            if unflatten_fn is not None:
                params = unflatten_fn(params_flat)
            else:
                # Different parameter classes may have different constructors
                # We need to handle this safely
                try:
                    # Try to construct the parameter object directly
                    # This might work for simple parameter classes
                    # Disable mypy for this line because parameter classes can have different constructors
                    params = initial_params.__class__(params_flat)  # type: ignore
                except TypeError:
                    # If direct construction fails, use a type assertion to tell mypy 
                    # that we know what we're doing
                    params_val: P = params_flat  # type: ignore
                    params = params_val
            
            # Switch to inference context
            with context(DataContext.INFERENCE):
                # Compute log probability
                return self.model.forward(params)
        
        # Flatten initial parameters if needed
        if flatten_fn is not None:
            initial_position = flatten_fn(initial_params)
        else:
            # Handle case when flatting function is not provided
            # Assume we're using BaseParams and access its raw parameters
            # This relies on BaseParams having a method or property to access raw parameters
            if hasattr(initial_params, 'params'):
                initial_position = initial_params.params
            else:
                # Fall back to treating the whole object as the parameters
                initial_position = jnp.array(initial_params)
        
        # Run inference
        samples = nuts_with_warmup(
            log_prob_fn,
            initial_position,
            subkey,
            num_samples=num_samples,
        )
        
        # Store the samples
        self.results.posterior_samples = samples
        
        # Unflatten samples if needed
        if unflatten_fn is not None:
            posterior_params = [unflatten_fn(s) for s in samples]
            self.results.posterior_params = posterior_params
        
        return samples
    
    def posterior_check(
        self,
        n_samples: int = 100,
        observed_data: Optional[Array] = None,
        plot: bool = True,
    ) -> SummaryDict:
        """Run posterior predictive checks.
        
        Args:
            n_samples: Number of posterior predictive samples to generate
            observed_data: Observed data to compare with
            plot: Whether to create plots
            
        Returns:
            Dictionary with summary statistics
        """
        if self.results.posterior_samples is None:
            raise ValueError("Must run inference before posterior predictive checks")
        
        # Split the random key
        self.rng_key, subkey = jax.random.split(self.rng_key)
        
        # Get a representative posterior sample to use for demonstration
        # In a real implementation, you would use multiple samples
        if not hasattr(self.model, "_posterior_data_generator"):
            # If model doesn't have a custom posterior data generator, use the standard method
            posterior_samples = check_posterior_predictive(
                self.model,
                self.results.posterior_samples,
                n_samples=n_samples,
                rng_key=subkey,
            )
        else:
            # Use the custom posterior data generator
            # We'll use up to 100 posterior samples for demonstration
            num_posterior_samples = min(100, len(self.results.posterior_samples))
            sample_indices = jnp.arange(num_posterior_samples)
            
            # Create a collection of posterior predictive samples
            samples = []
            if self.results.posterior_params is not None:
                # Use the posterior params if available
                posterior_params = self.results.posterior_params[:num_posterior_samples]
                for param in posterior_params:
                    # Generate data using the custom generator
                    data_fn = self.model._posterior_data_generator(param)
                    samples.append(data_fn())
            else:
                # Fall back to using raw posterior samples
                # We need unflatten_fn to create proper parameter objects from raw samples
                if not hasattr(self, '_unflatten_fn') or self._unflatten_fn is None:
                    raise ValueError("Cannot generate posterior predictive samples without unflatten_fn")
                
                for i in range(num_posterior_samples):
                    # Create params from the raw samples using the unflatten function
                    raw_sample = self.results.posterior_samples[i]
                    params = self._unflatten_fn(raw_sample)
                    
                    # Generate data using the custom generator
                    data_fn = self.model._posterior_data_generator(params)
                    samples.append(data_fn())
                    
            # Stack samples into an array
            posterior_samples = jnp.stack(samples)
        
        # Store the samples
        self.results.posterior_predictive_samples = posterior_samples
        
        # If we have observed data or plot is requested, summarize the results
        if observed_data is not None and plot:
            # Summarize the posterior predictive checks
            summary = summarize_posterior_predictive(
                posterior_samples,
                observed_data,
            )
            
            # Store the summary
            self.results.posterior_predictive_summary = summary
            
            # If we also have prior samples, plot the comparison
            if self.results.prior_samples is not None:
                plot_predictive_comparison(
                    self.results.prior_samples,
                    posterior_samples,
                    observed_data,
                )
            
            return summary
        
        return {"samples": posterior_samples}
    
    def run_workflow(
        self,
        initial_params: P,
        observed_data: Array,
        num_samples: int = 1000,
        num_predictive_samples: int = 100,
        flatten_fn: Optional[Callable[[P], Array]] = None,
        unflatten_fn: Optional[Callable[[Array], P]] = None,
    ) -> WorkflowResults:
        """Run the full Bayesian workflow.

        Args:
            initial_params: Initial parameter values
            observed_data: Observed data
            num_samples: Number of posterior samples to generate
            num_predictive_samples: Number of predictive samples to generate
            flatten_fn: Function to flatten structured parameters (if needed)
            unflatten_fn: Function to unflatten parameters (if needed)

        Returns:
            WorkflowResults containing all results from the workflow
        """
        # Run prior predictive checks
        self.prior_check(
            n_samples=num_predictive_samples,
            observed_data=observed_data,
        )

        # Run inference
        self.run_inference(
            initial_params,
            num_samples=num_samples,
            flatten_fn=flatten_fn,
            unflatten_fn=unflatten_fn,
        )

        # Run posterior predictive checks
        self.posterior_check(
            n_samples=num_predictive_samples,
            observed_data=observed_data,
        )

        return self.results
