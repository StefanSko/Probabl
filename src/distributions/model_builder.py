"""Enhanced probabilistic model builder with context-aware capabilities."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Generic, Optional, TypeVar, cast

import jax
import jax.numpy as jnp
from jaxtyping import Array

from distributions.continous import (
    BaseParams,
    DataFn,
    ParametricDensityFn,
)
from distributions.distribution import Distribution, data_from_distribution
from inference.simulation import ContextAwareDataFn, make_context_aware_data

P = TypeVar('P', bound=BaseParams)


@dataclass
class EnhancedProbabilisticModel(Generic[P]):
    """Enhanced probabilistic model with context-aware data functions."""

    data: ContextAwareDataFn
    parametric_density_fn: ParametricDensityFn[P]
    
    # Optional attribute for posterior data generation
    _posterior_data_generator: Optional[Callable[[P], Callable[[], Array]]] = None

    def forward(self, params: P) -> Array:
        """Compute the log probability of data given parameters."""
        return self.parametric_density_fn(params)(self.data())


class EnhancedModelBuilder(Generic[P]):
    """Builder for enhanced probabilistic models with context-aware data."""

    def __init__(self) -> None:
        """Initialize an empty model builder."""
        self.observed_data: Optional[DataFn] = None
        self.prior_simulator: Optional[DataFn] = None
        self.posterior_simulator: Optional[DataFn] = None
        self.parametric_density_fn: Optional[ParametricDensityFn[P]] = None
        self.rng_key: Optional[jax.Array] = None
        
        # Private attributes used by auto-simulation features
        self._posterior_data_generator: Optional[Callable[[P], Callable[[], Array]]] = None
        self._data_shape: Optional[tuple[int, ...]] = None
        self._x_data: Optional[Array] = None
        self._group_indices: Optional[list[Array]] = None

    def _copy(self) -> 'EnhancedModelBuilder[P]':
        """Create a copy of this builder to maintain immutability."""
        builder = EnhancedModelBuilder[P]()
        builder.observed_data = self.observed_data
        builder.prior_simulator = self.prior_simulator
        builder.posterior_simulator = self.posterior_simulator
        builder.parametric_density_fn = self.parametric_density_fn
        builder.rng_key = self.rng_key
        
        # Copy all custom attributes
        if hasattr(self, '_posterior_data_generator'):
            builder._posterior_data_generator = self._posterior_data_generator
        if hasattr(self, '_data_shape'):
            builder._data_shape = self._data_shape
        if hasattr(self, '_x_data'):
            builder._x_data = self._x_data
        if hasattr(self, '_group_indices'):
            builder._group_indices = self._group_indices
            
        return builder

    def with_observed_data(self, data_fn: DataFn) -> 'EnhancedModelBuilder[P]':
        """Set the observed data function for inference context."""
        builder = self._copy()
        builder.observed_data = data_fn
        return builder

    def with_rng_key(self, rng_key: jax.Array) -> 'EnhancedModelBuilder[P]':
        """Set the random number generator key for simulations."""
        builder = self._copy()
        builder.rng_key = rng_key
        return builder

    def with_prior_simulator(
        self,
        distribution: Distribution[P],
        params: P,
        rng_key: jax.Array,
        sample_shape: tuple[int, ...] = (1,),
    ) -> 'EnhancedModelBuilder[P]':
        """Set the prior simulator function for prior predictive context."""
        builder = self._copy()
        builder.prior_simulator = data_from_distribution(
            distribution, params, rng_key, sample_shape
        )
        return builder
    
    def with_custom_prior_simulator(
        self,
        simulator_fn: Callable[[], P]
    ) -> 'EnhancedModelBuilder[P]':
        """Set a custom prior simulator function."""
        builder = self._copy()
        # Type error: prior_simulator expects DataFn (Callable[[], Array])
        # but simulator_fn is Callable[[], P]
        # This is a known issue in the API design
        builder.prior_simulator = cast(DataFn, simulator_fn)
        return builder
        
    def with_prior_data_simulator(
        self,
        data_simulator_fn: Callable[[], Array]
    ) -> 'EnhancedModelBuilder[P]':
        """Set a custom prior data simulator function."""
        builder = self._copy()
        builder.prior_simulator = data_simulator_fn
        return builder

    def with_auto_prior_simulator(
        self, 
        default_params: P,
        data_shape: Optional[tuple[int, ...]] = None,
        n_groups: Optional[int] = None,
        x_data: Optional[Array] = None,
        group_data: Optional[Array] = None,
        group_indices: Optional[list[Array]] = None,
    ) -> 'EnhancedModelBuilder[P]':
        """
        Automatically generate a prior simulator based on the model specification.
        
        Args:
            default_params: Default parameter structure to use as a template
            data_shape: Shape of the data to simulate (if different from observed data)
            n_groups: Number of groups for hierarchical models
            x_data: Predictor variables for regression-type models
            group_data: Group assignments for hierarchical models
            group_indices: Pre-computed group indices for hierarchical models
            
        Returns:
            Builder with auto-generated prior simulator
        """
        if self.rng_key is None:
            raise ValueError("RNG key must be set before generating auto prior simulator")
            
        if self.parametric_density_fn is None:
            raise ValueError("Parametric density function must be set before generating auto prior simulator")
            
        builder = self._copy()
        
        # Create prior simulator using the current model specification
        def auto_prior_simulator() -> Array:
            # Generate a new key from the base key
            sim_key = jax.random.fold_in(builder.rng_key, 0)
            
            # Split keys for different sampling operations
            keys = jax.random.split(sim_key, 20)  # Get several keys
            key_idx = 0
            
            # Sample parameters from their prior distributions
            # This requires introspection into the parametric_density_fn to extract
            # prior distributions. For now, we'll use a simplified approach.
            # Convert the array of keys to a list for the function call
            keys_list = [keys[i] for i in range(key_idx, key_idx+5)]
            sampled_params = self._sample_params_from_prior(
                default_params, 
                builder.parametric_density_fn,
                keys_list
            )
            key_idx += 5
            
            # Now generate data given the sampled parameters
            # The approach depends on the model type
            
            # For standard (non-hierarchical) models
            if n_groups is None:
                # Get shape of data to generate
                shape = data_shape
                if shape is None and builder.observed_data is not None:
                    # Try to infer from observed data
                    observed = builder.observed_data()
                    shape = observed.shape
                
                # Generate data with appropriate noise
                sim_data = self._simulate_data_from_params(
                    sampled_params, 
                    shape=shape,
                    x_data=x_data,
                    rng_key=keys[key_idx]
                )
            
            # For hierarchical models
            else:
                if x_data is None:
                    raise ValueError("Hierarchical models require x_data for simulation")
                
                if group_indices is None:
                    # Create group indices if not provided
                    if group_data is None:
                        raise ValueError("Hierarchical models require group_data if group_indices not provided")
                    local_group_indices = []
                    for i in range(n_groups):
                        indices = jnp.where(group_data == i)[0]
                        local_group_indices.append(indices)
                else:
                    local_group_indices = group_indices
                
                # Generate hierarchical data
                sim_data = self._simulate_hierarchical_data(
                    sampled_params,
                    x_data=x_data,
                    group_indices=local_group_indices,
                    n_groups=n_groups,
                    rng_key=keys[key_idx]
                )
            
            return sim_data
            
        builder.prior_simulator = auto_prior_simulator
        return builder
    
    def _extract_log_priors(
        self, 
        parametric_density_fn: ParametricDensityFn[P],
        params: P
    ) -> dict[str, Callable[[Array], Array]]:
        """
        Extract prior log probability functions from the parametric density function.
        This is a placeholder for a more sophisticated implementation.
        """
        # This would be a more sophisticated implementation that extracts priors from the model
        # For now, we'll return a dummy implementation
        return {}
    
    def _sample_params_from_prior(
        self, 
        default_params: P, 
        parametric_density_fn: ParametricDensityFn[P],
        keys: list[jax.Array]
    ) -> P:
        """
        Sample parameters from their prior distributions.
        This is a placeholder for actual implementation that would
        introspect the parametric density function.
        """
        # For demonstration, returning a copy of default params
        # In a real implementation, you'd sample from extracted priors
        return default_params
    
    def _simulate_data_from_params(
        self, 
        params: P, 
        shape: Optional[tuple[int, ...]] = None,
        x_data: Optional[Array] = None,
        rng_key: Optional[jax.Array] = None
    ) -> Array:
        """
        Generate simulated data given parameters.
        This is a placeholder for model-specific simulation logic.
        """
        # Default implementation for simple models
        if shape is None:
            raise ValueError("Data shape must be provided for simulation")
            
        if rng_key is None:
            rng_key = jax.random.PRNGKey(0)
            
        # Basic simulation - noise around zero
        # In a real implementation, you'd use the likelihood function
        return jax.random.normal(rng_key, shape)
    
    def _simulate_hierarchical_data(
        self,
        params: P,
        x_data: Array,
        group_indices: list[Array],
        n_groups: int,
        rng_key: Optional[jax.Array] = None
    ) -> Array:
        """
        Generate simulated hierarchical data given parameters.
        This is a placeholder for hierarchical model simulation logic.
        """
        if rng_key is None:
            rng_key = jax.random.PRNGKey(0)
            
        # Initialize simulated data
        simulated_data = jnp.zeros_like(x_data)
        
        # This is a simplified implementation
        # In a real implementation, you would extract specific parameters
        # from the params object and use them to generate data
        
        # Check if params has alpha and beta attributes (common for hierarchical regression)
        has_alpha_beta = hasattr(params, 'alpha') and hasattr(params, 'beta')
        has_sigma = hasattr(params, 'sigma') or hasattr(params, 'log_sigma')
        
        if has_alpha_beta and has_sigma:
            # Extract parameters for hierarchical regression
            alpha = getattr(params, 'alpha')
            beta = getattr(params, 'beta')
            
            # Get sigma, handling log parameterization
            if hasattr(params, 'sigma'):
                sigma = getattr(params, 'sigma')
            else:
                log_sigma = getattr(params, 'log_sigma')
                sigma = jnp.exp(log_sigma)
                
            # Generate data for each group
            for i in range(n_groups):
                # Get indices for this group
                indices = group_indices[i]
                # Extract group data
                x_group = x_data[indices]
                
                # Generate group-specific mean prediction
                mean_pred = alpha[i] + beta[i] * x_group
                
                # Generate random noise
                key_noise = jax.random.fold_in(rng_key, i)
                noise = jax.random.normal(key_noise, shape=mean_pred.shape) * sigma
                
                # Set simulated data for this group
                simulated_data = simulated_data.at[indices].set(mean_pred + noise)
                
            return simulated_data
        
        # Default fallback for unknown hierarchical structure
        return jax.random.normal(rng_key, x_data.shape)

    def with_posterior_simulator(
        self,
        posterior_samples: Array,
        data_generator: Callable[[Array], Array],
    ) -> 'EnhancedModelBuilder[P]':
        """Set the posterior simulator function for posterior predictive context."""
        # Just use the first sample for simplicity
        # In a real implementation, you would select samples randomly or use all samples
        sample = posterior_samples[0]
        
        builder = self._copy()
        
        def posterior_data_fn() -> Array:
            return data_generator(sample)

        builder.posterior_simulator = posterior_data_fn
        return builder
        
    def with_posterior_data_simulator(
        self,
        data_simulator_fn: Callable[[P], Array]
    ) -> 'EnhancedModelBuilder[P]':
        """Set a custom posterior data simulator function that takes parameters."""
        builder = self._copy()
        
        # Create a wrapper function that will be used with actual parameters
        # during posterior predictive checks
        def posterior_wrapper(params: P) -> Callable[[], Array]:
            def data_fn() -> Array:
                return data_simulator_fn(params)
            return data_fn
            
        # Store the wrapper function for later use
        builder._posterior_data_generator = posterior_wrapper
        
        # Create a temporary simulator that will be replaced during workflow execution
        # This is just to satisfy the API until we have actual posterior samples
        def temp_simulator() -> Array:
            # This will be replaced with actual posterior samples
            raise NotImplementedError("Posterior simulator not yet initialized with samples")
            
        builder.posterior_simulator = temp_simulator
        return builder

    def with_auto_posterior_simulator(
        self,
        data_simulator_fn: Optional[Callable[[P], Array]] = None
    ) -> 'EnhancedModelBuilder[P]':
        """
        Automatically create a posterior simulator that reuses the prior simulator logic.
        
        Args:
            data_simulator_fn: Optional custom data simulator function. If not provided,
                               the simulator will be derived from the prior simulator.
        
        Returns:
            Builder with auto-generated posterior simulator
        """
        builder = self._copy()
        
        # If a custom simulator is provided, use it directly
        if data_simulator_fn is not None:
            return builder.with_posterior_data_simulator(data_simulator_fn)
        
        # If no simulator is provided but we have a prior simulator, use the same logic
        # This assumes the prior_simulator is using the same parameter structure
        # and can be applied to posterior parameters
        if not hasattr(self, '_posterior_data_generator'):
            # Create a wrapper that reuses the hierarchical data simulation logic
            def auto_posterior_simulator(params: P) -> Array:
                if hasattr(params, 'alpha') and hasattr(params, 'beta'):
                    # For hierarchical models, try to reuse the hierarchical data simulator
                    # This requires x_data and group_indices to be set globally
                    if not hasattr(self, '_x_data') or not hasattr(self, '_group_indices'):
                        raise ValueError(
                            "Automatic posterior simulation for hierarchical models requires "
                            "x_data and group_indices to be stored. Use with_hierarchical_data "
                            "to set these values."
                        )
                    
                    return self._simulate_hierarchical_data(
                        params,
                        x_data=self._x_data,
                        group_indices=self._group_indices,
                        n_groups=len(self._group_indices),
                        rng_key=self.rng_key
                    )
                else:
                    # For other models, use general data simulation
                    if not hasattr(self, '_data_shape'):
                        raise ValueError(
                            "Automatic posterior simulation requires data_shape to be stored. "
                            "Use with_data_shape to set this value."
                        )
                    
                    return self._simulate_data_from_params(
                        params,
                        shape=self._data_shape,
                        rng_key=self.rng_key
                    )
            
            builder = builder.with_posterior_data_simulator(auto_posterior_simulator)
        
        return builder

    def with_data_shape(self, data_shape: tuple[int, ...]) -> 'EnhancedModelBuilder[P]':
        """Store the data shape for use in automatic simulation."""
        builder = self._copy()
        builder._data_shape = data_shape
        return builder
        
    def with_hierarchical_data(
        self,
        x_data: Array,
        group_indices: list[Array]
    ) -> 'EnhancedModelBuilder[P]':
        """Store hierarchical data information for use in automatic simulation."""
        builder = self._copy()
        builder._x_data = x_data
        builder._group_indices = group_indices
        return builder

    def with_parametric_density_fn(
        self, parametric_density_fn: ParametricDensityFn[P]
    ) -> 'EnhancedModelBuilder[P]':
        """Set the parametric density function."""
        builder = self._copy()
        builder.parametric_density_fn = parametric_density_fn
        return builder

    def build(self) -> EnhancedProbabilisticModel[P]:
        """Build an enhanced probabilistic model."""
        if self.parametric_density_fn is None:
            raise ValueError("Parametric density function must be set")

        # Create a context-aware data function
        context_aware_data = make_context_aware_data(
            observed_data=self.observed_data,
            prior_simulator=self.prior_simulator,
            posterior_simulator=self.posterior_simulator,
        )

        model = EnhancedProbabilisticModel(
            data=context_aware_data,
            parametric_density_fn=self.parametric_density_fn,
        )
        
        # Transfer the posterior data generator if it exists
        if hasattr(self, '_posterior_data_generator'):
            model._posterior_data_generator = self._posterior_data_generator
            
        return model
