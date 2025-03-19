"""Enhanced probabilistic model builder with context-aware capabilities."""
from dataclasses import dataclass
from typing import Callable, Generic, TypeVar

import jax
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

    def forward(self, params: P) -> Array:
        """Compute the log probability of data given parameters."""
        return self.parametric_density_fn(params)(self.data())


class EnhancedModelBuilder(Generic[P]):
    """Builder for enhanced probabilistic models with context-aware data."""

    def __init__(self) -> None:
        """Initialize an empty model builder."""
        self.observed_data: DataFn | None = None
        self.prior_simulator: DataFn | None = None
        self.posterior_simulator: DataFn | None = None
        self.parametric_density_fn: ParametricDensityFn[P] | None = None

    def with_observed_data(self, data_fn: DataFn) -> 'EnhancedModelBuilder[P]':
        """Set the observed data function for inference context."""
        self.observed_data = data_fn
        return self

    def with_prior_simulator(
        self,
        distribution: Distribution[P],
        params: P,
        rng_key: jax.Array,
        sample_shape: tuple[int, ...] = (1,),
    ) -> 'EnhancedModelBuilder[P]':
        """Set the prior simulator function for prior predictive context."""
        self.prior_simulator = data_from_distribution(
            distribution, params, rng_key, sample_shape
        )
        return self
    
    def with_custom_prior_simulator(
        self,
        simulator_fn: Callable[[], P]
    ) -> 'EnhancedModelBuilder[P]':
        """Set a custom prior simulator function."""
        self.prior_simulator = simulator_fn
        return self
        
    def with_prior_data_simulator(
        self,
        data_simulator_fn: Callable[[], Array]
    ) -> 'EnhancedModelBuilder[P]':
        """Set a custom prior data simulator function."""
        self.prior_simulator = data_simulator_fn
        return self

    def with_posterior_simulator(
        self,
        posterior_samples: Array,
        data_generator: Callable[[Array], Array],
    ) -> 'EnhancedModelBuilder[P]':
        """Set the posterior simulator function for posterior predictive context."""
        # Just use the first sample for simplicity
        # In a real implementation, you would select samples randomly or use all samples
        sample = posterior_samples[0]

        def posterior_data_fn() -> Array:
            return data_generator(sample)

        self.posterior_simulator = posterior_data_fn
        return self
        
    def with_posterior_data_simulator(
        self,
        data_simulator_fn: Callable[[P], Array]
    ) -> 'EnhancedModelBuilder[P]':
        """Set a custom posterior data simulator function that takes parameters."""
        # Create a wrapper function that will be used with actual parameters
        # during posterior predictive checks
        def posterior_wrapper(params: P) -> Callable[[], Array]:
            def data_fn() -> Array:
                return data_simulator_fn(params)
            return data_fn
            
        # Store the wrapper function for later use
        self._posterior_data_generator = posterior_wrapper
        
        # Create a temporary simulator that will be replaced during workflow execution
        # This is just to satisfy the API until we have actual posterior samples
        def temp_simulator() -> Array:
            # This will be replaced with actual posterior samples
            raise NotImplementedError("Posterior simulator not yet initialized with samples")
            
        self.posterior_simulator = temp_simulator
        return self

    def with_parametric_density_fn(
        self, parametric_density_fn: ParametricDensityFn[P]
    ) -> 'EnhancedModelBuilder[P]':
        """Set the parametric density function."""
        self.parametric_density_fn = parametric_density_fn
        return self

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

        return EnhancedProbabilisticModel(
            data=context_aware_data,
            parametric_density_fn=self.parametric_density_fn,
        )
