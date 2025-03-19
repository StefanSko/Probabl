"""Enhanced probabilistic model builder with context-aware capabilities."""
from dataclasses import dataclass
from typing import Callable, Generic, TypeVar

import jax
from jaxtyping import Array

from distributions.continous import (
    BaseParams,
    DataFn,
    LogDensityFn,
    ParametricDensityFn,
    ProbabilisticModel,
)
from distributions.distribution import Distribution, data_from_distribution
from inference.context import DataContext
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