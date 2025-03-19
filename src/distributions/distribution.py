from abc import ABC, abstractmethod
from typing import Generic, Protocol, TypeVar

import jax
import jax.numpy as jnp
from jaxtyping import Array

from distributions.continous import BaseParams, LogDensityFn


P = TypeVar('P', bound=BaseParams)


class Distribution(Generic[P], ABC):
    """Abstract base class for probability distributions.
    
    Encapsulates both the log PDF and sampling functionality for a distribution.
    """
    
    @abstractmethod
    def log_prob(self, params: P) -> LogDensityFn:
        """Returns a function that computes the log probability of data given parameters.
        
        Args:
            params: Parameters for the distribution
            
        Returns:
            A function that takes data and returns log probabilities
        """
        ...
    
    @abstractmethod
    def sample(self, params: P, rng_key: jax.Array, sample_shape: tuple[int, ...] = (1,)) -> Array:
        """Generate samples from the distribution.
        
        Args:
            params: Parameters for the distribution
            rng_key: JAX random key for reproducibility
            sample_shape: Shape of the output samples
            
        Returns:
            Array of samples from the distribution
        """
        ...


class NormalDistribution(Distribution[P]):
    """Normal (Gaussian) distribution."""
    
    def log_prob(self, params: P) -> LogDensityFn:
        """Returns a function that computes the log probability density of a normal distribution.
        
        Args:
            params: Parameters with loc and scale attributes
            
        Returns:
            A function that takes data and returns log probabilities
        """
        loc = params.loc if hasattr(params, 'loc') else 0.0
        scale = params.scale if hasattr(params, 'scale') else 1.0
        
        def log_prob_fn(data: Array) -> Array:
            return jnp.sum(jax.scipy.stats.norm.logpdf(data, loc, scale))
        
        return log_prob_fn
    
    def sample(self, params: P, rng_key: jax.Array, sample_shape: tuple[int, ...] = (1,)) -> Array:
        """Generate samples from a normal distribution.
        
        Args:
            params: Parameters with loc and scale attributes
            rng_key: JAX random key for reproducibility
            sample_shape: Shape of the output samples
            
        Returns:
            Array of samples from the normal distribution
        """
        loc = params.loc if hasattr(params, 'loc') else 0.0
        scale = params.scale if hasattr(params, 'scale') else 1.0
        
        return jax.random.normal(rng_key, sample_shape) * scale + loc


class BetaDistribution(Distribution[P]):
    """Beta distribution."""
    
    def log_prob(self, params: P) -> LogDensityFn:
        """Returns a function that computes the log probability density of a beta distribution.
        
        Args:
            params: Parameters with a and b attributes
            
        Returns:
            A function that takes data and returns log probabilities
        """
        a = params.a if hasattr(params, 'a') else 1.0
        b = params.b if hasattr(params, 'b') else 1.0
        
        def log_prob_fn(data: Array) -> Array:
            return jnp.sum(jax.scipy.stats.beta.logpdf(data, a, b))
        
        return log_prob_fn
    
    def sample(self, params: P, rng_key: jax.Array, sample_shape: tuple[int, ...] = (1,)) -> Array:
        """Generate samples from a beta distribution.
        
        Args:
            params: Parameters with a and b attributes
            rng_key: JAX random key for reproducibility
            sample_shape: Shape of the output samples
            
        Returns:
            Array of samples from the beta distribution
        """
        a = params.a if hasattr(params, 'a') else 1.0
        b = params.b if hasattr(params, 'b') else 1.0
        
        return jax.random.beta(rng_key, a, b, sample_shape)


class GammaDistribution(Distribution[P]):
    """Gamma distribution."""
    
    def log_prob(self, params: P) -> LogDensityFn:
        """Returns a function that computes the log probability density of a gamma distribution.
        
        Args:
            params: Parameters with shape and scale attributes
            
        Returns:
            A function that takes data and returns log probabilities
        """
        shape = params.shape if hasattr(params, 'shape') else 1.0
        scale = params.scale if hasattr(params, 'scale') else 1.0
        
        def log_prob_fn(data: Array) -> Array:
            return jnp.sum(jax.scipy.stats.gamma.logpdf(data, shape, scale=scale))
        
        return log_prob_fn
    
    def sample(self, params: P, rng_key: jax.Array, sample_shape: tuple[int, ...] = (1,)) -> Array:
        """Generate samples from a gamma distribution.
        
        Args:
            params: Parameters with shape and scale attributes
            rng_key: JAX random key for reproducibility
            sample_shape: Shape of the output samples
            
        Returns:
            Array of samples from the gamma distribution
        """
        shape = params.shape if hasattr(params, 'shape') else 1.0
        scale = params.scale if hasattr(params, 'scale') else 1.0
        
        return jax.random.gamma(rng_key, shape, sample_shape) * scale


# Create distribution instances
normal_distribution = NormalDistribution()
beta_distribution = BetaDistribution()
gamma_distribution = GammaDistribution()


def data_from_distribution(
    distribution: Distribution[P],
    params: P,
    rng_key: jax.Array,
    sample_shape: tuple[int, ...] = (1,),
) -> Callable[[], Array]:
    """Create a data function that returns samples from a distribution.
    
    Args:
        distribution: The distribution to sample from
        params: Parameters for the distribution
        rng_key: JAX random key for reproducibility
        sample_shape: Shape of the output samples
        
    Returns:
        A data function that returns samples from the distribution
    """
    # Pre-generate samples for efficiency
    samples = distribution.sample(params, rng_key, sample_shape)
    
    def data_fn() -> Array:
        return samples
    
    return data_fn