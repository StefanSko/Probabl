"""Utilities for handling structured parameters in Bayesian models."""
from dataclasses import dataclass, field, is_dataclass
from typing import Any, Callable, Dict, Generic, List, Tuple, TypeVar

import jax
import jax.numpy as jnp
from jaxtyping import Array, Float

from distributions.continous import BaseParams


T = TypeVar('T')
P = TypeVar('P', bound=BaseParams)


@dataclass
class ParameterStructure(Generic[P]):
    """Captures the structure of parameters for flattening and unflattening."""
    
    original_shape: dict[str, tuple[int, ...]] = field(default_factory=dict)
    param_names: list[str] = field(default_factory=list)
    param_sizes: list[int] = field(default_factory=list)
    param_class: type[P] | None = None
    
    @classmethod
    def from_params(cls, params: P) -> 'ParameterStructure[P]':
        """Create a parameter structure from a parameter object."""
        if not is_dataclass(params):
            raise ValueError("Parameters must be a dataclass")
        
        structure = cls()
        structure.param_class = params.__class__
        
        # Extract parameter structure from the dataclass
        for field_name, field_value in params.__dict__.items():
            if isinstance(field_value, Array):
                structure.param_names.append(field_name)
                structure.original_shape[field_name] = field_value.shape
                structure.param_sizes.append(jnp.size(field_value))
            
        return structure
    
    def flatten(self, params: P) -> Array:
        """Flatten structured parameters into a 1D array."""
        if not isinstance(params, self.param_class):
            raise TypeError(f"Expected parameters of type {self.param_class}, got {type(params)}")
        
        flat_params = []
        for name in self.param_names:
            param_value = getattr(params, name)
            flat_params.append(param_value.flatten())
        
        return jnp.concatenate(flat_params)
    
    def unflatten(self, flat_params: Array) -> P:
        """Unflatten a 1D array into structured parameters."""
        if self.param_class is None:
            raise ValueError("Parameter class not set")
        
        param_dict = {}
        start_idx = 0
        
        for name, size in zip(self.param_names, self.param_sizes):
            end_idx = start_idx + size
            shape = self.original_shape[name]
            param_dict[name] = flat_params[start_idx:end_idx].reshape(shape)
            start_idx = end_idx
        
        return self.param_class(**param_dict)


def flatten_params(params: P) -> Tuple[Array, ParameterStructure[P]]:
    """Flatten structured parameters into a 1D array.
    
    Args:
        params: Structured parameters (dataclass)
        
    Returns:
        Tuple of flattened parameters and their structure
    """
    structure = ParameterStructure.from_params(params)
    flat_params = structure.flatten(params)
    return flat_params, structure


def unflatten_params(flat_params: Array, structure: ParameterStructure[P]) -> P:
    """Unflatten a 1D array into structured parameters.
    
    Args:
        flat_params: 1D array of flattened parameters
        structure: Structure of the parameters
        
    Returns:
        Structured parameters (dataclass)
    """
    return structure.unflatten(flat_params)


# Parameter transformation utilities
@dataclass
class ParameterConstraint:
    """Constraint for a parameter (e.g., positive, bounded, etc.)."""
    
    transform: Callable[[Array], Array]
    inverse_transform: Callable[[Array], Array]
    log_det_jacobian: Callable[[Array], Array]


class ConstraintRegistry:
    """Registry of common parameter constraints."""
    
    @staticmethod
    def positive() -> ParameterConstraint:
        """Constraint for positive parameters (using softplus)."""
        return ParameterConstraint(
            transform=lambda x: jnp.log(jnp.exp(x) + 1),  # softplus
            inverse_transform=lambda y: jnp.log(jnp.exp(y) - 1),  # inverse softplus
            log_det_jacobian=lambda y: -jnp.log(1 - jnp.exp(-y)),  # log det of Jacobian
        )
    
    @staticmethod
    def bounded(lower: float, upper: float) -> ParameterConstraint:
        """Constraint for bounded parameters (using sigmoid)."""
        scale = upper - lower
        
        def transform(x):
            return lower + scale * jax.nn.sigmoid(x)
        
        def inverse_transform(y):
            z = (y - lower) / scale
            return jnp.log(z / (1 - z))
        
        def log_det_jacobian(y):
            z = (y - lower) / scale
            return -jnp.log(z) - jnp.log(1 - z) - jnp.log(scale)
        
        return ParameterConstraint(
            transform=transform,
            inverse_transform=inverse_transform,
            log_det_jacobian=log_det_jacobian,
        )


def transform_params(
    params: P, 
    constraints: dict[str, ParameterConstraint]
) -> Tuple[P, Callable[[P], Float[Array, ""]]]:
    """Apply constraints to parameters.
    
    Args:
        params: Structured parameters
        constraints: Dictionary mapping parameter names to constraints
        
    Returns:
        Tuple of transformed parameters and a function to compute the log determinant of the Jacobian
    """
    param_dict = {}
    
    for name in params.__dict__:
        if name in constraints:
            param_dict[name] = constraints[name].transform(getattr(params, name))
        else:
            param_dict[name] = getattr(params, name)
    
    transformed_params = params.__class__(**param_dict)
    
    def log_det_fn(transformed: P) -> Float[Array, ""]:
        log_det = 0.0
        
        for name, constraint in constraints.items():
            log_det = log_det + jnp.sum(constraint.log_det_jacobian(getattr(transformed, name)))
        
        return log_det
    
    return transformed_params, log_det_fn


def untransform_params(
    transformed_params: P, 
    constraints: dict[str, ParameterConstraint]
) -> P:
    """Reverse constraints applied to parameters.
    
    Args:
        transformed_params: Transformed parameters
        constraints: Dictionary mapping parameter names to constraints
        
    Returns:
        Original untransformed parameters
    """
    param_dict = {}
    
    for name in transformed_params.__dict__:
        if name in constraints:
            param_dict[name] = constraints[name].inverse_transform(getattr(transformed_params, name))
        else:
            param_dict[name] = getattr(transformed_params, name)
    
    return transformed_params.__class__(**param_dict)