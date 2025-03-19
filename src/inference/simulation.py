from typing import Protocol

from jaxtyping import Array

from distributions.continous import DataFn
from inference.context import DataContext, get_current_context


class ContextAwareDataFn(Protocol):
    """Protocol for context-aware data provider functions."""

    def __call__(self, context: DataContext | None = None) -> Array: ...


def make_context_aware_data(
    observed_data: DataFn | None = None,
    prior_simulator: DataFn | None = None,
    posterior_simulator: DataFn | None = None,
) -> ContextAwareDataFn:
    """Creates a context-aware data function that behaves differently based on the current context.
    
    Args:
        observed_data: Data function to use in INFERENCE context
        prior_simulator: Data function to use in PRIOR_PREDICTIVE context
        posterior_simulator: Data function to use in POSTERIOR_PREDICTIVE context
        
    Returns:
        A context-aware data function that returns appropriate data based on the context

    """
    def context_aware_data_fn(context: DataContext | None = None) -> Array:
        # Use provided context or get from global context manager
        ctx = context or get_current_context()

        if ctx == DataContext.INFERENCE:
            if observed_data is None:
                raise ValueError("No observed data function provided for INFERENCE context")
            return observed_data()

        if ctx == DataContext.PRIOR_PREDICTIVE:
            if prior_simulator is None:
                raise ValueError("No prior simulator function provided for PRIOR_PREDICTIVE context")
            return prior_simulator()

        if ctx == DataContext.POSTERIOR_PREDICTIVE:
            if posterior_simulator is None:
                raise ValueError("No posterior simulator function provided for POSTERIOR_PREDICTIVE context")
            return posterior_simulator()

        raise ValueError(f"Unknown context: {ctx}")

    return context_aware_data_fn
