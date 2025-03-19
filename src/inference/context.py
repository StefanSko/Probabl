from enum import Enum, auto
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Iterator

class DataContext(Enum):
    """Enum for different contexts in the Bayesian workflow."""
    PRIOR_PREDICTIVE = auto()
    INFERENCE = auto()
    POSTERIOR_PREDICTIVE = auto()


@dataclass
class DataContextManager:
    """Manages the current context for data functions.
    
    This class provides a way to track and modify the current context
    in the Bayesian workflow, allowing data functions to behave differently
    based on whether they're being used for prior simulation, inference,
    or posterior prediction.
    """
    _current_context: DataContext = DataContext.INFERENCE
    _context_stack: list[DataContext] = field(default_factory=list)
    
    @property
    def current_context(self) -> DataContext:
        """Get the current data context."""
        return self._current_context
    
    def set_context(self, context: DataContext) -> None:
        """Set the current data context."""
        self._current_context = context
    
    @contextmanager
    def context(self, ctx: DataContext) -> Iterator[None]:
        """Context manager for temporarily changing the data context."""
        old_context = self._current_context
        self._context_stack.append(old_context)
        self._current_context = ctx
        try:
            yield
        finally:
            self._current_context = self._context_stack.pop()


# Global default context manager
default_context_manager = DataContextManager()


def get_current_context() -> DataContext:
    """Get the current data context from the default context manager."""
    return default_context_manager.current_context


def set_context(context: DataContext) -> None:
    """Set the current data context in the default context manager."""
    default_context_manager.set_context(context)


@contextmanager
def context(ctx: DataContext) -> Iterator[None]:
    """Context manager for temporarily changing the data context.
    
    Args:
        ctx: The context to use temporarily
    """
    with default_context_manager.context(ctx):
        yield