from contextlib import contextmanager
from typing import Dict, Generator

import jax.numpy as jnp

from core.reader import Reader
from core.variable import Variable


class Model:
    """Base class for probabilistic models."""

    def __init__(self):
        self._variables = {}
        self._data = {}
        self._relationships = []
        self._current_context = None

    def data(self, name: str, *, prior=None) -> Variable:
        """Register a data variable with optional prior for simulation."""
        var = Variable(name, prior=prior)
        self._data[name] = var
        return var

    @contextmanager
    def model_context(self)  -> Generator['Model', None, None]:
        """Context manager for model building."""
        prev_context = self._current_context
        self._current_context = self
        try:
            yield self
        finally:
            self._current_context = prev_context

    def __enter__(self) -> 'Model':
        return self.model_context().__enter__()

    def __exit__(self, *args) -> bool:  # noqa: ANN002
        return self.model_context().__exit__(*args)

    def log_prob(self, params: dict[str, jnp.ndarray]) -> float:
        """Compute log probability for BlackJAX."""
        reader = Reader(params)
        log_prob = 0.0

        # Accumulate log probs from all relationships
        for relationship in self._relationships:
            log_prob += relationship.log_prob(reader)

        # Add log det jacobian from parameter transforms
        log_prob += reader.log_jacobian()

        return log_prob
