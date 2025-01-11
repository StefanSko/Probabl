# src/probflow/core/variable.py
from dataclasses import dataclass
from typing import Optional


@dataclass
class Variable:
    """Represents a random variable in the model."""

    name: str
    prior: Optional['Distribution'] = None
    _value: Optional[Any] = None

    def __invert__(self):
        """Handle the ~ operator for distribution assignment."""
        # Get current model context
        model = Model.get_current()
        if model is None:
            raise RuntimeError("Variable definition must be within a model context")
        return DistributionAssignment(self)

    def __rshift__(self, distribution):
        """Handle the >> operator for generative assignments."""
        model = Model.get_current()
        if model is None:
            raise RuntimeError("Generative assignment must be within a model context")

        relationship = Relationship(self, distribution)
        model._relationships.append(relationship)
        return relationship


class Relationship:
    """Represents a generative relationship between variables."""

    def __init__(self, target, distribution):
        self.target = target
        self.distribution = distribution

    def log_prob(self, reader):
        """Compute log probability of this relationship."""
        # Get parameter values from reader
        params = {}
        for name, param in self.distribution.parameters.items():
            params[name] = reader.read(param)

        # Compute log prob
        return self.distribution.log_prob(self.target, **params)