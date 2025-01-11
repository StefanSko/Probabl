from core.model import Model
from core.reader import Reader


class Variable:
    def __init__(self, name) -> None:
        self.name = name
        self.distribution = None

    def __invert__(self):
        """Called when ~ is used (e.g., y ~ Normal(...))."""
        # Get the current model context
        model = ModelContext.get_current()
        if model is None:
            raise RuntimeError("Variable definition must be within a model context")

        # Return a special object that will handle the distribution assignment
        return DistributionAssignment(self, model)


class DistributionAssignment:
    def __init__(self, variable: Variable, model: Model) -> None:
        self.variable = variable
        self.model = model

    def __rrshift__(self, distribution):
        """Handle the actual distribution assignment."""
        # Record this relationship in the model
        relationship = Relationship(self.variable, distribution)
        self.model._relationships.append(relationship)
        return relationship

class Relationship:
    """Represents a generative relationship between variables."""

    def __init__(self, target, distribution):
        self.target = target
        self.distribution = distribution

    def log_prob(self, reader: Reader):
        """Compute log probability of this relationship."""
        # Get parameter values from reader
        params = {}
        for name, param in self.distribution.parameters.items():
            params[name] = reader.read(param)

        # Compute log prob
        return self.distribution.log_prob(self.target, **params)
