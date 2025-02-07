from dataclasses import dataclass
from typing import Callable, Generic, TypeAlias, TypeVar

import jax.numpy as jnp
from jax.scipy import stats
from jaxtyping import Array


@dataclass(frozen=True)
class BaseParams:
    pass

@dataclass(frozen=True)
class Parameters(BaseParams):
    params: Array

@dataclass(frozen=True)
class LocationScaleParams(BaseParams):
    loc: Array | float
    scale: Array | float

@dataclass(frozen=True)
class BetaParams(BaseParams):
    a: Array | float
    b: Array | float

@dataclass(frozen=True)
class GammaParams(BaseParams):
    shape: Array | float
    scale: Array | float

P = TypeVar('P', bound=BaseParams)

# Type aliases for Bayesian probability functions
LogDensityFn: TypeAlias = Callable[[Array], Array]  # log p(x|θ)
ParametricDensityFn: TypeAlias = Callable[[P], LogDensityFn]  # p(x|θ)
LogPDFFn: TypeAlias = Callable[[Array, Array | float, Array | float], Array]  # raw PDF computation


#Type alias for a data provider
Data: TypeAlias = Callable[[], Array]


def make_distribution(
    logpdf_fn: LogPDFFn
) -> ParametricDensityFn[LocationScaleParams]:
    def distribution(params: LocationScaleParams) -> LogDensityFn:
        def log_prob(data: Array) -> Array:
            return jnp.sum(logpdf_fn(data, params.loc, params.scale))
        return log_prob
    return distribution


# Standard distributions with loc and scale
normal = make_distribution(stats.norm.logpdf)
laplace = make_distribution(stats.laplace.logpdf)
cauchy = make_distribution(stats.cauchy.logpdf)
exp = make_distribution(stats.expon.logpdf)


# Custom parameter distributions
def beta(a: Array, b: Array) -> LogDensityFn:
    def log_prob(data: Array) -> Array:
        return jnp.sum(stats.beta.logpdf(data, a, b))
    return log_prob

def gamma(shape: Array, scale: Array) -> LogDensityFn:
    def log_prob(data: Array) -> Array:
        return jnp.sum(stats.gamma.logpdf(data, shape, scale=scale))
    return log_prob


class ProbabilisticModel(Generic[P]):
    data: Data
    parametric_density_fn: ParametricDensityFn[P]

    def __init__(self, data: Data, parametric_density_fn: ParametricDensityFn[P]) -> None:
        self.data = data
        self.parametric_density_fn = parametric_density_fn

    def forward(self, params: P) -> Array:
        return self.parametric_density_fn(params)(self.data())

class ProbabilisticModelBuilder(Generic[P]):
    data: Data | None
    parametric_density_fn: ParametricDensityFn[P] | None

    def __init__(self) -> None:
        self.data = None
        self.parametric_density_fn = None

    def with_data(self, data: Data) -> 'ProbabilisticModelBuilder[P]':
        self.data = data
        return self

    def with_parametric_density_fn(self, parametric_density_fn: ParametricDensityFn[P]) \
            -> 'ProbabilisticModelBuilder[P]':
        self.parametric_density_fn = parametric_density_fn
        return self

    def from_model(self, model: ProbabilisticModel[P]) -> 'ProbabilisticModelBuilder[P]':
        self.data = model.data
        self.parametric_density_fn = model.parametric_density_fn
        return self

    def build(self) -> ProbabilisticModel[P]:
        if self.data is None or self.parametric_density_fn is None:
            raise ValueError("Both data and parametric_density_fn must be set")
        return ProbabilisticModel(self.data, self.parametric_density_fn)

