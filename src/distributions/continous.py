from typing import Callable, TypeAlias

import jax.numpy as jnp
from jax.scipy import stats
from jaxtyping import Array

# Type aliases for Bayesian probability functions
LogDensityFn: TypeAlias = Callable[[Array], Array]  # log p(x|θ)
#TODO: FIXME!!
Parameters = tuple[Array | float, ...]  # θ = (θ₁, θ₂, ...)
ParametricDensityFn: TypeAlias = Callable[[Parameters], LogDensityFn]  # p(x|θ)
LogPDFFn: TypeAlias = Callable[[Array, Array | float, Array | float], Array]  # raw PDF computation


#Type alias for a data provider
Data: TypeAlias = Callable[[], Array]


def make_distribution(
    logpdf_fn: LogPDFFn
) -> ParametricDensityFn:
    def distribution(loc: Array | float, scale: Array | float) -> LogDensityFn:
        def log_prob(data: Array) -> Array:
            return jnp.sum(logpdf_fn(data, loc, scale))
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


class ProbabilisticModel:


    data: Data
    parametric_density: ParametricDensityFn

    def __init__(self, data: Data, parametric_density_fn: ParametricDensityFn) -> None:
        self.data = data
        self.parametric_density_fn = parametric_density_fn

    def forward(self, params: Array) -> Array:
        return self.parametric_density_fn(params)(self.data())

class ProbabilisticModelBuilder:

    data: Data | None
    parametric_density_fn: ParametricDensityFn | None

    def __init__(self) -> None:
        self.data = None
        self.parametric_density_fn = None

    def with_data(self, data: Data) -> 'ProbabilisticModelBuilder':
        self.data = data
        return self

    def with_parametric_density_fn(self, parametric_density_fn: ParametricDensityFn) \
            -> 'ProbabilisticModelBuilder':
        self.parametric_density_fn = parametric_density_fn
        return self

    def from_model(self, model: ProbabilisticModel) -> 'ProbabilisticModelBuilder':
        self.data = model.data
        self.parametric_density_fn = model.parametric_density_fn
        return self

    def build(self) -> ProbabilisticModel:
        return ProbabilisticModel(self.data, self.parametric_density_fn)

