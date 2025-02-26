from dataclasses import dataclass
from typing import Callable, Generic, TypeAlias, TypeVar

import jax
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
    loc: Array | float = 0.0
    scale: Array | float = 1.0

@dataclass(frozen=True)
class BetaParams(BaseParams):
    a: Array | float
    b: Array | float

@dataclass(frozen=True)
class GammaParams(BaseParams):
    shape: Array | float
    scale: Array | float

@dataclass(frozen=True)
class JointParams(Parameters):

    """Parameters for joint model containing both model parameters and data."""
    param_size: int  # Indicates where to split between parameters and data

    @property
    def model_params(self) -> Array:
        """Extract just the model parameters from the joint parameters."""
        return self.params[:self.param_size]

    @property
    def data_params(self) -> Array:
        """Extract the data portion from the joint parameters."""
        return self.params[self.param_size:]

    @classmethod
    def from_model_params_and_data(cls, model_params: Array, data: Array) -> 'JointParams':
        """Create joint parameters by concatenating model parameters and data."""
        param_size = model_params.shape[0]
        return cls(params=jnp.concatenate([model_params, data]), param_size=param_size)

P = TypeVar('P', bound=BaseParams)

# Type aliases for Bayesian probability functions
LogDensityFn: TypeAlias = Callable[[Array], Array]  # log p(x|θ)
ParametricDensityFn: TypeAlias = Callable[[P], LogDensityFn]  # p(x|θ)
LogPDFFn: TypeAlias = Callable[[Array, Array | float, Array | float], Array]  # raw PDF computation
PriorDensityFn: TypeAlias = Callable[[P], Array]  # p(θ)
LikelihoodFn: TypeAlias = Callable[[P, Array], Array]  # p(data|params)

# Type alias for a data provider
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
    likelihood_fn: LikelihoodFn[P] | None = None
    prior_density_fn: PriorDensityFn[P] | None = None

    def __init__(self, data: Data, parametric_density_fn: ParametricDensityFn[P]) -> None:
        self.data = data
        self.parametric_density_fn = parametric_density_fn

    def forward(self, params: P) -> Array:
        return self.parametric_density_fn(params)(self.data())


class ProbabilisticModelBuilder(Generic[P]):
    data: Data | None
    parametric_density_fn: ParametricDensityFn[P] | None
    likelihood_fn: LikelihoodFn[P] | None
    prior_density_fn: PriorDensityFn[P] | None

    def __init__(self) -> None:
        self.data = None
        self.parametric_density_fn = None
        self.likelihood_fn = None
        self.prior_density_fn = None

    def with_data(self, data: Data) -> 'ProbabilisticModelBuilder[P]':
        self.data = data
        return self

    def with_parametric_density_fn(self, parametric_density_fn: ParametricDensityFn[P]) \
            -> 'ProbabilisticModelBuilder[P]':
        self.parametric_density_fn = parametric_density_fn
        return self

    def with_likelihood_fn(self, likelihood_fn: LikelihoodFn[P]) \
            -> 'ProbabilisticModelBuilder[P]':
        """Set the likelihood function that will be used for data generation."""
        self.likelihood_fn = likelihood_fn
        return self

    def with_prior_density_fn(self, prior_density_fn: PriorDensityFn[P]) \
            -> 'ProbabilisticModelBuilder[P]':
        """Set the prior density function for model parameters."""
        self.prior_density_fn = prior_density_fn
        return self

    def from_model(self, model: ProbabilisticModel[P]) -> 'ProbabilisticModelBuilder[P]':
        self.data = model.data
        self.parametric_density_fn = model.parametric_density_fn
        self.likelihood_fn = model.likelihood_fn
        self.prior_density_fn = model.prior_density_fn
        return self

    def build(self) -> ProbabilisticModel[P]:
        if self.data is None or self.parametric_density_fn is None:
            raise ValueError("Both data and parametric_density_fn must be set")

        model = ProbabilisticModel(self.data, self.parametric_density_fn)

        if self.likelihood_fn:
            model.likelihood_fn = self.likelihood_fn

        if self.prior_density_fn:
            model.prior_density_fn = self.prior_density_fn

        return model


class JointModelBuilder(Generic[P]):
    """Builder for joint models that treat both parameters and data as variables to sample."""

    def __init__(self, model: ProbabilisticModel[P]) -> None:
        self.model = model

    def build(self) -> ProbabilisticModel[JointParams]:
        """Build a joint model that samples both parameters and data."""
        if not hasattr(self.model, 'likelihood_fn') or self.model.likelihood_fn is None:
            raise ValueError("Model must have a likelihood function for joint sampling")

        if not hasattr(self.model, 'prior_density_fn') or self.model.prior_density_fn is None:
            raise ValueError("Model must have a prior density function for joint sampling")

        # Original data shape for reconstruction
        data_shape = self.model.data().shape

        # Create a joint log probability function
        def joint_parametric_density_fn(joint_params: JointParams) -> LogDensityFn:
            def joint_log_prob(_: Array) -> Array:
                # Extract model parameters and data from joint parameters
                raw_model_params = joint_params.model_params
                data = joint_params.data_params.reshape(data_shape)

                # Get the parameter type from the model's type annotation
                # This handles both generic and concrete parameter types
                param_type = None
                try:
                    # Try to get the parameter type from __annotations__
                    param_type = self.model.parametric_density_fn.__annotations__['params']
                    # If it's a generic type with args, extract the concrete type
                    if hasattr(param_type, '__args__'):
                        param_type = param_type.__args__[0]
                except (KeyError, AttributeError, IndexError):
                    # Fallback: try to infer from the model's type parameter
                    import inspect
                    for base in inspect.getmro(type(self.model)):
                        if hasattr(base, '__orig_bases__'):
                            for orig_base in base.__orig_bases__:
                                if hasattr(orig_base, '__args__') and len(orig_base.__args__) > 0:
                                    param_type = orig_base.__args__[0]
                                    break
                            if param_type is not None:
                                break

                # If we still couldn't determine the parameter type, raise an error
                if param_type is None:
                    raise ValueError("Could not determine parameter type for the model")

                # Create the model parameters with the appropriate type
                model_params = param_type(raw_model_params)

                # Calculate prior log probability (p(θ))
                prior_log_prob = self.model.prior_density_fn(model_params)

                # Calculate likelihood log probability (p(y|θ))
                likelihood_log_prob = self.model.likelihood_fn(model_params, data)

                # Return the joint probability: p(θ, y) = p(θ) × p(y|θ)
                return prior_log_prob + likelihood_log_prob

            return joint_log_prob

        # Create a dummy data function that returns an empty array
        def dummy_data_fn() -> Array:
            return jnp.array([])

        # Build and return the joint model
        return ProbabilisticModel(dummy_data_fn, joint_parametric_density_fn)


def run_prior_predictive_check(
    model: ProbabilisticModel[P],
    initial_params: P,
    num_samples: int = 1000,
    seed: int = 0
) -> Array:

    # Create joint model
    joint_model = JointModelBuilder(model).build()

    # Get the original data shape for initialization
    data_shape = model.data().shape
    data_size = jnp.prod(jnp.array(data_shape))

    # Create initial joint parameters (parameters + dummy data)
    initial_data = jnp.zeros(data_size)
    joint_params = JointParams.from_model_params_and_data(
        initial_params.params,
        initial_data
    )

    # Create posterior log probability function
    def joint_posterior_log_prob(params: Array) -> Array:
        param_size = initial_params.params.shape[0]
        return joint_model.forward(JointParams(params, param_size))

    # Run HMC on joint distribution
    from inference.samplers import nuts_with_warmup
    rng_key = jax.random.PRNGKey(seed)
    samples = nuts_with_warmup(
        joint_posterior_log_prob,
        joint_params.params,
        rng_key,
        num_samples=num_samples
    )

    # Extract and return data samples
    param_size = initial_params.params.shape[0]
    return samples[:, param_size:].reshape((-1, *data_shape))


