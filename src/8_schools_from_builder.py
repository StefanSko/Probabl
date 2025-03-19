from dataclasses import dataclass
from typing import Callable

import jax
import jax.numpy as jnp
from jaxtyping import Array, Float

from distributions.continous import (
    LocationScaleParams,
    Parameters,
    ProbabilisticModel,
    ProbabilisticModelBuilder,
    normal,
)
from inference.samplers import nuts_with_warmup

# The 8 schools data
treatment_effects = jnp.array([28., 8., -3., 7., -1., 1., 18., 12.])
standard_errors = jnp.array([15., 10., 16., 11., 9., 11., 10., 18.])
n_schools = len(treatment_effects)


@dataclass(frozen=True)
class HierarchicalParams(Parameters):
    params: Float[Array, "10"]  # [mu, log_tau, theta(8)]

    @property
    def mu(self) -> Float[Array, " "]:
        return self.params[0]

    @property
    def log_tau(self) -> Float[Array, " "]:
        return self.params[1]

    @property
    def tau(self) -> Float[Array, " "]:
        return jnp.exp(self.log_tau)

    @property
    def theta(self) -> Array:
        return self.params[2:]


def build_eight_schools_model() -> ProbabilisticModel[HierarchicalParams]:
    # Data provider
    def data_fn() -> Array:
        return treatment_effects

    def parametric_density_fn(params: HierarchicalParams) -> Callable[[Array], Array]:
        def log_prob(data: Array) -> Array:
            # Priors
            prior_mu = normal(LocationScaleParams(0., 1.))(params.mu)
            prior_tau = normal(LocationScaleParams(5., 1.))(params.log_tau)
            prior_theta = normal(LocationScaleParams(params.mu, params.tau))(params.theta)

            # Likelihood
            likelihood = normal(LocationScaleParams(params.theta, standard_errors))(data)

            return prior_mu + prior_tau + prior_theta + likelihood

        return log_prob

    return (ProbabilisticModelBuilder[HierarchicalParams]()
            .with_data(data_fn)
            .with_parametric_density_fn(parametric_density_fn)
            .build())


def main() -> None:
    # Build the model
    model = build_eight_schools_model()

    # Initialize from random position
    rng_key = jax.random.PRNGKey(1000)
    initial_params = HierarchicalParams(jnp.zeros(10))

    # Create the posterior log probability function
    def posterior_log_prob(params: Array) -> Array:
        return model.forward(HierarchicalParams(params))

    # Run the sampler
    samples = nuts_with_warmup(
        posterior_log_prob,
        initial_params.params,
        rng_key,
        num_samples=2000
    )

    # Extract parameters
    mu_samples = samples[:, 0]
    tau_samples = jnp.exp(samples[:, 1])
    theta_samples = samples[:, 2:]

    # Print results
    print("Population parameters:")
    print(f"mu: {jnp.mean(mu_samples):.1f} ± {jnp.std(mu_samples):.1f}")
    print(f"tau: {jnp.mean(tau_samples):.1f} ± {jnp.std(tau_samples):.1f}")

    print("\nSchool-specific effects:")
    for i in range(n_schools):
        school_samples = theta_samples[:, i]
        print(f"School {i + 1}: {jnp.mean(school_samples):.1f} ± {jnp.std(school_samples):.1f}")

    # Visualize the posterior distributions
    import seaborn as sns
    from matplotlib import pyplot as plt

    plt.figure(figsize=(10, 6))
    sns.swarmplot(data=theta_samples[::10, :], s=2, zorder=0)
    plt.show()


if __name__ == "__main__":
    main()