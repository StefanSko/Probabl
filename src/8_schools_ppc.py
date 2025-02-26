import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import seaborn as sns

from distributions.continous import run_prior_predictive_check
from src.eigth_schools_from_builder import (
    HierarchicalParams,
    build_eight_schools_model,
    n_schools,
    standard_errors,
    treatment_effects,
)


def main() -> None:
    # Build the model
    model = build_eight_schools_model()

    # Initialize from random position
    rng_key = jax.random.PRNGKey(1000)
    initial_params = HierarchicalParams(jnp.zeros(10))

    # Run prior predictive check
    prior_predictive_samples = run_prior_predictive_check(
        model=model,
        initial_params=initial_params,
        num_samples=1000,
        seed=1000
    )

    # Visualize the prior predictive distribution vs. observed data
    plt.figure(figsize=(12, 6))

    # Plot prior predictive samples
    sns.violinplot(data=prior_predictive_samples, color="lightblue", inner=None, alpha=0.6)

    # Overlay the observed data points
    plt.scatter(range(n_schools), treatment_effects, color='red', s=50, zorder=10, label='Observed')

    # Add error bars for the observed data
    plt.errorbar(
        range(n_schools),
        treatment_effects,
        yerr=standard_errors,
        fmt='none',
        color='red',
        capsize=5
    )

    plt.title('Prior Predictive Check: 8 Schools Model')
    plt.xlabel('School')
    plt.ylabel('Treatment Effect')
    plt.legend()
    plt.grid(True, alpha=0.3)

    # Add a second plot to compare the distribution of all samples
    plt.figure(figsize=(10, 6))

    # Flatten all prior predictive samples
    all_samples = prior_predictive_samples.flatten()

    # Plot histogram of prior predictive samples
    sns.histplot(all_samples, kde=True, stat="density", label="Prior Predictive", color="blue", alpha=0.5)

    # Plot observed data points as rug plot
    sns.rugplot(treatment_effects, color="red", label="Observed", height=0.1)

    plt.title('Prior Predictive Distribution vs. Observed Data')
    plt.xlabel('Treatment Effect')
    plt.ylabel('Density')
    plt.legend()
    plt.grid(True, alpha=0.3)

    # Calculate summary statistics
    print("Prior Predictive Check Summary:")
    print(f"Mean of prior predictive samples: {jnp.mean(prior_predictive_samples):.2f}")
    print(f"Std of prior predictive samples: {jnp.std(prior_predictive_samples):.2f}")
    print(f"Mean of observed data: {jnp.mean(treatment_effects):.2f}")
    print(f"Std of observed data: {jnp.std(treatment_effects):.2f}")

    # Calculate the proportion of prior predictive samples that are more extreme than the observed data
    mean_observed = jnp.mean(treatment_effects)
    mean_samples = jnp.mean(prior_predictive_samples, axis=1)
    p_value = jnp.mean(jnp.abs(mean_samples) >= jnp.abs(mean_observed))
    print(f"Proportion of samples with more extreme mean than observed: {p_value:.3f}")

    plt.show()


if __name__ == "__main__":
    main()