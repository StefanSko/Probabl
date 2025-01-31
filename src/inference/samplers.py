from typing import Callable, cast

import blackjax
import jax
from blackjax.base import State
from blackjax.mcmc.hmc import HMCState
from jax import Array


def nuts_with_warmup(
    logprob_fn: Callable[[Array], Array],
    initial_position: Array,
    rng_key: Array,
    num_samples: int = 1000,
    target_acceptance_rate: float = 0.8,
) -> Array:

    # Initialize the warmup adaptation
    warmup = blackjax.window_adaptation(
        blackjax.nuts,
        logprob_fn,
        target_acceptance_rate=target_acceptance_rate,
        is_mass_matrix_diagonal=True,
    )

    # Run the warmup phase
    (state, adapted_params), _ = warmup.run(
        rng_key,
        initial_position
    )

    # Define the NUTS kernel using the adapted parameters
    nuts_kernel = blackjax.nuts(logprob_fn, **adapted_params).step

    # Function to perform one sampling step
    def one_step(state: State, rng_key: jax.Array) -> tuple[State, State]:
        state, _ = nuts_kernel(rng_key, state)
        return state, state

    # Generate random keys for sampling
    sampling_keys = jax.random.split(rng_key, num_samples)

    # Run the sampling phase
    _, states = jax.lax.scan(one_step, state, sampling_keys)
    states = cast(HMCState, states)

    return states.position

