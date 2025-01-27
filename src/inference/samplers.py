from typing import Callable

import blackjax
import jax
from blackjax.base import State
from jaxtyping import Array, Float


def sample_nuts(log_prob_fn: Callable[[Array], Array], initial_position: Array, num_samples: int,
                seed: int) -> Float[Array, "n_samples 1"]:

    nuts = blackjax.nuts(log_prob_fn)

    # Run warmup
    rng_key = jax.random.PRNGKey(seed)
    state = nuts.init(initial_position)

    # Sample
    @jax.jit
    def one_step(
            state: State,
            rng_key: Array
    ) -> tuple[State, State]:
        state, _ = nuts.step(rng_key, state)
        return state, state

    keys = jax.random.split(rng_key, num_samples)
    states, samples = jax.lax.scan(one_step, state, keys)

    return samples

