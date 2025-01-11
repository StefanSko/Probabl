import blackjax
import jax
import jax.numpy as jnp


def sample_nuts(model, num_samples, *, num_warmup=1000, seed=0):
    """Sample from a model using NUTS via BlackJAX."""

    def log_prob_fn(params):
        return model.log_prob(params)

    # Initialize the sampler
    initial_position = model.initial_values()
    nuts = blackjax.nuts(log_prob_fn)

    # Run warmup
    rng_key = jax.random.PRNGKey(seed)
    state = nuts.init(initial_position)
    state, kernel = blackjax.window_adaptation(
        nuts,
        rng_key,
        initial_position,
        num_warmup
    )

    # Sample
    def one_step(state, rng_key):
        state, _ = kernel(rng_key, state)
        return state, state

    keys = jax.random.split(rng_key, num_samples)
    states, samples = jax.lax.scan(one_step, state, keys)

    return samples