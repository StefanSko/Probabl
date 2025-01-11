import jax.numpy as jnp


class Reader:
    """Handles parameter reading and transformations during inference."""

    def __init__(self, params: dict[str, jnp.ndarray]) -> None:
        # Raw parameters from the sampler
        self._params = params
        # Track accumulated log Jacobian terms
        self._log_jacobian = 0.0
        # Keep track of which parameters we've read
        self._read_params: set = set()

    def read(self, name: str, transform=None) -> jnp.ndarray:
        """Read a parameter, optionally applying a transformation."""
        # Mark this parameter as read
        self._read_params.add(name)

        # Get the raw parameter value
        value = self._params[name]

        if transform is not None:
            # Apply transformation and accumulate Jacobian
            value, log_det_jacobian = transform(value)
            self._log_jacobian += log_det_jacobian

        return value

    def log_jacobian(self) -> float:
        """Get the accumulated log Jacobian term."""
        return self._log_jacobian

    def check_all_used(self) -> None:
        """Verify that all parameters were used."""
        unused = set(self._params.keys()) - self._read_params
        if unused:
            raise ValueError(f"Unused parameters: {unused}")

