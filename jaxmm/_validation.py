"""Precision and scalar parameter checks shared by molecular entry points."""

import jax
import jax.numpy as jnp
import numpy as np


def require_float64(value=None):
    """Require x64 mode and, if supplied, a float64 array.

    This checks dtype metadata without inspecting traced values. It never
    changes global precision settings or silently converts float32 arrays.
    """
    if not jax.config.jax_enable_x64:
        raise RuntimeError(
            "jaxmm requires float64. Call jax.config.update('jax_enable_x64', True) "
            "before importing jaxmm or using its functions."
        )
    if value is not None:
        if not hasattr(value, "dtype"):
            raise TypeError("Expected a float64 coordinate array with dtype metadata.")
        if value.dtype != jnp.float64:
            raise TypeError(f"Expected float64 coordinates, got {value.dtype}.")


def validate_temperature(value):
    """Require a positive finite real scalar in Kelvin when concrete.

    Shape and real numeric dtype are checked under tracing. The plain object
    placeholder used by pytree reconstruction has no metadata and is skipped.
    """
    if type(value) is object:
        return
    traced = isinstance(value, jax.core.Tracer)
    shape = value.shape if traced else np.shape(value)
    if shape != ():
        raise ValueError(f"temperature must be a scalar in Kelvin, got shape {shape}.")
    dtype = value.dtype if traced else np.asarray(value).dtype
    if not (np.issubdtype(dtype, np.floating) or np.issubdtype(dtype, np.integer)):
        raise ValueError(f"temperature must be a real numeric scalar in Kelvin, got {dtype}.")
    if not traced and (not np.isfinite(value) or value <= 0):
        raise ValueError(f"temperature must be positive and finite in Kelvin, got {value}.")


def _check_positions_shape(positions, term):
    """Raise unless positions has shape (..., n_atoms, 3).

    Args:
        positions: Candidate coordinate array.
        term: Name of the calling energy term, used in the message.

    Raises:
        ValueError: If the array is not at least 2D with a trailing axis of 3.
    """
    if jnp.ndim(positions) < 2:
        raise ValueError(
            f"{term}: positions must have shape (..., n_atoms, 3), got "
            f"{jnp.shape(positions)}."
        )
    if positions.shape[-1] != 3:
        raise ValueError(
            f"{term}: positions last dimension is {positions.shape[-1]}, "
            "expected 3 (x, y, z)."
        )


def _check_atom_count(positions, n_atoms, term):
    """Raise unless positions holds exactly n_atoms atoms.

    Used where the atom count is known statically, so the check costs nothing
    and survives jit.

    Args:
        positions: Atom coordinates.
        n_atoms: Atom count the parameters describe.
        term: Name of the calling energy term, used in the message.

    Raises:
        ValueError: On a shape or count mismatch.
    """
    _check_positions_shape(positions, term)
    if positions.shape[-2] != n_atoms:
        raise ValueError(
            f"{term}: positions has {positions.shape[-2]} atoms but parameters "
            f"describe {n_atoms}. Check that positions matches the system."
        )


def _check_atom_indices(positions, arrays, term):
    """Raise if any atom index falls outside positions.

    JAX clamps an out-of-bounds gather to the last valid element instead of
    raising, so a parameter set built for a larger system silently yields a
    finite, wrong energy. Index values are only readable when they are
    concrete, so this runs eagerly and is skipped under jit and vmap, where the
    static shape checks are all that remain.

    Args:
        positions: Atom coordinates, shape (..., n_atoms, 3).
        arrays: Sequence of (name, index_array) pairs to bounds check.
        term: Name of the calling energy term, used in the message.

    Raises:
        ValueError: If any index is negative or >= n_atoms.
    """
    n_atoms = positions.shape[-2]
    for name, array in arrays:
        if isinstance(array, jax.core.Tracer) or array.size == 0:
            continue
        values = np.asarray(array)
        highest, lowest = int(values.max()), int(values.min())
        if highest >= n_atoms or lowest < 0:
            offender = highest if highest >= n_atoms else lowest
            raise ValueError(
                f"{term}: {name} references atom {offender} but positions has "
                f"{n_atoms} atoms (valid indices 0 to {n_atoms - 1}). JAX would "
                "clamp this to a valid index and return a wrong energy."
            )


def _check_positions(positions, params):
    """Raise if positions shape does not match params.n_atoms."""
    _check_positions_shape(positions, "energy")
    if positions.shape[-2] != params.n_atoms:
        raise ValueError(
            f"positions has {positions.shape[-2]} atoms but params expects "
            f"{params.n_atoms}. Check that positions matches the system."
        )


def _check_log_density(values, batch_shape, name="log_density"):
    """Validate callback shape/type and concrete arithmetic errors in pullbacks."""
    values = jnp.asarray(values)
    if values.shape != batch_shape:
        raise ValueError(f"{name} must have shape {batch_shape}, got {values.shape}.")
    if not (jnp.issubdtype(values.dtype, jnp.floating)
            or jnp.issubdtype(values.dtype, jnp.integer)):
        raise TypeError(f"{name} must be real valued, got {values.dtype}.")
    valid = jnp.all(~(jnp.isnan(values) | jnp.isposinf(values)))
    if not isinstance(valid, jax.core.Tracer) and not bool(valid):
        raise ValueError(f"{name} must not contain NaN or +inf.")
    return values
