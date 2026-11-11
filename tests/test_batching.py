"""Event mapping preserves shapes and values, including empty batch axes."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from jaxmm import map_events


@pytest.mark.parametrize("batch", [(), (2, 3), (0,), (2, 0)])
def test_event_axes_and_pytree_outputs(batch):
    """Only batch axes are collapsed; event reductions have an algebraic oracle."""
    event = jnp.arange(6., dtype=float).reshape(2, 3)+0.4
    values = jnp.broadcast_to(event, batch+event.shape)
    def operation(x):
        return {"sum": x.sum(), "columns": x.sum(0), "positions": x*2}
    actual = jax.jit(lambda x: map_events(operation, x, 2))(values)
    np.testing.assert_allclose(actual["sum"], values.sum(axis=(-2, -1)))
    np.testing.assert_allclose(actual["columns"], values.sum(axis=-2))
    np.testing.assert_allclose(actual["positions"], values*2)


@pytest.mark.parametrize("rank,error,match", [(0, ValueError, "event_ndim in"),
                                            (3, ValueError, "event_ndim in"),
                                            (True, TypeError, "static integer"),
                                            (1.0, TypeError, "static integer")])
def test_invalid_event_rank_is_diagnostic(rank, error, match):
    with pytest.raises(error, match=match):
        map_events(lambda x: x.sum(), jnp.ones((2, 3)), rank)


def test_noncallable_is_rejected():
    with pytest.raises(TypeError, match="callable"):
        map_events(None, jnp.ones(3), 1)


def test_missing_array_metadata_is_diagnostic():
    with pytest.raises(TypeError, match="array with shape"):
        map_events(lambda x: x, [1., 2.], 1)
