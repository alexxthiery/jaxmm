"""Map single-event functions over arbitrary leading batch axes."""

import math
from numbers import Integral

import jax


def map_events(function, values, event_ndim):
    """Apply a single-event function while preserving leading batch axes.

    Args:
        function: Callable taking one event. It may return an array or pytree.
        values: Array with shape ``batch_shape + event_shape``.
        event_ndim: Positive static integer, the number of trailing event axes.

    Returns:
        Function outputs with ``batch_shape`` prepended to each output's shape.
        Multiple and empty batch axes are supported. No event is flattened.

    Raises:
        TypeError: If function is not callable or event_ndim is not an integer.
        ValueError: If event_ndim is not in [1, values.ndim].
    """
    if not callable(function):
        raise TypeError("function must be callable.")
    if not all(hasattr(values, name) for name in ("shape", "ndim", "reshape")):
        raise TypeError("values must be an array with shape and ndim metadata.")
    if isinstance(event_ndim, bool) or not isinstance(event_ndim, Integral):
        raise TypeError("event_ndim must be a static integer.")
    if not 1 <= event_ndim <= values.ndim:
        raise ValueError(f"Expected event_ndim in [1, {values.ndim}], got {event_ndim}.")
    batch = values.shape[:-event_ndim]
    events = values.reshape((math.prod(batch),) + values.shape[-event_ndim:])
    result = jax.vmap(function)(events)
    return jax.tree_util.tree_map(lambda v: v.reshape(batch + v.shape[1:]), result)
