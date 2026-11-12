"""Log/logit sampling coordinates with explicit unchanged circular axes."""

from dataclasses import dataclass
from numbers import Integral

import jax
import jax.numpy as jnp

from jaxmm.extract import _register_pytree
from jaxmm.internal import _array, _reject_concrete_invalid
from jaxmm._validation import require_float64, _check_log_density


@dataclass(frozen=True)
class SamplingCoordinates:
    """Map real coordinates to physical intervals, keeping circles explicit.

    Half-lines use ``q = lower + (reference-lower)*exp(rate*z)``. Finite
    intervals use a logit map. Both are centered at ``reference`` and have
    derivative ``scale`` at zero. Circular axes are unchanged, with unit
    derivative, and must stay in their supplied half-open principal interval.
    This map adds no molecular volume factor; compose it with an internal
    density that already includes that factor.

    Args:
        reference: Physical center, shape (dim,), float64.
        scale: Positive local physical scales, shape (dim,), float64.
            Circular entries must equal one because circles are unchanged.
        lower: Finite lower bounds, shape (dim,), float64.
        upper: Upper bounds greater than lower, finite or +inf, shape (dim,).
        periodic_indices: Distinct static circular-axis indices. Their bounds
            must be finite. Other intervals are open at both ends.
    """

    reference: jax.Array
    scale: jax.Array
    lower: jax.Array
    upper: jax.Array
    periodic_indices: tuple = ()

    def __post_init__(self):
        require_float64()
        if not isinstance(self.periodic_indices, tuple) or any(
                isinstance(i, bool) or not isinstance(i, Integral)
                for i in self.periodic_indices):
            raise TypeError("periodic_indices must be a tuple of integer indices.")
        if len(set(self.periodic_indices)) != len(self.periodic_indices):
            raise ValueError("periodic_indices must be distinct.")
        if any(type(v) is object for v in (self.reference, self.scale, self.lower, self.upper)):
            return
        for name in ("reference", "scale", "lower", "upper"):
            value = jnp.asarray(getattr(self, name))
            require_float64(value)
            object.__setattr__(self, name, value)
        if self.reference.ndim != 1 or self.reference.size == 0:
            raise ValueError("reference must have nonempty shape (dim,).")
        for name in ("scale", "lower", "upper"):
            if getattr(self, name).shape != self.reference.shape:
                raise ValueError(f"{name}: expected shape {self.reference.shape}, got {getattr(self, name).shape}.")
        if any(i < 0 or i >= self.dim for i in self.periodic_indices):
            raise ValueError(f"periodic_indices must be in [0, {self.dim}).")
        if not any(isinstance(v, jax.core.Tracer) for v in
                   (self.reference, self.scale, self.lower, self.upper)):
            if not bool(jnp.all(jnp.isfinite(self.scale) & (self.scale > 0))):
                raise ValueError("scale must be positive and finite.")
            if not bool(jnp.all(jnp.isfinite(self.lower) & (self.upper > self.lower))):
                raise ValueError("Expected finite lower bounds and upper > lower (upper may be +inf).")
            if not bool(jnp.all(jnp.where(self._periodic, jnp.isfinite(self.upper) & (self.scale == 1), True))):
                raise ValueError("Circular axes require finite upper bounds and unit scale.")
            if not bool(self._in_domain(self.reference)):
                raise ValueError("reference must lie inside the physical domain.")

    @property
    def dim(self):
        """Event dimension, derived from the reference."""
        return self.reference.shape[0]

    @property
    def _periodic(self):
        return jnp.zeros(self.dim, dtype=bool).at[jnp.asarray(self.periodic_indices, dtype=jnp.int32)].set(True)

    def _in_domain(self, q):
        left = jnp.where(self._periodic, q >= self.lower, q > self.lower)
        return jnp.all(jnp.isfinite(q) & left & (q < self.upper), axis=-1)

    def _coefficients(self):
        bounded = jnp.isfinite(self.upper) & ~self._periodic
        positive = ~jnp.isfinite(self.upper)
        # Every inactive branch has regular operands, including at a circle's
        # lower endpoint. Otherwise zero adjoints can still encounter NaN.
        width = jnp.where(bounded, self.upper, self.lower + 1) - self.lower
        distance = jnp.where(positive, self.reference - self.lower, 1.)
        fraction = jnp.where(bounded, (self.reference-self.lower)/width, .5)
        offset = jnp.log(fraction) - jnp.log1p(-fraction)
        rate = self.scale / jnp.where(bounded, width*fraction*(1-fraction), distance)
        return bounded, positive, width, distance, offset, rate

    def _forward(self, z):
        bounded, positive, width, distance, offset, rate = self._coefficients()
        exponent = rate*jnp.where(positive, z, 0.)
        eta = offset + rate*jnp.where(bounded, z, 0.)
        q = jnp.where(self._periodic, z, jnp.where(
            bounded, self.lower + width*jax.nn.sigmoid(eta),
            self.lower + distance*jnp.exp(exponent)))
        log_det = jnp.where(self._periodic, 0., jnp.where(
            bounded, jnp.log(width) + jnp.log(rate) - jax.nn.softplus(-eta) - jax.nn.softplus(eta),
            jnp.log(self.scale) + exponent))
        return q, jnp.sum(log_det, axis=-1)

    def to_physical(self, z):
        """Convert ``(..., dim)`` sampling events to physical coordinates.

        Concrete nonfinite, noncanonical circular or numerically saturated
        events raise. Under JIT use ``pullback`` for safe density rejection.

        Args:
            z: Float64 sampling coordinates, shape (..., dim).

        Returns:
            Physical coordinates of shape (..., dim).
        """
        z = _array(z, (self.dim,), "sampling coordinates")
        q, _ = self._forward(z)
        _reject_concrete_invalid(self._in_domain(q), "Sampling event is outside the representable physical domain.")
        return q

    def to_sampling(self, q):
        """Invert physical events; invalid concrete events raise.

        Args:
            q: Float64 physical coordinates, shape (..., dim).

        Returns:
            Sampling coordinates of shape (..., dim), with circles unchanged.
        """
        q = _array(q, (self.dim,), "physical coordinates")
        _reject_concrete_invalid(self._in_domain(q), "Physical event is outside the coordinate domain.")
        bounded, positive, width, distance, offset, rate = self._coefficients()
        fraction = jnp.where(bounded, (q-self.lower)/width, .5)
        logit = jnp.log(fraction) - jnp.log1p(-fraction)
        exponent = jnp.log(jnp.where(positive, q-self.lower, distance)/distance)
        return jnp.where(self._periodic, q, jnp.where(bounded, (logit-offset)/rate, exponent/rate))

    def log_abs_det_jacobian(self, z):
        """Log determinant in the sampling-to-physical direction.

        Args:
            z: Float64 sampling coordinates, shape (..., dim).

        Returns:
            Log absolute determinant of shape (...).
        """
        z = _array(z, (self.dim,), "sampling coordinates")
        q, log_det = self._forward(z)
        _reject_concrete_invalid(self._in_domain(q), "Sampling event is outside the representable physical domain.")
        return log_det

    def pullback(self, log_density, z):
        """Evaluate ``log_density(q(z)) + log|dq/dz|`` on ``(..., dim)``.

        The callback must return exactly the leading batch shape and safely
        reject its own support. Unrepresentable map events return -inf with
        zero gradients, including exponential overflow and logit saturation.
        Callback NaN/+inf remain arithmetic errors, not support rejections.

        Args:
            log_density: Physical-density callable taking (..., dim) and
                returning exactly the leading batch shape, with safe support.
            z: Float64 sampling coordinates, shape (..., dim).

        Returns:
            Transformed log density of shape (...).
        """
        if not callable(log_density):
            raise TypeError("log_density must be callable.")
        z = _array(z, (self.dim,), "sampling coordinates")
        fallback = self.to_sampling(self.reference)
        finite = jnp.all(jnp.isfinite(z), axis=-1)
        trial, trial_det = self._forward(jnp.where(
            finite[..., None], jax.lax.stop_gradient(z), fallback))
        valid = finite & self._in_domain(trial) & jnp.isfinite(trial_det)
        # Probe representability without differentiating overflow, then evaluate
        # only safe inputs. Masking an overflowing exp afterwards is too late.
        q, log_det = self._forward(jnp.where(valid[..., None], z, fallback))
        values = _check_log_density(log_density(q), z.shape[:-1])
        return jnp.where(valid & ~jnp.isneginf(values), values+log_det, -jnp.inf)


_register_pytree(SamplingCoordinates, aux_field_names=("periodic_indices",))
