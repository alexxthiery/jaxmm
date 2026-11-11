"""Packed internal coordinates and the Cartesian conformational measure.

Lengths are nm and angles are radians. Rigid translation and orientation are
integrated out in ``log_volume``; the omitted orientation volume is 8*pi**2.
The fixed-frame Cartesian slice has a different, explicitly named Jacobian.
"""

from dataclasses import dataclass

import jax
import jax.numpy as jnp

from jaxmm.coordinates import (
    ZMatrix, cartesian_to_zmatrix, validate_zmatrix,
    zmatrix_log_abs_det_jacobian, zmatrix_to_cartesian,
)
from jaxmm.extract import _register_pytree
from jaxmm._validation import require_float64
from jaxmm.batching import map_events


def _array(value, shape, name):
    """Check event metadata even when values are traced."""
    require_float64()
    value = jnp.asarray(value)
    if value.shape[-len(shape):] != shape:
        raise ValueError(f"{name}: expected trailing shape {shape}, got {value.shape}.")
    if not jnp.issubdtype(value.dtype, jnp.floating):
        raise TypeError(f"{name}: expected real floating coordinates, got {value.dtype}.")
    require_float64(value)
    return value


_SUPPORT_ERROR = (
    "Expected finite bonds > 0, angles in (0, pi), torsions in [-pi, pi), "
    "and noncollinear reference triples."
)


def _reject_concrete_invalid(valid, message=_SUPPORT_ERROR):
    """A captured concrete input can still produce a traced predicate."""
    valid = jnp.all(valid)
    if not isinstance(valid, jax.core.Tracer) and not bool(valid):
        raise ValueError(message)


@dataclass(frozen=True)
class InternalCoordinates:
    """A Z-matrix with a packed ``[bonds, angles, torsions]`` interface.

    Args:
        z_matrix: Valid construction references and optional atom permutation.

    Shapes and dtypes are checked under tracing. Concrete geometry conversions
    also reject nonfinite or out-of-support coordinates. Density evaluation
    uses ``in_support`` instead, returning zero probability outside support.
    """

    z_matrix: ZMatrix

    def __post_init__(self):
        require_float64()
        if type(self.z_matrix) is object:
            return
        if not isinstance(self.z_matrix, ZMatrix):
            raise TypeError("z_matrix must be a ZMatrix.")
        refs = (self.z_matrix.bond_ref, self.z_matrix.angle_ref,
                self.z_matrix.torsion_ref)
        if any(type(ref) is object for ref in refs):
            return
        if any(not hasattr(ref, "shape") or not hasattr(ref, "dtype") for ref in refs):
            raise TypeError("ZMatrix references must be integer arrays; use jnp.asarray(..., dtype=jnp.int32).")
        if jnp.ndim(refs[0]) != 1 or refs[0].shape[0] < 3:
            raise ValueError("bond_ref: expected a vector describing at least three atoms.")
        n = self.z_matrix.n_atoms
        for name, ref in zip(("bond_ref", "angle_ref", "torsion_ref"), refs):
            if ref.shape != (n,) or not jnp.issubdtype(ref.dtype, jnp.integer):
                raise ValueError(f"{name}: expected integer shape {(n,)}, got {ref.shape}, {ref.dtype}.")
        order = self.z_matrix.atom_order
        if order is not None and type(order) is not object:
            if not hasattr(order, "shape") or not hasattr(order, "dtype"):
                raise TypeError("atom_order must be an integer array.")
            if order.shape != (n,) or not jnp.issubdtype(order.dtype, jnp.integer):
                raise ValueError(f"atom_order: expected integer shape {(n,)}, got {order.shape}, {order.dtype}.")
        if not any(isinstance(v, jax.core.Tracer)
                   for v in jax.tree_util.tree_leaves(self.z_matrix)):
            validate_zmatrix(self.z_matrix)

    @property
    def n_atoms(self):
        """Number of atoms."""
        return self.z_matrix.n_atoms

    @property
    def dim(self):
        """Number of internal coordinates, 3*N-6."""
        return 3 * self.n_atoms - 6

    @property
    def bond_slice(self):
        return slice(0, self.n_atoms - 1)

    @property
    def angle_slice(self):
        return slice(self.n_atoms - 1, 2 * self.n_atoms - 3)

    @property
    def torsion_slice(self):
        return slice(2 * self.n_atoms - 3, self.dim)

    @property
    def periodic_indices(self):
        """Packed torsion indices; period 2*pi, encoded in [-pi, pi)."""
        return tuple(range(self.torsion_slice.start, self.dim))

    @property
    def bounds(self):
        """Lower/upper arrays; bonds and angles open, torsions [-pi, pi)."""
        n = self.n_atoms
        return (jnp.concatenate((jnp.zeros(2*n-3), jnp.full(n-3, -jnp.pi))),
                jnp.concatenate((jnp.full(n-1, jnp.inf),
                                 jnp.full(n-2, jnp.pi), jnp.full(n-3, jnp.pi))))

    def pack(self, bonds, angles, torsions):
        """Pack blocks with identical batch shapes into ``(..., dim)``."""
        n = self.n_atoms
        blocks = [_array(v, (size,), name) for v, size, name in
                  ((bonds, n-1, "bonds"), (angles, n-2, "angles"),
                   (torsions, n-3, "torsions"))]
        if any(v.shape[:-1] != blocks[0].shape[:-1] for v in blocks):
            raise ValueError("pack: expected identical batch shapes; broadcasting is not supported.")
        return jnp.concatenate(blocks, axis=-1)

    def unpack(self, q):
        """Return bond, angle and torsion blocks from ``(..., dim)``."""
        q = _array(q, (self.dim,), "internal coordinates")
        return q[..., self.bond_slice], q[..., self.angle_slice], q[..., self.torsion_slice]

    def _in_bounds(self, q):
        b, a, t = self.unpack(q)
        return (jnp.all(jnp.isfinite(q), axis=-1)
                & jnp.all(b > 0, axis=-1)
                & jnp.all((a > 0) & (a < jnp.pi), axis=-1)
                & jnp.all((t >= -jnp.pi) & (t < jnp.pi), axis=-1))

    def in_support(self, q):
        """Boolean ``(...)``: finite bounds and regular construction geometry."""
        q = _array(q, (self.dim,), "internal coordinates")
        bounds_ok = self._in_bounds(q)
        safe = jnp.where(bounds_ok[..., None], q, self._fallback())
        return bounds_ok & self._regular_geometry(self._reconstruct(safe))

    def _fallback(self):
        n = self.n_atoms
        return self.pack(jnp.ones(n-1), jnp.ones(n-2), jnp.linspace(0.2, 2., n-3))

    def _reconstruct(self, q):
        return map_events(lambda v: zmatrix_to_cartesian(
            self.z_matrix, *self.unpack(v)), q, 1)

    def _regular_geometry(self, x):
        """Exclude coincident bonds and collinear construction reference triples."""
        z = self.z_matrix
        ordered = x[..., z.construction_order, :]
        bonds = ordered[..., 1:, :] - ordered[..., z.bond_ref[1:], :]
        u = ordered[..., 0, :] - ordered[..., 1, :]
        v = ordered[..., 2, :] - ordered[..., 1, :]
        root_ok = jnp.sum(jnp.cross(u, v)**2, axis=-1) > 0
        u = ordered[..., z.torsion_ref[3:], :] - ordered[..., z.angle_ref[3:], :]
        v = ordered[..., z.bond_ref[3:], :] - ordered[..., z.angle_ref[3:], :]
        refs_ok = jnp.all(jnp.sum(jnp.cross(u, v)**2, axis=-1) > 0, axis=-1)
        return (root_ok & refs_ok & jnp.all(jnp.sum(bonds**2, axis=-1) > 0, axis=-1)
                & jnp.all(jnp.isfinite(x), axis=(-2, -1)))

    def validate(self, q):
        """Raise on concrete invalid coordinates; setup only, not a JIT callback."""
        valid = self.in_support(q)
        if isinstance(valid, jax.core.Tracer):
            raise TypeError("validate requires concrete coordinates; use in_support inside jit.")
        _reject_concrete_invalid(valid)

    def to_cartesian(self, q):
        """Reconstruct ``(..., N, 3)`` in the canonical frame and original atom order."""
        q = _array(q, (self.dim,), "internal coordinates")
        positions = self._reconstruct(q)
        _reject_concrete_invalid(self._in_bounds(q) & self._regular_geometry(positions))
        return positions

    def to_internal(self, x):
        """Convert Cartesian geometry, discarding its rigid pose.

        Args:
            x: Float64 positions in nm, shape (..., N, 3).

        Returns:
            Packed bonds/angles/torsions, shape (..., dim), in nm/radians.

        Raises:
            ValueError: On malformed or concrete nonfinite/degenerate geometry.
            TypeError: On coordinates other than float64.
        """
        x = _array(x, (self.n_atoms, 3), "positions")
        _reject_concrete_invalid(jnp.isfinite(x), "positions must be finite.")
        _reject_concrete_invalid(self._regular_geometry(x),
                                 "Expected finite bonds > 0 and noncollinear reference triples in positions.")
        def convert(v):
            b, a, t = cartesian_to_zmatrix(self.z_matrix, v)
            return self.pack(b, a, (t + jnp.pi) % (2*jnp.pi) - jnp.pi)
        q = map_events(convert, x, 2)
        # Input geometry was checked above; rebuilding it would repeat the
        # complete Z-matrix construction solely to recheck the same references.
        _reject_concrete_invalid(self._in_bounds(q))
        return q

    def log_volume(self, q):
        """Physical conformational log volume, shape ``(...)``; -inf off support.

        This integrates rigid rotations with normalized Haar measure. The
        constant 8*pi**2 and the external translation volume are omitted.

        Args:
            q: Packed float64 coordinates, shape (..., dim), in nm/radians.

        Returns:
            Log volume of shape (...), or -inf outside the chart.
        """
        q = _array(q, (self.dim,), "internal coordinates")
        valid = self.in_support(q)
        safe = jnp.where(valid[..., None], q, self._fallback())
        return jnp.where(valid, self._log_volume(safe), -jnp.inf)

    def _log_volume(self, q, physical=True):
        """Evaluate the chosen volume on already validated internal events."""
        if physical:
            b, a, _ = self.unpack(q)
            return (2*jnp.sum(jnp.log(b), axis=-1)
                    + jnp.sum(jnp.log(jnp.sin(a)), axis=-1))
        return map_events(lambda v: zmatrix_log_abs_det_jacobian(
            self.z_matrix, *self.unpack(v)[:2]), q, 1)

    def fixed_frame_log_det(self, q):
        """Log determinant into the 3*N-6 free Cartesian slice entries."""
        return self._log_volume(_array(q, (self.dim,), "internal coordinates"), False)

    def pullback(self, log_density, q, *, reference=None, measure="physical"):
        """Express a Cartesian log density in packed internal coordinates.

        The callback must represent a rotation/translation-invariant target.
        This operation owns chart rejection, safe reconstruction and the volume
        factor, so callers need not repeat those operations for each molecule.
        The callback remains responsible for safely evaluating its own support
        restrictions, including finite gradients where it returns -inf.

        Args:
            log_density: Callable taking (..., N, 3) positions and returning
                exactly the leading batch shape. It may return -inf to reject
                additional events (for example, a chirality sector).
            q: Float64 internal events, shape (..., dim), in nm/radians.
            reference: Optional valid fallback event, shape (dim,). Supply a
                molecular reference if the generic geometric fallback has a
                singular energy for your model.
            measure: 'physical' (rigid motions marginalized) or 'fixed_frame'
                (flat Cartesian slice). A static choice, never inferred.

        Returns:
            Log density of shape (...), with -inf and zero rejection gradients
            for events outside either the chart or the callback's support.

        Raises:
            TypeError: If the callback is not callable.
            ValueError: On incorrect shapes, measure, reference, or a concrete
                callback result containing NaN or +inf. Arithmetic errors under
                JIT remain visible; they are never silently rejected as support.
        """
        if not callable(log_density):
            raise TypeError("log_density must be callable.")
        if not isinstance(measure, str) or measure not in ("physical", "fixed_frame"):
            raise ValueError(f"Expected measure 'physical' or 'fixed_frame', got {measure!r}.")
        q = _array(q, (self.dim,), "internal coordinates")
        if reference is None:
            reference = self._fallback()
        else:
            reference = _array(reference, (self.dim,), "reference")
            if reference.shape != (self.dim,):
                raise ValueError(f"reference must have exactly shape {(self.dim,)}, got {reference.shape}.")
            _reject_concrete_invalid(self.in_support(reference), "reference must be a valid internal event.")
        valid = self.in_support(q)
        safe = jnp.where(valid[..., None], q, reference)
        values = jnp.asarray(log_density(self._reconstruct(safe)))
        if values.shape != q.shape[:-1]:
            raise ValueError(f"Cartesian log density must have shape {q.shape[:-1]}, got {values.shape}.")
        if not (jnp.issubdtype(values.dtype, jnp.floating)
                or jnp.issubdtype(values.dtype, jnp.integer)):
            raise TypeError(f"Cartesian log density must be real valued, got {values.dtype}.")
        _reject_concrete_invalid(~(jnp.isnan(values) | jnp.isposinf(values)),
                                 "Cartesian log density must not contain NaN or +inf.")
        # The Cartesian target may impose further support. Its rejection must
        # also suppress the volume derivative, rather than differentiating
        # (-inf + log_volume) and returning a spurious nonzero gradient.
        valid = valid & ~jnp.isneginf(values)
        result = values + self._log_volume(safe, measure == "physical")
        return jnp.where(valid, result, -jnp.inf)


_register_pytree(InternalCoordinates)
