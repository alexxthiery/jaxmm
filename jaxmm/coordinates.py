"""Internal-coordinate geometry utilities.

This module contains pure JAX coordinate transforms.  The transforms use a
fixed global frame: atom 0 is at the origin, atom 1 lies on the positive
x-axis, and atom 2 lies in the xy-plane.  This removes rigid translation and
rotation degrees of freedom and gives a square map between internal
coordinates and Cartesian molecular shapes.

The fixed frame constrains the first three atoms: ``bond_ref[2]`` must be 1
and ``angle_ref[2]`` must be 0, because ``zmatrix_to_cartesian`` places atom 2
relative to atom 1 at an angle measured against atom 0.  ``validate_zmatrix``
enforces this.  Every reference triple must also be made of distinct atoms,
otherwise the angle or torsion it names is undefined.

The map is singular where a bond length is zero or a reference triple is
collinear.  That is a property of internal coordinates, not a defect:
``zmatrix_log_abs_det_jacobian`` is a function of the bond lengths and
angles alone, so it diverges to ``-inf`` exactly at a zero bond length or an
angle of 0 or pi.  A collinear reference triple leaves it finite while making
the construction ill-posed, and ``zmatrix_in_domain`` is what answers whether a
configuration lies on the chart the transform inverts.  Away
from that measure-zero set all transforms return finite gradients.
"""

from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np

from jaxmm.energy import _check_positions_shape, _check_x64
from jaxmm.extract import _register_pytree


def _check_zmatrix_positions(z_matrix, positions, term):
    """Raise unless positions holds exactly the z-matrix's atom count.

    JAX clamps out-of-bounds gathers, so a 22-atom z-matrix applied to 10
    positions would otherwise return a full-size, all-finite, wrong answer.
    Both sizes are static, so this check survives jit and costs nothing.

    Args:
        z_matrix: Z-matrix metadata.
        positions: Candidate coordinates.
        term: Name of the calling function, used in the message.

    Raises:
        ValueError: On a shape or atom-count mismatch.
    """
    _check_positions_shape(positions, term)
    n_atoms = z_matrix.n_atoms
    if positions.shape[-2] != n_atoms:
        raise ValueError(
            f"{term}: positions has {positions.shape[-2]} atoms but the z-matrix "
            f"describes {n_atoms}. JAX would clamp the out-of-range reads and "
            "return a plausible, wrong result."
        )


def _check_internal_shapes(z_matrix, term, bonds=None, angles=None, torsions=None):
    """Raise unless each internal-coordinate array matches the z-matrix size.

    Args:
        z_matrix: Z-matrix metadata.
        term: Name of the calling function, used in the message.
        bonds: Bond lengths, or None to skip.
        angles: Bond angles, or None to skip.
        torsions: Torsion angles, or None to skip.

    Raises:
        ValueError: If any supplied array has the wrong trailing length.
    """
    n_atoms = z_matrix.n_atoms
    for name, array, expected in (
        ("bonds", bonds, n_atoms - 1),
        ("angles", angles, n_atoms - 2),
        ("torsions", torsions, n_atoms - 3),
    ):
        if array is None:
            continue
        if jnp.ndim(array) < 1 or array.shape[-1] != expected:
            raise ValueError(
                f"{term}: {name} has trailing length "
                f"{array.shape[-1] if jnp.ndim(array) else 'scalar'} but a "
                f"{n_atoms}-atom z-matrix needs {expected}."
            )


@dataclass(frozen=True)
class ZMatrix:
    """Reference atoms for a fixed-frame z-matrix.

    Args:
        bond_ref: Bond reference atom for each atom, shape ``(n_atoms,)``.
            Entry 0 must be ``-1``.  For atom ``i > 0``, ``bond_ref[i]`` is the
            atom bonded to ``i``.
        angle_ref: Angle reference atom for each atom, shape ``(n_atoms,)``.
            Entries 0 and 1 must be ``-1``.  For atom ``i > 1``, the angle is
            formed by ``i - bond_ref[i] - angle_ref[i]``.
        torsion_ref: Torsion reference atom for each atom, shape
            ``(n_atoms,)``.  Entries 0, 1, and 2 must be ``-1``.  For atom
            ``i > 2``, the torsion is formed by
            ``torsion_ref[i] - angle_ref[i] - bond_ref[i] - i``.
        atom_order: Optional permutation, shape ``(n_atoms,)``.
            ``atom_order[c]`` is the atom placed at construction step ``c``.
            When ``None`` (the default) construction order is atom order and
            nothing changes.

    The three reference arrays are indexed by **construction step**, and their
    values are construction steps too, so ``ref[c] < c`` always holds and the
    frame is steps 0, 1, 2.  Only the two boundaries permute:
    ``cartesian_to_zmatrix`` gathers positions into construction order on the
    way in, and ``zmatrix_to_cartesian`` scatters back to atom order on the way
    out, so a caller never handles a construction index.  This is what lets the
    frame be the rigid backbone rather than whichever atoms happen to be
    numbered first, and it is what an automatically built z-matrix needs, since
    those are not produced in atom order.
    """

    bond_ref: jax.Array
    angle_ref: jax.Array
    torsion_ref: jax.Array
    atom_order: jax.Array | None = None

    @property
    def construction_order(self) -> jax.Array:
        """Which atom each construction step places, identity when unset."""
        if self.atom_order is None:
            return jnp.arange(self.n_atoms, dtype=jnp.int32)
        return jnp.asarray(self.atom_order, dtype=jnp.int32)

    @property
    def n_atoms(self) -> int:
        """Number of atoms described by this z-matrix."""

        return int(self.bond_ref.shape[0])


_register_pytree(ZMatrix)


def validate_zmatrix(z_matrix: ZMatrix) -> None:
    """Validate z-matrix reference shapes and construction ordering.

    This is a host-side metadata check intended for setup code and tests.  The
    numerical transform functions remain JAX-traceable and do not call this
    validator internally.

    Args:
        z_matrix: Z-matrix metadata to validate.

    Raises:
        ValueError: If shapes, sentinel values, or reference ordering are
            invalid.
    """

    bond_ref = np.asarray(z_matrix.bond_ref)
    angle_ref = np.asarray(z_matrix.angle_ref)
    torsion_ref = np.asarray(z_matrix.torsion_ref)
    if bond_ref.ndim != 1:
        raise ValueError(f"bond_ref must be one-dimensional, got {bond_ref.shape}")
    if angle_ref.shape != bond_ref.shape or torsion_ref.shape != bond_ref.shape:
        raise ValueError(
            "bond_ref, angle_ref, and torsion_ref must have matching shape, "
            f"got {bond_ref.shape}, {angle_ref.shape}, and {torsion_ref.shape}"
        )
    n_atoms = bond_ref.shape[0]
    if n_atoms < 3:
        raise ValueError("z_matrix must contain at least three atoms")
    if bond_ref[0] != -1:
        raise ValueError("bond_ref[0] must be -1")
    if np.any(angle_ref[:2] != -1):
        raise ValueError("angle_ref[:2] must be -1")
    if np.any(torsion_ref[:3] != -1):
        raise ValueError("torsion_ref[:3] must be -1")
    atom_indices = np.arange(n_atoms)
    if np.any(bond_ref[1:] < 0) or np.any(bond_ref[1:] >= atom_indices[1:]):
        raise ValueError("bond references for atoms i > 0 must be in [0, i)")
    if np.any(angle_ref[2:] < 0) or np.any(angle_ref[2:] >= atom_indices[2:]):
        raise ValueError("angle references for atoms i > 1 must be in [0, i)")
    if np.any(torsion_ref[3:] < 0) or np.any(torsion_ref[3:] >= atom_indices[3:]):
        raise ValueError("torsion references for atoms i > 2 must be in [0, i)")

    # The fixed frame pins atom 2: zmatrix_to_cartesian places it relative to
    # atom 1 at an angle measured against atom 0. Any other choice would be
    # accepted here and then silently reconstructed in the wrong place.
    if bond_ref[2] != 1:
        raise ValueError(
            "bond_ref[2] must be 1: the fixed frame places atom 2 relative to "
            f"atom 1, got {bond_ref[2]}"
        )
    if angle_ref[2] != 0:
        raise ValueError(
            "angle_ref[2] must be 0: the fixed frame measures atom 2's angle "
            f"against atom 0, got {angle_ref[2]}"
        )
    if z_matrix.atom_order is not None:
        order = np.asarray(z_matrix.atom_order)
        if order.shape != (n_atoms,):
            raise ValueError(
                f"atom_order must have length {n_atoms}, got {order.shape}")
        if not np.array_equal(np.sort(order), np.arange(n_atoms)):
            raise ValueError(
                "atom_order must be a permutation of every atom exactly once, "
                f"got {order.tolist()}")

    # Repeated reference atoms leave the angle or torsion undefined.
    _reject_repeat(angle_ref, bond_ref, 2, "angle_ref", "bond_ref")
    _reject_repeat(torsion_ref, bond_ref, 3, "torsion_ref", "bond_ref")
    _reject_repeat(torsion_ref, angle_ref, 3, "torsion_ref", "angle_ref")


def _reject_repeat(first, second, start, first_name, second_name):
    """Raise if two reference arrays name the same atom at any index >= start."""

    clashes = np.nonzero(first[start:] == second[start:])[0]
    if clashes.size:
        atom = int(clashes[0]) + start
        raise ValueError(
            f"{first_name} must differ from {second_name}: atom {atom} uses "
            f"reference atom {int(first[atom])} for both"
        )


def _safe_normalize(x: jax.Array, eps: float = 1e-30) -> jax.Array:
    """Normalize vectors with a small denominator floor."""

    return x / jnp.sqrt(jnp.sum(x * x, axis=-1, keepdims=True) + eps)


def _dihedral_openmm(a: jax.Array, b: jax.Array, c: jax.Array, d: jax.Array) -> jax.Array:
    """Return the OpenMM-sign dihedral for four points."""

    b1 = b - a
    b2 = c - b
    b3 = d - c
    n1 = jnp.cross(b1, b2)
    n2 = jnp.cross(b2, b3)
    b2_hat = _safe_normalize(b2)
    m1 = jnp.cross(n1, b2_hat)
    x = jnp.sum(n1 * n2)
    y = jnp.sum(m1 * n2)
    return jnp.arctan2(y, x)


def zmatrix_to_cartesian(
    z_matrix: ZMatrix,
    bonds: jax.Array,
    angles: jax.Array,
    torsions: jax.Array,
) -> jax.Array:
    """Map fixed-frame internal coordinates to Cartesian positions.

    Args:
        z_matrix: Z-matrix references for ``n_atoms`` atoms.
        bonds: Bond lengths in nm, shape ``(n_atoms - 1,)``.
        angles: Bond angles in radians, shape ``(n_atoms - 2,)``.
        torsions: Torsion angles in radians, shape ``(n_atoms - 3,)``.

    Returns:
        Cartesian positions in nm, shape ``(n_atoms, 3)``.  Atom 0 is at the
        origin, atom 1 lies on the positive x-axis, and atom 2 lies in the
        xy-plane.
    """

    _check_x64()
    # Promote across all three inputs: keying the output dtype off `bonds`
    # alone would silently downcast float64 angles and torsions.
    dtype = jnp.result_type(bonds, angles, torsions)
    bonds = jnp.asarray(bonds, dtype=dtype)
    angles = jnp.asarray(angles, dtype=dtype)
    torsions = jnp.asarray(torsions, dtype=dtype)
    _check_internal_shapes(
        z_matrix, "zmatrix_to_cartesian", bonds=bonds, angles=angles, torsions=torsions)
    n_atoms = z_matrix.n_atoms

    positions = jnp.zeros((n_atoms, 3), dtype=dtype)
    positions = positions.at[1].set(jnp.array([bonds[0], 0.0, 0.0], dtype=dtype))
    atom2 = positions[1] + bonds[1] * jnp.array(
        [-jnp.cos(angles[0]), jnp.sin(angles[0]), 0.0],
        dtype=dtype,
    )
    positions = positions.at[2].set(atom2)

    def place_atom(atom_index, pos):
        j = z_matrix.bond_ref[atom_index]
        k = z_matrix.angle_ref[atom_index]
        l = z_matrix.torsion_ref[atom_index]
        r = bonds[atom_index - 1]
        theta = angles[atom_index - 2]
        phi = torsions[atom_index - 3]

        a = pos[l]
        b = pos[k]
        c = pos[j]
        bc = _safe_normalize(c - b)
        normal = _safe_normalize(jnp.cross(b - a, bc))
        m = jnp.cross(normal, bc)
        direction = (
            -jnp.cos(theta) * bc
            + jnp.sin(theta) * (jnp.cos(phi) * m - jnp.sin(phi) * normal)
        )
        return pos.at[atom_index].set(c + r * direction)

    # fori_loop traces its body even when the trip count is zero, and the body
    # indexes `torsions`, which is empty for a triatomic. Skip it outright.
    if n_atoms == 3:
        return positions
    built = jax.lax.fori_loop(3, n_atoms, place_atom, positions)
    # Built in construction order; hand back atom order, because every caller
    # and every force-field index is in atom order.
    if z_matrix.atom_order is None:
        return built
    return jnp.zeros_like(built).at[z_matrix.construction_order].set(built)


def cartesian_to_zmatrix(z_matrix: ZMatrix, positions: jax.Array) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Map Cartesian positions to fixed-frame internal coordinates.

    Args:
        z_matrix: Z-matrix references for ``n_atoms`` atoms.
        positions: Cartesian positions in nm, shape ``(n_atoms, 3)``.

    Returns:
        Tuple ``(bonds, angles, torsions)`` with shapes ``(n_atoms - 1,)``,
        ``(n_atoms - 2,)``, and ``(n_atoms - 3,)``.
    """

    _check_x64()
    positions = jnp.asarray(positions)
    _check_zmatrix_positions(z_matrix, positions, "cartesian_to_zmatrix")
    # The references below are construction steps, so read the positions in
    # construction order. The returned internals are indexed the same way.
    if z_matrix.atom_order is not None:
        positions = positions[z_matrix.construction_order]
    n_atoms = z_matrix.n_atoms

    def bond_for_atom(atom_index):
        j = z_matrix.bond_ref[atom_index]
        dr = positions[atom_index] - positions[j]
        # Safe norm rather than jnp.linalg.norm: norm has a NaN gradient at
        # zero separation.
        return jnp.sqrt(jnp.sum(dr**2) + 1e-30)

    def angle_for_atom(atom_index):
        j = z_matrix.bond_ref[atom_index]
        k = z_matrix.angle_ref[atom_index]
        u = positions[atom_index] - positions[j]
        v = positions[k] - positions[j]
        # atan2(|u x v|, u . v), the same form as energy.py:angle_energy.
        # arccos has an infinite derivative at 0 and pi, so a collinear triple
        # produces a NaN gradient. atan2 is scale free, so u and v need no
        # normalization.
        cross = jnp.cross(u, v)
        cross_norm = jnp.sqrt(jnp.sum(cross**2) + 1e-30)
        return jnp.arctan2(cross_norm, jnp.sum(u * v))

    def torsion_for_atom(atom_index):
        j = z_matrix.bond_ref[atom_index]
        k = z_matrix.angle_ref[atom_index]
        l = z_matrix.torsion_ref[atom_index]
        return _dihedral_openmm(positions[l], positions[k], positions[j], positions[atom_index])

    atom_indices = jnp.arange(n_atoms)
    bonds = jax.vmap(bond_for_atom)(atom_indices[1:])
    angles = jax.vmap(angle_for_atom)(atom_indices[2:])
    torsions = jax.vmap(torsion_for_atom)(atom_indices[3:])
    return bonds, angles, torsions


def canonicalize_cartesian(z_matrix: ZMatrix, positions: jax.Array) -> jax.Array:
    """Remove rigid translation and rotation from Cartesian positions.

    Args:
        z_matrix: Z-matrix references for ``n_atoms`` atoms.
        positions: Cartesian positions in nm, shape ``(n_atoms, 3)``.

    Returns:
        Fixed-frame reconstruction with shape ``(n_atoms, 3)``.
    """

    _check_x64()
    _check_zmatrix_positions(
        z_matrix, jnp.asarray(positions), "canonicalize_cartesian")
    bonds, angles, torsions = cartesian_to_zmatrix(z_matrix, positions)
    return zmatrix_to_cartesian(z_matrix, bonds, angles, torsions)


def zmatrix_log_abs_det_jacobian(
    z_matrix: ZMatrix,
    bonds: jax.Array,
    angles: jax.Array,
) -> jax.Array:
    """Compute the fixed-frame IC-to-Cartesian log absolute Jacobian.

    The fixed-frame volume element is ``|r_2|`` for atom 2 and
    ``r_i^2 |sin(theta_i)|`` for every atom ``i >= 3``.  Atom 1 contributes
    unit Jacobian because it lies on the x-axis.

    Returns ``-inf`` exactly at the zeros of the determinant, a zero bond
    length or a collinear angle, and a finite value everywhere else including
    off the chart. Use ``zmatrix_in_domain`` to ask whether a configuration is
    on the chart at all; that is a different question from the determinant.

    Args:
        z_matrix: Z-matrix references for ``n_atoms`` atoms.
        bonds: Bond lengths in nm, shape ``(n_atoms - 1,)``.
        angles: Bond angles in radians, shape ``(n_atoms - 2,)``.

    Returns:
        Scalar log absolute determinant.
    """

    _check_x64()
    bonds = jnp.asarray(bonds)
    angles = jnp.asarray(angles)
    _check_internal_shapes(
        z_matrix, "zmatrix_log_abs_det_jacobian", bonds=bonds, angles=angles)
    # Absolute values, as the function's name says. Without them a negative
    # bond length or an angle outside (0, pi) returns NaN rather than a number,
    # and NaN is far worse than a wrong value downstream: a sampler that
    # threshold-tests the result gets False from every comparison and discards
    # the configuration silently. The determinant is defined wherever it is
    # nonzero; whether the configuration is *on the chart* is a separate
    # question, answered by `zmatrix_in_domain`.
    atom2_term = jnp.log(jnp.abs(bonds[1]))
    later_terms = jnp.sum(2.0 * jnp.log(jnp.abs(bonds[2:]))
                          + jnp.log(jnp.abs(jnp.sin(angles[1:]))))
    return atom2_term + later_terms


def zmatrix_in_domain(bonds: jax.Array, angles: jax.Array) -> jax.Array:
    """Whether internals lie on the chart the transform inverts.

    The internal-to-Cartesian map is a bijection only on ``r > 0`` and
    ``theta in (0, pi)``, both open. Outside it the map is exactly 2-to-1,
    since ``x(theta, phi) == x(-theta, phi + pi)``, so a density built from
    ``zmatrix_log_abs_det_jacobian`` there would double count and no amount of
    care with the Jacobian would fix it. At the closed ends the map is not
    invertible at all and the determinant is zero.

    This is a predicate, not a validator: it returns an array rather than
    raising, because the caller is typically a sampler that must mask a few
    rows of a large batch inside ``jit`` rather than abort.

    Args:
        bonds: Bond lengths in nm, shape ``(..., n_atoms - 1)``.
        angles: Bond angles in radians, shape ``(..., n_atoms - 2)``.

    Returns:
        Boolean of shape ``(...)``, one flag per sample.
    """
    bonds = jnp.asarray(bonds)
    angles = jnp.asarray(angles)
    return (jnp.all(bonds > 0.0, axis=-1)
            & jnp.all(angles > 0.0, axis=-1)
            & jnp.all(angles < jnp.pi, axis=-1))


def aldp_zmatrix() -> ZMatrix:
    """Return the ALDP z-matrix for openmmtools atom ordering.

    Built outward from the rigid backbone: construction steps 0, 1, 2 place
    C, CA and N of the alanine residue, and ``atom_order`` carries the
    permutation, since those are atoms 14, 8 and 6.

    Two properties follow from that choice and neither is free:

    * **Chirality is a coordinate.** CA's substituents are measured against a
      reference triple lying entirely in the frame, so each is a stiff improper
      near +/-120 degrees whose sign flips under reflection.  Torsion 0 is one,
      so restricting it to half a circle is an exact fundamental domain that
      selects a single enantiomer.  Rooted anywhere that moves with phi, the
      substituents sweep together and no single torsion carries the sign; a
      half-circle restriction then cuts the Ramachandran circle instead, which
      deletes the alpha-L basin.
    * **phi and psi are sampled directly**, at torsions 12 and 5, so the slow
      collective variables are coordinates rather than sums of coordinates.

    Within each methyl one hydrogen is referenced to the backbone and the other
    two to that hydrogen, so they are pinned near +/-120 rather than sweeping
    together.  Three coordinates moving as one is a near-deterministic
    dependence, and those are what destroy an importance-sampling ESS.

    Returns:
        ``ZMatrix`` for the 22-atom alanine dipeptide systems in
        ``openmmtools.testsystems``.
    """

    # Construction step -> atom.  C, CA, N first, then out along each branch.
    atom_order = jnp.array(
        [14, 8, 6, 10, 9, 11, 12, 13, 16, 15, 18, 17, 19, 20, 21, 4, 7, 5, 1, 0, 2, 3],
        dtype=jnp.int32,
    )
    # References are construction steps, so every entry is below its own index.
    bond_ref = jnp.array(
        [-1, 0, 1, 1, 1, 3, 3, 3, 0, 0, 8, 8, 10, 10, 10, 2, 2, 15, 15, 18, 18, 18],
        dtype=jnp.int32,
    )
    angle_ref = jnp.array(
        [-1, -1, 0, 2, 2, 1, 1, 1, 1, 1, 0, 0, 8, 8, 8, 1, 1, 2, 2, 15, 15, 15],
        dtype=jnp.int32,
    )
    torsion_ref = jnp.array(
        [-1, -1, -1, 0, 0, 2, 5, 5, 2, 8, 1, 10, 0, 12, 12, 0, 15, 1, 17, 2, 19, 19],
        dtype=jnp.int32,
    )
    return ZMatrix(bond_ref=bond_ref, angle_ref=angle_ref, torsion_ref=torsion_ref,
                   atom_order=atom_order)


# Per-type whitening scales, in nm and radians. Declared rather than fitted,
# which is what both reference implementations do: FAB's `default_std` is
# {'bond': 0.005, 'angle': 0.15, 'dih': 0.2}
# (fab/target_distributions/aldp.py), and TA-BG and CMT use an effective 0.07
# and 0.5730 in their own [0, 1]-normalized units. Torsions are left alone here:
# they are already O(1) in radians and they are the coordinates that carry the
# multimodality, so rescaling them buys nothing.
BOND_SCALE = 0.005
ANGLE_SCALE = 0.15


@dataclass(frozen=True)
class InternalWhitener:
    """Affine whitening of bond lengths and bond angles.

    Bond lengths vary by about 0.005 nm around 0.1, so a flow handed raw
    internals spends its capacity on the scale difference rather than on the
    torsions. The centre is a reference structure's own internals, so that
    structure whitens to the origin, where a flow's base sits.

    Args:
        bond_center: Reference bond lengths in nm, shape ``(n_atoms - 1,)``.
        bond_scale: Bond scales in nm, same shape, all positive.
        angle_center: Reference bond angles in radians, shape ``(n_atoms - 2,)``.
        angle_scale: Angle scales in radians, same shape, all positive.

    Torsions are not whitened and do not appear here.
    """

    bond_center: jax.Array
    bond_scale: jax.Array
    angle_center: jax.Array
    angle_scale: jax.Array


_register_pytree(InternalWhitener)


def internal_whitener(
    z_matrix: ZMatrix,
    positions: jax.Array,
    bond_scale: float = BOND_SCALE,
    angle_scale: float = ANGLE_SCALE,
) -> InternalWhitener:
    """Build a whitener centred on a reference structure.

    Args:
        z_matrix: Z-matrix references.
        positions: The reference structure in nm, shape ``(n_atoms, 3)``.
            Normally an energy-minimized one, as in FAB and TA-BG.
        bond_scale: Bond scale in nm. See ``BOND_SCALE`` for the default's source.
        angle_scale: Angle scale in radians.

    Returns:
        ``InternalWhitener`` whose centre is this structure's internals.

    Raises:
        ValueError: If either scale is not positive.
    """

    _check_x64()
    for name, value in (("bond_scale", bond_scale), ("angle_scale", angle_scale)):
        if not float(value) > 0.0:
            raise ValueError(f"{name} must be positive, got {value}")
    bonds, angles, _ = cartesian_to_zmatrix(z_matrix, positions)
    # Scales are broadcast to arrays at construction, so every consumer below
    # is a plain elementwise operation and the log-Jacobian is a plain sum.
    return InternalWhitener(
        bond_center=bonds, bond_scale=jnp.full_like(bonds, bond_scale),
        angle_center=angles, angle_scale=jnp.full_like(angles, angle_scale),
    )


def whiten_internals(whitener: InternalWhitener, bonds: jax.Array,
                     angles: jax.Array) -> tuple[jax.Array, jax.Array]:
    """Map bonds and angles to whitened coordinates.

    Args:
        whitener: The whitener.
        bonds: Bond lengths in nm, shape ``(..., n_atoms - 1)``.
        angles: Bond angles in radians, shape ``(..., n_atoms - 2)``.

    Returns:
        ``(u_bonds, u_angles)``, the same shapes.
    """

    return ((bonds - whitener.bond_center) / whitener.bond_scale,
            (angles - whitener.angle_center) / whitener.angle_scale)


def unwhiten_internals(whitener: InternalWhitener, u_bonds: jax.Array,
                       u_angles: jax.Array) -> tuple[jax.Array, jax.Array]:
    """Map whitened coordinates back to bonds and angles. Inverse of ``whiten_internals``."""

    return (u_bonds * whitener.bond_scale + whitener.bond_center,
            u_angles * whitener.angle_scale + whitener.angle_center)


def whitener_log_abs_det_jacobian(whitener: InternalWhitener) -> jax.Array:
    """``log |det d(bonds, angles) / du|``, a constant.

    This is the *unwhitening* direction, because that is what a density over
    whitened coordinates needs: ``p(u) = p(eta(u)) |d eta / du|``. The map is
    diagonal with the scales on the diagonal, so the determinant is their
    product. It is a constant, which is precisely why it is easy to omit and
    why omitting it is invisible during training and fatal to any free energy.

    Returns:
        Scalar.
    """

    _check_x64()
    return (jnp.sum(jnp.log(whitener.bond_scale))
            + jnp.sum(jnp.log(whitener.angle_scale)))


def whitened_chart_bounds(whitener: InternalWhitener):
    """The whitened image of the chart, per coordinate.

    The transform inverts only on ``r > 0`` and ``theta in (0, pi)``
    (``zmatrix_in_domain``), so a flow over whitened coordinates must be
    bounded by the image of that set rather than by anything measured from
    data. A bond length has no upper bound, so its upper edge is ``+inf``.

    Returns:
        ``((bond_low, bond_high), (angle_low, angle_high))``, each the shape of
        its coordinate block.
    """

    bond_low = (0.0 - whitener.bond_center) / whitener.bond_scale
    bond_high = jnp.full_like(bond_low, jnp.inf)
    angle_low = (0.0 - whitener.angle_center) / whitener.angle_scale
    angle_high = (jnp.pi - whitener.angle_center) / whitener.angle_scale
    return (bond_low, bond_high), (angle_low, angle_high)
