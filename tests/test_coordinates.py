"""Tests for fixed-frame internal-coordinate transforms."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import jaxmm
from jaxmm.energy import total_energy


def _angle_diff(a, b):
    """Periodic difference between angles."""

    return jnp.arctan2(jnp.sin(a - b), jnp.cos(a - b))


def test_toy_zmatrix_fixed_frame_positions():
    """A four-atom chain is placed in the expected fixed frame."""

    z_matrix = jaxmm.ZMatrix(
        bond_ref=jnp.array([-1, 0, 1, 2], dtype=jnp.int32),
        angle_ref=jnp.array([-1, -1, 0, 1], dtype=jnp.int32),
        torsion_ref=jnp.array([-1, -1, -1, 0], dtype=jnp.int32),
    )
    bonds = jnp.array([1.0, 1.0, 1.0])
    angles = jnp.array([jnp.pi / 2.0, jnp.pi / 2.0])
    torsions = jnp.array([0.0])

    positions = jaxmm.zmatrix_to_cartesian(z_matrix, bonds, angles, torsions)

    np.testing.assert_allclose(positions[0], [0.0, 0.0, 0.0], atol=1e-12)
    np.testing.assert_allclose(positions[1], [1.0, 0.0, 0.0], atol=1e-12)
    assert abs(float(positions[2, 2])) < 1e-12
    np.testing.assert_allclose(
        jnp.linalg.norm(positions[1:] - positions[:-1], axis=-1),
        bonds,
        atol=1e-12,
    )


def test_zmatrix_roundtrip_internal_coordinates():
    """IC -> Cartesian -> IC round trip preserves nondegenerate coordinates."""

    z_matrix = jaxmm.ZMatrix(
        bond_ref=jnp.array([-1, 0, 1, 2, 3], dtype=jnp.int32),
        angle_ref=jnp.array([-1, -1, 0, 1, 2], dtype=jnp.int32),
        torsion_ref=jnp.array([-1, -1, -1, 0, 1], dtype=jnp.int32),
    )
    bonds = jnp.array([0.12, 0.15, 0.14, 0.16])
    angles = jnp.array([1.1, 2.0, 1.4])
    torsions = jnp.array([-0.8, 1.7])

    positions = jaxmm.zmatrix_to_cartesian(z_matrix, bonds, angles, torsions)
    got_bonds, got_angles, got_torsions = jaxmm.cartesian_to_zmatrix(z_matrix, positions)

    np.testing.assert_allclose(got_bonds, bonds, rtol=1e-10, atol=1e-12)
    np.testing.assert_allclose(got_angles, angles, rtol=1e-10, atol=1e-12)
    np.testing.assert_allclose(_angle_diff(got_torsions, torsions), 0.0, atol=1e-10)


def _random_rigid_motion(seed):
    """A proper rotation and a translation."""

    rng = np.random.default_rng(seed)
    rotation, _ = np.linalg.qr(rng.normal(size=(3, 3)))
    if np.linalg.det(rotation) < 0:
        rotation[:, 0] *= -1.0
    return jnp.array(rotation), jnp.array(rng.normal(size=3) * 0.5)


def _distance_matrix(positions):
    """All pairwise distances, invariant under any rigid motion."""

    return jnp.linalg.norm(positions[:, None] - positions[None, :], axis=-1)


def test_canonicalize_is_invariant_to_rigid_motion(aldp_positions_jnp):
    """Rotating and translating the input does not change the canonical form.

    This is what canonicalize_cartesian claims to do. Idempotence alone is a
    much weaker property that a wrong frame construction would also satisfy,
    so it is asserted here as a secondary check rather than on its own.
    """

    z_matrix = jaxmm.aldp_zmatrix()
    rotation, translation = _random_rigid_motion(seed=20260923)

    canonical = jaxmm.canonicalize_cartesian(z_matrix, aldp_positions_jnp)
    moved = jaxmm.canonicalize_cartesian(
        z_matrix, aldp_positions_jnp @ rotation.T + translation
    )

    np.testing.assert_allclose(moved, canonical, rtol=1e-10, atol=1e-10)
    np.testing.assert_allclose(
        jaxmm.canonicalize_cartesian(z_matrix, canonical), canonical,
        rtol=1e-10, atol=1e-10,
    )


def test_canonicalize_preserves_distance_matrix(aldp_positions_jnp):
    """Canonicalization removes gauge, not shape."""

    z_matrix = jaxmm.aldp_zmatrix()
    canonical = jaxmm.canonicalize_cartesian(z_matrix, aldp_positions_jnp)

    np.testing.assert_allclose(
        _distance_matrix(canonical), _distance_matrix(aldp_positions_jnp),
        rtol=1e-10, atol=1e-12,
    )


def test_canonicalize_preserves_chirality(aldp_positions_jnp):
    """A mirror image is a different molecule and must not collapse onto the original.

    Reflection preserves every pairwise distance, so the distance-matrix check
    above cannot see it. Signed torsions can, and must.
    """

    z_matrix = jaxmm.aldp_zmatrix()
    mirrored = aldp_positions_jnp * jnp.array([1.0, 1.0, -1.0])

    # Control: the reflection really is distance preserving.
    np.testing.assert_allclose(
        _distance_matrix(mirrored), _distance_matrix(aldp_positions_jnp),
        rtol=1e-10, atol=1e-12,
    )

    canonical = jaxmm.canonicalize_cartesian(z_matrix, aldp_positions_jnp)
    canonical_mirror = jaxmm.canonicalize_cartesian(z_matrix, mirrored)
    assert float(jnp.max(jnp.abs(canonical - canonical_mirror))) > 1e-3

    # The torsions are what carry the handedness.
    torsions = jaxmm.cartesian_to_zmatrix(z_matrix, aldp_positions_jnp)[2]
    torsions_mirror = jaxmm.cartesian_to_zmatrix(z_matrix, mirrored)[2]
    np.testing.assert_allclose(torsions_mirror, -torsions, rtol=1e-10, atol=1e-10)


def test_zmatrix_transform_jit_vmap_and_grad():
    """Coordinate transforms work with JIT, vmap, and autodiff."""

    z_matrix = jaxmm.ZMatrix(
        bond_ref=jnp.array([-1, 0, 1, 2], dtype=jnp.int32),
        angle_ref=jnp.array([-1, -1, 0, 1], dtype=jnp.int32),
        torsion_ref=jnp.array([-1, -1, -1, 0], dtype=jnp.int32),
    )
    bonds = jnp.array([0.11, 0.14, 0.13])
    angles = jnp.array([1.2, 1.7])
    torsions = jnp.array([0.4])

    @jax.jit
    def loss(b):
        pos = jaxmm.zmatrix_to_cartesian(z_matrix, b, angles, torsions)
        return jnp.sum(pos * pos)

    value = loss(bonds)
    grad = jax.grad(loss)(bonds)
    assert jnp.isfinite(value)
    assert jnp.all(jnp.isfinite(grad))

    batched_bonds = jnp.stack([bonds, bonds + 0.01])
    batched = jax.vmap(lambda b: jaxmm.zmatrix_to_cartesian(z_matrix, b, angles, torsions))(
        batched_bonds
    )
    assert batched.shape == (2, 4, 3)


def _reduced_cartesian(z_matrix, flat, n_atoms):
    """The 3 * n_atoms - 6 free Cartesian coordinates in the fixed frame.

    Atom 0 is pinned at the origin, atom 1 to the x-axis, and atom 2 to the
    xy-plane, so those are the coordinates the transform actually controls.
    """

    bonds = flat[: n_atoms - 1]
    angles = flat[n_atoms - 1 : 2 * n_atoms - 3]
    torsions = flat[2 * n_atoms - 3 :]
    positions = jaxmm.zmatrix_to_cartesian(z_matrix, bonds, angles, torsions)
    return jnp.concatenate([positions[1, :1], positions[2, :2], positions[3:].ravel()])


def _assert_log_jacobian_matches_autodiff(z_matrix, bonds, angles, torsions):
    """Compare the analytic volume element against jacfwd + slogdet."""

    n_atoms = z_matrix.n_atoms
    flat = jnp.concatenate([bonds, angles, torsions])
    jacobian = jax.jacfwd(lambda y: _reduced_cartesian(z_matrix, y, n_atoms))(flat)
    assert jacobian.shape == (3 * n_atoms - 6, 3 * n_atoms - 6)

    autodiff_logdet = jnp.linalg.slogdet(jacobian)[1]
    analytic = jaxmm.zmatrix_log_abs_det_jacobian(z_matrix, bonds, angles)
    np.testing.assert_allclose(analytic, autodiff_logdet, rtol=1e-9, atol=1e-9)


@pytest.mark.parametrize("n_atoms", [3, 4, 5, 6, 9])
def test_log_jacobian_matches_autodiff_linear_chain(n_atoms):
    """The analytic volume element holds at every size, not just four atoms."""

    rng = np.random.default_rng(100 + n_atoms)
    z_matrix = _chain_zmatrix(n_atoms)
    jaxmm.validate_zmatrix(z_matrix)
    bonds, angles, torsions = _safe_internals(rng, n_atoms)
    _assert_log_jacobian_matches_autodiff(z_matrix, bonds, angles, torsions)


def test_log_jacobian_matches_autodiff_branched_topology():
    """The volume element does not depend on the z-matrix being a chain."""

    z_matrix = jaxmm.ZMatrix(
        bond_ref=jnp.array([-1, 0, 1, 1, 2, 2, 3], dtype=jnp.int32),
        angle_ref=jnp.array([-1, -1, 0, 0, 1, 1, 1], dtype=jnp.int32),
        torsion_ref=jnp.array([-1, -1, -1, 2, 0, 0, 2], dtype=jnp.int32),
    )
    jaxmm.validate_zmatrix(z_matrix)
    rng = np.random.default_rng(7)
    bonds, angles, torsions = _safe_internals(rng, 7)
    _assert_log_jacobian_matches_autodiff(z_matrix, bonds, angles, torsions)


def test_log_jacobian_matches_autodiff_aldp(aldp_positions_jnp):
    """The volume element holds on the real 22-atom system at a real geometry."""

    z_matrix = jaxmm.aldp_zmatrix()
    bonds, angles, torsions = jaxmm.cartesian_to_zmatrix(z_matrix, aldp_positions_jnp)
    _assert_log_jacobian_matches_autodiff(z_matrix, bonds, angles, torsions)


def test_aldp_zmatrix_openmmtools_metadata():
    """ALDP helper has the expected openmmtools dimensions and references."""

    z_matrix = jaxmm.aldp_zmatrix()
    jaxmm.validate_zmatrix(z_matrix)
    assert z_matrix.n_atoms == 22
    assert z_matrix.bond_ref.shape == (22,)
    assert z_matrix.angle_ref.shape == (22,)
    assert z_matrix.torsion_ref.shape == (22,)
    assert int(z_matrix.bond_ref[0]) == -1
    assert jnp.all(z_matrix.bond_ref[1:] < jnp.arange(1, 22))


def test_validate_zmatrix_rejects_bad_metadata():
    """Host-side validator catches malformed z-matrix references."""

    with pytest.raises(ValueError, match="matching shape"):
        jaxmm.validate_zmatrix(
            jaxmm.ZMatrix(
                bond_ref=jnp.array([-1, 0, 1]),
                angle_ref=jnp.array([-1, -1]),
                torsion_ref=jnp.array([-1, -1, -1]),
            )
        )
    with pytest.raises(ValueError, match="bond references"):
        jaxmm.validate_zmatrix(
            jaxmm.ZMatrix(
                bond_ref=jnp.array([-1, 0, 2]),
                angle_ref=jnp.array([-1, -1, 0]),
                torsion_ref=jnp.array([-1, -1, -1]),
            )
        )


def test_aldp_zmatrix_reconstructs_finite_energy(aldp_implicit_positions, aldp_implicit_params):
    """ALDP reference geometry remains finite after IC canonicalization."""

    z_matrix = jaxmm.aldp_zmatrix()
    positions = jnp.asarray(aldp_implicit_positions)
    bonds, angles, torsions = jaxmm.cartesian_to_zmatrix(z_matrix, positions)
    reconstructed = jaxmm.zmatrix_to_cartesian(z_matrix, bonds, angles, torsions)

    energy_original = total_energy(positions, aldp_implicit_params)
    energy_reconstructed = total_energy(reconstructed, aldp_implicit_params)
    assert jnp.isfinite(energy_reconstructed)
    np.testing.assert_allclose(energy_reconstructed, energy_original, rtol=1e-7, atol=1e-5)


# ---------------------------------------------------------------------------
# Helpers for z-matrix construction
# ---------------------------------------------------------------------------

def _chain_zmatrix(n_atoms):
    """Linear-chain z-matrix: atom i references i-1, i-2, i-3."""
    return jaxmm.ZMatrix(
        bond_ref=jnp.array([-1] + list(range(n_atoms - 1)), dtype=jnp.int32),
        angle_ref=jnp.array([-1, -1] + list(range(n_atoms - 2)), dtype=jnp.int32),
        torsion_ref=jnp.array([-1, -1, -1] + list(range(n_atoms - 3)), dtype=jnp.int32),
    )


def _random_valid_zmatrix(rng, n_atoms):
    """Random z-matrix obeying the fixed-frame contract, with distinct references."""
    bond_ref = [-1, 0, 1]
    angle_ref = [-1, -1, 0]
    torsion_ref = [-1, -1, -1]
    for i in range(3, n_atoms):
        j = int(rng.integers(0, i))
        k = int(rng.choice([a for a in range(i) if a != j]))
        l = int(rng.choice([a for a in range(i) if a not in (j, k)]))
        bond_ref.append(j)
        angle_ref.append(k)
        torsion_ref.append(l)
    return jaxmm.ZMatrix(
        bond_ref=jnp.array(bond_ref, dtype=jnp.int32),
        angle_ref=jnp.array(angle_ref, dtype=jnp.int32),
        torsion_ref=jnp.array(torsion_ref, dtype=jnp.int32),
    )


def _safe_internals(rng, n_atoms):
    """Internal coordinates well away from the coordinate singularity."""
    return (
        jnp.array(rng.uniform(0.10, 0.16, n_atoms - 1)),
        jnp.array(rng.uniform(0.7, 2.4, n_atoms - 2)),
        jnp.array(rng.uniform(-np.pi, np.pi, max(n_atoms - 3, 0))),
    )


# ---------------------------------------------------------------------------
# F1: the fixed frame constrains the first three atoms
# ---------------------------------------------------------------------------

def test_validate_rejects_noncanonical_bond_ref_2():
    """Atom 2 must hang off atom 1: zmatrix_to_cartesian hardcodes that frame."""
    z = jaxmm.ZMatrix(
        bond_ref=jnp.array([-1, 0, 0, 1], dtype=jnp.int32),
        angle_ref=jnp.array([-1, -1, 1, 0], dtype=jnp.int32),
        torsion_ref=jnp.array([-1, -1, -1, 2], dtype=jnp.int32),
    )
    with pytest.raises(ValueError, match=r"bond_ref\[2\]"):
        jaxmm.validate_zmatrix(z)


def test_validate_rejects_noncanonical_angle_ref_2():
    """Atom 2's angle must be measured against atom 0."""
    z = jaxmm.ZMatrix(
        bond_ref=jnp.array([-1, 0, 1, 1], dtype=jnp.int32),
        angle_ref=jnp.array([-1, -1, 1, 0], dtype=jnp.int32),
        torsion_ref=jnp.array([-1, -1, -1, 2], dtype=jnp.int32),
    )
    with pytest.raises(ValueError, match=r"angle_ref\[2\]"):
        jaxmm.validate_zmatrix(z)


def test_every_accepted_zmatrix_roundtrips():
    """Validator acceptance implies the IC -> Cartesian -> IC bijection holds.

    This is the contract F1 broke: a z-matrix with bond_ref[2] == 0 passed
    validation and silently round-tripped to different bond lengths.
    """
    rng = np.random.default_rng(20260923)
    for n_atoms in (4, 5, 7, 11):
        for z in (_chain_zmatrix(n_atoms), _random_valid_zmatrix(rng, n_atoms)):
            jaxmm.validate_zmatrix(z)
            bonds, angles, torsions = _safe_internals(rng, n_atoms)
            pos = jaxmm.zmatrix_to_cartesian(z, bonds, angles, torsions)
            got_b, got_a, got_t = jaxmm.cartesian_to_zmatrix(z, pos)
            np.testing.assert_allclose(got_b, bonds, rtol=1e-10, atol=1e-12)
            np.testing.assert_allclose(got_a, angles, rtol=1e-10, atol=1e-12)
            np.testing.assert_allclose(_angle_diff(got_t, torsions), 0.0, atol=1e-10)


# ---------------------------------------------------------------------------
# F2: reference atoms must be distinct
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "bond_ref,angle_ref,torsion_ref,match",
    [
        ([-1, 0, 1, 1], [-1, -1, 0, 1], [-1, -1, -1, 0], r"angle_ref"),
        ([-1, 0, 1, 2], [-1, -1, 0, 1], [-1, -1, -1, 2], r"torsion_ref"),
        ([-1, 0, 1, 2], [-1, -1, 0, 1], [-1, -1, -1, 1], r"torsion_ref"),
    ],
    ids=["angle_ref==bond_ref", "torsion_ref==bond_ref", "torsion_ref==angle_ref"],
)
def test_validate_rejects_degenerate_references(bond_ref, angle_ref, torsion_ref, match):
    """Repeated reference atoms do not define an angle or torsion."""
    z = jaxmm.ZMatrix(
        bond_ref=jnp.array(bond_ref, dtype=jnp.int32),
        angle_ref=jnp.array(angle_ref, dtype=jnp.int32),
        torsion_ref=jnp.array(torsion_ref, dtype=jnp.int32),
    )
    with pytest.raises(ValueError, match=match):
        jaxmm.validate_zmatrix(z)


# ---------------------------------------------------------------------------
# F3: triatomics
# ---------------------------------------------------------------------------

def test_three_atom_molecule_roundtrips():
    """A 3-atom molecule works in all four functions.

    The validator accepts n_atoms >= 3, and the inverse and Jacobian already
    handled it, but zmatrix_to_cartesian raised IndexError on the empty
    torsion array.
    """
    z = jaxmm.ZMatrix(
        bond_ref=jnp.array([-1, 0, 1], dtype=jnp.int32),
        angle_ref=jnp.array([-1, -1, 0], dtype=jnp.int32),
        torsion_ref=jnp.array([-1, -1, -1], dtype=jnp.int32),
    )
    jaxmm.validate_zmatrix(z)
    bonds = jnp.array([0.0957, 0.0957])
    angles = jnp.array([1.8242])
    torsions = jnp.zeros((0,))

    pos = jaxmm.zmatrix_to_cartesian(z, bonds, angles, torsions)
    assert pos.shape == (3, 3)
    got_b, got_a, got_t = jaxmm.cartesian_to_zmatrix(z, pos)
    np.testing.assert_allclose(got_b, bonds, rtol=1e-12)
    np.testing.assert_allclose(got_a, angles, rtol=1e-12)
    assert got_t.shape == (0,)

    # Hand-derived: for three atoms the fixed-frame volume element is r_2.
    analytic = jaxmm.zmatrix_log_abs_det_jacobian(z, bonds, angles)
    np.testing.assert_allclose(analytic, np.log(0.0957), rtol=1e-12)


# ---------------------------------------------------------------------------
# F4, F5: gradient safety off the coordinate singularity
# ---------------------------------------------------------------------------

def test_cartesian_to_zmatrix_gradients_finite_at_collinear_triple():
    """arccos returns a NaN gradient at a collinear triple; atan2 does not."""
    z = _chain_zmatrix(4)
    # atoms 0, 1, 2 exactly collinear -> the angle for atom 2 is pi
    positions = jnp.array([[0.0, 0.0, 0.0], [0.1, 0.0, 0.0],
                           [0.2, 0.0, 0.0], [0.3, 0.01, 0.0]])
    grad = jax.grad(lambda p: jnp.sum(jaxmm.cartesian_to_zmatrix(z, p)[1]))(positions)
    assert jnp.all(jnp.isfinite(grad)), f"non-finite angle gradient:\n{grad}"


def test_cartesian_to_zmatrix_gradients_finite_at_coincident_atoms():
    """jnp.linalg.norm returns a NaN gradient at zero separation; safe sqrt does not."""
    z = _chain_zmatrix(4)
    # atoms 1 and 2 coincide -> bond length 0
    positions = jnp.array([[0.0, 0.0, 0.0], [0.1, 0.0, 0.0],
                           [0.1, 0.0, 0.0], [0.2, 0.01, 0.0]])
    grad = jax.grad(lambda p: jnp.sum(jaxmm.cartesian_to_zmatrix(z, p)[0]))(positions)
    assert jnp.all(jnp.isfinite(grad)), f"non-finite bond gradient:\n{grad}"


# ---------------------------------------------------------------------------
# F6, F8: house invariants
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("call", ["forward", "inverse", "canonicalize", "logdet"])
def test_x64_required(monkeypatch, call):
    """Every public entry point refuses to run without float64."""
    z = _chain_zmatrix(4)
    bonds = jnp.array([0.12, 0.15, 0.14])
    angles = jnp.array([1.2, 1.8])
    torsions = jnp.array([0.7])
    positions = jaxmm.zmatrix_to_cartesian(z, bonds, angles, torsions)

    monkeypatch.setattr(type(jax.config), "jax_enable_x64", False)
    calls = {
        "forward": lambda: jaxmm.zmatrix_to_cartesian(z, bonds, angles, torsions),
        "inverse": lambda: jaxmm.cartesian_to_zmatrix(z, positions),
        "canonicalize": lambda: jaxmm.canonicalize_cartesian(z, positions),
        "logdet": lambda: jaxmm.zmatrix_log_abs_det_jacobian(z, bonds, angles),
    }
    with pytest.raises(RuntimeError, match="float64"):
        calls[call]()


def test_zmatrix_to_cartesian_promotes_dtype_across_inputs():
    """Output precision follows all three inputs, not just `bonds`."""
    z = _chain_zmatrix(4)
    positions = jaxmm.zmatrix_to_cartesian(
        z,
        jnp.array([0.12, 0.15, 0.14], dtype=jnp.float32),
        jnp.array([1.2, 1.8], dtype=jnp.float64),
        jnp.array([0.7], dtype=jnp.float64),
    )
    assert positions.dtype == jnp.float64, (
        f"float32 bonds silently downcast the result to {positions.dtype}"
    )


# ---------------------------------------------------------------------------
# Stronger oracles for the shipped ALDP z-matrix and the transform conventions
# ---------------------------------------------------------------------------

def test_aldp_zmatrix_references_are_real_bonds(aldp_topology):
    """Every reference in the shipped ALDP z-matrix is chemically meaningful.

    The 66 indices in aldp_zmatrix() are hand written. The OpenMM topology
    bond graph is an independent oracle for them; shape and sentinel checks
    are not.
    """

    bonded = {i: set() for i in range(aldp_topology.getNumAtoms())}
    for bond in aldp_topology.bonds():
        bonded[bond.atom1.index].add(bond.atom2.index)
        bonded[bond.atom2.index].add(bond.atom1.index)

    z_matrix = jaxmm.aldp_zmatrix()
    bond_ref = np.asarray(z_matrix.bond_ref)
    angle_ref = np.asarray(z_matrix.angle_ref)

    for i in range(1, z_matrix.n_atoms):
        assert bond_ref[i] in bonded[i], (
            f"atom {i} uses bond reference {bond_ref[i]}, which is not bonded to it"
        )
    for i in range(2, z_matrix.n_atoms):
        assert angle_ref[i] in bonded[bond_ref[i]], (
            f"atom {i} uses angle reference {angle_ref[i]}, which is not bonded "
            f"to its bond reference {bond_ref[i]}"
        )


def test_torsion_roundtrip_across_the_branch_cut():
    """Torsions recover correctly across the whole circle, including at +-pi."""

    z_matrix = _chain_zmatrix(4)
    bonds = jnp.array([0.12, 0.15, 0.14])
    angles = jnp.array([1.2, 1.8])

    worst = 0.0
    for phi in np.linspace(-np.pi, np.pi, 41):
        positions = jaxmm.zmatrix_to_cartesian(z_matrix, bonds, angles, jnp.array([phi]))
        recovered = jaxmm.cartesian_to_zmatrix(z_matrix, positions)[2][0]
        worst = max(worst, abs(float(_angle_diff(recovered, phi))))
    assert worst < 1e-10, f"max periodic torsion error {worst:.3e}"


def test_angle_convention_matches_independent_reference():
    """Switching arccos to atan2 must not change the value, only the gradient."""

    z_matrix = _chain_zmatrix(5)
    rng = np.random.default_rng(11)
    bonds, angles, torsions = _safe_internals(rng, 5)
    positions = np.asarray(jaxmm.zmatrix_to_cartesian(z_matrix, bonds, angles, torsions))

    bond_ref = np.asarray(z_matrix.bond_ref)
    angle_ref = np.asarray(z_matrix.angle_ref)
    expected = []
    for i in range(2, 5):
        u = positions[i] - positions[bond_ref[i]]
        v = positions[angle_ref[i]] - positions[bond_ref[i]]
        cosine = np.dot(u, v) / (np.linalg.norm(u) * np.linalg.norm(v))
        expected.append(np.arccos(np.clip(cosine, -1.0, 1.0)))

    got = jaxmm.cartesian_to_zmatrix(z_matrix, jnp.asarray(positions))[1]
    np.testing.assert_allclose(got, np.array(expected), rtol=1e-10, atol=1e-12)


def test_inverse_and_canonicalize_under_jit_and_vmap(aldp_positions_jnp):
    """Batched and compiled paths agree with the plain path, value for value."""

    z_matrix = jaxmm.aldp_zmatrix()
    batch = jnp.stack([aldp_positions_jnp, aldp_positions_jnp * 1.01])

    plain = jaxmm.cartesian_to_zmatrix(z_matrix, aldp_positions_jnp)
    compiled = jax.jit(lambda p: jaxmm.cartesian_to_zmatrix(z_matrix, p))(aldp_positions_jnp)
    for want, got in zip(plain, compiled):
        np.testing.assert_allclose(got, want, rtol=1e-12, atol=1e-14)

    batched = jax.vmap(lambda p: jaxmm.cartesian_to_zmatrix(z_matrix, p))(batch)
    for want, got in zip(plain, batched):
        np.testing.assert_allclose(got[0], want, rtol=1e-12, atol=1e-14)

    batched_canonical = jax.vmap(lambda p: jaxmm.canonicalize_cartesian(z_matrix, p))(batch)
    np.testing.assert_allclose(
        batched_canonical[0], jaxmm.canonicalize_cartesian(z_matrix, aldp_positions_jnp),
        rtol=1e-12, atol=1e-14,
    )


def test_log_jacobian_diverges_at_the_coordinate_singularity():
    """The volume element reports vanishing measure as a triple becomes collinear.

    This is a property of internal coordinates, not a defect. It is pinned so
    that nobody "fixes" it into a finite value later. Note that sin(pi) is
    1.2e-16 rather than 0 in float64, so the divergence is asserted as a limit
    rather than by comparing against -inf at exactly pi.
    """

    z_matrix = _chain_zmatrix(4)
    bonds = jnp.array([0.12, 0.15, 0.14])

    values = np.array([
        float(jaxmm.zmatrix_log_abs_det_jacobian(z_matrix, bonds, jnp.array([1.2, np.pi - eps])))
        for eps in (1e-2, 1e-4, 1e-6, 1e-8)
    ])
    assert all(later < earlier for earlier, later in zip(values, values[1:])), values

    # The bond terms are constant across these calls, so they cancel in the
    # differences. What is left is log sin(angle), which must fall by exactly
    # log(100) for every two decades the angle approaches pi.
    np.testing.assert_allclose(np.diff(values), -np.log(100.0), atol=1e-4)

    # An angle of exactly zero is representable, and there the measure is zero.
    exactly_collinear = jnp.array([1.2, 0.0])
    assert jnp.isneginf(
        jaxmm.zmatrix_log_abs_det_jacobian(z_matrix, bonds, exactly_collinear)
    )


# ---------------------------------------------------------------------------
# Defensive validation: size mismatches must raise, not clamp
#
# JAX clamps out-of-bounds gathers, so a 22-atom z-matrix applied to 10
# positions used to return a full-size, all-finite, entirely wrong answer.
# Every size here is known statically, so these checks cost nothing and work
# under jit.
# ---------------------------------------------------------------------------

def test_cartesian_to_zmatrix_rejects_wrong_atom_count():
    """A 22-atom z-matrix against 10 positions must raise."""
    with pytest.raises(ValueError, match="22"):
        jaxmm.cartesian_to_zmatrix(jaxmm.aldp_zmatrix(), jnp.zeros((10, 3)))


def test_canonicalize_cartesian_rejects_wrong_atom_count():
    """Same contract through the composed entry point."""
    with pytest.raises(ValueError, match="22"):
        jaxmm.canonicalize_cartesian(jaxmm.aldp_zmatrix(), jnp.zeros((10, 3)))


@pytest.mark.parametrize("bad", [(22, 2), (22,)], ids=["2d-coords", "1d"])
def test_cartesian_to_zmatrix_rejects_malformed_shape(bad):
    """positions must be (..., n_atoms, 3)."""
    with pytest.raises(ValueError):
        jaxmm.cartesian_to_zmatrix(jaxmm.aldp_zmatrix(), jnp.zeros(bad))


@pytest.mark.parametrize(
    "n_bonds,n_angles,n_torsions,match",
    [(5, 20, 19, "bonds"), (21, 3, 19, "angles"), (21, 20, 4, "torsions")],
    ids=["short-bonds", "short-angles", "short-torsions"],
)
def test_zmatrix_to_cartesian_rejects_wrong_internal_lengths(
    n_bonds, n_angles, n_torsions, match
):
    """Internal-coordinate arrays must match the z-matrix size.

    Before this check, short arrays produced a finite (22, 3) result built from
    clamped reads.
    """
    z_matrix = jaxmm.aldp_zmatrix()
    with pytest.raises(ValueError, match=match):
        jaxmm.zmatrix_to_cartesian(
            z_matrix, jnp.full(n_bonds, 0.15), jnp.full(n_angles, 1.9),
            jnp.full(n_torsions, 1.0),
        )


def test_log_jacobian_rejects_wrong_internal_lengths():
    """The volume element silently ignored its z_matrix argument; now it checks it."""
    z_matrix = jaxmm.aldp_zmatrix()
    with pytest.raises(ValueError, match="bonds"):
        jaxmm.zmatrix_log_abs_det_jacobian(z_matrix, jnp.full(5, 0.15), jnp.full(20, 1.9))


# ---------------------------------------------------------------------------
# Boltzmann density in internal coordinates
# ---------------------------------------------------------------------------

def test_log_boltzmann_internal_matches_autodiff_change_of_variables(
    aldp_positions_jnp, aldp_params
):
    """The internal-coordinate density equals the Cartesian one plus log |det J|.

    The oracle builds the Jacobian with jacfwd and slogdet, so it never touches
    the analytic volume-element formula the implementation uses.
    """
    z_matrix = jaxmm.aldp_zmatrix()
    bonds, angles, torsions = jaxmm.cartesian_to_zmatrix(z_matrix, aldp_positions_jnp)
    n_atoms = z_matrix.n_atoms

    flat = jnp.concatenate([bonds, angles, torsions])
    jacobian = jax.jacfwd(lambda y: _reduced_cartesian(z_matrix, y, n_atoms))(flat)
    expected = (
        float(jaxmm.log_boltzmann(aldp_positions_jnp, aldp_params, 300.0))
        + float(jnp.linalg.slogdet(jacobian)[1])
    )

    got = jaxmm.log_boltzmann_internal(
        z_matrix, bonds, angles, torsions, aldp_params, 300.0
    )
    np.testing.assert_allclose(float(got), expected, rtol=1e-9)


def test_log_boltzmann_internal_differs_from_cartesian_by_the_jacobian(
    aldp_positions_jnp, aldp_params
):
    """Forgetting the Jacobian is a large, silent error; pin its size.

    For alanine dipeptide the term is roughly -84 nats, about 84 kT. A user who
    calls log_boltzmann on internal-coordinate samples is wrong by that much.
    """
    z_matrix = jaxmm.aldp_zmatrix()
    bonds, angles, torsions = jaxmm.cartesian_to_zmatrix(z_matrix, aldp_positions_jnp)

    cartesian = float(jaxmm.log_boltzmann(aldp_positions_jnp, aldp_params, 300.0))
    internal = float(jaxmm.log_boltzmann_internal(
        z_matrix, bonds, angles, torsions, aldp_params, 300.0))
    log_det = float(jaxmm.zmatrix_log_abs_det_jacobian(z_matrix, bonds, angles))

    # log_boltzmann_internal evaluates the energy at the reconstructed
    # canonical frame while the baseline uses the original positions. The
    # energy is rigid-motion invariant to about 1e-10 kJ/mol in float64, which
    # sets the floor on this comparison at roughly 1e-12 relative.
    np.testing.assert_allclose(internal - cartesian, log_det, rtol=1e-9)
    assert abs(log_det) > 50.0


def test_log_boltzmann_internal_is_jittable_and_differentiable(
    aldp_positions_jnp, aldp_params
):
    """The density must be usable as a flow training target."""
    z_matrix = jaxmm.aldp_zmatrix()
    bonds, angles, torsions = jaxmm.cartesian_to_zmatrix(z_matrix, aldp_positions_jnp)

    def target(b, a, t):
        return jaxmm.log_boltzmann_internal(z_matrix, b, a, t, aldp_params, 300.0)

    np.testing.assert_allclose(
        float(jax.jit(target)(bonds, angles, torsions)),
        float(target(bonds, angles, torsions)), rtol=1e-8,
    )
    grads = jax.grad(target, argnums=(0, 1, 2))(bonds, angles, torsions)
    for g in grads:
        assert jnp.all(jnp.isfinite(g))


# ---------------------------------------------------------------------------
# Sign convention between the two public dihedral APIs
# ---------------------------------------------------------------------------

def test_zmatrix_torsions_are_negated_dihedral_angle(aldp_positions_jnp, aldp_topology):
    """The two public APIs disagree by a sign, on purpose. Pin it.

    coordinates.py follows OpenMM's PeriodicTorsionForce; utils.dihedral_angle
    negates to match mdtraj. Backbone phi and psi are z-matrix torsions 11 and
    13, so anyone plotting a Ramachandran from them will hit this.
    """
    z_matrix = jaxmm.aldp_zmatrix()
    torsions = jaxmm.cartesian_to_zmatrix(z_matrix, aldp_positions_jnp)[2]

    phi = jaxmm.dihedral_angle(aldp_positions_jnp, jaxmm.phi_indices(aldp_topology))[0]
    psi = jaxmm.dihedral_angle(aldp_positions_jnp, jaxmm.psi_indices(aldp_topology))[0]

    np.testing.assert_allclose(_angle_diff(torsions[11], -phi), 0.0, atol=1e-10)
    np.testing.assert_allclose(_angle_diff(torsions[13], -psi), 0.0, atol=1e-10)


def test_backbone_phi_psi_are_explicit_zmatrix_torsions(aldp_topology):
    """phi and psi are coordinates of the transform, not functions of them.

    This is the property that makes the shipped ALDP z-matrix suitable for a
    normalizing flow: the slow collective variables are sampled directly.
    """
    z_matrix = jaxmm.aldp_zmatrix()
    bond_ref = np.asarray(z_matrix.bond_ref)
    angle_ref = np.asarray(z_matrix.angle_ref)
    torsion_ref = np.asarray(z_matrix.torsion_ref)

    def quadruple(atom):
        return [torsion_ref[atom], angle_ref[atom], bond_ref[atom], atom]

    np.testing.assert_array_equal(quadruple(14), np.asarray(jaxmm.phi_indices(aldp_topology))[0])
    np.testing.assert_array_equal(quadruple(16), np.asarray(jaxmm.psi_indices(aldp_topology))[0])
