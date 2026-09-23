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


# ---------------------------------------------------------------------------
# validate_zmatrix: every rejection branch
#
# The checks added for the fixed-frame and degenerate-reference cases were
# tested when they landed, but the original nine branches were not. A
# validator whose branches are unexercised is a validator nobody knows works.
# ---------------------------------------------------------------------------

def _zmatrix(bond_ref, angle_ref, torsion_ref):
    """Build a ZMatrix from plain lists, bypassing any validation."""
    return jaxmm.ZMatrix(
        bond_ref=jnp.array(bond_ref, dtype=jnp.int32),
        angle_ref=jnp.array(angle_ref, dtype=jnp.int32),
        torsion_ref=jnp.array(torsion_ref, dtype=jnp.int32),
    )


@pytest.mark.parametrize(
    "bond_ref,angle_ref,torsion_ref,match",
    [
        # bond_ref is not one-dimensional
        ([[-1, 0], [1, 2]], [[-1, -1], [0, 1]], [[-1, -1], [-1, 0]],
         "one-dimensional"),
        # fewer than three atoms
        ([-1, 0], [-1, -1], [-1, -1], "at least three atoms"),
        # atom 0 must have no bond reference
        ([0, 0, 1, 2], [-1, -1, 0, 1], [-1, -1, -1, 0], r"bond_ref\[0\]"),
        # atoms 0 and 1 must have no angle reference
        ([-1, 0, 1, 2], [-1, 0, 0, 1], [-1, -1, -1, 0], r"angle_ref\[:2\]"),
        # atoms 0, 1 and 2 must have no torsion reference
        ([-1, 0, 1, 2], [-1, -1, 0, 1], [-1, -1, 0, 0], r"torsion_ref\[:3\]"),
        # an angle reference at or beyond its own atom index
        ([-1, 0, 1, 2], [-1, -1, 0, 3], [-1, -1, -1, 0], "angle references"),
        # a torsion reference at or beyond its own atom index
        ([-1, 0, 1, 2], [-1, -1, 0, 1], [-1, -1, -1, 3], "torsion references"),
    ],
    ids=["bond_ref-2d", "too-few-atoms", "bond_ref[0]", "angle_ref[:2]",
         "torsion_ref[:3]", "angle-ref-out-of-range", "torsion-ref-out-of-range"],
)
def test_validate_zmatrix_rejection_branches(bond_ref, angle_ref, torsion_ref, match):
    """Each malformed z-matrix is rejected with a message naming the problem."""
    with pytest.raises(ValueError, match=match):
        jaxmm.validate_zmatrix(_zmatrix(bond_ref, angle_ref, torsion_ref))


def test_validate_zmatrix_rejects_negative_references():
    """A negative reference for an atom past the frame is out of range.

    -1 is the sentinel for the first three atoms only; later atoms must name a
    real earlier atom.
    """
    with pytest.raises(ValueError, match="bond references"):
        jaxmm.validate_zmatrix(_zmatrix([-1, 0, 1, -1], [-1, -1, 0, 1], [-1, -1, -1, 0]))


def test_validate_zmatrix_accepts_the_shipped_aldp_matrix():
    """Positive control: the validator must not reject a z-matrix that is correct."""
    jaxmm.validate_zmatrix(jaxmm.aldp_zmatrix())


# ---------------------------------------------------------------------------
# Domain of the chart.
#
# The internal-to-Cartesian map is a valid chart only on r > 0 and
# theta in (0, pi). Outside it the map is exactly 2-to-1, because
# x(theta, phi) == x(-theta, phi + pi), so a density built from the Jacobian
# there would double count. These tests pin the two halves of the contract
# separately: the determinant is a determinant everywhere it is defined, and
# the *density* is what vanishes off the chart.
# ---------------------------------------------------------------------------

def _valid_internals(seed=0):
    """Internals of a non-degenerate structure, as float64 numpy."""
    z = jaxmm.aldp_zmatrix()
    x = jax.random.normal(jax.random.PRNGKey(seed), (z.n_atoms, 3), jnp.float64) * 0.15
    b, a, t = jaxmm.cartesian_to_zmatrix(z, x)
    return z, b, a, t


def _ldj_oracle(bonds, angles):
    """log|det J| written out as a float64 loop: log|r_1| + sum 2log|r_i| + log|sin th_i|."""
    b = np.asarray(bonds, np.float64)
    a = np.asarray(angles, np.float64)
    total = np.log(np.abs(b[1]))
    for i in range(2, len(b)):
        total += 2.0 * np.log(np.abs(b[i]))
    for i in range(1, len(a)):
        total += np.log(np.abs(np.sin(a[i])))
    return total


class TestChartDomain:

    def test_the_jacobian_never_returns_nan_off_the_chart(self):
        """
        Claim: the determinant is |r|^2 |sin theta|, so it is finite for a
        negative bond length or an angle outside (0, pi).
        Bug it catches: the defect this test was written for. `log(sin theta)`
        and `log(r)` without absolute values return NaN there, and NaN is far
        worse than a wrong number downstream: in LTR it either aborts a run or,
        through `NaN >= log_tau` evaluating False and a cumulative AND, silently
        closes a source's entire reach row.
        Oracle: the absolute-value product formula as a float64 loop.
        """
        z, b, a, _ = _valid_internals()

        for name, bb, aa in (("angle > pi", b, a.at[5].set(jnp.pi + 0.01)),
                             ("angle < 0", b, a.at[5].set(-0.01)),
                             ("bond < 0", b.at[5].set(-0.11), a)):
            value = float(jaxmm.zmatrix_log_abs_det_jacobian(z, bb, aa))

            assert np.isfinite(value), f"{name} gave {value}"
            np.testing.assert_allclose(value, _ldj_oracle(bb, aa), rtol=1e-12)

    def test_the_jacobian_is_unchanged_on_the_chart(self):
        """
        Regression: taking absolute values must not move any value that was
        already correct, since every saved number depends on this.
        Oracle: the same float64 loop, on a physical structure.
        """
        z, b, a, _ = _valid_internals()

        np.testing.assert_allclose(
            float(jaxmm.zmatrix_log_abs_det_jacobian(z, b, a)), _ldj_oracle(b, a), rtol=1e-12)

    def test_the_determinant_vanishes_only_where_it_should(self):
        """
        Claim: log|det J| is -inf exactly at a zero bond length or a collinear
        angle, which are the genuine zeros of the determinant.
        Bug it catches: clamping the singularity to a finite floor, which jaxmm
        documents as forbidden because it would make a singular configuration
        look merely unlikely.
        Oracle: the definition at the two zeros.
        """
        z, b, a, _ = _valid_internals()

        assert float(jaxmm.zmatrix_log_abs_det_jacobian(z, b.at[5].set(0.0), a)) == -np.inf
        assert float(jaxmm.zmatrix_log_abs_det_jacobian(z, b, a.at[5].set(0.0))) == -np.inf

    @pytest.mark.parametrize("name, bond, angle, expected", [
        ("physical", None, None, True),
        ("bond zero", 0.0, None, False),
        ("bond negative", -0.11, None, False),
        ("angle zero", None, 0.0, False),
        ("angle pi", None, np.pi, False),
        ("angle above pi", None, np.pi + 0.01, False),
        ("angle negative", None, -0.01, False),
    ])
    def test_the_domain_predicate_is_the_open_chart(self, name, bond, angle, expected):
        """
        Claim: `zmatrix_in_domain` is exactly r > 0 and theta in (0, pi), open
        at both ends.
        Bug it catches: a closed interval, which would admit the collinear
        configurations where the map is not invertible and the determinant is 0.
        Oracle: a hand table over every boundary case.
        """
        z, b, a, _ = _valid_internals()
        if bond is not None:
            b = b.at[5].set(bond)
        if angle is not None:
            a = a.at[5].set(angle)

        assert bool(jaxmm.zmatrix_in_domain(b, a)) is expected

    def test_the_domain_predicate_is_per_sample_over_a_batch(self):
        """
        Claim: the predicate maps a batch of internals to a boolean per sample.
        Bug it catches: reducing over the batch, which would reject a whole
        refresh because one sample of thousands left the chart.
        Oracle: a batch built with known good and bad rows.
        """
        z, b, a, _ = _valid_internals()
        bonds = jnp.stack([b, b, b.at[5].set(-0.11)])
        angles = jnp.stack([a, a.at[5].set(jnp.pi + 0.01), a])

        np.testing.assert_array_equal(
            np.asarray(jaxmm.zmatrix_in_domain(bonds, angles)), [True, False, False])


# ---------------------------------------------------------------------------
# The density off the chart.
#
# `log_boltzmann_internal` is the function a flow trains against, so the chart
# restriction has to bite here: off the chart the density is zero, i.e. -inf.
# Built on a synthetic four-atom force field rather than an OpenMM system, so
# the contract is checked by arithmetic and runs in any environment.
# ---------------------------------------------------------------------------

def _tiny_params():
    """A four-atom force field, every term present and none of them zero."""
    from jaxmm.extract import (AngleParams, BondParams, ForceFieldParams,
                               NonbondedParams, TorsionParams)
    i32 = lambda v: jnp.asarray(v, jnp.int32)
    f64 = lambda v: jnp.asarray(v, jnp.float64)
    return ForceFieldParams(
        bonds=BondParams(atom_i=i32([0, 1, 2]), atom_j=i32([1, 2, 3]),
                         r0=f64([0.15, 0.15, 0.15]), k=f64([2.0e5, 2.0e5, 2.0e5])),
        angles=AngleParams(atom_i=i32([0, 1]), atom_j=i32([1, 2]), atom_k=i32([2, 3]),
                           theta0=f64([1.91, 1.91]), k=f64([400.0, 400.0])),
        torsions=TorsionParams(atom_i=i32([0]), atom_j=i32([1]), atom_k=i32([2]),
                               atom_l=i32([3]), periodicity=i32([3]),
                               phase=f64([0.0]), k=f64([5.0])),
        nonbonded=NonbondedParams(
            charges=f64([0.1, -0.1, 0.1, -0.1]), sigmas=f64([0.3] * 4),
            epsilons=f64([0.4] * 4), n_atoms=4,
            exclusion_pairs=i32([[0, 1], [1, 2], [2, 3], [0, 2], [1, 3]]),
            exception_pairs=i32([[0, 3]]), exception_chargeprod=f64([-0.005]),
            exception_sigma=f64([0.3]), exception_epsilon=f64([0.2]),
            cutoff=None, switch_distance=None),
        masses=f64([12.0, 12.0, 12.0, 12.0]), n_atoms=4)


def _tiny_zmatrix():
    from jaxmm.coordinates import ZMatrix
    return ZMatrix(bond_ref=jnp.asarray([-1, 0, 1, 2], jnp.int32),
                   angle_ref=jnp.asarray([-1, -1, 0, 1], jnp.int32),
                   torsion_ref=jnp.asarray([-1, -1, -1, 0], jnp.int32))


class TestInternalDensityOffChart:

    B = jnp.asarray([0.15, 0.15, 0.15], jnp.float64)
    A = jnp.asarray([1.91, 1.91], jnp.float64)
    T = jnp.asarray([1.0], jnp.float64)

    def test_the_density_is_minus_inf_off_the_chart(self):
        """
        Claim: an angle outside (0, pi) or a non-positive bond length gives
        -inf, a legitimate zero, not NaN.
        Bug it catches: the defect this was written for. The Jacobian returned
        NaN there, and an importance-weighted trainer treats NaN as neither
        pass nor fail: it either aborts the run or silently drops the level.
        Oracle: the two boundary violations, one each.
        """
        z, ff = _tiny_zmatrix(), _tiny_params()

        for bad_b, bad_a in ((self.B, self.A.at[1].set(jnp.pi + 0.01)),
                             (self.B.at[1].set(-0.15), self.A)):
            value = float(jaxmm.log_boltzmann_internal(z, bad_b, bad_a, self.T, ff, 300.0))

            assert value == -np.inf, f"got {value}"

    def test_the_density_is_unchanged_on_the_chart(self):
        """
        Regression: masking must not move a value that was already right.
        Oracle: the Cartesian density plus the log Jacobian, formed separately.
        """
        z, ff = _tiny_zmatrix(), _tiny_params()
        x = jaxmm.zmatrix_to_cartesian(z, self.B, self.A, self.T)
        expected = (float(jaxmm.log_boltzmann(x, ff, 300.0))
                    + float(jaxmm.zmatrix_log_abs_det_jacobian(z, self.B, self.A)))

        got = float(jaxmm.log_boltzmann_internal(z, self.B, self.A, self.T, ff, 300.0))

        np.testing.assert_allclose(got, expected, rtol=1e-12)

    def test_the_gradient_off_the_chart_is_finite(self):
        """
        Claim: the gradient at an off-chart point carries no NaN.
        Bug it catches: masking only the output. `jnp.where` applied to the
        result still differentiates the unmasked branch, so `0 * NaN = NaN`
        reaches the caller. Only masking the *inputs* as well avoids it. This
        matters because gradient-based samplers and trainers differentiate this
        function, and one NaN poisons an entire parameter update.
        Oracle: every gradient entry finite, at a point that is off the chart.
        """
        z, ff = _tiny_zmatrix(), _tiny_params()
        bad_a = self.A.at[1].set(jnp.pi + 0.01)

        grads = jax.grad(lambda b, a, t: jaxmm.log_boltzmann_internal(z, b, a, t, ff, 300.0),
                         argnums=(0, 1, 2))(self.B, bad_a, self.T)

        for g in grads:
            assert np.all(np.isfinite(np.asarray(g))), f"non-finite gradient: {g}"

    def test_it_vmaps_over_a_batch_with_mixed_validity(self):
        """
        Claim: a batch containing off-chart rows returns -inf for those rows
        and correct values for the rest.
        Bug it catches: a Python-level branch on the predicate, which cannot
        trace and would force the caller out of jit or, worse, reject the whole
        batch.
        Oracle: the per-row results against the same rows evaluated singly.
        """
        z, ff = _tiny_zmatrix(), _tiny_params()
        bonds = jnp.stack([self.B, self.B, self.B.at[0].set(-0.15)])
        angles = jnp.stack([self.A, self.A.at[0].set(-0.01), self.A])
        torsions = jnp.stack([self.T, self.T, self.T])

        out = jax.vmap(lambda b, a, t: jaxmm.log_boltzmann_internal(z, b, a, t, ff, 300.0))(
            bonds, angles, torsions)

        assert np.isfinite(float(out[0]))
        assert float(out[1]) == -np.inf and float(out[2]) == -np.inf


# ---------------------------------------------------------------------------
# Construction order separate from atom order.
#
# Without `atom_order` the z-matrix forces construction order == atom index,
# so the frame must be atoms 0, 1, 2 and every reference must point backwards
# in atom numbering. For a real molecule that is the wrong constraint: the
# rigid backbone is rarely numbered first, and an automatically built z-matrix
# (bgmol's ZMatrixFactory, FAB's hand-written one) is not in atom order at all.
#
# `atom_order[c] = a` means construction step c places atom a. The references
# stay in construction space, so the `ref[c] < c` invariant is untouched and
# only the two boundaries permute.
# ---------------------------------------------------------------------------

CHAIN_REFS = dict(bond_ref=jnp.asarray([-1, 0, 1, 2], jnp.int32),
                  angle_ref=jnp.asarray([-1, -1, 0, 1], jnp.int32),
                  torsion_ref=jnp.asarray([-1, -1, -1, 0], jnp.int32))


def _distance_matrix_np(x):
    x = np.asarray(x, np.float64)
    return np.linalg.norm(x[:, None, :] - x[None, :, :], axis=-1)


class TestAtomOrder:

    def test_no_atom_order_is_the_identity(self):
        """
        Claim: omitting `atom_order` leaves every existing result unchanged.
        Bug it catches: a permutation applied unconditionally, which would
        silently scramble every z-matrix already in use, including the shipped
        one and every saved result derived from it.
        Oracle: the same call with an explicit identity permutation.
        """
        from jaxmm.coordinates import ZMatrix
        # Built from one set of references two ways, since aldp_zmatrix now
        # carries a real permutation and would not test this.
        unset = ZMatrix(**CHAIN_REFS)
        identity = ZMatrix(**CHAIN_REFS,
                           atom_order=jnp.arange(unset.n_atoms, dtype=jnp.int32))
        z = unset
        x = jax.random.normal(jax.random.PRNGKey(4), (unset.n_atoms, 3), jnp.float64) * 0.15

        a = jaxmm.cartesian_to_zmatrix(z, x)
        b = jaxmm.cartesian_to_zmatrix(identity, x)

        for u, v in zip(a, b):
            np.testing.assert_array_equal(np.asarray(u), np.asarray(v))

    def test_a_permuted_order_describes_the_same_molecule(self):
        """
        Claim: a z-matrix that builds the same molecule in a different order
        reproduces the same geometry, since a distance matrix does not care
        which atom was placed first.
        Bug it catches: permuting on one boundary and not the other, which
        round trips consistently while describing a different molecule.
        Oracle: the distance matrix of the input, which is frame independent.
        """
        from jaxmm.coordinates import ZMatrix
        reversed_order = ZMatrix(**CHAIN_REFS,
                                 atom_order=jnp.asarray([3, 2, 1, 0], jnp.int32))
        x = jnp.asarray([[0.0, 0.0, 0.0], [0.15, 0.0, 0.0],
                         [0.20, 0.14, 0.0], [0.34, 0.16, 0.05]], jnp.float64)

        b, a, t = jaxmm.cartesian_to_zmatrix(reversed_order, x)
        rebuilt = jaxmm.zmatrix_to_cartesian(reversed_order, b, a, t)

        np.testing.assert_allclose(_distance_matrix_np(rebuilt), _distance_matrix_np(x),
                                   atol=1e-12)

    def test_the_round_trip_returns_atoms_in_their_own_order(self):
        """
        Claim: `zmatrix_to_cartesian` returns row i for atom i, whatever the
        construction order, so the caller never sees construction indices.
        Bug it catches: returning positions in construction order, which would
        silently mismatch every force-field index, since those are in atom
        order. The energy would be computed for a scrambled molecule and look
        entirely plausible.
        Oracle: the canonicalized input, atom by atom.
        """
        from jaxmm.coordinates import ZMatrix
        reversed_order = ZMatrix(**CHAIN_REFS,
                                 atom_order=jnp.asarray([3, 2, 1, 0], jnp.int32))
        x = jnp.asarray([[0.0, 0.0, 0.0], [0.15, 0.0, 0.0],
                         [0.20, 0.14, 0.0], [0.34, 0.16, 0.05]], jnp.float64)

        b, a, t = jaxmm.cartesian_to_zmatrix(reversed_order, x)
        rebuilt = jaxmm.zmatrix_to_cartesian(reversed_order, b, a, t)
        canonical = jaxmm.canonicalize_cartesian(reversed_order, x)

        np.testing.assert_allclose(np.asarray(rebuilt), np.asarray(canonical), atol=1e-12)

    def test_the_jacobian_ignores_the_construction_order(self):
        """
        Claim: log|det J| is a product over bonds and angles, so relabelling
        which atom each step places cannot change it.
        Bug it catches: permuting the internals inside the Jacobian, which
        would make the density depend on a bookkeeping choice.
        Oracle: the same internals under two orders.
        """
        from jaxmm.coordinates import ZMatrix
        plain = ZMatrix(**CHAIN_REFS)
        permuted = ZMatrix(**CHAIN_REFS, atom_order=jnp.asarray([3, 2, 1, 0], jnp.int32))
        b = jnp.asarray([0.15, 0.15, 0.15], jnp.float64)
        a = jnp.asarray([1.91, 1.91], jnp.float64)

        assert (float(jaxmm.zmatrix_log_abs_det_jacobian(plain, b, a))
                == float(jaxmm.zmatrix_log_abs_det_jacobian(permuted, b, a)))

    @pytest.mark.parametrize("bad, match", [
        ([0, 1, 2], "length"),
        ([0, 1, 2, 2], "permutation"),
        ([0, 1, 2, 4], "permutation"),
        ([-1, 1, 2, 3], "permutation"),
    ])
    def test_a_bad_atom_order_is_refused(self, bad, match):
        """
        Claim: `validate_zmatrix` rejects an `atom_order` that is not a
        permutation of every atom exactly once.
        Bug it catches: a duplicated or missing entry, which makes the scatter
        drop atoms and leave others at the origin, a wrong answer that looks
        like a plausible structure.
        Oracle: a hand table of the ways a permutation can fail.
        """
        from jaxmm.coordinates import ZMatrix
        z = ZMatrix(**CHAIN_REFS, atom_order=jnp.asarray(bad, jnp.int32))

        with pytest.raises(ValueError, match=match):
            jaxmm.validate_zmatrix(z)

    def test_a_valid_atom_order_is_accepted(self):
        """Fixture strength: the validator is not simply rejecting everything."""
        from jaxmm.coordinates import ZMatrix
        jaxmm.validate_zmatrix(ZMatrix(**CHAIN_REFS,
                                       atom_order=jnp.asarray([3, 2, 1, 0], jnp.int32)))


# ---------------------------------------------------------------------------
# The rebuilt ALDP z-matrix.
#
# The first one was rooted in the acetyl methyl, which forced every reference
# backwards in atom numbering. Three substituents of CA then shared one
# reference triple whose third atom was not in the frame, so HA, CB and C all
# swept together with phi. That made chirality a *difference* of coordinates
# rather than a coordinate, so no domain restriction on a single torsion could
# select an enantiomer: restricting one would cut the Ramachandran circle and
# delete the alpha-L basin.
#
# With construction order free of atom order the frame is the rigid backbone
# C, CA, N. CA's substituents are then pinned against a frame triple, so
# chirality is a coordinate, and phi and psi are still sampled directly.
#
# Structure below is the standard extended conformation, in nm, embedded so
# these tests need neither OpenMM nor a data file.
# ---------------------------------------------------------------------------

ALDP_XYZ = np.array([
    [0.20000, 0.10000, -0.00000], [0.20000, 0.20900, 0.00000],
    [0.14860, 0.24540, 0.08900], [0.14860, 0.24540, -0.08900],
    [0.34270, 0.26410, -0.00000], [0.43910, 0.18770, -0.00000],
    [0.35550, 0.39700, -0.00000], [0.27330, 0.45560, -0.00000],
    [0.48530, 0.46140, -0.00000], [0.54080, 0.43160, 0.08900],
    [0.56610, 0.42210, -0.12320], [0.51230, 0.45210, -0.21310],
    [0.66300, 0.47190, -0.12060], [0.58090, 0.31410, -0.12410],
    [0.47130, 0.61290, 0.00000], [0.36010, 0.66530, 0.00000],
    [0.58460, 0.68350, 0.00000], [0.67370, 0.63590, -0.00000],
    [0.58460, 0.82840, 0.00000], [0.48190, 0.86480, 0.00000],
    [0.63600, 0.86480, 0.08900], [0.63600, 0.86480, -0.08900],
], dtype=np.float64)

CA, N_ALA, C_ALA, C_ACE, N_NME = 8, 6, 14, 4, 16
PHI_TORSION, PSI_TORSION, CHIRALITY_TORSION = 12, 5, 0


class TestALDPZMatrix:

    def test_the_frame_is_the_backbone(self):
        """
        Claim: the first three construction steps place C, CA and N of the
        alanine residue.
        Bug it catches: the defect this rebuild exists for, a frame inside a
        freely rotating terminal methyl, which forces CA's substituents to
        share a reference triple that moves with phi.
        Oracle: the atom indices, written out.
        """
        z = jaxmm.aldp_zmatrix()

        np.testing.assert_array_equal(np.asarray(z.construction_order)[:3],
                                      [C_ALA, CA, N_ALA])

    def test_it_validates(self):
        """The rebuilt z-matrix satisfies every structural rule, including the permutation."""
        jaxmm.validate_zmatrix(jaxmm.aldp_zmatrix())

    def test_the_round_trip_is_exact(self):
        """
        Claim: internals derived from a real structure rebuild it exactly, in
        atom order.
        Bug it catches: a permutation applied on one boundary only, or refs
        that do not match the construction order.
        Oracle: the canonicalized input, which is the frame-fixed form of the
        same molecule.
        """
        z = jaxmm.aldp_zmatrix()
        x = jnp.asarray(ALDP_XYZ)

        b, a, t = jaxmm.cartesian_to_zmatrix(z, x)
        rebuilt = jaxmm.zmatrix_to_cartesian(z, b, a, t)

        np.testing.assert_allclose(np.asarray(rebuilt),
                                   np.asarray(jaxmm.canonicalize_cartesian(z, x)), atol=1e-12)
        np.testing.assert_allclose(_distance_matrix_np(rebuilt),
                                   _distance_matrix_np(x), atol=1e-9)

    def test_phi_and_psi_are_sampled_directly(self):
        """
        Claim: torsion PHI_TORSION is the IUPAC phi and PSI_TORSION is psi, so
        the slow collective variables are coordinates rather than nonlinear
        functions of coordinates.
        Bug it catches: a construction order in which a backbone rotor is
        placed after its siblings, which leaves phi recoverable only as a sum.
        Oracle: `dihedral_angle` on the same structure, which is an independent
        implementation of the dihedral.
        """
        z = jaxmm.aldp_zmatrix()
        _, _, t = jaxmm.cartesian_to_zmatrix(z, jnp.asarray(ALDP_XYZ))
        quads = jnp.asarray([[C_ACE, N_ALA, CA, C_ALA], [N_ALA, CA, C_ALA, N_NME]], jnp.int32)
        phi, psi = np.asarray(jaxmm.dihedral_angle(jnp.asarray(ALDP_XYZ), quads))

        # dihedral_angle negates to the biochemistry convention; z-matrix
        # torsions follow OpenMM's. Compare as a wrapped magnitude.
        wrap = lambda v: np.degrees((np.asarray(v) + np.pi) % (2 * np.pi) - np.pi)
        assert abs(abs(wrap(t[PHI_TORSION])) - abs(wrap(phi))) < 1e-6
        assert abs(abs(wrap(t[PSI_TORSION])) - abs(wrap(psi))) < 1e-6

    def test_chirality_is_a_single_coordinate_that_flips_under_reflection(self):
        """
        Claim: torsion CHIRALITY_TORSION carries the sign of the alpha carbon's
        handedness on its own, so restricting it to half a circle is an exact
        fundamental domain that selects one enantiomer.
        Bug it catches: the whole reason for the rebuild. If this torsion is a
        rotor instead of a stiff improper, a half-circle restriction cuts the
        Ramachandran circle rather than the enantiomer.
        Oracle: the value on an L structure and on its mirror image.
        """
        z = jaxmm.aldp_zmatrix()
        x = jnp.asarray(ALDP_XYZ)

        _, _, t = jaxmm.cartesian_to_zmatrix(z, x)
        _, _, t_mirror = jaxmm.cartesian_to_zmatrix(z, x * jnp.array([1.0, 1.0, -1.0]))

        value = float(np.degrees(t[CHIRALITY_TORSION]))
        mirrored = float(np.degrees(t_mirror[CHIRALITY_TORSION]))
        assert 100.0 < abs(value) < 140.0, f"expected a stiff improper near 120, got {value}"
        assert np.isclose(value, -mirrored, atol=1e-6), f"{value} did not flip to {-mirrored}"

    def test_every_reference_triple_is_well_conditioned(self):
        """
        Claim: no reference triple is near collinear, so the construction never
        approaches the degenerate case where the placement normal vanishes.
        Bug it catches: a construction order that defines an atom through a
        near-linear triple, which is a silently wrong geometry rather than a
        failure.
        Oracle: the angle at each reference triple, from the structure.
        """
        z = jaxmm.aldp_zmatrix()
        order = np.asarray(z.construction_order)
        br, ar, tr = (np.asarray(v) for v in (z.bond_ref, z.angle_ref, z.torsion_ref))

        worst = 180.0
        for c in range(3, len(order)):
            p = ALDP_XYZ[order[[tr[c], ar[c], br[c]]]]
            u, v = p[0] - p[1], p[2] - p[1]
            cos = np.dot(u, v) / (np.linalg.norm(u) * np.linalg.norm(v))
            angle = np.degrees(np.arccos(np.clip(cos, -1, 1)))
            worst = min(worst, min(angle, 180.0 - angle))

        assert worst > 15.0, f"a reference triple is only {worst:.1f} degrees from collinear"

    def test_each_methyl_has_one_rotor_and_two_pinned_torsions(self):
        """
        Claim: within a methyl, one hydrogen is referenced to the backbone and
        the other two to that hydrogen, so they are pinned near +/-120 rather
        than sweeping together.
        Bug it catches: all three referenced to the backbone, which is what the
        first z-matrix did. Three coordinates then move as one, which is a
        near-deterministic dependence between coordinates and the classic way
        to destroy an importance-sampling ESS.
        Oracle: the pinned torsions' values on the real structure.
        """
        z = jaxmm.aldp_zmatrix()
        _, _, t = jaxmm.cartesian_to_zmatrix(z, jnp.asarray(ALDP_XYZ))
        order = np.asarray(z.construction_order)
        tr = np.asarray(z.torsion_ref)

        pinned = [c for c in range(3, len(order))
                  if order[c] in (12, 13, 20, 21)]       # the second and third methyl H of each
        assert len(pinned) == 4
        for c in pinned:
            assert tr[c] >= 3, "a pinned hydrogen must reference its sibling, not the frame"
            value = abs(float(np.degrees(t[c - 3])))
            assert 100.0 < value < 140.0, f"step {c} torsion {value} is not a pinned +/-120"
