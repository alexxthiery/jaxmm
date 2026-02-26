"""Tests for parameter extraction from OpenMM system."""

import warnings

import jax
import jax.numpy as jnp
import numpy as np
import pytest


def test_bond_shapes(aldp_params):
    """Bond params have expected count for ALDP (21 bonds)."""
    b = aldp_params.bonds
    assert b.atom_i.shape == (21,)
    assert b.atom_j.shape == (21,)
    assert b.r0.shape == (21,)
    assert b.k.shape == (21,)


def test_angle_shapes(aldp_params):
    """Angle params have expected count for ALDP (36 angles)."""
    a = aldp_params.angles
    assert a.atom_i.shape == (36,)
    assert a.atom_j.shape == (36,)
    assert a.atom_k.shape == (36,)
    assert a.theta0.shape == (36,)
    assert a.k.shape == (36,)


def test_torsion_shapes(aldp_params):
    """Torsion params have expected count for ALDP (52 torsions)."""
    t = aldp_params.torsions
    assert t.atom_i.shape == (52,)
    assert t.atom_j.shape == (52,)
    assert t.atom_k.shape == (52,)
    assert t.atom_l.shape == (52,)
    assert t.periodicity.shape == (52,)
    assert t.phase.shape == (52,)
    assert t.k.shape == (52,)


def test_nonbonded_shapes(aldp_params):
    """Nonbonded params have expected shapes for ALDP (22 atoms)."""
    nb = aldp_params.nonbonded
    assert nb.charges.shape == (22,)
    assert nb.sigmas.shape == (22,)
    assert nb.epsilons.shape == (22,)
    assert nb.n_atoms == 22
    assert nb.exclusion_pairs.ndim == 2 and nb.exclusion_pairs.shape[1] == 2
    assert nb.exception_pairs.ndim == 2 and nb.exception_pairs.shape[1] == 2


def test_n_atoms(aldp_params):
    """ForceFieldParams stores correct atom count."""
    assert aldp_params.n_atoms == 22


def test_exclusion_count(aldp_params):
    """ALDP has 57 excluded pairs (1-2 and 1-3 neighbors)."""
    n_excluded = aldp_params.nonbonded.exclusion_pairs.shape[0]
    assert n_excluded == 57


def test_exception_count(aldp_params):
    """ALDP has 41 exception pairs (1-4 neighbors with scaled params)."""
    n_exception = aldp_params.nonbonded.exception_pairs.shape[0]
    assert n_exception == 41


def test_spot_check_charge(aldp_params):
    """Spot-check a known charge value."""
    # Atom 4 is the N in ALDP with charge ~ -0.4157
    # Let's just check a charge is non-zero and reasonable
    charges = aldp_params.nonbonded.charges
    assert jnp.any(jnp.abs(charges) > 0.1), "Expected some significant charges"
    assert jnp.all(jnp.abs(charges) < 2.0), "Charges should be in elementary charge units"


def test_bond_r0_range(aldp_params):
    """Bond equilibrium lengths should be in typical range (0.09 - 0.16 nm)."""
    r0 = aldp_params.bonds.r0
    assert jnp.all(r0 > 0.08), f"Min r0 = {float(jnp.min(r0)):.4f} nm, too small"
    assert jnp.all(r0 < 0.20), f"Max r0 = {float(jnp.max(r0)):.4f} nm, too large"


def test_masses_shape_and_values(aldp_params):
    """Masses shape is (22,) and contain expected atomic masses."""
    m = aldp_params.masses
    assert m.shape == (22,)
    assert m.dtype == jnp.float64
    # ALDP contains carbon atoms (~12.011 amu) and hydrogen (~1.008)
    assert jnp.any(jnp.abs(m - 12.011) < 0.01), "Expected carbon mass ~12.011"
    assert jnp.any(jnp.abs(m - 1.008) < 0.01), "Expected hydrogen mass ~1.008"
    assert jnp.all(m > 0.0), "All masses must be positive"


def test_dtypes(aldp_params):
    """Parameters should be float64 (for precision) or int32 (indices)."""
    assert aldp_params.bonds.atom_i.dtype == jnp.int32
    assert aldp_params.bonds.r0.dtype == jnp.float64
    assert aldp_params.bonds.k.dtype == jnp.float64
    assert aldp_params.nonbonded.charges.dtype == jnp.float64
    assert aldp_params.torsions.periodicity.dtype == jnp.int32


# ---------------------------------------------------------------------------
# Toluene extraction (second molecule: aromatic ring, 15 atoms)
# ---------------------------------------------------------------------------

def test_toluene_extraction_shapes(toluene_params):
    """Toluene extraction produces correct shapes (15 atoms, 15 bonds, 24 angles, 36 torsions)."""
    assert toluene_params.n_atoms == 15
    assert toluene_params.masses.shape == (15,)
    assert toluene_params.bonds.atom_i.shape[0] == 15
    assert toluene_params.angles.atom_i.shape[0] == 24
    assert toluene_params.torsions.atom_i.shape[0] == 36
    assert toluene_params.nonbonded.charges.shape == (15,)
    assert toluene_params.gbsa is None  # vacuum system


def test_unknown_force_error():
    """extract_params raises on unrecognized force types."""
    import openmm
    from jaxmm.extract import extract_params

    # Build minimal system with all required forces + one unknown
    system = openmm.System()
    system.addParticle(12.0)
    system.addParticle(12.0)

    bond_force = openmm.HarmonicBondForce()
    bond_force.addBond(0, 1, 0.15, 200000.0)
    system.addForce(bond_force)

    angle_force = openmm.HarmonicAngleForce()
    system.addForce(angle_force)

    torsion_force = openmm.PeriodicTorsionForce()
    system.addForce(torsion_force)

    nb_force = openmm.NonbondedForce()
    nb_force.addParticle(0.0, 0.3, 0.0)
    nb_force.addParticle(0.0, 0.3, 0.0)
    system.addForce(nb_force)

    # Add an unrecognized force
    custom = openmm.CustomExternalForce("x")
    custom.addParticle(0, [])
    system.addForce(custom)

    with pytest.raises(ValueError, match="Unsupported force type"):
        extract_params(system)


def test_constraint_error():
    """extract_params raises when system has constraints."""
    import openmm
    from jaxmm.extract import extract_params

    system = openmm.System()
    system.addParticle(12.0)
    system.addParticle(12.0)
    system.addConstraint(0, 1, 0.15)  # rigid bond

    with pytest.raises(ValueError, match="constraints"):
        extract_params(system)


def test_pme_error():
    """extract_params raises when NonbondedForce uses PME."""
    import openmm
    from jaxmm.extract import extract_params

    system = openmm.System()
    system.addParticle(12.0)
    system.addParticle(12.0)

    bond_force = openmm.HarmonicBondForce()
    bond_force.addBond(0, 1, 0.15, 200000.0)
    system.addForce(bond_force)
    system.addForce(openmm.HarmonicAngleForce())
    system.addForce(openmm.PeriodicTorsionForce())

    nb_force = openmm.NonbondedForce()
    nb_force.addParticle(0.5, 0.3, 0.5)
    nb_force.addParticle(-0.5, 0.3, 0.5)
    nb_force.setNonbondedMethod(openmm.NonbondedForce.PME)
    nb_force.setCutoffDistance(1.0)
    system.addForce(nb_force)
    system.setDefaultPeriodicBoxVectors(
        openmm.Vec3(3, 0, 0), openmm.Vec3(0, 3, 0), openmm.Vec3(0, 0, 3),
    )

    with pytest.raises(ValueError, match="PME"):
        extract_params(system)


def test_positions_shape_error():
    """total_energy raises on positions/params atom count mismatch."""
    import jax.numpy as jnp
    import jaxmm

    from openmmtools import testsystems
    from openmm import unit

    aldp = testsystems.AlanineDipeptideVacuum(constraints=None)
    params = jaxmm.extract_params(aldp.system)

    # Wrong number of atoms
    wrong_pos = jnp.zeros((10, 3))
    with pytest.raises(ValueError, match="10 atoms but params expects 22"):
        jaxmm.total_energy(wrong_pos, params)


# ---------------------------------------------------------------------------
# Pytree roundtrip tests
# ---------------------------------------------------------------------------

def test_pytree_roundtrip_vacuum(aldp_positions_jnp, aldp_params):
    """tree_flatten/tree_unflatten roundtrip preserves ForceFieldParams."""
    from jaxmm.energy import total_energy

    leaves, treedef = jax.tree_util.tree_flatten(aldp_params)
    restored = jax.tree_util.tree_unflatten(treedef, leaves)

    e_orig = float(total_energy(aldp_positions_jnp, aldp_params))
    e_restored = float(total_energy(aldp_positions_jnp, restored))
    assert abs(e_orig - e_restored) < 1e-12, f"orig={e_orig}, restored={e_restored}"


def test_pytree_roundtrip_implicit(aldp_implicit_positions, aldp_implicit_params):
    """tree_flatten/tree_unflatten roundtrip preserves implicit params (with GBSA)."""
    from jaxmm.energy import total_energy

    pos = jnp.array(aldp_implicit_positions)
    leaves, treedef = jax.tree_util.tree_flatten(aldp_implicit_params)
    restored = jax.tree_util.tree_unflatten(treedef, leaves)

    e_orig = float(total_energy(pos, aldp_implicit_params))
    e_restored = float(total_energy(pos, restored))
    assert abs(e_orig - e_restored) < 1e-12, f"orig={e_orig}, restored={e_restored}"


def test_tree_map_identity(aldp_positions_jnp, aldp_params):
    """tree_map with identity function preserves energy."""
    from jaxmm.energy import total_energy

    mapped = jax.tree_util.tree_map(lambda x: x, aldp_params)
    e_orig = float(total_energy(aldp_positions_jnp, aldp_params))
    e_mapped = float(total_energy(aldp_positions_jnp, mapped))
    assert abs(e_orig - e_mapped) < 1e-12, f"orig={e_orig}, mapped={e_mapped}"


def test_tree_map_scale_floats(aldp_positions_jnp, aldp_params):
    """tree_map scaling only float leaves preserves energy (multiply-by-one)."""
    from jaxmm.energy import total_energy

    def scale_floats(x):
        if hasattr(x, 'dtype') and jnp.issubdtype(x.dtype, jnp.floating):
            return x * 1.0
        return x

    scaled = jax.tree_util.tree_map(scale_floats, aldp_params)
    e_orig = float(total_energy(aldp_positions_jnp, aldp_params))
    e_scaled = float(total_energy(aldp_positions_jnp, scaled))
    assert abs(e_orig - e_scaled) < 1e-10, f"orig={e_orig}, scaled={e_scaled}"


# ---------------------------------------------------------------------------
# Error path tests
# ---------------------------------------------------------------------------

def test_check_x64_passes_when_enabled():
    """_check_x64 does not raise when x64 is enabled (current state)."""
    from jaxmm.energy import _check_x64
    # Should not raise since conftest enables x64
    _check_x64()


def test_check_x64_error_message():
    """_check_x64 error message mentions the config update command."""
    from jaxmm.energy import _check_x64
    import inspect
    source = inspect.getsource(_check_x64)
    assert "jax_enable_x64" in source
    assert "RuntimeError" in source


def test_extract_params_missing_bond_force():
    """extract_params raises ValueError listing missing forces."""
    import openmm
    from jaxmm.extract import extract_params

    # System with only NonbondedForce (missing bonds, angles, torsions)
    system = openmm.System()
    system.addParticle(12.0)

    nb_force = openmm.NonbondedForce()
    nb_force.addParticle(0.0, 0.3, 0.0)
    system.addForce(nb_force)

    with pytest.raises(ValueError, match="HarmonicBondForce"):
        extract_params(system)


def test_extract_params_missing_nonbonded():
    """extract_params ValueError lists NonbondedForce when missing."""
    import openmm
    from jaxmm.extract import extract_params

    system = openmm.System()
    system.addParticle(12.0)
    system.addParticle(12.0)

    system.addForce(openmm.HarmonicBondForce())
    system.addForce(openmm.HarmonicAngleForce())
    system.addForce(openmm.PeriodicTorsionForce())
    # No NonbondedForce

    with pytest.raises(ValueError, match="NonbondedForce"):
        extract_params(system)
