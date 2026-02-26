"""Tests for Ryckaert-Bellemans torsion energy."""
import copy

import numpy as np
import jax
import jax.numpy as jnp
import openmm
from openmm import unit
import pytest

from jaxmm.extract import extract_params
from jaxmm.energy import rb_torsion_energy, total_energy, energy_components


def _make_rb_system():
    """Build a 4-atom system with one RB torsion for testing.

    Returns (system, positions_nm). The system has:
    - 4 particles (mass 12 amu each)
    - 1 RB torsion with known C0..C5 coefficients
    - A NonbondedForce with zero charges (required by extract_params)
    - HarmonicBondForce, HarmonicAngleForce, PeriodicTorsionForce (all empty)
    """
    system = openmm.System()
    for _ in range(4):
        system.addParticle(12.0 * unit.amu)

    # Required empty forces so extract_params doesn't raise
    system.addForce(openmm.HarmonicBondForce())
    system.addForce(openmm.HarmonicAngleForce())
    system.addForce(openmm.PeriodicTorsionForce())
    nb = openmm.NonbondedForce()
    for _ in range(4):
        nb.addParticle(0.0, 0.1, 0.0)
    system.addForce(nb)

    # RB torsion with known coefficients (kJ/mol)
    rb = openmm.RBTorsionForce()
    rb.addTorsion(0, 1, 2, 3,
                  2.0 * unit.kilojoule_per_mole,
                  -1.5 * unit.kilojoule_per_mole,
                  0.5 * unit.kilojoule_per_mole,
                  0.3 * unit.kilojoule_per_mole,
                  -0.1 * unit.kilojoule_per_mole,
                  0.05 * unit.kilojoule_per_mole)
    system.addForce(rb)

    # Non-planar positions for a nontrivial dihedral angle
    positions = np.array([
        [0.0, 0.0, 0.0],
        [0.15, 0.0, 0.0],
        [0.15, 0.15, 0.0],
        [0.15, 0.15, 0.15],
    ], dtype=np.float64)

    return system, positions


def _openmm_rb_energy(system, positions_nm):
    """Get RB torsion energy from OpenMM using isolated system."""
    iso = openmm.System()
    for i in range(system.getNumParticles()):
        iso.addParticle(system.getParticleMass(i))
    for i in range(system.getNumForces()):
        f = system.getForce(i)
        if isinstance(f, openmm.RBTorsionForce):
            iso.addForce(copy.deepcopy(f))
            break
    ctx = openmm.Context(iso, openmm.VerletIntegrator(0.001),
                         openmm.Platform.getPlatformByName("CPU"))
    ctx.setPositions(positions_nm * unit.nanometer)
    e = ctx.getState(getEnergy=True).getPotentialEnergy().value_in_unit(
        unit.kilojoule_per_mole)
    del ctx
    return e


@pytest.fixture
def rb_system_and_pos():
    return _make_rb_system()


def test_rb_torsion_energy_vs_openmm(rb_system_and_pos):
    system, pos = rb_system_and_pos
    params = extract_params(system)
    assert params.rb_torsions is not None

    jax_e = float(rb_torsion_energy(jnp.array(pos), params.rb_torsions))
    omm_e = _openmm_rb_energy(system, pos)
    assert abs(jax_e - omm_e) < 1e-4, f"jax={jax_e}, openmm={omm_e}"


def test_rb_torsion_energy_grad(rb_system_and_pos):
    system, pos = rb_system_and_pos
    params = extract_params(system)
    pos_jnp = jnp.array(pos)
    grad = jax.grad(rb_torsion_energy)(pos_jnp, params.rb_torsions)
    assert not jnp.any(jnp.isnan(grad))
    assert not jnp.any(jnp.isinf(grad))


def test_rb_torsion_energy_jit(rb_system_and_pos):
    system, pos = rb_system_and_pos
    params = extract_params(system)
    pos_jnp = jnp.array(pos)
    ref = float(rb_torsion_energy(pos_jnp, params.rb_torsions))
    jit_e = float(jax.jit(rb_torsion_energy)(pos_jnp, params.rb_torsions))
    assert abs(jit_e - ref) < 1e-10


def test_rb_torsion_in_total_energy(rb_system_and_pos):
    system, pos = rb_system_and_pos
    params = extract_params(system)
    pos_jnp = jnp.array(pos)
    comps = energy_components(pos_jnp, params)
    assert "rb_torsions" in comps
    total = sum(comps.values())
    assert abs(float(total) - float(total_energy(pos_jnp, params))) < 1e-10


def test_rb_torsion_none_when_absent(aldp_params):
    """Existing ALDP system has no RB torsions."""
    assert aldp_params.rb_torsions is None
