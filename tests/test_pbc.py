"""Tests for periodic boundary conditions and nonbonded cutoff."""
import copy

import numpy as np
import jax
import jax.numpy as jnp
import openmm
from openmm import unit
import pytest

from jaxmm.energy import _minimum_image, _lj_switch
from jaxmm.extract import extract_params
from jaxmm.energy import nonbonded_energy, total_energy


def test_minimum_image_orthorhombic():
    """Displacement wraps to nearest image."""
    box = jnp.array([2.0, 2.0, 2.0])
    dr = jnp.array([1.5, -1.5, 0.3])
    wrapped = _minimum_image(dr, box)
    expected = jnp.array([-0.5, 0.5, 0.3])
    assert jnp.allclose(wrapped, expected, atol=1e-10)


def test_minimum_image_no_wrap_needed():
    """Small displacements unchanged."""
    box = jnp.array([3.0, 3.0, 3.0])
    dr = jnp.array([0.1, -0.2, 0.5])
    wrapped = _minimum_image(dr, box)
    assert jnp.allclose(wrapped, dr, atol=1e-10)


def test_switching_function_endpoints():
    """S(r_sw) = 1, S(r_cut) = 0, monotonic."""
    r_sw, r_cut = 0.8, 1.0
    assert abs(float(_lj_switch(jnp.array(r_sw), r_sw, r_cut)) - 1.0) < 1e-10
    assert abs(float(_lj_switch(jnp.array(r_cut), r_sw, r_cut))) < 1e-10
    assert abs(float(_lj_switch(jnp.array(0.5), r_sw, r_cut)) - 1.0) < 1e-10
    # Monotonic in between
    rs = jnp.linspace(r_sw, r_cut, 50)
    vals = _lj_switch(rs, r_sw, r_cut)
    assert jnp.all(jnp.diff(vals) <= 1e-10)


def test_switching_function_grad():
    """Gradient of switching function is smooth (no NaN)."""
    r_sw, r_cut = 0.8, 1.0
    grad_fn = jax.grad(lambda r: _lj_switch(r, r_sw, r_cut))
    for r in [0.7, 0.8, 0.85, 0.9, 0.95, 1.0, 1.1]:
        g = grad_fn(jnp.array(r))
        assert not jnp.isnan(g)


def _make_periodic_lj_system():
    """Build a 2-particle LJ system with CutoffPeriodic method.

    Returns (system, positions_nm).
    """
    system = openmm.System()
    system.addParticle(12.0 * unit.amu)
    system.addParticle(12.0 * unit.amu)

    # Set periodic box
    box_length = 3.0 * unit.nanometer
    system.setDefaultPeriodicBoxVectors(
        openmm.Vec3(box_length.value_in_unit(unit.nanometer), 0, 0) * unit.nanometer,
        openmm.Vec3(0, box_length.value_in_unit(unit.nanometer), 0) * unit.nanometer,
        openmm.Vec3(0, 0, box_length.value_in_unit(unit.nanometer)) * unit.nanometer,
    )

    # Required forces
    system.addForce(openmm.HarmonicBondForce())
    system.addForce(openmm.HarmonicAngleForce())
    system.addForce(openmm.PeriodicTorsionForce())

    nb = openmm.NonbondedForce()
    nb.setNonbondedMethod(openmm.NonbondedForce.CutoffPeriodic)
    nb.setCutoffDistance(1.2 * unit.nanometer)
    nb.setUseSwitchingFunction(True)
    nb.setSwitchingDistance(1.0 * unit.nanometer)
    nb.setUseDispersionCorrection(False)
    # Two neutral LJ particles
    nb.addParticle(0.0, 0.3, 1.0)  # charge=0, sigma=0.3nm, eps=1.0 kJ/mol
    nb.addParticle(0.0, 0.3, 1.0)
    system.addForce(nb)

    positions = np.array([
        [0.5, 0.5, 0.5],
        [1.0, 0.5, 0.5],
    ], dtype=np.float64)

    return system, positions


def _openmm_total_energy(system, positions_nm):
    """Get total energy from OpenMM."""
    ctx = openmm.Context(system, openmm.VerletIntegrator(0.001),
                         openmm.Platform.getPlatformByName("CPU"))
    ctx.setPositions(positions_nm * unit.nanometer)
    e = ctx.getState(getEnergy=True).getPotentialEnergy().value_in_unit(
        unit.kilojoule_per_mole)
    del ctx
    return e


def test_pbc_extraction():
    """Extract params from periodic system gets cutoff, switch, box."""
    system, pos = _make_periodic_lj_system()
    params = extract_params(system)
    assert params.nonbonded.cutoff == pytest.approx(1.2)
    assert params.nonbonded.switch_distance == pytest.approx(1.0)
    assert params.box is not None
    assert jnp.allclose(params.box, jnp.array([3.0, 3.0, 3.0]))


def test_vacuum_has_no_cutoff(aldp_params):
    """Vacuum system has no cutoff or box."""
    assert aldp_params.nonbonded.cutoff is None
    assert aldp_params.nonbonded.switch_distance is None
    assert aldp_params.box is None


def test_pbc_nonbonded_vs_openmm():
    """Compare periodic nonbonded energy against OpenMM CutoffPeriodic."""
    system, pos = _make_periodic_lj_system()
    params = extract_params(system)
    pos_jnp = jnp.array(pos)

    jax_e = float(total_energy(pos_jnp, params))
    omm_e = _openmm_total_energy(system, pos)
    assert abs(jax_e - omm_e) < 1e-4, f"jax={jax_e}, openmm={omm_e}"


def test_pbc_cutoff_masks_distant_pair():
    """Pair beyond cutoff contributes zero nonbonded energy."""
    system, _ = _make_periodic_lj_system()
    # Place particles 1.3 nm apart (beyond 1.2 nm cutoff, within half box)
    pos_far = np.array([
        [0.5, 0.5, 0.5],
        [0.5, 0.5, 0.5 + 1.3],
    ], dtype=np.float64)
    params = extract_params(system)
    # Nonbonded should be zero (only two particles, beyond cutoff)
    e = float(total_energy(jnp.array(pos_far), params))
    assert abs(e) < 1e-10, f"expected ~0, got {e}"


def test_pbc_minimum_image_wrapping():
    """Particles close via PBC wrapping interact correctly."""
    system, _ = _make_periodic_lj_system()
    # Particles at 0.2 and 2.8 in a box of 3.0: minimum image distance = 0.6 nm
    pos_wrap = np.array([
        [0.2, 0.5, 0.5],
        [2.8, 0.5, 0.5],
    ], dtype=np.float64)
    params = extract_params(system)
    pos_jnp = jnp.array(pos_wrap)

    jax_e = float(total_energy(pos_jnp, params))
    omm_e = _openmm_total_energy(system, pos_wrap)
    assert abs(jax_e - omm_e) < 1e-4, f"jax={jax_e}, openmm={omm_e}"


def test_pbc_nonbonded_jit():
    """JIT consistency for periodic nonbonded."""
    system, pos = _make_periodic_lj_system()
    params = extract_params(system)
    pos_jnp = jnp.array(pos)
    ref = float(total_energy(pos_jnp, params))
    jit_e = float(jax.jit(total_energy)(pos_jnp, params))
    assert abs(jit_e - ref) < 1e-10


def test_pbc_nonbonded_grad():
    """Gradient is finite for periodic system."""
    system, pos = _make_periodic_lj_system()
    params = extract_params(system)
    pos_jnp = jnp.array(pos)
    grad = jax.grad(total_energy)(pos_jnp, params)
    assert not jnp.any(jnp.isnan(grad))
    assert not jnp.any(jnp.isinf(grad))
