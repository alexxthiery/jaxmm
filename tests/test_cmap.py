"""Tests for CMAP torsion correction energy."""
import copy

import numpy as np
import jax
import jax.numpy as jnp
import openmm
from openmm import unit
import pytest

from jaxmm.extract import extract_params
from jaxmm.energy import cmap_energy, total_energy, energy_components


def _make_cmap_system():
    """Build an 8-atom system with a synthetic CMAPTorsionForce.

    Two overlapping dihedrals: atoms 0-1-2-3 (phi) and 1-2-3-4 (psi),
    plus extra atoms 5-6-7 as padding. Uses a small 6x6 CMAP grid.

    Returns (system, positions_nm).
    """
    system = openmm.System()
    for _ in range(8):
        system.addParticle(12.0 * unit.amu)

    # Required forces so extract_params doesn't raise
    system.addForce(openmm.HarmonicBondForce())
    system.addForce(openmm.HarmonicAngleForce())
    system.addForce(openmm.PeriodicTorsionForce())
    nb = openmm.NonbondedForce()
    for _ in range(8):
        nb.addParticle(0.0, 0.1, 0.0)
    system.addForce(nb)

    # Build a 6x6 CMAP grid with known values
    size = 6
    rng = np.random.RandomState(42)
    grid_values = rng.randn(size * size).tolist()

    cmap_force = openmm.CMAPTorsionForce()
    cmap_force.addMap(size, grid_values)
    # phi: atoms 0,1,2,3; psi: atoms 1,2,3,4
    cmap_force.addTorsion(0, 0, 1, 2, 3, 1, 2, 3, 4)
    system.addForce(cmap_force)

    # Non-planar positions
    positions = np.array([
        [0.0, 0.0, 0.0],
        [0.15, 0.0, 0.0],
        [0.15, 0.15, 0.0],
        [0.15, 0.15, 0.15],
        [0.0, 0.15, 0.15],
        [0.3, 0.0, 0.0],
        [0.3, 0.15, 0.0],
        [0.3, 0.15, 0.15],
    ], dtype=np.float64)

    return system, positions


def _openmm_cmap_energy(system, positions_nm):
    """Get CMAP energy from OpenMM using isolated system."""
    iso = openmm.System()
    for i in range(system.getNumParticles()):
        iso.addParticle(system.getParticleMass(i))
    for i in range(system.getNumForces()):
        f = system.getForce(i)
        if isinstance(f, openmm.CMAPTorsionForce):
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
def cmap_system_and_pos():
    return _make_cmap_system()


def test_cmap_energy_vs_openmm(cmap_system_and_pos):
    system, pos = cmap_system_and_pos
    params = extract_params(system)
    assert params.cmap is not None

    jax_e = float(cmap_energy(jnp.array(pos), params.cmap))
    omm_e = _openmm_cmap_energy(system, pos)
    # Bicubic interpolation may differ slightly from OpenMM's natural cubic spline
    assert abs(jax_e - omm_e) < 0.5, f"jax={jax_e}, openmm={omm_e}"


def test_cmap_energy_grad(cmap_system_and_pos):
    system, pos = cmap_system_and_pos
    params = extract_params(system)
    pos_jnp = jnp.array(pos)
    grad = jax.grad(cmap_energy)(pos_jnp, params.cmap)
    assert not jnp.any(jnp.isnan(grad))
    assert not jnp.any(jnp.isinf(grad))


def test_cmap_energy_jit(cmap_system_and_pos):
    system, pos = cmap_system_and_pos
    params = extract_params(system)
    pos_jnp = jnp.array(pos)
    ref = float(cmap_energy(pos_jnp, params.cmap))
    jit_e = float(jax.jit(cmap_energy)(pos_jnp, params.cmap))
    assert abs(jit_e - ref) < 1e-10


def test_cmap_in_total_energy(cmap_system_and_pos):
    system, pos = cmap_system_and_pos
    params = extract_params(system)
    pos_jnp = jnp.array(pos)
    comps = energy_components(pos_jnp, params)
    assert "cmap" in comps
    total = sum(comps.values())
    assert abs(float(total) - float(total_energy(pos_jnp, params))) < 1e-10


def test_cmap_none_when_absent(aldp_params):
    """Existing ALDP system (AMBER) has no CMAP."""
    assert aldp_params.cmap is None
