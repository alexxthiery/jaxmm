"""Shared test fixtures for jaxmm tests.

Provides alanine dipeptide system, positions, MD frames, and a helper
to compute OpenMM energy for an isolated force (single-force system).
"""

import warnings
warnings.filterwarnings("ignore")

import copy

import numpy as np
import pytest

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

import openmm
from openmm import unit
from openmmtools import testsystems

from jaxmm.extract import extract_params


@pytest.fixture(scope="session")
def aldp_testsystem():
    """Create alanine dipeptide vacuum test system (cached for session)."""
    return testsystems.AlanineDipeptideVacuum(constraints=None)


@pytest.fixture(scope="session")
def aldp_system(aldp_testsystem):
    """OpenMM System for alanine dipeptide."""
    return aldp_testsystem.system


@pytest.fixture(scope="session")
def aldp_topology(aldp_testsystem):
    """OpenMM Topology for alanine dipeptide."""
    return aldp_testsystem.topology


@pytest.fixture(scope="session")
def aldp_positions(aldp_testsystem):
    """Energy-minimized positions as numpy (22, 3) in nm."""
    system = aldp_testsystem.system
    topology = aldp_testsystem.topology
    positions = aldp_testsystem.positions

    sim = openmm.app.Simulation(
        topology,
        system,
        openmm.LangevinIntegrator(
            300 * unit.kelvin,
            1.0 / unit.picosecond,
            1.0 * unit.femtosecond,
        ),
        openmm.Platform.getPlatformByName("CPU"),
    )
    sim.context.setPositions(positions)
    sim.minimizeEnergy()

    state = sim.context.getState(getPositions=True)
    pos = state.getPositions(asNumpy=True).value_in_unit(unit.nanometer)
    return pos.astype(np.float64)


@pytest.fixture(scope="session")
def aldp_positions_jnp(aldp_positions):
    """Energy-minimized positions as JAX array (22, 3) float64."""
    return jnp.array(aldp_positions)


@pytest.fixture(scope="session")
def aldp_md_frames(aldp_testsystem, aldp_positions):
    """50 MD frames as numpy (50, 22, 3) in nm."""
    system = aldp_testsystem.system
    topology = aldp_testsystem.topology

    sim = openmm.app.Simulation(
        topology,
        system,
        openmm.LangevinIntegrator(
            300 * unit.kelvin,
            1.0 / unit.picosecond,
            1.0 * unit.femtosecond,
        ),
        openmm.Platform.getPlatformByName("CPU"),
    )
    sim.context.setPositions(aldp_positions * unit.nanometer)

    n_frames = 50
    save_every = 100
    frames = np.empty((n_frames, 22, 3), dtype=np.float64)

    for i in range(n_frames):
        sim.step(save_every)
        state = sim.context.getState(getPositions=True)
        frames[i] = state.getPositions(asNumpy=True).value_in_unit(unit.nanometer)

    return frames


@pytest.fixture(scope="session")
def aldp_params(aldp_system):
    """Extracted ForceFieldParams for alanine dipeptide."""
    return extract_params(aldp_system)


@pytest.fixture(scope="session")
def aldp_implicit_testsystem():
    """Create alanine dipeptide implicit solvent test system (cached for session)."""
    return testsystems.AlanineDipeptideImplicit(constraints=None)


@pytest.fixture(scope="session")
def aldp_implicit_system(aldp_implicit_testsystem):
    """OpenMM System for alanine dipeptide with implicit solvent."""
    return aldp_implicit_testsystem.system


@pytest.fixture(scope="session")
def aldp_implicit_positions(aldp_implicit_testsystem):
    """Energy-minimized positions as numpy (22, 3) in nm."""
    system = aldp_implicit_testsystem.system
    topology = aldp_implicit_testsystem.topology
    positions = aldp_implicit_testsystem.positions

    sim = openmm.app.Simulation(
        topology,
        system,
        openmm.LangevinIntegrator(
            300 * unit.kelvin,
            1.0 / unit.picosecond,
            1.0 * unit.femtosecond,
        ),
        openmm.Platform.getPlatformByName("CPU"),
    )
    sim.context.setPositions(positions)
    sim.minimizeEnergy()

    state = sim.context.getState(getPositions=True)
    pos = state.getPositions(asNumpy=True).value_in_unit(unit.nanometer)
    return pos.astype(np.float64)


@pytest.fixture(scope="session")
def aldp_implicit_md_frames(aldp_implicit_testsystem, aldp_implicit_positions):
    """50 MD frames as numpy (50, 22, 3) in nm."""
    system = aldp_implicit_testsystem.system
    topology = aldp_implicit_testsystem.topology

    sim = openmm.app.Simulation(
        topology,
        system,
        openmm.LangevinIntegrator(
            300 * unit.kelvin,
            1.0 / unit.picosecond,
            1.0 * unit.femtosecond,
        ),
        openmm.Platform.getPlatformByName("CPU"),
    )
    sim.context.setPositions(aldp_implicit_positions * unit.nanometer)

    n_frames = 50
    save_every = 100
    frames = np.empty((n_frames, 22, 3), dtype=np.float64)

    for i in range(n_frames):
        sim.step(save_every)
        state = sim.context.getState(getPositions=True)
        frames[i] = state.getPositions(asNumpy=True).value_in_unit(unit.nanometer)

    return frames


@pytest.fixture(scope="session")
def aldp_implicit_params(aldp_implicit_system):
    """Extracted ForceFieldParams for alanine dipeptide with implicit solvent."""
    return extract_params(aldp_implicit_system)


# ---------------------------------------------------------------------------
# Toluene vacuum fixtures (structurally different from ALDP: aromatic ring)
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def toluene_testsystem():
    """Create toluene vacuum test system (cached for session)."""
    return testsystems.TolueneVacuum(constraints=None)


@pytest.fixture(scope="session")
def toluene_system(toluene_testsystem):
    """OpenMM System for toluene."""
    return toluene_testsystem.system


@pytest.fixture(scope="session")
def toluene_positions(toluene_testsystem):
    """Energy-minimized positions as numpy (15, 3) in nm."""
    system = toluene_testsystem.system
    topology = toluene_testsystem.topology
    positions = toluene_testsystem.positions

    sim = openmm.app.Simulation(
        topology,
        system,
        openmm.LangevinIntegrator(
            300 * unit.kelvin,
            1.0 / unit.picosecond,
            1.0 * unit.femtosecond,
        ),
        openmm.Platform.getPlatformByName("CPU"),
    )
    sim.context.setPositions(positions)
    sim.minimizeEnergy()

    state = sim.context.getState(getPositions=True)
    pos = state.getPositions(asNumpy=True).value_in_unit(unit.nanometer)
    return pos.astype(np.float64)


@pytest.fixture(scope="session")
def toluene_positions_jnp(toluene_positions):
    """Energy-minimized toluene positions as JAX array (15, 3) float64."""
    return jnp.array(toluene_positions)


@pytest.fixture(scope="session")
def toluene_params(toluene_system):
    """Extracted ForceFieldParams for toluene."""
    return extract_params(toluene_system)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def get_openmm_force_energy(system, force_class, positions_nm):
    """Compute energy from a single force type using an isolated system.

    Creates a new system containing only the specified force, evaluates
    energy at the given positions. This avoids the unreliable force group
    API.

    Args:
        system: Original OpenMM System.
        force_class: OpenMM force class (e.g., openmm.HarmonicBondForce).
        positions_nm: Atom positions as numpy (n_atoms, 3) in nm.

    Returns:
        Energy in kJ/mol as a float.
    """
    # Build a minimal system with only the target force
    iso_system = openmm.System()
    for i in range(system.getNumParticles()):
        iso_system.addParticle(system.getParticleMass(i))

    for i in range(system.getNumForces()):
        force = system.getForce(i)
        if isinstance(force, force_class):
            iso_system.addForce(copy.deepcopy(force))
            break

    integrator = openmm.VerletIntegrator(0.001 * unit.picosecond)
    context = openmm.Context(
        iso_system, integrator, openmm.Platform.getPlatformByName("CPU")
    )
    context.setPositions(positions_nm * unit.nanometer)

    state = context.getState(getEnergy=True)
    energy = state.getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole)

    del context
    return energy


def get_openmm_total_energy(system, positions_nm):
    """Compute total energy from the full system.

    Args:
        system: OpenMM System.
        positions_nm: Atom positions as numpy (n_atoms, 3) in nm.

    Returns:
        Energy in kJ/mol as a float.
    """
    integrator = openmm.VerletIntegrator(0.001 * unit.picosecond)
    context = openmm.Context(
        system, integrator, openmm.Platform.getPlatformByName("CPU")
    )
    context.setPositions(positions_nm * unit.nanometer)

    state = context.getState(getEnergy=True)
    energy = state.getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole)

    del context
    return energy


def get_openmm_forces(system, positions_nm):
    """Compute forces from the full system.

    Args:
        system: OpenMM System.
        positions_nm: Atom positions as numpy (n_atoms, 3) in nm.

    Returns:
        Forces as numpy (n_atoms, 3) in kJ/mol/nm.
    """
    integrator = openmm.VerletIntegrator(0.001 * unit.picosecond)
    context = openmm.Context(
        system, integrator, openmm.Platform.getPlatformByName("CPU")
    )
    context.setPositions(positions_nm * unit.nanometer)

    state = context.getState(getForces=True)
    forces = state.getForces(asNumpy=True).value_in_unit(
        unit.kilojoule_per_mole / unit.nanometer
    )

    del context
    return forces.astype(np.float64)
