"""Tests for bond energy against OpenMM reference."""

import jax
import jax.numpy as jnp
import numpy as np
import openmm

from jaxmm.energy import bond_energy
from tests.conftest import get_openmm_force_energy


def test_bond_energy_initial(aldp_system, aldp_positions, aldp_positions_jnp, aldp_params):
    """Bond energy at minimized positions matches OpenMM."""
    ref = get_openmm_force_energy(aldp_system, openmm.HarmonicBondForce, aldp_positions)
    jax_e = float(bond_energy(aldp_positions_jnp, aldp_params.bonds))
    assert abs(jax_e - ref) < 1e-4, f"jaxmm={jax_e:.6f}, openmm={ref:.6f}"


def test_bond_energy_md_frames(aldp_system, aldp_md_frames, aldp_params):
    """Bond energy matches OpenMM across 10 MD frames."""
    for i in range(10):
        pos = aldp_md_frames[i]
        ref = get_openmm_force_energy(aldp_system, openmm.HarmonicBondForce, pos)
        jax_e = float(bond_energy(jnp.array(pos), aldp_params.bonds))
        assert abs(jax_e - ref) < 1e-4, f"frame {i}: jaxmm={jax_e:.6f}, openmm={ref:.6f}"


def test_bond_energy_grad(aldp_positions_jnp, aldp_params):
    """Gradient of bond energy has no NaN or Inf."""
    grad = jax.grad(bond_energy)(aldp_positions_jnp, aldp_params.bonds)
    assert not jnp.any(jnp.isnan(grad)), "NaN in bond energy gradient"
    assert not jnp.any(jnp.isinf(grad)), "Inf in bond energy gradient"


def test_bond_energy_jit(aldp_positions_jnp, aldp_params):
    """JIT-compiled bond energy matches non-jit."""
    ref = float(bond_energy(aldp_positions_jnp, aldp_params.bonds))
    jit_e = float(jax.jit(bond_energy)(aldp_positions_jnp, aldp_params.bonds))
    assert abs(jit_e - ref) < 1e-10, f"jit={jit_e}, nojit={ref}"


def test_bond_energy_grad_near_zero_distance():
    """Gradient at near-zero bond distance is finite (safe norm)."""
    from jaxmm.extract import BondParams

    params = BondParams(
        atom_i=jnp.array([0], dtype=jnp.int32),
        atom_j=jnp.array([1], dtype=jnp.int32),
        r0=jnp.array([0.15]),
        k=jnp.array([200000.0]),
    )
    # Two atoms at the same position (degenerate)
    positions = jnp.array([[0.1, 0.0, 0.0], [0.1, 0.0, 0.0]])
    grad = jax.grad(bond_energy)(positions, params)
    assert not jnp.any(jnp.isnan(grad)), "NaN in gradient at zero distance"
    assert not jnp.any(jnp.isinf(grad)), "Inf in gradient at zero distance"
