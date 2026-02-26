"""Tests for nonbonded energy against OpenMM reference."""

import jax
import jax.numpy as jnp
import openmm

from jaxmm.energy import nonbonded_energy
from tests.conftest import get_openmm_force_energy


def test_nonbonded_energy_initial(aldp_system, aldp_positions, aldp_positions_jnp, aldp_params):
    """Nonbonded energy at minimized positions matches OpenMM."""
    ref = get_openmm_force_energy(aldp_system, openmm.NonbondedForce, aldp_positions)
    jax_e = float(nonbonded_energy(aldp_positions_jnp, aldp_params.nonbonded))
    assert abs(jax_e - ref) < 1e-3, f"jaxmm={jax_e:.6f}, openmm={ref:.6f}"


def test_nonbonded_energy_md_frames(aldp_system, aldp_md_frames, aldp_params):
    """Nonbonded energy matches OpenMM across 10 MD frames."""
    for i in range(10):
        pos = aldp_md_frames[i]
        ref = get_openmm_force_energy(aldp_system, openmm.NonbondedForce, pos)
        jax_e = float(nonbonded_energy(jnp.array(pos), aldp_params.nonbonded))
        assert abs(jax_e - ref) < 1e-3, f"frame {i}: jaxmm={jax_e:.6f}, openmm={ref:.6f}"


def test_nonbonded_energy_grad(aldp_positions_jnp, aldp_params):
    """Gradient of nonbonded energy has no NaN or Inf."""
    grad = jax.grad(nonbonded_energy)(aldp_positions_jnp, aldp_params.nonbonded)
    assert not jnp.any(jnp.isnan(grad)), "NaN in nonbonded energy gradient"
    assert not jnp.any(jnp.isinf(grad)), "Inf in nonbonded energy gradient"


def test_nonbonded_energy_jit(aldp_positions_jnp, aldp_params):
    """JIT-compiled nonbonded energy matches non-jit."""
    ref = float(nonbonded_energy(aldp_positions_jnp, aldp_params.nonbonded))
    jit_e = float(jax.jit(nonbonded_energy)(aldp_positions_jnp, aldp_params.nonbonded))
    assert abs(jit_e - ref) < 1e-8, f"jit={jit_e}, nojit={ref}"


def test_nonbonded_energy_negative(aldp_positions_jnp, aldp_params):
    """Nonbonded energy should be negative for reasonable configurations (attraction dominates)."""
    e = float(nonbonded_energy(aldp_positions_jnp, aldp_params.nonbonded))
    assert e < 0, f"Expected negative nonbonded energy, got {e:.4f}"
