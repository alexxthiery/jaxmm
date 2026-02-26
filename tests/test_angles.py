"""Tests for angle energy against OpenMM reference."""

import jax
import jax.numpy as jnp
import openmm

from jaxmm.energy import angle_energy
from tests.conftest import get_openmm_force_energy


def test_angle_energy_initial(aldp_system, aldp_positions, aldp_positions_jnp, aldp_params):
    """Angle energy at minimized positions matches OpenMM."""
    ref = get_openmm_force_energy(aldp_system, openmm.HarmonicAngleForce, aldp_positions)
    jax_e = float(angle_energy(aldp_positions_jnp, aldp_params.angles))
    assert abs(jax_e - ref) < 1e-4, f"jaxmm={jax_e:.6f}, openmm={ref:.6f}"


def test_angle_energy_md_frames(aldp_system, aldp_md_frames, aldp_params):
    """Angle energy matches OpenMM across 10 MD frames."""
    for i in range(10):
        pos = aldp_md_frames[i]
        ref = get_openmm_force_energy(aldp_system, openmm.HarmonicAngleForce, pos)
        jax_e = float(angle_energy(jnp.array(pos), aldp_params.angles))
        assert abs(jax_e - ref) < 1e-4, f"frame {i}: jaxmm={jax_e:.6f}, openmm={ref:.6f}"


def test_angle_energy_grad(aldp_positions_jnp, aldp_params):
    """Gradient of angle energy has no NaN or Inf."""
    grad = jax.grad(angle_energy)(aldp_positions_jnp, aldp_params.angles)
    assert not jnp.any(jnp.isnan(grad)), "NaN in angle energy gradient"
    assert not jnp.any(jnp.isinf(grad)), "Inf in angle energy gradient"


def test_angle_energy_jit(aldp_positions_jnp, aldp_params):
    """JIT-compiled angle energy matches non-jit."""
    ref = float(angle_energy(aldp_positions_jnp, aldp_params.angles))
    jit_e = float(jax.jit(angle_energy)(aldp_positions_jnp, aldp_params.angles))
    assert abs(jit_e - ref) < 1e-10, f"jit={jit_e}, nojit={ref}"


def test_angle_energy_grad_near_linear():
    """Gradient at near-linear angle is finite (atan2 is stable near pi)."""
    from jaxmm.extract import AngleParams

    params = AngleParams(
        atom_i=jnp.array([0], dtype=jnp.int32),
        atom_j=jnp.array([1], dtype=jnp.int32),
        atom_k=jnp.array([2], dtype=jnp.int32),
        theta0=jnp.array([jnp.pi]),
        k=jnp.array([500.0]),
    )
    # Nearly linear configuration: atoms along x-axis
    positions = jnp.array([[0.0, 0.0, 0.0], [0.1, 0.0, 0.0], [0.2, 1e-15, 0.0]])
    grad = jax.grad(angle_energy)(positions, params)
    assert not jnp.any(jnp.isnan(grad)), "NaN in gradient at near-linear angle"
    assert not jnp.any(jnp.isinf(grad)), "Inf in gradient at near-linear angle"
