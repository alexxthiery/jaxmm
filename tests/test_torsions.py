"""Tests for torsion energy against OpenMM reference."""

import jax
import jax.numpy as jnp
import openmm

from jaxmm.energy import torsion_energy
from jaxmm.extract import TorsionParams
from tests.conftest import get_openmm_force_energy


def test_torsion_energy_initial(aldp_system, aldp_positions, aldp_positions_jnp, aldp_params):
    """Torsion energy at minimized positions matches OpenMM."""
    ref = get_openmm_force_energy(aldp_system, openmm.PeriodicTorsionForce, aldp_positions)
    jax_e = float(torsion_energy(aldp_positions_jnp, aldp_params.torsions))
    assert abs(jax_e - ref) < 1e-4, f"jaxmm={jax_e:.6f}, openmm={ref:.6f}"


def test_torsion_energy_md_frames(aldp_system, aldp_md_frames, aldp_params):
    """Torsion energy matches OpenMM across 10 MD frames."""
    for i in range(10):
        pos = aldp_md_frames[i]
        ref = get_openmm_force_energy(aldp_system, openmm.PeriodicTorsionForce, pos)
        jax_e = float(torsion_energy(jnp.array(pos), aldp_params.torsions))
        assert abs(jax_e - ref) < 1e-3, f"frame {i}: jaxmm={jax_e:.6f}, openmm={ref:.6f}"


def test_torsion_energy_grad(aldp_positions_jnp, aldp_params):
    """Gradient of torsion energy has no NaN or Inf."""
    grad = jax.grad(torsion_energy)(aldp_positions_jnp, aldp_params.torsions)
    assert not jnp.any(jnp.isnan(grad)), "NaN in torsion energy gradient"
    assert not jnp.any(jnp.isinf(grad)), "Inf in torsion energy gradient"


def test_torsion_energy_jit(aldp_positions_jnp, aldp_params):
    """JIT-compiled torsion energy matches non-jit."""
    ref = float(torsion_energy(aldp_positions_jnp, aldp_params.torsions))
    jit_e = float(jax.jit(torsion_energy)(aldp_positions_jnp, aldp_params.torsions))
    assert abs(jit_e - ref) < 1e-10, f"jit={jit_e}, nojit={ref}"


def test_torsion_energy_grad_collinear():
    """Gradient is finite when three atoms are nearly collinear.

    Collinear atoms make the cross product (and thus the normal vectors)
    near-zero. The atan2 formula should still produce finite gradients.
    Completes the pattern from bonds (near-zero dist) and angles (near-linear).
    """
    # 4 atoms: first three nearly collinear along x-axis, fourth off-axis
    positions = jnp.array([
        [0.0, 0.0, 0.0],
        [0.15, 1e-10, 0.0],   # nearly on x-axis
        [0.30, 0.0, 0.0],
        [0.45, 0.15, 0.0],
    ])
    params = TorsionParams(
        atom_i=jnp.array([0], dtype=jnp.int32),
        atom_j=jnp.array([1], dtype=jnp.int32),
        atom_k=jnp.array([2], dtype=jnp.int32),
        atom_l=jnp.array([3], dtype=jnp.int32),
        periodicity=jnp.array([2], dtype=jnp.int32),
        phase=jnp.array([0.0]),
        k=jnp.array([10.0]),
    )
    grad = jax.grad(torsion_energy)(positions, params)
    assert not jnp.any(jnp.isnan(grad)), "NaN in gradient at near-collinear configuration"
    assert not jnp.any(jnp.isinf(grad)), "Inf in gradient at near-collinear configuration"
