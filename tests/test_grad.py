"""Tests for gradient validation against OpenMM forces and finite differences."""

import jax
import jax.numpy as jnp
import numpy as np

from jaxmm.energy import total_energy, bond_energy, angle_energy, torsion_energy, nonbonded_energy
from tests.conftest import get_openmm_forces


# ---------------------------------------------------------------------------
# Helper: central finite differences for any energy function
# ---------------------------------------------------------------------------

def _finite_diff_grad(energy_fn, positions, params, h=1e-5):
    """Central finite difference gradient for an energy function."""
    pos_np = np.array(positions)
    fd = np.zeros_like(pos_np)
    for i in range(pos_np.shape[0]):
        for j in range(3):
            p_plus = pos_np.copy()
            p_minus = pos_np.copy()
            p_plus[i, j] += h
            p_minus[i, j] -= h
            fd[i, j] = (
                float(energy_fn(jnp.array(p_plus), params))
                - float(energy_fn(jnp.array(p_minus), params))
            ) / (2 * h)
    return fd


def test_grad_vs_openmm_forces(aldp_system, aldp_md_frames, aldp_params):
    """JAX gradient matches OpenMM forces across 10 MD frames.

    OpenMM forces = -dE/dx, so jax.grad(E) should equal -forces.
    """
    grad_fn = jax.grad(total_energy)

    for i in range(10):
        pos = aldp_md_frames[i]
        openmm_forces = get_openmm_forces(aldp_system, pos)
        jax_grad = np.array(grad_fn(jnp.array(pos), aldp_params))

        # jax_grad = dE/dx, openmm_forces = -dE/dx
        # Tolerance is 1e-2 because OpenMM includes CMMotionRemover force
        # which adds a small constant correction we don't model
        max_err = np.max(np.abs(jax_grad + openmm_forces))
        assert max_err < 1e-2, (
            f"frame {i}: max gradient error = {max_err:.6f} kJ/mol/nm"
        )


def test_grad_vs_finite_diff(aldp_positions_jnp, aldp_params):
    """JAX gradient matches finite differences (independent reference).

    Uses central differences: dE/dx_i ~ (E(x+h) - E(x-h)) / (2h)
    """
    grad_fn = jax.grad(total_energy)
    jax_grad = np.array(grad_fn(aldp_positions_jnp, aldp_params))

    positions = np.array(aldp_positions_jnp)
    h = 1e-5  # nm
    fd_grad = np.zeros_like(positions)

    for i in range(positions.shape[0]):
        for j in range(3):
            pos_plus = positions.copy()
            pos_minus = positions.copy()
            pos_plus[i, j] += h
            pos_minus[i, j] -= h
            e_plus = float(total_energy(jnp.array(pos_plus), aldp_params))
            e_minus = float(total_energy(jnp.array(pos_minus), aldp_params))
            fd_grad[i, j] = (e_plus - e_minus) / (2 * h)

    max_err = np.max(np.abs(jax_grad - fd_grad))
    assert max_err < 1e-3, f"max finite diff error = {max_err:.6f}"


def test_grad_no_nan(aldp_md_frames, aldp_params):
    """Gradient has no NaN or Inf across all 50 MD frames."""
    grad_fn = jax.jit(jax.grad(total_energy))

    for i in range(50):
        grad = grad_fn(jnp.array(aldp_md_frames[i]), aldp_params)
        assert not jnp.any(jnp.isnan(grad)), f"NaN in gradient at frame {i}"
        assert not jnp.any(jnp.isinf(grad)), f"Inf in gradient at frame {i}"


# ---------------------------------------------------------------------------
# Per-term finite difference gradient tests
# ---------------------------------------------------------------------------

def test_grad_bond_vs_finite_diff(aldp_positions_jnp, aldp_params):
    """Bond energy gradient matches central finite differences."""
    jax_grad = np.array(jax.grad(bond_energy)(aldp_positions_jnp, aldp_params.bonds))
    fd_grad = _finite_diff_grad(bond_energy, aldp_positions_jnp, aldp_params.bonds)
    max_err = np.max(np.abs(jax_grad - fd_grad))
    assert max_err < 1e-3, f"bond grad finite diff error = {max_err:.6f}"


def test_grad_angle_vs_finite_diff(aldp_positions_jnp, aldp_params):
    """Angle energy gradient matches central finite differences."""
    jax_grad = np.array(jax.grad(angle_energy)(aldp_positions_jnp, aldp_params.angles))
    fd_grad = _finite_diff_grad(angle_energy, aldp_positions_jnp, aldp_params.angles)
    max_err = np.max(np.abs(jax_grad - fd_grad))
    assert max_err < 1e-3, f"angle grad finite diff error = {max_err:.6f}"


def test_grad_torsion_vs_finite_diff(aldp_positions_jnp, aldp_params):
    """Torsion energy gradient matches central finite differences."""
    jax_grad = np.array(jax.grad(torsion_energy)(aldp_positions_jnp, aldp_params.torsions))
    fd_grad = _finite_diff_grad(torsion_energy, aldp_positions_jnp, aldp_params.torsions)
    max_err = np.max(np.abs(jax_grad - fd_grad))
    assert max_err < 1e-3, f"torsion grad finite diff error = {max_err:.6f}"


def test_grad_nonbonded_vs_finite_diff(aldp_positions_jnp, aldp_params):
    """Nonbonded energy gradient matches central finite differences."""
    jax_grad = np.array(jax.grad(nonbonded_energy)(aldp_positions_jnp, aldp_params.nonbonded))
    fd_grad = _finite_diff_grad(nonbonded_energy, aldp_positions_jnp, aldp_params.nonbonded)
    max_err = np.max(np.abs(jax_grad - fd_grad))
    assert max_err < 1e-3, f"nonbonded grad finite diff error = {max_err:.6f}"


# ---------------------------------------------------------------------------
# Hessian (second derivatives)
# ---------------------------------------------------------------------------

def test_hessian_diatomic_finite_and_symmetric():
    """Hessian of total_energy on a diatomic is finite and symmetric.

    Uses a minimal 2-atom harmonic oscillator. The Hessian should be
    a (6, 6) matrix (2 atoms x 3 coords flattened), symmetric, and finite.
    This verifies JAX can compute second derivatives through the energy code.
    """
    from jaxmm.extract import (
        ForceFieldParams, BondParams, AngleParams, TorsionParams, NonbondedParams,
    )

    bonds = BondParams(
        atom_i=jnp.array([0], dtype=jnp.int32),
        atom_j=jnp.array([1], dtype=jnp.int32),
        r0=jnp.array([0.15]),
        k=jnp.array([200000.0]),
    )
    angles = AngleParams(
        atom_i=jnp.array([], dtype=jnp.int32),
        atom_j=jnp.array([], dtype=jnp.int32),
        atom_k=jnp.array([], dtype=jnp.int32),
        theta0=jnp.array([]),
        k=jnp.array([]),
    )
    torsions = TorsionParams(
        atom_i=jnp.array([], dtype=jnp.int32),
        atom_j=jnp.array([], dtype=jnp.int32),
        atom_k=jnp.array([], dtype=jnp.int32),
        atom_l=jnp.array([], dtype=jnp.int32),
        periodicity=jnp.array([], dtype=jnp.int32),
        phase=jnp.array([]),
        k=jnp.array([]),
    )
    nonbonded = NonbondedParams(
        charges=jnp.zeros(2),
        sigmas=jnp.ones(2) * 0.3,
        epsilons=jnp.zeros(2),
        n_atoms=2,
        exclusion_pairs=jnp.array([[0, 1]], dtype=jnp.int32),
        exception_pairs=jnp.empty((0, 2), dtype=jnp.int32),
        exception_chargeprod=jnp.empty(0),
        exception_sigma=jnp.empty(0),
        exception_epsilon=jnp.empty(0),
    )
    params = ForceFieldParams(
        bonds=bonds, angles=angles, torsions=torsions,
        nonbonded=nonbonded, masses=jnp.array([12.0, 12.0]), n_atoms=2,
    )
    positions = jnp.array([[0.0, 0.0, 0.0], [0.15, 0.0, 0.0]])

    def energy_flat(pos_flat):
        return total_energy(pos_flat.reshape(2, 3), params)

    H = jax.hessian(energy_flat)(positions.ravel())

    # Shape: (6, 6)
    assert H.shape == (6, 6), f"Hessian shape = {H.shape}, expected (6, 6)"
    # Finite
    assert not jnp.any(jnp.isnan(H)), "NaN in Hessian"
    assert not jnp.any(jnp.isinf(H)), "Inf in Hessian"
    # Symmetric (Schwarz's theorem)
    max_asym = float(jnp.max(jnp.abs(H - H.T)))
    assert max_asym < 1e-6, f"Hessian not symmetric: max asymmetry = {max_asym}"
