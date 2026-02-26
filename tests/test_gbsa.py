"""Tests for GBSA/OBC2 implicit solvent energy."""

import copy

import jax
import jax.numpy as jnp
import numpy as np
import openmm
from openmm import unit

from jaxmm.energy import gbsa_energy, _born_radii, _gb_energy, _sa_energy, _distance_matrix, total_energy
from tests.conftest import get_openmm_force_energy, get_openmm_total_energy, get_openmm_forces


# ---------------------------------------------------------------------------
# Extraction tests
# ---------------------------------------------------------------------------

def test_gbsa_params_shapes(aldp_implicit_params):
    """GBSA params have expected shapes for ALDP (22 atoms)."""
    g = aldp_implicit_params.gbsa
    assert g is not None
    assert g.charges.shape == (22,)
    assert g.radii.shape == (22,)
    assert g.scale_factors.shape == (22,)


def test_gbsa_params_dtypes(aldp_implicit_params):
    """GBSA arrays are float64."""
    g = aldp_implicit_params.gbsa
    assert g.charges.dtype == jnp.float64
    assert g.radii.dtype == jnp.float64
    assert g.scale_factors.dtype == jnp.float64


def test_gbsa_global_params(aldp_implicit_params):
    """GBSA global parameters match OpenMM defaults."""
    g = aldp_implicit_params.gbsa
    assert g.solute_dielectric == 1.0
    assert g.solvent_dielectric == 78.5
    assert abs(g.probe_radius - 0.14) < 1e-10


def test_gbsa_charges_match_nonbonded(aldp_implicit_params):
    """GBSA charges should match NonbondedForce charges."""
    gb_q = aldp_implicit_params.gbsa.charges
    nb_q = aldp_implicit_params.nonbonded.charges
    assert jnp.allclose(gb_q, nb_q, atol=1e-10)


def test_gbsa_radii_range(aldp_implicit_params):
    """Atomic radii should be in a reasonable range (0.1 - 0.3 nm)."""
    r = aldp_implicit_params.gbsa.radii
    assert jnp.all(r > 0.05), f"Min radius = {float(jnp.min(r)):.4f} nm"
    assert jnp.all(r < 0.35), f"Max radius = {float(jnp.max(r)):.4f} nm"


def test_gbsa_scale_factors_range(aldp_implicit_params):
    """Scale factors should be positive and in typical range (0.5 - 1.2)."""
    sf = aldp_implicit_params.gbsa.scale_factors
    assert jnp.all(sf > 0.0)
    assert jnp.all(sf < 2.0)


def test_vacuum_has_no_gbsa(aldp_params):
    """Vacuum system should have gbsa=None."""
    assert aldp_params.gbsa is None


# ---------------------------------------------------------------------------
# Born radii reference implementation (loop-based, direct from OpenMM C++)
# ---------------------------------------------------------------------------

def _born_radii_reference(positions_np, params):
    """Pure-Python reference Born radii, matching OpenMM CustomGBForce expressions.

    Uses the same formula as the CustomGBForce: no "inside" correction,
    and tanh coefficients from params (alpha, beta, gamma).
    """
    n = positions_np.shape[0]
    dielectric_offset = 0.009

    radii = np.array(params.radii)
    scale_factors = np.array(params.scale_factors)
    offset_radii = radii - dielectric_offset

    born = np.zeros(n)
    for i in range(n):
        I_sum = 0.0
        for j in range(n):
            if i == j:
                continue
            r = np.linalg.norm(positions_np[i] - positions_np[j])
            sr_j = offset_radii[j] * scale_factors[j]
            r_scaled = r + sr_j

            # step(r + sr2 - or1): skip if or_i >= r_scaled
            if offset_radii[i] >= r_scaled:
                continue

            D = abs(r - sr_j)
            L = max(offset_radii[i], D)
            U = r_scaled

            # Matching: 0.5*(1/L - 1/U + 0.25*(r - sr^2/r)*(1/U^2 - 1/L^2) + 0.5*log(L/U)/r)
            term = 0.5 * (
                1.0 / L - 1.0 / U
                + 0.25 * (r - sr_j**2 / r) * (1.0 / U**2 - 1.0 / L**2)
                + 0.5 * np.log(L / U) / r
            )
            I_sum += term

        psi = I_sum * offset_radii[i]
        tanh_val = np.tanh(
            params.alpha * psi + params.beta * psi**2 + params.gamma * psi**3
        )
        born[i] = 1.0 / (1.0 / offset_radii[i] - tanh_val / radii[i])

    return born


def test_born_radii_vs_reference(aldp_implicit_positions, aldp_implicit_params):
    """JAX Born radii match loop-based reference on minimized positions."""
    pos_jnp = jnp.array(aldp_implicit_positions)
    dist, _ = _distance_matrix(pos_jnp)
    jax_br = np.array(_born_radii(pos_jnp, aldp_implicit_params.gbsa, dist))
    ref_br = _born_radii_reference(aldp_implicit_positions, aldp_implicit_params.gbsa)
    max_err = np.max(np.abs(jax_br - ref_br))
    assert max_err < 1e-10, f"max Born radii error = {max_err}"


def test_born_radii_vs_reference_md_frames(aldp_implicit_md_frames, aldp_implicit_params):
    """JAX Born radii match reference across 10 MD frames."""
    for i in range(10):
        pos = aldp_implicit_md_frames[i]
        pos_j = jnp.array(pos)
        dist, _ = _distance_matrix(pos_j)
        jax_br = np.array(_born_radii(pos_j, aldp_implicit_params.gbsa, dist))
        ref_br = _born_radii_reference(pos, aldp_implicit_params.gbsa)
        max_err = np.max(np.abs(jax_br - ref_br))
        assert max_err < 1e-10, f"frame {i}: max Born radii error = {max_err}"


# ---------------------------------------------------------------------------
# Helper: create isolated GBSA system for OpenMM comparison
# ---------------------------------------------------------------------------

def _make_gbsa_only_system(system):
    """Create a system with only the GB force (for isolated energy comparison).

    Handles both GBSAOBCForce and CustomGBForce (openmmtools uses the latter).
    """
    iso = openmm.System()
    for i in range(system.getNumParticles()):
        iso.addParticle(system.getParticleMass(i))
    for i in range(system.getNumForces()):
        force = system.getForce(i)
        if isinstance(force, (openmm.GBSAOBCForce, openmm.CustomGBForce)):
            iso.addForce(copy.deepcopy(force))
            break
    return iso


def _get_gbsa_energy_openmm(system, positions_nm):
    """Compute GBSA energy from isolated GBSAOBCForce."""
    iso = _make_gbsa_only_system(system)
    integrator = openmm.VerletIntegrator(0.001 * unit.picosecond)
    ctx = openmm.Context(iso, integrator, openmm.Platform.getPlatformByName("CPU"))
    ctx.setPositions(positions_nm * unit.nanometer)
    state = ctx.getState(getEnergy=True)
    energy = state.getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole)
    del ctx
    return energy


# ---------------------------------------------------------------------------
# GBSA energy tests
# ---------------------------------------------------------------------------

def test_gbsa_energy_initial(aldp_implicit_system, aldp_implicit_positions, aldp_implicit_params):
    """GBSA energy at minimized positions matches isolated OpenMM GBSAOBCForce."""
    ref = _get_gbsa_energy_openmm(aldp_implicit_system, aldp_implicit_positions)
    pos_jnp = jnp.array(aldp_implicit_positions)
    jax_e = float(gbsa_energy(pos_jnp, aldp_implicit_params.gbsa))
    assert abs(jax_e - ref) < 1e-3, f"jaxmm={jax_e:.6f}, openmm={ref:.6f}"


def test_gbsa_energy_md_frames(aldp_implicit_system, aldp_implicit_md_frames, aldp_implicit_params):
    """GBSA energy matches OpenMM across 50 MD frames."""
    for i in range(50):
        pos = aldp_implicit_md_frames[i]
        ref = _get_gbsa_energy_openmm(aldp_implicit_system, pos)
        jax_e = float(gbsa_energy(jnp.array(pos), aldp_implicit_params.gbsa))
        assert abs(jax_e - ref) < 1e-3, (
            f"frame {i}: jaxmm={jax_e:.6f}, openmm={ref:.6f}, diff={abs(jax_e-ref):.6f}"
        )


# ---------------------------------------------------------------------------
# Total energy (implicit system)
# ---------------------------------------------------------------------------

def test_total_energy_implicit_initial(
    aldp_implicit_system, aldp_implicit_positions, aldp_implicit_params
):
    """Total energy with GBSA matches full OpenMM implicit system."""
    ref = get_openmm_total_energy(aldp_implicit_system, aldp_implicit_positions)
    jax_e = float(total_energy(jnp.array(aldp_implicit_positions), aldp_implicit_params))
    assert abs(jax_e - ref) < 1e-3, f"jaxmm={jax_e:.6f}, openmm={ref:.6f}"


def test_total_energy_implicit_md_frames(
    aldp_implicit_system, aldp_implicit_md_frames, aldp_implicit_params
):
    """Total energy matches OpenMM across 50 MD frames for implicit system."""
    for i in range(50):
        pos = aldp_implicit_md_frames[i]
        ref = get_openmm_total_energy(aldp_implicit_system, pos)
        jax_e = float(total_energy(jnp.array(pos), aldp_implicit_params))
        assert abs(jax_e - ref) < 1e-3, (
            f"frame {i}: jaxmm={jax_e:.6f}, openmm={ref:.6f}"
        )


# ---------------------------------------------------------------------------
# Gradient tests
# ---------------------------------------------------------------------------

def test_grad_gbsa_vs_openmm(
    aldp_implicit_system, aldp_implicit_md_frames, aldp_implicit_params
):
    """JAX gradient matches OpenMM forces for implicit system across 10 frames."""
    grad_fn = jax.grad(total_energy)

    for i in range(10):
        pos = aldp_implicit_md_frames[i]
        openmm_forces = get_openmm_forces(aldp_implicit_system, pos)
        jax_grad = np.array(grad_fn(jnp.array(pos), aldp_implicit_params))

        # jax_grad = dE/dx, openmm_forces = -dE/dx
        max_err = np.max(np.abs(jax_grad + openmm_forces))
        assert max_err < 1e-2, (
            f"frame {i}: max gradient error = {max_err:.6f} kJ/mol/nm"
        )


def test_grad_gbsa_vs_finite_diff(aldp_implicit_positions, aldp_implicit_params):
    """GBSA gradient matches central finite differences."""
    pos_jnp = jnp.array(aldp_implicit_positions)
    grad_fn = jax.grad(gbsa_energy)
    jax_grad = np.array(grad_fn(pos_jnp, aldp_implicit_params.gbsa))

    positions = np.array(pos_jnp)
    h = 1e-5
    fd_grad = np.zeros_like(positions)

    for i in range(positions.shape[0]):
        for j in range(3):
            pos_plus = positions.copy()
            pos_minus = positions.copy()
            pos_plus[i, j] += h
            pos_minus[i, j] -= h
            e_plus = float(gbsa_energy(jnp.array(pos_plus), aldp_implicit_params.gbsa))
            e_minus = float(gbsa_energy(jnp.array(pos_minus), aldp_implicit_params.gbsa))
            fd_grad[i, j] = (e_plus - e_minus) / (2 * h)

    max_err = np.max(np.abs(jax_grad - fd_grad))
    assert max_err < 1e-3, f"max finite diff error = {max_err:.6f}"


def test_grad_gbsa_no_nan(aldp_implicit_md_frames, aldp_implicit_params):
    """GBSA gradient has no NaN or Inf across 50 MD frames."""
    grad_fn = jax.jit(jax.grad(gbsa_energy))

    for i in range(50):
        grad = grad_fn(jnp.array(aldp_implicit_md_frames[i]), aldp_implicit_params.gbsa)
        assert not jnp.any(jnp.isnan(grad)), f"NaN in gradient at frame {i}"
        assert not jnp.any(jnp.isinf(grad)), f"Inf in gradient at frame {i}"


# ---------------------------------------------------------------------------
# JIT and vmap
# ---------------------------------------------------------------------------

def test_gbsa_jit(aldp_implicit_positions, aldp_implicit_params):
    """JIT-compiled GBSA energy matches non-jit."""
    pos_jnp = jnp.array(aldp_implicit_positions)
    ref = float(gbsa_energy(pos_jnp, aldp_implicit_params.gbsa))
    jit_e = float(jax.jit(gbsa_energy)(pos_jnp, aldp_implicit_params.gbsa))
    assert abs(jit_e - ref) < 1e-10, f"jit={jit_e}, nojit={ref}"


def test_gbsa_vmap(aldp_implicit_system, aldp_implicit_md_frames, aldp_implicit_params):
    """vmap over batch of 50 configs matches individual evaluations."""
    batch = jnp.array(aldp_implicit_md_frames)
    vmap_energies = jax.vmap(gbsa_energy, in_axes=(0, None))(
        batch, aldp_implicit_params.gbsa
    )

    for i in range(50):
        ref = _get_gbsa_energy_openmm(aldp_implicit_system, aldp_implicit_md_frames[i])
        assert abs(float(vmap_energies[i]) - ref) < 1e-3, (
            f"frame {i}: vmap={float(vmap_energies[i]):.6f}, openmm={ref:.6f}"
        )


def test_total_energy_implicit_vmap(
    aldp_implicit_system, aldp_implicit_md_frames, aldp_implicit_params
):
    """vmap total_energy with GBSA matches OpenMM across 50 frames."""
    batch = jnp.array(aldp_implicit_md_frames)
    vmap_energies = jax.vmap(total_energy, in_axes=(0, None))(batch, aldp_implicit_params)

    for i in range(50):
        ref = get_openmm_total_energy(aldp_implicit_system, aldp_implicit_md_frames[i])
        assert abs(float(vmap_energies[i]) - ref) < 1e-3, (
            f"frame {i}: vmap={float(vmap_energies[i]):.6f}, openmm={ref:.6f}"
        )
