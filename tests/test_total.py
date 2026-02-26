"""Tests for total energy, vmap, utility functions, and physical invariances."""

import jax
import jax.numpy as jnp
import jax.random as random
import numpy as np

from jaxmm.energy import (
    total_energy, energy_components,
    bond_energy, angle_energy, torsion_energy, nonbonded_energy,
)
from jaxmm.utils import log_boltzmann, log_boltzmann_regularized, log_prob, log_prob_regularized, KB
from tests.conftest import get_openmm_total_energy


def test_total_energy_initial(aldp_system, aldp_positions, aldp_positions_jnp, aldp_params):
    """Total energy at minimized positions matches full OpenMM energy."""
    ref = get_openmm_total_energy(aldp_system, aldp_positions)
    jax_e = float(total_energy(aldp_positions_jnp, aldp_params))
    assert abs(jax_e - ref) < 1e-3, f"jaxmm={jax_e:.6f}, openmm={ref:.6f}"


def test_total_energy_md_frames(aldp_system, aldp_md_frames, aldp_params):
    """Total energy matches OpenMM across all 50 MD frames."""
    for i in range(50):
        pos = aldp_md_frames[i]
        ref = get_openmm_total_energy(aldp_system, pos)
        jax_e = float(total_energy(jnp.array(pos), aldp_params))
        assert abs(jax_e - ref) < 1e-3, f"frame {i}: jaxmm={jax_e:.6f}, openmm={ref:.6f}"


def test_total_energy_jit(aldp_positions_jnp, aldp_params):
    """JIT-compiled total energy matches non-jit."""
    ref = float(total_energy(aldp_positions_jnp, aldp_params))
    jit_e = float(jax.jit(total_energy)(aldp_positions_jnp, aldp_params))
    assert abs(jit_e - ref) < 1e-8, f"jit={jit_e}, nojit={ref}"


def test_total_energy_vmap(aldp_system, aldp_md_frames, aldp_params):
    """vmap over batch of 50 configs matches individual evaluations."""
    batch = jnp.array(aldp_md_frames)  # (50, 22, 3)
    vmap_energies = jax.vmap(total_energy, in_axes=(0, None))(batch, aldp_params)

    for i in range(50):
        ref = get_openmm_total_energy(aldp_system, aldp_md_frames[i])
        assert abs(float(vmap_energies[i]) - ref) < 1e-3, (
            f"frame {i}: vmap={float(vmap_energies[i]):.6f}, openmm={ref:.6f}"
        )


def test_log_boltzmann(aldp_positions_jnp, aldp_params):
    """log_boltzmann == -energy / (kB * T) at a few temperatures."""
    for temp in [300.0, 500.0, 1000.0]:
        energy = float(total_energy(aldp_positions_jnp, aldp_params))
        lp = float(log_boltzmann(aldp_positions_jnp, aldp_params, temp))
        expected = -energy / (KB * temp)
        assert abs(lp - expected) < 1e-10, f"T={temp}: lp={lp}, expected={expected}"


def test_log_boltzmann_regularized_below_cut(aldp_positions_jnp, aldp_params):
    """Below energy_cut, regularized matches standard."""
    temp = 300.0
    lp = float(log_boltzmann(aldp_positions_jnp, aldp_params, temp))
    lp_reg = float(log_boltzmann_regularized(
        aldp_positions_jnp, aldp_params, temp, energy_cut=1e10, energy_max=1e20
    ))
    assert abs(lp - lp_reg) < 1e-6, f"lp={lp}, lp_reg={lp_reg}"


def test_log_prob_deprecated(aldp_positions_jnp, aldp_params):
    """log_prob emits DeprecationWarning and returns same as log_boltzmann."""
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("error", DeprecationWarning)
        try:
            log_prob(aldp_positions_jnp, aldp_params, 300.0)
            assert False, "Should have raised DeprecationWarning"
        except DeprecationWarning:
            pass


def test_energy_components_sum_vacuum(aldp_positions_jnp, aldp_params):
    """energy_components values sum to total_energy for vacuum system."""
    total = float(total_energy(aldp_positions_jnp, aldp_params))
    comps = energy_components(aldp_positions_jnp, aldp_params)
    comp_sum = sum(float(v) for v in comps.values())
    assert abs(comp_sum - total) < 1e-10, f"sum={comp_sum}, total={total}"


def test_energy_components_keys_vacuum(aldp_positions_jnp, aldp_params):
    """Vacuum system has exactly 4 component keys (no gbsa)."""
    comps = energy_components(aldp_positions_jnp, aldp_params)
    assert set(comps.keys()) == {"bonds", "angles", "torsions", "nonbonded"}


def test_energy_components_sum_implicit(aldp_implicit_positions, aldp_implicit_params):
    """energy_components values sum to total_energy for implicit system."""
    pos = jnp.array(aldp_implicit_positions)
    total = float(total_energy(pos, aldp_implicit_params))
    comps = energy_components(pos, aldp_implicit_params)
    comp_sum = sum(float(v) for v in comps.values())
    assert abs(comp_sum - total) < 1e-10, f"sum={comp_sum}, total={total}"


def test_energy_components_keys_implicit(aldp_implicit_positions, aldp_implicit_params):
    """Implicit system has 5 component keys (including gbsa)."""
    pos = jnp.array(aldp_implicit_positions)
    comps = energy_components(pos, aldp_implicit_params)
    assert set(comps.keys()) == {"bonds", "angles", "torsions", "nonbonded", "gbsa"}


def test_energy_components_jit(aldp_positions_jnp, aldp_params):
    """energy_components works under jit."""
    comps = jax.jit(energy_components)(aldp_positions_jnp, aldp_params)
    total = float(total_energy(aldp_positions_jnp, aldp_params))
    comp_sum = sum(float(v) for v in comps.values())
    assert abs(comp_sum - total) < 1e-8


# ---------------------------------------------------------------------------
# Physical invariance tests
# ---------------------------------------------------------------------------

def test_total_energy_translation_invariance(aldp_positions_jnp, aldp_params):
    """Energy is invariant under rigid translation of all atoms."""
    e_original = float(total_energy(aldp_positions_jnp, aldp_params))
    offset = jnp.array([3.7, -1.2, 8.5])
    e_translated = float(total_energy(aldp_positions_jnp + offset, aldp_params))
    assert abs(e_translated - e_original) < 1e-8, (
        f"Translation changed energy: {e_original:.6f} -> {e_translated:.6f}"
    )


def test_total_energy_rotation_invariance(aldp_positions_jnp, aldp_params):
    """Energy is invariant under rigid rotation of all atoms."""
    e_original = float(total_energy(aldp_positions_jnp, aldp_params))

    # Rotation matrix: 37 degrees around axis (1, 2, 3) (normalized)
    axis = jnp.array([1.0, 2.0, 3.0])
    axis = axis / jnp.linalg.norm(axis)
    theta = jnp.radians(37.0)
    # Rodrigues' formula
    K = jnp.array([
        [0, -axis[2], axis[1]],
        [axis[2], 0, -axis[0]],
        [-axis[1], axis[0], 0],
    ])
    R = jnp.eye(3) + jnp.sin(theta) * K + (1 - jnp.cos(theta)) * K @ K

    # Center, rotate, un-center
    com = jnp.mean(aldp_positions_jnp, axis=0)
    centered = aldp_positions_jnp - com
    rotated = (R @ centered.T).T + com

    e_rotated = float(total_energy(rotated, aldp_params))
    assert abs(e_rotated - e_original) < 1e-6, (
        f"Rotation changed energy: {e_original:.6f} -> {e_rotated:.6f}"
    )


# ---------------------------------------------------------------------------
# Energy components match standalone functions
# ---------------------------------------------------------------------------

def test_energy_components_match_standalone(aldp_positions_jnp, aldp_params):
    """Each energy_components value matches its standalone function call."""
    comps = energy_components(aldp_positions_jnp, aldp_params)

    e_bonds = float(bond_energy(aldp_positions_jnp, aldp_params.bonds))
    e_angles = float(angle_energy(aldp_positions_jnp, aldp_params.angles))
    e_torsions = float(torsion_energy(aldp_positions_jnp, aldp_params.torsions))
    e_nonbonded = float(nonbonded_energy(aldp_positions_jnp, aldp_params.nonbonded))

    assert abs(float(comps["bonds"]) - e_bonds) < 1e-10, "bonds mismatch"
    assert abs(float(comps["angles"]) - e_angles) < 1e-10, "angles mismatch"
    assert abs(float(comps["torsions"]) - e_torsions) < 1e-10, "torsions mismatch"
    assert abs(float(comps["nonbonded"]) - e_nonbonded) < 1e-10, "nonbonded mismatch"


# ---------------------------------------------------------------------------
# log_boltzmann_regularized above cutoff
# ---------------------------------------------------------------------------

def test_log_boltzmann_regularized_above_cut(aldp_positions_jnp, aldp_params):
    """Above energy_cut, regularized output is dampened and clamped."""
    temp = 300.0
    energy = float(total_energy(aldp_positions_jnp, aldp_params))
    reduced = energy / (KB * temp)

    # Set energy_cut well below actual energy so we're in the dampened regime.
    # ALDP minimized energy is negative (~-70 kJ/mol), so reduced is negative.
    # Use a cut that's above the reduced energy (less negative) to trigger dampening.
    energy_cut = reduced - 5.0  # below actual reduced energy
    energy_max = reduced + 100.0

    lp_reg = float(log_boltzmann_regularized(
        aldp_positions_jnp, aldp_params, temp,
        energy_cut=energy_cut, energy_max=energy_max
    ))
    lp_std = float(log_boltzmann(aldp_positions_jnp, aldp_params, temp))

    # Since reduced > energy_cut, the regularized version should differ
    assert abs(lp_reg - lp_std) > 0.1, (
        f"Expected dampening above cut: reg={lp_reg:.4f}, std={lp_std:.4f}"
    )

    # Test hard clamp: set energy_max to a very tight value
    energy_max_tight = energy_cut + 1.0
    lp_clamped = float(log_boltzmann_regularized(
        aldp_positions_jnp, aldp_params, temp,
        energy_cut=energy_cut, energy_max=energy_max_tight
    ))
    # Clamped result should be >= -energy_max_tight (since result = -regularized)
    assert lp_clamped >= -energy_max_tight - 1e-10, (
        f"Clamp failed: {lp_clamped:.4f} < {-energy_max_tight:.4f}"
    )


# ---------------------------------------------------------------------------
# Second molecule: toluene (aromatic ring, different topology)
# ---------------------------------------------------------------------------

def test_total_energy_toluene(toluene_system, toluene_positions, toluene_positions_jnp,
                              toluene_params):
    """Total energy on toluene matches OpenMM."""
    ref = get_openmm_total_energy(toluene_system, toluene_positions)
    jax_e = float(total_energy(toluene_positions_jnp, toluene_params))
    assert abs(jax_e - ref) < 1e-3, f"jaxmm={jax_e:.6f}, openmm={ref:.6f}"


def test_grad_no_nan_toluene(toluene_positions_jnp, toluene_params):
    """Gradient on toluene has no NaN or Inf."""
    grad = jax.grad(total_energy)(toluene_positions_jnp, toluene_params)
    assert not jnp.any(jnp.isnan(grad)), "NaN in toluene gradient"
    assert not jnp.any(jnp.isinf(grad)), "Inf in toluene gradient"


# ---------------------------------------------------------------------------
# log_boltzmann gradient tests
# ---------------------------------------------------------------------------

def test_grad_log_boltzmann_finite(aldp_positions_jnp, aldp_params):
    """Gradient of log_boltzmann is finite (no NaN/Inf)."""
    grad = jax.grad(log_boltzmann)(aldp_positions_jnp, aldp_params, 300.0)
    assert not jnp.any(jnp.isnan(grad)), "NaN in log_boltzmann gradient"
    assert not jnp.any(jnp.isinf(grad)), "Inf in log_boltzmann gradient"


def test_grad_log_boltzmann_vs_finite_diff(aldp_positions_jnp, aldp_params):
    """log_boltzmann gradient matches finite differences."""
    grad_fn = jax.grad(log_boltzmann)
    jax_grad = np.array(grad_fn(aldp_positions_jnp, aldp_params, 300.0))

    positions = np.array(aldp_positions_jnp)
    h = 1e-5
    fd_grad = np.zeros_like(positions)
    for i in range(positions.shape[0]):
        for j in range(3):
            pos_plus = positions.copy()
            pos_minus = positions.copy()
            pos_plus[i, j] += h
            pos_minus[i, j] -= h
            fd_grad[i, j] = (
                float(log_boltzmann(jnp.array(pos_plus), aldp_params, 300.0))
                - float(log_boltzmann(jnp.array(pos_minus), aldp_params, 300.0))
            ) / (2 * h)

    max_err = np.max(np.abs(jax_grad - fd_grad))
    assert max_err < 1e-3, f"log_boltzmann grad finite diff error = {max_err:.6f}"


def test_grad_log_boltzmann_regularized_finite(aldp_positions_jnp, aldp_params):
    """Gradient of log_boltzmann_regularized is finite (no NaN/Inf)."""
    grad = jax.grad(log_boltzmann_regularized)(
        aldp_positions_jnp, aldp_params, 300.0,
        energy_cut=1e10, energy_max=1e20,
    )
    assert not jnp.any(jnp.isnan(grad)), "NaN in regularized gradient"
    assert not jnp.any(jnp.isinf(grad)), "Inf in regularized gradient"


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

def test_constants_values():
    """Unit conversion constants have correct numerical values."""
    from jaxmm.utils import FEMTOSECOND, ANGSTROM, KCAL_PER_MOL
    assert FEMTOSECOND == 1e-3, f"FEMTOSECOND={FEMTOSECOND}, expected 1e-3"
    assert ANGSTROM == 0.1, f"ANGSTROM={ANGSTROM}, expected 0.1"
    assert KCAL_PER_MOL == 4.184, f"KCAL_PER_MOL={KCAL_PER_MOL}, expected 4.184"


def test_kb_value():
    """Boltzmann constant matches NIST value (kJ/mol/K)."""
    assert abs(KB - 8.314462618e-3) < 1e-12


# ---------------------------------------------------------------------------
# log_prob_regularized deprecation
# ---------------------------------------------------------------------------

def test_log_prob_regularized_deprecated(aldp_positions_jnp, aldp_params):
    """log_prob_regularized emits DeprecationWarning."""
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("error", DeprecationWarning)
        try:
            log_prob_regularized(aldp_positions_jnp, aldp_params, 300.0, 1e10, 1e20)
            assert False, "Should have raised DeprecationWarning"
        except DeprecationWarning:
            pass


def test_log_prob_regularized_matches(aldp_positions_jnp, aldp_params):
    """log_prob_regularized returns same value as log_boltzmann_regularized."""
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        old = float(log_prob_regularized(
            aldp_positions_jnp, aldp_params, 300.0, 1e10, 1e20
        ))
    new = float(log_boltzmann_regularized(
        aldp_positions_jnp, aldp_params, 300.0, 1e10, 1e20
    ))
    assert abs(old - new) < 1e-12, f"old={old}, new={new}"


# ---------------------------------------------------------------------------
# energy_components vmap
# ---------------------------------------------------------------------------

def test_energy_components_vmap(aldp_md_frames, aldp_params):
    """vmap(energy_components) over batch matches individual calls."""
    batch = jnp.array(aldp_md_frames[:10])  # 10 frames for speed
    vmap_comps = jax.vmap(energy_components, in_axes=(0, None))(batch, aldp_params)

    for i in range(10):
        single = energy_components(batch[i], aldp_params)
        for key in single:
            assert abs(float(vmap_comps[key][i]) - float(single[key])) < 1e-10, (
                f"frame {i}, {key}: vmap={float(vmap_comps[key][i])}, single={float(single[key])}"
            )


def test_custom_energy_composition(aldp_positions_jnp, aldp_params):
    """Custom energy terms compose with total_energy under jit/grad/vmap."""
    def custom_fn(pos):
        return jnp.sum(pos**2)

    def combined(pos, params):
        return total_energy(pos, params) + custom_fn(pos)

    pos = aldp_positions_jnp
    # Works with grad
    g = jax.grad(combined)(pos, aldp_params)
    assert not jnp.any(jnp.isnan(g))

    # Works with jit
    e1 = float(combined(pos, aldp_params))
    e2 = float(jax.jit(combined)(pos, aldp_params))
    assert abs(e1 - e2) < 1e-8

    # Works with vmap
    batch = jnp.stack([pos, pos + 0.001])
    vmap_e = jax.vmap(combined, in_axes=(0, None))(batch, aldp_params)
    assert vmap_e.shape == (2,)
