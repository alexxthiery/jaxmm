"""Tests for harmonic position restraints."""
import jax
import jax.numpy as jnp
import pytest
from dataclasses import replace

from jaxmm.extract import make_restraints
from jaxmm.energy import restraint_energy, total_energy, energy_components


def test_restraint_energy_at_reference(aldp_params, aldp_positions_jnp):
    """Energy is zero when positions match reference."""
    n_atoms = aldp_params.n_atoms
    restraints = make_restraints(
        jnp.arange(n_atoms), aldp_positions_jnp, k=1000.0)
    e = float(restraint_energy(aldp_positions_jnp, restraints))
    assert abs(e) < 1e-10


def test_restraint_energy_displaced(aldp_params, aldp_positions_jnp):
    """Known displacement gives expected energy: 0.5 * k * d^2 * n_atoms * 3."""
    n_atoms = aldp_params.n_atoms
    k = 100.0
    d = 0.01  # nm displacement
    ref = aldp_positions_jnp
    displaced = ref + d
    restraints = make_restraints(jnp.arange(n_atoms), ref, k=k)
    e = float(restraint_energy(displaced, restraints))
    expected = 0.5 * k * d**2 * n_atoms * 3
    assert abs(e - expected) < 1e-6


def test_restraint_energy_grad(aldp_params, aldp_positions_jnp):
    """Gradient matches analytical: k * (x - x_ref)."""
    ref = aldp_positions_jnp
    displaced = ref + 0.01
    restraints = make_restraints(jnp.arange(aldp_params.n_atoms), ref, k=100.0)
    grad = jax.grad(restraint_energy)(displaced, restraints)
    assert not jnp.any(jnp.isnan(grad))
    expected_grad = 100.0 * (displaced - ref)
    assert jnp.allclose(grad, expected_grad, atol=1e-8)


def test_restraint_in_total_energy(aldp_params, aldp_positions_jnp):
    """Restraints included in total_energy when set."""
    restraints = make_restraints(jnp.arange(3), aldp_positions_jnp[:3] + 0.01, k=100.0)
    params_r = replace(aldp_params, restraints=restraints)
    e_total = float(total_energy(aldp_positions_jnp, params_r))
    e_base = float(total_energy(aldp_positions_jnp, aldp_params))
    e_restr = float(restraint_energy(aldp_positions_jnp, restraints))
    assert abs(e_total - (e_base + e_restr)) < 1e-10


def test_restraint_in_energy_components(aldp_params, aldp_positions_jnp):
    """'restraints' key appears in energy_components when set."""
    restraints = make_restraints(jnp.arange(3), aldp_positions_jnp[:3] + 0.01, k=100.0)
    params_r = replace(aldp_params, restraints=restraints)
    comps = energy_components(aldp_positions_jnp, params_r)
    assert "restraints" in comps


def test_restraint_selective_atoms(aldp_positions_jnp):
    """Only selected atoms contribute to restraint energy."""
    k = 1000.0
    ref = aldp_positions_jnp
    displaced = ref.at[0].set(ref[0] + 0.1)  # displace only atom 0
    # Restrain only atom 0
    r1 = make_restraints(jnp.array([0]), ref[:1], k=k)
    e1 = float(restraint_energy(displaced, r1))
    assert e1 > 0
    # Restrain only atom 5 (not displaced)
    r5 = make_restraints(jnp.array([5]), ref[5:6], k=k)
    e5 = float(restraint_energy(displaced, r5))
    assert abs(e5) < 1e-10


def test_restraint_jit(aldp_params, aldp_positions_jnp):
    restraints = make_restraints(jnp.arange(5), aldp_positions_jnp[:5] + 0.01, k=100.0)
    ref = float(restraint_energy(aldp_positions_jnp, restraints))
    jit_e = float(jax.jit(restraint_energy)(aldp_positions_jnp, restraints))
    assert abs(jit_e - ref) < 1e-10
