"""Defensive-validation tests: malformed input must fail loudly, not silently.

JAX clamps out-of-bounds gathers rather than raising, so an index that overruns
`positions` produces a plausible, finite, wrong number. These tests pin the
boundary checks that turn that class of silent error into a ValueError.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import jaxmm
from jaxmm.extract import BondParams, AngleParams, TorsionParams


def _bonds(n_atoms_referenced):
    """BondParams whose largest atom index is n_atoms_referenced - 1."""
    top = n_atoms_referenced - 1
    return BondParams(
        atom_i=jnp.array([0, top], dtype=jnp.int32),
        atom_j=jnp.array([1, top - 1], dtype=jnp.int32),
        r0=jnp.array([0.1, 0.1]),
        k=jnp.array([1000.0, 1000.0]),
    )


# ---------------------------------------------------------------------------
# Out-of-range atom indices
# ---------------------------------------------------------------------------

def test_bond_energy_rejects_indices_beyond_positions():
    """A bond referencing atom 21 against 10 positions must raise, not clamp."""
    positions = jnp.zeros((10, 3))
    with pytest.raises(ValueError, match="atom 21"):
        jaxmm.bond_energy(positions, _bonds(22))


def test_angle_energy_rejects_indices_beyond_positions():
    """Same contract for angles."""
    positions = jnp.zeros((10, 3))
    params = AngleParams(
        atom_i=jnp.array([0], dtype=jnp.int32),
        atom_j=jnp.array([1], dtype=jnp.int32),
        atom_k=jnp.array([15], dtype=jnp.int32),
        theta0=jnp.array([1.9]),
        k=jnp.array([100.0]),
    )
    with pytest.raises(ValueError, match="atom 15"):
        jaxmm.angle_energy(positions, params)


def test_torsion_energy_rejects_indices_beyond_positions():
    """Same contract for periodic torsions."""
    positions = jnp.zeros((6, 3))
    params = TorsionParams(
        atom_i=jnp.array([0], dtype=jnp.int32),
        atom_j=jnp.array([1], dtype=jnp.int32),
        atom_k=jnp.array([2], dtype=jnp.int32),
        atom_l=jnp.array([9], dtype=jnp.int32),
        periodicity=jnp.array([2], dtype=jnp.int32),
        phase=jnp.array([0.0]),
        k=jnp.array([5.0]),
    )
    with pytest.raises(ValueError, match="atom 9"):
        jaxmm.torsion_energy(positions, params)


def test_in_range_indices_are_accepted():
    """The check must not fire on valid input."""
    positions = jnp.zeros((22, 3)).at[:, 0].set(jnp.linspace(0.0, 2.1, 22))
    assert jnp.isfinite(jaxmm.bond_energy(positions, _bonds(22)))


# ---------------------------------------------------------------------------
# Per-atom parameter arrays: the count is known statically
# ---------------------------------------------------------------------------

def test_nonbonded_energy_rejects_wrong_atom_count(aldp_params):
    """NonbondedParams carries n_atoms as static metadata; use it."""
    with pytest.raises(ValueError, match="22"):
        jaxmm.nonbonded_energy(jnp.zeros((10, 3)), aldp_params.nonbonded)


def test_gbsa_energy_rejects_wrong_atom_count(aldp_implicit_params):
    """GBSAParams has one radius per atom; a mismatch is a static error."""
    with pytest.raises(ValueError, match="22"):
        jaxmm.gbsa_energy(jnp.zeros((10, 3)), aldp_implicit_params.gbsa)


# ---------------------------------------------------------------------------
# Position shape
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("bad", [(22, 2), (22, 4), (22,)], ids=["2d-coords", "4d-coords", "1d"])
def test_energy_rejects_malformed_position_shape(bad):
    """positions must be (..., n_atoms, 3)."""
    with pytest.raises(ValueError):
        jaxmm.bond_energy(jnp.zeros(bad), _bonds(22))


# ---------------------------------------------------------------------------
# The check must not break the compiled path
# ---------------------------------------------------------------------------

def test_validation_does_not_break_jit_or_vmap(aldp_positions_jnp, aldp_params):
    """Index values are tracers under jit and vmap, so the bound check is skipped there.

    Compiled results must still match the eager ones. The tolerance is the
    project's documented 1e-8 for JIT consistency, not something tighter: XLA
    reorders floating-point work and shifts results by around 1e-12 relative.
    """
    eager = float(jaxmm.total_energy(aldp_positions_jnp, aldp_params))
    compiled = float(jax.jit(jaxmm.total_energy)(aldp_positions_jnp, aldp_params))
    np.testing.assert_allclose(compiled, eager, rtol=1e-8)

    batch = jnp.stack([aldp_positions_jnp, aldp_positions_jnp])
    batched = jax.vmap(jaxmm.total_energy, in_axes=(0, None))(batch, aldp_params)
    np.testing.assert_allclose(np.asarray(batched), [eager, eager], rtol=1e-8)


def test_gradients_still_flow_through_validated_functions(aldp_positions_jnp, aldp_params):
    """Validation is a host-side guard and must not perturb autodiff."""
    grad = jax.grad(jaxmm.total_energy)(aldp_positions_jnp, aldp_params)
    assert grad.shape == aldp_positions_jnp.shape
    assert jnp.all(jnp.isfinite(grad))


# ---------------------------------------------------------------------------
# float64 enforcement in utils.py
# ---------------------------------------------------------------------------

def test_dihedral_angle_requires_x64(monkeypatch):
    """dihedral_angle used to run silently at float32 while everything else raised."""
    positions = jnp.zeros((5, 3))
    indices = jnp.array([[0, 1, 2, 3]], dtype=jnp.int32)
    monkeypatch.setattr(type(jax.config), "jax_enable_x64", False)
    with pytest.raises(RuntimeError, match="float64"):
        jaxmm.dihedral_angle(positions, indices)
