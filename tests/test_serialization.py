"""Tests for parameter serialization (save/load roundtrip)."""

import os
import tempfile

import jax.numpy as jnp

from jaxmm.utils import save_params, load_params
from jaxmm.energy import total_energy


def test_save_load_roundtrip_vacuum(aldp_positions_jnp, aldp_params):
    """Save and load vacuum params, energy is identical."""
    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "params.npz")
        save_params(aldp_params, path)
        loaded = load_params(path)

    e_orig = float(total_energy(aldp_positions_jnp, aldp_params))
    e_loaded = float(total_energy(aldp_positions_jnp, loaded))
    assert abs(e_orig - e_loaded) < 1e-10, f"orig={e_orig}, loaded={e_loaded}"
    assert loaded.n_atoms == aldp_params.n_atoms
    assert loaded.gbsa is None


def test_save_load_roundtrip_implicit(aldp_implicit_positions, aldp_implicit_params):
    """Save and load implicit params, energy is identical."""
    pos = jnp.array(aldp_implicit_positions)
    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "params.npz")
        save_params(aldp_implicit_params, path)
        loaded = load_params(path)

    e_orig = float(total_energy(pos, aldp_implicit_params))
    e_loaded = float(total_energy(pos, loaded))
    assert abs(e_orig - e_loaded) < 1e-10, f"orig={e_orig}, loaded={e_loaded}"
    assert loaded.gbsa is not None
    assert loaded.gbsa.solute_dielectric == aldp_implicit_params.gbsa.solute_dielectric
