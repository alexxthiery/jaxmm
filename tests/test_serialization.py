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


# ---------------------------------------------------------------------------
# Optional fields
#
# ForceFieldParams carries gbsa, rb_torsions, cmap, restraints and box as
# optional fields with their own save/load branches. Those branches had no
# coverage, and a silently dropped field loses physics without raising. Each
# test below perturbs the system so the term contributes a non-zero energy,
# otherwise dropping it would not change the total and the test would pass
# against a broken implementation.
#
# The system builders live next to the tests for their own energy term. pytest
# puts <repo>/tests on sys.path, so they import directly.
# ---------------------------------------------------------------------------

import dataclasses

import numpy as np

import jaxmm
from jaxmm.extract import extract_params, make_restraints
from jaxmm.energy import energy_components
from test_rb_torsions import _make_rb_system
from test_cmap import _make_cmap_system
from test_pbc import _make_periodic_lj_system


def _roundtrip(params):
    """Save and reload a ForceFieldParams through a temporary .npz."""
    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "params.npz")
        save_params(params, path)
        return load_params(path)


def test_save_load_roundtrip_rb_torsions():
    """RB torsion coefficients survive the round trip."""
    system, positions = _make_rb_system()
    params = extract_params(system)
    pos = jnp.asarray(positions)
    assert params.rb_torsions is not None
    assert abs(float(energy_components(pos, params)["rb_torsions"])) > 1e-6

    loaded = _roundtrip(params)

    assert loaded.rb_torsions is not None, "rb_torsions silently dropped"
    for name in ("c0", "c1", "c2", "c3", "c4", "c5"):
        np.testing.assert_allclose(
            getattr(loaded.rb_torsions, name), getattr(params.rb_torsions, name))
    np.testing.assert_allclose(
        float(total_energy(pos, loaded)), float(total_energy(pos, params)), rtol=1e-12)


def test_save_load_roundtrip_cmap():
    """The CMAP energy grid is a 3D array; check it survives shape and value."""
    system, positions = _make_cmap_system()
    params = extract_params(system)
    pos = jnp.asarray(positions)
    assert params.cmap is not None
    assert abs(float(energy_components(pos, params)["cmap"])) > 1e-6

    loaded = _roundtrip(params)

    assert loaded.cmap is not None, "cmap silently dropped"
    assert loaded.cmap.maps.shape == params.cmap.maps.shape
    np.testing.assert_allclose(loaded.cmap.maps, params.cmap.maps)
    assert loaded.cmap.map_size == params.cmap.map_size
    assert isinstance(loaded.cmap.map_size, int), "map_size must stay a static int"
    np.testing.assert_allclose(
        float(total_energy(pos, loaded)), float(total_energy(pos, params)), rtol=1e-12)


def test_save_load_roundtrip_restraints(aldp_positions_jnp, aldp_params):
    """Restraints survive, and the reference positions come back exactly.

    The restrained atoms are displaced from their reference so the term
    contributes a non-zero energy; with a zero contribution a dropped field
    would be invisible.
    """
    indices = jnp.array([0, 4, 8], dtype=jnp.int32)
    reference = aldp_positions_jnp[indices] + 0.02
    params = dataclasses.replace(
        aldp_params, restraints=make_restraints(indices, reference, 500.0))
    assert float(energy_components(aldp_positions_jnp, params)["restraints"]) > 1e-6

    loaded = _roundtrip(params)

    assert loaded.restraints is not None, "restraints silently dropped"
    np.testing.assert_allclose(
        loaded.restraints.reference_positions, params.restraints.reference_positions)
    np.testing.assert_allclose(loaded.restraints.k, params.restraints.k)
    np.testing.assert_allclose(
        float(total_energy(aldp_positions_jnp, loaded)),
        float(total_energy(aldp_positions_jnp, params)), rtol=1e-12)


def test_save_load_roundtrip_periodic_box():
    """The box vector survives, so the minimum-image convention still applies."""
    system, positions = _make_periodic_lj_system()
    params = extract_params(system)
    pos = jnp.asarray(positions)
    assert params.box is not None

    loaded = _roundtrip(params)

    assert loaded.box is not None, "box silently dropped"
    np.testing.assert_allclose(loaded.box, params.box)
    assert loaded.nonbonded.cutoff == params.nonbonded.cutoff
    np.testing.assert_allclose(
        float(total_energy(pos, loaded)), float(total_energy(pos, params)), rtol=1e-12)


def test_save_load_preserves_absent_optional_fields(aldp_params):
    """Absent optional fields stay absent rather than materializing as empties."""
    loaded = _roundtrip(aldp_params)
    assert loaded.gbsa is None
    assert loaded.rb_torsions is None
    assert loaded.cmap is None
    assert loaded.restraints is None
    assert loaded.box is None


def test_optional_param_containers_roundtrip_as_pytrees(aldp_implicit_params,
                                                        aldp_positions_jnp):
    """Every exported parameter container flattens and unflattens losslessly.

    RBTorsionParams, CmapParams, RestraintParams and GBSAParams are public
    exports that no test referenced, so a missing _register_pytree call, or an
    array field wrongly declared static, would not have been caught. This
    checks the behavior that matters rather than the registry internals: real
    instances must survive a flatten/unflatten round trip with their arrays and
    their static metadata intact.
    """
    import jax

    containers = {
        "GBSAParams": aldp_implicit_params.gbsa,
        "RBTorsionParams": extract_params(_make_rb_system()[0]).rb_torsions,
        "CmapParams": extract_params(_make_cmap_system()[0]).cmap,
        "RestraintParams": make_restraints(
            jnp.array([0, 1], dtype=jnp.int32), aldp_positions_jnp[:2], 10.0),
    }

    for name, container in containers.items():
        assert container is not None, f"{name} fixture produced nothing"
        leaves, treedef = jax.tree_util.tree_flatten(container)
        assert leaves, f"{name} flattened to no leaves; array fields may be static"
        rebuilt = jax.tree_util.tree_unflatten(treedef, leaves)

        assert type(rebuilt) is type(container)
        for field in dataclasses.fields(container):
            original = getattr(container, field.name)
            restored = getattr(rebuilt, field.name)
            if hasattr(original, "shape"):
                np.testing.assert_allclose(
                    restored, original, err_msg=f"{name}.{field.name} changed")
            else:
                assert restored == original, f"{name}.{field.name} changed"

        # A hashable treedef is what jit needs to cache on.
        assert hash(treedef) is not None
