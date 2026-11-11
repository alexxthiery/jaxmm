"""Tests for jaxmm.notebook helpers.

notebook.py is not re-exported from jaxmm/__init__.py and had no coverage at
all. The plotting and 3D-viewer helpers are presentation code whose failure
mode is a bad picture, but free_energy_1d, free_energy_2d, phi_psi_degrees and
the PDB writer compute numbers that end up in figures and analyses, so they get
real oracles here rather than smoke checks.

py3Dmol is an optional dependency; the viewer tests skip when it is absent.
"""

import matplotlib
matplotlib.use("Agg")  # headless backend, must precede pyplot

import jax.numpy as jnp
import numpy as np
import pytest

import jaxmm
from jaxmm.notebook import (
    _write_multimodel_pdb,
    free_energy_1d,
    free_energy_2d,
    phi_psi_degrees,
    plot_ramachandran,
)


# Exact histogram counts give known free energies without sampling error.
# Powers of two make the expected differences integer multiples of kBT*log(2).


def test_free_energy_1d_matches_known_population_ratios():
    centers = np.array([-4., -2., 0., 2., 4.])
    samples = np.repeat(centers, [1, 2, 4, 0, 8])
    temperature = 427.
    actual_centers, fe = free_energy_1d(samples, temperature, bins=5,
                                      sample_range=(-5., 5.))
    expected = 0.008314462618*temperature*np.log(2)*np.array([3., 2., 1., np.nan, 0.])
    np.testing.assert_allclose(actual_centers, centers)
    np.testing.assert_allclose(fe, expected, rtol=1e-12, equal_nan=True)


def test_free_energy_2d_matches_known_asymmetric_population_table():
    counts = np.array([[1, 2], [4, 0], [8, 16]])
    ix, iy = np.indices(counts.shape)
    x = np.repeat(np.array([-2., 0., 2.])[ix.ravel()], counts.ravel())
    y = np.repeat(np.array([1., 3.])[iy.ravel()], counts.ravel())
    temperature = 381.
    xc, yc, fe = free_energy_2d(x, y, temperature, bins=(3, 2),
                               sample_range=((-3., 3.), (0., 4.)))
    expected = 0.008314462618*temperature*np.log(2)*np.array([[4., 3.], [2., np.nan], [1., 0.]])
    np.testing.assert_allclose(xc, [-2., 0., 2.])
    np.testing.assert_allclose(yc, [1., 3.])
    np.testing.assert_allclose(fe, expected, rtol=1e-12, equal_nan=True)


# ---------------------------------------------------------------------------
# Backbone angles
# ---------------------------------------------------------------------------

def test_phi_psi_degrees_matches_dihedral_angle(aldp_positions_jnp, aldp_topology):
    """Differential check against dihedral_angle, itself validated against mdtraj."""
    phi_deg, psi_deg = phi_psi_degrees(aldp_positions_jnp, aldp_topology)

    expected_phi = np.degrees(float(
        jaxmm.dihedral_angle(aldp_positions_jnp, jaxmm.phi_indices(aldp_topology))[0]))
    expected_psi = np.degrees(float(
        jaxmm.dihedral_angle(aldp_positions_jnp, jaxmm.psi_indices(aldp_topology))[0]))

    np.testing.assert_allclose(float(phi_deg), expected_phi, rtol=1e-10)
    np.testing.assert_allclose(float(psi_deg), expected_psi, rtol=1e-10)
    assert -180.0 <= float(phi_deg) <= 180.0


def test_phi_psi_degrees_handles_single_frame_and_batch(aldp_positions_jnp, aldp_topology):
    """Both branches of the ndim test are exercised, and they agree."""
    single_phi, single_psi = phi_psi_degrees(aldp_positions_jnp, aldp_topology)
    assert np.ndim(single_phi) == 0

    trajectory = jnp.stack([aldp_positions_jnp] * 3)
    batch_phi, batch_psi = phi_psi_degrees(trajectory, aldp_topology)
    assert batch_phi.shape == (3,)

    np.testing.assert_allclose(batch_phi, float(single_phi), rtol=1e-10)
    np.testing.assert_allclose(batch_psi, float(single_psi), rtol=1e-10)


# ---------------------------------------------------------------------------
# PDB writer
# ---------------------------------------------------------------------------

def test_write_multimodel_pdb_roundtrips_coordinates(aldp_positions_jnp, aldp_topology):
    """Coordinates parsed back out of the PDB text match the input.

    PDB stores Angstroms to three decimals, so the achievable tolerance is
    5e-5 nm. The check catches a unit error, an axis swap or a dropped frame,
    none of which a line count would notice.
    """
    frames = np.asarray(jnp.stack([aldp_positions_jnp, aldp_positions_jnp + 0.01]))
    text = _write_multimodel_pdb(frames, aldp_topology)

    assert text.count("MODEL ") == 2
    assert text.count("ENDMDL") == 2
    assert "CONECT" in text, "bonds must be repeated in every model for the viewer"
    assert text.rstrip().endswith("END")

    blocks = text.split("ENDMDL")[:2]
    for frame_index, block in enumerate(blocks):
        parsed = np.array([
            [float(line[30:38]), float(line[38:46]), float(line[46:54])]
            for line in block.splitlines() if line.startswith(("ATOM", "HETATM"))
        ]) * 0.1  # Angstrom to nm
        assert parsed.shape == frames[frame_index].shape
        np.testing.assert_allclose(parsed, frames[frame_index], atol=5e-5)


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def test_plot_ramachandran_sets_standard_axes():
    """Ramachandran plots must always span the full periodic range, squared up."""
    rng = np.random.default_rng(3)
    phi = rng.uniform(-180.0, 180.0, 500)
    psi = rng.uniform(-180.0, 180.0, 500)

    ax = plot_ramachandran(phi, psi, title="test", colorbar=False)

    assert ax.get_xlim() == (-180.0, 180.0)
    assert ax.get_ylim() == (-180.0, 180.0)
    assert ax.get_aspect() == 1.0
    assert ax.get_title() == "test"
    assert ax.collections, "hexbin produced no artist"


# ---------------------------------------------------------------------------
# 3D viewers (optional dependency)
# ---------------------------------------------------------------------------

def test_show_structure_returns_a_viewer(aldp_positions_jnp, aldp_topology):
    """Smoke check only; py3Dmol renders to a browser, not to a value we can assert."""
    pytest.importorskip("py3Dmol")
    from jaxmm.notebook import show_structure
    assert show_structure(aldp_positions_jnp, aldp_topology) is not None


def test_animate_trajectory_returns_a_viewer(aldp_positions_jnp, aldp_topology):
    """Smoke check; the subsampling and COM removal run before py3Dmol is touched."""
    pytest.importorskip("py3Dmol")
    from jaxmm.notebook import animate_trajectory
    trajectory = jnp.stack([aldp_positions_jnp + 0.001 * i for i in range(5)])
    assert animate_trajectory(trajectory, aldp_topology, n_frames=3) is not None
