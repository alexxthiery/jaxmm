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
from jaxmm.utils import KB
from jaxmm.notebook import (
    _write_multimodel_pdb,
    free_energy_1d,
    free_energy_2d,
    phi_psi_degrees,
    plot_ramachandran,
)


# ---------------------------------------------------------------------------
# Free energy: analytic oracle
#
# For samples from N(0, sigma) the free energy is exactly parabolic,
# F(x) = kB*T * x^2 / (2 sigma^2) + const, so the fitted curvature has a
# closed form to compare against. Tolerances below were measured, not guessed:
# with these sample counts the 1D fit lands within 0.3% and the 2D all-bin fit
# within 0.8%.
# ---------------------------------------------------------------------------

SIGMA = 0.5
TEMPERATURE = 300.0


def _expected_curvature():
    """kB*T / (2 sigma^2), the analytic curvature of a Gaussian free energy."""
    return KB * TEMPERATURE / (2.0 * SIGMA**2)


def test_free_energy_1d_recovers_gaussian_curvature():
    """F = -kBT ln P must reproduce the parabola of a known Gaussian."""
    rng = np.random.default_rng(0)
    samples = rng.normal(0.0, SIGMA, 400_000)

    centers, fe = free_energy_1d(
        samples, TEMPERATURE, bins=80, sample_range=(-2 * SIGMA, 2 * SIGMA))

    finite = np.isfinite(fe)
    curvature = np.polyfit(centers[finite], fe[finite], 2)[0]
    np.testing.assert_allclose(curvature, _expected_curvature(), rtol=0.02)


def test_free_energy_1d_is_shifted_to_zero_minimum():
    """The reported minimum is exactly zero, which is what plots assume."""
    rng = np.random.default_rng(1)
    _, fe = free_energy_1d(rng.normal(0.0, SIGMA, 20_000), TEMPERATURE, bins=40)
    assert np.nanmin(fe) == 0.0
    assert np.all(fe[np.isfinite(fe)] >= 0.0)


def test_free_energy_1d_marks_empty_bins_nan():
    """Unvisited bins are NaN, not a spurious finite barrier height."""
    samples = np.concatenate([np.zeros(1000), np.full(1000, 10.0)])
    _, fe = free_energy_1d(samples, TEMPERATURE, bins=20, sample_range=(0.0, 10.0))
    assert np.isnan(fe).any(), "the gap between the two populations should be NaN"
    assert np.isfinite(fe[0]) and np.isfinite(fe[-1])


def test_free_energy_2d_recovers_isotropic_gaussian_curvature():
    """The 2D surface is fitted over every populated bin, not one slice."""
    rng = np.random.default_rng(1)
    x = rng.normal(0.0, SIGMA, 600_000)
    y = rng.normal(0.0, SIGMA, 600_000)

    x_centers, y_centers, fe = free_energy_2d(
        x, y, TEMPERATURE, bins=40,
        sample_range=((-2 * SIGMA, 2 * SIGMA), (-2 * SIGMA, 2 * SIGMA)))

    grid_x, grid_y = np.meshgrid(x_centers, y_centers, indexing="ij")
    radius_sq = (grid_x**2 + grid_y**2).ravel()
    flat = fe.ravel()
    finite = np.isfinite(flat)
    design = np.vstack([radius_sq[finite], np.ones(finite.sum())]).T
    curvature = np.linalg.lstsq(design, flat[finite], rcond=None)[0][0]

    np.testing.assert_allclose(curvature, _expected_curvature(), rtol=0.03)
    assert np.nanmin(fe) == 0.0


def test_free_energy_2d_orientation_is_x_by_y():
    """fe is indexed [x, y]; the docstring tells users to transpose for contourf."""
    rng = np.random.default_rng(2)
    x = rng.normal(0.0, 1.0, 20_000)
    y = rng.normal(0.0, 0.2, 20_000)   # deliberately narrower in y
    x_centers, y_centers, fe = free_energy_2d(x, y, TEMPERATURE, bins=(30, 20))

    assert fe.shape == (len(x_centers), len(y_centers)) == (30, 20)


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
