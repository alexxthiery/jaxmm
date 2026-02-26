"""Notebook utilities for interactive exploration with jaxmm.

Visualization, plotting, and analysis helpers for Jupyter notebooks.
Not re-exported by jaxmm.__init__; import explicitly::

    from jaxmm.notebook import show_structure, plot_ramachandran

Dependencies beyond core jaxmm: matplotlib, py3Dmol, openmm.
These are imported lazily inside each function.
"""

import io

import numpy as np
import jax.numpy as jnp

from jaxmm.utils import KB, dihedral_angle
from jaxmm.extract import phi_indices, psi_indices


# ---------------------------------------------------------------------------
# 3D visualization (py3Dmol)
# ---------------------------------------------------------------------------

def show_structure(positions, topology, label="", width=400, height=300):
    """Display a molecular structure in 3D via py3Dmol.

    Args:
        positions: Atom coordinates (n_atoms, 3) in nm. JAX or numpy.
        topology: OpenMM Topology for PDB conversion.
        label: Optional text printed above the viewer.
        width: Viewer width in pixels.
        height: Viewer height in pixels.

    Returns:
        py3Dmol view object. Call .show() to display.
    """
    import py3Dmol
    from openmm import app, unit

    if label:
        print(label)

    buf = io.StringIO()
    app.PDBFile.writeFile(topology, np.array(positions) * unit.nanometer, buf)

    view = py3Dmol.view(width=width, height=height)
    view.addModel(buf.getvalue(), "pdb")
    view.setStyle({}, {"stick": {}, "sphere": {"radius": 0.3}})
    view.zoomTo()
    return view


def _write_multimodel_pdb(frames, topology):
    """Build a multi-model PDB string with CONECT records in each model.

    Args:
        frames: (n_frames, n_atoms, 3) positions in nm, numpy array.
        topology: OpenMM Topology.

    Returns:
        PDB string with MODEL/ENDMDL blocks.
    """
    from openmm import app, unit

    # Extract CONECT lines from first frame
    ref_buf = io.StringIO()
    app.PDBFile.writeFile(topology, frames[0] * unit.nanometer, ref_buf)
    conect_lines = [l for l in ref_buf.getvalue().splitlines()
                    if l.startswith("CONECT")]

    model_blocks = []
    for i in range(len(frames)):
        frame_buf = io.StringIO()
        app.PDBFile.writeFile(topology, frames[i] * unit.nanometer, frame_buf)
        atom_lines = [l for l in frame_buf.getvalue().splitlines()
                      if l.startswith(("ATOM", "HETATM", "TER"))]
        model_blocks.append(atom_lines)

    buf = io.StringIO()
    for i, atom_lines in enumerate(model_blocks):
        buf.write(f"MODEL     {i + 1:4d}\n")
        buf.write("\n".join(atom_lines) + "\n")
        buf.write("\n".join(conect_lines) + "\n")
        buf.write("ENDMDL\n")
    buf.write("END\n")
    return buf.getvalue()


def animate_trajectory(trajectory, topology, n_frames=50, masses=None,
                       width=600, height=400):
    """Create a py3Dmol animation from an MD trajectory.

    Subsamples uniformly to n_frames and removes center-of-mass drift
    so the molecule stays centered in the viewer.

    Args:
        trajectory: Positions (n_total, n_atoms, 3) in nm. JAX or numpy.
        topology: OpenMM Topology.
        n_frames: Number of frames to display.
        masses: Atomic masses (n_atoms,) for COM removal. Equal masses if None.
        width: Viewer width in pixels.
        height: Viewer height in pixels.

    Returns:
        py3Dmol view with animation. Call .show() to display.
    """
    import py3Dmol

    n_total = trajectory.shape[0]
    indices = np.linspace(0, n_total - 1, n_frames, dtype=int)
    frames = np.array(trajectory[indices])

    # Remove center-of-mass drift
    m = np.array(masses) if masses is not None else np.ones(frames.shape[1])
    total_mass = m.sum()
    for i in range(len(frames)):
        com = (m[:, None] * frames[i]).sum(axis=0) / total_mass
        frames[i] -= com

    pdb_string = _write_multimodel_pdb(frames, topology)

    view = py3Dmol.view(width=width, height=height)
    view.addModelsAsFrames(pdb_string, "pdb")
    view.setStyle({}, {"stick": {}, "sphere": {"radius": 0.3}})
    view.zoomTo()
    view.animate({"loop": "forward", "reps": 0})
    return view


def animate_mode(pos_eq, mode_vector, masses, topology, amplitude=0.05,
                 n_frames=20, label="", width=500, height=350):
    """Animate a normal mode as oscillating displacement via py3Dmol.

    The mode vector is un-mass-weighted to Cartesian displacements,
    normalized, and scaled by amplitude.

    Args:
        pos_eq: Equilibrium positions (n_atoms, 3) in nm.
        mode_vector: Mass-weighted eigenvector (3*n_atoms,).
        masses: Atomic masses (n_atoms,) in amu.
        topology: OpenMM Topology.
        amplitude: Max displacement in nm.
        n_frames: Frames per oscillation cycle.
        label: Optional text printed above the viewer.
        width: Viewer width in pixels.
        height: Viewer height in pixels.

    Returns:
        py3Dmol view with back-and-forth animation. Call .show() to display.
    """
    import py3Dmol

    if label:
        print(label)

    pos_eq_np = np.array(pos_eq)
    mode = np.array(mode_vector)
    m = np.array(masses)

    # Un-mass-weight to Cartesian displacements
    inv_sqrt_m = np.repeat(1.0 / np.sqrt(m), 3)
    cart_mode = (mode * inv_sqrt_m).reshape(-1, 3)
    cart_mode = cart_mode / np.linalg.norm(cart_mode) * amplitude

    # Sinusoidal oscillation
    scales = np.sin(np.linspace(0, 2 * np.pi, n_frames))
    frames = np.array([pos_eq_np + s * cart_mode for s in scales])

    pdb_string = _write_multimodel_pdb(frames, topology)

    view = py3Dmol.view(width=width, height=height)
    view.addModelsAsFrames(pdb_string, "pdb")
    view.setStyle({}, {"stick": {}, "sphere": {"radius": 0.3}})
    view.zoomTo()
    view.animate({"loop": "backAndForth", "reps": 0})
    return view


# ---------------------------------------------------------------------------
# Dihedral analysis
# ---------------------------------------------------------------------------

def phi_psi_degrees(trajectory, topology):
    """Compute backbone phi/psi dihedral angles in degrees.

    Extracts the first phi and first psi dihedral from the topology.
    Works for single frames or batches.

    Args:
        trajectory: Positions (n_frames, n_atoms, 3) or (n_atoms, 3) in nm.
        topology: OpenMM Topology.

    Returns:
        (phi_deg, psi_deg) as numpy arrays, shape (n_frames,) or scalar.
    """
    phi_idx = jnp.array(phi_indices(topology))
    psi_idx = jnp.array(psi_indices(topology))
    phi = dihedral_angle(trajectory, phi_idx)
    psi = dihedral_angle(trajectory, psi_idx)
    # First (usually only) dihedral pair
    if phi.ndim > 1:
        phi, psi = phi[:, 0], psi[:, 0]
    else:
        phi, psi = phi[0], psi[0]
    return np.degrees(np.array(phi)), np.degrees(np.array(psi))


# ---------------------------------------------------------------------------
# Plotting (matplotlib)
# ---------------------------------------------------------------------------

def plot_ramachandran(phi_deg, psi_deg, ax=None, gridsize=40, cmap="viridis",
                      title=None, colorbar=True):
    """Hexbin Ramachandran plot with standard formatting.

    Args:
        phi_deg: Phi angles in degrees, shape (n,).
        psi_deg: Psi angles in degrees, shape (n,).
        ax: Matplotlib axes. Creates new figure if None.
        gridsize: Hexbin grid resolution.
        cmap: Colormap name.
        title: Plot title.
        colorbar: Add a colorbar.

    Returns:
        Matplotlib axes.
    """
    import matplotlib.pyplot as plt

    if ax is None:
        _, ax = plt.subplots(figsize=(6, 5))

    hb = ax.hexbin(phi_deg, psi_deg, gridsize=gridsize, cmap=cmap, mincnt=1)
    ax.set_xlabel(r"$\phi$ (deg)")
    ax.set_ylabel(r"$\psi$ (deg)")
    ax.set_xlim(-180, 180)
    ax.set_ylim(-180, 180)
    ax.set_aspect("equal")
    if title:
        ax.set_title(title)
    if colorbar:
        plt.colorbar(hb, ax=ax, label="Count")
    return ax


# ---------------------------------------------------------------------------
# Free energy estimation
# ---------------------------------------------------------------------------

def free_energy_1d(samples, temperature, bins=72, sample_range=None):
    """Estimate 1D free energy from samples: F = -kBT ln P(x).

    Args:
        samples: 1D array of sample values.
        temperature: Temperature in K.
        bins: Number of bins or array of bin edges.
        sample_range: (min, max) for histogram. Data range if None.

    Returns:
        (centers, fe) where centers is bin-center array and fe is free
        energy in kJ/mol, shifted so the minimum is 0. Empty bins are NaN.
    """
    kBT = KB * temperature
    hist, edges = np.histogram(samples, bins=bins, range=sample_range,
                               density=True)
    centers = 0.5 * (edges[:-1] + edges[1:])
    mask = hist > 0
    fe = np.full_like(hist, np.nan, dtype=float)
    fe[mask] = -kBT * np.log(hist[mask])
    fe -= np.nanmin(fe)
    return centers, fe


def free_energy_2d(x, y, temperature, bins=60, sample_range=None):
    """Estimate 2D free energy from samples: F = -kBT ln P(x, y).

    Args:
        x: First coordinate samples, 1D array.
        y: Second coordinate samples, 1D array.
        temperature: Temperature in K.
        bins: Number of bins per axis, or array of bin edges.
        sample_range: ((x_min, x_max), (y_min, y_max)). Data range if None.

    Returns:
        (x_centers, y_centers, fe) where fe has shape (n_x, n_y) in kJ/mol,
        shifted so the minimum is 0. Empty bins are NaN.
        For contourf with meshgrid: use fe.T (transpose).
    """
    kBT = KB * temperature
    hist, xedges, yedges = np.histogram2d(x, y, bins=bins, range=sample_range)
    x_centers = 0.5 * (xedges[:-1] + xedges[1:])
    y_centers = 0.5 * (yedges[:-1] + yedges[1:])
    hist_masked = np.where(hist > 0, hist, np.nan)
    fe = -kBT * np.log(hist_masked / hist.sum())
    fe -= np.nanmin(fe)
    return x_centers, y_centers, fe
