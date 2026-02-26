"""Tests for backbone dihedral angle computation."""

import numpy as np
import jax.numpy as jnp
import jaxmm
from jaxmm.energy import torsion_energy
from jaxmm.extract import TorsionParams


def test_phi_indices_aldp(aldp_topology):
    """ALDP has one phi angle: C(ACE)-N(ALA)-CA(ALA)-C(ALA) = [4, 6, 8, 14]."""
    idx = jaxmm.phi_indices(aldp_topology)
    assert idx.shape == (1, 4)
    assert list(idx[0]) == [4, 6, 8, 14]


def test_psi_indices_aldp(aldp_topology):
    """ALDP has one psi angle: N(ALA)-CA(ALA)-C(ALA)-N(NME) = [6, 8, 14, 16]."""
    idx = jaxmm.psi_indices(aldp_topology)
    assert idx.shape == (1, 4)
    assert list(idx[0]) == [6, 8, 14, 16]


def test_dihedral_angle_single_config(aldp_positions):
    """dihedral_angle works on a single (n_atoms, 3) configuration."""
    idx = np.array([[4, 6, 8, 14]], dtype=np.int32)
    angles = jaxmm.dihedral_angle(aldp_positions, idx)
    assert angles.shape == (1,)
    # Angle should be in [-pi, pi]
    assert -np.pi <= angles[0] <= np.pi


def test_dihedral_angle_batch(aldp_md_frames):
    """dihedral_angle works on a batch (n_frames, n_atoms, 3)."""
    idx = np.array([[4, 6, 8, 14], [6, 8, 14, 16]], dtype=np.int32)
    angles = jaxmm.dihedral_angle(aldp_md_frames, idx)
    assert angles.shape == (50, 2)
    assert np.all(angles >= -np.pi) and np.all(angles <= np.pi)


def test_dihedral_angle_matches_mdtraj(aldp_topology, aldp_md_frames):
    """Dihedral angles match mdtraj reference."""
    mdtraj = __import__("mdtraj")

    phi_idx = jaxmm.phi_indices(aldp_topology)
    psi_idx = jaxmm.psi_indices(aldp_topology)
    phi_jaxmm = jaxmm.dihedral_angle(aldp_md_frames, phi_idx)
    psi_jaxmm = jaxmm.dihedral_angle(aldp_md_frames, psi_idx)

    top = mdtraj.Topology.from_openmm(aldp_topology)
    traj = mdtraj.Trajectory(aldp_md_frames, top)
    _, phi_ref = mdtraj.compute_phi(traj)
    _, psi_ref = mdtraj.compute_psi(traj)

    assert np.max(np.abs(phi_jaxmm - phi_ref)) < 1e-6
    assert np.max(np.abs(psi_jaxmm - psi_ref)) < 1e-6


def test_dihedral_torsion_sign_convention():
    """dihedral_angle and torsion_energy use opposite sign conventions.

    dihedral_angle uses biochemistry convention (negated atan2, matches mdtraj).
    torsion_energy uses OpenMM convention (un-negated atan2).
    On the same geometry, they should produce angles of opposite sign.
    """
    # Non-degenerate geometry: 4 atoms forming a clear dihedral
    positions = jnp.array([
        [0.0, 0.0, 0.0],
        [0.15, 0.0, 0.0],
        [0.15, 0.15, 0.0],
        [0.15, 0.15, 0.15],
    ])
    indices = jnp.array([[0, 1, 2, 3]], dtype=jnp.int32)

    # dihedral_angle from utils
    angle_bio = float(jaxmm.dihedral_angle(positions, indices)[0])

    # Extract the angle that torsion_energy sees by computing the energy
    # with k=1, periodicity=1, phase=0: E = k*(1 + cos(1*phi - 0)) = 1 + cos(phi)
    # So cos(phi_openmm) = E - 1. And we know phi_bio = -phi_openmm.
    torsion_params = TorsionParams(
        atom_i=jnp.array([0], dtype=jnp.int32),
        atom_j=jnp.array([1], dtype=jnp.int32),
        atom_k=jnp.array([2], dtype=jnp.int32),
        atom_l=jnp.array([3], dtype=jnp.int32),
        periodicity=jnp.array([1], dtype=jnp.int32),
        phase=jnp.array([0.0]),
        k=jnp.array([1.0]),
    )
    e = float(torsion_energy(positions, torsion_params))
    cos_phi_openmm = e - 1.0  # E = 1 + cos(phi)

    # cos(-phi_bio) should equal cos(phi_openmm)
    assert abs(np.cos(-angle_bio) - cos_phi_openmm) < 1e-10, (
        f"Sign convention mismatch: cos(-angle_bio)={np.cos(-angle_bio):.8f}, "
        f"cos(phi_openmm)={cos_phi_openmm:.8f}"
    )
