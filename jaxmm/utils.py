"""Utility functions: minimization, Boltzmann factors, dihedral angles, serialization.

Provides log-Boltzmann factors under the canonical ensemble, with optional
energy regularization for numerical stability. Also provides geometry helpers
for backbone analysis, L-BFGS-B energy minimization, and parameter
serialization.
"""

import json
import warnings

import numpy as np
import jax
import jax.numpy as jnp
import jaxopt

from jaxmm.extract import ForceFieldParams
from jaxmm.energy import total_energy

# Boltzmann constant in kJ/(mol*K)
KB = 8.314462618e-3

# ---------------------------------------------------------------------------
# Unit conversion constants. Multiply to convert to internal units.
#   Internal units: nm, ps, kJ/mol, K, amu, elementary charge
#
# Usage:
#   dt = 1.0 * FEMTOSECOND          # 0.001 ps
#   pos_nm = 1.5 * ANGSTROM         # 0.15 nm
#   e_kj = 10.0 * KCAL_PER_MOL     # 41.84 kJ/mol
# ---------------------------------------------------------------------------
FEMTOSECOND = 1e-3    # 1 fs in ps
ANGSTROM = 0.1        # 1 A in nm
KCAL_PER_MOL = 4.184  # 1 kcal/mol in kJ/mol


def minimize_energy(
    positions: jax.Array,
    params: ForceFieldParams,
    tolerance: float = 10.0,
    max_iterations: int = 500,
) -> jax.Array:
    """Find a local energy minimum via L-BFGS.

    Pure JAX implementation using jaxopt.LBFGS, GPU-compatible.

    Args:
        positions: Initial atom coordinates, shape (n_atoms, 3) in nm.
        params: Force field parameters.
        tolerance: Gradient norm convergence threshold in kJ/mol/nm.
            Default 10 (matches OpenMM's default).
        max_iterations: Maximum L-BFGS iterations. Default 500.

    Returns:
        Minimized positions, shape (n_atoms, 3) in nm.
    """
    def _objective(pos_flat):
        return total_energy(pos_flat.reshape(positions.shape), params)

    solver = jaxopt.LBFGS(fun=_objective, maxiter=max_iterations, tol=tolerance)
    result = solver.run(positions.ravel())
    return result.params.reshape(positions.shape)


def log_boltzmann(positions: jax.Array, params: ForceFieldParams, temperature: float) -> jax.Array:
    """Compute log unnormalized Boltzmann factor: -E(x) / (kB * T).

    Args:
        positions: Atom coordinates, shape (n_atoms, 3) in nm.
        params: Force field parameters.
        temperature: Temperature in Kelvin.

    Returns:
        Scalar (dimensionless).
    """
    energy = total_energy(positions, params)
    return -energy / (KB * temperature)


def log_boltzmann_regularized(
    positions: jax.Array,
    params: ForceFieldParams,
    temperature: float,
    energy_cut: float,
    energy_max: float,
) -> jax.Array:
    """Log Boltzmann factor with energy regularization.

    For numerical stability when training generative models. Energy is
    clamped to avoid extreme values from unphysical configurations.

    Below energy_cut (in kBT): energy reported as-is.
    Above energy_cut: grows logarithmically.
    Above energy_max: hard-clamped.

    Args:
        positions: Atom coordinates, shape (n_atoms, 3) in nm.
        params: Force field parameters.
        temperature: Temperature in Kelvin.
        energy_cut: Threshold for log-dampening, in kBT units.
        energy_max: Hard upper bound, in kBT units.

    Returns:
        Regularized log Boltzmann factor (dimensionless).
    """
    energy = total_energy(positions, params)
    reduced = energy / (KB * temperature)  # E / kBT

    # Log-dampen above energy_cut
    regularized = jnp.where(
        reduced < energy_cut,
        reduced,
        energy_cut + jnp.log1p(reduced - energy_cut),
    )

    # Hard clamp at energy_max
    regularized = jnp.minimum(regularized, energy_max)

    return -regularized


def log_prob(positions: jax.Array, params: ForceFieldParams, temperature: float) -> jax.Array:
    """Deprecated: use log_boltzmann instead."""
    warnings.warn("log_prob is deprecated, use log_boltzmann", DeprecationWarning, stacklevel=2)
    return log_boltzmann(positions, params, temperature)


def log_prob_regularized(
    positions: jax.Array,
    params: ForceFieldParams,
    temperature: float,
    energy_cut: float,
    energy_max: float,
) -> jax.Array:
    """Deprecated: use log_boltzmann_regularized instead."""
    warnings.warn(
        "log_prob_regularized is deprecated, use log_boltzmann_regularized",
        DeprecationWarning, stacklevel=2,
    )
    return log_boltzmann_regularized(positions, params, temperature, energy_cut, energy_max)


def dihedral_angle(positions: jax.Array, indices: jax.Array) -> jax.Array:
    """Compute dihedral angles for given atom index quadruplets.

    Uses the same cross-product/atan2 formula as the torsion energy code.
    Works on single configurations or batches. Pure JAX, compatible with
    jit/vmap/grad.

    Args:
        positions: Atom coordinates, shape (n_atoms, 3) or (n_frames, n_atoms, 3).
        indices: Atom index quadruplets, shape (n_dihedrals, 4) int.

    Returns:
        Dihedral angles in radians, shape (n_dihedrals,) or (n_frames, n_dihedrals).
    """
    single = positions.ndim == 2
    if single:
        positions = positions[jnp.newaxis]  # (1, n_atoms, 3)

    # Gather atom positions: (n_frames, n_dihedrals, 3)
    p0 = positions[:, indices[:, 0]]
    p1 = positions[:, indices[:, 1]]
    p2 = positions[:, indices[:, 2]]
    p3 = positions[:, indices[:, 3]]

    b1 = p1 - p0
    b2 = p2 - p1
    b3 = p3 - p2

    n1 = jnp.cross(b1, b2)
    n2 = jnp.cross(b2, b3)
    # Safe norm: avoids NaN gradients at zero-length bond vectors
    b2_hat = b2 / jnp.sqrt(jnp.sum(b2**2, axis=-1, keepdims=True) + 1e-30)
    m1 = jnp.cross(n1, b2_hat)

    # Negate to match the standard biochemistry sign convention (same as mdtraj).
    # Note: torsion_energy() in energy.py uses the OPPOSITE sign (no negation) to match
    # OpenMM's PeriodicTorsionForce convention. Do NOT unify them.
    x = jnp.sum(n1 * n2, axis=-1)
    y = jnp.sum(m1 * n2, axis=-1)
    angles = -jnp.arctan2(y, x)

    if single:
        return angles[0]
    return angles


# ---------------------------------------------------------------------------
# Parameter serialization
# ---------------------------------------------------------------------------

def save_params(params: ForceFieldParams, path: str) -> None:
    """Save ForceFieldParams to a .npz file (no pickle).

    Arrays are stored as named numpy arrays. Scalar metadata (n_atoms,
    GBSA dielectrics, etc.) is stored as a JSON string.

    Args:
        params: ForceFieldParams to save.
        path: File path (should end in .npz).
    """
    from jaxmm.extract import (
        BondParams, AngleParams, TorsionParams, NonbondedParams, GBSAParams,
        RBTorsionParams, CmapParams, RestraintParams,
    )

    arrays = {
        # Bonds
        "bonds_atom_i": np.array(params.bonds.atom_i),
        "bonds_atom_j": np.array(params.bonds.atom_j),
        "bonds_r0": np.array(params.bonds.r0),
        "bonds_k": np.array(params.bonds.k),
        # Angles
        "angles_atom_i": np.array(params.angles.atom_i),
        "angles_atom_j": np.array(params.angles.atom_j),
        "angles_atom_k": np.array(params.angles.atom_k),
        "angles_theta0": np.array(params.angles.theta0),
        "angles_k": np.array(params.angles.k),
        # Torsions
        "torsions_atom_i": np.array(params.torsions.atom_i),
        "torsions_atom_j": np.array(params.torsions.atom_j),
        "torsions_atom_k": np.array(params.torsions.atom_k),
        "torsions_atom_l": np.array(params.torsions.atom_l),
        "torsions_periodicity": np.array(params.torsions.periodicity),
        "torsions_phase": np.array(params.torsions.phase),
        "torsions_k": np.array(params.torsions.k),
        # Nonbonded
        "nb_charges": np.array(params.nonbonded.charges),
        "nb_sigmas": np.array(params.nonbonded.sigmas),
        "nb_epsilons": np.array(params.nonbonded.epsilons),
        "nb_exclusion_pairs": np.array(params.nonbonded.exclusion_pairs),
        "nb_exception_pairs": np.array(params.nonbonded.exception_pairs),
        "nb_exception_chargeprod": np.array(params.nonbonded.exception_chargeprod),
        "nb_exception_sigma": np.array(params.nonbonded.exception_sigma),
        "nb_exception_epsilon": np.array(params.nonbonded.exception_epsilon),
        # Masses
        "masses": np.array(params.masses),
    }

    metadata = {
        "n_atoms": params.n_atoms,
        "has_gbsa": params.gbsa is not None,
        "has_rb_torsions": params.rb_torsions is not None,
        "has_cmap": params.cmap is not None,
        "has_restraints": params.restraints is not None,
        "has_box": params.box is not None,
        "nb_cutoff": params.nonbonded.cutoff,
        "nb_switch_distance": params.nonbonded.switch_distance,
    }

    if params.gbsa is not None:
        g = params.gbsa
        arrays["gbsa_charges"] = np.array(g.charges)
        arrays["gbsa_radii"] = np.array(g.radii)
        arrays["gbsa_scale_factors"] = np.array(g.scale_factors)
        metadata.update({
            "gbsa_solute_dielectric": g.solute_dielectric,
            "gbsa_solvent_dielectric": g.solvent_dielectric,
            "gbsa_probe_radius": g.probe_radius,
            "gbsa_sa_energy": g.sa_energy,
            "gbsa_alpha": g.alpha,
            "gbsa_beta": g.beta,
            "gbsa_gamma": g.gamma,
        })

    if params.rb_torsions is not None:
        rb = params.rb_torsions
        arrays["rb_atom_i"] = np.array(rb.atom_i)
        arrays["rb_atom_j"] = np.array(rb.atom_j)
        arrays["rb_atom_k"] = np.array(rb.atom_k)
        arrays["rb_atom_l"] = np.array(rb.atom_l)
        for ci in range(6):
            arrays[f"rb_c{ci}"] = np.array(getattr(rb, f"c{ci}"))

    if params.cmap is not None:
        cm = params.cmap
        arrays["cmap_phi_atoms"] = np.array(cm.phi_atoms)
        arrays["cmap_psi_atoms"] = np.array(cm.psi_atoms)
        arrays["cmap_map_indices"] = np.array(cm.map_indices)
        arrays["cmap_maps"] = np.array(cm.maps)
        metadata["cmap_map_size"] = cm.map_size

    if params.restraints is not None:
        r = params.restraints
        arrays["restraint_atom_indices"] = np.array(r.atom_indices)
        arrays["restraint_reference_positions"] = np.array(r.reference_positions)
        arrays["restraint_k"] = np.array(r.k)

    if params.box is not None:
        arrays["box"] = np.array(params.box)

    # Store scalar metadata as a JSON string in a 0-d numpy array,
    # avoiding pickle while keeping everything in a single .npz file.
    arrays["_metadata"] = np.array(json.dumps(metadata))
    np.savez(path, **arrays)


def load_params(path: str) -> ForceFieldParams:
    """Load ForceFieldParams from a .npz file saved by save_params.

    Args:
        path: File path to the .npz file.

    Returns:
        Reconstructed ForceFieldParams.
    """
    from jaxmm.extract import (
        BondParams, AngleParams, TorsionParams, NonbondedParams, GBSAParams,
        RBTorsionParams, CmapParams, RestraintParams,
    )

    data = np.load(path, allow_pickle=False)
    metadata = json.loads(str(data["_metadata"]))

    bonds = BondParams(
        atom_i=jnp.array(data["bonds_atom_i"]),
        atom_j=jnp.array(data["bonds_atom_j"]),
        r0=jnp.array(data["bonds_r0"]),
        k=jnp.array(data["bonds_k"]),
    )
    angles = AngleParams(
        atom_i=jnp.array(data["angles_atom_i"]),
        atom_j=jnp.array(data["angles_atom_j"]),
        atom_k=jnp.array(data["angles_atom_k"]),
        theta0=jnp.array(data["angles_theta0"]),
        k=jnp.array(data["angles_k"]),
    )
    torsions = TorsionParams(
        atom_i=jnp.array(data["torsions_atom_i"]),
        atom_j=jnp.array(data["torsions_atom_j"]),
        atom_k=jnp.array(data["torsions_atom_k"]),
        atom_l=jnp.array(data["torsions_atom_l"]),
        periodicity=jnp.array(data["torsions_periodicity"]),
        phase=jnp.array(data["torsions_phase"]),
        k=jnp.array(data["torsions_k"]),
    )
    nonbonded = NonbondedParams(
        charges=jnp.array(data["nb_charges"]),
        sigmas=jnp.array(data["nb_sigmas"]),
        epsilons=jnp.array(data["nb_epsilons"]),
        n_atoms=metadata["n_atoms"],
        exclusion_pairs=jnp.array(data["nb_exclusion_pairs"]),
        exception_pairs=jnp.array(data["nb_exception_pairs"]),
        exception_chargeprod=jnp.array(data["nb_exception_chargeprod"]),
        exception_sigma=jnp.array(data["nb_exception_sigma"]),
        exception_epsilon=jnp.array(data["nb_exception_epsilon"]),
        cutoff=metadata.get("nb_cutoff"),
        switch_distance=metadata.get("nb_switch_distance"),
    )

    gbsa = None
    if metadata.get("has_gbsa"):
        gbsa = GBSAParams(
            charges=jnp.array(data["gbsa_charges"]),
            radii=jnp.array(data["gbsa_radii"]),
            scale_factors=jnp.array(data["gbsa_scale_factors"]),
            solute_dielectric=metadata["gbsa_solute_dielectric"],
            solvent_dielectric=metadata["gbsa_solvent_dielectric"],
            probe_radius=metadata["gbsa_probe_radius"],
            sa_energy=metadata["gbsa_sa_energy"],
            alpha=metadata["gbsa_alpha"],
            beta=metadata["gbsa_beta"],
            gamma=metadata["gbsa_gamma"],
        )

    rb_torsions = None
    if metadata.get("has_rb_torsions"):
        rb_torsions = RBTorsionParams(
            atom_i=jnp.array(data["rb_atom_i"]),
            atom_j=jnp.array(data["rb_atom_j"]),
            atom_k=jnp.array(data["rb_atom_k"]),
            atom_l=jnp.array(data["rb_atom_l"]),
            c0=jnp.array(data["rb_c0"]), c1=jnp.array(data["rb_c1"]),
            c2=jnp.array(data["rb_c2"]), c3=jnp.array(data["rb_c3"]),
            c4=jnp.array(data["rb_c4"]), c5=jnp.array(data["rb_c5"]),
        )

    cmap = None
    if metadata.get("has_cmap"):
        cmap = CmapParams(
            phi_atoms=jnp.array(data["cmap_phi_atoms"]),
            psi_atoms=jnp.array(data["cmap_psi_atoms"]),
            map_indices=jnp.array(data["cmap_map_indices"]),
            maps=jnp.array(data["cmap_maps"]),
            map_size=metadata["cmap_map_size"],
        )

    restraints = None
    if metadata.get("has_restraints"):
        restraints = RestraintParams(
            atom_indices=jnp.array(data["restraint_atom_indices"]),
            reference_positions=jnp.array(data["restraint_reference_positions"]),
            k=jnp.array(data["restraint_k"]),
        )

    return ForceFieldParams(
        bonds=bonds,
        angles=angles,
        torsions=torsions,
        nonbonded=nonbonded,
        masses=jnp.array(data["masses"]),
        n_atoms=metadata["n_atoms"],
        gbsa=gbsa,
        rb_torsions=rb_torsions,
        cmap=cmap,
        restraints=restraints,
        box=jnp.array(data["box"]) if metadata.get("has_box") else None,
    )
