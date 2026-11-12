"""jaxmm: Pure JAX molecular potential energy evaluation.

Extracts force field parameters from OpenMM systems and evaluates
potential energy using pure JAX functions (jittable, vmappable, differentiable).
"""

__version__ = "0.2.0"

from jaxmm.extract import (
    extract_params, ForceFieldParams, GBSAParams, RBTorsionParams, CmapParams,
    RestraintParams, make_restraints, phi_indices, psi_indices,
)
from jaxmm.energy import (
    bond_energy,
    angle_energy,
    torsion_energy,
    rb_torsion_energy,
    cmap_energy,
    restraint_energy,
    nonbonded_energy,
    gbsa_energy,
    total_energy,
    energy_components,
)
from jaxmm.utils import (
    minimize_energy, log_boltzmann, log_boltzmann_regularized,
    log_boltzmann_internal, log_boltzmann_fixed_frame,
    log_prob, log_prob_regularized,
    dihedral_angle, save_params, load_params,
    KB, FEMTOSECOND, ANGSTROM, KCAL_PER_MOL,
)
from jaxmm.internal import InternalCoordinates
from jaxmm.sampling_coordinates import SamplingCoordinates
from jaxmm.batching import map_events
from jaxmm._validation import require_float64, validate_temperature
from jaxmm.integrate import verlet, langevin_baoab, baoab_step, kinetic_energy, MDTrajectory
from jaxmm.coordinates import (
    ANGLE_SCALE,
    BOND_SCALE,
    InternalWhitener,
    ZMatrix,
    aldp_zmatrix,
    aldp_in_l_sector,
    canonicalize_cartesian,
    cartesian_to_zmatrix,
    validate_zmatrix,
    internal_whitener,
    unwhiten_internals,
    whiten_internals,
    whitened_chart_bounds,
    whitener_log_abs_det_jacobian,
    zmatrix_in_domain,
    zmatrix_log_abs_det_jacobian,
    zmatrix_to_cartesian,
)

__all__ = [
    "extract_params",
    "ForceFieldParams",
    "GBSAParams",
    "RBTorsionParams",
    "CmapParams",
    "RestraintParams",
    "make_restraints",
    "phi_indices",
    "psi_indices",
    "bond_energy",
    "angle_energy",
    "torsion_energy",
    "rb_torsion_energy",
    "cmap_energy",
    "restraint_energy",
    "nonbonded_energy",
    "gbsa_energy",
    "total_energy",
    "energy_components",
    "minimize_energy",
    "log_boltzmann",
    "log_boltzmann_regularized",
    "log_boltzmann_internal",
    "log_boltzmann_fixed_frame",
    "InternalCoordinates",
    "SamplingCoordinates",
    "map_events",
    "require_float64",
    "validate_temperature",
    "log_prob",
    "log_prob_regularized",
    "dihedral_angle",
    "save_params",
    "load_params",
    "verlet",
    "langevin_baoab",
    "baoab_step",
    "kinetic_energy",
    "MDTrajectory",
    "ZMatrix",
    "aldp_zmatrix",
    "aldp_in_l_sector",
    "canonicalize_cartesian",
    "cartesian_to_zmatrix",
    "validate_zmatrix",
    "ANGLE_SCALE",
    "BOND_SCALE",
    "InternalWhitener",
    "internal_whitener",
    "unwhiten_internals",
    "whiten_internals",
    "whitened_chart_bounds",
    "whitener_log_abs_det_jacobian",
    "zmatrix_in_domain",
    "zmatrix_log_abs_det_jacobian",
    "zmatrix_to_cartesian",
    "KB",
    "FEMTOSECOND",
    "ANGSTROM",
    "KCAL_PER_MOL",
]
