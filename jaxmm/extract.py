"""Extract force field parameters from an OpenMM System into JAX arrays.

Converts OpenMM's object-oriented force descriptions into flat arrays
suitable for pure-function JAX energy evaluation.
"""

from dataclasses import dataclass, fields

import jax
import jax.numpy as jnp
import numpy as np



# Dielectric offset in nm, hardcoded in OpenMM's OBC GB implementation
DIELECTRIC_OFFSET = 0.009


def _register_pytree(cls, aux_field_names=()):
    """Register a frozen dataclass as a JAX pytree.

    Array fields become tree children; fields in aux_field_names become
    static auxiliary data (must be hashable).
    """
    child_names = [f.name for f in fields(cls) if f.name not in aux_field_names]

    def flatten(obj):
        children = [getattr(obj, name) for name in child_names]
        aux = {name: getattr(obj, name) for name in aux_field_names}
        return children, aux

    def unflatten(aux, children):
        kwargs = dict(zip(child_names, children))
        kwargs.update(aux)
        return cls(**kwargs)

    jax.tree_util.register_pytree_node(cls, flatten, unflatten)


@dataclass(frozen=True)
class BondParams:
    """Harmonic bond parameters.

    Args:
        atom_i: First atom indices, shape (n_bonds,) int32.
        atom_j: Second atom indices, shape (n_bonds,) int32.
        r0: Equilibrium distances in nm, shape (n_bonds,) float64.
        k: Force constants in kJ/mol/nm^2, shape (n_bonds,) float64.
    """
    atom_i: jax.Array
    atom_j: jax.Array
    r0: jax.Array
    k: jax.Array


@dataclass(frozen=True)
class AngleParams:
    """Harmonic angle parameters.

    Args:
        atom_i: First atom indices, shape (n_angles,) int32.
        atom_j: Central atom indices, shape (n_angles,) int32.
        atom_k: Third atom indices, shape (n_angles,) int32.
        theta0: Equilibrium angles in radians, shape (n_angles,) float64.
        k: Force constants in kJ/mol/rad^2, shape (n_angles,) float64.
    """
    atom_i: jax.Array
    atom_j: jax.Array
    atom_k: jax.Array
    theta0: jax.Array
    k: jax.Array


@dataclass(frozen=True)
class TorsionParams:
    """Periodic torsion parameters.

    Args:
        atom_i: First atom indices, shape (n_torsions,) int32.
        atom_j: Second atom indices, shape (n_torsions,) int32.
        atom_k: Third atom indices, shape (n_torsions,) int32.
        atom_l: Fourth atom indices, shape (n_torsions,) int32.
        periodicity: Periodicities, shape (n_torsions,) int32.
        phase: Phase offsets in radians, shape (n_torsions,) float64.
        k: Force constants in kJ/mol, shape (n_torsions,) float64.
    """
    atom_i: jax.Array
    atom_j: jax.Array
    atom_k: jax.Array
    atom_l: jax.Array
    periodicity: jax.Array
    phase: jax.Array
    k: jax.Array


@dataclass(frozen=True)
class RBTorsionParams:
    """Ryckaert-Bellemans torsion parameters (OPLS/GROMOS).

    E = sum_{i=0}^{5} C_i * cos^i(phi) for each torsion.

    Args:
        atom_i: First atom indices, shape (n_rb,) int32.
        atom_j: Second atom indices, shape (n_rb,) int32.
        atom_k: Third atom indices, shape (n_rb,) int32.
        atom_l: Fourth atom indices, shape (n_rb,) int32.
        c0: Coefficient of cos^0(phi) in kJ/mol, shape (n_rb,).
        c1: Coefficient of cos^1(phi) in kJ/mol, shape (n_rb,).
        c2: Coefficient of cos^2(phi) in kJ/mol, shape (n_rb,).
        c3: Coefficient of cos^3(phi) in kJ/mol, shape (n_rb,).
        c4: Coefficient of cos^4(phi) in kJ/mol, shape (n_rb,).
        c5: Coefficient of cos^5(phi) in kJ/mol, shape (n_rb,).
    """
    atom_i: jax.Array
    atom_j: jax.Array
    atom_k: jax.Array
    atom_l: jax.Array
    c0: jax.Array
    c1: jax.Array
    c2: jax.Array
    c3: jax.Array
    c4: jax.Array
    c5: jax.Array


@dataclass(frozen=True)
class CmapParams:
    """CMAP torsion correction parameters (CHARMM).

    Each term couples two dihedrals (typically phi/psi) with a 2D energy
    correction map interpolated via bicubic splines.

    Args:
        phi_atoms: Atom indices for first dihedral, shape (n_terms, 4) int32.
        psi_atoms: Atom indices for second dihedral, shape (n_terms, 4) int32.
        map_indices: Which map each term uses, shape (n_terms,) int32.
        maps: Energy grids, shape (n_maps, size, size) float64, in kJ/mol.
            Grid is uniformly spaced over [0, 2*pi) in both dimensions.
        map_size: Grid dimension (auxiliary).
    """
    phi_atoms: jax.Array
    psi_atoms: jax.Array
    map_indices: jax.Array
    maps: jax.Array
    map_size: int


@dataclass(frozen=True)
class RestraintParams:
    """Harmonic position restraint parameters.

    Args:
        atom_indices: Restrained atom indices, shape (n_restrained,) int32.
        reference_positions: Target positions in nm, shape (n_restrained, 3) float64.
        k: Spring constants in kJ/mol/nm^2, shape (n_restrained,) float64.
    """
    atom_indices: jax.Array
    reference_positions: jax.Array
    k: jax.Array


@dataclass(frozen=True)
class NonbondedParams:
    """Nonbonded interaction parameters.

    Per-atom parameters are used with Lorentz-Berthelot combining rules
    for normal pairs. Exception pairs (1-4 neighbors) use pre-computed
    parameters directly. Excluded pairs (1-2, 1-3) have zero interaction.

    Exclusions and exceptions are stored as sparse pair lists (O(n_pairs) memory)
    rather than dense NxN matrices (O(n_atoms^2) memory).

    Args:
        charges: Atomic partial charges in elementary charge units, shape (n_atoms,).
        sigmas: LJ sigma in nm, shape (n_atoms,).
        epsilons: LJ epsilon in kJ/mol, shape (n_atoms,).
        n_atoms: Number of atoms in the system (auxiliary, not a JAX array).
        exclusion_pairs: Excluded pair indices (1-2, 1-3), shape (n_excl, 2) int32. i < j.
        exception_pairs: Exception pair indices (1-4), shape (n_exn, 2) int32. i < j.
        exception_chargeprod: Pre-computed charge products, shape (n_exn,).
        exception_sigma: Pre-computed sigma, shape (n_exn,).
        exception_epsilon: Pre-computed epsilon, shape (n_exn,).
        cutoff: Nonbonded cutoff distance in nm (auxiliary). None for no cutoff.
        switch_distance: LJ switching function start in nm (auxiliary). None for no switch.
    """
    charges: jax.Array
    sigmas: jax.Array
    epsilons: jax.Array
    n_atoms: int
    exclusion_pairs: jax.Array
    exception_pairs: jax.Array
    exception_chargeprod: jax.Array
    exception_sigma: jax.Array
    exception_epsilon: jax.Array
    cutoff: float | None = None
    switch_distance: float | None = None


@dataclass(frozen=True)
class GBSAParams:
    """GBSA/OBC implicit solvent parameters.

    Supports both OBC1 and OBC2 variants via the tanh coefficients.
    The Born radius tanh formula is: tanh(alpha*psi + beta*psi^2 + gamma*psi^3).

    Args:
        charges: Partial charges in e, shape (n_atoms,).
        radii: Atomic radii in nm, shape (n_atoms,).
        scale_factors: OBC scaling factors, shape (n_atoms,).
        solute_dielectric: Interior dielectric constant (typically 1.0).
        solvent_dielectric: Exterior dielectric constant (typically 78.5).
        probe_radius: Solvent probe radius in nm (typically 0.14).
        sa_energy: Surface area energy coefficient in kJ/mol/nm^2 (typically 2.25936).
        alpha: Coefficient of psi in tanh (OBC1: 0.8, OBC2: 1.0).
        beta: Coefficient of psi^2 in tanh (OBC1: 0.0, OBC2: -0.8).
        gamma: Coefficient of psi^3 in tanh (OBC1: 2.909125, OBC2: 4.85).
    """
    charges: jax.Array
    radii: jax.Array
    scale_factors: jax.Array
    solute_dielectric: float
    solvent_dielectric: float
    probe_radius: float
    sa_energy: float
    alpha: float
    beta: float
    gamma: float


@dataclass(frozen=True)
class ForceFieldParams:
    """Complete force field parameters for a molecular system.

    Args:
        bonds: Harmonic bond parameters.
        angles: Harmonic angle parameters.
        torsions: Periodic torsion parameters.
        nonbonded: Nonbonded interaction parameters.
        masses: Atomic masses in amu (dalton), shape (n_atoms,) float64.
        n_atoms: Number of atoms in the system.
        gbsa: Optional GBSA/OBC implicit solvent parameters (OBC1 or OBC2).
        rb_torsions: Optional Ryckaert-Bellemans torsion parameters (OPLS/GROMOS).
        cmap: Optional CMAP torsion correction parameters (CHARMM).
        restraints: Optional harmonic position restraint parameters.
        box: Optional (3,) orthorhombic box lengths in nm for periodic systems.
    """
    bonds: BondParams
    angles: AngleParams
    torsions: TorsionParams
    nonbonded: NonbondedParams
    masses: jax.Array
    n_atoms: int
    gbsa: GBSAParams | None = None
    rb_torsions: RBTorsionParams | None = None
    cmap: CmapParams | None = None
    restraints: RestraintParams | None = None
    box: jax.Array | None = None  # (3,) orthorhombic box lengths in nm


# Register all param dataclasses as JAX pytrees so they work with jit/vmap/grad
_register_pytree(BondParams)
_register_pytree(AngleParams)
_register_pytree(TorsionParams)
_register_pytree(NonbondedParams, aux_field_names=("n_atoms", "cutoff", "switch_distance"))
_register_pytree(GBSAParams, aux_field_names=(
    "solute_dielectric", "solvent_dielectric", "probe_radius", "sa_energy",
    "alpha", "beta", "gamma",
))
_register_pytree(RBTorsionParams)
_register_pytree(CmapParams, aux_field_names=("map_size",))
_register_pytree(RestraintParams)


def _flatten_ffparams(obj):
    """Flatten ForceFieldParams, handling optional fields."""
    children = [
        obj.bonds, obj.angles, obj.torsions, obj.nonbonded, obj.masses,
        obj.gbsa, obj.rb_torsions, obj.cmap, obj.restraints, obj.box,
    ]
    aux = {"n_atoms": obj.n_atoms}
    return children, aux


def _unflatten_ffparams(aux, children):
    """Unflatten ForceFieldParams, handling optional fields."""
    bonds, angles, torsions, nonbonded, masses, gbsa, rb_torsions, cmap, restraints, box = children
    return ForceFieldParams(
        bonds=bonds, angles=angles, torsions=torsions,
        nonbonded=nonbonded, masses=masses, n_atoms=aux["n_atoms"],
        gbsa=gbsa, rb_torsions=rb_torsions, cmap=cmap, restraints=restraints,
        box=box,
    )


jax.tree_util.register_pytree_node(ForceFieldParams, _flatten_ffparams, _unflatten_ffparams)


def make_restraints(atom_indices, reference_positions, k) -> RestraintParams:
    """Create position restraint parameters.

    Args:
        atom_indices: Which atoms to restrain, shape (n,) int array.
        reference_positions: Target positions in nm, shape (n, 3).
        k: Spring constant(s) in kJ/mol/nm^2, scalar or shape (n,).

    Returns:
        RestraintParams ready to use in ForceFieldParams.
    """
    atom_indices = jnp.asarray(atom_indices, dtype=jnp.int32)
    reference_positions = jnp.asarray(reference_positions, dtype=jnp.float64)
    k = jnp.broadcast_to(jnp.asarray(k, dtype=jnp.float64), atom_indices.shape)
    return RestraintParams(
        atom_indices=atom_indices,
        reference_positions=reference_positions,
        k=k,
    )


def _backbone_atoms(topology):
    """Find backbone N, CA, C atom indices per residue.

    Returns a list of dicts, one per residue. Each dict maps atom name
    to index for whichever backbone atoms are present (N, CA, C).
    Residues missing all three are skipped.
    """
    residues = []
    for res in topology.residues():
        atoms = {}
        for atom in res.atoms():
            if atom.name in ("N", "CA", "C"):
                atoms[atom.name] = atom.index
        if atoms:
            residues.append(atoms)
    return residues


def phi_indices(topology) -> np.ndarray:
    """Get atom indices defining backbone phi angles from an OpenMM Topology.

    Phi = C(i-1) - N(i) - CA(i) - C(i). Requires the preceding residue to
    have a C atom and the current residue to have N, CA, C.

    Args:
        topology: An openmm.app.Topology.

    Returns:
        Array of shape (n_phi, 4) with atom indices, dtype int32.
        Empty (0, 4) array if no phi angles are found.
    """
    residues = _backbone_atoms(topology)
    indices = []
    for i in range(1, len(residues)):
        prev, curr = residues[i - 1], residues[i]
        if "C" in prev and all(k in curr for k in ("N", "CA", "C")):
            indices.append([prev["C"], curr["N"], curr["CA"], curr["C"]])
    if not indices:
        return np.empty((0, 4), dtype=np.int32)
    return np.array(indices, dtype=np.int32)


def psi_indices(topology) -> np.ndarray:
    """Get atom indices defining backbone psi angles from an OpenMM Topology.

    Psi = N(i) - CA(i) - C(i) - N(i+1). Requires the current residue to
    have N, CA, C and the following residue to have an N atom.

    Args:
        topology: An openmm.app.Topology.

    Returns:
        Array of shape (n_psi, 4) with atom indices, dtype int32.
        Empty (0, 4) array if no psi angles are found.
    """
    residues = _backbone_atoms(topology)
    indices = []
    for i in range(len(residues) - 1):
        curr, nxt = residues[i], residues[i + 1]
        if all(k in curr for k in ("N", "CA", "C")) and "N" in nxt:
            indices.append([curr["N"], curr["CA"], curr["C"], nxt["N"]])
    if not indices:
        return np.empty((0, 4), dtype=np.int32)
    return np.array(indices, dtype=np.int32)


def extract_params(system) -> ForceFieldParams:
    """Extract force field parameters from an OpenMM System.

    Iterates over all forces in the system and extracts parameters
    for bonds, angles, torsions, nonbonded interactions, and optionally
    GBSA/OBC2 implicit solvent.

    Args:
        system: An openmm.System object with assigned force field parameters.

    Returns:
        ForceFieldParams with all parameters as JAX arrays.

    Raises:
        ValueError: If the system has constraints, virtual sites,
            PME/Ewald electrostatics, unsupported force types, or
            is missing required forces (bonds, angles, torsions, nonbonded).
    """
    import openmm

    n_particles = system.getNumParticles()

    # Check for constraints (rigid bonds, e.g. in water models)
    n_constraints = system.getNumConstraints()
    if n_constraints > 0:
        raise ValueError(
            f"System has {n_constraints} constraints. jaxmm does not support "
            f"constraints; use constraints=None when building the system."
        )

    # Check for virtual sites (e.g. TIP4P/TIP5P extra interaction sites)
    virtual_sites = [i for i in range(n_particles) if system.isVirtualSite(i)]
    if virtual_sites:
        raise ValueError(
            f"System has {len(virtual_sites)} virtual sites (particles "
            f"{virtual_sites[:5]}{'...' if len(virtual_sites) > 5 else ''}). "
            f"jaxmm does not support virtual sites."
        )

    bonds = None
    angles = None
    torsions = None
    nonbonded = None
    gbsa = None
    rb_torsions = None
    cmap = None

    for i in range(system.getNumForces()):
        force = system.getForce(i)

        if isinstance(force, openmm.HarmonicBondForce):
            bonds = _extract_bonds(force)
        elif isinstance(force, openmm.HarmonicAngleForce):
            angles = _extract_angles(force)
        elif isinstance(force, openmm.PeriodicTorsionForce):
            torsions = _extract_torsions(force)
        elif isinstance(force, openmm.NonbondedForce):
            nonbonded = _extract_nonbonded(force, n_particles)
        elif isinstance(force, openmm.GBSAOBCForce):
            gbsa = _extract_gbsa_obc(force)
        elif isinstance(force, openmm.CustomGBForce):
            gbsa = _extract_gbsa_custom(force)
        elif isinstance(force, openmm.RBTorsionForce):
            rb_torsions = _extract_rb_torsions(force)
        elif isinstance(force, openmm.CMAPTorsionForce):
            cmap = _extract_cmap(force)
        elif isinstance(force, openmm.CMMotionRemover):
            pass  # intentionally ignored, not modeled in jaxmm
        else:
            raise ValueError(
                f"Unsupported force type: {type(force).__name__}. "
                f"jaxmm only supports: HarmonicBondForce, HarmonicAngleForce, "
                f"PeriodicTorsionForce, NonbondedForce, RBTorsionForce, "
                f"CMAPTorsionForce, GBSAOBCForce, CustomGBForce."
            )

    if any(x is None for x in [bonds, angles, torsions, nonbonded]):
        missing = []
        if bonds is None:
            missing.append("HarmonicBondForce")
        if angles is None:
            missing.append("HarmonicAngleForce")
        if torsions is None:
            missing.append("PeriodicTorsionForce")
        if nonbonded is None:
            missing.append("NonbondedForce")
        raise ValueError(f"Missing forces in system: {missing}")

    masses = _extract_masses(system)

    # Extract box vectors if system uses periodic boundary conditions
    box = None
    if nonbonded is not None and nonbonded.cutoff is not None:
        import openmm.unit as unit
        a, b, c = system.getDefaultPeriodicBoxVectors()
        box_np = np.array([
            a[0].value_in_unit(unit.nanometer),
            b[1].value_in_unit(unit.nanometer),
            c[2].value_in_unit(unit.nanometer),
        ], dtype=np.float64)
        box = jnp.array(box_np)

    return ForceFieldParams(
        bonds=bonds,
        angles=angles,
        torsions=torsions,
        nonbonded=nonbonded,
        masses=masses,
        n_atoms=system.getNumParticles(),
        gbsa=gbsa,
        rb_torsions=rb_torsions,
        cmap=cmap,
        box=box,
    )


def _extract_masses(system) -> jax.Array:
    """Extract atomic masses from an OpenMM System.

    Args:
        system: An openmm.System object.

    Returns:
        Masses in amu (dalton), shape (n_atoms,) float64.
    """
    import openmm.unit as unit

    n = system.getNumParticles()
    masses = np.empty(n, dtype=np.float64)
    for i in range(n):
        masses[i] = system.getParticleMass(i).value_in_unit(unit.dalton)
    return jnp.array(masses)


def _extract_bonds(force) -> BondParams:
    """Extract harmonic bond parameters from an OpenMM HarmonicBondForce."""
    import openmm.unit as unit

    n = force.getNumBonds()
    atom_i = np.empty(n, dtype=np.int32)
    atom_j = np.empty(n, dtype=np.int32)
    r0 = np.empty(n, dtype=np.float64)
    k = np.empty(n, dtype=np.float64)

    for idx in range(n):
        i, j, length, force_const = force.getBondParameters(idx)
        atom_i[idx] = i
        atom_j[idx] = j
        r0[idx] = length.value_in_unit(unit.nanometer)
        k[idx] = force_const.value_in_unit(unit.kilojoule_per_mole / unit.nanometer**2)

    return BondParams(
        atom_i=jnp.array(atom_i),
        atom_j=jnp.array(atom_j),
        r0=jnp.array(r0),
        k=jnp.array(k),
    )


def _extract_angles(force) -> AngleParams:
    """Extract harmonic angle parameters from an OpenMM HarmonicAngleForce."""
    import openmm.unit as unit

    n = force.getNumAngles()
    atom_i = np.empty(n, dtype=np.int32)
    atom_j = np.empty(n, dtype=np.int32)
    atom_k = np.empty(n, dtype=np.int32)
    theta0 = np.empty(n, dtype=np.float64)
    k_arr = np.empty(n, dtype=np.float64)

    for idx in range(n):
        i, j, k_atom, angle, force_const = force.getAngleParameters(idx)
        atom_i[idx] = i
        atom_j[idx] = j
        atom_k[idx] = k_atom
        theta0[idx] = angle.value_in_unit(unit.radian)
        k_arr[idx] = force_const.value_in_unit(
            unit.kilojoule_per_mole / unit.radian**2
        )

    return AngleParams(
        atom_i=jnp.array(atom_i),
        atom_j=jnp.array(atom_j),
        atom_k=jnp.array(atom_k),
        theta0=jnp.array(theta0),
        k=jnp.array(k_arr),
    )


def _extract_torsions(force) -> TorsionParams:
    """Extract periodic torsion parameters from an OpenMM PeriodicTorsionForce."""
    import openmm.unit as unit

    n = force.getNumTorsions()
    atom_i = np.empty(n, dtype=np.int32)
    atom_j = np.empty(n, dtype=np.int32)
    atom_k = np.empty(n, dtype=np.int32)
    atom_l = np.empty(n, dtype=np.int32)
    periodicity = np.empty(n, dtype=np.int32)
    phase = np.empty(n, dtype=np.float64)
    k_arr = np.empty(n, dtype=np.float64)

    for idx in range(n):
        i, j, k_atom, l, per, ph, force_const = force.getTorsionParameters(idx)
        atom_i[idx] = i
        atom_j[idx] = j
        atom_k[idx] = k_atom
        atom_l[idx] = l
        periodicity[idx] = per
        phase[idx] = ph.value_in_unit(unit.radian)
        k_arr[idx] = force_const.value_in_unit(unit.kilojoule_per_mole)

    return TorsionParams(
        atom_i=jnp.array(atom_i),
        atom_j=jnp.array(atom_j),
        atom_k=jnp.array(atom_k),
        atom_l=jnp.array(atom_l),
        periodicity=jnp.array(periodicity),
        phase=jnp.array(phase),
        k=jnp.array(k_arr),
    )


def _extract_rb_torsions(force) -> RBTorsionParams:
    """Extract RB torsion parameters from an OpenMM RBTorsionForce."""
    import openmm.unit as unit

    n = force.getNumTorsions()
    atom_i = np.empty(n, dtype=np.int32)
    atom_j = np.empty(n, dtype=np.int32)
    atom_k = np.empty(n, dtype=np.int32)
    atom_l = np.empty(n, dtype=np.int32)
    c = [np.empty(n, dtype=np.float64) for _ in range(6)]

    for idx in range(n):
        i, j, k_atom, l, c0, c1, c2, c3, c4, c5 = force.getTorsionParameters(idx)
        atom_i[idx] = i
        atom_j[idx] = j
        atom_k[idx] = k_atom
        atom_l[idx] = l
        for ci, val in enumerate([c0, c1, c2, c3, c4, c5]):
            c[ci][idx] = val.value_in_unit(unit.kilojoule_per_mole)

    return RBTorsionParams(
        atom_i=jnp.array(atom_i), atom_j=jnp.array(atom_j),
        atom_k=jnp.array(atom_k), atom_l=jnp.array(atom_l),
        c0=jnp.array(c[0]), c1=jnp.array(c[1]), c2=jnp.array(c[2]),
        c3=jnp.array(c[3]), c4=jnp.array(c[4]), c5=jnp.array(c[5]),
    )


def _extract_cmap(force) -> CmapParams:
    """Extract CMAP parameters from an OpenMM CMAPTorsionForce."""
    import openmm.unit as unit

    n_maps = force.getNumMaps()
    size = None
    map_list = []
    for i in range(n_maps):
        sz, energy_flat = force.getMapParameters(i)
        if size is None:
            size = sz
        # energy_flat is a list of Quantity objects in kJ/mol
        grid = np.array(
            [v.value_in_unit(unit.kilojoule_per_mole) for v in energy_flat],
            dtype=np.float64,
        ).reshape(sz, sz)
        map_list.append(grid)
    maps = np.stack(map_list)  # (n_maps, size, size)

    n_terms = force.getNumTorsions()
    phi_atoms = np.empty((n_terms, 4), dtype=np.int32)
    psi_atoms = np.empty((n_terms, 4), dtype=np.int32)
    map_indices = np.empty(n_terms, dtype=np.int32)

    for i in range(n_terms):
        result = force.getTorsionParameters(i)
        map_idx = result[0]
        a1, a2, a3, a4 = result[1], result[2], result[3], result[4]
        b1, b2, b3, b4 = result[5], result[6], result[7], result[8]
        map_indices[i] = map_idx
        phi_atoms[i] = [a1, a2, a3, a4]
        psi_atoms[i] = [b1, b2, b3, b4]

    return CmapParams(
        phi_atoms=jnp.array(phi_atoms),
        psi_atoms=jnp.array(psi_atoms),
        map_indices=jnp.array(map_indices),
        maps=jnp.array(maps),
        map_size=size,
    )


def _extract_nonbonded(force, n_atoms: int) -> NonbondedParams:
    """Extract nonbonded parameters from an OpenMM NonbondedForce.

    Builds per-atom arrays (charges, sigmas, epsilons) and sparse pair lists
    for exclusions and exceptions. Exception pairs store pre-computed
    chargeProd, sigma, epsilon values from OpenMM directly.
    """
    import openmm.unit as unit

    # Per-atom parameters
    charges = np.empty(n_atoms, dtype=np.float64)
    sigmas = np.empty(n_atoms, dtype=np.float64)
    epsilons = np.empty(n_atoms, dtype=np.float64)

    for i in range(n_atoms):
        q, sig, eps = force.getParticleParameters(i)
        charges[i] = q.value_in_unit(unit.elementary_charge)
        sigmas[i] = sig.value_in_unit(unit.nanometer)
        epsilons[i] = eps.value_in_unit(unit.kilojoule_per_mole)

    # Sparse pair lists for exclusions and exceptions
    exclusion_list = []
    exception_list = []
    exception_cp_list = []
    exception_sig_list = []
    exception_eps_list = []

    for idx in range(force.getNumExceptions()):
        i, j, chargeprod, sig, eps = force.getExceptionParameters(idx)
        chargeprod_val = chargeprod.value_in_unit(unit.elementary_charge**2)
        sig_val = sig.value_in_unit(unit.nanometer)
        eps_val = eps.value_in_unit(unit.kilojoule_per_mole)

        # Canonical ordering: i < j
        a, b = min(i, j), max(i, j)

        if chargeprod_val == 0.0 and eps_val == 0.0:
            exclusion_list.append((a, b))
        else:
            exception_list.append((a, b))
            exception_cp_list.append(chargeprod_val)
            exception_sig_list.append(sig_val)
            exception_eps_list.append(eps_val)

    # Convert to arrays (empty arrays get shape (0, 2) or (0,))
    if exclusion_list:
        exclusion_pairs = np.array(exclusion_list, dtype=np.int32)
    else:
        exclusion_pairs = np.empty((0, 2), dtype=np.int32)

    if exception_list:
        exception_pairs = np.array(exception_list, dtype=np.int32)
        exception_chargeprod = np.array(exception_cp_list, dtype=np.float64)
        exception_sigma = np.array(exception_sig_list, dtype=np.float64)
        exception_epsilon = np.array(exception_eps_list, dtype=np.float64)
    else:
        exception_pairs = np.empty((0, 2), dtype=np.int32)
        exception_chargeprod = np.empty(0, dtype=np.float64)
        exception_sigma = np.empty(0, dtype=np.float64)
        exception_epsilon = np.empty(0, dtype=np.float64)

    # Extract cutoff and switching (only for periodic methods)
    import openmm as _openmm
    cutoff = None
    switch_distance = None
    method = force.getNonbondedMethod()

    # PME/Ewald require long-range electrostatic corrections that jaxmm
    # does not implement. Only NoCutoff, CutoffNonPeriodic, and
    # CutoffPeriodic are supported.
    _pme_methods = {
        _openmm.NonbondedForce.PME: "PME",
        _openmm.NonbondedForce.LJPME: "LJPME",
        _openmm.NonbondedForce.Ewald: "Ewald",
    }
    if method in _pme_methods:
        raise ValueError(
            f"NonbondedForce uses {_pme_methods[method]} electrostatics. "
            f"jaxmm does not support long-range electrostatic methods "
            f"(PME/Ewald). Use NoCutoff, CutoffNonPeriodic, or "
            f"CutoffPeriodic instead."
        )

    if method == _openmm.NonbondedForce.CutoffPeriodic:
        cutoff = force.getCutoffDistance().value_in_unit(unit.nanometer)
        if force.getUseSwitchingFunction():
            switch_distance = force.getSwitchingDistance().value_in_unit(unit.nanometer)

    return NonbondedParams(
        charges=jnp.array(charges),
        sigmas=jnp.array(sigmas),
        epsilons=jnp.array(epsilons),
        n_atoms=n_atoms,
        exclusion_pairs=jnp.array(exclusion_pairs),
        exception_pairs=jnp.array(exception_pairs),
        exception_chargeprod=jnp.array(exception_chargeprod),
        exception_sigma=jnp.array(exception_sigma),
        exception_epsilon=jnp.array(exception_epsilon),
        cutoff=cutoff,
        switch_distance=switch_distance,
    )


def _extract_gbsa_obc(force) -> GBSAParams:
    """Extract GBSA/OBC2 parameters from an OpenMM GBSAOBCForce.

    GBSAOBCForce always uses OBC2 (alpha=1.0, beta=-0.8, gamma=4.85).
    """
    import openmm.unit as unit

    n = force.getNumParticles()
    charges = np.empty(n, dtype=np.float64)
    radii = np.empty(n, dtype=np.float64)
    scale_factors = np.empty(n, dtype=np.float64)

    for i in range(n):
        q, r, sf = force.getParticleParameters(i)
        charges[i] = q.value_in_unit(unit.elementary_charge)
        radii[i] = r.value_in_unit(unit.nanometer)
        scale_factors[i] = sf

    return GBSAParams(
        charges=jnp.array(charges),
        radii=jnp.array(radii),
        scale_factors=jnp.array(scale_factors),
        solute_dielectric=force.getSoluteDielectric(),
        solvent_dielectric=force.getSolventDielectric(),
        probe_radius=force.getProbeRadius().value_in_unit(unit.nanometer),
        sa_energy=force.getSurfaceAreaEnergy().value_in_unit(
            unit.kilojoule_per_mole / unit.nanometer**2
        ),
        alpha=1.0,
        beta=-0.8,
        gamma=4.85,
    )


def _extract_gbsa_custom(force) -> GBSAParams:
    """Extract GBSA parameters from an OpenMM CustomGBForce.

    Parses the OBC variant (OBC1/OBC2) from the Born radius expression.
    Per-particle params are (charge, offset_radius, scaled_radius).
    """
    import re

    n = force.getNumParticles()
    charges = np.empty(n, dtype=np.float64)
    offset_radii = np.empty(n, dtype=np.float64)
    scaled_radii = np.empty(n, dtype=np.float64)

    for i in range(n):
        params = force.getParticleParameters(i)
        charges[i] = params[0]
        offset_radii[i] = params[1]
        scaled_radii[i] = params[2]

    radii = offset_radii + DIELECTRIC_OFFSET
    scale_factors = scaled_radii / offset_radii

    # Parse tanh coefficients from Born radius expression (2nd computed value)
    _, born_expr, _ = force.getComputedValueParameters(1)
    tanh_match = re.search(r'tanh\(([^)]+)\)', born_expr)
    if not tanh_match:
        raise ValueError(f"Cannot parse tanh from Born expression: {born_expr}")

    tanh_arg = tanh_match.group(1)
    alpha, beta, gamma = 0.0, 0.0, 0.0
    for m in re.finditer(r'([+-]?\d*\.?\d+)\*psi(?:\^(\d+))?', tanh_arg):
        coeff = float(m.group(1))
        power = int(m.group(2)) if m.group(2) else 1
        if power == 1:
            alpha = coeff
        elif power == 2:
            beta = coeff
        elif power == 3:
            gamma = coeff

    # Parse dielectrics from energy expressions
    solute_dielectric = 1.0
    solvent_dielectric = 78.5
    probe_radius = 0.14
    sa_factor = 4.0 * np.pi * 2.25936  # default

    for e_idx in range(force.getNumEnergyTerms()):
        expr, _ = force.getEnergyTermParameters(e_idx)

        sd_match = re.search(r'soluteDielectric=([0-9.]+)', expr)
        if sd_match:
            solute_dielectric = float(sd_match.group(1))

        svd_match = re.search(r'solventDielectric=([0-9.]+)', expr)
        if svd_match:
            solvent_dielectric = float(svd_match.group(1))

        # SA term: factor*(radius+probe)^2*(radius/B)^6
        sa_match = re.search(r'(\d+\.?\d*)\*\(radius\+(\d+\.?\d*)\)\^2', expr)
        if sa_match:
            sa_factor = float(sa_match.group(1))
            probe_radius = float(sa_match.group(2))

    sa_energy = sa_factor / (4.0 * np.pi)

    return GBSAParams(
        charges=jnp.array(charges),
        radii=jnp.array(radii),
        scale_factors=jnp.array(scale_factors),
        solute_dielectric=solute_dielectric,
        solvent_dielectric=solvent_dielectric,
        probe_radius=probe_radius,
        sa_energy=sa_energy,
        alpha=alpha,
        beta=beta,
        gamma=gamma,
    )
