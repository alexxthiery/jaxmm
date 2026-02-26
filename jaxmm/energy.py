"""Pure JAX energy functions for molecular potential energy evaluation.

All functions are pure: (positions, params) -> scalar energy in kJ/mol.
Compatible with jax.jit, jax.vmap, and jax.grad.
"""

import jax
import jax.numpy as jnp

from jaxmm.extract import (
    BondParams,
    AngleParams,
    TorsionParams,
    RBTorsionParams,
    CmapParams,
    RestraintParams,
    NonbondedParams,
    GBSAParams,
    ForceFieldParams,
    DIELECTRIC_OFFSET,
)


def _check_x64():
    """Raise if JAX float64 is not enabled."""
    if not jax.config.jax_enable_x64:
        raise RuntimeError(
            "jaxmm requires float64. Call jax.config.update('jax_enable_x64', True) "
            "before importing jaxmm or using its functions."
        )


def _check_positions(positions, params):
    """Raise if positions shape does not match params.n_atoms."""
    if positions.shape[-2] != params.n_atoms:
        raise ValueError(
            f"positions has {positions.shape[-2]} atoms but params expects "
            f"{params.n_atoms}. Check that positions matches the system."
        )
    if positions.shape[-1] != 3:
        raise ValueError(
            f"positions last dimension is {positions.shape[-1]}, expected 3 (x, y, z)."
        )

# Coulomb constant in OpenMM units: kJ*nm/(mol*e^2)
ONE_4PI_EPS0 = 138.93545764438198


def bond_energy(positions: jax.Array, params: BondParams) -> jax.Array:
    """Compute harmonic bond energy.

    E = sum_bonds 0.5 * k * (r - r0)^2

    Urey-Bradley 1-3 terms (used in CHARMM force fields) are stored
    by OpenMM as regular entries in HarmonicBondForce and are
    automatically included without special handling.

    Args:
        positions: Atom coordinates, shape (n_atoms, 3) in nm.
        params: Bond parameters.

    Returns:
        Total bond energy in kJ/mol.
    """
    _check_x64()
    ri = positions[params.atom_i]  # (n_bonds, 3)
    rj = positions[params.atom_j]  # (n_bonds, 3)
    dr = ri - rj
    # Safe norm: avoids NaN gradients at zero distance
    r = jnp.sqrt(jnp.sum(dr**2, axis=-1) + 1e-30)  # (n_bonds,)
    return jnp.sum(0.5 * params.k * (r - params.r0) ** 2)


def angle_energy(positions: jax.Array, params: AngleParams) -> jax.Array:
    """Compute harmonic angle energy.

    E = sum_angles 0.5 * k * (theta - theta0)^2

    Args:
        positions: Atom coordinates, shape (n_atoms, 3) in nm.
        params: Angle parameters.

    Returns:
        Total angle energy in kJ/mol.
    """
    _check_x64()
    ri = positions[params.atom_i]
    rj = positions[params.atom_j]  # central atom
    rk = positions[params.atom_k]

    # Vectors from central atom to outer atoms
    v1 = ri - rj  # (n_angles, 3)
    v2 = rk - rj  # (n_angles, 3)

    # Angle via atan2: numerically stable, no arccos clipping needed.
    # atan2(|v1 x v2|, v1 . v2) gives theta in [0, pi].
    cross = jnp.cross(v1, v2)
    cross_norm = jnp.sqrt(jnp.sum(cross**2, axis=-1) + 1e-30)
    dot = jnp.sum(v1 * v2, axis=-1)
    theta = jnp.arctan2(cross_norm, dot)

    return jnp.sum(0.5 * params.k * (theta - params.theta0) ** 2)


def torsion_energy(positions: jax.Array, params: TorsionParams) -> jax.Array:
    """Compute periodic torsion energy.

    E = sum_torsions k * (1 + cos(n * phi - phase))

    Args:
        positions: Atom coordinates, shape (n_atoms, 3) in nm.
        params: Torsion parameters.

    Returns:
        Total torsion energy in kJ/mol.
    """
    _check_x64()
    ri = positions[params.atom_i]
    rj = positions[params.atom_j]
    rk = positions[params.atom_k]
    rl = positions[params.atom_l]

    # Dihedral angle via cross products.
    # Sign convention: matches OpenMM's PeriodicTorsionForce (atan2(y, x) without negation).
    # Note: dihedral_angle() in utils.py uses the OPPOSITE sign (negated) to match
    # the biochemistry/mdtraj convention. Do NOT unify them.
    b1 = rj - ri  # (n_torsions, 3)
    b2 = rk - rj
    b3 = rl - rk

    # Normal vectors to the two planes
    n1 = jnp.cross(b1, b2)  # (n_torsions, 3)
    n2 = jnp.cross(b2, b3)

    # Normalized b2 for the atan2 formula (safe norm to avoid NaN gradients)
    b2_norm = b2 / jnp.sqrt(jnp.sum(b2**2, axis=-1, keepdims=True) + 1e-30)

    # m1 = n1 x b2_hat (lies in the plane of n1 and b2)
    m1 = jnp.cross(n1, b2_norm)

    # phi = atan2(m1 . n2, n1 . n2)
    x = jnp.sum(n1 * n2, axis=-1)
    y = jnp.sum(m1 * n2, axis=-1)
    phi = jnp.arctan2(y, x)

    return jnp.sum(params.k * (1.0 + jnp.cos(params.periodicity * phi - params.phase)))


def rb_torsion_energy(positions: jax.Array, params: RBTorsionParams) -> jax.Array:
    """Compute Ryckaert-Bellemans torsion energy.

    E = sum_torsions sum_{i=0}^{5} C_i * cos^i(phi)

    Args:
        positions: Atom coordinates, shape (n_atoms, 3) in nm.
        params: RB torsion parameters.

    Returns:
        Total RB torsion energy in kJ/mol.
    """
    _check_x64()
    ri = positions[params.atom_i]
    rj = positions[params.atom_j]
    rk = positions[params.atom_k]
    rl = positions[params.atom_l]

    # Dihedral angle via cross products (same convention as torsion_energy)
    b1 = rj - ri
    b2 = rk - rj
    b3 = rl - rk

    n1 = jnp.cross(b1, b2)
    n2 = jnp.cross(b2, b3)
    b2_norm = b2 / jnp.sqrt(jnp.sum(b2**2, axis=-1, keepdims=True) + 1e-30)
    m1 = jnp.cross(n1, b2_norm)

    x = jnp.sum(n1 * n2, axis=-1)
    y = jnp.sum(m1 * n2, axis=-1)
    phi = jnp.arctan2(y, x)
    cos_phi = jnp.cos(phi)

    # Horner's method: c0 + cos*(c1 + cos*(c2 + cos*(c3 + cos*(c4 + cos*c5))))
    e = params.c0 + cos_phi * (
        params.c1 + cos_phi * (
            params.c2 + cos_phi * (
                params.c3 + cos_phi * (
                    params.c4 + cos_phi * params.c5))))

    return jnp.sum(e)


def cmap_energy(positions: jax.Array, params: CmapParams) -> jax.Array:
    """Compute CMAP torsion correction energy via 2D interpolation.

    Args:
        positions: Atom coordinates, shape (n_atoms, 3) in nm.
        params: CMAP parameters.

    Returns:
        Total CMAP correction energy in kJ/mol.
    """
    _check_x64()
    size = params.map_size

    def _dihedral(atoms):
        """Compute dihedral for one term, returns angle in [0, 2*pi)."""
        ri = positions[atoms[0]]
        rj = positions[atoms[1]]
        rk = positions[atoms[2]]
        rl = positions[atoms[3]]
        b1 = rj - ri
        b2 = rk - rj
        b3 = rl - rk
        n1 = jnp.cross(b1, b2)
        n2 = jnp.cross(b2, b3)
        b2_hat = b2 / jnp.sqrt(jnp.sum(b2**2) + 1e-30)
        m1 = jnp.cross(n1, b2_hat)
        phi = jnp.arctan2(jnp.sum(m1 * n2), jnp.sum(n1 * n2))
        # Map from [-pi, pi] to [0, 2*pi) to match OpenMM grid convention
        return phi % (2.0 * jnp.pi)

    def _one_term(i):
        phi = _dihedral(params.phi_atoms[i])
        psi = _dihedral(params.psi_atoms[i])
        grid = params.maps[params.map_indices[i]]  # (size, size)
        # Convert angles to fractional grid coordinates
        phi_idx = phi * size / (2.0 * jnp.pi)
        psi_idx = psi * size / (2.0 * jnp.pi)
        # Bilinear interpolation with periodic wrapping
        coords = jnp.array([[phi_idx], [psi_idx]])
        return jax.scipy.ndimage.map_coordinates(grid, coords, order=1, mode='wrap')[0]

    # Sum over all CMAP terms
    energies = jax.vmap(_one_term)(jnp.arange(params.phi_atoms.shape[0]))
    return jnp.sum(energies)


def restraint_energy(positions: jax.Array, params: RestraintParams) -> jax.Array:
    """Compute harmonic position restraint energy.

    E = 0.5 * sum_i k_i * |x_i - x_ref_i|^2

    Args:
        positions: Atom coordinates, shape (n_atoms, 3) in nm.
        params: Restraint parameters.

    Returns:
        Total restraint energy in kJ/mol.
    """
    _check_x64()
    dr = positions[params.atom_indices] - params.reference_positions
    return 0.5 * jnp.sum(params.k[:, None] * dr**2)


def _minimum_image(dr, box):
    """Apply minimum image convention for orthorhombic box.

    Args:
        dr: Displacement vectors, shape (..., 3).
        box: Box edge lengths, shape (3,).

    Returns:
        Wrapped displacement vectors.
    """
    return dr - box * jnp.round(dr / box)


def _lj_switch(r, switch_distance, cutoff):
    """Compute LJ switching function S(r).

    S = 1 - 6*x^5 + 15*x^4 - 10*x^3  where x = (r - r_sw) / (r_cut - r_sw).
    S(r_sw) = 1, S(r_cut) = 0. First and second derivatives are zero at both ends.

    Args:
        r: Distances, any shape.
        switch_distance: Start of switching region in nm.
        cutoff: End of switching region in nm.

    Returns:
        Switching factors, same shape as r.
    """
    x = jnp.clip((r - switch_distance) / (cutoff - switch_distance), 0.0, 1.0)
    return 1.0 - 6.0 * x**5 + 15.0 * x**4 - 10.0 * x**3


def _pair_energy(dist, sigma, epsilon, chargeprod):
    """Compute LJ + Coulomb energy for a single pair at given distance.

    Args:
        dist: Scalar distance (with epsilon offset for safe grad).
        sigma: LJ sigma for this pair.
        epsilon: LJ epsilon for this pair.
        chargeprod: Product of charges for this pair.

    Returns:
        Scalar pair energy in kJ/mol.
    """
    sig_over_r = sigma / dist
    sig_over_r6 = sig_over_r**6
    lj = 4.0 * epsilon * (sig_over_r6**2 - sig_over_r6)
    coulomb = ONE_4PI_EPS0 * chargeprod / dist
    return lj + coulomb


def _nonbonded_energy_with_dist(
    params: NonbondedParams, dist: jax.Array,
) -> jax.Array:
    """Compute nonbonded energy from a precomputed distance matrix.

    Args:
        params: Nonbonded parameters.
        dist: Distance matrix, shape (n_atoms, n_atoms) with safe sqrt offset.

    Returns:
        Total nonbonded energy in kJ/mol.
    """
    n_atoms = params.n_atoms
    upper = jnp.triu(jnp.ones((n_atoms, n_atoms), dtype=bool), k=1)

    sigma_ij = (params.sigmas[:, None] + params.sigmas[None, :]) / 2.0
    epsilon_ij = jnp.sqrt(params.epsilons[:, None] * params.epsilons[None, :])
    chargeprod_ij = params.charges[:, None] * params.charges[None, :]

    if params.cutoff is not None:
        # Compute LJ and Coulomb separately for switching
        sig_over_r = sigma_ij / dist
        sig_over_r6 = sig_over_r**6
        lj_ij = 4.0 * epsilon_ij * (sig_over_r6**2 - sig_over_r6)
        coul_ij = ONE_4PI_EPS0 * chargeprod_ij / dist

        # Apply switching to LJ only
        if params.switch_distance is not None:
            sw = _lj_switch(dist, params.switch_distance, params.cutoff)
            lj_ij = lj_ij * sw

        pair_e = lj_ij + coul_ij

        # Mask pairs beyond cutoff
        within_cutoff = dist < params.cutoff
        mask = upper & within_cutoff
    else:
        pair_e = _pair_energy(dist, sigma_ij, epsilon_ij, chargeprod_ij)
        mask = upper

    total = jnp.sum(jnp.where(mask, pair_e, 0.0))

    # Exclusions and exceptions (bonded neighbors, always within cutoff)
    exc_i = params.exclusion_pairs[:, 0]
    exc_j = params.exclusion_pairs[:, 1]
    total = total - jnp.sum(_pair_energy(
        dist[exc_i, exc_j], sigma_ij[exc_i, exc_j],
        epsilon_ij[exc_i, exc_j], chargeprod_ij[exc_i, exc_j],
    ))

    exn_i = params.exception_pairs[:, 0]
    exn_j = params.exception_pairs[:, 1]
    total = total - jnp.sum(_pair_energy(
        dist[exn_i, exn_j], sigma_ij[exn_i, exn_j],
        epsilon_ij[exn_i, exn_j], chargeprod_ij[exn_i, exn_j],
    ))
    total = total + jnp.sum(_pair_energy(
        dist[exn_i, exn_j], params.exception_sigma,
        params.exception_epsilon, params.exception_chargeprod,
    ))

    return total


def nonbonded_energy(positions: jax.Array, params: NonbondedParams) -> jax.Array:
    """Compute nonbonded energy (Coulomb + Lennard-Jones).

    Handles three pair categories:
    - Excluded pairs (1-2, 1-3): zero interaction
    - Exception pairs (1-4): use pre-computed parameters
    - Normal pairs: Lorentz-Berthelot combining rules

    Uses sparse pair lists for exclusions/exceptions (O(n_pairs) memory).

    Args:
        positions: Atom coordinates, shape (n_atoms, 3) in nm.
        params: Nonbonded parameters.

    Returns:
        Total nonbonded energy in kJ/mol.
    """
    _check_x64()
    dist, _ = _distance_matrix(positions)
    return _nonbonded_energy_with_dist(params, dist)


def _distance_matrix(positions: jax.Array, box=None):
    """Compute all-pairs distance matrix with safe sqrt for diagonal.

    Args:
        positions: (n_atoms, 3) in nm.
        box: Optional (3,) orthorhombic box lengths. If provided,
            applies minimum image convention.

    Returns:
        (dist, dist_sq) where dist is (n_atoms, n_atoms) with epsilon
        offset to avoid NaN gradients at r=0 on the diagonal.
    """
    dr = positions[:, None, :] - positions[None, :, :]  # (n, n, 3)
    if box is not None:
        dr = _minimum_image(dr, box)
    dist_sq = jnp.sum(dr**2, axis=-1)  # (n, n)
    dist = jnp.sqrt(dist_sq + 1e-30)  # (n, n)
    return dist, dist_sq


def _born_radii(positions: jax.Array, params: GBSAParams,
                dist: jax.Array) -> jax.Array:
    """Compute OBC Born radii for all atoms.

    Implements the HCT pairwise integral with OBC tanh scaling.
    Matches the CustomGBForce expressions used by OpenMM/openmmtools.
    Reference: Onufriev, Bashford, Case (2004), Proteins 55:383-394.

    Args:
        positions: Atom coordinates, shape (n_atoms, 3) in nm.
        params: GBSA parameters with alpha/beta/gamma tanh coefficients.
        dist: Precomputed distance matrix, shape (n_atoms, n_atoms).

    Returns:
        Born radii in nm, shape (n_atoms,).
    """
    n_atoms = positions.shape[0]
    offset_radii = params.radii - DIELECTRIC_OFFSET  # (n_atoms,)
    scaled_radii = offset_radii * params.scale_factors  # (n_atoms,)

    # Broadcast: or_i[i,j] = offset_radii[i], sr_j[i,j] = scaled_radii[j]
    or_i = offset_radii[:, None]  # (n, 1)
    sr_j = scaled_radii[None, :]  # (1, n)

    r_scaled = dist + sr_j  # (n, n)

    # L and U: matching OpenMM CustomGBForce convention
    # L = max(or_i, |r - sr_j|), U = r + sr_j
    abs_r_minus_sr = jnp.abs(dist - sr_j)
    L_raw = jnp.maximum(or_i, abs_r_minus_sr)
    L = 1.0 / L_raw
    U = 1.0 / r_scaled

    # HCT integral per pair: matches OpenMM expression
    # 0.5*(1/L - 1/U + 0.25*(r - sr^2/r)*(1/U^2 - 1/L^2) + 0.5*log(L/U)/r)
    L2 = L**2
    U2 = U**2
    # Safe reciprocal for r (diagonal is ~0)
    r_safe = jnp.where(dist > 1e-15, dist, 1.0)
    inv_r = jnp.where(dist > 1e-15, 1.0 / r_safe, 0.0)

    # Using reciprocal L, U: L-U = 1/L_raw - 1/U_raw
    # Combined middle terms: 0.25*(r - sr^2/r)*(U^2 - L^2) in reciprocal form
    term = (
        L - U
        + 0.25 * (dist - sr_j**2 * inv_r) * (U2 - L2)
        + 0.5 * inv_r * jnp.log(L_raw / r_scaled)
    )
    # The 0.5 outer factor from the OpenMM expression
    term = 0.5 * term

    # Only include pairs where or_i < r + sr_j (step function), and i != j
    active = (or_i < r_scaled) & ~jnp.eye(n_atoms, dtype=bool)
    term = jnp.where(active, term, 0.0)

    # psi = I * or, where I = sum_j(term)
    psi = offset_radii * jnp.sum(term, axis=1)  # (n_atoms,)

    # OBC tanh scaling: tanh(alpha*psi + beta*psi^2 + gamma*psi^3)
    tanh_sum = jnp.tanh(
        params.alpha * psi + params.beta * psi**2 + params.gamma * psi**3
    )
    born_radii = 1.0 / (1.0 / offset_radii - tanh_sum / params.radii)

    return born_radii


def _gb_energy(born_radii: jax.Array, dist_sq: jax.Array,
               params: GBSAParams) -> jax.Array:
    """Compute Generalized Born electrostatic solvation energy.

    Uses the Still equation: E = -C * sum_{i,j} q_i*q_j / f_GB_ij
    where C = 0.5 * ONE_4PI_EPS0 * (1/eps_solute - 1/eps_solvent).
    Self-energy (diagonal) gets factor 0.5, off-diagonal pairs counted once.

    Args:
        born_radii: OBC Born radii in nm, shape (n_atoms,).
        dist_sq: Precomputed squared distance matrix, shape (n_atoms, n_atoms).
        params: GBSA parameters.

    Returns:
        GB solvation energy in kJ/mol.
    """
    n_atoms = born_radii.shape[0]
    # Matches OpenMM sign convention: solvation energy is negative (favorable)
    pre_factor = (
        -ONE_4PI_EPS0
        * (1.0 / params.solute_dielectric - 1.0 / params.solvent_dielectric)
    )

    # f_GB = sqrt(r_ij^2 + B_i * B_j * exp(-r_ij^2 / (4 * B_i * B_j)))
    alpha2 = born_radii[:, None] * born_radii[None, :]  # (n, n)
    D = dist_sq / (4.0 * alpha2)
    f_GB = jnp.sqrt(dist_sq + alpha2 * jnp.exp(-D))  # (n, n)

    charge_prod = params.charges[:, None] * params.charges[None, :]  # (n, n)
    Gpol = pre_factor * charge_prod / f_GB  # (n, n)

    # Upper triangle (i < j): full interaction. Diagonal (i == j): half.
    upper = jnp.triu(jnp.ones((n_atoms, n_atoms), dtype=bool), k=1)
    diag = jnp.eye(n_atoms, dtype=bool)
    energy = jnp.sum(jnp.where(upper, Gpol, 0.0)) + 0.5 * jnp.sum(jnp.where(diag, Gpol, 0.0))

    return energy


def _sa_energy(born_radii: jax.Array, params: GBSAParams) -> jax.Array:
    """Compute ACE non-polar surface area solvation energy.

    E_SA = 4*pi*sa_energy * sum_i (r_i + probe_radius)^2 * (r_i / B_i)^6

    Args:
        born_radii: OBC Born radii in nm, shape (n_atoms,).
        params: GBSA parameters.

    Returns:
        SA energy in kJ/mol.
    """
    surface_area_factor = 4.0 * jnp.pi * params.sa_energy  # kJ/mol/nm^2
    r_plus_probe = params.radii + params.probe_radius
    ratio6 = (params.radii / born_radii) ** 6
    return jnp.sum(surface_area_factor * r_plus_probe**2 * ratio6)


def _gbsa_energy_with_dist(
    positions: jax.Array, params: GBSAParams,
    dist: jax.Array, dist_sq: jax.Array,
) -> jax.Array:
    """Compute GBSA energy from precomputed distance matrix.

    Args:
        positions: Atom coordinates, shape (n_atoms, 3) in nm.
        params: GBSA parameters.
        dist: Distance matrix, shape (n_atoms, n_atoms).
        dist_sq: Squared distance matrix, shape (n_atoms, n_atoms).

    Returns:
        GBSA solvation energy in kJ/mol.
    """
    br = _born_radii(positions, params, dist)
    return _gb_energy(br, dist_sq, params) + _sa_energy(br, params)


def gbsa_energy(positions: jax.Array, params: GBSAParams) -> jax.Array:
    """Compute GBSA/OBC implicit solvent energy (GB + SA).

    Args:
        positions: Atom coordinates, shape (n_atoms, 3) in nm.
        params: GBSA parameters.

    Returns:
        Total GBSA solvation energy in kJ/mol.
    """
    _check_x64()
    dist, dist_sq = _distance_matrix(positions)
    return _gbsa_energy_with_dist(positions, params, dist, dist_sq)


def total_energy(positions: jax.Array, params: ForceFieldParams) -> jax.Array:
    """Compute total potential energy as sum of all terms.

    Computes the all-pairs distance matrix once and shares it between
    nonbonded and GBSA terms for efficiency.

    Args:
        positions: Atom coordinates, shape (n_atoms, 3) in nm.
        params: Complete force field parameters.

    Returns:
        Total potential energy in kJ/mol.

    Note:
        To add custom energy terms, compose with total_energy directly::

            def my_energy(pos, params):
                return jaxmm.total_energy(pos, params) + my_custom_term(pos)

            grad_fn = jax.grad(my_energy)  # works with jit, vmap, grad
    """
    _check_x64()
    _check_positions(positions, params)
    dist, dist_sq = _distance_matrix(positions, box=params.box)
    e = (
        bond_energy(positions, params.bonds)
        + angle_energy(positions, params.angles)
        + torsion_energy(positions, params.torsions)
        + _nonbonded_energy_with_dist(params.nonbonded, dist)
    )
    if params.gbsa is not None:
        e = e + _gbsa_energy_with_dist(positions, params.gbsa, dist, dist_sq)
    if params.rb_torsions is not None:
        e = e + rb_torsion_energy(positions, params.rb_torsions)
    if params.cmap is not None:
        e = e + cmap_energy(positions, params.cmap)
    if params.restraints is not None:
        e = e + restraint_energy(positions, params.restraints)
    return e


def energy_components(positions: jax.Array, params: ForceFieldParams) -> dict:
    """Compute individual energy terms as a dictionary.

    Computes the all-pairs distance matrix once and shares it between
    nonbonded and GBSA terms for efficiency.

    Args:
        positions: Atom coordinates, shape (n_atoms, 3) in nm.
        params: Complete force field parameters.

    Returns:
        Dict with keys "bonds", "angles", "torsions", "nonbonded", and
        optionally "gbsa" (only when params.gbsa is not None). Values are
        scalar energies in kJ/mol.
    """
    _check_x64()
    _check_positions(positions, params)
    dist, dist_sq = _distance_matrix(positions, box=params.box)
    components = {
        "bonds": bond_energy(positions, params.bonds),
        "angles": angle_energy(positions, params.angles),
        "torsions": torsion_energy(positions, params.torsions),
        "nonbonded": _nonbonded_energy_with_dist(params.nonbonded, dist),
    }
    if params.gbsa is not None:
        components["gbsa"] = _gbsa_energy_with_dist(
            positions, params.gbsa, dist, dist_sq,
        )
    if params.rb_torsions is not None:
        components["rb_torsions"] = rb_torsion_energy(positions, params.rb_torsions)
    if params.cmap is not None:
        components["cmap"] = cmap_energy(positions, params.cmap)
    if params.restraints is not None:
        components["restraints"] = restraint_energy(positions, params.restraints)
    return components
