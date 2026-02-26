"""Tests for MD integrators (Verlet and Langevin BAOAB)."""

import numpy as np
import pytest

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import jax.random as random

import openmm
from openmm import unit

from jaxmm.extract import (
    ForceFieldParams,
    BondParams,
    AngleParams,
    TorsionParams,
    NonbondedParams,
)
from jaxmm.integrate import verlet, langevin_baoab, baoab_step, kinetic_energy, KB


# ---------------------------------------------------------------------------
# Diatomic harmonic oscillator fixture (built directly, no OpenMM)
# ---------------------------------------------------------------------------

def _diatomic_params():
    """Minimal ForceFieldParams for a 2-atom harmonic oscillator.

    Two carbon-like atoms (12 amu), one harmonic bond with
    k=200000 kJ/mol/nm^2, r0=0.15 nm. No angles, torsions, or nonbonded.
    """
    n = 2
    bonds = BondParams(
        atom_i=jnp.array([0], dtype=jnp.int32),
        atom_j=jnp.array([1], dtype=jnp.int32),
        r0=jnp.array([0.15]),
        k=jnp.array([200000.0]),
    )
    # Empty angle/torsion params
    angles = AngleParams(
        atom_i=jnp.array([], dtype=jnp.int32),
        atom_j=jnp.array([], dtype=jnp.int32),
        atom_k=jnp.array([], dtype=jnp.int32),
        theta0=jnp.array([]),
        k=jnp.array([]),
    )
    torsions = TorsionParams(
        atom_i=jnp.array([], dtype=jnp.int32),
        atom_j=jnp.array([], dtype=jnp.int32),
        atom_k=jnp.array([], dtype=jnp.int32),
        atom_l=jnp.array([], dtype=jnp.int32),
        periodicity=jnp.array([], dtype=jnp.int32),
        phase=jnp.array([]),
        k=jnp.array([]),
    )
    # Minimal nonbonded: zero charges and epsilons, bond pair excluded
    nonbonded = NonbondedParams(
        charges=jnp.zeros(n),
        sigmas=jnp.ones(n) * 0.3,
        epsilons=jnp.zeros(n),
        n_atoms=n,
        exclusion_pairs=jnp.array([[0, 1]], dtype=jnp.int32),
        exception_pairs=jnp.empty((0, 2), dtype=jnp.int32),
        exception_chargeprod=jnp.empty(0),
        exception_sigma=jnp.empty(0),
        exception_epsilon=jnp.empty(0),
    )
    masses = jnp.array([12.0, 12.0])
    return ForceFieldParams(
        bonds=bonds,
        angles=angles,
        torsions=torsions,
        nonbonded=nonbonded,
        masses=masses,
        n_atoms=n,
    )


@pytest.fixture(scope="module")
def diatomic():
    """Diatomic system: params, initial positions (stretched to 0.16 nm), zero velocities."""
    params = _diatomic_params()
    # Place atoms along x-axis, stretched from equilibrium 0.15 to 0.16 nm
    positions = jnp.array([[0.0, 0.0, 0.0], [0.16, 0.0, 0.0]])
    velocities = jnp.zeros((2, 3))
    return params, positions, velocities


# ---------------------------------------------------------------------------
# Test 1: Verlet vs OpenMM on diatomic
# ---------------------------------------------------------------------------

def test_verlet_vs_openmm_diatomic(diatomic):
    """Verlet trajectory matches OpenMM VerletIntegrator on diatomic system."""
    params, positions, velocities = diatomic
    dt = 0.001  # ps
    n_steps = 100

    # --- jaxmm Verlet ---
    _, _, traj_pos, _ = verlet(positions, velocities, params, dt, n_steps, save_every=1)

    # --- OpenMM Verlet (leapfrog scheme) ---
    system = openmm.System()
    system.addParticle(12.0 * unit.dalton)
    system.addParticle(12.0 * unit.dalton)
    bond_force = openmm.HarmonicBondForce()
    bond_force.addBond(0, 1, 0.15 * unit.nanometer,
                       200000.0 * unit.kilojoule_per_mole / unit.nanometer**2)
    system.addForce(bond_force)

    integrator = openmm.VerletIntegrator(dt * unit.picosecond)
    context = openmm.Context(system, integrator, openmm.Platform.getPlatformByName("CPU"))
    context.setPositions(np.array(positions) * unit.nanometer)

    # OpenMM's VerletIntegrator is leapfrog: setVelocities sets v(t - dt/2).
    # Our velocity Verlet starts from v(t). To match position trajectories,
    # set OpenMM's initial velocity to v(t) - (dt/2) * F(x_0) / m.
    state = context.getState(getForces=True)
    forces_init = state.getForces(asNumpy=True).value_in_unit(
        unit.kilojoule_per_mole / unit.nanometer
    )
    masses_np = np.array([12.0, 12.0])
    v_adjusted = np.array(velocities) - 0.5 * dt * forces_init / masses_np[:, None]
    context.setVelocities(v_adjusted * unit.nanometer / unit.picosecond)

    omm_traj = np.empty((n_steps, 2, 3))
    for i in range(n_steps):
        integrator.step(1)
        state = context.getState(getPositions=True)
        omm_traj[i] = state.getPositions(asNumpy=True).value_in_unit(unit.nanometer)
    del context

    # Compare frame-by-frame
    max_diff = float(jnp.max(jnp.abs(traj_pos - omm_traj)))
    assert max_diff < 1e-6, f"Max position diff = {max_diff}, expected < 1e-6 nm"


# ---------------------------------------------------------------------------
# Test 2: Verlet energy conservation
# ---------------------------------------------------------------------------

def test_verlet_energy_conservation(diatomic):
    """Verlet conserves total energy on diatomic oscillator."""
    from jaxmm.energy import total_energy

    params, positions, velocities = diatomic
    dt = 0.0005  # small timestep for tight conservation
    n_steps = 1000

    _, _, traj_pos, traj_vel = verlet(positions, velocities, params, dt, n_steps, save_every=1)

    # Compute total energy at each frame
    def frame_energy(pos, vel):
        pe = total_energy(pos, params)
        ke = kinetic_energy(vel, params.masses)
        return pe + ke

    energies = jax.vmap(frame_energy)(traj_pos, traj_vel)
    # Velocity Verlet conserves a shadow Hamiltonian, not the exact energy.
    # Energy oscillates with amplitude O(dt^2). For dt=0.0005, omega~183,
    # (dt*omega)^2 ~ 0.008, so energy error is ~1% of PE amplitude.
    drift = float(jnp.max(energies) - jnp.min(energies))
    assert drift < 0.05, f"Energy drift = {drift} kJ/mol, expected < 0.05"


# ---------------------------------------------------------------------------
# Test 3: Langevin temperature equilibrium
# ---------------------------------------------------------------------------

def test_langevin_temperature_equilibrium(diatomic):
    """Langevin BAOAB thermalizes to correct temperature (equipartition)."""
    params, positions, velocities = diatomic
    dt = 0.001
    n_steps = 50000
    save_every = 10
    temperature = 300.0
    friction = 10.0

    # Disable COM removal so thermostat targets all 6 DOF cleanly.
    # (For a 2-atom system, COM removal fights the thermostat significantly.)
    _, _, traj_pos, traj_vel = langevin_baoab(
        positions, velocities, params, dt, temperature, friction,
        n_steps, save_every=save_every, key=random.key(42), remove_com=False,
    )

    # Discard warmup (first 500 saved frames = 5000 steps)
    warmup = 500
    vel_equil = traj_vel[warmup:]

    # Average KE over equilibrated trajectory
    ke_per_frame = jax.vmap(kinetic_energy, in_axes=(0, None))(vel_equil, params.masses)
    avg_ke = float(jnp.mean(ke_per_frame))
    n_samples = len(ke_per_frame)

    # Expected from equipartition: <KE> = (n_dof/2) * kB * T, n_dof = 6
    # Var(KE) = (n_dof/2) * (kB*T)^2 from the Gamma distribution
    n_dof = 6
    expected_ke = (n_dof / 2.0) * KB * temperature
    theoretical_std = jnp.sqrt(n_dof / 2.0) * KB * temperature
    # Naive std of mean (ignoring autocorrelation, so z is inflated)
    naive_se = float(theoretical_std) / jnp.sqrt(n_samples)
    z = abs(avg_ke - expected_ke) / naive_se
    assert z < 5.0, (
        f"Avg KE = {avg_ke:.4f}, expected {expected_ke:.4f}, "
        f"z = {z:.1f} (>5, accounting for autocorrelation inflation)"
    )


# ---------------------------------------------------------------------------
# Test 4: Langevin bond length variance
# ---------------------------------------------------------------------------

def test_langevin_position_variance(diatomic):
    """Bond length variance matches kB*T/k for harmonic oscillator."""
    params, positions, velocities = diatomic
    dt = 0.001
    n_steps = 50000
    save_every = 10
    temperature = 300.0
    friction = 10.0
    k_bond = 200000.0  # kJ/mol/nm^2

    _, _, traj_pos, _ = langevin_baoab(
        positions, velocities, params, dt, temperature, friction,
        n_steps, save_every=save_every, key=random.key(123)
    )

    # Discard warmup
    warmup = 500
    pos_equil = traj_pos[warmup:]

    # Bond length: distance between atoms 0 and 1
    dr = pos_equil[:, 1, :] - pos_equil[:, 0, :]
    bond_lengths = jnp.linalg.norm(dr, axis=-1)
    var_r = float(jnp.var(bond_lengths))
    n_samples = len(bond_lengths)

    # For harmonic bond, variance of bond length = kB*T / k_bond.
    # For Gaussian samples, Var(sample_var) = 2*sigma^4 / (N-1).
    expected_var = KB * temperature / k_bond
    naive_se_var = jnp.sqrt(2.0) * expected_var / jnp.sqrt(n_samples - 1)
    z = abs(var_r - expected_var) / float(naive_se_var)
    assert z < 5.0, (
        f"Bond length variance = {var_r:.6e}, expected {expected_var:.6e}, "
        f"z = {z:.1f} (>5, accounting for autocorrelation inflation)"
    )


# ---------------------------------------------------------------------------
# Test 5: Verlet on ALDP (stability check)
# ---------------------------------------------------------------------------

def test_verlet_aldp_stable(aldp_params, aldp_positions_jnp):
    """Verlet on ALDP runs without NaN for 100 steps at small timestep."""
    from jaxmm.energy import total_energy

    dt = 0.0005  # small for stability with full force field
    n_steps = 100
    velocities = jnp.zeros_like(aldp_positions_jnp)

    final_pos, final_vel, traj_pos, traj_vel = verlet(
        aldp_positions_jnp, velocities, aldp_params, dt, n_steps, save_every=10
    )

    assert not jnp.any(jnp.isnan(final_pos)), "NaN in final positions"
    assert not jnp.any(jnp.isnan(final_vel)), "NaN in final velocities"

    # Energy should stay in a reasonable range (not exploded)
    e_init = float(total_energy(aldp_positions_jnp, aldp_params))
    e_final = float(total_energy(final_pos, aldp_params))
    # Starting from minimized positions with zero velocity, energy should stay close
    assert abs(e_final - e_init) < 10.0, (
        f"Energy changed too much: {e_init:.2f} -> {e_final:.2f} kJ/mol"
    )


# ---------------------------------------------------------------------------
# Test 6: Langevin JIT consistency
# ---------------------------------------------------------------------------

def test_langevin_jit(diatomic):
    """JIT-compiled Langevin matches non-JIT output."""
    params, positions, velocities = diatomic
    dt = 0.001
    n_steps = 100
    temperature = 300.0
    friction = 10.0
    key = random.key(7)

    result_nojit = langevin_baoab(
        positions, velocities, params, dt, temperature, friction,
        n_steps, key=key
    )

    result_jit = jax.jit(
        langevin_baoab, static_argnames=("n_steps", "save_every")
    )(
        positions, velocities, params, dt, temperature, friction,
        n_steps, key=key
    )

    for a, b in zip(result_nojit, result_jit):
        assert jnp.allclose(a, b, atol=1e-10), (
            f"JIT mismatch: max diff = {float(jnp.max(jnp.abs(a - b)))}"
        )


# ---------------------------------------------------------------------------
# Test 7: kinetic_energy unit test
# ---------------------------------------------------------------------------

def test_kinetic_energy():
    """Kinetic energy = 0.5 * sum(m * v^2) for known input."""
    masses = jnp.array([2.0, 3.0])  # amu
    velocities = jnp.array([[1.0, 0.0, 0.0], [0.0, 2.0, 0.0]])  # nm/ps

    ke = kinetic_energy(velocities, masses)
    # atom 0: 0.5 * 2 * 1^2 = 1.0
    # atom 1: 0.5 * 3 * 4 = 6.0
    # total = 7.0 kJ/mol
    expected = 7.0
    assert abs(float(ke) - expected) < 1e-10, f"KE = {float(ke)}, expected {expected}"


# ---------------------------------------------------------------------------
# Test 8: baoab_step single step
# ---------------------------------------------------------------------------

def test_baoab_step_single(diatomic):
    """baoab_step produces valid output: no NaN, positions change, forces update."""
    from jaxmm.energy import total_energy

    params, positions, velocities = diatomic
    forces = -jax.grad(total_energy)(positions, params)
    key = random.key(99)

    new_pos, new_vel, new_forces, new_key = baoab_step(
        positions, velocities, forces, key, params,
        temperature=300.0, dt=0.001, friction=10.0, remove_com=False,
    )

    assert not jnp.any(jnp.isnan(new_pos)), "NaN in positions"
    assert not jnp.any(jnp.isnan(new_vel)), "NaN in velocities"
    assert not jnp.any(jnp.isnan(new_forces)), "NaN in forces"
    # Positions should change (system is stretched from equilibrium)
    assert not jnp.allclose(positions, new_pos), "Positions unchanged after step"
    # Forces should differ (positions changed)
    assert not jnp.allclose(forces, new_forces), "Forces unchanged after step"


def test_md_trajectory_named_fields(diatomic):
    """MDTrajectory supports named field access."""
    from jaxmm.integrate import MDTrajectory

    params, positions, velocities = diatomic
    result = verlet(positions, velocities, params, 0.001, 10, save_every=1)

    # Named field access
    assert result.positions.shape == (2, 3)
    assert result.velocities.shape == (2, 3)
    assert result.trajectory_positions.shape == (10, 2, 3)
    assert result.trajectory_velocities.shape == (10, 2, 3)

    # Tuple unpacking still works
    pos, vel, traj_pos, traj_vel = result
    assert jnp.allclose(pos, result.positions)
    assert isinstance(result, MDTrajectory)


# ---------------------------------------------------------------------------
# Test 9: baoab_step matches langevin_baoab (regression test)
# ---------------------------------------------------------------------------

def test_baoab_step_matches_langevin_baoab(diatomic):
    """One baoab_step call matches one step of langevin_baoab.

    This is the key regression test for the refactor: langevin_baoab now
    delegates to baoab_step internally. Running n_steps=1 of langevin_baoab
    and comparing to a direct baoab_step call verifies they're identical.
    """
    from jaxmm.energy import total_energy

    params, positions, velocities = diatomic
    key = random.key(42)
    dt, temperature, friction = 0.001, 300.0, 10.0

    # Run langevin_baoab for 1 step
    result = langevin_baoab(
        positions, velocities, params, dt, temperature, friction,
        n_steps=1, save_every=1, key=key, remove_com=False,
    )

    # Run baoab_step directly with same initial forces
    init_forces = -jax.grad(total_energy)(positions, params)
    new_pos, new_vel, _, _ = baoab_step(
        positions, velocities, init_forces, key, params,
        temperature, dt, friction, remove_com=False,
    )

    assert jnp.allclose(result.positions, new_pos, atol=1e-12), (
        f"Position mismatch: {float(jnp.max(jnp.abs(result.positions - new_pos)))}"
    )
    assert jnp.allclose(result.velocities, new_vel, atol=1e-12), (
        f"Velocity mismatch: {float(jnp.max(jnp.abs(result.velocities - new_vel)))}"
    )


# ---------------------------------------------------------------------------
# Test 10: baoab_step JIT consistency
# ---------------------------------------------------------------------------

def test_baoab_step_jit(diatomic):
    """JIT-compiled baoab_step matches eager execution."""
    from jaxmm.energy import total_energy

    params, positions, velocities = diatomic
    forces = -jax.grad(total_energy)(positions, params)
    key = random.key(77)
    dt, temperature, friction = 0.001, 300.0, 10.0

    result_eager = baoab_step(
        positions, velocities, forces, key, params,
        temperature, dt, friction,
    )
    result_jit = jax.jit(baoab_step)(
        positions, velocities, forces, key, params,
        temperature, dt, friction,
    )

    for a, b in zip(result_eager, result_jit):
        assert jnp.allclose(a, b, atol=1e-10), (
            f"JIT mismatch: max diff = {float(jnp.max(jnp.abs(a - b)))}"
        )


# ---------------------------------------------------------------------------
# Test 11: baoab_step vmap across temperatures
# ---------------------------------------------------------------------------

def test_baoab_step_vmap(diatomic):
    """vmap(baoab_step) across temperatures matches sequential calls.

    This tests the parallel tempering use case: same initial state,
    different temperatures per replica.
    """
    from jaxmm.energy import total_energy

    params, positions, velocities = diatomic
    forces = -jax.grad(total_energy)(positions, params)
    dt, friction = 0.001, 10.0
    temperatures = jnp.array([300.0, 500.0, 800.0])
    keys = random.split(random.key(0), 3)

    # Sequential calls
    results_seq = [
        baoab_step(positions, velocities, forces, keys[i], params,
                   temperatures[i], dt, friction, False)
        for i in range(3)
    ]

    # Vmapped call
    batch_pos = jnp.broadcast_to(positions, (3,) + positions.shape)
    batch_vel = jnp.broadcast_to(velocities, (3,) + velocities.shape)
    batch_forces = jnp.broadcast_to(forces, (3,) + forces.shape)

    vmapped = jax.vmap(baoab_step, in_axes=(0, 0, 0, 0, None, 0, None, None, None))
    vmap_pos, vmap_vel, vmap_forces, vmap_keys = vmapped(
        batch_pos, batch_vel, batch_forces, keys, params,
        temperatures, dt, friction, False,
    )

    for i in range(3):
        assert jnp.allclose(vmap_pos[i], results_seq[i][0], atol=1e-10), (
            f"Replica {i} position mismatch"
        )
        assert jnp.allclose(vmap_vel[i], results_seq[i][1], atol=1e-10), (
            f"Replica {i} velocity mismatch"
        )

    # Different temperatures should produce different velocities (O-step noise scales with T)
    assert not jnp.allclose(vmap_vel[0], vmap_vel[2], atol=1e-3), (
        "300K and 800K replicas have same velocities"
    )


# ---------------------------------------------------------------------------
# Test 12: remove_com zeroes center-of-mass velocity
# ---------------------------------------------------------------------------

def test_remove_com_zeroes_com_velocity(diatomic):
    """remove_com=True zeroes COM velocity; remove_com=False does not."""
    from jaxmm.energy import total_energy

    params, positions, velocities = diatomic
    forces = -jax.grad(total_energy)(positions, params)
    key = random.key(55)
    dt, temperature, friction = 0.001, 300.0, 10.0

    # With remove_com=True
    _, vel_com, _, _ = baoab_step(
        positions, velocities, forces, key, params,
        temperature, dt, friction, remove_com=True,
    )
    com_vel = jnp.sum(params.masses[:, None] * vel_com, axis=0) / jnp.sum(params.masses)
    assert jnp.allclose(com_vel, 0.0, atol=1e-14), (
        f"COM velocity should be zero: {com_vel}"
    )

    # With remove_com=False, thermostat noise gives non-zero COM velocity
    _, vel_nocom, _, _ = baoab_step(
        positions, velocities, forces, key, params,
        temperature, dt, friction, remove_com=False,
    )
    com_vel_nocom = jnp.sum(params.masses[:, None] * vel_nocom, axis=0) / jnp.sum(params.masses)
    assert not jnp.allclose(com_vel_nocom, 0.0, atol=1e-10), (
        "COM velocity should be non-zero without removal"
    )


# ---------------------------------------------------------------------------
# Test 13: Langevin determinism (same key = same trajectory)
# ---------------------------------------------------------------------------

def test_langevin_deterministic(diatomic):
    """Same PRNG key produces identical Langevin trajectories."""
    params, positions, velocities = diatomic
    key = random.key(314)

    r1 = langevin_baoab(
        positions, velocities, params, 0.001, 300.0, 10.0,
        n_steps=50, key=key,
    )
    r2 = langevin_baoab(
        positions, velocities, params, 0.001, 300.0, 10.0,
        n_steps=50, key=key,
    )

    assert jnp.allclose(r1.positions, r2.positions, atol=1e-14), "Non-deterministic positions"
    assert jnp.allclose(r1.velocities, r2.velocities, atol=1e-14), "Non-deterministic velocities"


# ---------------------------------------------------------------------------
# Test 14: Trajectory frame indexing (save_every correctness)
# ---------------------------------------------------------------------------

def test_trajectory_frame_indexing(diatomic):
    """Frame i in trajectory corresponds to state after step (i+1)*save_every.

    Runs with save_every=5 and verifies frame 0 matches a separate run
    of 5 steps, and the final positions match the last frame.
    """
    params, positions, velocities = diatomic
    dt = 0.001
    save_every = 5
    n_steps = 10  # 2 frames saved

    result = verlet(positions, velocities, params, dt, n_steps, save_every=save_every)

    # Frame 0 should match state after 5 steps
    result_5 = verlet(positions, velocities, params, dt, 5, save_every=1)
    assert jnp.allclose(result.trajectory_positions[0], result_5.positions, atol=1e-12), (
        "Frame 0 doesn't match state after 5 steps"
    )

    # Frame 1 (last) should match final positions
    assert jnp.allclose(result.trajectory_positions[1], result.positions, atol=1e-12), (
        "Last frame doesn't match final positions"
    )


# ---------------------------------------------------------------------------
# Test 15: Verlet stability on implicit solvent
# ---------------------------------------------------------------------------

def test_verlet_implicit_stable(aldp_implicit_params, aldp_implicit_positions):
    """Verlet on implicit solvent ALDP runs without NaN for 50 steps."""
    from jaxmm.energy import total_energy

    pos = jnp.array(aldp_implicit_positions)
    dt = 0.0005  # small for stability with GBSA forces
    n_steps = 50
    velocities = jnp.zeros_like(pos)

    final_pos, final_vel, traj_pos, traj_vel = verlet(
        pos, velocities, aldp_implicit_params, dt, n_steps, save_every=10
    )

    assert not jnp.any(jnp.isnan(final_pos)), "NaN in final positions"
    assert not jnp.any(jnp.isnan(final_vel)), "NaN in final velocities"

    e_init = float(total_energy(pos, aldp_implicit_params))
    e_final = float(total_energy(final_pos, aldp_implicit_params))
    # Starting from minimized + zero velocity, energy should stay close
    assert abs(e_final - e_init) < 20.0, (
        f"Energy changed too much: {e_init:.2f} -> {e_final:.2f} kJ/mol"
    )


# ---------------------------------------------------------------------------
# Test 16: Langevin stability on implicit solvent
# ---------------------------------------------------------------------------

def test_langevin_implicit_stable(aldp_implicit_params, aldp_implicit_positions):
    """Langevin BAOAB on implicit solvent runs without NaN for 100 steps."""
    pos = jnp.array(aldp_implicit_positions)
    velocities = jnp.zeros_like(pos)

    result = langevin_baoab(
        pos, velocities, aldp_implicit_params,
        dt=0.001, temperature=300.0, friction=10.0,
        n_steps=100, save_every=10, key=random.key(55),
    )

    assert not jnp.any(jnp.isnan(result.positions)), "NaN in final positions"
    assert not jnp.any(jnp.isnan(result.velocities)), "NaN in final velocities"
    assert not jnp.any(jnp.isnan(result.trajectory_positions)), "NaN in trajectory"
