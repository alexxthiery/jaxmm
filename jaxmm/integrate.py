"""Pure JAX molecular dynamics integrators.

Velocity Verlet (deterministic, energy-conserving) and Langevin BAOAB
(thermostatted, ergodic). Both use nested jax.lax.scan for jittable
trajectory generation.

Units follow OpenMM internal conventions:
  positions: nm, velocities: nm/ps, forces: kJ/mol/nm,
  masses: amu, time: ps, temperature: K.

Since 1 kJ/mol = 1 amu*nm^2/ps^2, no unit conversion is needed
between force and acceleration in these units.
"""

from typing import NamedTuple

import jax
import jax.numpy as jnp
import jax.random as random

from jaxmm.energy import total_energy
from jaxmm.utils import KB


class MDTrajectory(NamedTuple):
    """Output of MD integrators.

    NamedTuple so existing tuple unpacking still works and JAX treats it
    as a pytree automatically.

    Attributes:
        positions: Final positions, shape (n_atoms, 3).
        velocities: Final velocities, shape (n_atoms, 3).
        trajectory_positions: Saved positions, shape (n_frames, n_atoms, 3).
        trajectory_velocities: Saved velocities, shape (n_frames, n_atoms, 3).
    """
    positions: jax.Array
    velocities: jax.Array
    trajectory_positions: jax.Array
    trajectory_velocities: jax.Array


def kinetic_energy(velocities: jax.Array, masses: jax.Array) -> jax.Array:
    """Compute kinetic energy.

    KE = 0.5 * sum(m_i * |v_i|^2)

    Args:
        velocities: (n_atoms, 3) nm/ps.
        masses: (n_atoms,) amu.

    Returns:
        Scalar kinetic energy in kJ/mol.
    """
    return 0.5 * jnp.sum(masses[:, None] * velocities**2)


def verlet(positions: jax.Array, velocities: jax.Array, params, dt: float,
           n_steps: int, save_every: int = 1, remove_com: bool = True):
    """Velocity Verlet integration (kick-drift-kick).

    Matches OpenMM's VerletIntegrator. Symplectic, time-reversible,
    energy-conserving up to floating point.

    Args:
        positions: (n_atoms, 3) nm.
        velocities: (n_atoms, 3) nm/ps.
        params: ForceFieldParams (must have masses field).
        dt: Timestep in ps.
        n_steps: Total integration steps (must be divisible by save_every).
        save_every: Save trajectory every N steps.
        remove_com: Remove center-of-mass motion each step (default True).

    Returns:
        MDTrajectory with final state and saved frames.
        Trajectory shapes are (n_steps // save_every, n_atoms, 3).
    """
    if n_steps % save_every != 0:
        raise ValueError(f"n_steps ({n_steps}) must be divisible by save_every ({save_every})")
    inv_mass = (1.0 / params.masses)[:, None]  # (n_atoms, 1)
    total_mass = jnp.sum(params.masses)
    half_dt = 0.5 * dt
    n_outer = n_steps // save_every

    def _force(pos):
        return -jax.grad(total_energy)(pos, params)

    # Carry forces in scan state to avoid double force evaluation per step.
    # Force at end of step N is reused as force at start of step N+1.
    init_forces = _force(positions)

    def _one_step(carry, _):
        pos, vel, forces = carry
        vel = vel + half_dt * forces * inv_mass
        pos = pos + dt * vel
        forces = _force(pos)
        vel = vel + half_dt * forces * inv_mass
        if remove_com:
            vel = vel - jnp.sum(params.masses[:, None] * vel, axis=0) / total_mass
        return (pos, vel, forces), None

    def _outer_step(carry, _):
        (pos, vel, forces), _ = jax.lax.scan(_one_step, carry, None, length=save_every)
        return (pos, vel, forces), (pos, vel)

    (final_pos, final_vel, _), (traj_pos, traj_vel) = jax.lax.scan(
        _outer_step, (positions, velocities, init_forces), None, length=n_outer
    )
    return MDTrajectory(final_pos, final_vel, traj_pos, traj_vel)


def baoab_step(pos: jax.Array, vel: jax.Array, forces: jax.Array,
               key: jax.Array, params, temperature: float, dt: float,
               friction: float, remove_com: bool = True):
    """Single BAOAB Langevin step.

    BAOAB splitting: B (kick) - A (drift) - O (thermostat) - A (drift) - B (kick).
    The force from the second B is reused as the first B of the next step,
    so only one force evaluation per step.

    Useful as a building block for custom integrators (e.g. parallel tempering
    with per-replica temperatures via vmap).

    Args:
        pos: (n_atoms, 3) positions in nm.
        vel: (n_atoms, 3) velocities in nm/ps.
        forces: (n_atoms, 3) forces in kJ/mol/nm (from previous step).
        key: JAX random key.
        params: ForceFieldParams.
        temperature: Temperature in K (scalar; can vary per-replica under vmap).
        dt: Timestep in ps.
        friction: Friction coefficient in 1/ps.
        remove_com: Remove center-of-mass velocity (default True).

    Returns:
        (pos, vel, forces, key) tuple with updated state.
    """
    key, subkey = random.split(key)
    inv_mass = (1.0 / params.masses)[:, None]  # (n_atoms, 1)
    half_dt = 0.5 * dt

    # O-step coefficients
    c1 = jnp.exp(-friction * dt)
    c2 = jnp.sqrt((1.0 - c1**2) * KB * temperature * inv_mass)

    # B: kick with current force
    vel = vel + half_dt * forces * inv_mass
    # A: half drift
    pos = pos + half_dt * vel
    # O: Ornstein-Uhlenbeck thermostat
    noise = random.normal(subkey, vel.shape)
    vel = c1 * vel + c2 * noise
    # A: half drift
    pos = pos + half_dt * vel
    # Recompute force after position update
    forces = -jax.grad(total_energy)(pos, params)
    # B: kick with new force
    vel = vel + half_dt * forces * inv_mass

    if remove_com:
        total_mass = jnp.sum(params.masses)
        vel = vel - jnp.sum(params.masses[:, None] * vel, axis=0) / total_mass

    return pos, vel, forces, key


def langevin_baoab(positions: jax.Array, velocities: jax.Array, params,
                   dt: float, temperature: float, friction: float,
                   n_steps: int, save_every: int = 1, *, key: jax.Array,
                   remove_com: bool = True):
    """Langevin dynamics with BAOAB splitting.

    BAOAB is a second-order symmetric splitting that gives excellent
    configurational sampling. One force evaluation per step.

    Splitting per step:
      B: v += (dt/2) * F / m          (kick)
      A: x += (dt/2) * v              (drift)
      O: v = c1*v + c2*R              (Ornstein-Uhlenbeck thermostat)
      A: x += (dt/2) * v              (drift)
      B: v += (dt/2) * F_new / m      (kick, recompute force after drift)

    The force from the second B is reused as the first B of the next step,
    so only one force evaluation happens per step.

    Args:
        positions: (n_atoms, 3) nm.
        velocities: (n_atoms, 3) nm/ps.
        params: ForceFieldParams.
        dt: Timestep in ps.
        temperature: Temperature in K.
        friction: Friction coefficient in 1/ps.
        n_steps: Total integration steps (must be divisible by save_every).
        save_every: Save every N steps.
        key: JAX random key (required).
        remove_com: Remove center-of-mass motion each step (default True).

    Returns:
        MDTrajectory with final state and saved frames.
        Trajectory shapes are (n_steps // save_every, n_atoms, 3).
    """
    if n_steps % save_every != 0:
        raise ValueError(f"n_steps ({n_steps}) must be divisible by save_every ({save_every})")
    n_outer = n_steps // save_every

    # Initial force for the first step's first B
    init_forces = -jax.grad(total_energy)(positions, params)

    def _one_step(carry, _):
        pos, vel, forces, k = carry
        pos, vel, forces, k = baoab_step(
            pos, vel, forces, k, params, temperature, dt, friction, remove_com
        )
        return (pos, vel, forces, k), None

    def _outer_step(carry, _):
        (pos, vel, forces, k), _ = jax.lax.scan(
            _one_step, carry, None, length=save_every
        )
        return (pos, vel, forces, k), (pos, vel)

    init_carry = (positions, velocities, init_forces, key)
    (final_pos, final_vel, _, _), (traj_pos, traj_vel) = jax.lax.scan(
        _outer_step, init_carry, None, length=n_outer
    )
    return MDTrajectory(final_pos, final_vel, traj_pos, traj_vel)
