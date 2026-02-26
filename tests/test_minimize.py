"""Tests for jaxmm.minimize_energy (L-BFGS-B minimization)."""

import numpy as np
import jax.numpy as jnp
import jaxmm


def test_minimize_converges(aldp_positions_jnp, aldp_params):
    """Minimized energy is lower than initial."""
    e_init = float(jaxmm.total_energy(aldp_positions_jnp, aldp_params))
    pos_min = jaxmm.minimize_energy(aldp_positions_jnp, aldp_params)
    e_min = float(jaxmm.total_energy(pos_min, aldp_params))
    assert e_min <= e_init


def test_minimize_matches_openmm(aldp_system, aldp_positions, aldp_positions_jnp, aldp_params):
    """Minimized energy within kBT of OpenMM minimized energy."""
    import openmm
    from openmm import unit

    # OpenMM minimization via Context + LocalEnergyMinimizer
    ctx = openmm.Context(
        aldp_system,
        openmm.LangevinIntegrator(300 * unit.kelvin, 1 / unit.picosecond, 1 * unit.femtosecond),
        openmm.Platform.getPlatformByName("CPU"),
    )
    ctx.setPositions(aldp_positions * unit.nanometer)
    openmm.LocalEnergyMinimizer.minimize(ctx)
    omm_energy = ctx.getState(getEnergy=True).getPotentialEnergy().value_in_unit(
        unit.kilojoule_per_mole
    )
    del ctx

    # jaxmm minimization
    pos_min = jaxmm.minimize_energy(aldp_positions_jnp, aldp_params)
    jax_energy = float(jaxmm.total_energy(pos_min, aldp_params))

    # Both should find minima within kBT = 2.5 kJ/mol of each other
    assert abs(jax_energy - omm_energy) < 2.5


# ---------------------------------------------------------------------------
# Implicit solvent minimization
# ---------------------------------------------------------------------------

def test_minimize_implicit_converges(aldp_implicit_positions, aldp_implicit_params):
    """Minimization on implicit solvent system lowers energy."""
    pos = jnp.array(aldp_implicit_positions)
    e_init = float(jaxmm.total_energy(pos, aldp_implicit_params))
    pos_min = jaxmm.minimize_energy(pos, aldp_implicit_params)
    e_min = float(jaxmm.total_energy(pos_min, aldp_implicit_params))
    assert e_min <= e_init, f"Energy increased: {e_init:.4f} -> {e_min:.4f}"


def test_minimize_implicit_no_nan(aldp_implicit_positions, aldp_implicit_params):
    """Minimized positions on implicit system contain no NaN."""
    pos = jnp.array(aldp_implicit_positions)
    pos_min = jaxmm.minimize_energy(pos, aldp_implicit_params)
    assert not jnp.any(jnp.isnan(pos_min)), "NaN in minimized positions"
