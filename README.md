# jaxmm

Pure JAX molecular potential energy evaluation. OpenMM handles one-time force field setup; at runtime, energy evaluation is pure JAX: jittable, vmappable, differentiable via `jax.grad`.

## Setup

```bash
pip install -e .
```

This installs jaxmm in editable mode: `import jaxmm` works from anywhere, and source edits take effect immediately without reinstalling (activate your environment first, e.g. `conda activate myenv`).

Dependencies: `jax`, `numpy`, `jaxopt`. For extraction/tests: `openmm`, `openmmtools`, `pytest`.

**Precision**: float64 required. Set `jax.config.update("jax_enable_x64", True)` before using jaxmm.

## Quick start

```python
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
from openmmtools import testsystems
import jaxmm

# 1. Build molecule via OpenMM (one-time)
aldp = testsystems.AlanineDipeptideVacuum(constraints=None)
# Implicit solvent: testsystems.AlanineDipeptideImplicit(constraints=None)

# 2. Extract parameters into JAX arrays (one-time)
params = jaxmm.extract_params(aldp.system)

# 3. Save/load params (no OpenMM needed after this)
jaxmm.save_params(params, "aldp_params.npz")
params = jaxmm.load_params("aldp_params.npz")

# 4. Evaluate energy (pure JAX, no OpenMM dependency at runtime)
positions = jnp.array(...)  # (n_atoms, 3) float64, in nm
energy = jaxmm.total_energy(positions, params)  # kJ/mol

# 5. Per-term energy decomposition
components = jaxmm.energy_components(positions, params)
# {"bonds": ..., "angles": ..., "torsions": ..., "nonbonded": ...}

# 6. Differentiate
forces = -jax.grad(jaxmm.total_energy)(positions, params)

# 7. Batch evaluate
batch_energy = jax.vmap(jaxmm.total_energy, in_axes=(0, None))
energies = batch_energy(batch_positions, params)  # (batch,)

# 8. Log Boltzmann factor for sampling
lp = jaxmm.log_boltzmann(positions, params, temperature=300.0)
```

## API

All energy functions follow the same signature: `(positions, params) -> scalar`.

| Function | Description |
|----------|-------------|
| `extract_params(system)` | OpenMM System to `ForceFieldParams` (one-time, requires openmm) |
| `save_params(params, path)` | Save ForceFieldParams to .npz (no pickle) |
| `load_params(path)` | Load ForceFieldParams from .npz |
| `bond_energy(pos, params.bonds)` | Harmonic bonds: `0.5 * k * (r - r0)^2` |
| `angle_energy(pos, params.angles)` | Harmonic angles: `0.5 * k * (theta - theta0)^2` |
| `torsion_energy(pos, params.torsions)` | Periodic torsions: `k * (1 + cos(n*phi - phase))` |
| `rb_torsion_energy(pos, params.rb_torsions)` | Ryckaert-Bellemans torsions (OPLS/GROMOS) |
| `cmap_energy(pos, params.cmap)` | CMAP torsion correction (CHARMM backbone phi/psi) |
| `restraint_energy(pos, params.restraints)` | Harmonic position restraints: `0.5 * k * |x - x_ref|^2` |
| `nonbonded_energy(pos, params.nonbonded)` | Coulomb + Lennard-Jones (sparse exclusions/exceptions, optional cutoff+switching) |
| `gbsa_energy(pos, params.gbsa)` | GBSA/OBC implicit solvent (GB electrostatic + SA non-polar) |
| `total_energy(pos, params)` | Sum of all terms (optional terms included when present) |
| `energy_components(pos, params)` | Dict of per-term energies (shared distance matrix) |
| `make_restraints(indices, ref_pos, k)` | Create RestraintParams for use in ForceFieldParams |
| `log_boltzmann(pos, params, T)` | `-E / (kB * T)` |
| `log_boltzmann_regularized(pos, params, T, cut, max)` | With energy clamping for numerical stability |
| `phi_indices(topology)` | Backbone phi dihedral atom indices from OpenMM Topology |
| `psi_indices(topology)` | Backbone psi dihedral atom indices from OpenMM Topology |
| `dihedral_angle(positions, indices)` | Compute dihedral angles from positions + index array |
| `verlet(pos, vel, params, dt, n, ...)` | Velocity Verlet integrator (symplectic, energy-conserving) |
| `langevin_baoab(pos, vel, params, dt, T, friction, n, *, key)` | Langevin BAOAB thermostat (second-order, ergodic) |
| `kinetic_energy(vel, masses)` | `0.5 * sum(m * v^2)` in kJ/mol |
| `baoab_step(pos, vel, forces, key, params, T, dt, friction)` | Single BAOAB Langevin step (building block for custom integrators) |
| `minimize_energy(pos, params, tol, max_iter)` | L-BFGS energy minimization via jaxopt (pure JAX, GPU-compatible) |

Both integrators return `MDTrajectory(positions, velocities, trajectory_positions, trajectory_velocities)`, a NamedTuple that also supports tuple unpacking.

**Units**: positions in nm, velocities in nm/ps, energies in kJ/mol, masses in amu, time in ps, angles in radians, charges in elementary charge units.

**Unit constants**: `FEMTOSECOND` (1e-3 ps), `ANGSTROM` (0.1 nm), `KCAL_PER_MOL` (4.184 kJ/mol), `KB` (Boltzmann constant, kJ/(mol*K)). Multiply to convert: `dt = 2.0 * jaxmm.FEMTOSECOND` gives 0.002 ps.

## Project structure

```
jaxmm/
  __init__.py        public API (37 exports)
  extract.py         OpenMM System -> ForceFieldParams dataclass
  energy.py          pure JAX energy functions
  integrate.py       Verlet and Langevin BAOAB integrators (pure JAX)
  utils.py           minimize_energy, log_boltzmann, dihedral_angle, serialization
  notebook.py        visualization and analysis helpers for Jupyter (not re-exported)
tests/
  conftest.py        ALDP fixtures (vacuum + implicit), OpenMM reference helpers
  test_extraction.py parameter shapes, values, and input validation (24 tests)
  test_bonds.py      bond energy vs OpenMM (5 tests)
  test_angles.py     angle energy vs OpenMM (5 tests)
  test_torsions.py   torsion energy vs OpenMM (5 tests)
  test_rb_torsions.py RB torsion energy vs OpenMM (5 tests)
  test_cmap.py       CMAP torsion correction vs OpenMM (5 tests)
  test_restraints.py harmonic position restraints (7 tests)
  test_nonbonded.py  nonbonded energy vs OpenMM (5 tests)
  test_pbc.py        periodic boundary conditions + cutoff (11 tests)
  test_gbsa.py       GBSA energy, Born radii, gradients, jit, vmap (19 tests)
  test_total.py      total energy, vmap, log_boltzmann, energy_components, composability (27 tests)
  test_dihedral.py   phi/psi dihedral angles (6 tests)
  test_grad.py       gradients vs OpenMM forces + finite diff (8 tests)
  test_integrate.py  Verlet + Langevin BAOAB integrators (17 tests)
  test_minimize.py   L-BFGS-B minimization vs OpenMM (4 tests)
  test_serialization.py save/load roundtrip (2 tests)
examples/
  quickstart.ipynb           core API in 5 minutes
  energy_landscape.ipynb     PES visualization, free energy surfaces, basin analysis
  differentiable_md.ipynb    gradients through MD, Hessian, parameter sensitivity
  custom_energy.ipynb        restraints, dihedral bias, umbrella sampling + WHAM
  normal_modes.ipynb         Hessian eigendecomposition, vibrational frequencies
  solvent_comparison.ipynb   vacuum vs implicit solvent side-by-side
  free_energy.ipynb          histogram FES, convergence, log_boltzmann_regularized
  custom_samplers.ipynb      simulated tempering, HMC via baoab_step
  jaxmm_demo.ipynb           energy comparison, vmap, gradients, timing, MD
  parallel_tempering.ipynb   replica exchange MD, Ramachandran comparison
  aldp_potential_jaxmm.ipynb jaxmm version of aldp_potential
```

## Tests

```bash
python -m pytest tests/ -v
```

155 tests, ~44s. Energy terms validated against OpenMM on alanine dipeptide (22 atoms) across 50 MD frames for both vacuum and implicit solvent systems. Integrators validated against OpenMM trajectories and statistical mechanics (equipartition, harmonic variance).

## Validation summary

| Check | Result |
|-------|--------|
| Per-term energy vs OpenMM | < 1e-4 kJ/mol (bonds, angles), < 1e-3 (torsions, nonbonded) |
| RB torsions vs OpenMM | < 1e-4 kJ/mol across MD frames |
| CMAP correction vs OpenMM | < 0.5 kJ/mol (bilinear interpolation, JAX order<=1 limit) |
| PBC nonbonded vs OpenMM | < 1e-4 kJ/mol (CutoffPeriodic with switching) |
| GBSA energy vs OpenMM | < 1e-3 kJ/mol across 50 MD frames (implicit solvent) |
| Total energy vs OpenMM | < 1e-3 kJ/mol across 50 MD frames (vacuum and implicit) |
| Gradients vs finite differences | < 1e-3 kJ/mol/nm |
| Gradients vs OpenMM forces | < 1e-2 kJ/mol/nm (residual from CMMotionRemover) |
| jit+vmap speedup | ~234x over sequential OpenMM (200 configs, CPU) |

## Scope and guardrails

This library computes potential energy (vacuum, implicit solvent, or periodic systems with cutoff) and runs MD for small molecules (< ~1500 atoms). It does **not** include:

- Neighbor lists (all-pairs computation, O(N^2))
- Force field parameter assignment (OpenMM handles this)
- Long-range electrostatics (PME/Ewald)
- Long-range dispersion correction

`extract_params` raises `ValueError` for unsupported systems:

- **Constraints**: use `constraints=None` when building the OpenMM system
- **Virtual sites**: TIP4P/TIP5P water models are not supported
- **PME/Ewald/LJPME**: only NoCutoff, CutoffNonPeriodic, and CutoffPeriodic are supported
- **Unknown force types**: only the forces listed in the API table are handled

`total_energy` and `energy_components` validate that positions shape matches `params.n_atoms`.

## Notebook utilities

`jaxmm.notebook` provides reusable helpers for Jupyter notebooks. Not re-exported by `jaxmm.__init__`; import explicitly:

```python
from jaxmm.notebook import show_structure, animate_trajectory, plot_ramachandran
```

| Function | Description |
|----------|-------------|
| `show_structure(pos, topology)` | 3D structure viewer via py3Dmol |
| `animate_trajectory(traj, topology)` | Animated MD trajectory (subsampled, COM-removed) |
| `animate_mode(pos_eq, mode, masses, topology)` | Normal mode oscillation animation |
| `phi_psi_degrees(traj, topology)` | Backbone phi/psi dihedrals in degrees |
| `plot_ramachandran(phi, psi)` | Hexbin Ramachandran plot |
| `free_energy_1d(samples, T)` | 1D free energy F = -kBT ln P |
| `free_energy_2d(x, y, T)` | 2D free energy surface |

Extra dependencies (lazy-imported): `matplotlib`, `py3Dmol`, `openmm`.
