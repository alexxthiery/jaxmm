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
| `log_boltzmann(pos, params, T)` | `-E / (kB * T)`, the **Cartesian** density |
| `log_boltzmann_internal(z, bonds, angles, torsions, params, T)` | Boltzmann density in internal coordinates, Jacobian included |
| `log_boltzmann_regularized(pos, params, T, cut, max)` | With energy clamping for numerical stability |
| `phi_indices(topology)` | Backbone phi dihedral atom indices from OpenMM Topology |
| `psi_indices(topology)` | Backbone psi dihedral atom indices from OpenMM Topology |
| `dihedral_angle(positions, indices)` | Compute dihedral angles from positions + index array |
| `zmatrix_to_cartesian(z, bonds, angles, torsions)` | Internal coordinates to positions in the fixed frame |
| `cartesian_to_zmatrix(z, pos)` | Positions to internal coordinates |
| `canonicalize_cartesian(z, pos)` | Remove rigid translation and rotation |
| `zmatrix_log_abs_det_jacobian(z, bonds, angles)` | `log \|det J\|` of the internal-to-Cartesian map |
| `validate_zmatrix(z)` | Host-side check of z-matrix metadata (raises `ValueError`) |
| `zmatrix_in_domain(bonds, angles)` | Whether internals lie on the chart, `r > 0` and `theta` in `(0, pi)`, per sample |
| `internal_whitener(z, positions)` | Affine whitener for bonds and angles, centred on a reference structure |
| `whiten_internals(w, bonds, angles)` | Bonds and angles to whitened coordinates |
| `unwhiten_internals(w, u_bonds, u_angles)` | The inverse |
| `whitener_log_abs_det_jacobian(w)` | `log \|det d(bonds, angles)/du\|`, a constant |
| `whitened_chart_bounds(w)` | The whitened image of the chart, per coordinate |
| `aldp_zmatrix()` | Z-matrix for the 22-atom openmmtools alanine dipeptide, rooted on the backbone |
| `verlet(pos, vel, params, dt, n, ...)` | Velocity Verlet integrator (symplectic, energy-conserving) |
| `langevin_baoab(pos, vel, params, dt, T, friction, n, *, key)` | Langevin BAOAB thermostat (second-order, ergodic) |
| `kinetic_energy(vel, masses)` | `0.5 * sum(m * v^2)` in kJ/mol |
| `baoab_step(pos, vel, forces, key, params, T, dt, friction)` | Single BAOAB Langevin step (building block for custom integrators) |
| `minimize_energy(pos, params, tol, max_iter)` | L-BFGS energy minimization via jaxopt (pure JAX, GPU-compatible) |

Both integrators return `MDTrajectory(positions, velocities, trajectory_positions, trajectory_velocities)`, a NamedTuple that also supports tuple unpacking.

**Units**: positions in nm, velocities in nm/ps, energies in kJ/mol, masses in amu, time in ps, angles in radians, charges in elementary charge units.

**Unit constants**: `FEMTOSECOND` (1e-3 ps), `ANGSTROM` (0.1 nm), `KCAL_PER_MOL` (4.184 kJ/mol), `KB` (Boltzmann constant, kJ/(mol*K)). Multiply to convert: `dt = 2.0 * jaxmm.FEMTOSECOND` gives 0.002 ps.

## Internal coordinates

For training normalizing flows against a Boltzmann target it is usually better to
work in internal coordinates than in Cartesian ones: bond lengths and angles are
stiff and nearly Gaussian, torsions carry the interesting multimodality, and the
six rigid-body degrees of freedom disappear.

`jaxmm.coordinates` provides that change of variables and the log Jacobian the
change of variables formula needs.

```python
z = jaxmm.aldp_zmatrix()          # 22-atom alanine dipeptide
jaxmm.validate_zmatrix(z)         # host-side check, do this once at setup

bonds, angles, torsions = jaxmm.cartesian_to_zmatrix(z, positions)
positions = jaxmm.zmatrix_to_cartesian(z, bonds, angles, torsions)

# Boltzmann density in internal coordinates, change of variables included
logp = jaxmm.log_boltzmann_internal(z, bonds, angles, torsions, params, 300.0)
```

**Use `log_boltzmann_internal`, not `log_boltzmann`, for internal-coordinate
samples.** `log_boltzmann` returns the density with respect to Cartesian
coordinates. The change of variables adds `log |det J|`, which for alanine
dipeptide is about `-84` nats, or 210 kJ/mol, or 84 kBT. Omitting it is silent
and large enough to make any reweighting or free-energy estimate meaningless.

**Backbone phi and psi are explicit z-matrix torsions.** For the shipped ALDP
z-matrix, `torsions[12]` is phi and `torsions[5]` is psi, matching
`phi_indices` and `psi_indices` exactly. This is what makes these coordinates
suitable for a flow: the slow collective variables are sampled directly rather
than being nonlinear functions of the sampled variables.

**Whiten bonds and angles before handing them to a flow.** Measured on a
300 K trajectory of alanine dipeptide, raw bond standard deviations run 0.0016
to 0.0036 nm while angles run 0.049 to 0.088 rad, two orders of magnitude
apart. After whitening both blocks sit at 0.32 to 0.72, so a flow spends its
capacity on the torsions, which carry the multimodality, rather than on the
scale difference.

The scales are declared, not fitted, as in both reference implementations
(`BOND_SCALE`, `ANGLE_SCALE`). The centre is a reference structure's own
internals, normally an energy-minimized one, so that structure whitens to the
origin where a flow's base sits.

**Add `whitener_log_abs_det_jacobian` to the density, not just the z-matrix
term.** For alanine dipeptide the two are about `-85` and `-149` nats, so the
whitening term is the larger of the pair. Being a constant is exactly what
makes it easy to omit, invisible during training, and fatal to any free energy.

`whitened_chart_bounds` gives the whitened image of `r > 0` and
`theta in (0, pi)`, which is what a bounded flow domain needs. Bounds from data
would be wrong: they would let a flow place mass on a negative bond length. On
the same trajectory the nearest bound sits about 8 standard deviations from the
sampled range, so bounding costs nothing physical.

**Construction order is separate from atom order.** `ZMatrix.atom_order[c]` is
the atom placed at construction step `c`; the reference arrays are indexed by
construction step, so `ref[c] < c` always holds and the frame is steps 0, 1, 2.
Only the two boundaries permute, so a caller never handles a construction
index: `cartesian_to_zmatrix` takes and `zmatrix_to_cartesian` returns
positions in atom order. Leave `atom_order` as `None` and construction order is
atom order, as before.

This is what lets the ALDP frame be the rigid backbone, C, CA and N, rather
than whichever atoms happen to be numbered first. It is also what an
automatically built z-matrix needs, since those are not produced in atom order.

**Chirality is a single coordinate.** Because CA's substituents are measured
against a reference triple lying entirely in the frame, each is a stiff
improper near `+/-120` degrees whose sign flips under reflection. `torsions[0]`
is one of them, so restricting it to half a circle is an exact fundamental
domain that selects one enantiomer, at no cost and with no rejection.

Rooting the frame anywhere that moves with phi loses this: the substituents
then sweep together, no single torsion carries the sign, and a half-circle
restriction cuts the Ramachandran circle instead, deleting the alpha-L basin.
The mirror map in these coordinates is exactly `t -> -t` on every torsion, with
bonds and angles unchanged.

**Z-matrix torsions carry the opposite sign to `dihedral_angle`.**
`coordinates.py` follows OpenMM's `PeriodicTorsionForce`; `utils.dihedral_angle`
negates to match the biochemistry and mdtraj convention. Both are correct in
their own context and neither will change. A Ramachandran plot built from
z-matrix torsions without negating is mirrored, and looks entirely plausible:

```python
phi_zmat = -torsions[12]   # now matches jaxmm.dihedral_angle
psi_zmat = -torsions[5]
```

**The fixed frame.** Atom 0 sits at the origin, atom 1 on the positive x-axis, and
atom 2 in the xy-plane. That removes the six rigid degrees of freedom and makes the
map square: `3N - 6` internal coordinates in, `3N - 6` free Cartesian coordinates out.

**The frame constrains the first three atoms.** `bond_ref[2]` must be 1 and
`angle_ref[2]` must be 0, because atom 2 is placed relative to atom 1 at an angle
measured against atom 0. Every reference triple must also name three distinct atoms.
`validate_zmatrix` enforces both. Call it once when you build a z-matrix; the
transforms themselves stay jit-traceable and do not re-check.

**The volume element** is `r_2` for atom 2 and `r_i^2 sin(theta_i)` for each
atom `i >= 3`. Atom 1 contributes 1 because it is pinned to the x-axis.

**The map is singular** where a bond length is zero or a reference triple is
`zmatrix_log_abs_det_jacobian` is a function of the bond lengths and angles alone, so it is `-inf` exactly at a zero bond length or an angle of 0 or pi, and finite everywhere else, including off the chart. It says nothing about whether a reference triple is collinear: that makes the *construction* ill-posed while the determinant stays finite. Ask `zmatrix_in_domain` whether a configuration is on the chart; that is a different question.
That is a property of internal coordinates, not a defect. Away from that
measure-zero set every transform returns finite gradients, including at geometries
that are exactly collinear.

**Chirality is preserved.** Torsions are signed, so a molecule and its mirror image
map to different internal coordinates. `canonicalize_cartesian` removes rigid motion
without collapsing enantiomers.

## Project structure

```
jaxmm/
  __init__.py        public API (44 exports)
  extract.py         OpenMM System -> ForceFieldParams dataclass
  energy.py          pure JAX energy functions
  integrate.py       Verlet and Langevin BAOAB integrators (pure JAX)
  utils.py           minimize_energy, log_boltzmann, dihedral_angle, serialization
  coordinates.py     fixed-frame z-matrix transforms and their log Jacobian
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
  test_total.py      total energy, vmap, log_boltzmann, energy_components, composability (28 tests)
  test_dihedral.py   phi/psi dihedral angles (6 tests)
  test_grad.py       gradients vs OpenMM forces + finite diff (8 tests)
  test_integrate.py  Verlet + Langevin BAOAB integrators (17 tests)
  test_minimize.py   L-BFGS-B minimization vs OpenMM (4 tests)
  test_serialization.py save/load roundtrip incl. every optional field (8 tests)
  test_coordinates.py  z-matrix transforms, Jacobian, gradient safety, density (57 tests)
  test_validation.py   defensive input validation across energy terms (12 tests)
  test_notebook.py     free energy, backbone angles, PDB writer (11 tests)
  test_examples.py     examples compile, use real API, docs in sync, one runs (82 tests)
examples/
  quickstart.py           core API in 5 minutes
  energy_landscape.py     PES visualization, free energy surfaces, basin analysis
  differentiable_md.py    gradients through MD, Hessian, parameter sensitivity
  custom_energy.py        restraints, dihedral bias, umbrella sampling + WHAM
  normal_modes.py         Hessian eigendecomposition, vibrational frequencies
  solvent_comparison.py   vacuum vs implicit solvent side-by-side
  free_energy.py          histogram FES, convergence, log_boltzmann_regularized
  custom_samplers.py      simulated tempering, HMC via baoab_step
  jaxmm_demo.py           energy comparison, vmap, gradients, timing, MD
  parallel_tempering.py   replica exchange MD, Ramachandran comparison
  aldp_potential_jaxmm.py jaxmm version of aldp_potential
```

## Examples

Each example is a self-contained folder under `examples/`: a jupytext "percent"
Python script, a README, and a gitignored `output/` for figures. See
[examples/README.md](examples/README.md) for the index.

```bash
pip install -e ".[examples]"                      # matplotlib, py3Dmol, jupytext
python examples/quickstart/quickstart.py          # run one
python tools/render_examples.py                   # run all, save figures
python tools/render_examples.py --only quickstart # just one
```

These are plain Python, not notebooks. `# %%` marks a cell, so Jupyter and VS Code
open them as notebooks with jupytext installed, and they also run as ordinary
scripts. Notebooks are deliberately not committed: stored outputs are base64 blobs
that bloated this directory to ten times the size of the library and produced diffs
nobody could review.

Figures are not committed either. The examples draw 35 of them, about 40 KB each,
so committing them would restore roughly the 1.2 MB the notebooks cost. Run the
renderer to produce them locally; it redirects `plt.show()` to `savefig` so the
examples need no changes.

Because they are plain Python, the examples are tested. `tests/test_examples.py`
walks each one's AST and fails if it references a jaxmm function that no longer
exists, checks the generated READMEs are in sync, and runs one end to end.

## Tests

```bash
python -m pytest tests/ -v
```

324 tests (322 run, 2 skip without py3Dmol). Energy terms validated against OpenMM on alanine dipeptide (22 atoms) across 50 MD frames for both vacuum and implicit solvent systems. Integrators validated against OpenMM trajectories and statistical mechanics (equipartition, harmonic variance).

## Validation summary

| Check | Result |
|-------|--------|
| Per-term energy vs OpenMM | < 1e-4 kJ/mol (bonds, angles), < 1e-3 (torsions, nonbonded) |
| RB torsions vs OpenMM | < 1e-4 kJ/mol across MD frames |
| CMAP correction vs OpenMM | < 0.5 kJ/mol (bilinear interpolation, JAX order<=1 limit). See the note below |
| PBC nonbonded vs OpenMM | < 1e-4 kJ/mol (CutoffPeriodic with switching) |
| GBSA energy vs OpenMM | < 1e-3 kJ/mol across 50 MD frames (implicit solvent) |
| Total energy vs OpenMM | < 1e-3 kJ/mol across 50 MD frames (vacuum and implicit) |
| Gradients vs finite differences | < 1e-3 kJ/mol/nm |
| Gradients vs OpenMM forces | < 1e-2 kJ/mol/nm (residual from CMMotionRemover) |
| jit+vmap speedup | ~234x over sequential OpenMM (200 configs, CPU) |

**A note on the CMAP tolerance.** At 300 K, kB*T is 2.4943 kJ/mol. Every other term
above agrees with OpenMM to under 1e-3 kT, but CMAP's bilinear interpolation sits at
roughly 0.05 kT typical and 0.2 kT worst case. That is harmless for structure and
dynamics, where kT-scale fluctuations dominate, but it is large enough to bias a free
energy difference or a reweighting estimate. If you need CMAP at full accuracy for a
thermodynamic quantity, evaluate that term in OpenMM. The cause is a JAX limitation:
`jax.scipy.ndimage.map_coordinates` supports order <= 1, so OpenMM's bicubic
interpolation cannot be reproduced.

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

`jaxmm.notebook` provides reusable helpers for interactive work (Jupyter, or an
example opened as a notebook). Not re-exported by `jaxmm.__init__`; import explicitly:

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
