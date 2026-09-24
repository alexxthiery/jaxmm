# AGENTS.md

Context file for LLM agents working on this codebase.
For tool-specific instructions, see also `.claude/CLAUDE.md` (Claude Code).

## What this is

Pure JAX molecular potential energy library. OpenMM extracts force field
parameters once into frozen dataclasses; energy evaluation is pure JAX
(jit, vmap, grad). Built for training normalizing flows against Boltzmann
targets without the CPU-GPU bottleneck of calling OpenMM per sample.

## Architecture

Two phases:

1. **Extraction** (one-time, requires OpenMM): `extract_params(system)` reads
   an `openmm.System` and returns `ForceFieldParams`, a frozen dataclass
   containing JAX arrays. Parameters can be saved to `.npz` and loaded
   without OpenMM.

2. **Evaluation** (pure JAX, no OpenMM): energy functions are pure
   `(positions, params) -> scalar`, compatible with `jit`, `vmap`, `grad`.

```
OpenMM System -> extract_params() -> ForceFieldParams (frozen dataclass, pytree)
                                          |
                                          v
                                 total_energy(positions, params) -> scalar kJ/mol
                                          |
                                 +--------+--------+
                                 |                 |
                                 v                 v
                            jax.grad          jax.vmap / jax.jit
                            (forces)          (batch eval)
                                 |
                                 v
                      minimize_energy() / verlet() / langevin_baoab()
```

## Module dependency graph

```
extract.py          <-- jax, numpy (openmm imported lazily, only in extract_params)
    |
    v
energy.py           <-- jax, extract.py (dataclass types only)
    |
    +--------+-------------+
    |        |             |
    v        v             v
integrate.py  coordinates.py ----+
    |               |              |
    v               v              v
  jax, energy.py, jax, numpy    utils.py
  utils.py (KB)   (energy.py    (extract.py, energy.py,
                   _check_x64,   coordinates.py for
                   _check_       log_boltzmann_internal)
                   positions_    |
                   shape)        v
                               jaxopt

notebook.py         <-- utils.py (KB, dihedral_angle), extract.py (phi/psi_indices)
                        matplotlib, py3Dmol, openmm (all lazy-imported)
```

No circular dependencies. `energy.py` imports only type definitions from
`extract.py`, never extraction logic. `coordinates.py` is pure geometry: it
touches no energy term and borrows only private helpers (`_check_x64`,
`_check_positions_shape`, `_register_pytree`). `utils.py` depends on
`coordinates.py` for `log_boltzmann_internal`. `notebook.py` is not re-exported from `__init__.py`.

## Key patterns

- **Energy function signature**: `(positions: jax.Array, params: ParamType) -> jax.Array`
- **Parameter containers**: frozen `@dataclass(frozen=True)`, registered as JAX pytrees via `_register_pytree()`
- **Batch evaluation**: `jax.vmap(jaxmm.total_energy, in_axes=(0, None))`
- **Gradient safety**: `jnp.sqrt(r_sq + 1e-30)` instead of `jnp.linalg.norm` (avoids NaN grad at zero distance)
- **Integrators**: nested `jax.lax.scan` (inner for stepping, outer for trajectory saving)
- **Optional fields**: `ForceFieldParams` has optional fields (gbsa, rb_torsions, cmap, restraints) that are `None` when absent; custom flatten/unflatten handles this
- **Internal coordinates**: `coordinates.py` maps between a fixed-frame z-matrix and
  Cartesian positions and supplies `log |det J|`, the pieces a normalizing flow needs
- **Metadata validation**: host-side (numpy) and separate from the traced transforms, so
  `validate_zmatrix` raises clear errors at setup without breaking jit
- **Defensive boundary checks**: validate at every public entry point and fail loudly. A
  wrong answer that looks right is worse than an exception
- **Examples are plain Python in one folder each**: `examples/<name>/<name>.py` in
  jupytext percent format, where `# %%` marks a cell. Edit them like any Python file.
  Do not add `.ipynb` or commit figures; `tests/test_examples.py` fails if one appears
- **Example READMEs are generated**, not written: `tools/sync_example_docs.py` derives
  them from each script's leading markdown cell. Edit the cell, then rerun the tool.
  Figures come from `tools/render_examples.py`, which redirects `plt.show` to
  `savefig` so the examples themselves stay unchanged
- **Testing**: each energy term validated against an isolated single-force OpenMM system, not the force group API

## Invariants

Breaking any of these will break downstream users:

1. **Pure functions.** No side effects, no hidden state, no closures over mutable data.
2. **JAX pytree compatibility.** All param dataclasses registered via `_register_pytree()`. Non-array fields go in `aux_field_names`.
3. **float64 precision.** All arrays float64 (or int32 for indices). Runtime check via `_check_x64()`.
4. **Units.** Positions nm, energies kJ/mol, angles radians, charges elementary charge, time ps, masses amu.
5. **Gradient safety.** Finite gradients for any physically reasonable configuration.

## Dev commands

```bash
# Environment: any env with jax, numpy, jaxopt, openmm, openmmtools, pytest.
# The env name is machine specific, so check what you have rather than
# assuming one: `conda env list`, then verify with
#   python -c "import jax, openmm, openmmtools, jaxopt"

# Run all tests (324 tests; 2 skip without py3Dmol)
python -m pytest tests/ -v

# Run a single test file
python -m pytest tests/test_bonds.py -v

# Run tests matching a keyword
python -m pytest tests/ -v -k "bond"

# Quick smoke test (fastest subset)
python -m pytest tests/test_bonds.py tests/test_angles.py tests/test_torsions.py -v

# Internal-coordinate transforms only
python -m pytest tests/test_coordinates.py -v

# Float64 must be enabled before importing jaxmm
python -c "import jax; jax.config.update('jax_enable_x64', True); import jaxmm; print('OK')"
```

## Gotchas

These are the most common sources of bugs and confusion:

- **JAX float64**: requires `jax.config.update("jax_enable_x64", True)` before any jaxmm import. Energy functions check at runtime via `_check_x64()`.
- **NaN gradients from `jnp.linalg.norm`**: when the input vector can be zero (e.g., self-distance), norm returns 0 but its gradient is NaN. Use `jnp.sqrt(jnp.sum(x**2) + 1e-30)`.
- **Frozen dataclass pytree registration**: JAX cannot flatten/unflatten frozen dataclasses automatically. Call `_register_pytree(YourClass)` after definition. Non-array fields (int, str) go in `aux_field_names`.
- **OpenMM force group API**: unreliable for per-force energies. Use isolated single-force systems (`get_openmm_force_energy` in `conftest.py`).
- **CMMotionRemover**: OpenMM adds a small force correction (~6e-3 kJ/mol/nm) not modeled in jaxmm. Gradient tests use 1e-2 tolerance.
- **CMAP bilinear limit**: `jax.scipy.ndimage.map_coordinates` only supports order<=1. CMAP uses bilinear interpolation with ~0.13 kJ/mol difference vs OpenMM's bicubic. In kT at 300 K that is ~0.05 typical, ~0.2 worst case: fine for structure and dynamics, but it can bias a free energy or reweighting estimate. Every other term agrees to under 1e-3 kT. Judge tolerances in kT, not kJ/mol.
- **OpenMM Verlet is leapfrog**: `setVelocities` sets v(t-dt/2), not v(t). Pre-kick by -dt/2*F/m to match velocity Verlet.
- **JIT reordering**: tiny floating-point differences (~1e-10). Use 1e-8 tolerance for JIT consistency tests.
- **Constraints**: `extract_params` raises `ValueError` for constrained systems. Always use `constraints=None` when building OpenMM systems.
- **JAX clamps out-of-bounds gathers instead of raising**: an atom index past the end of
  `positions`, or a z-matrix sized for a different molecule, silently yields a finite,
  plausible, wrong number. Every public entry point now validates. Static checks (shape,
  atom count) always run; index-bound checks read values, so they run eagerly and are
  skipped under jit and vmap where the arrays are tracers. If you add an entry point that
  gathers by index, add the check.
- **`log_boltzmann` is the Cartesian density**: for internal-coordinate samples use
  `log_boltzmann_internal`, which adds `log |det J|`. The term is about -84 nats for
  alanine dipeptide, roughly 84 kBT. Omitting it is the standard silent error when
  training a flow in internal coordinates.
- **Backbone phi and psi are explicit z-matrix torsions**: `torsions[11]` and
  `torsions[13]` for the shipped ALDP z-matrix, matching `phi_indices` and `psi_indices`.
  That is why these coordinates suit a flow, and why the sign gotcha below matters.
- **Z-matrix torsions are the negative of `dihedral_angle`**: `coordinates.py` follows
  OpenMM, `utils.dihedral_angle` negates to match mdtraj. A Ramachandran plot built from
  z-matrix torsions without negating is mirrored and looks plausible.
- **Fixed frame pins the first three atoms**: `zmatrix_to_cartesian` places atom 2
  relative to atom 1 at an angle measured against atom 0, so `bond_ref[2]` must be 1
  and `angle_ref[2]` must be 0. `validate_zmatrix` enforces this. Before 2026-09-23 it
  did not, and a z-matrix with `bond_ref[2] == 0` silently round-tripped to different
  bond lengths with no error.
- **Z-matrix references must be distinct**: `angle_ref[i] != bond_ref[i]`, and
  `torsion_ref[i]` differs from both. A repeated reference leaves the angle or torsion
  undefined. Also enforced by `validate_zmatrix`, also silently wrong before.
- **`arccos` has a NaN gradient at 0 and pi**: use `atan2(|u x v|, u . v)` for angles, the
  same form as `energy.py:angle_energy`. This is the angle analogue of the
  `jnp.linalg.norm` rule and bit `cartesian_to_zmatrix` for exactly the same reason.
- **Internal coordinates are genuinely singular** at zero bond length and at collinear
  an angle of 0 or pi, where `zmatrix_log_abs_det_jacobian` goes to `-inf`. That is correct.
  It is a function of the bonds and angles alone, so a collinear reference triple leaves it finite
  while making the construction ill-posed. `zmatrix_in_domain` answers the chart question instead.
  Do not clamp it. Note `sin(pi)` is 1.2e-16 rather than 0 in float64, so the divergence
  shows up as a limit rather than an exact `-inf` at exactly pi.
- **`lax.fori_loop` traces its body even when the trip count is zero**, so a triatomic
  (`n_atoms == 3`, empty torsion array) needs an explicit early return, not just a loop
  that happens not to run.
- **Five copies of the dihedral formula exist on purpose**: three in `energy.py`, one in
  `utils.py`, one in `coordinates.py`. `utils.dihedral_angle` negates to match the
  biochemistry/mdtraj convention; the rest match OpenMM's `PeriodicTorsionForce`. Do not
  unify them.
- **A stray top-level `tests` package can shadow this repo's**: test files import
  `from conftest import ...`, not `from tests.conftest import ...`. `tests/` has no
  `__init__.py`, so it is only a namespace portion, and any regular `tests` package
  elsewhere on `sys.path` (an editable install of an unrelated project, for example) wins
  outright. A `pythonpath` setting does not fix that; the plain `conftest` import does.
- **GBSA OBC variant**: openmmtools `AlanineDipeptideImplicit` uses `CustomGBForce` with OBC1 tanh parameters (alpha=0.8, beta=0, gamma=2.909125), not OBC2 or `GBSAOBCForce`.

## How to add a new energy term

See CONTRIBUTING.md for the step-by-step template. Summary:

1. Add frozen dataclass in `extract.py`, register as pytree
2. Add extraction function in `extract.py`, wire into `extract_params()`
3. Add energy function in `energy.py` following `(positions, params) -> scalar` pattern
4. Wire into `total_energy()` and `energy_components()`
5. Write tests in `tests/test_new_term.py` (vs OpenMM, gradients, JIT)
6. Export from `__init__.py`
7. Update README.md and CONTRIBUTING.md
8. Run full test suite: `python -m pytest tests/ -v`

## File reading order

To orient in the codebase, read in this order:

1. `README.md` -- what it does, quick start, API overview
2. `CONTRIBUTING.md` -- architecture, patterns, how to add features
3. `CODEMAP.md` -- structural overview, dependency graph
4. `jaxmm/__init__.py` -- public API at a glance (44 exports)
5. `jaxmm/energy.py` (first 100 lines) -- energy function pattern
6. `jaxmm/extract.py` (first 100 lines) -- parameter dataclass pattern
7. `jaxmm/coordinates.py` (module docstring) -- fixed-frame conventions and singularities
8. `tests/conftest.py` (first 80 lines) -- test fixture setup
9. `examples/quickstart/quickstart.py` -- jupytext percent format, plain Python

## Scope and limitations

- Small molecules only (< ~1500 atoms, O(N^2) all-pairs computation)
- No neighbor lists, no PME/Ewald, no long-range dispersion correction
- No force field parameter assignment (OpenMM handles this)
- No explicit solvent (implicit solvent via GBSA is supported)
- `zmatrix_in_domain` is the chart predicate, `r > 0` and `theta in (0, pi)`, both open. Off that
  chart the internal-to-Cartesian map is exactly 2-to-1, since `x(theta, phi) == x(-theta, phi + pi)`,
  so a density built from the Jacobian there double counts. `log_boltzmann_internal` returns `-inf`
  off the chart and substitutes its inputs first, so a caller that differentiates it gets no NaN.
- The Jacobian takes absolute values. Without them an out-of-range angle returned NaN, which is worse
  than a wrong number for a sampler: a threshold test against NaN is False, so the configuration is
  discarded in silence rather than rejected loudly.
- Whitening is a separate map from the z-matrix, with its own constant log-Jacobian, and the two
  compose additively. For alanine dipeptide they are about `-85` and `-149` nats, so the whitening
  term is the larger; it is a constant, which is why omitting it never shows up in training.
- `whitened_chart_bounds` bounds a flow domain from the *chart*, never from data. Data bounds would
  admit a negative bond length. On a 300 K trajectory the nearest bound is about 8 standard
  deviations away, so the restriction costs nothing physical.
- Construction order is separate from atom order: `ZMatrix.atom_order[c]` is the atom placed at
  step `c`, references are indexed by construction step so `ref[c] < c` holds, and only the two
  boundaries permute. `None` means construction order is atom order. This is what lets the ALDP
  frame be the backbone, and what an automatically built z-matrix needs.
- The ALDP frame is C, CA, N of the alanine residue, so CA's substituents are pinned against a
  frame triple and chirality is a single coordinate (`torsions[0]`), not a difference. `torsions[12]`
  is phi and `torsions[5]` is psi. Rooting anywhere that moves with phi loses all three properties.
- Internal coordinates are fixed-frame only: no free rigid-body degrees of freedom,
  so the map covers molecular shape, not absolute position or orientation
