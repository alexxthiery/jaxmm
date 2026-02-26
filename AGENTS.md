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
    +--------+
    |        |
    v        v
utils.py   integrate.py
    |          |
    v          v
  jaxopt     jax, energy.py, utils.py (KB constant)

notebook.py         <-- utils.py (KB, dihedral_angle), extract.py (phi/psi_indices)
                        matplotlib, py3Dmol, openmm (all lazy-imported)
```

No circular dependencies. `energy.py` imports only type definitions from
`extract.py`, never extraction logic. `notebook.py` is not re-exported
from `__init__.py`.

## Key patterns

- **Energy function signature**: `(positions: jax.Array, params: ParamType) -> jax.Array`
- **Parameter containers**: frozen `@dataclass(frozen=True)`, registered as JAX pytrees via `_register_pytree()`
- **Batch evaluation**: `jax.vmap(jaxmm.total_energy, in_axes=(0, None))`
- **Gradient safety**: `jnp.sqrt(r_sq + 1e-30)` instead of `jnp.linalg.norm` (avoids NaN grad at zero distance)
- **Integrators**: nested `jax.lax.scan` (inner for stepping, outer for trajectory saving)
- **Optional fields**: `ForceFieldParams` has optional fields (gbsa, rb_torsions, cmap, restraints) that are `None` when absent; custom flatten/unflatten handles this
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
# Environment
conda activate chemistry

# Run all tests (155 tests, ~44s)
python -m pytest tests/ -v

# Run a single test file
python -m pytest tests/test_bonds.py -v

# Run tests matching a keyword
python -m pytest tests/ -v -k "bond"

# Quick smoke test (fastest subset)
python -m pytest tests/test_bonds.py tests/test_angles.py tests/test_torsions.py -v

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
- **CMAP bilinear limit**: `jax.scipy.ndimage.map_coordinates` only supports order<=1. CMAP uses bilinear interpolation with ~0.13 kJ/mol difference vs OpenMM's bicubic.
- **OpenMM Verlet is leapfrog**: `setVelocities` sets v(t-dt/2), not v(t). Pre-kick by -dt/2*F/m to match velocity Verlet.
- **JIT reordering**: tiny floating-point differences (~1e-10). Use 1e-8 tolerance for JIT consistency tests.
- **Constraints**: `extract_params` raises `ValueError` for constrained systems. Always use `constraints=None` when building OpenMM systems.
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
4. `jaxmm/__init__.py` -- public API at a glance (40 exports)
5. `jaxmm/energy.py` (first 100 lines) -- energy function pattern
6. `jaxmm/extract.py` (first 100 lines) -- parameter dataclass pattern
7. `tests/conftest.py` (first 80 lines) -- test fixture setup
8. `examples/quickstart.ipynb` -- working example code

## Scope and limitations

- Small molecules only (< ~1500 atoms, O(N^2) all-pairs computation)
- No neighbor lists, no PME/Ewald, no long-range dispersion correction
- No force field parameter assignment (OpenMM handles this)
- No explicit solvent (implicit solvent via GBSA is supported)
