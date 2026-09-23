# Contributing to jaxmm

## Architecture

jaxmm has two phases:

1. **Extraction** (one-time, uses OpenMM): `extract.py` reads an `openmm.System`, pulls out force parameters, and stores them in frozen dataclasses containing JAX arrays.

2. **Evaluation** (pure JAX, no OpenMM): `energy.py` contains pure functions `(positions, params) -> scalar` that are compatible with `jax.jit`, `jax.vmap`, and `jax.grad`.

OpenMM is only imported inside `extract.py` (and only at call time, not at module level). The energy functions never touch OpenMM.

### Data flow

```
OpenMM System
    |
    v
extract_params()  -->  ForceFieldParams (frozen dataclass, JAX pytree)
                            |
                            v
                   total_energy(positions, params) --> scalar (kJ/mol)
                            |
                            v
                   jax.grad / jax.vmap / jax.jit (all work)
```

### Key files

| File | Role | Depends on |
|------|------|------------|
| `jaxmm/extract.py` | Dataclasses + extraction from OpenMM + backbone indices | openmm (import-time: jax, numpy only) |
| `jaxmm/energy.py` | Energy functions | jax, extract.py (dataclass types only) |
| `jaxmm/integrate.py` | Verlet, Langevin BAOAB, and `baoab_step` (single-step building block) | jax, energy.py |
| `jaxmm/utils.py` | minimize_energy, log_boltzmann, log_boltzmann_internal, dihedral angles, unit constants, serialization | energy.py, coordinates.py, numpy, jaxopt |
| `jaxmm/coordinates.py` | Fixed-frame z-matrix transforms, log Jacobian, ALDP z-matrix | jax, numpy, `_check_x64` from energy.py, `_register_pytree` from extract.py |
| `jaxmm/notebook.py` | Jupyter helpers: 3D viz, Ramachandran, free energy estimation | matplotlib, py3Dmol, openmm (all lazy) |
| `tests/conftest.py` | ALDP fixtures (vacuum + implicit) + OpenMM reference helpers | openmm, openmmtools, jaxmm |

### Entry points

- **Library**: `import jaxmm; params = jaxmm.extract_params(system)`
- **Tests**: `python -m pytest tests/ -v` (324 tests, 93% branch coverage)
- **Examples**: `examples/quickstart/quickstart.py` (start here), plus 10 topic folders.
  Each is a jupytext "percent" script with a generated README and a gitignored
  `output/`. Run them directly, or open as notebooks with jupytext installed. Do not
  commit `.ipynb` or figures. After editing an example's leading markdown cell, run
  `python tools/sync_example_docs.py`; a test fails if the READMEs are stale.
- **Demo**: `examples/jaxmm_demo/jaxmm_demo.py`

### Dev commands

```bash
# Environment: any env with jax, numpy, jaxopt, openmm, openmmtools, pytest.
# The env name is machine specific. Check with `conda env list`, then verify:
#   python -c "import jax, openmm, openmmtools, jaxopt"

# Run all tests (324 tests; 2 skip without py3Dmol)
python -m pytest tests/ -v

# Run a single test file
python -m pytest tests/test_bonds.py -v

# Run tests matching a keyword
python -m pytest tests/ -v -k "bond"

# Quick smoke test (fastest subset)
python -m pytest tests/test_bonds.py tests/test_angles.py tests/test_torsions.py -v

# Validate a change: run full suite, check exit code
python -m pytest tests/ -v && echo "ALL PASS"
```

Float64 must be enabled before importing jaxmm. The test suite handles
this via conftest.py, but standalone scripts need:

```python
import jax
jax.config.update("jax_enable_x64", True)
```

### Patterns

- Frozen dataclasses as parameter containers, registered as JAX pytrees
- All energy functions: `(positions, params) -> scalar` (pure functions)
- Batch eval: `jax.vmap(jaxmm.total_energy, in_axes=(0, None))`
- Test validation: each energy term tested against an isolated OpenMM force
- Integrators use nested `jax.lax.scan`: inner scan for stepping, outer scan for trajectory saving
- Metadata validation is host-side numpy and separate from the traced transforms, so it can
  raise clear errors at setup without breaking jit (see `validate_zmatrix` in `coordinates.py`)

## Invariants

These must stay true. Breaking any of them will break downstream users.

1. **Pure functions.** Every energy function is `(positions, params) -> scalar`. No side effects, no hidden state, no closures over mutable data.

2. **JAX pytree compatibility.** All param dataclasses are registered as JAX pytrees (see `_register_pytree` in `extract.py`). New dataclasses must also be registered. Fields that are not JAX arrays (like `n_atoms: int`) go in `aux_field_names`.

3. **float64 precision.** All modules call `jax.config.update("jax_enable_x64", True)`. All arrays are float64 (or int32 for indices). Do not downcast to float32.

4. **Units.** Positions in nm. Energies in kJ/mol. Angles in radians. Charges in elementary charge units. These match OpenMM's internal unit system.

5. **Gradient safety.** Energy functions and coordinate transforms must produce finite gradients for any physically reasonable configuration. Avoid `jnp.linalg.norm` on vectors that can be zero (use `jnp.sqrt(r_sq + 1e-30)` instead) and avoid `arccos` for angles (use `atan2(|u x v|, u . v)` instead).

6. **Validate at every public entry point.** Static checks (position shape, atom count, internal-coordinate lengths) always run and survive jit. Index-bound checks need concrete values, so they run eagerly and are skipped under jit and vmap. Use `_check_positions_shape`, `_check_atom_count` and `_check_atom_indices` in `energy.py`; do not write a new ad hoc check.

7. **Loud rejection over silent wrongness.** If a configuration cannot be represented faithfully, raise with a message naming the offending index. `validate_zmatrix` exists because a z-matrix the transform could not honour used to be accepted and then silently reconstructed in the wrong place.

## How to add a new energy term

Follow this pattern. The existing terms (bond, angle, torsion, rb_torsion, cmap, restraint, nonbonded, gbsa) are all implemented this way.

### 1. Add the parameter dataclass in `extract.py`

```python
@dataclass(frozen=True)
class NewTermParams:
    """Docstring with units and shapes."""
    atom_i: jnp.ndarray      # (n_terms,) int32
    some_param: jnp.ndarray   # (n_terms,) float64, units

# Register as pytree (after the class definition, before extract_params)
_register_pytree(NewTermParams)
```

### 2. Add extraction logic in `extract.py`

Add a new `_extract_new_term(force) -> NewTermParams` function. Add a branch in `extract_params()` that dispatches on the OpenMM force type. Add the field to `ForceFieldParams`. For optional fields (like `gbsa`), use a custom flatten/unflatten instead of `_register_pytree` to handle `None` values.

### 3. Add the energy function in `energy.py`

```python
def new_term_energy(positions: jnp.ndarray, params: NewTermParams) -> float:
    """Docstring with formula, args, returns."""
    # Pure JAX computation
    return jnp.sum(...)
```

Add it to `total_energy()`.

### 4. Write tests

Create `tests/test_new_term.py` following the existing pattern:

```python
def test_new_term_energy_initial(aldp_system, aldp_positions, aldp_positions_jnp, aldp_params):
    """Energy at minimized positions matches OpenMM."""
    ref = get_openmm_force_energy(aldp_system, openmm.TheForceClass, aldp_positions)
    jax_e = float(new_term_energy(aldp_positions_jnp, aldp_params.new_term))
    assert abs(jax_e - ref) < TOLERANCE

def test_new_term_energy_md_frames(aldp_system, aldp_md_frames, aldp_params):
    """Energy matches OpenMM across 10 MD frames."""
    ...

def test_new_term_energy_grad(aldp_positions_jnp, aldp_params):
    """Gradient has no NaN or Inf."""
    ...

def test_new_term_energy_jit(aldp_positions_jnp, aldp_params):
    """JIT matches non-jit."""
    ...
```

### 5. Export from `__init__.py`

Add the new function to `__init__.py` imports and `__all__`.

### 6. Run the full suite

```bash
python -m pytest tests/ -v
```

All 155+ tests must pass before merging.

## Testing conventions

- **Reference method**: each energy term is tested against an isolated OpenMM system containing only that force (not the force group API, which is unreliable). See `get_openmm_force_energy()` in `conftest.py`.
- **Multi-frame**: test across MD frames, not just the minimized configuration. MD frames expose edge cases (close contacts, extreme torsion angles).
- **Gradient check**: every energy function must have a test that `jax.grad` produces no NaN/Inf.
- **JIT check**: every energy function must have a test that `jax.jit(fn)` matches the non-jit result.
- **Independent oracles**: prefer an oracle the implementation cannot trivially satisfy. For the coordinate transforms that means `jacfwd` + `slogdet` for the Jacobian, the OpenMM topology bond graph for the ALDP z-matrix, and rigid-motion invariance plus distance-matrix preservation for `canonicalize_cartesian`. Idempotence alone is too weak: a wrong frame construction satisfies it.
- **Validation tests**: every entry point that gathers by index needs a test that a stale index raises rather than clamping. See `tests/test_validation.py`.
- **Mutation check**: after adding tests for a fix, revert the fix and confirm the tests go red. A test that cannot fail is not evidence.
- **Tolerances**: bonds/angles/RB torsions 1e-4 kJ/mol, periodic torsions 1e-3, nonbonded 1e-3, GBSA 1e-3, CMAP 0.5 (bilinear interpolation limit), PBC nonbonded 1e-4, total 1e-3, gradients 1e-2 (CMMotionRemover residual).

## Gotchas

These are the most common sources of bugs and confusion, distilled from
development experience.

### JAX and numerical

**JAX float64 must be enabled first.** Call `jax.config.update("jax_enable_x64", True)` before any jaxmm import. Energy functions check at runtime via `_check_x64()` and raise if float64 is off.

**NaN gradients from `jnp.linalg.norm`.** When the input vector can be zero (e.g., self-distance on the diagonal), `jnp.linalg.norm` returns 0 but its gradient is NaN. Use `jnp.sqrt(jnp.sum(x**2) + 1e-30)` instead. The epsilon does not affect forward-pass accuracy but keeps gradients finite.

**JAX clamps out-of-bounds gathers.** `positions[idx]` with `idx >= n_atoms` returns the last row rather than raising, so parameters built for a larger system produce a finite, plausible, wrong energy. This is the single most dangerous failure mode in the codebase because nothing looks wrong. Every public entry point validates; keep it that way when adding one.

**NaN gradients from `arccos`.** `arccos` has an infinite derivative at both ends of its domain, so clipping the cosine to `[-1, 1]` makes the forward pass safe and leaves the gradient NaN at exactly 0 and pi. Use `atan2(|u x v|, u . v)`, which is also scale free so the input vectors need no normalization. This is the angle analogue of the `jnp.linalg.norm` rule above and bit `cartesian_to_zmatrix` for the same reason.

**`lax.fori_loop` traces its body even when the trip count is zero.** A loop from 3 to 3 still traces, so indexing an empty array inside it raises `IndexError` rather than being skipped. Guard with an explicit early return when the degenerate size is legal, as `zmatrix_to_cartesian` does for triatomics.

**Internal coordinates are genuinely singular** at zero bond length and at an angle of 0 or pi. `zmatrix_log_abs_det_jacobian` is a function of the bond lengths and angles alone, so it is `-inf` exactly at a zero bond length or an angle of 0 or pi, and finite everywhere else, including off the chart. It says nothing about whether a reference triple is collinear: that makes the *construction* ill-posed while the determinant stays finite. Ask `zmatrix_in_domain` whether a configuration is on the chart; that is a different question. That is correct behavior for the coordinate system, not a bug to clamp. Note that `sin(pi)` evaluates to 1.2e-16 rather than 0 in float64, so a test must assert the divergence as a limit rather than comparing against `-inf` at exactly pi.

**Five copies of the dihedral formula exist on purpose.** Three in `energy.py` (periodic torsions, RB torsions, CMAP), one in `utils.py`, one in `coordinates.py`. `utils.dihedral_angle` negates to match the biochemistry/mdtraj convention; the others match OpenMM's `PeriodicTorsionForce`. Unifying them would silently flip a sign somewhere. Leave them alone.

**JIT causes tiny floating-point reordering.** JIT-compiled functions may produce results differing by ~1e-10 from non-JIT. Use 1e-8 tolerance for JIT consistency tests.

**CMAP bilinear interpolation limit.** `jax.scipy.ndimage.map_coordinates` only supports order<=1 (no bicubic). CMAP uses bilinear interpolation, resulting in ~0.13 kJ/mol difference vs OpenMM on 6x6 grids. This is a known JAX limitation. Express it in kT before deciding whether it matters: at 300 K that is about 0.05 kT typical and 0.2 kT worst case, negligible for structure and dynamics but large enough to bias a free energy or reweighting estimate. Every other term agrees to under 1e-3 kT.

### Dataclasses and pytrees

**Frozen dataclass not JIT-compatible by default.** JAX needs to know how to flatten/unflatten your dataclass. Call `_register_pytree(YourClass)` after defining it. Non-array fields (int, str) must go in `aux_field_names`.

**Optional fields need custom flatten/unflatten.** `ForceFieldParams` has optional fields (gbsa, rb_torsions, cmap, restraints) that can be `None`. The standard `_register_pytree` cannot handle `None` children; these use a custom `tree_flatten`/`tree_unflatten` pair. See the `ForceFieldParams` registration in `extract.py`.

### Test collection

**A stray top-level `tests` package can shadow this repo's.** Test files import `from conftest import ...`, not `from tests.conftest import ...`. `tests/` has no `__init__.py`, so it is only a namespace portion, and Python's `PathFinder` gives an unconditional win to any regular `tests` package (one with `__init__.py`) found anywhere on `sys.path`, such as an editable install of an unrelated project. Path ordering does not help and neither does a `pythonpath` setting; importing `conftest` directly does, because pytest's default prepend import mode puts `<repo>/tests` at `sys.path[0]`.

### OpenMM

**Force group API is unreliable.** Do not use `context.getState(groups=...)` to get per-force energies. Instead, create a separate system with only the target force (see `get_openmm_force_energy` in `conftest.py`).

**CMMotionRemover adds a small force correction** (~6e-3 kJ/mol/nm) not modeled in jaxmm. Gradient tests against OpenMM use 1e-2 tolerance to account for this.

**OpenMM Verlet is leapfrog.** `setVelocities` sets v(t-dt/2), not v(t). When comparing against OpenMM Verlet trajectories, pre-kick by -dt/2*F/m to match velocity Verlet.

**Implicit solvent uses CustomGBForce, not GBSAOBCForce.** `openmmtools.testsystems.AlanineDipeptideImplicit` creates a `CustomGBForce`. The extraction code handles both types. The CustomGBForce uses OBC1 tanh parameters (alpha=0.8, beta=0, gamma=2.909125); coefficients are parsed from the expression string.

**CMAP Quantity objects.** `CMAPTorsionForce.getMapParameters` returns OpenMM `Quantity` objects. Use `value_in_unit` conversion when extracting.

**Unsupported systems fail at extraction time.** `extract_params` raises `ValueError` for systems with constraints, virtual sites, PME/Ewald electrostatics, or unknown force types. This is intentional: jaxmm computes all-pairs interactions without long-range corrections, so silently extracting a PME system would give wrong energies. Always use `constraints=None` when building OpenMM systems.

**Long-range dispersion correction** is not implemented. This is a constant offset depending on N/V and does not affect forces or relative energies.

## Style

- Google-style docstrings on all public functions.
- Comments explain "why", not "what".
- Match existing patterns. If the other energy functions do it one way, do it the same way.
- No unnecessary abstractions. A new energy term is one dataclass + one function + one test file.
