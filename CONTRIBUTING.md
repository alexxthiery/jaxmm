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
| `jaxmm/utils.py` | minimize_energy, log_boltzmann, dihedral angles, unit constants, serialization | energy.py, numpy, jaxopt |
| `jaxmm/notebook.py` | Jupyter helpers: 3D viz, Ramachandran, free energy estimation | matplotlib, py3Dmol, openmm (all lazy) |
| `tests/conftest.py` | ALDP fixtures (vacuum + implicit) + OpenMM reference helpers | openmm, openmmtools, jaxmm |

### Entry points

- **Library**: `import jaxmm; params = jaxmm.extract_params(system)`
- **Tests**: `python -m pytest tests/ -v` (155 tests, ~44s)
- **Notebooks**: `examples/quickstart.ipynb` (start here), plus 9 topic notebooks
- **Demo**: `examples/jaxmm_demo.ipynb` (chemistry kernel)

### Dev commands

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

## Invariants

These must stay true. Breaking any of them will break downstream users.

1. **Pure functions.** Every energy function is `(positions, params) -> scalar`. No side effects, no hidden state, no closures over mutable data.

2. **JAX pytree compatibility.** All param dataclasses are registered as JAX pytrees (see `_register_pytree` in `extract.py`). New dataclasses must also be registered. Fields that are not JAX arrays (like `n_atoms: int`) go in `aux_field_names`.

3. **float64 precision.** All modules call `jax.config.update("jax_enable_x64", True)`. All arrays are float64 (or int32 for indices). Do not downcast to float32.

4. **Units.** Positions in nm. Energies in kJ/mol. Angles in radians. Charges in elementary charge units. These match OpenMM's internal unit system.

5. **Gradient safety.** Energy functions must produce finite gradients for any physically reasonable configuration. Avoid `jnp.linalg.norm` on vectors that can be zero (use `jnp.sqrt(r_sq + 1e-30)` instead).

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
- **Tolerances**: bonds/angles/RB torsions 1e-4 kJ/mol, periodic torsions 1e-3, nonbonded 1e-3, GBSA 1e-3, CMAP 0.5 (bilinear interpolation limit), PBC nonbonded 1e-4, total 1e-3, gradients 1e-2 (CMMotionRemover residual).

## Gotchas

These are the most common sources of bugs and confusion, distilled from
development experience.

### JAX and numerical

**JAX float64 must be enabled first.** Call `jax.config.update("jax_enable_x64", True)` before any jaxmm import. Energy functions check at runtime via `_check_x64()` and raise if float64 is off.

**NaN gradients from `jnp.linalg.norm`.** When the input vector can be zero (e.g., self-distance on the diagonal), `jnp.linalg.norm` returns 0 but its gradient is NaN. Use `jnp.sqrt(jnp.sum(x**2) + 1e-30)` instead. The epsilon does not affect forward-pass accuracy but keeps gradients finite.

**JIT causes tiny floating-point reordering.** JIT-compiled functions may produce results differing by ~1e-10 from non-JIT. Use 1e-8 tolerance for JIT consistency tests.

**CMAP bilinear interpolation limit.** `jax.scipy.ndimage.map_coordinates` only supports order<=1 (no bicubic). CMAP uses bilinear interpolation, resulting in ~0.13 kJ/mol difference vs OpenMM on 6x6 grids. This is a known JAX limitation.

### Dataclasses and pytrees

**Frozen dataclass not JIT-compatible by default.** JAX needs to know how to flatten/unflatten your dataclass. Call `_register_pytree(YourClass)` after defining it. Non-array fields (int, str) must go in `aux_field_names`.

**Optional fields need custom flatten/unflatten.** `ForceFieldParams` has optional fields (gbsa, rb_torsions, cmap, restraints) that can be `None`. The standard `_register_pytree` cannot handle `None` children; these use a custom `tree_flatten`/`tree_unflatten` pair. See the `ForceFieldParams` registration in `extract.py`.

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
