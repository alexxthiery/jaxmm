# Codemap: jaxmm
> Last updated: 2026-02-13

Pure JAX molecular potential energy. OpenMM extracts force field params once;
energy evaluation is pure JAX (jit, vmap, grad). Built for training normalizing
flows against Boltzmann targets without the CPU-GPU bottleneck of calling OpenMM
per sample.

## Structure

```
jaxmm/                  # core library (1361 lines)
  __init__.py            public API, 25 exports
  extract.py             OpenMM System -> frozen dataclasses (614 lines)
  energy.py              pure JAX energy functions (340 lines)
  integrate.py           Verlet + Langevin BAOAB integrators (166 lines)
  utils.py               minimize, log_prob, dihedrals, unit constants (193 lines)
tests/                   # 70 tests, ~20s (1346 lines)
  conftest.py            ALDP fixtures (vacuum + implicit), OpenMM reference helpers
  test_extraction.py     param shapes/values (11 tests)
  test_bonds.py          bond energy vs OpenMM (4)
  test_angles.py         angle energy vs OpenMM (4)
  test_torsions.py       torsion energy vs OpenMM (4)
  test_nonbonded.py      nonbonded energy vs OpenMM (5)
  test_gbsa.py           GBSA energy, Born radii, gradients (19)
  test_total.py          total energy, vmap, log_prob (6)
  test_grad.py           gradients vs OpenMM + finite diff (3)
  test_dihedral.py       phi/psi dihedral angles (5)
  test_integrate.py      Verlet + Langevin BAOAB (7)
  test_minimize.py       L-BFGS-B minimization vs OpenMM (2)
examples/
  jaxmm_demo.ipynb       energy comparison, vmap, gradients, timing, MD
  parallel_tempering.ipynb  replica exchange MD, Ramachandran comparison
  aldp_potential.ipynb      original OpenMM + boltzgen exploration
  aldp_potential_jaxmm.ipynb  jaxmm version of the above
```

## Modules

| File | Role | Key symbols |
|------|------|-------------|
| `extract.py` | Dataclasses + extraction from OpenMM | `BondParams`, `AngleParams`, `TorsionParams`, `NonbondedParams`, `GBSAParams`, `ForceFieldParams`, `extract_params()`, `phi_indices()`, `psi_indices()` |
| `energy.py` | Pure JAX energy functions | `bond_energy()`, `angle_energy()`, `torsion_energy()`, `nonbonded_energy()`, `gbsa_energy()`, `total_energy()` |
| `integrate.py` | MD integrators (pure JAX, use `jax.lax.scan`) | `verlet()`, `langevin_baoab()`, `kinetic_energy()`, `KB` |
| `utils.py` | Minimization, log-prob, geometry, units | `minimize_energy()`, `log_prob()`, `log_prob_regularized()`, `dihedral_angle()`, `FEMTOSECOND`, `PICOSECOND`, `NANOMETER`, `ANGSTROM`, `KJ_PER_MOL`, `KCAL_PER_MOL`, `KELVIN` |

## Data flow

```
OpenMM System
    |
    v
extract_params()  -->  ForceFieldParams (frozen dataclass, JAX pytree)
                            |
                            v
                   total_energy(positions, params) --> scalar (kJ/mol)
                            |
                   +--------+--------+
                   |                 |
                   v                 v
              jax.grad          jax.vmap / jax.jit
              (forces)          (batch eval)
                   |
                   v
         minimize_energy()  /  verlet()  /  langevin_baoab()
```

## Entry points

- **Library**: `import jaxmm; params = jaxmm.extract_params(system)`
- **Tests**: `python -m pytest tests/ -v` (70 tests, ~20s)
- **Demo**: `examples/jaxmm_demo.ipynb`

## Patterns

- All energy functions: `(positions, params) -> scalar` (pure, no side effects)
- Frozen dataclasses as param containers, registered as JAX pytrees
- Batch eval: `jax.vmap(jaxmm.total_energy, in_axes=(0, None))`
- Integrators: nested `jax.lax.scan` (inner for stepping, outer for saving)
- Test validation: each term tested against isolated OpenMM force (not force group API)
- Optional fields (GBSA): `None` for vacuum, handled in custom pytree flatten/unflatten
- Gradient safety: `jnp.sqrt(r_sq + 1e-30)` instead of `jnp.linalg.norm` (avoids NaN grad at zero)

## Units

Internal: nm, ps, kJ/mol, K, amu, elementary charge (matches OpenMM).
Unit constants multiply to convert: `dt = 1.0 * FEMTOSECOND` gives `0.001` ps.

## Dependencies

Core: `jax`, `numpy`, `scipy` (L-BFGS-B for minimization).
Extraction only: `openmm`, `openmmtools`.
Tests: `pytest`, `openmm`, `openmmtools`.
