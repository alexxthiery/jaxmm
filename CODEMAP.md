# Codemap: jaxmm

> Last updated: 2026-02-26

Pure JAX molecular potential energy. OpenMM extracts force field params once;
energy evaluation is pure JAX (jit, vmap, grad). Built for training normalizing
flows against Boltzmann targets without the CPU-GPU bottleneck of calling OpenMM
per sample.

## Structure

```
jaxmm/                  # core library (~1361 lines, 5 modules)
  __init__.py            public API, 40 exports
  extract.py             OpenMM System -> frozen dataclasses (614 lines)
  energy.py              pure JAX energy functions (340 lines)
  integrate.py           Verlet + Langevin BAOAB integrators (166 lines)
  utils.py               minimize, log_boltzmann, dihedrals, serialization, constants (409 lines)
  notebook.py            Jupyter helpers: viz, Ramachandran, free energy (297 lines, not re-exported)
tests/                   # 155 tests, ~44s
  conftest.py            ALDP fixtures (vacuum + implicit), OpenMM reference helpers
  test_extraction.py     param shapes/values/validation (24 tests)
  test_bonds.py          bond energy vs OpenMM (5)
  test_angles.py         angle energy vs OpenMM (5)
  test_torsions.py       periodic torsion energy vs OpenMM (5)
  test_rb_torsions.py    Ryckaert-Bellemans torsions vs OpenMM (5)
  test_cmap.py           CMAP torsion correction vs OpenMM (5)
  test_nonbonded.py      Coulomb + LJ vs OpenMM (5)
  test_pbc.py            periodic boundary conditions + cutoff (11)
  test_gbsa.py           GBSA energy, Born radii, gradients, jit, vmap (19)
  test_total.py          total energy, vmap, composability, energy_components (27)
  test_dihedral.py       phi/psi dihedral angles (6)
  test_grad.py           gradients vs OpenMM + finite diff (8)
  test_integrate.py      Verlet + Langevin BAOAB (17)
  test_minimize.py       L-BFGS minimization (4)
  test_serialization.py  save/load roundtrip (2)
examples/                # 11 Jupyter notebooks
  quickstart.ipynb           core API in 5 minutes
  energy_landscape.ipynb     PES visualization, free energy surfaces
  differentiable_md.ipynb    gradients through MD, Hessian, parameter sensitivity
  custom_energy.ipynb        restraints, dihedral bias, umbrella sampling
  normal_modes.ipynb         Hessian eigendecomposition, vibrational frequencies
  solvent_comparison.ipynb   vacuum vs implicit solvent side-by-side
  free_energy.ipynb          histogram FES, convergence, log_boltzmann_regularized
  custom_samplers.ipynb      simulated tempering, HMC via baoab_step
  jaxmm_demo.ipynb           energy comparison, vmap, gradients, timing, MD
  parallel_tempering.ipynb   replica exchange MD, Ramachandran comparison
  aldp_potential_jaxmm.ipynb jaxmm version of classic ALDP exploration
```

## Module dependency graph

```
extract.py              jax, numpy
    |                   (openmm imported lazily, only inside extract_params)
    |
    +---> energy.py     jax (imports dataclass types from extract.py)
    |         |
    |         +---> integrate.py    jax (imports total_energy from energy.py,
    |         |                          KB from utils.py)
    |         |
    |         +---> utils.py        jax, numpy, jaxopt
    |                               (imports ForceFieldParams from extract.py,
    |                                total_energy from energy.py)
    |
    +---> notebook.py   numpy, jax
                        (imports KB, dihedral_angle from utils.py,
                         phi_indices, psi_indices from extract.py;
                         matplotlib, py3Dmol, openmm lazy-imported)
```

No circular dependencies. `energy.py` imports only type definitions from
`extract.py`. `notebook.py` is not re-exported from `__init__.py`.

## Modules

| File | Role | Key symbols |
|------|------|-------------|
| `extract.py` | Dataclasses + extraction from OpenMM | `BondParams`, `AngleParams`, `TorsionParams`, `RBTorsionParams`, `CmapParams`, `RestraintParams`, `NonbondedParams`, `GBSAParams`, `ForceFieldParams`, `extract_params()`, `phi_indices()`, `psi_indices()`, `make_restraints()` |
| `energy.py` | Pure JAX energy functions | `bond_energy()`, `angle_energy()`, `torsion_energy()`, `rb_torsion_energy()`, `cmap_energy()`, `restraint_energy()`, `nonbonded_energy()`, `gbsa_energy()`, `total_energy()`, `energy_components()` |
| `integrate.py` | MD integrators (pure JAX, `jax.lax.scan`) | `verlet()`, `langevin_baoab()`, `baoab_step()`, `kinetic_energy()`, `MDTrajectory` |
| `utils.py` | Minimization, log-Boltzmann, geometry, serialization, constants | `minimize_energy()`, `log_boltzmann()`, `log_boltzmann_regularized()`, `dihedral_angle()`, `save_params()`, `load_params()`, `KB`, `FEMTOSECOND`, `ANGSTROM`, `KCAL_PER_MOL` |
| `notebook.py` | Jupyter helpers (not in public API) | `show_structure()`, `animate_trajectory()`, `animate_mode()`, `phi_psi_degrees()`, `plot_ramachandran()`, `free_energy_1d()`, `free_energy_2d()` |

## Data flow

```
OpenMM System
    |
    v
extract_params()  -->  ForceFieldParams (frozen dataclass, JAX pytree)
                            |
                       save_params() / load_params()  (.npz, no pickle)
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
- **Tests**: `python -m pytest tests/ -v` (155 tests, ~44s)
- **Notebooks**: `examples/quickstart.ipynb` (start here)
- **Demo**: `examples/jaxmm_demo.ipynb`

## Patterns

- All energy functions: `(positions, params) -> scalar` (pure, no side effects)
- Frozen dataclasses as param containers, registered as JAX pytrees via `_register_pytree()`
- Batch eval: `jax.vmap(jaxmm.total_energy, in_axes=(0, None))`
- Integrators: nested `jax.lax.scan` (inner for stepping, outer for saving)
- Test validation: each term tested against isolated OpenMM force (not force group API)
- Optional fields (gbsa, rb_torsions, cmap, restraints): `None` when absent, custom pytree flatten/unflatten
- Gradient safety: `jnp.sqrt(r_sq + 1e-30)` instead of `jnp.linalg.norm` (avoids NaN grad at zero)

## Units

Internal: nm, ps, kJ/mol, K, amu, elementary charge (matches OpenMM).
Unit constants multiply to convert: `dt = 1.0 * FEMTOSECOND` gives `0.001` ps.

## Dependencies

Core: `jax`, `numpy`, `jaxopt`.
Extraction only: `openmm`, `openmmtools`.
Tests: `pytest`, `openmm`, `openmmtools`.
Notebooks: `matplotlib`, `py3Dmol` (lazy-imported).
