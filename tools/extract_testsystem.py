"""Extract an openmmtools test system's force field into a jaxmm .npz.

OpenMM is needed to *define* a force field; jaxmm only evaluates one. So this
is the one step that needs it, and it is a setup step rather than a runtime
one: `extract_params` imports OpenMM lazily and `save_params` writes a
pickle-free .npz, so everything downstream runs in a plain JAX environment with
no OpenMM installed.

Run it once per system, in an environment that has openmm and openmmtools:

    python tools/extract_testsystem.py AlanineDipeptideImplicit --out ../LTR/results/forcefields

Writes `<name>.npz` (the parameters) and `<name>_positions.npy` (the system's
own starting coordinates, in nm), plus `<name>.json` recording what produced
them.

`constraints=None` is passed to every system: constrained bond lengths change
the Boltzmann measure, and all three reference papers (FAB, TA-BG, CMT) run
alanine dipeptide unconstrained.
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import date
from pathlib import Path

import numpy as np


def extract(name: str, out_dir: Path) -> None:
    import openmm
    from openmm import unit
    from openmmtools import testsystems

    import jaxmm
    from jaxmm.extract import extract_params
    from jaxmm.utils import save_params

    if not hasattr(testsystems, name):
        raise SystemExit(f"openmmtools.testsystems has no {name!r}")
    system = getattr(testsystems, name)(constraints=None)

    params = extract_params(system.system)
    positions = system.positions.value_in_unit(unit.nanometer)
    positions = np.asarray(positions, dtype=np.float64)

    out_dir.mkdir(parents=True, exist_ok=True)
    save_params(params, str(out_dir / f"{name}.npz"))
    np.save(out_dir / f"{name}_positions.npy", positions)

    meta = dict(
        system=name,
        constraints=None,
        n_atoms=int(params.n_atoms),
        n_bonds=int(params.bonds.r0.shape[0]),
        n_angles=int(params.angles.theta0.shape[0]),
        n_torsions=int(params.torsions.k.shape[0]),
        has_gbsa=params.gbsa is not None,
        openmm_version=openmm.__version__,
        openmmtools_version=getattr(testsystems, "__version__", "unknown"),
        extracted=str(date.today()),
        note="constraints=None; positions in nm from the test system itself",
    )
    (out_dir / f"{name}.json").write_text(json.dumps(meta, indent=1))

    print(f"{name}: {meta['n_atoms']} atoms, {meta['n_bonds']} bonds, "
          f"{meta['n_angles']} angles, {meta['n_torsions']} torsions, "
          f"gbsa={meta['has_gbsa']}")
    print(f"  wrote {out_dir / (name + '.npz')}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("systems", nargs="+", help="openmmtools.testsystems class names")
    p.add_argument("--out", type=Path, required=True, help="output directory")
    args = p.parse_args()
    for name in args.systems:
        extract(name, args.out)


if __name__ == "__main__":
    sys.exit(main())
