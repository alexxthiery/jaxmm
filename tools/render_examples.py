#!/usr/bin/env python
"""Run the examples and save their figures to examples/<name>/output/.

Examples call plt.show(), which is idiomatic and reads correctly when one is
opened as a notebook. On a headless machine that is a silent no-op, so every
figure is computed and thrown away. This runs them with plt.show redirected to
savefig instead, which needs no change to the examples themselves.

Output is gitignored. Committing 35 figures at about 40 KB each would put back
roughly the 1.2 MB of notebook base64 this repository deliberately removed.

Usage:
    python tools/render_examples.py                 # all examples
    python tools/render_examples.py --only quickstart
    python tools/render_examples.py --list
"""

from __future__ import annotations

import argparse
import os
import pathlib
import subprocess
import sys
import time

ROOT = pathlib.Path(__file__).resolve().parent.parent
EXAMPLES = ROOT / "examples"

# Runs the example in a fresh interpreter with plt.show redirected. A
# subprocess keeps JAX and matplotlib state from leaking between examples.
RUNNER = r'''
import os, sys, runpy
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

outdir, script = sys.argv[1], sys.argv[2]
os.makedirs(outdir, exist_ok=True)
count = {"n": 0}

def _save(*args, **kwargs):
    figures = [plt.figure(num) for num in plt.get_fignums()]
    for figure in figures:
        count["n"] += 1
        figure.savefig(os.path.join(outdir, "fig%02d.png" % count["n"]),
                       dpi=110, bbox_inches="tight")
    plt.close("all")

plt.show = _save
sys.argv = [script]
try:
    runpy.run_path(script, run_name="__main__")
finally:
    _save()                      # catch figures left open at the end
    print("FIGURES=%d" % count["n"])
'''


def example_dirs():
    """Every examples/<name>/ that holds a matching script."""
    return sorted(
        d for d in EXAMPLES.iterdir()
        if d.is_dir() and (d / f"{d.name}.py").exists()
    )


def render(directory: pathlib.Path, timeout: int) -> dict:
    """Run one example, returning a status record."""
    script = directory / f"{directory.name}.py"
    outdir = directory / "output"
    env = dict(os.environ, MPLBACKEND="Agg", JAX_PLATFORMS="cpu")

    started = time.perf_counter()
    try:
        result = subprocess.run(
            [sys.executable, "-c", RUNNER, str(outdir), str(script)],
            capture_output=True, text=True, env=env, timeout=timeout,
        )
    except subprocess.TimeoutExpired:
        return {"name": directory.name, "ok": False, "figures": 0,
                "seconds": timeout, "error": f"timed out after {timeout}s"}

    seconds = time.perf_counter() - started
    if result.returncode != 0:
        tail = (result.stderr or result.stdout).strip().splitlines()
        reason = tail[-1] if tail else "unknown error"
        if "py3Dmol" in result.stderr:
            reason = "needs py3Dmol (pip install -e '.[examples]')"
        return {"name": directory.name, "ok": False, "figures": 0,
                "seconds": seconds, "error": reason}

    figures = 0
    for line in result.stdout.splitlines():
        if line.startswith("FIGURES="):
            figures = int(line.split("=", 1)[1])
    written = sorted(outdir.glob("fig*.png")) if outdir.exists() else []
    size = sum(p.stat().st_size for p in written)
    return {"name": directory.name, "ok": True, "figures": len(written),
            "seconds": seconds, "bytes": size}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--only", metavar="NAME", help="render a single example")
    parser.add_argument("--list", action="store_true", help="list examples and exit")
    parser.add_argument("--timeout", type=int, default=900,
                        help="per-example timeout in seconds (default 900)")
    args = parser.parse_args()

    directories = example_dirs()
    if args.list:
        for d in directories:
            print(d.name)
        return 0
    if args.only:
        directories = [d for d in directories if d.name == args.only]
        if not directories:
            print(f"no example named {args.only!r}", file=sys.stderr)
            return 1

    results = [render(d, args.timeout) for d in directories]
    for record in results:
        if record["ok"]:
            print(f"  {record['name']:26s} {record['figures']:2d} figures  "
                  f"{record['bytes'] / 1024:6.0f} KB  {record['seconds']:6.1f}s")
        else:
            print(f"  {record['name']:26s} SKIPPED: {record['error']}")

    ok = [r for r in results if r["ok"]]
    failed = [r for r in results if not r["ok"]]
    print(f"\n{len(ok)} rendered, {len(failed)} skipped, "
          f"{sum(r['figures'] for r in ok)} figures, "
          f"{sum(r['bytes'] for r in ok) / 1024:.0f} KB total")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
