"""Tests for examples/.

The examples are jupytext "percent" Python, not notebooks, specifically so
they can be checked here. Before the conversion nothing exercised them, and
untested examples rot: they drift from the API and nobody notices until a
reader hits the error.

The API check is the load-bearing one. It walks the AST of every example and
asserts each jaxmm attribute it touches actually exists, so renaming or
removing a public function fails here rather than in a user's face.
"""

import ast
import os
import pathlib
import subprocess
import sys

import pytest

import jaxmm
import jaxmm.notebook

EXAMPLES_DIR = pathlib.Path(__file__).resolve().parent.parent / "examples"
EXAMPLES = sorted(EXAMPLES_DIR.glob("*.py"))

# Examples that call the py3Dmol-backed viewers in jaxmm.notebook.
NEEDS_PY3DMOL = {
    "custom_energy.py", "energy_landscape.py", "jaxmm_demo.py",
    "normal_modes.py", "quickstart.py", "solvent_comparison.py",
}


def test_examples_directory_is_not_empty():
    """Guard against the glob silently matching nothing."""
    assert len(EXAMPLES) >= 11, f"found only {len(EXAMPLES)} examples"


def test_no_notebooks_are_committed():
    """Examples are .py; .ipynb carries base64 output and unreviewable diffs."""
    stray = list(EXAMPLES_DIR.glob("*.ipynb"))
    assert not stray, f"unexpected notebooks in examples/: {[p.name for p in stray]}"


@pytest.mark.parametrize("path", EXAMPLES, ids=lambda p: p.name)
def test_example_compiles(path):
    """Every example is syntactically valid Python."""
    compile(path.read_text(), str(path), "exec")


@pytest.mark.parametrize("path", EXAMPLES, ids=lambda p: p.name)
def test_example_is_percent_format(path):
    """Every example keeps the jupytext header and at least one cell marker.

    Without these, Jupyter and VS Code open the file as a plain script rather
    than as a notebook, which is the whole point of the format.
    """
    text = path.read_text()
    assert text.startswith("# ---\n# jupyter:"), "missing jupytext header"
    assert "format_name: percent" in text
    assert "\n# %%" in text, "no cell markers"
    assert "display_name: chemistry" not in text, "machine-specific kernel name"


@pytest.mark.parametrize("path", EXAMPLES, ids=lambda p: p.name)
def test_example_uses_only_real_jaxmm_api(path):
    """Every jaxmm attribute an example touches must exist.

    This is what catches drift. A renamed or removed public function fails
    here instead of when a reader runs the example.
    """
    tree = ast.parse(path.read_text(), filename=str(path))
    modules = {"jaxmm": jaxmm, "jaxmm.notebook": jaxmm.notebook}

    unknown = []
    for node in ast.walk(tree):
        # jaxmm.something
        if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name):
            if node.value.id == "jaxmm" and not hasattr(jaxmm, node.attr):
                unknown.append(f"jaxmm.{node.attr} (line {node.lineno})")
        # from jaxmm[.sub] import something
        if isinstance(node, ast.ImportFrom) and (node.module or "").startswith("jaxmm"):
            module = modules.get(node.module)
            if module is None:
                continue
            for alias in node.names:
                if not hasattr(module, alias.name):
                    unknown.append(f"{node.module}.{alias.name} (line {node.lineno})")

    assert not unknown, f"{path.name} references nonexistent jaxmm API: {unknown}"


@pytest.mark.parametrize("path", EXAMPLES, ids=lambda p: p.name)
def test_example_enables_float64_before_use(path):
    """jaxmm requires float64; an example that forgets it fails at the first call."""
    text = path.read_text()
    if "import jaxmm" not in text:
        pytest.skip("example does not use jaxmm directly")
    assert 'jax.config.update("jax_enable_x64", True)' in text, (
        "example must enable float64 before calling jaxmm"
    )


def test_example_runs_end_to_end():
    """Execute one example for real.

    aldp_potential_jaxmm is the cheapest example that does not need py3Dmol:
    about 17 seconds. Static checks cannot catch a shape error or a bad
    argument, so one example is actually run.
    """
    path = EXAMPLES_DIR / "aldp_potential_jaxmm.py"
    assert path.name not in NEEDS_PY3DMOL

    env = dict(os.environ, MPLBACKEND="Agg", JAX_PLATFORMS="cpu")
    result = subprocess.run(
        [sys.executable, str(path)], capture_output=True, text=True,
        env=env, timeout=600,
    )
    assert result.returncode == 0, (
        f"{path.name} failed:\n{result.stdout[-2000:]}\n{result.stderr[-2000:]}"
    )


@pytest.mark.parametrize(
    "name", sorted(NEEDS_PY3DMOL), ids=lambda n: n.replace(".py", ""))
def test_py3dmol_examples_declare_their_dependency(name):
    """Examples using the 3D viewers must exist and be listed accurately.

    The set above drives which examples the runner may skip, so it has to stay
    true. A viewer call appearing in an example not listed here would go
    unnoticed otherwise.
    """
    path = EXAMPLES_DIR / name
    assert path.exists(), f"{name} listed in NEEDS_PY3DMOL but missing"
    text = path.read_text()
    assert any(fn in text for fn in
               ("show_structure", "animate_trajectory", "animate_mode"))


@pytest.mark.parametrize("path", EXAMPLES, ids=lambda p: p.name)
def test_example_not_listed_as_needing_py3dmol_does_not_use_it(path):
    """Keep NEEDS_PY3DMOL honest in the other direction too."""
    if path.name in NEEDS_PY3DMOL:
        pytest.skip("listed as needing py3Dmol")
    text = path.read_text()
    for fn in ("show_structure", "animate_trajectory", "animate_mode"):
        assert fn not in text, (
            f"{path.name} calls {fn} but is not listed in NEEDS_PY3DMOL"
        )
