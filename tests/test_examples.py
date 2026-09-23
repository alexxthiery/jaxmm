"""Tests for examples/.

Each example is a self-contained folder: a jupytext "percent" Python script, a
generated README, and a gitignored output/ for figures. They are plain Python
rather than notebooks specifically so they can be checked here. Untested
examples rot: they drift from the API and nobody notices until a reader hits
the error.

The API check is the load-bearing one. It walks each example's AST and asserts
every jaxmm attribute it touches exists, so renaming or removing a public
function fails here rather than in a reader's face.
"""


import ast
import os
import pathlib
import subprocess
import sys

import pytest

import jaxmm
import jaxmm.notebook

ROOT = pathlib.Path(__file__).resolve().parent.parent
EXAMPLES_DIR = ROOT / "examples"
EXAMPLE_DIRS = sorted(
    d for d in EXAMPLES_DIR.iterdir()
    if d.is_dir() and (d / f"{d.name}.py").exists()
)
EXAMPLES = [d / f"{d.name}.py" for d in EXAMPLE_DIRS]

VIEWER_CALLS = ("show_structure", "animate_trajectory", "animate_mode")


def _ids(path):
    return path.parent.name if path.suffix == ".py" else path.name


# ---------------------------------------------------------------------------
# Layout
# ---------------------------------------------------------------------------

def test_examples_directory_is_not_empty():
    """Guard against the glob silently matching nothing."""
    assert len(EXAMPLES) >= 11, f"found only {len(EXAMPLES)} examples"


def test_no_notebooks_are_committed():
    """Examples are .py; .ipynb carries base64 output and unreviewable diffs."""
    stray = list(EXAMPLES_DIR.rglob("*.ipynb"))
    assert not stray, f"unexpected notebooks: {[str(p) for p in stray]}"


@pytest.mark.parametrize("directory", EXAMPLE_DIRS, ids=lambda d: d.name)
def test_example_folder_layout(directory):
    """Every example folder holds a script named after it, plus a README."""
    assert (directory / f"{directory.name}.py").exists()
    assert (directory / "README.md").exists(), "run tools/sync_example_docs.py"


def test_example_docs_are_in_sync():
    """READMEs are generated from each example's own markdown header.

    They are derived rather than written so the prose cannot drift from the
    example. This fails if someone edits an example's header without
    regenerating, or hand-edits a generated file.
    """
    result = subprocess.run(
        [sys.executable, str(ROOT / "tools" / "sync_example_docs.py"), "--check"],
        capture_output=True, text=True,
    )
    assert result.returncode == 0, (
        f"example docs are stale:\n{result.stderr}"
    )


# ---------------------------------------------------------------------------
# The scripts themselves
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("path", EXAMPLES, ids=_ids)
def test_example_compiles(path):
    """Every example is syntactically valid Python."""
    compile(path.read_text(), str(path), "exec")


@pytest.mark.parametrize("path", EXAMPLES, ids=_ids)
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


@pytest.mark.parametrize("path", EXAMPLES, ids=_ids)
def test_example_has_a_leading_markdown_title(path):
    """The first markdown cell supplies the generated README's title."""
    text = path.read_text()
    marker = "# %% [markdown]"
    assert marker in text, "example needs a leading markdown cell"
    after = text.split(marker, 1)[1]
    assert after.lstrip().startswith("# # "), "markdown cell needs a '# Title' line"


@pytest.mark.parametrize("path", EXAMPLES, ids=_ids)
def test_example_uses_only_real_jaxmm_api(path):
    """Every jaxmm attribute an example touches must exist.

    This is what catches drift. A renamed or removed public function fails
    here instead of when a reader runs the example.
    """
    tree = ast.parse(path.read_text(), filename=str(path))
    modules = {"jaxmm": jaxmm, "jaxmm.notebook": jaxmm.notebook}

    unknown = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name):
            if node.value.id == "jaxmm" and not hasattr(jaxmm, node.attr):
                unknown.append(f"jaxmm.{node.attr} (line {node.lineno})")
        if isinstance(node, ast.ImportFrom) and (node.module or "").startswith("jaxmm"):
            module = modules.get(node.module)
            if module is None:
                continue
            for alias in node.names:
                if not hasattr(module, alias.name):
                    unknown.append(f"{node.module}.{alias.name} (line {node.lineno})")

    assert not unknown, f"{path.name} references nonexistent jaxmm API: {unknown}"


@pytest.mark.parametrize("path", EXAMPLES, ids=_ids)
def test_example_enables_float64_before_use(path):
    """jaxmm requires float64; an example that forgets it fails at the first call."""
    text = path.read_text()
    if "import jaxmm" not in text:
        pytest.skip("example does not use jaxmm directly")
    assert 'jax.config.update("jax_enable_x64", True)' in text, (
        "example must enable float64 before calling jaxmm"
    )


@pytest.mark.parametrize("path", EXAMPLES, ids=_ids)
def test_example_does_not_write_outside_its_output_dir(path):
    """Examples must not litter the repository when run.

    Figures are saved by tools/render_examples.py, which redirects plt.show;
    an example calling savefig itself would write wherever it was launched
    from and escape the gitignored output directory.
    """
    text = path.read_text()
    assert "savefig" not in text, (
        "let tools/render_examples.py handle saving, so output stays in output/"
    )


# ---------------------------------------------------------------------------
# Execution
# ---------------------------------------------------------------------------

def test_example_runs_end_to_end():
    """Execute one example for real.

    aldp_potential_jaxmm is the cheapest example that does not need py3Dmol:
    about 17 seconds. Static checks cannot catch a shape error or a bad
    argument, so one example is actually run.
    """
    pytest.importorskip("openmm")
    directory = EXAMPLES_DIR / "aldp_potential_jaxmm"
    script = directory / f"{directory.name}.py"
    assert not any(call in script.read_text() for call in VIEWER_CALLS)

    env = dict(os.environ, MPLBACKEND="Agg", JAX_PLATFORMS="cpu")
    result = subprocess.run(
        [sys.executable, str(script)], capture_output=True, text=True,
        env=env, timeout=600,
    )
    assert result.returncode == 0, (
        f"{script.name} failed:\n{result.stdout[-2000:]}\n{result.stderr[-2000:]}"
    )


def test_renderer_saves_figures_that_are_not_blank(tmp_path):
    """The renderer must produce real figures, not empty canvases.

    A blank PNG is the failure mode when plt.show is redirected but the figure
    has already been closed. Checking file size alone would not catch it, so
    this compares against a known-empty figure of the same size.
    """
    pytest.importorskip("matplotlib")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    blank = tmp_path / "blank.png"
    figure = plt.figure(figsize=(6, 4))
    figure.savefig(blank, dpi=110, bbox_inches="tight")
    plt.close(figure)

    drawn = tmp_path / "drawn.png"
    figure = plt.figure(figsize=(6, 4))
    figure.gca().plot([0, 1, 2], [0, 1, 4])
    figure.savefig(drawn, dpi=110, bbox_inches="tight")
    plt.close(figure)

    assert drawn.stat().st_size > blank.stat().st_size * 1.2, (
        "a plotted figure must be meaningfully larger than an empty one; "
        "if this fails the size heuristic below is not valid"
    )

    rendered = sorted((EXAMPLES_DIR / "aldp_potential_jaxmm" / "output").glob("fig*.png"))
    if not rendered:
        pytest.skip("no rendered output; run tools/render_examples.py")
    for path in rendered:
        assert path.stat().st_size > blank.stat().st_size, f"{path.name} looks blank"
