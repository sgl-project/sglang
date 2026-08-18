"""Every processor must reach preprocessing through the executor-backed helper.

`process_and_combine_mm_data` is the function the multimodal processor worker
pool actually runs. A processor that calls it directly can never use those
workers: it will build the thread pool and its processor clones on startup and
then route every request past them. That failure is silent -- the model just
serves at one-worker speed -- so pin the call site instead of the symptom.

`process_and_combine_mm_data_async` delegates straight to the sync function when
no executor exists, so using it costs nothing until a model opts into
concurrency.
"""

import ast
import pathlib

import pytest

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=3, suite="base-a-test-cpu")

_MULTIMODAL_ROOT = (
    pathlib.Path(__file__).resolve().parents[4]
    / "python"
    / "sglang"
    / "srt"
    / "multimodal"
)
# The async helper and the sync body live side by side here by design.
_EXEMPT = {"base_processor.py"}


def _enclosing_function(node, parents):
    current = parents.get(id(node))
    while current is not None:
        if isinstance(current, (ast.FunctionDef, ast.AsyncFunctionDef)):
            return current
        current = parents.get(id(current))
    return None


def _call_sites():
    """Yield (path, lineno, attribute, enclosing_function) for every call."""
    for path in sorted(_MULTIMODAL_ROOT.rglob("*.py")):
        if path.name in _EXEMPT:
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        parents = {
            id(child): parent
            for parent in ast.walk(tree)
            for child in ast.iter_child_nodes(parent)
        }
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            func = node.func
            if not (
                isinstance(func, ast.Attribute)
                and func.attr.startswith("process_and_combine_mm_data")
            ):
                continue
            yield path, node.lineno, func.attr, _enclosing_function(node, parents)


def test_no_processor_bypasses_the_worker_pool():
    offenders = [
        f"{path.relative_to(_MULTIMODAL_ROOT)}:{lineno}"
        for path, lineno, attr, _ in _call_sites()
        if not attr.endswith("_async")
    ]
    assert not offenders, (
        "these call sites bypass the multimodal processor worker pool; use "
        "`await self.process_and_combine_mm_data_async(...)`: " + ", ".join(offenders)
    )


def test_every_call_site_can_await():
    """An `await` needs an async def around it, so the migration stays possible."""
    offenders = [
        f"{path.relative_to(_MULTIMODAL_ROOT)}:{lineno}"
        for path, lineno, _, enclosing in _call_sites()
        if not isinstance(enclosing, ast.AsyncFunctionDef)
    ]
    assert not offenders, (
        "preprocessing is reached from a non-async function, so it cannot go "
        "through the worker pool: " + ", ".join(offenders)
    )


def test_default_worker_count_stays_at_the_measured_optimum():
    """Two workers, measured, not guessed.

    Image preprocessing releases the GIL, so a second worker overlaps it with
    the GPU; a third and fourth start spreading request arrivals far enough
    apart to fragment the prefill batches. Measured at 32-way concurrency on an
    H200 for one / two / four workers: PaddleOCR-VL 1080p pages 6.72 / 9.55 /
    8.92 req/s, Qwen2.5-VL small images with 512-token outputs 12.54 / 12.60 /
    12.25 req/s. Raising this needs a fresh sweep, not a hunch.
    """
    from sglang.srt.multimodal.processors.base_processor import (
        BaseMultimodalProcessor,
    )

    assert BaseMultimodalProcessor.supports_mm_processor_concurrency is True
    assert BaseMultimodalProcessor.auto_mm_processor_worker_num == 2


def test_the_scan_actually_finds_call_sites():
    """Guard against the scan silently matching nothing after a rename."""
    assert len(list(_call_sites())) > 20


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
