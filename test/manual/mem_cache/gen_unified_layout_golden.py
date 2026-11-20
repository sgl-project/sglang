"""Regenerate unified_layout_golden.json from the unified pool builder.

The golden pins entry formats and capacity results (test_unified_layout_golden.py).
Regenerate it only for an intended layout or sizing change, and explain the delta:
    python test/manual/mem_cache/gen_unified_layout_golden.py

The bs=1 floor terms are read from the builder's boot error at a one-byte
budget, so the golden pins what a too-small budget reports.
"""

import importlib.util
import json
import re
import subprocess
from pathlib import Path

import msgspec

from sglang.test.unified_allocator_fixtures import (
    build_unified_bundle,
    setup_allocator_context,
    unified_layout_summary,
)

_TEST_DIR = Path(__file__).resolve().parents[2] / "registered/unit/mem_cache"
_OUT = _TEST_DIR / "unified_layout_golden.json"


def _load_test_module():
    spec = importlib.util.spec_from_file_location(
        "test_unified_layout_golden", _TEST_DIR / "test_unified_layout_golden.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _floor_terms(geometry):
    """The (name, bytes) terms of the bs=1 floor, as a one-byte budget reports them."""
    setup_allocator_context()
    try:
        build_unified_bundle(msgspec.structs.replace(geometry, byte_budget=1))
    except RuntimeError as e:
        detail = re.search(r"bs=1 floor \d+ = (.*?)\)", str(e)).group(1)
        return [
            [name, int(nbytes)]
            for name, nbytes in (t.split("=") for t in detail.split(" + "))
        ]
    raise AssertionError("a one-byte budget built a pool")


def _budget_at_floor(geometry, floor: int) -> int:
    """The full budget includes the fixed allocations outside the shared arena."""
    setup_allocator_context()
    bundle = build_unified_bundle(msgspec.structs.replace(geometry, max_num_reqs=1))
    return floor + bundle.storage.sized.fixed_bytes


def _compact(value, indent=0) -> str:
    """JSON with dicts one key per line and every list on one line."""
    pad = " " * indent
    if isinstance(value, dict):
        items = [
            f"{pad} {json.dumps(k)}: {_compact(v, indent + 1).lstrip()}"
            for k, v in value.items()
        ]
        return "{\n" + ",\n".join(items) + f"\n{pad}}}"
    return json.dumps(value)


def main():
    module = _load_test_module()
    layouts = {}
    for name, geometry in module._CONFIGS.items():
        setup_allocator_context()
        summary = unified_layout_summary(build_unified_bundle(geometry))
        layouts[name] = json.loads(json.dumps(summary))
    floors = {}
    for name, geometry in module._FLOOR_CONFIGS.items():
        terms = _floor_terms(geometry)
        budget = _budget_at_floor(geometry, sum(b for _, b in terms))
        # Independently find the first feasible total budget; the regression
        # pins this value, rather than searching when the test runs.
        low, high = budget, build_unified_bundle(geometry).storage.sized.budget_bytes
        while low < high:
            mid = (low + high) // 2
            try:
                build_unified_bundle(msgspec.structs.replace(geometry, byte_budget=mid))
            except RuntimeError:
                low = mid + 1
            else:
                high = mid
        floors[name] = {
            "terms": terms,
            "budget_at_floor": budget,
            "minimum_usable_budget": low,
        }
    commit = subprocess.run(
        ["git", "rev-parse", "--short", "HEAD"],
        capture_output=True,
        text=True,
        cwd=_TEST_DIR,
    ).stdout.strip()
    golden = {
        "source": f"test/manual/mem_cache/gen_unified_layout_golden.py at {commit}",
        "layouts": layouts,
        "bs1_floors": floors,
    }
    _OUT.write_text(_compact(golden) + "\n")


if __name__ == "__main__":
    main()
