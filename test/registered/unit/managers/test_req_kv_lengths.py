"""ReqKvInfo owns its lengths: only its own methods assign them, and
check_lengths rejects an out-of-order row."""

import ast
import unittest
import warnings
from pathlib import Path

from sglang.srt.managers.schedule_batch import ReqKvInfo
from sglang.srt.mem_cache.unified_cache.component_type import ComponentType
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=10, suite="base-a-test-cpu")

_SGLANG_DIR = Path(__file__).resolve().parents[4] / "python" / "sglang"
assert _SGLANG_DIR.is_dir(), f"sglang dir not found: {_SGLANG_DIR}"

_LENGTH_FIELDS = {
    "cache_protected_len",
    "cache_inserted_len",
    "kv_committed_len",
    "kv_allocated_len",
    "swa_evict_floor",
    "component_evicted_seqlens",
}
_DICT_MUTATORS = {"clear", "pop", "popitem", "setdefault", "update"}

# {(path relative to python/sglang, enclosing scope): reason}. Expected to
# stay empty.
_ALLOWED_WRITERS: dict[tuple[str, str], str] = {}


def _writes(node):
    """The length fields a statement or call writes through another object."""
    if isinstance(node, (ast.Assign, ast.AugAssign, ast.AnnAssign)):
        targets = node.targets if isinstance(node, ast.Assign) else [node.target]
        for target in targets:
            for sub in ast.walk(target):
                if isinstance(sub, ast.Subscript):
                    sub = sub.value
                if (
                    isinstance(sub, ast.Attribute)
                    and sub.attr in _LENGTH_FIELDS
                    # Another class's own field of the same name.
                    and not (isinstance(sub.value, ast.Name) and sub.value.id == "self")
                ):
                    yield sub.attr
    elif isinstance(node, ast.Call):
        func = node.func
        if (
            isinstance(func, ast.Attribute)
            and func.attr in _DICT_MUTATORS
            and isinstance(func.value, ast.Attribute)
            and func.value.attr == "component_evicted_seqlens"
        ):
            yield "component_evicted_seqlens"
        name = getattr(func, "id", getattr(func, "attr", None))
        if name == "ReqKvInfo":
            for keyword in node.keywords:
                if keyword.arg in _LENGTH_FIELDS:
                    yield keyword.arg


def _scan(path):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", SyntaxWarning)
        tree = ast.parse(path.read_text(encoding="utf-8-sig"))
    found = []

    def visit(node, scope):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            scope = f"{scope}.{node.name}" if scope else node.name
        if scope.split(".")[0] == "ReqKvInfo":
            return
        for field in _writes(node):
            found.append((scope, field, node.lineno))
        for child in ast.iter_child_nodes(node):
            visit(child, scope)

    visit(tree, "")
    return found


class TestReqKvLengthWriters(CustomTestCase):
    def test_only_req_kv_info_methods_assign_lengths(self):
        found, allowed_seen = [], set()
        for path in sorted(_SGLANG_DIR.rglob("*.py")):
            rel = path.relative_to(_SGLANG_DIR).as_posix()
            for scope, field, line in _scan(path):
                if (rel, scope) in _ALLOWED_WRITERS:
                    allowed_seen.add((rel, scope))
                else:
                    found.append(f"{rel}:{line} {scope} writes {field}")
        self.assertFalse(
            found,
            "Write these lengths through a ReqKvInfo method:\n  " + "\n  ".join(found),
        )
        self.assertEqual(
            set(_ALLOWED_WRITERS) - allowed_seen,
            set(),
            "allowlisted writers that are gone; drop them from _ALLOWED_WRITERS",
        )


def _held(**lengths):
    kv = ReqKvInfo(req_pool_idx=0)
    kv.allocate_extend(lengths.pop("committed", 8))
    for name, value in lengths.items():
        setattr(kv, name, value)
    return kv


class TestReqKvLengthOrder(CustomTestCase):
    def test_in_order_row_passes(self):
        kv = _held(cache_protected_len=4, cache_inserted_len=8)
        kv.allocate_decode(1)
        kv.check_lengths()
        ReqKvInfo(kv_committed_len=9, kv_allocated_len=0).check_lengths()

    def test_out_of_order_rows_fail(self):
        cases = {
            "protected past committed": _held(cache_protected_len=9),
            "inserted past committed": _held(cache_inserted_len=9),
            "committed past allocated": _held(kv_allocated_len=7),
        }
        evicted = _held()
        evicted.set_evicted_seqlen(ComponentType.SWA, 9)
        cases["evicted past committed"] = evicted
        for name, kv in cases.items():
            with self.subTest(name), self.assertRaises(AssertionError):
                kv.check_lengths()

    def test_reset_for_retract_keeps_only_the_backup(self):
        kv = _held(cache_protected_len=4, swa_evict_floor=8)
        kv.set_evicted_seqlen(ComponentType.SWA, 2)
        kv.mamba_needs_clear = True
        kv.retraction_backup = backup = object()
        kv.req_pool_idx = None
        kv.mark_kv_released()
        kv.reset_for_retract()
        self.assertIs(kv.retraction_backup, backup)
        kv.retraction_backup = None
        self.assertEqual(kv, ReqKvInfo())


if __name__ == "__main__":
    unittest.main()
