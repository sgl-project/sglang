"""Ownership contract for per-decode-iter bookkeeping.

Each decode iter must tick `req.decode_batch_idx` and call
`batch.maybe_evict_swa()` exactly once, by exactly one owner: the scheduler
for non-spec, `EagleDraftInputV2Mixin.prepare_for_decode` for spec v2 (draft
workers must not repeat them), the worker itself for spec v1.

A fast clock fires SWA eviction in the first-decode-iter overlap race window
and releases the SWA prefix lock before the decode position truly passes the
window; neither shows up in e2e CI or the idle leak checker. AST-based so it
covers every code path without GPU, and picks up new `BaseDraftWorker`
subclasses automatically.
"""

import ast
import unittest
import warnings
from pathlib import Path

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=8, suite="base-a-test-cpu")

_REPO_ROOT = Path(__file__).resolve().parents[4]
_SRT_DIR = _REPO_ROOT / "python" / "sglang" / "srt"
_SPECULATIVE_DIR = _SRT_DIR / "speculative"

_CLOCK_ATTR = "decode_batch_idx"
_EVICT_METHOD = "maybe_evict_swa"

# Exhaustive allowlist of (path relative to srt/, scope, kind), where kind is
# "tick" (a `decode_batch_idx` mutation; `= 0` resets exempt) or "evict" (a
# `maybe_evict_swa()` call). Any added or removed site fails the test until
# reviewed here; a spec-v2 draft worker almost never qualifies as a new owner.
_OWNER_SITES = {
    # non-spec decode
    ("managers/schedule_batch.py", "ScheduleBatch.prepare_for_decode", "tick"),
    ("mem_cache/common.py", "alloc_for_extend", "evict"),
    ("mem_cache/common.py", "alloc_for_decode", "evict"),
    # spec v2, scheduler-driven via `batch.is_spec_v2`
    (
        "speculative/eagle_info_v2.py",
        "EagleDraftInputV2Mixin.prepare_for_decode",
        "tick",
    ),
    (
        "speculative/eagle_info_v2.py",
        "EagleDraftInputV2Mixin.prepare_for_decode",
        "evict",
    ),
}


def _iter_scoped_nodes(tree):
    """Yield (node, dotted Class.method scope) for every node."""
    scope_of = {}

    def visit(node, scope):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            scope = f"{scope}.{node.name}" if scope else node.name
        scope_of[node] = scope
        for child in ast.iter_child_nodes(node):
            visit(child, scope)

    visit(tree, "")
    return scope_of.items()


def _is_zero_reset(node):
    return isinstance(node, ast.Assign) and (
        isinstance(node.value, ast.Constant) and node.value.value == 0
    )


def _scan_tree(tree):
    """Return the set of (scope, kind) bookkeeping sites in an AST."""
    sites = set()
    for node, scope in _iter_scoped_nodes(tree):
        if isinstance(node, (ast.AugAssign, ast.Assign)):
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            for target in targets:
                if (
                    isinstance(target, ast.Attribute)
                    and target.attr == _CLOCK_ATTR
                    and not _is_zero_reset(node)
                ):
                    sites.add((scope, "tick"))
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == _EVICT_METHOD
        ):
            sites.add((scope, "evict"))
    return sites


def _parse(path: Path):
    # utf-8-sig: some srt files carry a BOM that breaks plain-utf-8 ast.parse.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", SyntaxWarning)
        return ast.parse(path.read_text(encoding="utf-8-sig"))


def _scan_file(path: Path):
    return _scan_tree(_parse(path))


def _scan_srt():
    """Return all bookkeeping sites in srt/ as (rel_path, scope, kind)."""
    found = set()
    for path in sorted(_SRT_DIR.rglob("*.py")):
        rel = path.relative_to(_SRT_DIR).as_posix()
        for scope, kind in _scan_file(path):
            found.add((rel, scope, kind))
    return found


def _base_draft_worker_classes():
    """Find every class under speculative/ that lists BaseDraftWorker as a
    direct base, as (rel_path, ClassDef)."""
    classes = []
    for path in sorted(_SPECULATIVE_DIR.glob("*.py")):
        tree = _parse(path)
        for node in ast.walk(tree):
            if not isinstance(node, ast.ClassDef):
                continue
            base_names = {
                base.id if isinstance(base, ast.Name) else getattr(base, "attr", None)
                for base in node.bases
            }
            if "BaseDraftWorker" in base_names:
                classes.append((path.relative_to(_SRT_DIR).as_posix(), node))
    return classes


class TestDecodeBookkeepingOwnership(CustomTestCase):
    def test_bookkeeping_sites_match_owner_allowlist(self):
        found = _scan_srt()
        unexpected = found - _OWNER_SITES
        missing = _OWNER_SITES - found
        msg = []
        if unexpected:
            msg.append(
                "New bookkeeping site(s):\n  "
                + "\n  ".join(map(str, sorted(unexpected)))
                + "\nUnder spec v2 these are owned by "
                "EagleDraftInputV2Mixin.prepare_for_decode -- do not repeat "
                "them; a genuinely new owner must be added to _OWNER_SITES."
            )
        if missing:
            msg.append(
                "Recorded site(s) no longer exist (update _OWNER_SITES):\n  "
                + "\n  ".join(map(str, sorted(missing)))
            )
        self.assertFalse(msg, "\n\n".join(msg))

    def test_spec_v2_draft_workers_do_no_scheduler_bookkeeping(self):
        classes = _base_draft_worker_classes()
        names = {node.name for _, node in classes}
        # Discovery sanity: fail loudly instead of silently guarding nothing.
        self.assertIn("EagleDraftWorker", names)
        self.assertIn("FrozenKVMTPDraftWorker", names)

        violations = []
        for rel, node in classes:
            for scope, kind in _scan_class_subtree(node):
                violations.append((rel, f"{node.name}.{scope}", kind))
        self.assertFalse(
            violations,
            "Spec-v2 draft worker(s) repeat scheduler-owned bookkeeping:\n  "
            + "\n  ".join(map(str, sorted(violations)))
            + "\nEagleDraftInputV2Mixin.prepare_for_decode already ticks "
            "`decode_batch_idx` and calls `maybe_evict_swa` every decode "
            "iter; doing either again double-runs them. Remove these calls.",
        )


def _scan_class_subtree(class_node):
    """Scan one ClassDef subtree; returns (method_scope, kind) sites."""
    module = ast.Module(body=[class_node], type_ignores=[])
    sites = set()
    for scope, kind in _scan_tree(module):
        # Strip the leading class name; keep method-level scope.
        sites.add((scope.split(".", 1)[1] if "." in scope else scope, kind))
    return sites


if __name__ == "__main__":
    unittest.main(verbosity=3)
