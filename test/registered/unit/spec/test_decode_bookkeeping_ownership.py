"""Ownership contract for per-decode-iter bookkeeping.

Each decode iteration must run the per-request bookkeeping -- the
`req.decode_batch_idx` clock tick and the `batch.maybe_evict_swa()` call --
exactly once, by exactly one owner:

- non-spec decode: `ScheduleBatch.prepare_for_decode` (+ `alloc_for_decode`)
- spec v2 (any algorithm where `supports_spec_v2()` is true):
  `EagleDraftInputV2Mixin.prepare_for_decode`, driven by the scheduler.
  Spec-v2 draft workers must NOT repeat either operation.
- spec v1 workers own their bookkeeping inside the worker (the scheduler's
  `prepare_for_decode` returns early for them).

Double-ticking the clock is silent and dangerous: `decode_batch_idx` gates
when SWA eviction may fire (the first-decode-iter overlap race guard) and
when the SWA prefix tree lock is released (`>= sliding_window_size` uses the
iter count as a lower-bound proxy for generated tokens). A clock that runs
fast releases KV that in-flight or in-window readers still need; a clock
that never ticks merely wastes memory. Neither failure shows up in e2e CI
(short sequences, no memory pressure) or in the idle leak checker (the
freed-too-early path is accounting-clean), hence this source-level guard.

Both tests are static (AST-based) so they cover every code path of every
worker without needing GPU forwards, and new spec-v2 draft workers are
picked up automatically: subclass `BaseDraftWorker` and the contract test
applies to you with no edits here.
"""

import ast
import unittest
import warnings
from pathlib import Path

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=8, suite="base-a-test-cpu")

_THIS_FILE = Path(__file__).resolve()
# test/registered/unit/spec/<this file> -> repo root is 4 levels up.
_REPO_ROOT = _THIS_FILE.parents[4]
_SRT_DIR = _REPO_ROOT / "python" / "sglang" / "srt"
_SPECULATIVE_DIR = _SRT_DIR / "speculative"

# The clock attribute and the eviction entry point this contract is about.
_CLOCK_ATTR = "decode_batch_idx"
_EVICT_METHOD = "maybe_evict_swa"

# Every site in python/sglang/srt that ticks the clock or calls eviction,
# as (path relative to srt/, enclosing scope, kind). Kind is "tick" for a
# `decode_batch_idx` mutation (resets to literal 0 are exempt) and "evict"
# for a `maybe_evict_swa()` call.
#
# This list is exhaustive on purpose: adding a bookkeeping site anywhere in
# srt/ fails this test until the site is reviewed and recorded here, and
# removing one fails until it is dropped here. If you are adding a spec-v2
# draft worker, the answer is almost certainly that your worker must NOT
# tick or evict at all -- the scheduler-driven
# `EagleDraftInputV2Mixin.prepare_for_decode` already does both.
_OWNER_SITES = {
    # non-spec decode owner
    ("managers/schedule_batch.py", "ScheduleBatch.prepare_for_decode", "tick"),
    # allocation paths shared by non-spec decode and (for extend) all modes
    ("mem_cache/common.py", "alloc_for_extend", "evict"),
    ("mem_cache/common.py", "alloc_for_decode", "evict"),
    # spec v2 owner, driven by the scheduler via `batch.is_spec_v2`
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
    """Yield (scope_name, node) for every node, where scope_name is the
    dotted Class.method / function path enclosing the node."""
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
    # utf-8-sig: a couple of srt files carry a UTF-8 BOM, which plain utf-8
    # surfaces as U+FEFF and breaks ast.parse. SyntaxWarnings (e.g. invalid
    # escape sequences in scanned sources) are not this test's business.
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
                "New per-decode-iter bookkeeping site(s) found:\n  "
                + "\n  ".join(map(str, sorted(unexpected)))
                + "\nEach decode iter must tick `decode_batch_idx` and call "
                "`maybe_evict_swa` exactly once, by exactly one owner (see "
                "module docstring). If your code runs under spec v2, the "
                "scheduler-driven EagleDraftInputV2Mixin.prepare_for_decode "
                "already owns both -- do not repeat them. If you are adding "
                "a genuinely new owner, update _OWNER_SITES in this test."
            )
        if missing:
            msg.append(
                "Recorded bookkeeping site(s) no longer exist:\n  "
                + "\n  ".join(map(str, sorted(missing)))
                + "\nIf the owning code moved or was removed, update "
                "_OWNER_SITES so this allowlist stays exact."
            )
        self.assertFalse(msg, "\n\n".join(msg))

    def test_spec_v2_draft_workers_do_no_scheduler_bookkeeping(self):
        classes = _base_draft_worker_classes()
        names = {node.name for _, node in classes}
        # Scanner sanity: if discovery breaks, fail loudly instead of
        # silently guarding nothing.
        self.assertIn("EagleDraftWorker", names)
        self.assertIn("FrozenKVMTPDraftWorker", names)

        violations = []
        for rel, node in classes:
            for scope, kind in _scan_class_subtree(node):
                violations.append((rel, f"{node.name}.{scope}", kind))
        self.assertFalse(
            violations,
            "Spec-v2 draft worker(s) perform scheduler-owned bookkeeping:\n  "
            + "\n  ".join(map(str, sorted(violations)))
            + "\nUnder spec v2 the scheduler calls "
            "EagleDraftInputV2Mixin.prepare_for_decode every decode iter, "
            "which already ticks `decode_batch_idx` and calls "
            "`maybe_evict_swa`. Doing either again in the worker double-runs "
            "the bookkeeping (the clock runs 2x fast, SWA eviction fires in "
            "the first-decode-iter overlap race window, and the SWA prefix "
            "lock releases early). Remove these calls from the worker.",
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
