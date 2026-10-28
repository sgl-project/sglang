"""ReqKvInfo owns its lengths: code outside the class changes them only
through its named methods."""

import ast
import dataclasses
import unittest
import warnings
from pathlib import Path

from sglang.srt.managers import schedule_batch
from sglang.srt.managers.schedule_batch import ReqKvInfo
from sglang.srt.mem_cache.unified_cache.component_type import ComponentType
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

_SRT_DIR = Path(schedule_batch.__file__).resolve().parents[1]

_LENGTH_FIELDS = {
    "cache_protected_len",
    "cache_inserted_len",
    "kv_committed_len",
    "kv_allocated_len",
    "swa_evict_floor",
    "component_evicted_seqlens",
}
_DICT_MUTATORS = {"clear", "pop", "popitem", "setdefault", "update"}


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


def _scan(path):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
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
        found = []
        for path in sorted(_SRT_DIR.rglob("*.py")):
            rel = path.relative_to(_SRT_DIR).as_posix()
            for scope, field, line in _scan(path):
                found.append(f"{rel}:{line} {scope} writes {field}")
        self.assertFalse(
            found,
            "Write these lengths through a ReqKvInfo method:\n  " + "\n  ".join(found),
        )


_LENGTHS = (
    "cache_protected_len",
    "cache_inserted_len",
    "kv_committed_len",
    "kv_allocated_len",
    "swa_evict_floor",
)


def _record(**lengths):
    kv = ReqKvInfo(**lengths)
    kv.set_evicted_seqlen(ComponentType.SWA, 6)
    return kv


def _lengths(kv):
    lengths = {name: getattr(kv, name) for name in _LENGTHS}
    lengths["swa_evicted"] = kv.get_evicted_seqlen(ComponentType.SWA)
    return lengths


class TestReqKvLengthMethods(CustomTestCase):
    def test_each_method_changes_only_its_lengths(self):
        start = dict(
            cache_protected_len=4,
            cache_inserted_len=8,
            kv_committed_len=8,
            kv_allocated_len=10,
            swa_evict_floor=2,
        )
        cases = [
            ("allocate_extend", (12,), dict(kv_committed_len=12, kv_allocated_len=12)),
            # Commits at allocation, not when the step's token is known.
            ("allocate_decode", (1,), dict(kv_committed_len=9, kv_allocated_len=11)),
            ("allocate_spec_decode", (12,), dict(kv_allocated_len=12)),
            ("allocate_spec_decode", (9,), {}),
            ("allocate_prealloc", (5,), dict(kv_allocated_len=5)),
            ("commit_prealloc", (5,), dict(kv_committed_len=5)),
            ("commit_decode", (2,), dict(kv_committed_len=10)),
            ("commit_finished", (9,), dict(kv_committed_len=9)),
            ("commit_finished", (12,), dict(kv_committed_len=10)),
            ("truncate_committed", (7,), dict(kv_committed_len=7)),
            ("truncate_committed", (9,), {}),
            (
                "truncate",
                (5,),
                dict(kv_committed_len=5, kv_allocated_len=5, swa_evicted=5),
            ),
            ("truncate", (12,), {}),
            (
                "free_tail",
                (5,),
                dict(kv_committed_len=5, kv_allocated_len=5, swa_evicted=5),
            ),
            # Unlike truncate, allocated becomes end even when below it.
            ("free_tail", (12,), dict(kv_allocated_len=12)),
            ("rewind", (5,), dict(kv_committed_len=5, kv_allocated_len=5)),
            ("protect_prefix", (6,), dict(cache_protected_len=6)),
            ("record_insert", (9,), dict(cache_inserted_len=9)),
            ("record_insert", (6,), {}),
            (
                "clear_tree_prefix",
                (),
                dict(cache_protected_len=0, cache_inserted_len=0),
            ),
            ("shield_swa_prefill", (9,), dict(swa_evict_floor=9)),
        ]
        for method, args, changed in cases:
            with self.subTest(method=method, args=args):
                kv = _record(**start)
                expected = {**_lengths(kv), **changed}
                getattr(kv, method)(*args)
                self.assertEqual(_lengths(kv), expected)

    def test_reset_for_retract_resets_only_its_subset(self):
        kv = ReqKvInfo(
            cache_protected_len=4,
            cache_inserted_len=8,
            kv_committed_len=8,
            kv_allocated_len=10,
            swa_evict_floor=2,
            retraction_backup=object(),
            restore_done_event=object(),
            mamba_ping_pong_empty=[True, False],
            mamba_next_track_idx=1,
            mamba_last_track_idx=0,
            mamba_last_track_seqlen=8,
            mamba_prev_track_seqlen=4,
            mamba_untracked_decode_batch_idx=3,
            mamba_needs_clear=True,
        )
        kv.set_evicted_seqlen(ComponentType.SWA, 6)
        reset = {
            "cache_protected_len",
            "cache_inserted_len",
            "kv_committed_len",
            "mamba_pool_idx",
            "mamba_ping_pong_track_buffer",
            "mamba_ping_pong_empty",
            "mamba_next_track_idx",
            "mamba_last_track_idx",
            "mamba_last_track_seqlen",
            "mamba_prev_track_seqlen",
            "mamba_untracked_decode_batch_idx",
            "mamba_cow_src_index",
            "mamba_needs_clear",
        }
        fresh = ReqKvInfo()
        before = {f.name: getattr(kv, f.name) for f in dataclasses.fields(kv)}
        kv.reset_for_retract()
        for name, value in before.items():
            expected = getattr(fresh, name) if name in reset else value
            with self.subTest(field=name):
                self.assertEqual(getattr(kv, name), expected)

        kv.req_pool_idx = 0
        with self.assertRaises(AssertionError):
            kv.reset_for_retract()


if __name__ == "__main__":
    unittest.main()
