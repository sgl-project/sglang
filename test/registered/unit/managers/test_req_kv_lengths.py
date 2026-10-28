"""ReqKvInfo owns its lengths: code outside the class changes them only
through its named methods."""

import ast
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

# (path relative to python/sglang/srt, enclosing scope) of writers not yet
# routed through a ReqKvInfo method.
_ALLOWED_WRITERS = {
    ("beam_search/fork.py", "free_member_rows"),
    ("disaggregation/decode.py", "_bind_root_prefix"),
    ("disaggregation/decode.py", "DecodePreallocQueue.pop_preallocated"),
    ("disaggregation/decode.py", "DecodePreallocQueue._pre_alloc"),
    ("disaggregation/decode.py", "alloc_for_decode_prealloc_hisparse"),
    ("disaggregation/decode.py", "alloc_for_decode_prealloc"),
    ("dllm/mixin/scheduler.py", "SchedulerDllmMixin._stash_dllm_context"),
    ("managers/schedule_batch.py", "Req.init_next_round_input"),
    ("managers/schedule_batch.py", "Req.reset_for_retract"),
    ("managers/schedule_policy.py", "PrefillAdder.add_one_req"),
    ("managers/scheduler.py", "Scheduler._get_new_batch_prefill_raw"),
    (
        "managers/scheduler_components/batch_result_processor.py",
        "SchedulerBatchResultProcessor.process_batch_result_prefill",
    ),
    (
        "managers/scheduler_components/batch_result_processor.py",
        "SchedulerBatchResultProcessor._resolve_spec_v2_tokens",
    ),
}


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
        found, allowed_seen = [], set()
        for path in sorted(_SRT_DIR.rglob("*.py")):
            rel = path.relative_to(_SRT_DIR).as_posix()
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
            _ALLOWED_WRITERS - allowed_seen,
            set(),
            "allowlisted writers that are gone; drop them from _ALLOWED_WRITERS",
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


if __name__ == "__main__":
    unittest.main()
