"""ReqKvInfo owns its lengths: code outside the class changes them only
through its named methods."""

import dataclasses
import unittest
from pathlib import Path

from sglang.srt.managers import schedule_batch
from sglang.srt.managers.schedule_batch import ReqKvInfo
from sglang.srt.mem_cache.unified_cache.component_type import ComponentType
from sglang.test import field_writes
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=20, suite="base-a-test-cpu")

_SRT_DIR = Path(schedule_batch.__file__).resolve().parents[1]

_LENGTH_FIELDS = {
    "cache_protected_len",
    "cache_inserted_len",
    "kv_committed_len",
    "kv_allocated_len",
    "swa_evict_floor",
    "component_evicted_seqlens",
}
_SCAN = dict(
    owner_scopes={"ReqKvInfo"},
    # The constructor and dataclasses.replace set fields by keyword.
    kwarg_funcs={"ReqKvInfo", "replace"},
    # A mutable field: any use outside ReqKvInfo can change it through an alias.
    owned_only_fields={"component_evicted_seqlens"},
)


class TestReqKvLengthWriters(CustomTestCase):
    def test_only_req_kv_info_methods_assign_lengths(self):
        found = field_writes.scan_tree(_SRT_DIR, _LENGTH_FIELDS, **_SCAN)
        self.assertFalse(
            found,
            "Write these lengths through a ReqKvInfo method:\n  " + "\n  ".join(found),
        )


# The int lengths; the evicted lengths are checked through their accessor.
_LENGTHS = sorted(_LENGTH_FIELDS - _SCAN["owned_only_fields"])


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
            ("set_row_len", (12,), dict(kv_committed_len=12, kv_allocated_len=12)),
            ("set_row_len", (5,), dict(kv_committed_len=5, kv_allocated_len=5)),
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
