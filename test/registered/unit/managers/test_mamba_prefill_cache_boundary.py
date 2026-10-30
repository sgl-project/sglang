"""Publish canonical Mamba prefix slots before selecting the next batch."""

import unittest
from collections import deque
from types import SimpleNamespace as NS
from unittest.mock import patch

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()

from sglang.srt.managers.scheduler import Scheduler

register_cpu_ci(est_time=10, suite="base-a-test-cpu")


class Batch:
    def __init__(self, name, req):
        self.name = name
        self.reqs = [req]
        self.decoding_reqs = None
        self.forward_mode = NS(is_extend=lambda: name == "prefill")

    def copy(self):
        return self


class OverlapHarness:
    event_loop_overlap = Scheduler.event_loop_overlap

    def _has_pending_mamba_cache_update(self):
        return Scheduler._has_pending_mamba_cache_update(self)

    def __init__(self, *, finish_in_prefill=False, fail_prefill=False):
        self.require_mlp_sync = False
        self.tree_cache = NS(disable=False, enable_mamba_extra_buffer=True)
        self.req = NS(
            kv=NS(mamba_last_track_seqlen=384),
            is_retracted=False,
            skip_radix_cache_insert=False,
            finished=lambda: self.done,
        )
        self.gracefully_exit = False
        self._engine_paused = False
        self.is_generation = True
        self.enable_unified_memory = False
        self.last_batch = None
        self.running_batch = None
        self.events = []
        self.iteration = 0
        self.scheduled = 0
        self.done = False
        self.finish_in_prefill = finish_in_prefill
        self.fail_prefill = fail_prefill
        self.private_slots = [91, 92, 93]
        self.canonical_slots = [1, 2, 3]
        self.mapping = list(self.private_slots)
        self.freed = []
        self.decode_mapping = None
        self.need_overlap_sync = False

    def ingest_requests(self):
        self.iteration += 1
        if self.iteration > 4:
            raise StopIteration

    def get_next_batch_to_run(self, *, running_batch, last_batch):
        del last_batch
        self.events.append("schedule")
        if self.done or self.scheduled == 2:
            batch = None
        elif self.scheduled == 0:
            batch = Batch("prefill", self.req)
            self.scheduled += 1
        else:
            # prepare_for_decode/verify captures the cache mapping here, before
            # run_batch. Processing the prefill result after selection is late.
            self.decode_mapping = list(self.mapping)
            batch = Batch("decode", self.req)
            self.scheduled += 1
        return NS(batch_to_run=batch, running_batch=running_batch)

    def is_disable_overlap_for_batch(self, batch, *, last_batch):
        return self.need_overlap_sync and last_batch is not None

    def run_batch(self, batch):
        self.events.append("run:" + batch.name)
        return object()

    def _apply_war_barrier(self):
        pass

    def process_batch_result(self, batch, result):
        self.events.append("process:" + batch.name)
        if batch.name == "prefill":
            if self.fail_prefill:
                raise RuntimeError("prefill failed")
            self.freed.extend(self.private_slots)
            self.mapping = list(self.canonical_slots)
            self.req.kv.mamba_last_track_seqlen = None
            if self.finish_in_prefill:
                self.done = True
        else:
            self.done = True

    def launch_batch_sample_if_needed(self, result, batch):
        pass

    def on_idle(self):
        pass


class TestMambaPrefillCacheBoundary(unittest.TestCase):
    def run_loop(self, harness):
        with (
            patch(
                "sglang.srt.managers.scheduler.envs."
                "SGLANG_ENABLE_STRICT_MEM_CHECK_DURING_BUSY.get",
                return_value=False,
            ),
            self.assertRaises(StopIteration),
        ):
            harness.event_loop_overlap()

    def test_publish_before_selection_and_process_each_result_once(self):
        for need_overlap_sync in (False, True):
            with self.subTest(need_overlap_sync=need_overlap_sync):
                h = OverlapHarness()
                h.need_overlap_sync = need_overlap_sync
                h.enable_unified_memory = True
                h.token_to_kv_pool_allocator = NS(
                    flush_opportunistic=lambda: h.events.append("flush")
                )
                self.run_loop(h)
                self.assertEqual(h.decode_mapping, h.canonical_slots)
                self.assertFalse(set(h.decode_mapping) & set(h.freed))
                self.assertEqual(h.events.count("process:prefill"), 1)
                self.assertEqual(h.events.count("process:decode"), 1)
                self.assertEqual(h.freed, h.private_slots)
                self.assertEqual(len(h.result_queue), 0)
                self.assertLess(
                    h.events.index("process:prefill"), h.events.index("flush")
                )
                self.assertLess(h.events.index("flush"), h.events.index("run:decode"))

    def test_finish_in_prefill_does_not_schedule_decode(self):
        h = OverlapHarness(finish_in_prefill=True)
        self.run_loop(h)
        self.assertNotIn("run:decode", h.events)
        self.assertIsNone(h.decode_mapping)
        self.assertEqual(h.events.count("process:prefill"), 1)

    def test_failed_prefill_does_not_launch_dependent_batch(self):
        h = OverlapHarness(fail_prefill=True)
        with self.assertRaisesRegex(RuntimeError, "prefill failed"):
            h.event_loop_overlap()
        self.assertNotIn("run:decode", h.events)
        self.assertIsNone(h.decode_mapping)

    def test_untracked_prefill_keeps_overlap(self):
        h = OverlapHarness()
        h.req.kv.mamba_last_track_seqlen = None
        self.run_loop(h)
        self.assertLess(h.events.index("run:decode"), h.events.index("process:prefill"))

    def test_only_pending_cache_publications_need_barrier(self):
        for case in (
            "dp",
            "disabled",
            "nonmamba",
            "untracked",
            "finished",
            "retracted",
            "skip_insert",
            "decode",
            "mixed_decode",
            "empty",
        ):
            with self.subTest(case=case):
                h = OverlapHarness()
                batch = Batch("prefill", h.req)
                h.result_queue = deque([(batch, object())])
                if case == "dp":
                    h.require_mlp_sync = True
                elif case == "disabled":
                    h.tree_cache.disable = True
                elif case == "nonmamba":
                    h.tree_cache.enable_mamba_extra_buffer = False
                elif case == "untracked":
                    h.req.kv.mamba_last_track_seqlen = None
                elif case == "finished":
                    h.done = True
                elif case == "retracted":
                    h.req.is_retracted = True
                elif case == "skip_insert":
                    h.req.skip_radix_cache_insert = True
                elif case == "decode":
                    h.result_queue[0] = (Batch("decode", h.req), object())
                elif case == "mixed_decode":
                    batch.decoding_reqs = [h.req]
                else:
                    h.result_queue.clear()
                self.assertFalse(h._has_pending_mamba_cache_update())


if __name__ == "__main__":
    unittest.main()
