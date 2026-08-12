"""Regression test: a parked chunked request must not be counted as in-flight.

`PrefillAdder.add_chunked_req` can return the request *without* adding it to
`can_run_list` (the hybrid-SWA branch, when the SWA pool cannot fit another
page). The request is "parked": no KV is computed for it this pass and it
produces no forward result. Incrementing `inflight_middle_chunks` for it leaks
the counter upward, and a request stuck above zero is treated as a middle chunk
forever -- it never appends a token and never finishes.
"""

import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase, maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()

from sglang.srt.managers.schedule_batch import Req
from sglang.srt.managers.schedule_policy import PrefillAdder
from sglang.srt.managers.scheduler import Scheduler
from sglang.srt.mem_cache.base_prefix_cache import DecLockRefResult, IncLockRefResult
from sglang.srt.server_args import ServerArgs, set_global_server_args_for_scheduler
from sglang.srt.utils.common import Range

register_cpu_ci(est_time=3, suite="base-a-test-cpu")

PAGE = 8
WINDOW = 128
PREFIX = 100


def _make_req(rid: str, extend_len: int) -> Req:
    req = MagicMock(spec=Req)
    req.rid = rid
    req.priority = 0
    req.prefix_indices = list(range(PREFIX))
    req.full_untruncated_fill_ids = list(range(PREFIX + extend_len))
    req.origin_input_ids = list(range(PREFIX + extend_len))
    req.output_ids = []
    req.sampling_params = SimpleNamespace(max_new_tokens=40, ignore_eos=False)
    req.retracted_stain = False
    req.host_hit_length = 0
    req.storage_hit_length = 0
    req.swa_host_hit_length = 0
    req.mamba_pool_idx = None
    req.last_node = MagicMock()
    req.inflight_middle_chunks = 0
    req.finished.return_value = False
    req.needs_host_load_back.return_value = False
    req.set_extend_range = MagicMock(
        side_effect=lambda s, e: setattr(req, "extend_range", Range(s, e))
    )
    return req


def _make_adder(*, swa_available: int) -> PrefillAdder:
    tree_cache = MagicMock()
    tree_cache.full_evictable_size.return_value = 0
    tree_cache.swa_evictable_size.return_value = 0
    tree_cache.evictable_size.return_value = 0
    tree_cache.disable = False
    tree_cache.sliding_window_size = WINDOW
    tree_cache.is_tree_cache.return_value = False
    tree_cache.inc_lock_ref.return_value = IncLockRefResult()
    tree_cache.dec_lock_ref.return_value = DecLockRefResult()

    allocator = MagicMock()
    allocator.full_available_size.return_value = 10_000_000
    allocator.available_size.return_value = 10_000_000
    allocator.swa_available_size.return_value = swa_available
    allocator.size_swa = 10_000_000

    running_batch = MagicMock()
    running_batch.reqs = []

    adder = PrefillAdder(
        page_size=PAGE,
        tree_cache=tree_cache,
        token_to_kv_pool_allocator=allocator,
        running_batch=running_batch,
        new_token_ratio=1.0,
        rem_input_tokens=100_000,
        rem_chunk_tokens=512,
        num_mixed_decode_tokens=0,
        priority_scheduling_preemption_threshold=0,
    )
    adder.is_hybrid_swa = True
    return adder


def _count_via_scheduler(adder: PrefillAdder, chunked_req: Req) -> None:
    """Run the scheduler's real accounting step over this pass's batch.

    Calls Scheduler._count_inflight_chunk itself rather than restating its
    logic, so deleting the parked-chunk guard fails these tests.
    """
    scheduler = Scheduler.__new__(Scheduler)
    scheduler.chunked_req = chunked_req
    Scheduler._count_inflight_chunk(scheduler, set(adder.can_run_list))


class TestParkedChunkAccounting(CustomTestCase):
    def setUp(self):
        super().setUp()
        set_global_server_args_for_scheduler(ServerArgs(model_path="dummy"))

    def test_parked_chunk_is_not_counted_as_inflight(self):
        # SWA pool below one page: add_chunked_req parks the request.
        adder = _make_adder(swa_available=PAGE // 2)
        req = _make_req("parked", extend_len=100_000)

        returned = adder.add_chunked_req(req)

        # Parked: still the chunked req, but absent from the batch.
        self.assertIs(returned, req)
        self.assertNotIn(req, adder.can_run_list)

        _count_via_scheduler(adder, returned)

        # Nothing will decrement it, so it must never have been incremented.
        self.assertEqual(req.inflight_middle_chunks, 0)

    def test_prefilled_chunk_is_counted_as_inflight(self):
        # Symmetric guard against over-gating: an ample SWA pool prefills the
        # chunk normally, and that one must still be counted.
        adder = _make_adder(swa_available=100_000)
        req = _make_req("prefilled", extend_len=100_000)

        returned = adder.add_chunked_req(req)

        self.assertIs(returned, req)  # still truncated
        self.assertIn(req, adder.can_run_list)

        _count_via_scheduler(adder, returned)

        self.assertEqual(req.inflight_middle_chunks, 1)


if __name__ == "__main__":
    unittest.main()
