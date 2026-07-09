"""Behavioral regression test for the chunked-prefill slot double-count in
``Scheduler._get_new_batch_prefill_raw``.

An in-flight chunked prefill already holds its ``req_to_token_pool`` slot (that
slot is deducted from ``available_size()``) AND is re-appended to the adder's
``can_run_list`` by ``add_chunked_req``. The admission gate

    num_allocatable = min(
        pp_max_micro_batch_size - running_bs,
        available_size() + int(chunked_req_holds_slot),
    )
    if len(can_run_list) >= num_allocatable:
        batch_is_full = True

must add that already-held slot back (the ``+ int(chunked_req_holds_slot)``
term). Without it the batch is declared full one request early, and because
``batch_is_full`` is sticky (reset only when a request finishes) the rank parks
at ``max_running_requests - 1`` with requests still queued -- the DP-attention
"caps at 15 instead of 16" symptom.

Rather than re-assert the arithmetic, these tests drive the real
``_get_new_batch_prefill_raw`` and check the observable outcome: is the queued
request actually *issued* (moved out of the waiting queue into the prefill
batch) or left parked? The admission gate lives in the method itself and only
reads ``can_run_list`` plus the result of ``add_one_req``, so a faithful fake
``PrefillAdder`` (add_chunked_req re-appends the in-flight chunk; add_one_req
always admits -- there is token budget, only the slot/ceiling gate limits us)
leaves the real gate under test. ``available_size()`` is held constant because
the real ``add_one_req`` allocates no ``req_to_token_pool`` slot -- those are
allocated later in ``prepare_for_extend`` -- so free-slot count does not change
across the admit loop.
"""

import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from sglang.srt.managers.schedule_policy import AddReqResult
from sglang.srt.managers.scheduler import Scheduler
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase, maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()

register_cpu_ci(est_time=8, suite="base-a-test-cpu")


class _FakeAdder:
    """Minimal stand-in for ``PrefillAdder``.

    The admission gate under test lives in ``_get_new_batch_prefill_raw`` itself
    and only reads ``can_run_list`` and the return of ``add_one_req``. This fake
    models exactly that: ``add_chunked_req`` re-appends the in-flight chunk
    (whose pool slot is already held from a previous pass), and ``add_one_req``
    always admits -- in these scenarios there is always token budget, so only the
    slot/ceiling gate can limit admission.
    """

    def __init__(self, *args, **kwargs):
        self.can_run_list = []
        self.preempt_list = []
        self.new_chunked_req = None

    def add_chunked_req(self, req):
        self.can_run_list.append(req)
        return req  # truncated=True: still chunking, not the final chunk

    def add_one_req(self, req, has_chunked_req=False, truncation_align_size=None):
        self.can_run_list.append(req)
        return AddReqResult.CONTINUE


def _make_scheduler(*, ceiling, running_bs, pool_avail, chunked_req, waiting_queue):
    """A ``Scheduler`` skeleton carrying only the attributes
    ``_get_new_batch_prefill_raw`` reads on the path exercised here."""
    s = Scheduler.__new__(Scheduler)
    # _run_prefill patches get_global_server_args() to return this, so the
    # ceiling term of the gate reads pp_max_micro_batch_size == ceiling.
    s._server_args_stub = SimpleNamespace(pp_max_micro_batch_size=ceiling)
    s.grammar_manager = MagicMock()
    s.grammar_manager.has_waiting_grammars.return_value = False
    s.enable_hierarchical_cache = False
    s.enable_priority_preemption = False
    s.is_hybrid_swa = False
    s.spec_algorithm = MagicMock()
    s.min_free_slots_delayer = None  # skip the free-slots prefill delay
    s.policy = MagicMock()
    s.chunked_prefill_size = 8192
    s.enable_dynamic_chunking = False
    s.page_size = 1
    s.tree_cache = MagicMock()
    s.token_to_kv_pool_allocator = MagicMock()
    s.new_token_ratio_tracker = SimpleNamespace(current=0.0)
    s.max_prefill_tokens = 1 << 20
    s.is_mixed_chunk = False
    s.priority_scheduling_preemption_threshold = 0
    s.max_prefill_bs = 0
    s.max_running_requests = ceiling
    s.server_args = SimpleNamespace(prefill_max_requests=None)
    s.prefill_delayer = None
    s.dllm_config = None
    s.enable_lora = False
    s.disaggregation_mode = None  # != DisaggregationMode.PREFILL
    s.truncation_align_size = None
    s.enable_hicache_storage = False
    s.enable_priority_scheduling = False
    s.load_inquirer = MagicMock()
    s.model_config = MagicMock()
    s.enable_overlap = False
    s.tp_worker = SimpleNamespace(model_runner=SimpleNamespace(prefill_aware_swa=False))
    # A pool with no ``mamba_allocator`` attr, so getattr(..., None) short-circuits.
    s.req_to_token_pool = SimpleNamespace(available_size=lambda: pool_avail)
    s.running_batch = SimpleNamespace(
        reqs=list(range(running_bs)),
        batch_is_full=False,
        return_logprob=False,
        is_empty=lambda: running_bs == 0,
    )
    s.chunked_req = chunked_req
    s.waiting_queue = list(waiting_queue)
    return s


def _run_prefill(s):
    """Drive the real gate; stub only the batch-construction tail (real KV pools
    / CUDA) and the ``PrefillAdder`` factory, capturing the adder instance so the
    resulting ``can_run_list`` can be inspected."""
    created = []

    def _factory(*args, **kwargs):
        adder = _FakeAdder()
        created.append(adder)
        return adder

    with patch(
        "sglang.srt.managers.scheduler.PrefillAdder", side_effect=_factory
    ), patch("sglang.srt.managers.scheduler.ScheduleBatch"), patch(
        "sglang.srt.managers.scheduler.PrefillStats"
    ), patch(
        "sglang.srt.managers.scheduler.get_global_server_args",
        return_value=s._server_args_stub,
    ):
        batch = Scheduler._get_new_batch_prefill_raw(
            s, prefill_delayer_single_pass=None
        )
    return batch, created[0]


class TestGetNewBatchPrefillChunkedSlot(CustomTestCase):
    def test_chunked_slot_added_back_so_queued_req_is_issued(self):
        # Live-trace scenario: ceiling 16, 14 decoding, a chunk in flight holding
        # the 15th slot, exactly one free req_to_token_pool slot, one queued req.
        #   buggy: min(16-14, 1)     = 1 -> can_run_list 1 >= 1 -> FULL, parked at 15
        #   fixed: min(16-14, 1 + 1) = 2 -> can_run_list 1 <  2 -> issue queued -> 16
        q = MagicMock(name="queued")
        chunk = MagicMock(name="chunked")
        s = _make_scheduler(
            ceiling=16,
            running_bs=14,
            pool_avail=1,
            chunked_req=chunk,
            waiting_queue=[q],
        )

        batch, adder = _run_prefill(s)

        self.assertIn(q, adder.can_run_list)  # the queued request was issued
        self.assertEqual(adder.can_run_list, [chunk, q])
        self.assertEqual(s.waiting_queue, [])  # ... and dequeued
        self.assertFalse(s.running_batch.batch_is_full)
        self.assertIsNotNone(batch)

    def test_ceiling_term_still_caps_total_no_over_admit(self):
        # The +1 compensation must not over-admit past the ceiling. At running_bs
        # 15 with a chunk in flight, 15 decoding + 1 chunk already == 16 == ceiling,
        # so the queued req must be refused even though the pool has spare slots.
        q = MagicMock(name="queued")
        chunk = MagicMock(name="chunked")
        s = _make_scheduler(
            ceiling=16,
            running_bs=15,
            pool_avail=8,
            chunked_req=chunk,
            waiting_queue=[q],
        )

        _, adder = _run_prefill(s)

        self.assertNotIn(q, adder.can_run_list)
        self.assertEqual(adder.can_run_list, [chunk])
        self.assertEqual(s.waiting_queue, [q])  # left queued
        self.assertTrue(s.running_batch.batch_is_full)

    def test_no_chunk_in_flight_is_unchanged_legacy_behavior(self):
        # No chunk -> no compensation (+0). With one free slot and ceiling room
        # for two, exactly one new req is admitted (legacy min(16-14, 1) == 1);
        # the second is left queued. Confirms the fix is scoped to the chunk case.
        q1 = MagicMock(name="q1")
        q2 = MagicMock(name="q2")
        s = _make_scheduler(
            ceiling=16,
            running_bs=14,
            pool_avail=1,
            chunked_req=None,
            waiting_queue=[q1, q2],
        )

        _, adder = _run_prefill(s)

        self.assertEqual(adder.can_run_list, [q1])
        self.assertEqual(s.waiting_queue, [q2])
        self.assertTrue(s.running_batch.batch_is_full)


if __name__ == "__main__":
    unittest.main()
