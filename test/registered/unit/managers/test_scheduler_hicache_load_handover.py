"""A new prefill batch issues its own host-to-device copies before its HiCache
load burst is handed over.

start_loading records the burst's start event on the stream it is called on, so
only a copy issued before the hand-over is ordered ahead of the burst. One issued
after it can wait for the whole burst on the copy engine, and the forward with it.
"""

import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase, maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()

from sglang.srt.managers.overlap_utils import resolve_forward_inputs
from sglang.srt.managers.schedule_batch import ScheduleBatch
from sglang.srt.managers.schedule_policy import AddReqResult, PrefillAdder
from sglang.srt.managers.scheduler import Scheduler
from sglang.srt.managers.scheduler_components.request_list import (
    IndexedReqList,
    KeyedReqList,
    ListKind,
)
from sglang.srt.runtime_context import get_context

register_cpu_ci(est_time=11, suite="base-a-test-cpu")

TOKENS = [1, 2, 3, 4]


class _Req:
    def __init__(self):
        self.rid = "r"
        self.kv = SimpleNamespace(holds_kv=False, holds_mamba=False)
        self.beam_group = None
        self.token_indices_to_pool = None

    def init_next_round_input(self, tree_cache=None):
        pass


class _Adder(PrefillAdder):
    def __init__(self, *args, **kwargs):
        self.can_run_list = []
        self.preempt_list = []
        self.new_chunked_req = None
        self.rem_chunk_tokens = 0

    def budget_state(self):
        return AddReqResult.CONTINUE

    def add_one_req(self, req, has_chunked_req, truncation_align_size):
        self.can_run_list.append(req)
        return AddReqResult.CONTINUE


class _PinnedTokens:
    """Stands in for the pinned host staging; records when it is uploaded."""

    def __init__(self, order):
        self.order = order

    def to(self, device, non_blocking=False):
        self.order.append("upload")
        return torch.tensor(TOKENS, dtype=torch.int64, device=device)


def _batch(order=None, **over):
    fields = dict(
        device="cpu",
        prefill_input_ids_cpu=_PinnedTokens([] if order is None else order),
        mix_running_indices=None,
        input_ids=None,
        hicache_consumer_index=-1,
    )
    fields.update(over)
    return SimpleNamespace(**fields)


def _handover_scheduler(order, hicache=True, linker=False):
    s = Scheduler.__new__(Scheduler)
    s.enable_hierarchical_cache = hicache
    s.enable_unified_cache_external_linker = linker
    s.tree_cache = SimpleNamespace(
        ready_to_load_host_cache=lambda: order.append("handover") or 7
    )
    return s


def _handover(order, batch, pp_size=1, **flags):
    with get_context().override_server_args(pp_size=pp_size):
        Scheduler._handover_hicache_load(_handover_scheduler(order, **flags), batch)


class TestHiCacheLoadHandover(CustomTestCase):
    def test_batch_formation_uploads_before_the_handover(self):
        order = []
        req = _Req()
        new_batch = _batch(
            order, prefill_input_ids_cpu=None, return_logprob=False, input_embeds=None
        )

        def prepare_for_extend():
            order.append("prepare")
            new_batch.prefill_input_ids_cpu = _PinnedTokens(order)

        new_batch.prepare_for_extend = prepare_for_extend

        s = _handover_scheduler(order)
        s.tree_cache.buffer_pipeline = None
        s.tree_cache.storage_prefetch_retries = None
        s.grammar_manager = SimpleNamespace(has_waiting_grammars=lambda: False)
        s.enable_priority_preemption = False
        s.enable_priority_scheduling = False
        s.is_hybrid_swa = False
        s.waiting_queue = IndexedReqList(ListKind.WAITING)
        s.waiting_queue.join([req])
        s.chunked_reqs = IndexedReqList(ListKind.CHUNKED)
        s.chunked_req = None
        s.running_reqs = KeyedReqList(ListKind.DECODING)
        s.min_free_slots_delayer = None
        s.invariant_checker = Mock()
        s.policy = Mock(calc_priority=lambda waiting, *args, **kwargs: list(waiting))
        s.policy.shortest_prefill_chunk_limit.return_value = None
        s.processed_tokens_counter = 0
        s.chunked_prefill_size = 8192
        s.dynamic_chunk_sizer = None
        s.tp_worker = SimpleNamespace(
            model_runner=SimpleNamespace(
                attn_backend=SimpleNamespace(), prefill_aware_swa=False
            )
        )
        s.get_prefill_adder_class = lambda: _Adder
        s.get_num_allocatable_reqs = lambda *args, **kwargs: 4
        s.request_retirer = None
        s.page_size = 1
        s.token_to_kv_pool_allocator = SimpleNamespace(state_slots=None)
        s.req_to_token_pool = None
        s.new_token_ratio_tracker = SimpleNamespace(current=0.0)
        s.max_prefill_tokens = 1 << 20
        s.is_mixed_chunk = False
        s.priority_scheduling_preemption_threshold = 0
        s.max_prefill_bs = 0
        s.max_running_requests = 4
        s.dllm_config = None
        s.enable_lora = False
        s.disaggregation_mode = None
        s.disagg_prefill_bootstrap_queue = None
        s.disagg_prefill_transfer_queue = []
        s.enable_hicache_storage = False
        s.enable_lmcache = False
        s.truncation_align_size = None
        s.model_config = None
        s.enable_overlap = False
        s.spec_algorithm = None
        s.hisparse_coordinator = None
        s.load_inquirer = SimpleNamespace(
            _get_num_pending_tokens=lambda chunk_deduct: 0
        )

        with (
            get_context().override_server_args(pp_size=1),
            patch.object(ScheduleBatch, "init_new", return_value=new_batch),
            patch("sglang.srt.managers.scheduler.set_time_batch"),
            patch("sglang.srt.managers.scheduler.PrefillStats"),
        ):
            built, _ = s._get_new_batch_prefill_raw(
                prefill_delayer_single_pass=None,
                running_batch=ScheduleBatch(reqs=[]),
            )

        self.assertIs(built, new_batch)
        self.assertEqual(order, ["prepare", "upload", "handover"])
        self.assertEqual(built.hicache_consumer_index, 7)

    def test_forward_entry_keeps_the_uploaded_input_ids(self):
        batch = _batch(
            req_pool_indices=torch.tensor([5, 6]),
            enable_overlap=False,
            spec_algorithm=SimpleNamespace(is_none=lambda: True),
        )
        _handover([], batch)
        uploaded = batch.input_ids
        self.assertEqual(uploaded.tolist(), TOKENS)
        self.assertIsNone(batch.prefill_input_ids_cpu)

        # A decode gather over the batch's rows would replace the prompt tokens.
        future_map = SimpleNamespace(
            takes_deferred_relay=lambda batch: True,
            output_tokens_buf=torch.arange(100, dtype=torch.int64),
            spec_algo=SimpleNamespace(is_none=lambda: True),
        )
        resolve_forward_inputs(batch, future_map)
        self.assertIs(batch.input_ids, uploaded)

    def test_mixed_batch_keeps_the_deferred_upload(self):
        # Its input_ids also gather the running rows' tokens at forward entry.
        order = []
        batch = _batch(order, mix_running_indices=torch.tensor([0, 1]))
        _handover(order, batch)
        self.assertEqual(order, ["handover"])
        self.assertIsNone(batch.input_ids)
        self.assertIsNotNone(batch.prefill_input_ids_cpu)

    def test_pipeline_stage_keeps_the_deferred_upload(self):
        # Its forward runs on another stream and outlives the batch's input_ids.
        order = []
        batch = _batch(order)
        _handover(order, batch, pp_size=2)
        self.assertEqual(order, ["handover"])
        self.assertIsNone(batch.input_ids)
        self.assertEqual(batch.hicache_consumer_index, 7)

    def test_no_host_loads_leaves_the_batch_untouched(self):
        order = []
        batch = _batch(order)
        _handover(order, batch, hicache=False)
        self.assertEqual(order, [])
        self.assertIsNotNone(batch.prefill_input_ids_cpu)
        self.assertEqual(batch.hicache_consumer_index, -1)

    def test_external_linker_alone_hands_over(self):
        order = []
        batch = _batch(order)
        _handover(order, batch, hicache=False, linker=True)
        self.assertEqual(order, ["upload", "handover"])


if __name__ == "__main__":
    unittest.main()
