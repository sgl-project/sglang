"""Regression tests for the SWA chunked-req stash gate (#24252)."""

import unittest
from array import array
from types import SimpleNamespace
from unittest.mock import MagicMock

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import (
    CustomTestCase,
    enter_scope,
    maybe_stub_sgl_kernel,
    published_topology,
)

maybe_stub_sgl_kernel()

from sglang.srt.disaggregation.utils import DisaggregationMode
from sglang.srt.managers.request_lists import CoreLists
from sglang.srt.managers.schedule_batch import Req, ReqKvInfo
from sglang.srt.managers.scheduler import Scheduler
from sglang.srt.managers.scheduler_assembly import init_mode_stages  # noqa: E402
from sglang.srt.managers.scheduler_components.mode_owners import ModeOwners
from sglang.srt.managers.scheduler_components.mode_stages import ScheduleSlot
from sglang.srt.managers.scheduler_components.step_planner import StepPlanner
from sglang.srt.mem_cache.base_prefix_cache import BasePrefixCache
from sglang.test.scheduler_request_lists import partial_scheduler  # noqa: E402

register_cpu_ci(est_time=11, suite="base-a-test-cpu")


def _make_req(
    *,
    req_pool_idx: int,
    fill_ids: list,
    prefix_len: int,
    fill_len: int,
) -> Req:
    req = Req.__new__(Req)
    req.rid = "test-req"
    req.origin_input_ids = array("q", fill_ids)
    req.output_ids = array("q")
    req.full_untruncated_fill_ids = array("q", fill_ids)
    req.prefix_len = prefix_len
    req.extend_end = fill_len
    req.host_hit_length = 0
    req.kv = ReqKvInfo(req_pool_idx=req_pool_idx)
    req.skip_radix_cache_insert = False
    req.finished_reason = None
    req.last_node = None
    req.lock = None
    req.session = None
    req.return_logprob = False
    req.logprob_start_len = -1
    req.positional_embed_overrides = None
    req.extra_key = None
    req.cache_salt = None
    req.kv.mamba_pool_idx = None
    req.sampling_params = SimpleNamespace(max_new_tokens=128, ignore_eos=False)
    return req


def _make_tree_cache() -> BasePrefixCache:
    # The gate under test lives in the scheduler; the cache is a stub.
    return MagicMock(spec=BasePrefixCache)


def _scheduler_for_get_next_batch(*, tree_cache, chunked_req) -> Scheduler:
    s = partial_scheduler()
    s.parts.mode_owners = ModeOwners()
    s.parts.lists = CoreLists()
    s.scheduler_stage_metrics = None
    s.parts.scripted_scheduler_hook = None
    s.mode.disaggregation_mode = DisaggregationMode.NULL
    s.parts.mode_owners.disagg_prefill = None
    s.parts.mode_owners.dllm = None
    s.enable_hisparse = False
    s.parts.metrics_sink = SimpleNamespace(enable_fpm=False)
    # Exercise the unconditional scheduler-loop HiCache event-drain point.
    s.enable_hierarchical_cache = True
    s.enable_unified_cache_external_linker = False
    s.parts.require_mlp_sync = False
    s.parts.spec_algorithm = MagicMock()
    s.parts.server_args = MagicMock(speculative_skip_dp_mlp_sync=True)
    s.parts.lists.decoding = []
    s.hub.step_planner = MagicMock(spec=StepPlanner)
    s.hub.request_intake = MagicMock()
    s.hub.step_planner.should_defer_prefill.return_value = False
    s.hub.step_planner.get_new_batch_prefill.return_value = None
    s.hub.step_planner.get_decode_step.return_value = None
    s.parts.dp_attn_adapter = MagicMock()
    s.parts.dp_attn_adapter.maybe_prepare_mlp_sync_batch = MagicMock(
        side_effect=lambda batch, **_: batch
    )
    s.parts.ngram_embedding_manager = MagicMock()
    s.parts.ngram_embedding_manager.prepare_for_forward = MagicMock(
        side_effect=lambda batch, **_: batch
    )
    tree_cache.check_hicache_events = MagicMock()
    s.parts.kv.tree_cache = tree_cache
    s.parts.lists.chunked_req = chunked_req
    s.parts.lists._pending_chunked_abort_req = None
    s.parts.mode_owners.disagg_decode = None
    s.parts.mode_owners.hisparse_coordinator = None
    s.parts.batch_result_processor = MagicMock()
    s.mode.stages = init_mode_stages(s.parts, s.loop_schedules(), lora_ids_in_use=set)
    return s


class TestStashGatePreservesPrefix(CustomTestCase):
    """The stash gate advances prefix_len iff `fill_len > prefix_len`, i.e. the
    chunk computed new KV; a parked chunk must be left untouched."""

    POOL_IDX = 4
    INITIAL_PREFIX_LEN = 8  # what was really cached last iter
    POST_RESET_FILL_LEN = 32  # length after init_next_round_input rebuilds

    def setUp(self):
        enter_scope(self, published_topology())

    def _build(self, *, fill_len: int):
        cache = _make_tree_cache()
        req = _make_req(
            req_pool_idx=self.POOL_IDX,
            fill_ids=list(range(self.POST_RESET_FILL_LEN)),
            prefix_len=self.INITIAL_PREFIX_LEN,
            fill_len=fill_len,
        )
        s = _scheduler_for_get_next_batch(tree_cache=cache, chunked_req=req)
        return s, req

    def test_parked_chunked_req_keeps_its_prefix(self):
        # A parked chunk has fill_len == prefix_len: no new KV was computed,
        # so the gate must skip stash and leave the prefix intact.
        s, req = self._build(fill_len=self.INITIAL_PREFIX_LEN)

        Scheduler.schedule_step(s, ScheduleSlot())

        self.assertEqual(req.prefix_len, self.INITIAL_PREFIX_LEN)
        s.parts.kv.tree_cache.checkpoint.assert_not_called()

    def test_scheduled_chunked_req_advances_prefix_via_real_stash(self):
        # Symmetric guard against over-gating: when fill_len has advanced past
        # the cached prefix, stash must run and advance prefix_len.
        s, req = self._build(fill_len=self.POST_RESET_FILL_LEN)

        Scheduler.schedule_step(s, ScheduleSlot())

        self.assertEqual(req.prefix_len, self.POST_RESET_FILL_LEN)
        s.parts.kv.tree_cache.checkpoint.assert_called_once_with(
            req, up_to=self.POST_RESET_FILL_LEN
        )


if __name__ == "__main__":
    unittest.main()
