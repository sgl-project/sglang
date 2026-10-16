"""Priority -> admission -> real Host lease/ticket/ACK regression, without GPU.

The tree walk and target allocator are controlled; ticket reconciliation,
publication, host generations/readers and both request match entrypoints are
real. Synthetic 64-byte pages preserve the failing 4096/8191-page counts.
"""

import sys

# The shared harness bootstraps CPU-only imports before serving imports.
# ruff: noqa: E402
from array import array
from collections import defaultdict
from concurrent.futures import Future
from contextlib import nullcontext
from enum import Enum, auto
from types import SimpleNamespace as NS
from unittest.mock import Mock, patch

import pytest
import torch
from test_compressed_hicache_queued_cancel import (
    CacheHarness,
    HiCacheAck,
    source_definition,
)

from sglang.srt.kv_compression.store import CompressedHostKVCache
from sglang.srt.kv_compression.types import EncodedPage, new_page_refs
from sglang.srt.mem_cache.l2_completion import RestoreTransferResult, TransferCompletion
from sglang.srt.mem_cache.unified_cache.component_type import ComponentType
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=8, suite="base-a-test-cpu")

RadixKey = source_definition("mem_cache/radix_cache.py", "RadixKey", array=array)
ENV = NS(SGLANG_RADIX_FORCE_MISS=NS(get=lambda: False))
MATCH = dict(RadixKey=RadixKey, MatchPrefixParams=NS, envs=ENV)
radix_key = source_definition(
    "mem_cache/common.py", "_req_radix_key", RadixKey=RadixKey
)
match_kv_cache = source_definition(
    "mem_cache/common.py", "match_kv_cache", _req_radix_key=radix_key, **MATCH
)
MATCH["match_kv_cache"] = match_kv_cache


def match_request(cache, req, token_ids=None, **kwargs):
    return match_kv_cache(
        req, cache, token_ids, cow_mamba=kwargs.get("cow_mamba", False)
    )


CacheAwarePolicy = source_definition(
    "managers/schedule_policy.py", "CacheAwarePolicy", Enum=Enum
)
CacheAgnosticPolicy = source_definition(
    "managers/schedule_policy.py", "CacheAgnosticPolicy", Enum=Enum
)
AddReqResult = source_definition(
    "managers/schedule_policy.py", "AddReqResult", Enum=Enum, auto=auto
)


class Request:
    needs_host_load_back = source_definition(
        "managers/schedule_batch.py", "needs_host_load_back", "Req"
    )
    _compute_max_prefix_len = source_definition(
        "managers/schedule_batch.py", "_compute_max_prefix_len", "Req"
    )
    init_next_round_input = source_definition(
        "managers/schedule_batch.py", "init_next_round_input", "Req", **MATCH
    )

    def __init__(self, length, *, output=0, logprob=-1):
        self.rid = "host-hit"
        self.origin_input_ids = array("q", range(length))
        self.output_ids = array("q", range(length, length + output))
        self.extra_key = "adapter"
        self.cache_salt = "salt"
        self.return_logprob = logprob >= 0
        self.logprob_start_len = logprob
        self.session = self.positional_embed_overrides = self.multimodal_inputs = None
        self.is_retracted = False
        self.finished_reason = self.to_finish = None
        self.kv = NS(cache_protected_len=0)
        self.sampling_params = NS(ignore_eos=False, max_new_tokens=1)
        self.host_loaded_length = 0
        self.full_untruncated_fill_ids = self.origin_input_ids + self.output_ids
        self.prefix_len = 0
        self.dllm_config = None
        self.retracted_stain = False
        self.set_finish_with_abort = Mock()

    def is_dllm(self):
        return False

    def refresh_fill_ids(self):
        self.full_untruncated_fill_ids = self.origin_input_ids + self.output_ids


class Policy:
    calc_priority = source_definition(
        "managers/schedule_policy.py",
        "calc_priority",
        "SchedulePolicy",
        CacheAwarePolicy=CacheAwarePolicy,
        CacheAgnosticPolicy=CacheAgnosticPolicy,
        get_disagg=lambda: NS(disaggregation_mode="prefill"),
    )
    _compute_prefix_matches = source_definition(
        "managers/schedule_policy.py",
        "_compute_prefix_matches",
        "SchedulePolicy",
        match_kv_cache=match_kv_cache,
        IN_BATCH_PREFIX_CACHING_CHECK_THRESHOLD=-1,
    )
    _sort_by_longest_prefix = staticmethod(
        source_definition(
            "managers/schedule_policy.py",
            "_sort_by_longest_prefix",
            "SchedulePolicy",
        )
    )

    def __init__(self, cache):
        self.tree_cache = cache
        self.policy = CacheAwarePolicy.LPM
        self.waiting_queue_radix_tree = NS(reset=Mock())
        self._determine_active_policy = lambda waiting, *args: self.policy
        self._touch_waiting_prefixes = lambda *args: None


# calc_priority references its class for the unchanged static sort.
Policy.calc_priority.__globals__["SchedulePolicy"] = Policy


class Adder:
    add_one_req = source_definition(
        "managers/schedule_policy.py",
        "add_one_req",
        "PrefillAdder",
        AddReqResult=AddReqResult,
        torch=torch,
        CLIP_MAX_NEW_TOKENS=4096,
        InitLoadBackParams=lambda **kw: NS(mem_quota=None, **kw),
        UnifiedRadixCache=CacheHarness,
        nullcontext=nullcontext,
        ComponentType=ComponentType,
        CacheTransferPhase=NS(LOAD_BACK="load"),
    )

    def __init__(self, cache, count):
        self.tree_cache = cache
        self.page_size = 1
        self.prefill_max_requests = self.prefill_delayer_single_pass = None
        self._mamba_gap_budget_for_req = lambda req, **kw: 0
        self._lock_node = lambda node, **kwargs: nullcontext()
        self._select_prefill_admission = lambda *a, **kw: NS(
            prefix_len=count, extend_len=1, max_new_tokens=1, is_chunked=False
        )
        self._commit_prefill_admission = Mock()
        self.budget_state = lambda: AddReqResult.CONTINUE
        self.rem_chunk_tokens = None
        self.rem_mamba_slots = None
        self.per_req_token_overhead = 0
        self._request_slot_reserve_for_req = lambda req: 0
        self._kv_shard_reserve_scratch = lambda **kw: True
        self.memory_budget = NS(
            load_back_needs_full_tokens=True,
            prepare_load_back=Mock(return_value=True),
        )


def result(count, anchor):
    return NS(
        device_prefix_len=0,
        last_device_node=0,
        last_host_node=anchor,
        best_match_node=anchor,
        host_hit_length=count,
        swa_host_hit_length=0,
        mamba_host_hit_length=0,
        swa_branching_seqlen=None,
        mamba_branching_seqlen=None,
        cache_protected_len=None,
    )


@pytest.mark.parametrize(
    "length,output,logprob,tail,expected",
    [
        (1, 0, -1, 0, 0),
        (32, 0, -1, 0, 31),
        (4097, 0, -1, 0, 4096),
        (8192, 0, -1, 0, 8191),
        (32, 3, -1, 0, 34),
        (32, 0, 7, 0, 7),
        (32, 0, 7, 30, 2),
        (32, 0, -1, 40, 0),
    ],
)
def test_priority_and_admission_use_same_match_span(
    length, output, logprob, tail, expected
):
    r = Request(length, output=output, logprob=logprob)
    seen = []

    def match(params):
        seen.append(params)
        return result(len(params.key), 1)

    c = NS(
        match_prefix=match,
        swa_reprefill_tail_tokens=lambda: tail,
        supports_mamba=lambda: False,
    )
    match_request(c, r, include_req=True)
    r.init_next_round_input(c)
    assert [len(p.key) for p in seen] == [expected, expected]
    assert all(
        p.req is r and p.key.extra_key == "adapter" and p.key.cache_salt == "salt"
        for p in seen
    )


@pytest.fixture
def restored_prefix():
    def build(length):
        count = length - 1
        req = Request(length)
        pool = CompressedHostKVCache(64, 128 * 1024**2, pin_memory=False)
        handles = pool.alloc(count)
        assert handles is not None
        refs = new_page_refs(count)
        stage = torch.zeros((min(count, 64), pool.reservation_bytes), dtype=torch.uint8)
        for start in range(0, count, 64):
            end = min(start + 64, count)
            write = pool.prepare_write(handles[start:end], [64] * (end - start))
            try:
                write.scatter(stage[: end - start])
                write.publish(
                    refs[start:end],
                    [
                        EncodedPage(stage[i - start, :64], "raw", 64)
                        for i in range(start, end)
                    ],
                    [None] * (end - start),
                )
            finally:
                write.close()
        future = Future()
        completion = TransferCompletion(future=future)
        state = NS(
            restore_ticket=None,
            quarantined=[],
            stats=defaultdict(int),
            pool=pool,
            runtime=NS(verify=False),
        )
        c = CacheHarness()
        c.cache_controller = NS(
            async_l2=state,
            ack_load_queue=[],
            load_fence_stream=None,
            l2_transfer_engine=NS(submit_async_restore=Mock(return_value=completion)),
            pp_prefetch_command_group=None,
        )
        c.tree_core = NS(
            node_by_id=lambda n: NS(
                key=range(count),
                component_data={
                    ComponentType.FULL: NS(
                        host_value=handles, metadata={"compression_page_refs": refs}
                    )
                },
            ),
            build_load_back_spec=Mock(
                return_value=(NS(host_indices=handles, nodes_to_load=[1]), {})
            ),
            commit_load_back=Mock(return_value=[]),
            finish_load_back=Mock(),
        )
        c.tree_core.build_hicache_transfers = Mock(
            return_value=[NS(host_indices=handles)]
        )
        c.linker = c.buffer_pipeline = None
        c.load_back_threshold = 1
        c.inc_host_lock_ref = Mock(return_value=NS(to_dec_params=lambda: "host"))
        c.inc_lock_ref = Mock(return_value=NS(delta=0, to_dec_params=lambda: "device"))
        c.dec_host_lock_ref = Mock()
        c.dec_lock_ref = Mock()
        c._apply_cache_actions = Mock()
        c.token_to_kv_pool_allocator = NS(
            available_size=lambda: 12000,
            alloc=Mock(return_value=torch.arange(count) * 2 + 3),
            free=Mock(),
        )
        c.swa_reprefill_tail_tokens = lambda: 0
        c.supports_mamba = lambda: False
        c.init_load_back = c.init_async_load_back
        c.load_back_is_cache_owned = lambda: True
        c.matches = []
        c.anchor = 1

        def match(params):
            # The two actual key spans choose distinct controlled tree anchors,
            # mirroring the live full-key vs capped-key split boundary.
            span = len(params.key)
            answer = result(span, c.anchor if span == count else 2)
            c.matches.append(params)
            c.reconcile_restore_ticket(params, answer)
            return answer

        c.match_prefix = match
        return req, c, state, pool, future, completion, count

    return build


@pytest.mark.parametrize("length", [4097, 8192])
def test_lpm_polling_preserves_ticket_then_adder_consumes_after_ack(
    restored_prefix, length
):
    r, c, state, pool, future, completion, count = restored_prefix(length)
    p = Policy(c)
    a = Adder(c, count)
    with patch.dict(
        sys.modules,
        {
            "sglang.srt.managers.cache_controller": NS(HiCacheAck=HiCacheAck),
            "sglang.srt.mem_cache.base_prefix_cache": NS(EvictParams=NS),
        },
    ):
        p.calc_priority([r])
        r.init_next_round_input(c)
        assert a.add_one_req(r, False, None) == AddReqResult.OTHER
        ticket = state.restore_ticket
        assert ticket is not None
        assert pool.snapshot()["active_readers"] == count
        for _ in range(3):
            p.calc_priority([r])
            r.init_next_round_input(c)
            assert not ticket.abandoned
            assert a.add_one_req(r, False, None) == AddReqResult.OTHER
            assert state.restore_ticket is ticket
        c.token_to_kv_pool_allocator.alloc.assert_called_once()
        c.token_to_kv_pool_allocator.free.assert_not_called()
        c.tree_core.commit_load_back.assert_not_called()
        future.set_result(RestoreTransferResult(count, count * 64, count * 64, 0, 0))
        c.loading_check(finish_count=1)
        p.calc_priority([r])
        r.init_next_round_input(c)
        assert not ticket.abandoned and ticket.ready
        assert c.get_restore_reserved_tokens(r) == count
        assert a.add_one_req(r, False, None) == AddReqResult.CONTINUE
        assert state.restore_ticket is None and ticket.consumed
        assert r.host_loaded_length == count
        assert a.memory_budget.prepare_load_back.call_args.kwargs["full_tokens"] == 3
        assert (
            a.memory_budget.prepare_load_back.call_args_list[0].kwargs["full_tokens"]
            == count + 3
        )
        assert pool.snapshot()["active_readers"] == 0
        c.token_to_kv_pool_allocator.free.assert_not_called()
        c.tree_core.commit_load_back.assert_called_once()
        assert (
            c.tree_core.commit_load_back.call_args.args[1].tolist()
            == (torch.arange(count) * 2 + 3).tolist()
        )
        a._commit_prefill_admission.assert_called_once()


def test_genuinely_changed_anchor_still_waits_for_ack_before_release(restored_prefix):
    r, c, state, pool, future, completion, count = restored_prefix(32)
    r.init_next_round_input(c)
    with patch.dict(
        sys.modules,
        {
            "sglang.srt.managers.cache_controller": NS(HiCacheAck=HiCacheAck),
            "sglang.srt.mem_cache.base_prefix_cache": NS(EvictParams=NS),
        },
    ):
        assert (
            c.init_async_load_back(NS(req=r, best_match_node=1, mem_quota=None)) is None
        )
        ticket = state.restore_ticket
        other = Request(32)
        c.reconcile_restore_ticket(NS(req=other), result(31, 2))
        assert not ticket.abandoned
        c.anchor = 2
        Policy(c).calc_priority([r])
        assert ticket.abandoned and state.restore_ticket is ticket
        assert pool.snapshot()["active_readers"] == count
        c.token_to_kv_pool_allocator.free.assert_not_called()
        future.set_result(RestoreTransferResult(count, count * 64, count * 64, 0, 0))
        c.loading_check(finish_count=1)
        assert state.restore_ticket is None
        assert pool.snapshot()["active_readers"] == 0
        c.token_to_kv_pool_allocator.free.assert_called_once_with(ticket.device_indices)
        c.tree_core.commit_load_back.assert_not_called()


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
