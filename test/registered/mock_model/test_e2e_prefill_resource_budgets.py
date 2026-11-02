"""Two-GPU coverage for rank consensus and FCFS mixed-prefill admission.

Only the memory observation in the consensus worker is injected. CUDA device
selection, NCCL progress, Gloo MIN reduction and scheduler decisions are real.
The serving cases use the existing dummy-weight/token-oracle/canary fixture;
they check scheduling and KV integrity, not pretrained-model accuracy.
"""

from __future__ import annotations

import unittest
from datetime import timedelta
from types import SimpleNamespace
from unittest.mock import patch

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from sglang.srt.environ import envs
from sglang.srt.kv_canary.token_oracle.oracle import HashOracle
from sglang.srt.managers.schedule_batch import FINISH_LENGTH
from sglang.srt.managers.scheduler import Scheduler
from sglang.srt.model_executor.forward_batch_info import (
    ForwardMode,
    _stable_hash_str_to_i64,
)
from sglang.srt.utils.network import get_open_port
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.mock_model.utils import MOCK_MODEL_PATH
from sglang.test.scripted_runtime.context import ScriptedContext
from sglang.test.scripted_runtime.test_case import ScriptedTestCase
from sglang.test.scripted_runtime_chunked_helpers import (
    DEFAULT_MAX_STEPS,
    advance_to_decode_step,
    advance_to_nth_chunk,
    run_until_finished,
    warmup_radix,
)
from sglang.test.test_utils import CustomTestCase, published_topology

register_cuda_ci(est_time=180, stage="extra-a", runner_config="2-gpu-large")


def _batch(tokens: int, *, prefill: bool = True):
    return SimpleNamespace(
        extend_num_tokens=tokens,
        forward_mode=ForwardMode.EXTEND if prefill else ForwardMode.DECODE,
        spec_algorithm=SimpleNamespace(is_none=lambda: True),
        grammar_needs_sync=lambda: False,
    )


def _consensus_worker(rank: int, port: int) -> None:
    torch.cuda.set_device(rank)
    dist.init_process_group(
        "nccl",
        init_method=f"tcp://127.0.0.1:{port}",
        rank=rank,
        world_size=2,
        timeout=timedelta(seconds=60),
    )
    cpu_group = dist.new_group(backend="gloo", timeout=timedelta(seconds=60))
    try:
        with published_topology(
            device="cuda", tp_size=2, ranks=dict(world_rank=rank, gpu_id=rank)
        ):
            scheduler = Scheduler.__new__(Scheduler)
            scheduler.require_mlp_sync = False
            scheduler.result_queue = []
            scheduler.device = "cuda"
            scheduler.tp_cpu_group = cpu_group
            scheduler.chunked_prefill_size = 64
            scheduler.waiting_queue = [object()]
            scheduler.chunked_req = None
            scheduler.dllm_config = None
            scheduler.dynamic_chunk_sizer = None
            scheduler.grammar_manager = SimpleNamespace(
                has_waiting_grammars=lambda: False
            )

            # The same ranks must still make progress on their device group
            # after every overlap decision, including asymmetric observations.
            def check(expected, previous):
                decision = scheduler.should_process_last_batch_early(previous)
                witness = torch.tensor([int(decision)], device=f"cuda:{rank}")
                dist.all_reduce(witness)
                assert decision == expected, (rank, decision, expected)
                assert witness.item() == 2 * int(expected), witness

            total = torch.cuda.get_device_properties(rank).total_memory
            for local_gb in ((4.0, 1.0), (1.0, 4.0), (2.0, 4.0)):
                with (
                    envs.SGLANG_DISABLE_OVERLAP_AFTER_PREFILL.override(False),
                    envs.SGLANG_OVERLAP_PREFILL_TOKEN_BUDGET.override(0),
                    envs.SGLANG_PREFILL_OVERLAP_MIN_FREE_GB.override(2.0),
                    patch(
                        "torch.cuda.mem_get_info",
                        return_value=(int(local_gb[rank] * (1 << 30)), total),
                    ) as observation,
                    patch(
                        "torch.cuda.empty_cache",
                        side_effect=AssertionError(
                            "the scheduler must not empty caches"
                        ),
                    ),
                ):
                    check(min(local_gb) < 2.0, _batch(16))
                    assert observation.call_count == 1

                    # A decode boundary must not enter the memory collective.
                    observation.reset_mock()
                    check(False, _batch(1, prefill=False))
                    check(False, None)
                    scheduler.waiting_queue = []
                    check(False, _batch(16))
                    scheduler.waiting_queue = [object()]
                    observation.assert_not_called()

            with (
                envs.SGLANG_DISABLE_OVERLAP_AFTER_PREFILL.override(False),
                envs.SGLANG_PREFILL_OVERLAP_MIN_FREE_GB.override(0),
                patch(
                    "torch.cuda.mem_get_info",
                    side_effect=AssertionError("disabled memory budget must not probe"),
                ),
            ):
                for budget, previous, expected in (
                    (0, 80, False),
                    (128, 64, False),
                    (128, 65, True),
                    (96, 32, False),
                    (95, 32, True),
                ):
                    with envs.SGLANG_OVERLAP_PREFILL_TOKEN_BUDGET.override(budget):
                        check(expected, _batch(previous))
    finally:
        dist.destroy_process_group(cpu_group)
        dist.destroy_process_group()


@unittest.skipUnless(torch.cuda.device_count() >= 2, "requires two CUDA GPUs")
class TestPrefillResourceBudgetConsensus(CustomTestCase):
    def test_asymmetric_memory_and_token_budget_agree_across_tp_ranks(self):
        mp.spawn(_consensus_worker, args=(get_open_port(),), nprocs=2, join=True)


class _PrefillResourceTP:
    RESERVE_TOKENS = 0
    ENGINE_KWARGS = dict(
        model_path=MOCK_MODEL_PATH,
        load_format="dummy",
        sampling_backend="token_oracle",
        tp_size=2,
        dtype="float16",
        attention_backend="triton",
        cuda_graph_backend_decode="disabled",
        mem_fraction_static=0.25,
        max_total_tokens=4096,
        max_running_requests=8,
        context_length=1024,
        chunked_prefill_size=64,
        max_prefill_tokens=1024,
        enable_mixed_chunk=True,
        schedule_policy="fcfs",
    )

    @classmethod
    def setUpClass(cls):
        with (
            envs.SGLANG_KV_CANARY_ENABLE_TOKEN_ORACLE.override(True),
            envs.SGLANG_KV_CANARY_ENABLE_WRITE_INPUT_ASSERT.override(True),
            envs.SGLANG_ENABLE_STRICT_MEM_CHECK_DURING_BUSY.override(True),
            envs.SGLANG_DISABLE_OVERLAP_AFTER_PREFILL.override(False),
            envs.SGLANG_OVERLAP_PREFILL_TOKEN_BUDGET.override(96),
            envs.SGLANG_PREFILL_OVERLAP_MIN_FREE_GB.override(0),
            envs.SGLANG_FCFS_WARM_PREFILL_RESERVE_TOKENS.override(cls.RESERVE_TOKENS),
        ):
            super().setUpClass()


class _MixedPrefillTP(_PrefillResourceTP):
    def test_warm_head_can_share_the_continuation_budget(self):
        self.server.execute_script(
            _script_mixed_fcfs, args=(self.RESERVE_TOKENS > 0, False)
        )

    def test_cold_head_keeps_fcfs_admission_order(self):
        self.server.execute_script(
            _script_mixed_fcfs, args=(self.RESERVE_TOKENS > 0, True)
        )


@unittest.skipUnless(torch.cuda.device_count() >= 2, "requires two CUDA GPUs")
class TestMixedPrefillTPDefault(_MixedPrefillTP, ScriptedTestCase):
    pass


@unittest.skipUnless(torch.cuda.device_count() >= 2, "requires two CUDA GPUs")
class TestMixedPrefillTPWarmReservation(_MixedPrefillTP, ScriptedTestCase):
    RESERVE_TOKENS = 32


@unittest.skipUnless(torch.cuda.device_count() >= 2, "requires two CUDA GPUs")
class TestPrefillTPInputBudget(_PrefillResourceTP, ScriptedTestCase):
    RESERVE_TOKENS = 32
    ENGINE_KWARGS = dict(
        _PrefillResourceTP.ENGINE_KWARGS,
        max_prefill_tokens=32,
        # A first cold chunk still uses the native chunk budget. Keep the
        # canary's write-entry capacity large enough for that existing path.
        max_running_requests=64,
        page_size=1,
        enable_mixed_chunk=False,
    )

    def test_warm_reservation_uses_the_smaller_input_budget(self):
        self.server.execute_script(_script_smaller_input_budget)


def _script_smaller_input_budget(t: ScriptedContext):
    assert t.scheduler.max_prefill_tokens == 32
    assert t.scheduler.chunked_prefill_size == 64
    yield from warmup_radix(t, [11] * 16)
    continuation = t.start_req(
        prompt_len=256, prompt_token=7, max_new_tokens=8, ignore_eos=True
    )
    yield from advance_to_nth_chunk(continuation, 1)
    warm = t.start_req(
        prompt_len=32, prompt_token=11, max_new_tokens=8, ignore_eos=True
    )

    hook = t._scheduler_hook
    original = hook.on_run_batch
    original_decision = t.scheduler.should_process_last_batch_early
    original_plan = t.scheduler.get_next_batch_to_run
    original_process = t.scheduler.process_batch_result
    batches = []
    decisions = []
    events = []

    def decide(last_batch):
        early = original_decision(last_batch)
        if last_batch is not None:
            prefill = last_batch.forward_mode.is_extend()
            possible_prefill = bool(
                t.scheduler.waiting_queue
                or t.scheduler.chunked_req is not None
                or t.scheduler.grammar_manager.has_waiting_grammars()
            )
            token_bound = (
                last_batch.extend_num_tokens or 0
            ) + t.scheduler.chunked_prefill_size
            decisions.append((prefill, possible_prefill, token_bound, bool(early)))
            events.append(("decision", bool(early)))
        return early

    def plan(*args, **kwargs):
        events.append(("plan",))
        return original_plan(*args, **kwargs)

    def process(batch, result):
        events.append(("process",))
        return original_process(batch, result)

    def capture(batch):
        events.append(("run",))
        if batch.forward_mode.is_extend():
            batches.append(
                (
                    batch.extend_num_tokens,
                    dict(
                        zip(
                            (req.rid for req in batch.reqs),
                            batch.extend_lens,
                            strict=True,
                        )
                    ),
                    t.scheduler.chunked_req.rid
                    if t.scheduler.chunked_req is not None
                    else None,
                )
            )
        original(batch)

    with (
        patch.object(hook, "on_run_batch", side_effect=capture),
        patch.object(
            t.scheduler, "should_process_last_batch_early", side_effect=decide
        ),
        patch.object(t.scheduler, "get_next_batch_to_run", side_effect=plan),
        patch.object(t.scheduler, "process_batch_result", side_effect=process),
    ):
        for _ in range(DEFAULT_MAX_STEPS):
            if batches:
                break
            yield
        else:
            raise AssertionError("no continuation batch was scheduled")

        # Check the actual forward batch while the warm head is waiting.
        # Removing the clamp gives the continuation 48 tokens and leaves no
        # input for the warm head.
        tokens, extend_lens, chunked_rid = batches[0]
        assert tokens == 32, batches
        assert extend_lens == {continuation.rid: 16, warm.rid: 16}, batches
        assert chunked_rid == continuation.rid, batches
        assert warm.req.num_matched_prefix_tokens == 16

        yield from run_until_finished(continuation)
        yield from run_until_finished(warm)

    # The real loop compares the previous prefill with the next step's fixed
    # chunk upper bound before planning. Equality stays overlapped; a shorter
    # actual next batch can drain conservatively. Decode overlap stays intact.
    assert (True, True, 96, False) in decisions, decisions
    assert any(
        prefill and possible and bound > 96 and early
        for prefill, possible, bound, early in decisions
    ), decisions
    assert any(not prefill and not early for prefill, _, _, early in decisions), (
        decisions
    )
    decision_indices = [i for i, event in enumerate(events) if event[0] == "decision"]
    for offset, index in enumerate(decision_indices):
        end = (
            decision_indices[offset + 1]
            if offset + 1 < len(decision_indices)
            else len(events)
        )
        step = tuple(event[0] for event in events[index + 1 : end])
        assert step.count("plan") == 1 and step.count("process") == 1, events
        if events[index][1]:
            assert step[:2] == ("process", "plan"), events
        else:
            assert step[0] == "plan", events
        if "run" in step:
            assert step.index("plan") < step.index("run"), events
            assert (step.index("process") < step.index("run")) == events[index][1], (
                events
            )

    for _ in range(8):
        if t.scheduler.is_fully_idle():
            break
        yield
    assert t.scheduler.is_fully_idle()
    t.scheduler.invariant_checker.self_check_during_busy()


def _script_mixed_fcfs(t: ScriptedContext, reserve_enabled: bool, cold_first: bool):
    # This is a genuine cache hit in the same running TP engine.
    yield from warmup_radix(t, [11] * 96)
    ballast = t.start_req(
        prompt_len=8, prompt_token=13, max_new_tokens=32, ignore_eos=True
    )
    yield from advance_to_decode_step(ballast, 1)
    held = {ballast.rid: ballast.req}

    continuation = t.start_req(
        prompt_len=256, prompt_token=7, max_new_tokens=8, ignore_eos=True
    )
    yield from advance_to_nth_chunk(continuation, 1)
    held[continuation.rid] = continuation.req
    offset = len(t._scheduler_hook._batch_log)

    # Both requests arrive before yielding the next scheduler iteration.
    waiters = []
    for cold in (True, False) if cold_first else (False, True):
        waiters.append(
            t.start_req(
                prompt_len=120 if cold else 112,
                prompt_token=12 if cold else 11,
                max_new_tokens=8,
                ignore_eos=True,
            )
        )
    cold, warm = waiters if cold_first else reversed(waiters)
    handles = [ballast, continuation, *waiters]
    for _ in range(DEFAULT_MAX_STEPS):
        for handle in handles:
            req = handle.req
            if req is not None:
                held[handle.rid] = req
        if all(handle.finished for handle in handles):
            break
        yield
    else:
        raise AssertionError("mixed-prefill requests did not finish")

    log = t._scheduler_hook._batch_log[offset:]
    first = {}
    for index, record in enumerate(log):
        for rid in record.rids:
            first.setdefault(rid, index)
    assert all(h.rid in first for h in waiters), log
    assert first[waiters[0].rid] <= first[waiters[1].rid], log

    warm_record = log[first[warm.rid]]
    if reserve_enabled and not cold_first:
        assert warm_record.chunked_rid == continuation.rid, log
        assert warm.rid in warm_record.extend_rids, log
        assert held[warm.rid].num_matched_prefix_tokens > 0
        assert first[warm.rid] < first[cold.rid], log
    elif not cold_first:
        # With the default zero reservation, the continuation consumes its
        # native chunks and the waiter first enters after its final chunk.
        assert warm_record.chunked_rid != continuation.rid, log

    assert any(
        record.mode == "mixed"
        and ballast.rid in record.rids
        and continuation.rid in record.rids
        for record in log
    ), log

    oracle = HashOracle(vocab_size=t.scheduler.model_config.vocab_size)
    for handle in handles:
        req = held[handle.rid]
        assert isinstance(req.finished_reason, FINISH_LENGTH), req.finished_reason
        expected = oracle.expected_tokens(
            generalized_req_ids=torch.full(
                (req.sampling_params.max_new_tokens,),
                _stable_hash_str_to_i64(handle.rid),
                dtype=torch.int64,
            ),
            positions=torch.arange(
                len(req.origin_input_ids),
                len(req.origin_input_ids) + req.sampling_params.max_new_tokens,
                dtype=torch.int64,
            ),
        )
        assert list(req.output_ids) == expected.tolist(), handle.rid

    # Drain the final overlap result and run the existing allocator invariant
    # check before another script reuses this engine.
    yield from run_until_finished(ballast)
    for _ in range(8):
        if t.scheduler.is_fully_idle():
            break
        yield
    assert t.scheduler.is_fully_idle()
    t.scheduler.invariant_checker.self_check_during_busy()


if __name__ == "__main__":
    unittest.main()
