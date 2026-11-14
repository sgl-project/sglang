"""CPU regressions through the real EAGLE V2 orchestration; GPU work is mocked."""

import contextlib
import copy
import inspect
import json
import pickle
import time
from dataclasses import FrozenInstanceError
from types import SimpleNamespace as NS
from unittest.mock import MagicMock, patch

import pytest
import torch

from sglang.srt.managers.utils import GenerationBatchResult
from sglang.srt.speculative.eagle_worker_v2 import EAGLEWorkerV2
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


def run_round(steps=1, device="cuda"):
    """Real forward_batch_generation; replace only draft/verify GPU boundaries."""
    worker = object.__new__(EAGLEWorkerV2)
    worker.device = device  # Select scalar CUDA telemetry, never allocate CUDA.
    worker.speculative_num_steps = steps
    worker.speculative_num_draft_tokens = steps + 1
    worker.topk = 1
    worker.adaptive_controller = NS()
    worker._post_verify_graphs = None
    worker._wait_for_fused_hicache_load = lambda batch: None
    worker.activate_step_by_batch = lambda size: None
    verify_input = NS(is_verify_input=lambda: True, iteration_ids=None)
    worker._build_trivial_verify_input = lambda batch: verify_input
    worker._stub_skipped_draft_extend = lambda batch, result: None

    def draft(batch, **kwargs):
        if execution := kwargs.get("execution"):
            execution.record("draft", graph=True, num_forwards=max(0, steps - 1))
        return verify_input

    def verify(batch, **kwargs):
        if execution := kwargs.get("execution"):
            execution.verify_num_tokens_per_req = steps + 1
            execution.record("target_verify", graph=True)
        return GenerationBatchResult(
            can_run_cuda_graph=True, speculative_num_draft_tokens=steps + 1
        )

    def extend(batch, result, **kwargs):
        if execution := kwargs.get("execution"):
            execution.record("draft_extend", graph=False)

    worker._draft_worker = NS(
        draft=draft, _draft_extend_for_decode=extend, parallel_layout=None
    )
    worker.verify = verify
    batch = NS(
        forward_mode=NS(is_extend=lambda: False, is_idle=lambda: False),
        is_extend_in_batch=False,
        seq_lens=torch.tensor([8, 8]),
        spec_info=NS(is_verify_input=lambda: False),
    )
    with (
        patch(
            "sglang.srt.speculative.eagle_worker_v2.get_parallel",
            return_value=NS(pp_size=1),
        ),
        patch(
            "sglang.srt.speculative.eagle_worker_v2.draft_tp_context",
            return_value=contextlib.nullcontext(),
        ),
        patch(
            "sglang.srt.speculative.eagle_worker_v2.speculative_moe_backend_context",
            return_value=contextlib.nullcontext(),
        ),
        patch(
            "sglang.srt.speculative.eagle_worker_v2.speculative_moe_a2a_backend_context",
            return_value=contextlib.nullcontext(),
        ),
        patch(
            "sglang.srt.speculative.eagle_worker_v2.spec_stage_span",
            side_effect=lambda *a: contextlib.nullcontext(),
        ),
        patch(
            "sglang.srt.speculative.eagle_worker_v2.envs",
            NS(SGLANG_SPEC_SKIP_ZERO_STEP_DRAFT_EXTEND=NS(get=lambda: True)),
        ),
    ):
        result = worker.forward_batch_generation(batch)
    return worker, result


@pytest.mark.parametrize("later_steps", [1, 3])
def test_completed_round_keeps_execution_tier(later_steps):
    worker, result = run_round(1)
    # Transition after execution, before scheduler/result processing.
    worker.speculative_num_steps = later_steps
    worker.speculative_num_draft_tokens = later_steps + 1
    execution = result.spec_execution
    assert execution.num_steps == 1
    assert execution.draft_num_tokens_per_req == 1
    assert execution.num_draft_tokens == execution.verify_num_tokens_per_req == 2
    assert execution.draft.graph_forwards == execution.draft.eager_forwards == 0
    assert execution.target_verify.graph_forwards == 1
    assert execution.draft_extend.eager_forwards == 1
    assert result.can_run_cuda_graph is True  # Existing target-only flag.
    with pytest.raises(FrozenInstanceError):
        execution.num_steps = 99


def test_zero_step_unexecuted_phases_are_not_eager():
    _, result = run_round(0)
    execution = result.spec_execution
    assert execution.num_steps == 0 and execution.verify_num_tokens_per_req == 1
    assert execution.draft.graph_forwards == execution.draft.eager_forwards == 0
    assert (
        execution.draft_extend.graph_forwards
        == execution.draft_extend.eager_forwards
        == 0
    )
    assert execution.target_verify.graph_forwards == 1


def test_uninstrumented_round_is_unknown_not_unexecuted():
    _, result = run_round(device="cpu")
    assert result.spec_execution is None


def test_result_defaults_and_pickle_preserve_unknown_vs_unexecuted():
    empty = GenerationBatchResult()
    assert empty.spec_execution is None
    assert empty.spec_execution_accounted is False
    _, completed = run_round(1)
    for result in (empty, completed):
        for restored in (copy.deepcopy(result), pickle.loads(pickle.dumps(result))):
            assert restored.spec_execution == result.spec_execution
            assert restored.spec_execution_accounted is False
    snapshot = pickle.loads(pickle.dumps(completed)).spec_execution
    assert snapshot is not None and snapshot.draft.graph_forwards == 0
    assert snapshot.draft.eager_forwards == 0


def test_existing_copy_to_cpu_preserves_snapshot_without_extra_copies():
    _, result = run_round(1)
    snapshot = result.spec_execution
    result.next_token_ids = torch.tensor([1, 2])
    result.copy_done = MagicMock()
    with patch(
        "sglang.srt.managers.utils._async_d2h", side_effect=lambda x: x, create=True
    ) as d2h:
        result.copy_to_cpu(return_logprob=False, return_hidden_states=False)
    assert result.spec_execution is snapshot
    # Only the pre-existing token tensor is copied; metadata stays host scalar.
    assert d2h.call_count == 1
    assert d2h.call_args.args[0] is result.next_token_ids
    result.copy_done.record.assert_called_once()


def test_multiple_model_forwards_keep_mixed_outcomes():
    from sglang.srt.speculative.execution_info import (
        SpecExecutionRecorder,
        observe_model_forward,
    )

    recorder = SpecExecutionRecorder(3, 1, 4)
    recorder.verify_num_tokens_per_req = 4
    # Same observer used at each real draft_forward ModelRunner call.
    runner = MagicMock(side_effect=[NS(can_run_graph=True), NS(can_run_graph=False)])
    observe_model_forward(recorder, "draft", runner)
    observe_model_forward(recorder, "draft", runner)
    snapshot = recorder.finish()
    assert runner.call_count == 2
    assert snapshot.draft.graph_forwards == snapshot.draft.eager_forwards == 1
    assert (
        snapshot.target_verify.graph_forwards
        == snapshot.target_verify.eager_forwards
        == 0
    )
    # A failed dispatch is not recorded as completed work.
    with pytest.raises(RuntimeError):
        observe_model_forward(recorder, "draft", MagicMock(side_effect=RuntimeError))
    assert recorder.finish() == snapshot


def test_real_draft_graph_dispatch_counts_captured_model_forwards():
    from sglang.srt.speculative.eagle_worker_v2 import EagleDraftWorker
    from sglang.srt.speculative.execution_info import SpecExecutionRecorder

    worker = object.__new__(EagleDraftWorker)
    worker.req_to_token_pool = None
    worker.topk = 1
    worker.speculative_num_steps = 3
    worker.speculative_num_draft_tokens = 4
    worker.device = "cuda"
    worker.tree_mask_mode = None
    worker.seed_dsa_topk_from_draft_extend = False
    worker.draft_runner = NS(canary_manager=None)
    worker.target_worker = NS(model_runner=NS(runner_ids=None))
    worker.cuda_graph_runner = NS(
        speculative_num_steps=3,
        execute=MagicMock(return_value=(None, None, None, None)),
    )
    worker.draft_runtime = NS(begin_draft_round=MagicMock())
    batch = NS(spec_info=NS())
    fb = NS(forward_mode=NS(is_idle=lambda: False))
    recorder = SpecExecutionRecorder(3, 1, 4)
    with (
        patch(
            "sglang.srt.speculative.eagle_worker_v2.prepare_for_draft",
            return_value=(fb, True, None),
        ),
        patch(
            "sglang.srt.speculative.eagle_worker_v2.build_eagle_verify_input",
            return_value="verify",
        ),
    ):
        assert worker.draft(batch, execution=recorder) == "verify"
    recorder.verify_num_tokens_per_req = 4
    assert recorder.finish().draft.graph_forwards == 2
    assert recorder.finish().draft.eager_forwards == 0
    worker.cuda_graph_runner.execute.assert_called_once()


def test_real_draft_loop_retains_two_different_dispatch_outcomes():
    from sglang.srt.speculative.eagle_worker_v2 import EagleDraftWorker
    from sglang.srt.speculative.execution_info import SpecExecutionRecorder

    worker = object.__new__(EagleDraftWorker)
    worker.topk = 1
    worker.speculative_num_steps = 3
    worker.hot_token_id = None
    worker.index_share_for_mtp_iteration = False
    worker.seed_dsa_topk_from_draft_extend = False
    worker._topk1_parents_prealloc = torch.zeros(2, 3)
    worker._topk1_score_indices_prealloc = torch.zeros(2, 3)
    worker.draft_attn_backend = NS(attn_backends=[None, None])
    logits = NS(next_token_logits=torch.zeros(2, 4), hidden_states=torch.zeros(2, 3))
    worker.draft_runner = NS(
        canary_manager=None,
        model_config=NS(
            hf_config=NS(architectures=["LlamaForCausalLM"]), model_is_mrope=False
        ),
        forward=MagicMock(
            side_effect=[
                NS(logits_output=logits, can_run_graph=True),
                NS(logits_output=logits, can_run_graph=False),
            ]
        ),
    )
    fb = NS(
        forward_mode=NS(is_idle=lambda: False),
        out_cache_loc=None,
        batch_size=2,
        spec_info=NS(
            topk_p=torch.ones(2, 1),
            topk_index=torch.zeros(2, 1),
            hidden_states=torch.zeros(2, 3),
        ),
        positions=torch.zeros(2),
    )
    recorder = SpecExecutionRecorder(3, 1, 4)
    with (
        patch("sglang.srt.speculative.eagle_worker_v2._is_cuda", False),
        patch("sglang.srt.speculative.eagle_worker_v2._is_hip", False),
        patch(
            "sglang.srt.speculative.eagle_worker_v2.get_spec",
            return_value=NS(speculative_use_rejection_sampling=False),
        ),
        patch(
            "sglang.srt.speculative.eagle_worker_v2.per_step_draft_out_cache_loc",
            return_value=[None, None, None],
        ),
        patch(
            "sglang.srt.speculative.eagle_worker_v2.select_top_k_tokens",
            return_value=(
                torch.zeros(2),
                torch.zeros(2, 3),
                None,
                (torch.zeros(2, 1), torch.zeros(2, 1), torch.zeros(2, 1)),
            ),
        ),
        patch(
            "sglang.srt.speculative.eagle_worker_v2.IndexTopKShareState.mtp_iteration",
            return_value=contextlib.nullcontext(),
        ),
        patch(
            "sglang.srt.speculative.eagle_worker_v2.forward_context",
            side_effect=lambda *a, **kw: contextlib.nullcontext(),
        ),
        patch("sglang.srt.speculative.eagle_worker_v2.ForwardContext"),
        patch("sglang.srt.speculative.eagle_worker_v2.maybe_detect_nan"),
        patch("sglang.srt.speculative.eagle_worker_v2.maybe_detect_inf"),
        patch("sglang.srt.speculative.eagle_worker_v2.maybe_detect_oob"),
    ):
        worker.draft_forward(fb, execution=recorder)
    recorder.verify_num_tokens_per_req = 4
    assert worker.draft_runner.forward.call_count == 2
    assert recorder.finish().draft.graph_forwards == 1
    assert recorder.finish().draft.eager_forwards == 1


@pytest.mark.parametrize("actual_graph", [False, True])
def test_target_uses_actual_dispatch_not_prepare_eligibility(actual_graph):
    from sglang.srt.speculative.eagle_worker_common import run_eagle_verify
    from sglang.srt.speculative.execution_info import SpecExecutionRecorder

    recorder = SpecExecutionRecorder(1, 1, 2)
    batch = NS(
        seq_lens=[8, 8],
        spec_info=NS(num_tokens_per_req=2),
        input_ids=None,
        out_cache_loc=None,
        has_grammar=False,
    )
    target = NS(
        forward_batch_generation=MagicMock(
            return_value=NS(logits_output=None, can_run_cuda_graph=actual_graph)
        )
    )
    # Stop after the real target forward observation, before GPU sampling.
    allocator = NS(get_kvcache=MagicMock(side_effect=StopIteration))
    # The newer provider binding adds a required runtime-hooks argument. This
    # test stops before sampling and only observes the target forward boundary.
    runtime_kwargs = (
        {"speculative_runtime_hooks": MagicMock()}
        if "speculative_runtime_hooks" in inspect.signature(run_eagle_verify).parameters
        else {}
    )
    with (
        patch(
            "sglang.srt.speculative.eagle_worker_common.torch.get_device_module",
            return_value=NS(current_stream=lambda: None),
        ),
        patch("sglang.srt.speculative.eagle_worker_common.record_stream_for_v2_verify"),
        patch("sglang.srt.speculative.eagle_worker_common.record_stream_each"),
        patch(
            "sglang.srt.speculative.eagle_worker_common.eagle_prepare_for_verify",
            return_value=(NS(), True),
        ),
        pytest.raises(StopIteration),
    ):
        run_eagle_verify(
            batch,
            target_worker=target,
            req_to_token_pool=None,
            token_to_kv_pool_allocator=allocator,
            plan_stream=None,
            plan_stream_ctx=contextlib.nullcontext(),
            topk=1,
            num_draft_tokens=2,
            device="cpu",
            metadata_ready_pre_pad=False,
            finalize_tree_path=True,
            execution=recorder,
            **runtime_kwargs,
        )
    result = recorder.finish()
    assert result.target_verify.graph_forwards == int(actual_graph)
    assert result.target_verify.eager_forwards == int(not actual_graph)
    target.forward_batch_generation.assert_called_once()


def reporter_for_window():
    from sglang.srt.managers.scheduler_components.metrics_reporter import (
        SchedulerMetricsReporter,
    )
    from sglang.srt.speculative.execution_info import SpecExecutionWindow

    r = object.__new__(SchedulerMetricsReporter)
    r.spec_execution_window = SpecExecutionWindow()
    r.is_stats_logging_rank = True
    r.current_scheduler_metrics_enabled = False
    r.forward_ct_decode = 1
    r.decode_log_interval = 2
    r.last_decode_stats_tic = time.perf_counter() - 1
    r.num_generated_tokens = 0
    r.enable_mfu_metrics = False
    r._graph_backend_label = "cuda graph"
    r.scheduler = NS(
        pool_stats_observer=NS(
            get_pool_stats=lambda: NS(get_decode_usage_msg_parts=lambda: [])
        ),
        forward_ct=1,
        spec_algorithm=NS(is_none=lambda: True),
        disaggregation_mode=None,
        waiting_queue=[],
        kv_events_publisher=MagicMock(),
    )
    return r


def test_report_boundary_counts_each_arrival_once_and_keeps_tier():
    r = reporter_for_window()
    worker, first = run_round(1)
    worker.speculative_num_steps = 3  # Current state must never be read here.
    r.observe_spec_execution(first)
    batch = NS(reqs=[None, None], forward_iter=None)
    with (
        patch(
            "sglang.srt.managers.scheduler_components.metrics_reporter.logger"
        ) as logger,
        patch(
            "sglang.srt.managers.scheduler_components.metrics_reporter.get_disagg",
            return_value=NS(language_only=False),
        ),
    ):
        r.report_decode_stats(True, running_batch=batch)  # Not a reporting tick.
        logger.info.assert_not_called()
        r.forward_ct_decode = 2
        r.report_decode_stats(True, running_batch=batch)
        summaries = [
            json.loads(c.args[1])
            for c in logger.info.call_args_list
            if c.args[0] == "Completed speculative execution: %s"
        ]
        assert len(summaries) == 1 and summaries[0][0]["rounds"] == 1
        assert summaries[0][0]["num_steps"] == 1
        r.observe_spec_execution(first)  # Even across a drain: no double count.
        _, second = run_round(3)
        r.observe_spec_execution(second)
        r.forward_ct_decode = 3
        r.report_decode_stats(True, running_batch=batch)
        r.forward_ct_decode = 4
        r.report_decode_stats(True, running_batch=batch)
        summaries = [
            json.loads(c.args[1])
            for c in logger.info.call_args_list
            if c.args[0] == "Completed speculative execution: %s"
        ]
        assert len(summaries) == 2 and summaries[1][0]["rounds"] == 1
        assert summaries[1][0]["num_steps"] == 3
        assert summaries[0][0]["phases"]["draft"]["unexecuted_rounds"] == 1
        assert summaries[0][0]["phases"]["draft_extend"]["eager_forwards"] == 1
        assert summaries[1][0]["phases"]["draft"]["graph_forwards"] == 2
        assert r.spec_execution_window.drain() == []
        assert first.spec_execution.num_steps == 1  # Snapshot remains readable.


def test_non_reporting_tp_rank_does_not_aggregate():
    r = reporter_for_window()
    r.is_stats_logging_rank = False
    _, result = run_round()
    r.observe_spec_execution(result)
    assert r.spec_execution_window.drain() == []
    assert not result.spec_execution_accounted


@pytest.mark.parametrize("instrumented", [False, True])
@pytest.mark.parametrize("live_rows", [(True, True), (True, False), (False, False)])
def test_real_result_processor_accounts_after_feedback_and_preserves_counter(
    instrumented,
    live_rows,
):
    from sglang.srt.managers.scheduler_components.batch_result_processor import (
        SchedulerBatchResultProcessor,
    )

    worker, result = run_round(1, device="cuda" if instrumented else "cpu")
    reporter = reporter_for_window()
    reporter.update_spec_metrics = MagicMock()
    processor = NS()
    processor.still_in_batch = lambda req: req.live
    processor._rows_still_in_batch = (
        SchedulerBatchResultProcessor._rows_still_in_batch.__get__(processor)
    )
    processor.metrics_reporter = reporter
    processor.metrics_collector = MagicMock()
    processor.snapshot_auxiliary_output_starts = MagicMock(return_value=None)
    processor.materialize_sampling_mask_output = MagicMock()

    def normalize(**kwargs):
        assert kwargs["still_in_batch_rows"] == list(live_rows)
        # Feedback may activate B before the completed result is accounted.
        worker.speculative_num_steps = 3
        worker.speculative_num_draft_tokens = 4
        return [[1], [1]], None

    processor._normalize_decode_outputs = normalize
    # Stop after the production metrics boundary, before request/KV mutation.
    processor.token_to_kv_pool_allocator = NS(
        free_group_begin=MagicMock(side_effect=StopIteration)
    )
    batch = NS(
        reqs=[NS(live=live) for live in live_rows],
        batch_size=lambda: 2,
        spec_algorithm=NS(is_none=lambda: False),
    )
    with (
        patch(
            "sglang.srt.managers.scheduler_components.batch_result_processor.get_observability",
            return_value=NS(enable_metrics=True),
        ),
        patch(
            "sglang.srt.managers.scheduler_components.batch_result_processor.materialize_typed_customized_info"
        ),
        pytest.raises(StopIteration),
    ):
        SchedulerBatchResultProcessor.process_batch_result_decode(
            processor, batch, result
        )
    rows = reporter.spec_execution_window.drain()
    assert worker.speculative_num_steps == 3
    if instrumented:
        assert len(rows) == 1 and rows[0]["rounds"] == 1
        assert rows[0]["num_steps"] == 1
    else:
        assert rows == []
    processor.metrics_collector.increment_decode_cuda_graph_pass.assert_called_once_with(
        value=True
    )


def test_partial_window_flushed_at_idle_and_not_repeated_at_shutdown():
    r = reporter_for_window()
    r.scheduler.enable_fpm = False
    r._reset_device_timer_window = MagicMock()
    _, result = run_round()
    r.observe_spec_execution(result)
    with patch(
        "sglang.srt.managers.scheduler_components.metrics_reporter.logger"
    ) as logger:
        r._maybe_log_idle_metrics()
        r._shutdown_fpm()
        r.observe_spec_execution(result)
        r._maybe_log_idle_metrics()
    summaries = [
        json.loads(c.args[1])
        for c in logger.info.call_args_list
        if c.args[0] == "Completed speculative execution: %s"
    ]
    assert len(summaries) == 1 and summaries[0][0]["rounds"] == 1


def test_existing_target_counter_modes_and_units_unchanged():
    from sglang.srt.observability.metrics_collector import SchedulerMetricsCollector

    collector = object.__new__(SchedulerMetricsCollector)
    collector.labels = {"model_name": "test"}
    collector.cuda_graph_passes_total = MagicMock()
    for graph in (True, False, True):
        SchedulerMetricsCollector.increment_decode_cuda_graph_pass(collector, graph)
    assert [
        c.kwargs["mode"]
        for c in collector.cuda_graph_passes_total.labels.call_args_list
    ] == ["decode_cuda_graph", "decode_none", "decode_cuda_graph"]
    assert [
        c.args
        for c in collector.cuda_graph_passes_total.labels.return_value.inc.call_args_list
    ] == [(1,), (1,), (1,)]


@pytest.mark.parametrize(
    "specialized_graph,actual_fallback_graph",
    [(True, False), (False, False), (False, True)],
)
def test_real_draft_extend_dispatch_observations(
    specialized_graph, actual_fallback_graph
):
    from sglang.srt.speculative.eagle_worker_v2 import EagleDraftWorker
    from sglang.srt.speculative.execution_info import SpecExecutionRecorder

    worker = object.__new__(EagleDraftWorker)
    worker.speculative_num_draft_tokens = 2
    worker.seed_dsa_topk_from_draft_extend = False
    worker.topk = 1
    worker.plan_stream = None
    worker.plan_stream_ctx = contextlib.nullcontext()
    logits = NS(next_token_logits=torch.zeros(4, 2), hidden_states=None)
    worker.draft_runner = NS(
        canary_manager=None,
        forward=MagicMock(
            return_value=NS(logits_output=logits, can_run_graph=actual_fallback_graph)
        ),
    )
    worker.cuda_graph_runner_for_draft_extend = NS(
        can_run_graph=lambda batch: specialized_graph,
        execute=MagicMock(return_value=logits),
    )
    worker.draft_runtime = NS(
        note_draft_extend=lambda: None,
        extend_tokens=lambda **kwargs: (None, torch.ones(2, 1), torch.zeros(2, 1)),
    )
    result = GenerationBatchResult(
        prepared_draft_extend_inputs=(
            torch.zeros(2),
            torch.tensor([0, 2]),
            torch.zeros(2),
        ),
        logits_output=NS(hidden_states=None),
        accept_lens=torch.ones(2),
        next_draft_input=NS(),
    )
    recorder = SpecExecutionRecorder(1, 1, 2)
    recorder.verify_num_tokens_per_req = 2
    with (
        patch("sglang.srt.speculative.eagle_worker_v2.EagleDraftExtendInput"),
        patch(
            "sglang.srt.speculative.eagle_worker_v2.prepare_for_draft_extend",
            return_value=NS(),
        ),
        patch("sglang.srt.speculative.eagle_worker_v2.maybe_detect_nan"),
        patch("sglang.srt.speculative.eagle_worker_v2.maybe_detect_inf"),
        patch(
            "sglang.srt.speculative.eagle_worker_v2.get_spec",
            return_value=NS(speculative_use_rejection_sampling=False),
        ),
    ):
        worker._draft_extend_for_decode(NS(), result, execution=recorder)
    observed_graph = specialized_graph or actual_fallback_graph
    counts = recorder.finish().draft_extend
    assert counts.graph_forwards == int(observed_graph)
    assert counts.eager_forwards == int(not observed_graph)
    assert worker.cuda_graph_runner_for_draft_extend.execute.call_count == int(
        specialized_graph
    )
    assert worker.draft_runner.forward.call_count == int(not specialized_graph)


def test_info_disabled_discards_interval_without_output_materialization():
    r = reporter_for_window()
    _, completed = run_round(1)
    snapshot = completed.spec_execution
    r.observe_spec_execution(completed)

    class UnmaterializedRows(dict):
        def __iter__(self):
            raise AssertionError("disabled INFO must not sort or materialize rows")

    r.spec_execution_window._totals = UnmaterializedRows(
        r.spec_execution_window._totals
    )
    with (
        patch(
            "sglang.srt.managers.scheduler_components.metrics_reporter.logger"
        ) as logger,
        patch(
            "sglang.srt.managers.scheduler_components.metrics_reporter.json.dumps",
            side_effect=AssertionError("disabled INFO must not serialize"),
        ),
    ):
        logger.isEnabledFor.return_value = False
        r.report_spec_execution()
        logger.info.assert_not_called()
    assert r.spec_execution_window.drain() == []
    r.observe_spec_execution(completed)
    assert r.spec_execution_window.drain() == []
    assert completed.spec_execution is snapshot
    assert completed.spec_execution_accounted

    _, next_result = run_round(3)
    r.observe_spec_execution(next_result)
    with patch(
        "sglang.srt.managers.scheduler_components.metrics_reporter.logger"
    ) as logger:
        logger.isEnabledFor.return_value = True
        r.report_spec_execution()
    rows = json.loads(logger.info.call_args.args[1])
    assert len(rows) == 1 and rows[0]["num_steps"] == 3
    assert rows[0]["rounds"] == 1  # Discarded K1 interval cannot leak forward.


def test_discard_empty_or_populated_window_keeps_duplicate_marker():
    from sglang.srt.speculative.execution_info import SpecExecutionWindow

    w = SpecExecutionWindow()
    assert w.drain(discard=True) is None
    _, result = run_round(1)
    w.observe(result)
    assert w.drain(discard=True) is None
    w.observe(result)
    assert w.drain() == []
    assert result.spec_execution_accounted


def test_existing_key_accumulates_mixed_phases_without_mutating_snapshot():
    from sglang.srt.speculative.execution_info import (
        SpecExecutionRecorder,
        SpecExecutionWindow,
    )

    rec = SpecExecutionRecorder(3, 1, 4)
    rec.verify_num_tokens_per_req = 4
    rec.record("draft", graph=True, num_forwards=2)
    rec.record("draft", graph=False)
    rec.record("target_verify", graph=False)
    snapshot = rec.finish()
    before = pickle.dumps(snapshot)
    first = GenerationBatchResult(spec_execution=snapshot)
    second = GenerationBatchResult(spec_execution=snapshot)
    w = SpecExecutionWindow()
    w.observe(first)
    w.observe(second)
    w.observe(first)
    rows = w.drain()
    assert len(rows) == 1 and rows[0]["rounds"] == 2
    assert rows[0]["phases"] == {
        "draft": {"graph_forwards": 4, "eager_forwards": 2, "unexecuted_rounds": 0},
        "target_verify": {
            "graph_forwards": 0,
            "eager_forwards": 2,
            "unexecuted_rounds": 0,
        },
        "draft_extend": {
            "graph_forwards": 0,
            "eager_forwards": 0,
            "unexecuted_rounds": 2,
        },
    }
    assert pickle.dumps(snapshot) == before
    w.observe(second)
    assert w.drain() == []


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v", "-x"]))
