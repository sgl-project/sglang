"""CPU contracts only. Fixture timings/progress are not performance evidence."""

from dataclasses import replace
from types import SimpleNamespace

import pytest
from sglang_simulator.simulation.sglang.dspark import (
    STEP_ATTRIBUTE,
    CommittedHistory,
    DSparkGroup,
    DSparkPhaseCost,
    DSparkProgress,
    DSparkRequest,
    DSparkSession,
    DSparkStep,
    DSparkStepTiming,
    DSparkTopology,
)
from sglang_simulator.simulation.sglang.dspark_worker import C_DSparkWorkerHook
from sglang_simulator.time_predictor.base import PredictorError


def request(rid="a", prefix=512, **kwargs):
    return DSparkRequest(
        rid=rid,
        target_prefix_len=prefix,
        extend_len=6,
        draft_valid_from=0,
        draft_valid_to=prefix,
        output_tokens_before=1,
        max_new_tokens=20,
        sampling_params_json='{"is_normalized":true,"max_new_tokens":20}',
        **kwargs,
    )


def step(tp=2, dp=1, fid=0, requests=None):
    groups = (
        DSparkGroup(0, "TARGET_VERIFY", requests or (request(),), '{"path":"fixture"}'),
    )
    if dp == 2:
        groups += (DSparkGroup(1, "IDLE", (), '{"path":"coordinated_idle"}'),)
    return DSparkStep(
        fid, DSparkTopology(tp, dp), groups, '{"source":"CPU protocol fixture"}'
    )


class Cost:
    calls = 0

    def predict_step(self, before):
        self.calls += 1
        return DSparkStepTiming(
            before.forward_id,
            tuple(
                DSparkPhaseCost(p, 0.01, "CPU fixture") for p in before.required_phases
            ),
        )


class Progress:
    calls = 0
    count = 2

    def predict_progress(self, before):
        self.calls += 1
        return DSparkProgress(
            before.forward_id,
            tuple((r.rid, self.count) for r in before.requests),
            "CPU fixture; no acceptance fit",
        )


def session(before, cost=None, progress=None):
    result = DSparkSession(cost or Cost(), progress or Progress())
    for r in before.requests:
        result.record_history(
            r.rid,
            CommittedHistory(r.target_prefix_len, r.draft_valid_from, r.draft_valid_to),
        )
    return result


@pytest.mark.parametrize("tp,dp", [(2, 1), (4, 1), (2, 2)])
def test_source_layout_and_idle_participants_are_retained_without_multiplier(tp, dp):
    before = step(tp, dp)
    result = session(before).execute(before)
    assert before.topology.attention_tp_size == tp // dp
    assert len(result.step.groups) == dp
    assert result.timing.seconds == pytest.approx(0.05)
    assert result.progress.committed_tokens == (("a", 2),)


def test_dp_partition_is_required_and_never_inferred_from_global_batch():
    before = step()
    with pytest.raises(PredictorError, match="every actual attention-DP partition"):
        replace(before, topology=DSparkTopology(2, 2))
    with pytest.raises(PredictorError, match="unique live requests"):
        replace(
            before,
            topology=DSparkTopology(2, 2),
            groups=(before.groups[0], replace(before.groups[0], attention_dp_rank=1)),
        )


def test_next_step_reads_only_previously_committed_history():
    before = step()
    run = session(before)
    run.execute(before)
    assert run.history_for("a") == CommittedHistory(514, 0, 514)
    with pytest.raises(PredictorError, match="committed causal history"):
        run.execute(replace(before, forward_id=1))
    second_request = replace(
        request(), target_prefix_len=514, draft_valid_to=514, output_tokens_before=3
    )
    second = step(fid=1, requests=(second_request,))
    run.execute(second)
    assert run.history_for("a") == CommittedHistory(516, 0, 516)


def test_long_target_prefix_does_not_supply_unobserved_draft_history():
    before = step(requests=(request(prefix=80000),))
    run = DSparkSession(Cost(), Progress())
    with pytest.raises(PredictorError, match="no prior injection/commit"):
        run.execute(before)


def test_missing_worker_cost_fails_before_progress_and_state_mutation():
    before = step()
    cost = Cost()
    full = cost.predict_step(before)
    cost.predict_step = lambda before: replace(full, phases=full.phases[:-1])
    progress = Progress()
    run = session(before, cost, progress)
    with pytest.raises(PredictorError, match="omits or reorders"):
        run.execute(before)
    assert progress.calls == 0
    assert run.history_for("a") == CommittedHistory(512, 0, 512)
    cost.predict_step = Cost().predict_step
    run.execute(before)  # failed calls do not consume the forward ID


@pytest.mark.parametrize("count", [True, 0, -1, 7, 25])
def test_progress_rejects_invalid_or_unexecuted_commit_lengths(count):
    before = step()
    progress = Progress()
    progress.count = count
    run = session(before, progress=progress)
    with pytest.raises(PredictorError):
        run.execute(before)
    assert run.history_for("a") == CommittedHistory(512, 0, 512)


def test_tail_native_commit_advances_full_kv_before_scheduler_output_clipping():
    r = replace(request(), output_tokens_before=19)
    before = step(requests=(r,))
    run = session(before)
    result = run.execute(before)
    assert r.remaining_output_tokens == 1
    assert result.progress.committed_tokens == (("a", 2),)
    assert run.history_for("a") == CommittedHistory(514, 0, 514)


def test_unknown_draft_gap_cannot_be_bridged_by_commit():
    r = replace(request(), draft_valid_to=400)
    before = step(requests=(r,))
    run = session(before)
    with pytest.raises(PredictorError, match="unknown draft KV gap"):
        run.execute(before)
    assert run.history_for("a") == CommittedHistory(512, 0, 400)


def test_prefill_progress_is_native_bonus_and_injected_work_not_spec_acceptance():
    before = step()
    before = replace(before, groups=(replace(before.groups[0], forward_mode="EXTEND"),))
    run = session(before)
    result = run.execute(before)
    assert result.progress.committed_tokens == (("a", 1),)
    assert run.progress_model.calls == 0
    assert run.history_for("a") == CommittedHistory(518, 0, 518)


def test_native_dspark_hook_routes_to_explicit_runtime_and_retains_result(monkeypatch):
    from sglang_simulator.simulation.sglang.scheduler import C_SchedulerHook

    class Worker:
        pass

    C_DSparkWorkerHook.hook(Worker)
    worker = Worker.__new__(Worker)
    before = step()
    prediction = session(before).execute(before)
    output = SimpleNamespace()
    setattr(output, STEP_ATTRIBUTE, prediction)
    calls = []

    def forward(**kwargs):
        calls.append(kwargs)
        return output

    monkeypatch.setattr(
        C_SchedulerHook,
        "INFERENCE_PREDICTOR",
        SimpleNamespace(
            supports_dspark=True,
            dspark_runtime=SimpleNamespace(forward_batch_generation=forward),
        ),
    )
    batch = object()
    assert worker.forward_batch_generation(batch) is output
    assert calls == [{"worker": worker, "batch": batch, "on_publish": None}]
    with pytest.raises(PredictorError, match="grammar-overlap"):
        worker.forward_batch_generation(batch, grammar_barrier=lambda: None)
    # Scheduler supplies the callback for every DSPark batch. The native worker
    # consumes it only when has_grammar is true; no callback runs for this case.
    grammar_free = SimpleNamespace(has_grammar=False)
    barrier_calls = []
    assert (
        worker.forward_batch_generation(
            grammar_free, grammar_barrier=lambda: barrier_calls.append("unexpected")
        )
        is output
    )
    assert calls[-1] == {"worker": worker, "batch": grammar_free, "on_publish": None}
    assert barrier_calls == []
    with pytest.raises(PredictorError, match="grammar-overlap"):
        worker.forward_batch_generation(
            SimpleNamespace(has_grammar=True), grammar_barrier=lambda: None
        )


def test_native_dspark_hook_has_no_latency_or_progress_fallback(monkeypatch):
    from sglang_simulator.simulation.sglang.scheduler import C_SchedulerHook

    class Worker:
        pass

    C_DSparkWorkerHook.hook(Worker)
    monkeypatch.setattr(C_SchedulerHook, "INFERENCE_PREDICTOR", SimpleNamespace())
    with pytest.raises(PredictorError, match="independent progress model"):
        Worker.__new__(Worker).forward_batch_generation(object())


def test_worker_cleanup_reaches_the_same_runtime_state(monkeypatch):
    from sglang_simulator.simulation.sglang.scheduler import C_SchedulerHook

    class Worker:
        pass

    C_DSparkWorkerHook.hook(Worker)
    worker = Worker.__new__(Worker)
    calls = []
    runtime = SimpleNamespace(
        note_request_finished=lambda **kwargs: calls.append(kwargs),
        clear_cache_pool=lambda: calls.append("clear"),
    )
    monkeypatch.setattr(
        C_SchedulerHook,
        "INFERENCE_PREDICTOR",
        SimpleNamespace(supports_dspark=True, dspark_runtime=runtime),
    )
    worker.note_request_finished(rid="a", natural_stop=False)
    worker.clear_cache_pool()
    assert calls == [{"rid": "a", "natural_stop": False}, "clear"]
    assert worker.dump_info_records() is None
    with pytest.raises(PredictorError, match="verify-budget"):
        worker.set_dspark_forced_budget_frac(0.5)


def test_result_validates_the_local_partition():
    before = step(dp=2)
    result = session(before).execute(before, local_attention_dp_rank=1)
    assert result.local_group.forward_mode == "IDLE"
    with pytest.raises(PredictorError, match="outside the declared topology"):
        replace(result, local_attention_dp_rank=2)
