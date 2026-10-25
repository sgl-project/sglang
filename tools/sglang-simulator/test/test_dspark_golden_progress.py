"""CPU source-law contracts; policy doubles are not qualified native evidence."""

import json
import sys
from dataclasses import replace
from types import ModuleType

import pytest
from sglang_simulator.simulation.sglang.dspark import (
    CommittedHistory,
    DSparkGroup,
    DSparkRequest,
    DSparkSession,
    DSparkStep,
    DSparkTopology,
)
from sglang_simulator.time_predictor import dspark_golden_progress
from sglang_simulator.time_predictor.base import PredictorError
from sglang_simulator.time_predictor.dspark_golden_progress import (
    LAW,
    SourceGoldenDSparkProgress,
)
from test_dspark_step import Cost

RUNTIME = '{"scope":"CPU policy seam, no native costs"}'


@pytest.fixture
def policy(monkeypatch):
    # Isolate the progress algorithm from the SDK's separately tested original
    # receipt loader. This type is deliberately a CPU fixture, never an artifact.
    class Policy:
        golden_law = LAW
        golden_material_sha256 = "a" * 64
        tp_size = 2
        runtime_profile_json = RUNTIME

    module = ModuleType("infercast.sdk.native_segment_projection")
    module.NativeSegmentSourcePolicy = Policy
    monkeypatch.setitem(sys.modules, module.__name__, module)
    return Policy()


def step(*, fid=0, rid="r", mode="TARGET_VERIFY", budget=20, **changes):
    recipe = {"ignore_eos": True, "is_normalized": True, "max_new_tokens": budget}
    req = DSparkRequest(
        rid,
        512,
        6,
        0,
        512,
        1,
        budget,
        json.dumps(recipe, sort_keys=True, separators=(",", ":")),
    )
    result = DSparkStep(
        fid,
        DSparkTopology(2, 1),
        (DSparkGroup(0, mode, (req,), '{"scope":"CPU phase seam"}'),),
        RUNTIME,
    )
    return replace(result, **changes)


def paired_step(*, fid=0, mode="TARGET_VERIFY"):
    before = step(fid=fid, mode=mode, budget=2)
    recipe = {"ignore_eos": True, "is_normalized": True, "max_new_tokens": 20}
    second = replace(
        before.requests[0],
        rid="second",
        target_prefix_len=8192,
        draft_valid_to=8192,
        max_new_tokens=20,
        sampling_params_json=json.dumps(recipe, sort_keys=True, separators=(",", ":")),
    )
    return replace(
        before,
        groups=(replace(before.groups[0], requests=before.requests + (second,)),),
    )


@pytest.mark.parametrize("seed", [None, True, -1, 1.0, "42"])
def test_explicit_seed_is_required(policy, seed):
    with pytest.raises(PredictorError, match="explicit nonnegative integer"):
        SourceGoldenDSparkProgress(policy, model_seed=seed)


@pytest.mark.parametrize(
    "field,value",
    [
        ("golden_law", None),
        ("golden_law", "natural"),
        ("golden_material_sha256", None),
        ("golden_material_sha256", "A" * 64),
        ("tp_size", 4),
    ],
)
def test_old_or_mismatched_policy_fails_closed(policy, field, value):
    setattr(policy, field, value)
    with pytest.raises(PredictorError, match="matching verified golden351"):
        SourceGoldenDSparkProgress(policy, model_seed=42)


def test_caller_approved_boolean_cannot_replace_sdk_policy(policy):
    with pytest.raises(PredictorError, match="SDK-verified"):
        SourceGoldenDSparkProgress({"approved": True}, model_seed=42)


def test_eventwise_law_retry_and_seed_identity(policy):
    model = SourceGoldenDSparkProgress(policy, model_seed=42)
    other = SourceGoldenDSparkProgress(policy, model_seed=43)
    assert model.model_ref != other.model_ref
    first = model.predict_progress(step(fid=77))
    assert model.predict_progress(step(fid=77)) == first
    # The native source chooses once per event, not per request identity.
    changed = model.predict_progress(step(fid=77, rid="another"))
    assert changed.committed_tokens[0][1] == first.committed_tokens[0][1]
    values = [
        model.predict_progress(step(fid=i)).committed_tokens[0][1] for i in range(3000)
    ]
    assert set(values) == {3, 4}
    assert sum(values) / len(values) == pytest.approx(3.51, abs=0.04)


@pytest.mark.parametrize("fid", [0, 1, 17, 77])
def test_b2_uses_one_shared_draw_with_ordered_requests_and_independent_budgets(
    policy, monkeypatch, fid
):
    model = SourceGoldenDSparkProgress(policy, model_seed=42)
    before = paired_step(fid=fid)
    expected = model.predict_progress(step(fid=fid)).committed_tokens[0][1]
    original_sha = dspark_golden_progress.hashlib.sha256
    calls = []

    def counted(*args, **kwargs):
        calls.append(args)
        return original_sha(*args, **kwargs)

    monkeypatch.setattr(dspark_golden_progress.hashlib, "sha256", counted)
    result = model.predict_progress(before)
    assert len(calls) == 1
    assert result.committed_tokens == (("r", expected), ("second", expected))
    assert model.predict_progress(before) == result
    reversed_step = replace(
        before,
        groups=(replace(before.groups[0], requests=tuple(reversed(before.requests))),),
    )
    assert model.predict_progress(reversed_step).committed_tokens == (
        ("second", expected),
        ("r", expected),
    )
    assert expected > before.requests[0].remaining_output_tokens
    assert expected < before.requests[1].remaining_output_tokens


@pytest.mark.parametrize("kind", ["runtime", "tp4", "batch3", "width5", "prefill"])
def test_unsupported_before_work_has_no_fallback(policy, kind):
    before = step()
    if kind == "runtime":
        before = replace(before, runtime_profile_json='{"different":true}')
    elif kind == "tp4":
        before = replace(before, topology=DSparkTopology(4, 1))
    elif kind == "batch3":
        requests = tuple(replace(before.requests[0], rid=f"r{i}") for i in range(3))
        before = replace(
            before,
            groups=(replace(before.groups[0], requests=requests),),
        )
    elif kind == "width5":
        before = replace(
            before,
            groups=(
                replace(
                    before.groups[0],
                    requests=(replace(before.requests[0], extend_len=5),),
                ),
            ),
        )
    else:
        before = step(mode="EXTEND")
    with pytest.raises(PredictorError):
        SourceGoldenDSparkProgress(policy, model_seed=42).predict_progress(before)


def test_second_b2_request_also_requires_static_q6(policy):
    before = paired_step()
    before = replace(
        before,
        groups=(
            replace(
                before.groups[0],
                requests=(
                    before.requests[0],
                    replace(before.requests[1], extend_len=5),
                ),
            ),
        ),
    )
    with pytest.raises(PredictorError, match="static Q6"):
        SourceGoldenDSparkProgress(policy, model_seed=42).predict_progress(before)


@pytest.mark.parametrize(
    "field,value",
    [
        ("ignore_eos", False),
        ("stop", ["done"]),
        ("stop_token_ids", [1]),
        ("json_schema", "{}"),
        ("regex", "abc"),
        ("structural_tag", '{"type":"structural_tag"}'),
    ],
)
@pytest.mark.parametrize("batch", [1, 2])
def test_content_dependent_finish_is_rejected(policy, field, value, batch):
    before = paired_step() if batch == 2 else step()
    req = before.requests[-1]
    recipe = json.loads(req.sampling_params_json)
    recipe[field] = value
    req = replace(
        req,
        sampling_params_json=json.dumps(recipe, sort_keys=True, separators=(",", ":")),
    )
    before = replace(
        before,
        groups=(replace(before.groups[0], requests=before.requests[:-1] + (req,)),),
    )
    with pytest.raises(PredictorError, match="length-only"):
        SourceGoldenDSparkProgress(policy, model_seed=42).predict_progress(before)


def test_invalid_second_request_fails_before_event_choice_or_history_change(
    policy, monkeypatch
):
    model = SourceGoldenDSparkProgress(policy, model_seed=42)
    before = paired_step()
    recipe = json.loads(before.requests[1].sampling_params_json)
    recipe["ignore_eos"] = False
    requests = (
        before.requests[0],
        replace(
            before.requests[1],
            sampling_params_json=json.dumps(
                recipe, sort_keys=True, separators=(",", ":")
            ),
        ),
    )
    before = replace(before, groups=(replace(before.groups[0], requests=requests),))

    def no_event(_):
        pytest.fail("invalid second request reached the event choice")

    monkeypatch.setattr(dspark_golden_progress, "_canonical", no_event)
    run = DSparkSession(Cost(), model)
    for request in before.requests:
        run.record_history(
            request.rid,
            CommittedHistory(
                request.target_prefix_len,
                request.draft_valid_from,
                request.draft_valid_to,
            ),
        )
    original = dict(run._history)
    with pytest.raises(PredictorError, match="length-only"):
        run.predict(before)
    assert run._history == original
    assert run._last_forward_id == -1


@pytest.mark.parametrize("batch", [1, 2])
def test_cost_failure_precedes_progress_and_retry_keeps_full_commit(policy, batch):
    before = paired_step() if batch == 2 else step(budget=2)

    class CountDraws(SourceGoldenDSparkProgress):
        calls = 0

        def predict_progress(self, before):
            self.calls += 1
            return super().predict_progress(before)

    model = CountDraws(policy, model_seed=42)

    class MissingCost:
        def predict_step(self, _):
            raise PredictorError("data_unavailable", "missing actual cost")

    run = DSparkSession(MissingCost(), model)
    for request in before.requests:
        run.record_history(
            request.rid,
            CommittedHistory(
                request.target_prefix_len,
                request.draft_valid_from,
                request.draft_valid_to,
            ),
        )
    original = dict(run._history)
    with pytest.raises(PredictorError, match="missing actual cost"):
        run.predict(before)
    assert model.calls == 0
    assert run._history == original
    run.cost_model = Cost()
    pending = run.predict(before)
    assert pending.progress == run.predict(before).progress
    count = pending.progress.committed_tokens[0][1]
    assert count in (3, 4) and count > before.requests[0].remaining_output_tokens
    run.commit(pending)
    for request, (_, committed) in zip(
        before.requests, pending.progress.committed_tokens, strict=True
    ):
        assert committed == count
        assert (
            run.history_for(request.rid).target_prefix_len
            == request.target_prefix_len + count
        )


@pytest.mark.parametrize("batch", [1, 2])
def test_prefill_never_calls_golden_draw(policy, batch):
    class NoDecode(SourceGoldenDSparkProgress):
        def predict_progress(self, _):
            pytest.fail("prefill must not draw golden progress")

    for emits in (False, True):
        before = paired_step(mode="EXTEND") if batch == 2 else step(mode="EXTEND")
        requests = tuple(
            replace(
                request,
                draft_valid_from=request.target_prefix_len,
                emits_output=emits if index == 0 else not emits,
            )
            for index, request in enumerate(before.requests)
        )
        before = replace(before, groups=(replace(before.groups[0], requests=requests),))
        run = DSparkSession(Cost(), NoDecode(policy, model_seed=42))
        assert run.predict(before).progress.committed_tokens == tuple(
            (request.rid, int(request.emits_output)) for request in requests
        )
