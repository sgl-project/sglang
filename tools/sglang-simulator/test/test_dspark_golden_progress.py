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


@pytest.mark.parametrize("kind", ["runtime", "tp4", "batch2", "width5", "prefill"])
def test_unsupported_before_work_has_no_fallback(policy, kind):
    before = step()
    if kind == "runtime":
        before = replace(before, runtime_profile_json='{"different":true}')
    elif kind == "tp4":
        before = replace(before, topology=DSparkTopology(4, 1))
    elif kind == "batch2":
        request = replace(before.requests[0], rid="second")
        before = replace(
            before,
            groups=(replace(before.groups[0], requests=before.requests + (request,)),),
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
def test_content_dependent_finish_is_rejected(policy, field, value):
    before = step()
    req = before.requests[0]
    recipe = json.loads(req.sampling_params_json)
    recipe[field] = value
    req = replace(
        req,
        sampling_params_json=json.dumps(recipe, sort_keys=True, separators=(",", ":")),
    )
    before = replace(before, groups=(replace(before.groups[0], requests=(req,)),))
    with pytest.raises(PredictorError, match="length-only"):
        SourceGoldenDSparkProgress(policy, model_seed=42).predict_progress(before)


def test_cost_failure_precedes_progress_and_retry_keeps_full_commit(policy):
    before = step(budget=2)

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
    run.record_history("r", CommittedHistory(512, 0, 512))
    with pytest.raises(PredictorError, match="missing actual cost"):
        run.predict(before)
    assert model.calls == 0
    assert run.history_for("r").target_prefix_len == 512
    run.cost_model = Cost()
    pending = run.predict(before)
    assert pending.progress == run.predict(before).progress
    count = pending.progress.committed_tokens[0][1]
    assert count in (3, 4) and count > before.requests[0].remaining_output_tokens
    run.commit(pending)
    assert run.history_for("r").target_prefix_len == 512 + count


def test_prefill_never_calls_golden_draw(policy):
    class NoDecode(SourceGoldenDSparkProgress):
        def predict_progress(self, _):
            pytest.fail("prefill must not draw golden progress")

    for emits in (False, True):
        before = step(mode="EXTEND")
        req = replace(before.requests[0], draft_valid_from=512, emits_output=emits)
        before = replace(before, groups=(replace(before.groups[0], requests=(req,)),))
        run = DSparkSession(Cost(), NoDecode(policy, model_seed=42))
        assert run.predict(before).progress.committed_tokens == (("r", int(emits)),)
