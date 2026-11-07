"""Conditional input validation and sampling; CPU fixtures are not native evidence."""

import hashlib
from dataclasses import replace

import pytest
from sglang_simulator.time_predictor.base import PredictorError
from sglang_simulator.time_predictor.dspark_given_mean_progress import (
    GivenMeanDSparkProgress,
)
from test_dspark_golden_progress import RUNTIME, paired_step
from test_dspark_golden_progress import policy as policy
from test_dspark_golden_progress import step


def given(numerator=184, denominator=100):
    return dict(
        schema_version=1,
        case_id="S17-C07",
        runtime_profile_sha256=hashlib.sha256(RUNTIME.encode()).hexdigest(),
        advance_tokens=numerator,
        decode_request_steps=denominator,
        token_definition="committed_including_target_before_tail_clip",
        formal_window_ns=[100, 200],
        warmup_excluded=True,
        prior_concurrency_excluded=True,
        native_counter_artifact_sha256="b" * 64,
    )


def test_given_ratio_event_retry_and_full_commit(policy):
    model = GivenMeanDSparkProgress(policy, model_seed=42, given=given())
    values = [
        model.predict_progress(step(fid=i)).committed_tokens[0][1] for i in range(4000)
    ]
    assert set(values) == {1, 2}
    assert sum(values) / len(values) == pytest.approx(1.84, abs=0.025)
    assert model.predict_progress(step(fid=19)) == model.predict_progress(step(fid=19))
    pair = paired_step(fid=27)
    result = model.predict_progress(pair)
    assert result.committed_tokens[0][1] == result.committed_tokens[1][1]
    assert model.identity["independent_acceptance_prediction"] is False


@pytest.mark.parametrize("n,d", [(0, 1), (7, 1), (True, 1), (2, False), (1, 0)])
def test_unsupported_ratios_fail(policy, n, d):
    with pytest.raises(PredictorError, match="counter ratio"):
        GivenMeanDSparkProgress(policy, model_seed=42, given=given(n, d))


@pytest.mark.parametrize(
    "key,value",
    [
        ("warmup_excluded", False),
        ("prior_concurrency_excluded", False),
        ("token_definition", "visible_output_after_clip"),
        ("formal_window_ns", [200, 100]),
        ("runtime_profile_sha256", "c" * 64),
        ("native_counter_artifact_sha256", "rounded_reported_mean"),
    ],
)
def test_mismatched_counter_context_fails(policy, key, value):
    document = given()
    document[key] = value
    with pytest.raises(PredictorError):
        GivenMeanDSparkProgress(policy, model_seed=42, given=document)


def test_default_source_guards_remain_and_mean_does_not_clip_tail(policy):
    model = GivenMeanDSparkProgress(policy, model_seed=42, given=given(4, 1))
    assert model.predict_progress(step(budget=2)).committed_tokens == (("r", 4),)
    with pytest.raises(PredictorError, match="runtime or topology"):
        model.predict_progress(
            replace(step(), runtime_profile_json='{"different":true}')
        )
    with pytest.raises(PredictorError, match="static Q6"):
        model.predict_progress(step(mode="EXTEND"))


def published(mean="2.20"):
    return dict(
        schema_version=2,
        case_id="S12-C07",
        runtime_profile_sha256=hashlib.sha256(RUNTIME.encode()).hexdigest(),
        mean_acceptance=mean,
        source_artifact_sha256="c" * 64,
        source_point_id="published-native-point",
        source_run_id=40712452076,
        source_precision="as_published",
        counter_verification="not_independently_verified",
    )


def test_published_mean_keeps_precision_and_does_not_invent_counters(policy):
    model = GivenMeanDSparkProgress(policy, model_seed=42, given=published())
    assert str(model.mean) == "11/5"
    assert model.identity["input_mode"] == "conditional_given_published_mean"
    assert model.identity["given"]["mean_acceptance"] == "2.20"
    assert "advance_tokens" not in model.identity["given"]
    assert "formal_window_ns" not in model.identity["given"]
    values = [
        model.predict_progress(step(fid=i)).committed_tokens[0][1] for i in range(4000)
    ]
    assert set(values) == {2, 3}
    assert sum(values) / len(values) == pytest.approx(2.2, abs=0.03)


@pytest.mark.parametrize("mean", [2.2, True, "NaN", "1e99", "0.9", "6.01", "", "2_2"])
def test_published_mean_must_be_an_explicit_supported_decimal(policy, mean):
    with pytest.raises(PredictorError):
        GivenMeanDSparkProgress(policy, model_seed=42, given=published(mean))


@pytest.mark.parametrize(
    "key,value",
    [
        ("source_artifact_sha256", "missing"),
        ("source_point_id", ""),
        ("source_run_id", True),
        ("source_precision", "unrounded"),
        ("counter_verification", "independently_verified"),
    ],
)
def test_published_identity_and_unverified_status_are_required(policy, key, value):
    given = published()
    given[key] = value
    with pytest.raises(PredictorError):
        GivenMeanDSparkProgress(policy, model_seed=42, given=given)
