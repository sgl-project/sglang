"""Synthetic record/JSON compatibility controls; not model timing evidence."""

import json
from pathlib import Path

import msgspec
import pytest
from msgspec.structs import replace
from sglang_simulator.simulation.sglang.dspark import (
    CommittedHistory,
    DSparkGroup,
    DSparkPhaseCost,
    DSparkProgress,
    DSparkRequest,
    DSparkStep,
    DSparkStepResult,
    DSparkStepTiming,
    DSparkTopology,
)
from sglang_simulator.simulation.sglang.dspark_runtime import (
    DSparkSimulatorRequestFacts,
    DSparkSimulatorSourceFacts,
)
from sglang_simulator.time_predictor.base import PredictorError


def records():
    request = DSparkRequest("a", 512, 6, 384, 512, 1, 20, '{"max_new_tokens":20}')
    group = DSparkGroup(
        0, "TARGET_VERIFY", (request,), '{"scope":"software representation fixture"}'
    )
    step = DSparkStep(
        0, DSparkTopology(2, 1), (group,), '{"scope":"software representation fixture"}'
    )
    timing = DSparkStepTiming(
        0,
        tuple(
            DSparkPhaseCost(name, 0.01, "synthetic leaf")
            for name in step.required_phases
        ),
    )
    progress = DSparkProgress(0, (("a", 2),), "synthetic progress")
    DSparkStepResult(step, timing, progress)
    facts = DSparkSimulatorSourceFacts(
        "DECODE",
        (DSparkSimulatorRequestFacts("a", 2, 20, False, False, False),),
        (512,),
        True,
        (768,),
    )
    return step, timing, progress, facts


def test_native_record_projection_preserves_original_json_bytes():
    step, timing, progress, facts = records()
    actual = {
        "step": msgspec.to_builtins(step),
        "timing": msgspec.to_builtins(timing),
        "progress": msgspec.to_builtins(progress),
        "source_facts": facts.to_dict(),
        "history": msgspec.to_builtins(CommittedHistory(512, 384, 512)),
    }
    expected = Path(__file__).parent / "fixtures/native-record-representation.json"
    assert (json.dumps(actual) + "\n").encode() == expected.read_bytes()


def test_native_replace_revalidates_instead_of_bypassing_invariants():
    step, timing, _, _ = records()
    with pytest.raises(PredictorError):
        replace(step.requests[0], extend_len=True)
    with pytest.raises(PredictorError):
        replace(timing.phases[0], seconds=float("nan"))
    assert replace(step) == step and replace(step) is not step


def test_source_scope_is_observed_metadata_not_a_constructor_input():
    _, _, _, facts = records()
    assert facts.observation_scope == "simulator_cpu_before_worker_v1"
    with pytest.raises(TypeError):
        DSparkSimulatorSourceFacts(
            facts.forward_mode,
            facts.requests,
            facts.seq_lens_cpu,
            facts.draft_input_present,
            facts.nxt_kv_lens_cpu,
            observation_scope=facts.observation_scope,
        )
    with pytest.raises(AttributeError):
        facts.observation_scope = "other"
    with pytest.raises(TypeError):
        replace(facts, observation_scope="other")
    assert list(facts.to_dict())[-1] == "observation_scope"
