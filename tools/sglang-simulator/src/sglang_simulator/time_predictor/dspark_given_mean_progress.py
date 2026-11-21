"""Conditional progress from explicit caller-supplied acceptance information.

This input is additional workload information, not an independent acceptance
prediction. A mean specifies neither its distribution nor request correlations.
The explicit model below uses adjacent integers and one shared draw per event.
Native provenance qualification belongs to the input producer, not this sampler.
"""

import hashlib
import re
from decimal import Decimal
from fractions import Fraction

from sglang_simulator.simulation.sglang.dspark import DSparkProgress
from sglang_simulator.time_predictor.dspark_golden_progress import (
    SourceGoldenDSparkProgress,
    _canonical,
    _require,
)


def _formal_counter_mean(given):
    _require(
        given["token_definition"] == "committed_including_target_before_tail_clip"
        and given["warmup_excluded"] is True
        and given["prior_concurrency_excluded"] is True,
        "given mean must isolate formal committed progress before tail clipping",
    )
    window = given["formal_window_ns"]
    _require(
        type(window) is list
        and len(window) == 2
        and all(type(x) is int for x in window)
        and 0 < window[0] < window[1],
        "given mean requires explicit formal-window bounds",
    )
    _require(
        type(given["native_counter_artifact_sha256"]) is str
        and re.fullmatch("[0-9a-f]{64}", given["native_counter_artifact_sha256"])
        is not None,
        "given mean requires a retained native-counter artifact digest",
    )
    numerator, denominator = given["advance_tokens"], given["decode_request_steps"]
    _require(
        type(numerator) is int
        and type(denominator) is int
        and denominator > 0
        and denominator <= numerator <= 6 * denominator,
        "given mean must be a counter ratio within static Q6 support",
    )
    return Fraction(numerator, denominator)


def _published_mean(given):
    text = given["mean_acceptance"]
    _require(
        type(text) is str
        and len(text) <= 64
        and re.fullmatch(r"[0-9]+(?:\.[0-9]+)?", text) is not None,
        "published acceptance mean requires its explicit decimal string",
    )
    mean = Decimal(text)
    _require(1 <= mean <= 6, "published acceptance mean is outside static Q6 support")
    _require(
        type(given["source_artifact_sha256"]) is str
        and re.fullmatch("[0-9a-f]{64}", given["source_artifact_sha256"]) is not None
        and type(given["source_point_id"]) is str
        and 0 < len(given["source_point_id"]) <= 256
        and type(given["source_run_id"]) is int
        and given["source_run_id"] > 0,
        "published mean requires original artifact, point and run identity",
    )
    _require(
        given["source_precision"] == "as_published"
        and given["counter_verification"] == "not_independently_verified",
        "published means must disclose retained precision and unverified counters",
    )
    return Fraction(mean)


class GivenMeanDSparkProgress(SourceGoldenDSparkProgress):
    """ADP1 conditional input; preserve all verified cost-source and Q6 guards."""

    def __init__(self, source_policy, *, model_seed, given):
        super().__init__(source_policy, model_seed=model_seed)
        _require(
            type(given) is dict
            and type(given.get("schema_version")) is int
            and given["schema_version"] in (1, 2),
            "given mean requires a supported versioned input contract",
        )
        version = given["schema_version"]
        common = {"schema_version", "case_id", "runtime_profile_sha256"}
        fields = (
            {
                "advance_tokens",
                "decode_request_steps",
                "token_definition",
                "formal_window_ns",
                "warmup_excluded",
                "prior_concurrency_excluded",
                "native_counter_artifact_sha256",
            }
            if version == 1
            else {
                "mean_acceptance",
                "source_artifact_sha256",
                "source_point_id",
                "source_run_id",
                "source_precision",
                "counter_verification",
            }
        )
        _require(
            set(given) == common | fields,
            "given mean requires the complete counter or published-mean contract",
        )
        _require(
            type(given["schema_version"]) is int
            and given["schema_version"] in (1, 2)
            and type(given["case_id"]) is str
            and re.fullmatch(r"S[0-9]{2}-C[0-9]{2}", given["case_id"]) is not None,
            "given mean requires an explicit versioned case identity",
        )
        runtime_sha = hashlib.sha256(self.runtime_profile_json.encode()).hexdigest()
        _require(
            given["runtime_profile_sha256"] == runtime_sha,
            "given mean runtime differs from verified cost context",
        )
        if version == 1:
            self.mean = _formal_counter_mean(given)
            input_mode = "conditional_given_formal_counters"
            law = "given_formal_mean_acceptance_v1"
        else:
            self.mean = _published_mean(given)
            input_mode = "conditional_given_published_mean"
            law = "given_published_mean_acceptance_v1"
        self.lower = self.mean.numerator // self.mean.denominator
        self.upper_probability = self.mean - self.lower
        self.identity = {
            "law": law,
            "algorithm": "sha256_event_adjacent_integers_shared_draw_v1",
            "input_mode": input_mode,
            "independent_acceptance_prediction": False,
            "distribution_and_correlation": "adjacent_integers_shared_event_not_native_replay",
            "cost_source_identity": self.identity,
            "given": given,
        }
        self.model_ref = (
            "sha256:" + hashlib.sha256(_canonical(self.identity).encode()).hexdigest()
        )

    def predict_progress(self, step):
        self._validate_step(step)
        event = {"model_ref": self.model_ref, "forward_id": step.forward_id}
        draw = int.from_bytes(
            hashlib.sha256(_canonical(event).encode()).digest(), "big"
        )
        committed = self.lower + (Fraction(draw, 2**256) < self.upper_probability)
        return DSparkProgress(
            step.forward_id,
            tuple((request.rid, committed) for request in step.requests),
            self.model_ref,
        ).validate_for(step)
