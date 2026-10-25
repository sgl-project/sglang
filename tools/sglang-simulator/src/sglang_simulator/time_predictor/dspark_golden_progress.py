"""Source-defined golden3.51 progress, separate from measured worker costs.

SHA256/mod100 is an explicit simulator sampling algorithm. It models the pinned
match-expected law; it does not replay native Torch CPU generator consumption.
The SDK owns original-source/material verification and the marginal cost contract.
"""

from __future__ import annotations

import hashlib
import json
import re

from sglang_simulator.simulation.sglang.dspark import DSparkProgress
from sglang_simulator.time_predictor.base import PredictorError

LAW = "dspark_match_expected_351_v1"
ALGORITHM = "sha256_decode_event_mod100_v1"


def _require(condition, message):
    if not condition:
        raise PredictorError("data_unavailable", message)


def _canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


class SourceGoldenDSparkProgress:
    """One modeled choice per DECODE event, with idempotent before-work retries.

    This law represents TP2/TP4 B1-B64, with ADP1/PP1. SDK policy verifies static
    Q5/Q6, captures and the golden cost realization. Session resolves every
    measured phase before calling this owner. Missing costs never draw progress.
    """

    def __init__(self, source_policy, *, model_seed):
        from infercast.sdk.native_segment_projection import NativeSegmentSourcePolicy

        _require(
            type(source_policy) is NativeSegmentSourcePolicy,
            "golden progress requires an SDK-verified native source policy",
        )
        _require(
            type(model_seed) is int and model_seed >= 0,
            "golden progress requires an explicit nonnegative integer model_seed",
        )
        material = getattr(source_policy, "golden_material_sha256", None)
        _require(
            getattr(source_policy, "golden_law", None) == LAW
            and type(material) is str
            and re.fullmatch("[0-9a-f]{64}", material) is not None
            and type(source_policy.tp_size) is int
            and source_policy.tp_size in (2, 4),
            "native source policy has no matching verified golden351 material",
        )
        self.topology = (source_policy.tp_size, 1, 1)
        self.runtime_profile_json = source_policy.runtime_profile_json
        self.material_sha256 = material
        self.seed = model_seed
        identity = {
            "law": LAW,
            "algorithm": ALGORITHM,
            "model_seed": model_seed,
            "runtime_profile_sha256": hashlib.sha256(
                self.runtime_profile_json.encode()
            ).hexdigest(),
            "golden_material_sha256": material,
        }
        self.model_ref = (
            "sha256:" + hashlib.sha256(_canonical(identity).encode()).hexdigest()
        )
        self.identity = identity

    def predict_progress(self, step):
        _require(
            step.runtime_profile_json == self.runtime_profile_json
            and (
                step.topology.tp_size,
                step.topology.attention_dp_size,
                step.topology.pp_size,
            )
            == self.topology,
            "golden progress runtime or topology differs from verified material",
        )
        _require(
            len(step.groups) == 1
            and step.groups[0].forward_mode == "TARGET_VERIFY"
            and 1 <= len(step.requests) <= 64
            and all(request.extend_len == 6 for request in step.requests),
            "golden progress requires static Q6 decode requests within its verified TP domain",
        )
        for request in step.requests:
            recipe = json.loads(request.sampling_params_json)
            _require(
                recipe.get("ignore_eos") is True
                and not any(
                    recipe.get(k)
                    for k in (
                        "stop",
                        "stop_regex",
                        "stop_strs",
                        "stop_regex_strs",
                        "stop_token_ids",
                        "json_schema",
                        "regex",
                        "ebnf",
                        "structural_tag",
                    )
                ),
                "golden progress requires length-only output without grammar or stop rules",
            )
        # Event identity, not per-RID sampling or a mutable RNG. Different request
        # batches cannot cause multiple draws for one native DECODE event.
        event = {"model_ref": self.model_ref, "forward_id": step.forward_id}
        choice = (
            int.from_bytes(hashlib.sha256(_canonical(event).encode()).digest(), "big")
            % 100
        )
        committed = 4 if choice < 51 else 3
        # Preserve the full commit even across remaining OSL. Native publication
        # advances KV and the native scheduler separately clips visible output.
        return DSparkProgress(
            step.forward_id,
            tuple((request.rid, committed) for request in step.requests),
            self.model_ref,
        ).validate_for(step)


class SourceGoldenDPProgress:
    """Separate local draws for attention-TP singletons, not global TP sharing.

    Independent per-DP hashes are an explicit modeled joint assumption. Native
    CPU generators and their correlations/consumption are not reproduced. The
    source-only SDK policy does not qualify costs or the caller's runtime JSON.
    """

    def __init__(self, source_policy, *, runtime_profile_json, model_seed):
        from infercast.sdk.native_dp_progress import NativeDPProgressPolicy

        _require(
            type(source_policy) is NativeDPProgressPolicy,
            "DP progress requires its SDK-verified singleton source policy",
        )
        _require(
            type(model_seed) is int and model_seed >= 0,
            "DP progress requires an explicit nonnegative integer model_seed",
        )
        try:
            runtime = json.loads(runtime_profile_json)
            valid = (
                type(runtime_profile_json) is str
                and isinstance(runtime, dict)
                and bool(runtime)
                and _canonical(runtime) == runtime_profile_json
            )
        except (TypeError, ValueError):
            valid = False
        _require(
            valid, "DP progress requires a canonical explicit cost-runtime context"
        )
        _require(source_policy.golden_law == LAW, "DP progress source law differs")
        self.runtime_profile_json = runtime_profile_json
        self.capacities = source_policy.sampler_capacities
        self.identity = {
            "law": LAW,
            "algorithm": "sha256_decode_event_dp_mod100_v1",
            "joint_assumption": "independent_local_hashes_not_native_rng_replay",
            "model_seed": model_seed,
            "source_identity_sha256": source_policy.source_identity_sha256,
            "runtime_profile_sha256": hashlib.sha256(
                runtime_profile_json.encode()
            ).hexdigest(),
        }
        self.model_ref = (
            "sha256:" + hashlib.sha256(_canonical(self.identity).encode()).hexdigest()
        )

    def predict_progress(self, step):
        _require(
            (
                step.topology.tp_size,
                step.topology.attention_dp_size,
                step.topology.pp_size,
            )
            == (2, 2, 1)
            and step.runtime_profile_json == self.runtime_profile_json
            and {group.forward_mode for group in step.groups} - {"IDLE"}
            == {"TARGET_VERIFY"},
            "DP progress requires complete serialized TP2/ADP2 decode groups and matching runtime context",
        )
        # Validate EVERY group before any event draw, including the last RID.
        for group, capacity in zip(step.groups, self.capacities, strict=True):
            _require(
                len(group.requests) <= capacity,
                "DP progress exceeds the source local sampler capacity",
            )
            for request in group.requests:
                recipe = json.loads(request.sampling_params_json)
                _require(
                    request.extend_len == 6
                    and request.emits_output is True
                    and recipe.get("ignore_eos") is True
                    and not any(
                        recipe.get(k)
                        for k in (
                            "stop",
                            "stop_regex",
                            "stop_strs",
                            "stop_regex_strs",
                            "stop_token_ids",
                            "json_schema",
                            "regex",
                            "ebnf",
                            "structural_tag",
                        )
                    ),
                    "DP progress requires static Q6 length-only requests",
                )
        counts = []
        for group in step.groups:
            if group.forward_mode == "IDLE":
                continue
            event = {
                "model_ref": self.model_ref,
                "forward_id": step.forward_id,
                "attention_dp_rank": group.attention_dp_rank,
            }
            draw = (
                int.from_bytes(
                    hashlib.sha256(_canonical(event).encode()).digest(), "big"
                )
                % 100
            )
            committed = 4 if draw < 51 else 3
            counts.extend((request.rid, committed) for request in group.requests)
        return DSparkProgress(
            step.forward_id, tuple(counts), self.model_ref
        ).validate_for(step)
