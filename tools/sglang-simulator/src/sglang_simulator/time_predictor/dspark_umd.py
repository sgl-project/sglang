"""DSPark phase costs through the existing InferCast UMD composition executor."""

from __future__ import annotations

import hashlib
import json

from sglang_simulator.simulation.sglang.dspark import DSparkPhaseCost, DSparkStepTiming
from sglang_simulator.time_predictor.base import PredictorError


class UMDDSparkCostModel:
    """Evaluate source-owned prepared graphs; no alternate latency table.

    Existing proposal preparation is wired directly. Other complete boundaries
    must be supplied by their owning UMD family implementation. Missing phases
    remain gaps, including target-call gaps omitted by the 44-leaf composition.
    """

    def __init__(self, database, *, proposal_profile_id=None, phase_preparers=None):
        self.database = database
        self.proposal_profile_id = proposal_profile_id
        self.preparers = dict(phase_preparers or {})
        if proposal_profile_id is not None:
            self.preparers["proposal"] = self._prepare_proposal
        self._proposal_profile = None

    def execution_for(self, *, group, runtime_profile_json):
        """Resolve draft capture from retained SDK profile metadata, not outputs.

        This resolves the existing source-bound proposal domain only. Target
        and prefill execution decisions remain owned by their UMD preparation.
        """
        from infercast.sdk.perf_database.slice_manifest import (
            read_verified_slice_artifact,
        )

        if group.forward_mode != "TARGET_VERIFY":
            return {"forward_mode": group.forward_mode}
        if self.proposal_profile_id is None:
            raise PredictorError(
                "data_unavailable", "DSPark draft execution profile is absent"
            )
        if self._proposal_profile is None:
            root = self.database.slice_ref.slice_dir
            identities = json.loads(
                read_verified_slice_artifact(root, "_meta/kernel_identity.json")
            )["identities"]
            refs = [
                entry.get("extra", {}).get("execution_profile")
                for entry in identities
                if entry.get("op") == "dsv41_dspark_proposal_module"
                and entry.get("config_token") == self.proposal_profile_id
            ]
            if not refs or any(ref != refs[0] for ref in refs):
                raise PredictorError(
                    "data_unavailable", "DSPark draft profile is missing or ambiguous"
                )
            ref = refs[0]
            if ref is None or ref.get("sha256") != self.proposal_profile_id:
                raise PredictorError(
                    "invalid_execution_profile", "DSPark draft profile identity differs"
                )
            self._proposal_profile = json.loads(
                read_verified_slice_artifact(
                    root, ref["path"], expected_sha256=ref["sha256"]
                )
            )
        definition = self._proposal_profile["graph_definition"]
        key = {
            "size": len(group.requests),
            "attention_variant": None,
            "stream_idx": None,
            "variant_label": None,
        }
        eligible = (
            definition.get("tokens_per_request") == 5
            and definition.get("ragged_verify_mode") is False
            and key in definition.get("capture_keys", [])
        )
        if not eligible:
            raise PredictorError(
                "data_unavailable",
                "DSPark draft capture has no matching source profile",
            )
        return {"draft_capture_key": key, "draft_graph_eligible": True}

    def _prepare_proposal(self, *, step, phase):
        from infercast.sdk.dspark_proposal import project_dspark_proposal_work
        from infercast.sdk.extend_forward import ExtendRequestShape
        from infercast.sdk.models.umd.templates.deepseek_v41_proposal import (
            prepare_dspark_proposal_module,
        )

        if (step.topology.tp_size, step.topology.attention_dp_size) != (2, 1):
            raise PredictorError(
                "data_unavailable",
                "proposal PerfDB has no matching TP4/ADP2 realization",
            )
        group = step.groups[0]
        execution = json.loads(group.execution_json)
        try:
            before = {
                "request_work": tuple(
                    ExtendRequestShape(prefix_len=r.target_prefix_len, extend_len=5)
                    for r in group.requests
                ),
                "draft_history_tokens": tuple(
                    r.bounded_draft_history_tokens(128) for r in group.requests
                ),
                "sampling_params": tuple(
                    json.loads(r.sampling_params_json) for r in group.requests
                ),
                "buffer_capacity": execution["draft_buffer_capacity"],
                "buffer_device_matches": execution["draft_buffer_device_matches"],
                "graph_eligible": execution["draft_graph_eligible"],
            }
            query = project_dspark_proposal_work(
                execution_profile_id=self.proposal_profile_id,
                capture_key=json.dumps(
                    execution["draft_capture_key"],
                    sort_keys=True,
                    separators=(",", ":"),
                ),
                **before,
            )
        except (KeyError, ValueError, TypeError) as error:
            raise PredictorError(
                "data_unavailable",
                "proposal is outside the source-bound causal cost class",
            ) from error
        description, point = prepare_dspark_proposal_module(query, **before)
        return description, point, "generation"

    def query_phase(self, phase, step):
        from infercast.sdk.models.umd.description import bind_runtime_point
        from infercast.sdk.models.umd.executor import evaluate_phase_composition

        if phase not in self.preparers:
            raise PredictorError(
                "data_unavailable", f"missing complete DSPark worker phase: {phase}"
            )
        description, point, graph_phase = self.preparers[phase](phase=phase, step=step)
        graph = description.op_graph
        if graph is None or graph_phase not in {"context", "generation"}:
            raise PredictorError(
                "invalid_execution_profile", "DSPark cost needs its UMD prepared graph"
            )
        if (
            phase != "proposal"
            and description.extra.get("complete_dspark_worker_phase") != phase
        ):
            raise PredictorError(
                "data_unavailable",
                "UMD graph does not cover the complete requested DSPark phase",
            )
        if phase == "target_verify" and description.extra.get(
            "unmodeled_target_call_units"
        ):
            raise PredictorError(
                "data_unavailable",
                "target module composition leaves target-call work uncovered",
            )
        nodes = getattr(graph, graph_phase)
        if not nodes or any(node.op == "const_latency" for node in nodes):
            raise PredictorError(
                "data_unavailable", "DSPark costs require measured UMD operations"
            )
        result = evaluate_phase_composition(
            self.database,
            nodes,
            getattr(graph, graph_phase + "_composition"),
            phase=graph_phase,
            env=bind_runtime_point(description, point),
            layer_repeat=graph.layer_repeat,
        )
        if result.source != "silicon":
            raise PredictorError(
                "data_unavailable", "DSPark requires measured silicon cost provenance"
            )
        identity = json.dumps(
            {
                "phase": phase,
                "runtime": point,
                "slice": str(self.database.slice_ref.slice_dir),
            },
            sort_keys=True,
            separators=(",", ":"),
        )
        return DSparkPhaseCost(
            phase,
            float(result) / 1000,
            "umd:" + hashlib.sha256(identity.encode()).hexdigest(),
        )

    def predict_step(self, step):
        missing = [
            phase for phase in step.required_phases if phase not in self.preparers
        ]
        if missing:
            raise PredictorError(
                "data_unavailable",
                "DSPark full worker costs are incomplete",
                missing_worker_phases=missing,
            )
        return DSparkStepTiming(
            step.forward_id,
            tuple(self.query_phase(phase, step) for phase in step.required_phases),
        ).validate_for(step)
