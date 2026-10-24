"""DSPark phase costs through the existing InferCast UMD composition executor."""

from __future__ import annotations

import hashlib
import json

from sglang_simulator.simulation.sglang.dspark import DSparkPhaseCost, DSparkStepTiming
from sglang_simulator.time_predictor.base import PredictorError


class UMDDSparkCostModel:
    """Evaluate source-owned prepared graphs; no alternate latency table.

    An explicit SDK native source policy enables its complete region graphs.
    The legacy proposal selector remains a separate incomplete cost grain.
    Registration alone supplies neither measured costs nor source event order.
    """

    def __init__(
        self,
        database,
        *,
        proposal_profile_id=None,
        phase_preparers=None,
        native_source_policy=None,
    ):
        self.database = database
        self.proposal_profile_id = proposal_profile_id
        self.preparers = dict(phase_preparers or {})
        self.native_source_policy = native_source_policy
        if native_source_policy is not None:
            from infercast.sdk.native_segment_projection import (
                NativeSegmentSourcePolicy,
            )

            if (
                type(native_source_policy) is not NativeSegmentSourcePolicy
                or proposal_profile_id is not None
                or self.preparers
            ):
                raise PredictorError(
                    "invalid_execution_profile",
                    "native regions require one verified SDK policy and one cost grain",
                )
            from sglang_simulator.time_predictor.dspark_native_regions import _REGIONS

            self.preparers = {
                phase: self._prepare_native
                for phases in _REGIONS.values()
                for phase in phases
            }
        if proposal_profile_id is not None:
            if "proposal" in self.preparers:
                raise PredictorError(
                    "invalid_execution_profile",
                    "DSPark proposal preparers select conflicting cost grains",
                )
            self.preparers["proposal"] = self._prepare_proposal
        self._proposal_profile = None

    def execution_for(
        self,
        *,
        group,
        runtime_profile_json,
        native_batch=None,
        buffer_capacity=None,
        buffer_device_matches=None,
        reference_view=None,
    ):
        """Resolve SDK work from explicit before-work inputs.

        A native policy requires the source driver's frozen reference CPU view
        and matching request metadata. The legacy proposal path resolves only
        its own capture and does not reinterpret reference planning inputs.
        """
        if self.native_source_policy is not None:
            return self._native_execution(
                group=group,
                runtime_profile_json=runtime_profile_json,
                native_batch=native_batch,
                buffer_capacity=buffer_capacity,
                buffer_device_matches=buffer_device_matches,
                reference_view=reference_view,
            )
        if reference_view is not None:
            raise PredictorError(
                "data_unavailable",
                "reference CPU inputs require the complete native SDK source policy",
            )
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

    def _native_execution(
        self,
        *,
        group,
        runtime_profile_json,
        native_batch,
        buffer_capacity,
        buffer_device_matches,
        reference_view,
    ):
        from sglang_simulator.simulation.sglang.dspark_reference import ReferenceCPUView
        from sglang_simulator.simulation.sglang.dspark_runtime import (
            capture_simulator_source_facts,
            sampling_snapshot,
        )
        from sglang_simulator.time_predictor.dspark_native_regions import _REGIONS

        if type(reference_view) is not ReferenceCPUView:
            raise PredictorError(
                "data_unavailable",
                "native projection requires the source driver's before-work reference CPU view",
                missing_source_facts=["reference_view"],
            )
        if native_batch is None:
            raise PredictorError(
                "data_unavailable",
                "native projection requires the matching before-work request metadata",
                missing_source_facts=["native_batch"],
            )
        facts = capture_simulator_source_facts(native_batch)
        decode = group.forward_mode == "TARGET_VERIFY"
        mode = "DECODE" if decode else "EXTEND"
        if (
            group.attention_dp_rank != 0
            or group.forward_mode not in _REGIONS
            or facts.forward_mode != mode
            or tuple((r.rid, r.max_new_tokens) for r in facts.requests)
            != tuple((r.rid, r.max_new_tokens) for r in group.requests)
            or tuple(sampling_snapshot(r.sampling_params) for r in native_batch.reqs)
            != tuple(r.sampling_params_json for r in group.requests)
        ):
            raise PredictorError(
                "data_unavailable",
                "native projection request order, mode or complete sampling differs from the reference step",
            )
        from infercast.sdk.extend_forward import ExtendRequestShape
        from infercast.sdk.native_segment_projection import project_native_segment_query

        try:
            query = project_native_segment_query(
                execution_profile_id=self.native_source_policy.execution_profile_id,
                segment_kind=next(iter(_REGIONS[group.forward_mode].values())),
                source_policy=self.native_source_policy,
                runtime_profile_json=runtime_profile_json,
                forward_mode=mode,
                request_work=tuple(
                    ExtendRequestShape(
                        prefix_len=r.target_prefix_len, extend_len=r.extend_len
                    )
                    for r in group.requests
                ),
                normalized_sampling=tuple(
                    json.loads(r.sampling_params_json) for r in group.requests
                ),
                request_metadata=tuple(
                    (
                        r.origin_input_len,
                        r.to_finish_present,
                        r.input_embeds_present,
                        r.multimodal_inputs_present,
                    )
                    for r in facts.requests
                ),
                seq_lens_cpu=reference_view.seq_lens_cpu,
                nxt_kv_lens_cpu=reference_view.nxt_kv_lens_cpu,
                buffer_capacity=buffer_capacity if decode else None,
                buffer_device_matches=buffer_device_matches if decode else None,
            )
        except (KeyError, ValueError, TypeError) as error:
            raise PredictorError(
                "data_unavailable",
                "native SDK projection lacks compatible causal inputs",
                reason=str(error),
            ) from error
        return {
            "native_segment_profile_id": query.execution_profile_id,
            "native_segment_module_variant": query.module_variant,
        }

    def _prepare_native(self, *, step, phase):
        from sglang_simulator.time_predictor.dspark_native_regions import _REGIONS

        try:
            from infercast.sdk.models.umd.templates.deepseek_v41_segments import (
                prepare_native_dspark_phase,
            )
            from infercast.sdk.native_segment_module import NativeSegmentQuery

            if (
                step.topology.tp_size,
                step.topology.attention_dp_size,
                step.topology.pp_size,
            ) != (2, 1, 1) or len(step.groups) != 1:
                raise ValueError("native region profile requires TP2/ADP1/PP1")
            group = step.groups[0]
            execution = json.loads(group.execution_json)
            if (
                execution["native_segment_profile_id"]
                != self.native_source_policy.execution_profile_id
                or step.runtime_profile_json
                != self.native_source_policy.runtime_profile_json
            ):
                raise ValueError(
                    "native region policy/runtime changed after projection"
                )
            query = NativeSegmentQuery(
                execution["native_segment_profile_id"],
                _REGIONS[group.forward_mode][phase],
                execution["native_segment_module_variant"],
            )
            return prepare_native_dspark_phase(
                query=query, phase=phase, runtime_profile_json=step.runtime_profile_json
            )
        except (KeyError, ValueError, TypeError) as error:
            raise PredictorError(
                "data_unavailable", "native SDK phase projection is unavailable"
            ) from error

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
        from infercast.sdk.perf_database.database import PerfDataNotAvailableError

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
        from sglang_simulator.time_predictor.dspark_native_regions import (
            validate_native_region_phase,
        )

        native_region = validate_native_region_phase(
            description=description,
            point=point,
            graph_phase=graph_phase,
            phase=phase,
            step=step,
        )
        if (phase != "proposal" or native_region) and description.extra.get(
            "complete_dspark_worker_phase"
        ) != phase:
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
        try:
            result = evaluate_phase_composition(
                self.database,
                nodes,
                getattr(graph, graph_phase + "_composition"),
                phase=graph_phase,
                env=bind_runtime_point(description, point),
                layer_repeat=graph.layer_repeat,
            )
        except PerfDataNotAvailableError as error:
            raise PredictorError(
                "data_unavailable", f"missing qualified DSPark cost for phase: {phase}"
            ) from error
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
