"""DSPark phase costs through the existing InferCast UMD composition executor."""

from __future__ import annotations

import hashlib
import json

from sglang_simulator.simulation.sglang.dspark import DSparkPhaseCost, DSparkStepTiming
from sglang_simulator.time_predictor.base import PredictorError


def _source_work_query_evidence(recorder, *, point, phase, policy, source_policy):
    """Bind one estimated SDK lookup to its original request and row anchors."""
    from infercast.sdk.diagnostics.prediction_evidence import linear_axis
    from infercast.sdk.native_segment_module import OP, NativeSegmentQuery

    def canonical(value):
        return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)

    def require(condition):
        if not condition:
            raise ValueError("missing matching SDK source-work query proof")

    try:
        shapes = json.loads(point["native_segment_queries"])
        require(
            len(shapes) == 1
            and recorder.dropped_queries == 0
            and len(recorder.nodes) == 1
        )
        requested = NativeSegmentQuery.from_shape(shapes[0])
        coordinate, prefix = requested.source_work_coordinate(policy=policy)
        node = recorder.nodes[0]
        require(
            node["op"] == OP
            and node["node_id"] == requested.segment_kind
            and node["phase"] == phase
            and node["status"] == "resolved"
            and node["source"] == "empirical"
            and type(node["count"]) is int
            and node["count"] == 1
            and type(node["repeat"]) is int
            and node["repeat"] == 1
            and canonical(node["shape"]) == canonical(requested.to_shape())
            and len(node["queries"]) == 1
        )
        query = node["queries"][0]
        metadata = query["row_metadata"]
        require(
            query["component"] == OP
            and query["role"] == "leaf"
            and query["source"] == "empirical"
            and query["kernel_source"] == "sglang"
            and query["kernel_impl"] == requested.to_shape()["kernel_impl"]
            and query["resolution"] in {"fallback", "interpolated"}
            and canonical(query["requested_key"]) == canonical(requested.to_shape())
            and requested.execution_profile_id == source_policy.execution_profile_id
            and metadata["execution_profile_id"] == requested.execution_profile_id
            and metadata["estimated_source_work_policy"]
            == metadata["interpolation_method"]
            == policy
            and metadata["source_binding_scope"] == "conditional_helper_file_inventory"
            and metadata["source_work_coordinate_sha256"]
            == hashlib.sha256(coordinate.encode()).hexdigest()
        )
        anchors = metadata["anchor_identities"]
        require(len(anchors) == 2)
        endpoints = []
        for shape in anchors:
            anchor = NativeSegmentQuery.from_shape(shape)
            group, position = anchor.source_work_coordinate(policy=policy)
            require(group == coordinate)
            endpoints.append(position)
        lower, upper = endpoints
        require(lower <= prefix <= upper)
        require(lower != upper or canonical(anchors[0]) == canonical(anchors[1]))
        weight = 0.0 if lower == upper else (prefix - lower) / (upper - lower)
        require(
            canonical(metadata["anchor_weights"]) == canonical([1 - weight, weight])
        )
        require(canonical(query["resolved_key"]) == canonical(anchors[0]))
        require(
            canonical(query["axes"])
            == canonical([linear_axis("prefix_len", prefix, lower, upper, weight)])
        )
        require(
            query["resolution"] == ("fallback" if lower == upper else "interpolated")
        )
        return query
    except (AttributeError, KeyError, IndexError, TypeError, ValueError) as error:
        raise PredictorError(
            "data_unavailable",
            "DSPark estimated cost lacks matching SDK source-work evidence",
        ) from error


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
        native_source_work_policy=None,
    ):
        self.database = database
        self.proposal_profile_id = proposal_profile_id
        self.preparers = dict(phase_preparers or {})
        self.native_source_policy = native_source_policy
        self.native_source_work_policy = native_source_work_policy
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
        if native_source_work_policy is not None:
            from infercast.sdk.native_golden_material import GOLDEN_LAW
            from infercast.sdk.native_segment_module import SOURCE_WORK_POLICY

            if (
                type(native_source_work_policy) is not str
                or native_source_work_policy != SOURCE_WORK_POLICY
                or native_source_policy is None
                or native_source_policy.golden_law != GOLDEN_LAW
                or native_source_policy.tp_size != 2
            ):
                raise PredictorError(
                    "invalid_execution_profile",
                    "native source work requires its versioned policy and verified golden source",
                )
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
            ) not in {(2, 1, 1), (4, 1, 1)} or len(step.groups) != 1:
                raise ValueError("native region profile requires TP2/TP4 ADP1/PP1")
            if step.topology.tp_size == 4 and len(step.requests) != 1:
                raise ValueError("TP4 native regions require B1")
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
            if not (
                self.native_source_policy.tp_size
                == step.topology.tp_size
                == query.native_input.to_dict()["tp_size"]
            ):
                raise ValueError(
                    "native region policy/query topology changed after projection"
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
        source_work = (
            self.native_source_work_policy is not None
            and native_region
            and len(step.groups) == 1
            and step.groups[0].forward_mode == "TARGET_VERIFY"
            and len(step.groups[0].requests) == 1
        )
        evaluation = {
            "phase": graph_phase,
            "env": bind_runtime_point(description, point),
            "layer_repeat": graph.layer_repeat,
        }
        recorder = None
        try:
            if source_work:
                from infercast.sdk.diagnostics.prediction_evidence import (
                    capture_prediction_evidence,
                )
                from infercast.sdk.native_segment_module import OP

                with capture_prediction_evidence() as recorder:
                    result = evaluate_phase_composition(
                        self.database,
                        nodes,
                        getattr(graph, graph_phase + "_composition"),
                        **evaluation,
                        operation_kwargs={
                            OP: {
                                "source_work_policy": self.native_source_work_policy,
                                "source_policy": self.native_source_policy,
                            }
                        },
                    )
            else:
                result = evaluate_phase_composition(
                    self.database,
                    nodes,
                    getattr(graph, graph_phase + "_composition"),
                    **evaluation,
                )
        except PerfDataNotAvailableError as error:
            raise PredictorError(
                "data_unavailable", f"missing qualified DSPark cost for phase: {phase}"
            ) from error
        source_work_evidence = None
        if result.source == "empirical" and source_work and recorder is not None:
            source_work_evidence = _source_work_query_evidence(
                recorder,
                point=point,
                phase=graph_phase,
                policy=self.native_source_work_policy,
                source_policy=self.native_source_policy,
            )
        elif result.source != "silicon":
            raise PredictorError(
                "data_unavailable", "DSPark requires measured silicon cost provenance"
            )
        identity_fields = {
            "phase": phase,
            "runtime": point,
            "slice": str(self.database.slice_ref.slice_dir),
        }
        if source_work:
            identity_fields["native_source_work_policy"] = (
                self.native_source_work_policy
            )
        if source_work_evidence is not None:
            identity_fields["source_work_query"] = source_work_evidence
        identity = json.dumps(
            identity_fields,
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
