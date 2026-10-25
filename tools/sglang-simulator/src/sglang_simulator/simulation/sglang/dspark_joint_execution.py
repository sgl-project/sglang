"""Prospective joint coordinates from the existing native reference owners.

This explicit library resolver neither predicts a cost nor manufactures a raw
observation. Completed native frames retain the SDK's separate observation
converter and its real completion witnesses. Public DP/clock guards are unchanged.
"""

import hashlib
import json
from dataclasses import asdict, replace

from sglang_simulator.simulation.sglang.dspark import DSparkStep, DSparkTopology
from sglang_simulator.simulation.sglang.dspark_reference import ReferenceCPUView
from sglang_simulator.simulation.sglang.dspark_runtime import (
    canonical,
    capture_simulator_source_facts,
    sampling_snapshot,
)
from sglang_simulator.time_predictor.base import PredictorError


def _require(valid, message):
    if not valid:
        raise PredictorError("invalid_dspark_step", "joint execution " + message)


class NativeJointExecutionResolver:
    """Resolve the prepared DP0/DP1 views without outcomes or GPU reads.

    Pass ``execution_for`` to the existing local reference preparation and this
    instance to ``forward_native_dp_peer(joint_execution_resolver=...)``. Local
    request facts come from its actual CPU SchedulerBatch. ReferenceCPUView owns
    mirror availability and the independent reserved planner vector; a completed
    replay key, device P, or current acceptance never replaces those inputs.
    """

    def __init__(self, *, source_policy, runtime_profile_json, profiles):
        from infercast.sdk.decode_forward import DecodeExecutionProfile
        from infercast.sdk.extend_forward import ExtendExecutionProfile
        from infercast.sdk.native_dp_progress import NativeDPProgressPolicy
        from infercast.sdk.native_graph_work import validate_native_padding_receipt
        from infercast.sdk.native_joint_forward import JOINT_FORWARD_REALIZATION
        from infercast.sdk.perf_database.dsv41_native_forward import (
            parse_forward_profile,
        )

        _require(
            type(source_policy) is NativeDPProgressPolicy, "requires its DP policy"
        )
        _require(type(runtime_profile_json) is str, "requires its canonical runtime")
        physical = json.loads(runtime_profile_json)
        _require(
            canonical(physical) == runtime_profile_json, "runtime is not canonical"
        )
        digest = hashlib.sha256(runtime_profile_json.encode()).hexdigest()
        _require(
            type(profiles) is dict and set(profiles) == {"EXTEND", "DECODE"},
            "requires both phase profiles",
        )
        _require(
            type(profiles["DECODE"]) is DecodeExecutionProfile
            and type(profiles["EXTEND"]) is ExtendExecutionProfile,
            "phase profile classes differ",
        )
        for profile in profiles.values():
            _require(
                parse_forward_profile(profile.to_dict()) == profile
                and profile.full_forward_realization == JOINT_FORWARD_REALIZATION
                and profile.component_realization
                == "global_worker_observer_excluded_v1"
                and profile.runtime_compatibility_manifest_sha256 == digest,
                "phase profile differs from its whole-worker runtime",
            )
        self.source_policy = source_policy
        self.runtime_profile_json = runtime_profile_json
        self.profiles = dict(profiles)
        self._graphs = {}
        for role in physical["roles"]:
            rank, kind = role["parallel"]["tp_rank"], role["role"]
            _require(
                type(rank) is int
                and rank in (0, 1)
                and kind in {"target", "draft"}
                and (rank, kind) not in self._graphs
                and role["parallel"]["tp_size"] == 2
                and role["parallel"]["attn_dp_size"] == 2
                and role["parallel"]["attn_dp_rank"] == rank
                and role["parallel"]["attn_tp_size"] == 1
                and role["parallel"]["attn_tp_rank"] == 0,
                "requires four distinct TP2/ADP2 graph roles",
            )
            graph = role["graph"]
            disabled = validate_native_padding_receipt(graph["padding_policy"])
            _require(
                disabled
                == (profiles["DECODE"].graph_batch_policy == "exact_capture_only")
                and type(graph["request_pool_size"]) is int
                and graph["request_pool_size"] > 0
                and graph["captured_req_width"] == (5 if kind == "draft" else 6),
                "graph padding, pool or width differs",
            )
            self._graphs[rank, kind] = json.loads(canonical(graph))
        _require(
            set(self._graphs)
            == {(rank, kind) for rank in (0, 1) for kind in ("draft", "target")},
            "graph role inventory is incomplete",
        )

    def execution_for(
        self,
        *,
        group,
        runtime_profile_json,
        native_batch,
        buffer_capacity,
        buffer_device_matches,
        reference_view,
    ):
        """Freeze actual local request metadata before the existing rendezvous."""
        _require(
            runtime_profile_json == self.runtime_profile_json, "local runtime differs"
        )
        self._view(group, reference_view)
        _require(
            str(native_batch.device) == "cpu"
            and type(buffer_capacity) is int
            and buffer_capacity >= 0
            and type(buffer_device_matches) is bool,
            "requires independent local before-buffer state",
        )
        if group.forward_mode == "IDLE":
            _require(
                native_batch.forward_mode.name == "IDLE" and not native_batch.reqs,
                "idle batch has live requests",
            )
            requests = []
        else:
            facts = capture_simulator_source_facts(native_batch)
            _require(
                facts.forward_mode
                == ("DECODE" if group.forward_mode == "TARGET_VERIFY" else "EXTEND")
                and tuple((r.rid, r.max_new_tokens) for r in facts.requests)
                == tuple((r.rid, r.max_new_tokens) for r in group.requests)
                and tuple(
                    sampling_snapshot(r.sampling_params) for r in native_batch.reqs
                )
                == tuple(r.sampling_params_json for r in group.requests),
                "local ordered requests or full sampling differ",
            )
            requests = [asdict(r) for r in facts.requests]
        return {
            "native_joint_request_facts": requests,
            "native_joint_source_identity_sha256": self.source_policy.source_identity_sha256,
            "local_attention_dp_rank": group.attention_dp_rank,
            "draft_buffer_capacity": buffer_capacity,
            "draft_buffer_device_matches": buffer_device_matches,
        }

    @staticmethod
    def _view(group, view):
        _require(
            type(view) is ReferenceCPUView
            and view.local_attention_dp_rank == group.attention_dp_rank
            and view.is_idle == (group.forward_mode == "IDLE")
            and all(
                v is None or len(v) == len(group.requests)
                for v in (view.seq_lens_cpu, view.nxt_kv_lens_cpu)
            ),
            "reference view has a different local owner or request count",
        )

    def __call__(self, *, step, reference_views, local_executions):
        from infercast.sdk.native_forward_input import NativeForwardInput
        from infercast.sdk.native_graph_work import native_capture_key
        from infercast.sdk.native_joint_forward import NativeJointForwardQuery
        from infercast.sdk.native_sources import NATIVE_SOURCE_0522

        _require(
            type(step) is DSparkStep
            and step.topology == DSparkTopology(2, 2)
            and step.runtime_profile_json == self.runtime_profile_json
            and type(reference_views) is tuple
            and len(reference_views) == 2
            and type(local_executions) is tuple
            and len(local_executions) == 2,
            "requires both prepared owners and its frozen joint step",
        )
        modes = {g.forward_mode for g in step.groups} - {"IDLE"}
        _require(
            modes in ({"EXTEND"}, {"TARGET_VERIFY"}), "mixed phases are unsupported"
        )
        mode = "EXTEND" if modes == {"EXTEND"} else "DECODE"
        profile = self.profiles[mode]
        batch = max(len(g.requests) for g in step.groups)
        groups, sampling, work = [], [], []
        for group, view, local in zip(
            step.groups, reference_views, local_executions, strict=True
        ):
            self._view(group, view)
            rank = group.attention_dp_rank
            before = json.loads(group.execution_json)
            _require(
                type(local) is dict
                and set(local)
                == {
                    "native_joint_request_facts",
                    "native_joint_source_identity_sha256",
                    "local_attention_dp_rank",
                    "draft_buffer_capacity",
                    "draft_buffer_device_matches",
                }
                and local["local_attention_dp_rank"] == rank
                and local["native_joint_source_identity_sha256"]
                == self.source_policy.source_identity_sha256
                and set(before)
                == {"selection", "draft_buffer_capacity", "draft_buffer_device_matches"}
                and before.get("selection") == "native_peer_before_work_unresolved"
                and type(before["draft_buffer_capacity"]) is int
                and type(before["draft_buffer_device_matches"]) is bool
                and type(local["draft_buffer_capacity"]) is int
                and local["draft_buffer_capacity"] >= 0
                and type(local["draft_buffer_device_matches"]) is bool
                and all(
                    before.get(k) == local[k]
                    for k in ("draft_buffer_capacity", "draft_buffer_device_matches")
                ),
                "local source owner or independent before-buffer changed",
            )
            facts = local["native_joint_request_facts"]
            _require(
                type(facts) is list and len(facts) == len(group.requests),
                "request facts are incomplete",
            )
            for req, fact in zip(group.requests, facts, strict=True):
                _require(
                    type(fact) is dict
                    and set(fact)
                    == {
                        "rid",
                        "origin_input_len",
                        "max_new_tokens",
                        "to_finish_present",
                        "input_embeds_present",
                        "multimodal_inputs_present",
                    }
                    and fact["rid"] == req.rid
                    and fact["max_new_tokens"] == req.max_new_tokens
                    and type(fact["max_new_tokens"]) is int
                    and type(fact["origin_input_len"]) is int
                    and fact["origin_input_len"] >= 0
                    and all(
                        type(fact[k]) is bool
                        for k in (
                            "to_finish_present",
                            "input_embeds_present",
                            "multimodal_inputs_present",
                        )
                    ),
                    "ordered request metadata changed",
                )
            bound = planner = None
            if mode == "EXTEND":
                _require(
                    view.nxt_kv_lens_cpu is None
                    and view.seq_lens_cpu
                    == tuple(
                        r.target_prefix_len + r.extend_len for r in group.requests
                    ),
                    "prefill requires its actual P+Q mirror and no decode reservation",
                )
                execution = {
                    k: None
                    for k in (
                        "buffer_capacity",
                        "buffer_device_matches",
                        "draft_read_extent_tokens",
                        "draft_capture_key",
                        "target_capture_key",
                        "verify_plan_cpu_max_seq_len",
                    )
                }
            else:
                if group.requests:
                    # Pinned worker selector: prefer the source reference host
                    # mirror, otherwise original input + full output budget +
                    # one pending verify width. Planner uses its own CPU vector.
                    bound = (
                        max(view.seq_lens_cpu)
                        if view.seq_lens_cpu is not None
                        else (
                            max(
                                f["origin_input_len"] + f["max_new_tokens"]
                                for f in facts
                            )
                            + 6
                            if all(
                                not any(
                                    f[k]
                                    for k in (
                                        "to_finish_present",
                                        "input_embeds_present",
                                        "multimodal_inputs_present",
                                    )
                                )
                                for f in facts
                            )
                            else None
                        )
                    )
                    lengths = view.planner_seq_lens_cpu
                    planner = None if lengths is None else max(lengths)
                execution = {
                    "buffer_capacity": local["draft_buffer_capacity"],
                    "buffer_device_matches": local["draft_buffer_device_matches"],
                    "draft_read_extent_tokens": [
                        min(128, r.target_prefix_len) for r in group.requests
                    ],
                    "verify_plan_cpu_max_seq_len": planner,
                }
                variant = (
                    "candidate_unfiltered"
                    if bound is not None and bound + 6 <= 16384
                    else "candidate_filtered"
                )
                for kind, selected_variant in (("draft", None), ("target", variant)):
                    key = native_capture_key(
                        batch,
                        profile.graph_capture_batches,
                        selected_variant,
                        disable_padding=profile.graph_batch_policy
                        == "exact_capture_only",
                    )
                    graph = self._graphs[rank, kind]
                    _require(
                        key in graph["graph_keys"]
                        and key["size"] <= graph["request_pool_size"],
                        "selected graph is absent from its actual role inventory",
                    )
                    execution[kind + "_capture_key"] = key
            groups.append(
                {
                    "attention_dp_rank": rank,
                    "phase": "IDLE" if not group.requests else mode,
                    "requests": [
                        {"prefix_len": r.target_prefix_len, "extend_len": r.extend_len}
                        for r in group.requests
                    ],
                    "target_seq_len_upper_bound": bound,
                }
            )
            sampling.append(
                [json.loads(r.sampling_params_json) for r in group.requests]
            )
            work.append(dict(attention_dp_rank=rank, **execution))
        native = NativeForwardInput.from_dict(
            {
                "schema_version": 1,
                "source_pin": NATIVE_SOURCE_0522.name,
                "sampling_recipe": {
                    "source_sha256": NATIVE_SOURCE_0522.sampling_params,
                    "per_attention_dp": sampling,
                },
                "groups": groups,
            }
        )
        query = NativeJointForwardQuery.from_input(
            native,
            profile=profile,
            source_policy=self.source_policy,
            runtime_profile_json=self.runtime_profile_json,
            group_execution=work,
        )
        encoded = canonical(query.to_runtime(self.source_policy))
        return replace(
            step, groups=tuple(replace(g, execution_json=encoded) for g in step.groups)
        )
