"""Translate causal worker facts to the public SimOpus prepared-region API.

No graph, native query construction, cost resolver or acceptance law lives here.
Evaluated query records are deliberately distinct from committed native events.
"""

import gzip
import hashlib
import json
import math
from pathlib import Path

from sglang_simulator.simulation.sglang.dspark import (
    _DECODE_PHASES,
    _PREFILL_PHASES,
    DSparkPhaseCost,
    DSparkStepTiming,
)
from sglang_simulator.time_predictor.base import PredictorError
from sglang_simulator.utils.evidence import (
    ZstdCodec,
    evidence_level,
    metrics_cpu_detail_enabled,
)


def _encoded(value):
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")


class SimOpusNativeCostAdapter:
    """One initialized public UMD model and its verified native assets."""

    @classmethod
    def from_sdk(
        cls,
        database,
        model_id,
        *,
        execution_profile,
        native_segment_profile_id,
        source_binding=None,
        analytic_model=None,
        source_work_policy=None,
        model_revision="main",
        cost_evidence_path=None,
    ):
        from simopus.sdk import build_umd_static_model, load_native_region_assets

        try:
            assets = load_native_region_assets(
                database,
                execution_profile=execution_profile,
                native_segment_profile_id=native_segment_profile_id,
                source_binding=source_binding,
                analytic_model=analytic_model,
                source_work_policy=source_work_policy,
            )
            model = build_umd_static_model(
                database,
                model_id,
                backend="sglang",
                revision=model_revision,
                allow_download=False,
                native_region_assets=assets,
            )
        except (OSError, TypeError, ValueError, KeyError, NotImplementedError) as error:
            raise PredictorError(
                getattr(error, "code", "provider_initialization_failed"),
                "public native SDK initialization failed",
            ) from error
        if metrics_cpu_detail_enabled():
            return cls(model, assets, cost_evidence_path=cost_evidence_path)
        return cls.from_prepared_sdk(
            model, assets, cost_evidence_path=cost_evidence_path
        )

    @classmethod
    def from_prepared_sdk(cls, model, assets, *, cost_evidence_path=None):
        """Reuse one public-factory model and its exact already-loaded assets."""
        from simopus.sdk import NativeRegionAssets, UMDStaticModel

        if (
            type(model) is not UMDStaticModel
            or type(assets) is not NativeRegionAssets
            or getattr(model, "_native_region_assets", None) is not assets
            or not callable(getattr(model, "estimate_prepared_forward_ms", None))
        ):
            raise PredictorError(
                "provider_initialization_failed",
                "metrics CPU detail off requires a public model bound to these assets",
            )
        return cls(
            model,
            assets,
            cost_evidence_path=cost_evidence_path,
            _numeric_forward=model.estimate_prepared_forward_ms,
        )

    def __init__(
        self, model, assets, *, cost_evidence_path=None, _numeric_forward=None
    ):
        self.model = model
        self.assets = assets
        # The native event driver consumes source observations, never cost options.
        self.native_source_policy = assets.source_policy
        self.available_phases = frozenset((*_PREFILL_PHASES, *_DECODE_PHASES))
        self.last_evaluations = []
        self._counts = {}
        self._sink = None
        self._sink_path = None
        self._sink_failed = False
        self._sink_gzip = False
        self._sink_zstd = False
        self._zstd_codec = None
        self._digest = hashlib.sha256()
        self._bytes = 0
        self._evidence_level = evidence_level()
        self._omit_cpu_detail = not metrics_cpu_detail_enabled()
        if self._omit_cpu_detail and not callable(_numeric_forward):
            raise PredictorError(
                "provider_initialization_failed",
                "metrics CPU detail off requires public SDK factory initialization",
            )
        self._numeric_forward = _numeric_forward
        self._last_metrics_phases = []
        self._requested_sink_path = (
            str(Path(cost_evidence_path).absolute())
            if cost_evidence_path is not None
            else None
        )
        if cost_evidence_path is not None and self._evidence_level == "full":
            self._sink_path = str(Path(cost_evidence_path).absolute())
            self._sink_gzip = self._sink_path.endswith(".gz")
            self._sink_zstd = self._sink_path.endswith(".zst")
            self._zstd_codec = ZstdCodec() if self._sink_zstd else None
            try:
                self._sink = open(self._sink_path, "xb", buffering=0)  # noqa: SIM115 -- run-owned sink
            except OSError as error:
                raise PredictorError(
                    "data_unavailable", "cannot create evaluated-cost evidence"
                ) from error

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
        from sglang_simulator.simulation.sglang.dspark_reference import ReferenceCPUView
        from sglang_simulator.simulation.sglang.dspark_runtime import (
            capture_simulator_source_facts,
            sampling_snapshot,
        )

        if type(reference_view) is not ReferenceCPUView or native_batch is None:
            raise PredictorError(
                "data_unavailable", "native costs require before-work reference facts"
            )
        facts = capture_simulator_source_facts(native_batch)
        mode = "DECODE" if group.forward_mode == "TARGET_VERIFY" else "EXTEND"
        if (
            group.attention_dp_rank != 0
            or group.forward_mode not in {"EXTEND", "TARGET_VERIFY"}
            or facts.forward_mode != mode
            or runtime_profile_json != self.assets.runtime_profile_json
            or tuple((r.rid, r.max_new_tokens) for r in facts.requests)
            != tuple((r.rid, r.max_new_tokens) for r in group.requests)
            or tuple(sampling_snapshot(r.sampling_params) for r in native_batch.reqs)
            != tuple(r.sampling_params_json for r in group.requests)
        ):
            raise PredictorError(
                "invalid_execution_profile",
                "native ordered requests, sampling, mode or runtime changed",
            )
        return {
            "native_region_stage_inputs": {
                "runtime_profile_json": runtime_profile_json,
                "normalized_sampling": [
                    json.loads(r.sampling_params_json) for r in group.requests
                ],
                "request_metadata": [
                    (
                        r.origin_input_len,
                        r.to_finish_present,
                        r.input_embeds_present,
                        r.multimodal_inputs_present,
                    )
                    for r in facts.requests
                ],
                "seq_lens_cpu": reference_view.seq_lens_cpu,
                "nxt_kv_lens_cpu": reference_view.nxt_kv_lens_cpu,
                "buffer_capacity": buffer_capacity if mode == "DECODE" else None,
                "buffer_device_matches": (
                    buffer_device_matches if mode == "DECODE" else None
                ),
            }
        }

    def query_phase(self, phase, step):
        from simopus.sdk import ExtendRequestShape, PredictionResult
        from simopus.sdk.perf_database.database import PerfDataNotAvailableError

        if (
            step.topology.tp_size != self.native_source_policy.tp_size
            or step.topology.tp_size not in (2, 4)
            or step.topology.attention_dp_size != 1
            or step.topology.pp_size != 1
            or len(step.groups) != 1
            or step.runtime_profile_json != self.assets.runtime_profile_json
            or phase not in step.required_phases
        ):
            raise PredictorError(
                "unsupported_execution_profile", "unsupported native regional step"
            )
        group = step.groups[0]
        mode = "EXTEND" if group.forward_mode == "EXTEND" else "DECODE"
        try:
            inputs = json.loads(group.execution_json)["native_region_stage_inputs"]
            if (
                set(inputs)
                != {
                    "runtime_profile_json",
                    "normalized_sampling",
                    "request_metadata",
                    "seq_lens_cpu",
                    "nxt_kv_lens_cpu",
                    "buffer_capacity",
                    "buffer_device_matches",
                }
                or inputs["runtime_profile_json"] != step.runtime_profile_json
                or inputs["normalized_sampling"]
                != [json.loads(r.sampling_params_json) for r in group.requests]
            ):
                raise ValueError("stage runtime differs from before-work step")
            arguments = {
                "forward_mode": mode,
                "requests": tuple(
                    ExtendRequestShape(
                        prefix_len=r.target_prefix_len, extend_len=r.extend_len
                    )
                    for r in group.requests
                ),
                "execution_profile": self.assets.capture_profile,
                "runtime": {"native_region_stage": {"phase": phase, **inputs}},
            }
            if self._omit_cpu_detail:
                milliseconds = self._numeric_forward(**arguments)
                if (
                    type(milliseconds) not in (float, int)
                    or not math.isfinite(milliseconds)
                    or milliseconds <= 0
                ):
                    raise ValueError(
                        "prepared numeric cost must be finite and positive"
                    )
                cost = DSparkPhaseCost(
                    phase,
                    milliseconds / 1000,
                    f"simopus-metrics:{self.assets.execution_profile_id}:"
                    f"{step.forward_id}:{phase}",
                )
                self._last_metrics_phases.append((step.forward_id, phase))
                counts = self._counts.setdefault(
                    (phase, "not_recorded"),
                    {
                        "phase": phase,
                        "source": "not_recorded",
                        "queries": 0,
                        "seconds": 0.0,
                    },
                )
                counts["queries"] += 1
                counts["seconds"] += cost.seconds
                return cost
            result = self.model.estimate_prepared_forward_result(**arguments)
        except PerfDataNotAvailableError as error:
            raise PredictorError(
                "data_unavailable", f"missing qualified DSPark cost for phase: {phase}"
            ) from error
        except (KeyError, TypeError, ValueError, NotImplementedError) as error:
            raise PredictorError(
                getattr(error, "code", "invalid_execution_profile"),
                f"native prepared stage refused: {phase}",
            ) from error
        if type(result) is not PredictionResult:
            raise PredictorError("data_unavailable", "missing structured SDK evidence")
        # Revalidate nested evidence before publishing or returning event time.
        try:
            document = result.to_dict()
            PredictionResult.from_dict(document)
            scope = result.runtime_resolved["work_scope"]
            expected_source = {"silicon"}
            if self.assets.source_work_policy is not None:
                expected_source.add("empirical")
            if self.assets.analytic_model is not None:
                expected_source.add("sol")
            if not (
                scope["complete_dspark_worker_phase"] == phase
                and scope["native_segment_profile_id"]
                == self.assets.execution_profile_id
                and scope["native_segment_runtime_profile_sha256"]
                == hashlib.sha256(step.runtime_profile_json.encode()).hexdigest()
                and scope["complete_worker_partition"] is False
                and scope["unmodeled_worker_regions"]
                and scope["token_progress_modeled"] is False
                and result.operation
                == ("prefill" if mode == "EXTEND" else "decode_forward")
                and result.source in expected_source
                and result.nodes
                and all(node.queries for node in result.nodes)
                and result.runtime_input["forward_mode"] == mode
                and result.runtime_input["requests"]
                == [request.to_dict() for request in arguments["requests"]]
                and _encoded(result.runtime_input["runtime"])
                == _encoded(arguments["runtime"])
                and result.runtime_input["execution_profile"]
                == self.assets.capture_profile.to_dict()
            ):
                raise ValueError("SDK result has incompatible work scope or provenance")
            record = {
                "event": "evaluated_phase_cost",
                "forward_id": step.forward_id,
                "request_ids": [r.rid for r in group.requests],
                "phase": phase,
                "committed_event": False,
                "prediction": document,
            }
            line = _encoded(record) + b"\n"
            cost = DSparkPhaseCost(
                phase,
                result.latency_ms / 1000,
                "simopus:" + hashlib.sha256(line).hexdigest(),
            )
        except (KeyError, TypeError, ValueError) as error:
            raise PredictorError(
                "data_unavailable", "invalid native stage result or query evidence"
            ) from error
        if self._sink_failed:
            raise PredictorError("data_unavailable", "evaluated-cost evidence failed")
        if self._sink is not None:
            try:
                # Complete members preserve prior rows on interrupted teardown.
                if self._zstd_codec is not None:
                    payload = self._zstd_codec.compress(line)
                elif self._sink_gzip:
                    payload = gzip.compress(line, compresslevel=1, mtime=0)
                else:
                    payload = line
                if self._sink.write(payload) != len(payload):
                    raise OSError("short evaluated-cost evidence write")
                self._sink.flush()
            except (OSError, ValueError) as error:
                self._sink_failed = True
                raise PredictorError(
                    "data_unavailable", "evaluated-cost evidence failed"
                ) from error
        self._digest.update(line)
        self._bytes += len(line)
        self.last_evaluations.append(record)
        counts = self._counts.setdefault(
            (phase, result.source),
            {"phase": phase, "source": result.source, "queries": 0, "seconds": 0.0},
        )
        counts["queries"] += 1
        counts["seconds"] += cost.seconds
        return cost

    def predict_step(self, step):
        self.last_evaluations = []
        self._last_metrics_phases = []
        return DSparkStepTiming(
            step.forward_id,
            tuple(self.query_phase(phase, step) for phase in step.required_phases),
        ).validate_for(step)

    def get_cost_metrics(self):
        if self._omit_cpu_detail:
            return {
                "query_scope": "evaluated_cost_queries_not_committed_events",
                "by_phase_source": [dict(row) for row in self._counts.values()],
                "evaluated_evidence": {
                    "path": None,
                    "bytes": None,
                    "sha256": None,
                    "persisted": False,
                    "write_failed": False,
                    "retention_policy": "metrics",
                    "cpu_detail": "off",
                    "omitted_by_policy": True,
                    "requested_path": self._requested_sink_path,
                    "full_native_cost_ledger_available": False,
                    "full_cost_digest_available": False,
                    "cost_reference_scheme": "profile/forward/phase; not a cost digest",
                    "query_source_classification_available": False,
                },
            }
        return {
            "query_scope": "evaluated_cost_queries_not_committed_events",
            "by_phase_source": [dict(row) for row in self._counts.values()],
            "evaluated_evidence": {
                "path": self._sink_path,
                "bytes": self._bytes,
                "sha256": self._digest.hexdigest(),
                "persisted": self._sink is not None,
                "write_failed": self._sink_failed,
                **(
                    {
                        "compression": "zstd" if self._sink_zstd else "gzip",
                        "digest_scope": "uncompressed_jsonl",
                    }
                    if self._sink_gzip or self._sink_zstd
                    else {}
                ),
                **(
                    {
                        "retention_policy": "metrics",
                        "omitted_by_policy": True,
                        "requested_path": self._requested_sink_path,
                    }
                    if self._evidence_level == "metrics"
                    else {}
                ),
            },
        }

    def evaluated_evidence_for(self, step):
        """Immutable full SDK documents for the exact result being published."""
        if self._omit_cpu_detail:
            if tuple(self._last_metrics_phases) != tuple(
                (step.forward_id, phase) for phase in step.required_phases
            ):
                raise PredictorError(
                    "data_unavailable", "native numeric phase coverage is incomplete"
                )
            return ()
        if tuple(
            (row["forward_id"], row["phase"]) for row in self.last_evaluations
        ) != tuple((step.forward_id, phase) for phase in step.required_phases):
            raise PredictorError(
                "data_unavailable", "native evaluated evidence is incomplete"
            )
        return tuple(_encoded(row).decode("utf-8") for row in self.last_evaluations)

    def close(self):
        """Close the run-owned evidence stream without altering query history."""
        if self._sink is not None:
            self._sink.close()
