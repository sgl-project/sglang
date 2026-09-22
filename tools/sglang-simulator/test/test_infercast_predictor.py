import hashlib
import json
import math
import sys
from pathlib import Path
from types import ModuleType

import pytest
from sglang_simulator.simulation.manager.state import StateManager
from sglang_simulator.simulation.sglang.scheduler import predict_schedule_batch
from sglang_simulator.simulation.types import SchedulerConfig
from sglang_simulator.spec.accelerator import AcceleratorInfo
from sglang_simulator.spec.data_type import DataType
from sglang_simulator.spec.model import ModelInfo
from sglang_simulator.time_predictor import (
    PredictorError,
    ScheduleBatch,
    ScheduleRequest,
)
from sglang_simulator.time_predictor.infercast import (
    PREFIX_REDUCTION_POLICY,
    RAGGED_REDUCTION_POLICY,
    REALIZATION_REDUCTION_POLICY,
    PROFILED_DECODE_REDUCTION_POLICY,
    InferCastTimePredictor,
    milliseconds_to_seconds,
    reduce_batch,
    validate_provider_contract,
    validate_topology,
)

_GRAPH_PROFILE = {
    "schema_version": 1,
    "prefill_graph_backend": "tc_piecewise",
    "decode_graph_backend": "disabled",
    "graph_capture_policy": "catalog_exact_shapes_v1",
    "kv_page_size": 1,
    "runtime_compatibility_id": "compat-v3",
    "profile_id": "b" * 64,
}

_REALIZATION_PROFILE = {
    "schema_version": 2,
    "prefill_graph_backend": "tc_piecewise",
    "decode_graph_backend": "disabled",
    "graph_capture_policy": "catalog_exact_shapes_v2",
    "kv_page_size": 1,
    "runtime_compatibility_id": "compat-v4",
    "prefill_graph_compiler": "eager",
    "prefill_graph_capture_tokens": [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 2048, 4096],
    "full_forward_realization": "sglang_tc_piecewise_fx_split_replay_v1",
    "component_realization": "sglang_isolated_mla_cudagraph_module_v1",
    "runtime_compatibility_manifest_sha256": "c" * 64,
    "profile_id": "d" * 64,
}

_DECODE_PROFILE = {
    "schema_version": 1,
    "decode_graph_backend": "full",
    "graph_capture_batches": [1, 2, 4, 8],
    "graph_batch_policy": "exact_capture_only",
    "fallback_policy": "reject",
    "kv_page_size": 1,
    "runtime_compatibility_id": "decode-v1",
    "full_forward_realization": "sglang_decode_full_graph_v1",
    "component_realization": "sglang_mla_cuda_graph_v1",
    "runtime_compatibility_manifest_sha256": "e" * 64,
    "profile_id": "f" * 64,
}


class _ExecutionProfile:
    def __init__(self, payload):
        self.payload = dict(payload)

    @classmethod
    def from_dict(cls, payload):
        return cls(payload)

    def validate_for_kernel_impl(self, kernel_impl):
        if kernel_impl != "cuda_graph":
            raise ValueError("profile requires cuda_graph")

    def to_dict(self):
        return dict(self.payload)


class _DecodeExecutionProfile:
    def __init__(self, payload):
        self.payload = dict(payload)

    @classmethod
    def from_dict(cls, payload):
        return cls(payload)

    @property
    def kernel_impl(self):
        return (
            "eager"
            if self.payload["decode_graph_backend"] == "disabled"
            else "cuda_graph"
        )

    def to_dict(self):
        return dict(self.payload)


def _batch(mode, *requests):
    return ScheduleBatch(
        [ScheduleRequest(extend, prefix) for extend, prefix in requests],
        forward_mode=mode,
    )


@pytest.mark.parametrize(
    "batch,method,expected,raw_scale",
    [
        (
            _batch("DECODE", (1, 127)),
            "estimate_decode_forward_ms",
            {"batch_size": 1, "history_len": 127},
            None,
        ),
        (
            _batch("DECODE", (1, 100), (1, 201)),
            "estimate_decode_forward_ms",
            {"batch_size": 2, "history_len": 150},
            None,
        ),
        (
            _batch("EXTEND", (1024, 0), (1024, 0)),
            "estimate_extend_forward_ms",
            {
                "batch_size": 2,
                "extend_len": 1024,
                "prefix_len": 0,
                "seq_imbalance_correction_scale": 1.0,
            },
            1.0,
        ),
        (
            _batch("EXTEND", (1, 0)),
            "estimate_extend_forward_ms",
            {
                "batch_size": 1,
                "extend_len": 1,
                "prefix_len": 0,
                "seq_imbalance_correction_scale": 1.0,
            },
            1.0,
        ),
        (
            _batch("EXTEND", (256, 768)),
            "estimate_extend_forward_ms",
            {
                "batch_size": 1,
                "extend_len": 256,
                "prefix_len": 768,
                "seq_imbalance_correction_scale": 1.0,
            },
            1.0,
        ),
        (
            _batch("EXTEND", (512, 0)),
            "estimate_extend_forward_ms",
            {
                "batch_size": 1,
                "extend_len": 512,
                "prefix_len": 0,
                "seq_imbalance_correction_scale": 1.0,
            },
            1.0,
        ),
        (
            _batch("EXTEND", (512, 512)),
            "estimate_extend_forward_ms",
            {
                "batch_size": 1,
                "extend_len": 512,
                "prefix_len": 512,
                "seq_imbalance_correction_scale": 1.0,
            },
            1.0,
        ),
        (
            _batch("EXTEND", (4, 10), (2, 20)),
            "estimate_extend_forward_ms",
            {
                "batch_size": 2,
                "extend_len": 3,
                "prefix_len": 15,
                "seq_imbalance_correction_scale": 10 / 11,
            },
            10 / 11,
        ),
        (
            _batch("MIXED", (1, 100), (8, 0)),
            "estimate_extend_forward_ms",
            {
                "batch_size": 2,
                "extend_len": 4,
                "prefix_len": 50,
                "seq_imbalance_correction_scale": 1.0,
            },
            530 / 1881,
        ),
        (
            _batch("EXTEND", (2, 0), (3, 1)),
            "estimate_extend_forward_ms",
            {
                "batch_size": 2,
                "extend_len": 3,
                "prefix_len": 0,
                "seq_imbalance_correction_scale": 38 / 35,
            },
            38 / 35,
        ),
    ],
)
def test_mean_attention_flops_v1(batch, method, expected, raw_scale):
    reduced = reduce_batch(batch)
    assert reduced.method == method
    assert reduced.arguments == pytest.approx(expected)
    assert reduced.raw_attention_scale == pytest.approx(raw_scale)


@pytest.mark.parametrize(
    "batch,code",
    [
        (_batch("DECODE"), "invalid_batch"),
        (_batch("EXTEND", (0, 0)), "invalid_batch"),
        (_batch("EXTEND", (1, -1)), "invalid_batch"),
        (_batch("EXTEND", (True, 0)), "invalid_batch"),
        (_batch("EXTEND", (1, 0.5)), "invalid_batch"),
        (_batch("DECODE", (2, 10)), "invalid_batch"),
        (_batch("TARGET_VERIFY", (1, 10)), "unsupported_forward_mode"),
        (_batch("FUTURE_MODE", (1, 10)), "unsupported_forward_mode"),
    ],
)
def test_reduction_rejects_invalid_batches(batch, code):
    with pytest.raises(PredictorError) as exc_info:
        reduce_batch(batch)
    assert exc_info.value.code == code


@pytest.mark.parametrize(
    "mode",
    ["DRAFT_EXTEND_V2", "SPLIT_PREFILL", "DLLM_EXTEND", "PREBUILT"],
)
def test_known_unsupported_modes_fail_explicitly(mode):
    with pytest.raises(PredictorError) as exc_info:
        reduce_batch(_batch(mode, (1, 10)))
    assert exc_info.value.code == "unsupported_forward_mode"


@pytest.mark.parametrize("value", [0, -1, math.nan, math.inf, "invalid"])
def test_provider_output_must_be_finite_and_positive(value):
    with pytest.raises(PredictorError) as exc_info:
        milliseconds_to_seconds(value)
    assert exc_info.value.code == "invalid_provider_output"


@pytest.mark.parametrize(
    "config",
    [
        SchedulerConfig(tp_size=2, dp_size=2),
        SchedulerConfig(tp_size=2, cp_size=2),
        SchedulerConfig(tp_size=3, ep_size=2),
    ],
)
def test_unsupported_topology_fails_closed(config):
    with pytest.raises(PredictorError) as exc_info:
        validate_topology(config)
    assert exc_info.value.code == "unsupported_topology"


class _Provider:
    def __init__(self, value=12.5, error=None):
        self.value = value
        self.error = error
        self.calls = []

    def _call(self, method, arguments):
        self.calls.append((method, arguments))
        if self.error:
            raise self.error
        return self.value

    def estimate_decode_forward_ms(self, **arguments):
        return self._call("estimate_decode_forward_ms", arguments)

    def estimate_profiled_decode_forward_ms(self, **arguments):
        return self._call("estimate_profiled_decode_forward_ms", arguments)

    def estimate_extend_forward_ms(self, **arguments):
        return self._call("estimate_extend_forward_ms", arguments)

    def estimate_ragged_extend_forward_ms(self, **arguments):
        return self._call("estimate_ragged_extend_forward_ms", arguments)


def _predictor(
    provider,
    *,
    model=None,
    config=None,
    model_revision="c" * 40,
    revision="a" * 40,
    contract_version=1,
    reduction_policy=None,
    execution_profile=None,
    decode_execution_profile=None,
    decode_attn_kernel_impl=None,
):
    return InferCastTimePredictor(
        model or ModelInfo(),
        AcceleratorInfo(
            name="MI350X",
            vendor="AMD",
            hbm_capacity_gb=1,
            hbm_bandwidth_gb=1,
        ),
        config or SchedulerConfig(backend_name="sglang", backend_version="0.5.17"),
        model_id="Qwen/Qwen3-32B-FP8",
        system="mi350x",
        systems_root="/unused",
        attn_kernel_impl="cuda_graph",
        decode_attn_kernel_impl=decode_attn_kernel_impl,
        attn_dtype="bfloat16",
        kv_cache_dtype="fp8",
        model_revision=model_revision,
        provider_revision=revision,
        contract_version=contract_version,
        reduction_policy=reduction_policy,
        execution_profile=(
            (
                _REALIZATION_PROFILE
                if contract_version in {4, 5}
                else _GRAPH_PROFILE
            )
            if contract_version in {3, 4, 5} and execution_profile is None
            else execution_profile
        ),
        decode_execution_profile=(
            _DECODE_PROFILE
            if contract_version == 5 and decode_execution_profile is None
            else decode_execution_profile
        ),
        _provider=provider,
        _request_shape_factory=lambda **values: values,
        _execution_profile_factory=_ExecutionProfile,
        _decode_execution_profile_factory=_DecodeExecutionProfile,
        _provider_version="0.1.0",
        _stack_digest="b" * 64,
    )


def test_adapter_calls_one_forward_and_converts_units():
    provider = _Provider()
    latency = _predictor(provider).predict_infer_time(_batch("DECODE", (1, 127)))
    assert latency == pytest.approx(0.0125)
    assert len(provider.calls) == 1
    method, arguments = provider.calls[0]
    assert method == "estimate_decode_forward_ms"
    assert arguments["history_len"] == 127
    assert arguments["tp_size"] == 1


def test_v4_selects_decode_kernel_impl_from_phase_profile():
    provider = _Provider()
    predictor = _predictor(provider, contract_version=4)

    predictor.predict_infer_time(_batch("DECODE", (1, 127)))

    assert provider.calls[0][1]["attn_kernel_impl"] == "eager"
    runtime = predictor.get_metrics()["infercast"]["runtime"]
    assert runtime["attn_kernel_impl"] == "cuda_graph"
    assert runtime["decode_attn_kernel_impl"] == "eager"


def test_v4_rejects_decode_kernel_impl_that_conflicts_with_profile():
    provider = _Provider()

    with pytest.raises(PredictorError, match="decode_attn_kernel_impl"):
        _predictor(
            provider,
            contract_version=4,
            decode_attn_kernel_impl="cuda_graph",
        )


@pytest.mark.parametrize("mode", ["EXTEND", "MIXED"])
def test_v2_passes_one_ordered_request_level_call(mode):
    provider = _Provider()
    predictor = _predictor(
        provider,
        contract_version=2,
        reduction_policy=RAGGED_REDUCTION_POLICY,
    )

    latency = predictor.predict_infer_time(
        _batch(mode, (1, 1024), (127, 0))
    )

    assert latency == pytest.approx(0.0125)
    assert provider.calls == [
        (
            "estimate_ragged_extend_forward_ms",
            {
                "forward_mode": mode,
                "requests": (
                    {"extend_len": 1, "prefix_len": 1024},
                    {"extend_len": 127, "prefix_len": 0},
                ),
                "reduction_policy": RAGGED_REDUCTION_POLICY,
                "tp_size": 1,
                "pp_size": 1,
                "moe_tp_size": 1,
                "moe_ep_size": 1,
                "attention_dp_size": 1,
                "attn_kernel_impl": "cuda_graph",
                "attn_dtype": "bfloat16",
                "kv_cache_dtype": "fp8",
            },
        )
    ]
    metrics = predictor.get_metrics()["infercast"]
    assert metrics["contract_version"] == 2
    assert metrics["reduction_policy"] == RAGGED_REDUCTION_POLICY


def test_v2_one_token_extend_is_not_decode():
    provider = _Provider()
    predictor = _predictor(provider, contract_version=2)

    predictor.predict_infer_time(_batch("EXTEND", (1, 63), (1, 65)))

    assert provider.calls[0][0] == "estimate_ragged_extend_forward_ms"
    assert provider.calls[0][1]["forward_mode"] == "EXTEND"


def test_v3_passes_profile_separately_from_the_ordered_workload():
    provider = _Provider()
    predictor = _predictor(
        provider,
        contract_version=3,
        reduction_policy=PREFIX_REDUCTION_POLICY,
    )

    predictor.predict_infer_time(_batch("MIXED", (1, 1024), (127, 0)))

    method, arguments = provider.calls[0]
    assert method == "estimate_ragged_extend_forward_ms"
    assert arguments["reduction_policy"] == PREFIX_REDUCTION_POLICY
    assert arguments["execution_profile"].to_dict() == _GRAPH_PROFILE
    assert arguments["requests"] == (
        {"extend_len": 1, "prefix_len": 1024},
        {"extend_len": 127, "prefix_len": 0},
    )
    metrics = predictor.get_metrics()["infercast"]
    assert metrics["contract_version"] == 3
    assert metrics["execution_profile"] == _GRAPH_PROFILE


def test_v4_passes_realization_profile_and_exact_request_vector_once():
    provider = _Provider()
    predictor = _predictor(
        provider,
        contract_version=4,
        reduction_policy=REALIZATION_REDUCTION_POLICY,
    )

    latency = predictor.predict_infer_time(
        _batch("MIXED", (1, 8194), (127, 66))
    )

    assert latency == pytest.approx(0.0125)
    assert len(provider.calls) == 1
    method, arguments = provider.calls[0]
    assert method == "estimate_ragged_extend_forward_ms"
    assert arguments["reduction_policy"] == REALIZATION_REDUCTION_POLICY
    assert arguments["execution_profile"].to_dict() == _REALIZATION_PROFILE
    assert arguments["requests"] == (
        {"extend_len": 1, "prefix_len": 8194},
        {"extend_len": 127, "prefix_len": 66},
    )
    metrics = predictor.get_metrics()["infercast"]
    assert metrics["contract_version"] == 4
    assert metrics["execution_profile"] == _REALIZATION_PROFILE


def test_v5_passes_ordered_decode_histories_and_phase_specific_profile_once():
    provider = _Provider()
    predictor = _predictor(
        provider,
        contract_version=5,
        reduction_policy=PROFILED_DECODE_REDUCTION_POLICY,
    )

    latency = predictor.predict_infer_time(
        _batch("DECODE", (1, 127), (1, 1025), (1, 4097), (1, 8191))
    )

    assert latency == pytest.approx(0.0125)
    assert len(provider.calls) == 1
    method, arguments = provider.calls[0]
    assert method == "estimate_profiled_decode_forward_ms"
    assert arguments["history_lengths"] == (127, 1025, 4097, 8191)
    assert arguments["execution_profile"].to_dict() == _DECODE_PROFILE
    assert "attn_kernel_impl" not in arguments
    metrics = predictor.get_metrics()["infercast"]
    assert metrics["contract_version"] == 5
    assert metrics["reduction_policy"] == PROFILED_DECODE_REDUCTION_POLICY
    assert metrics["decode_execution_profile"] == _DECODE_PROFILE
    assert metrics["runtime"]["attn_kernel_impl"] == "cuda_graph"
    assert metrics["runtime"]["decode_attn_kernel_impl"] == "cuda_graph"


def test_v5_context_keeps_v4_context_contract():
    provider = _Provider()
    predictor = _predictor(provider, contract_version=5)

    predictor.predict_infer_time(_batch("MIXED", (1, 8194), (127, 66)))

    method, arguments = provider.calls[0]
    assert method == "estimate_ragged_extend_forward_ms"
    assert arguments["reduction_policy"] == REALIZATION_REDUCTION_POLICY
    assert arguments["execution_profile"].to_dict() == _REALIZATION_PROFILE


def test_v2_portable_vectors_preserve_mode_order_and_lengths():
    fixture = json.loads(
        (Path(__file__).parent / "fixtures" / "infercast_provider_v2.json").read_text(
            encoding="utf-8"
        )
    )
    assert fixture["schema_version"] == 2
    assert fixture["reduction_policy"] == RAGGED_REDUCTION_POLICY
    for case in fixture["cases"]:
        provider = _Provider()
        predictor = _predictor(provider, contract_version=2)
        requests = tuple(
            (request["extend_length"], request["past_kv_length"])
            for request in case["requests"]
        )

        predictor.predict_infer_time(_batch(case["forward_mode"], *requests))

        assert len(provider.calls) == 1, case["id"]
        method, arguments = provider.calls[0]
        assert method == "estimate_ragged_extend_forward_ms", case["id"]
        assert arguments["forward_mode"] == case["forward_mode"], case["id"]
        assert arguments["requests"] == tuple(
            {
                "extend_len": request["extend_length"],
                "prefix_len": request["past_kv_length"],
            }
            for request in case["requests"]
        ), case["id"]


@pytest.mark.parametrize(
    "version,policy",
    [
        (1, RAGGED_REDUCTION_POLICY),
        (2, "mean_attention_flops_v1"),
        (3, RAGGED_REDUCTION_POLICY),
        (6, None),
    ],
)
def test_contract_and_reduction_policy_must_match(version, policy):
    with pytest.raises(PredictorError) as exc_info:
        validate_provider_contract(version, policy)
    assert exc_info.value.code == "unsupported_reduction_policy"


def test_prefix_forward_reaches_provider_and_fails_without_time_accounting():
    StateManager.reset()
    provider = _Provider(error=NotImplementedError("prefix_len must be 0"))
    with pytest.raises(PredictorError) as exc_info:
        predict_schedule_batch(
            _predictor(provider),
            _batch("EXTEND", (256, 768)),
        )
    assert exc_info.value.code == "prediction_failed"
    assert provider.calls[0][1]["prefix_len"] == 768
    assert StateManager.get_iteration() == 0
    assert StateManager.get_global_clock() == 0


def test_provider_errors_keep_stable_categories():
    class MissingData(RuntimeError):
        code = "perf_data_not_available"
        details = {"op": "gemm"}

    with pytest.raises(PredictorError) as exc_info:
        _predictor(_Provider(error=MissingData())).predict_infer_time(
            _batch("DECODE", (1, 127))
        )
    assert exc_info.value.code == "data_unavailable"
    assert exc_info.value.details["infercast_code"] == "perf_data_not_available"


@pytest.mark.parametrize(
    "infercast_code",
    [
        "invalid_execution_profile",
        "unsupported_execution_profile",
        "outside_calibrated_domain",
    ],
)
def test_v3_profile_errors_keep_stable_categories(infercast_code):
    error_type = type(
        "ProfileError",
        (RuntimeError,),
        {"code": infercast_code, "details": {"profile": "test"}},
    )
    predictor = _predictor(
        _Provider(error=error_type("profile failure")),
        contract_version=3,
    )

    with pytest.raises(PredictorError) as exc_info:
        predictor.predict_infer_time(_batch("EXTEND", (64, 128)))

    assert exc_info.value.code == infercast_code


@pytest.mark.parametrize(
    "model,config",
    [
        (
            ModelInfo(torch_dtype="float16"),
            SchedulerConfig(backend_name="sglang", backend_version="0.5.17"),
        ),
        (
            ModelInfo(),
            SchedulerConfig(
                backend_name="sglang",
                backend_version="0.5.17",
                kv_cache_data_type=DataType.BF16,
            ),
        ),
    ],
)
def test_runtime_dtype_mismatch_fails_closed(model, config):
    with pytest.raises(PredictorError) as exc_info:
        _predictor(_Provider(), model=model, config=config)
    assert exc_info.value.code == "incompatible_runtime"


def test_provider_revision_is_exact():
    with pytest.raises(PredictorError) as exc_info:
        _predictor(_Provider(), revision="unknown")
    assert exc_info.value.code == "provider_initialization_failed"


def test_model_revision_is_exact():
    with pytest.raises(PredictorError) as exc_info:
        _predictor(_Provider(), model_revision="main")
    assert exc_info.value.code == "provider_initialization_failed"


def test_production_binding_and_portable_provenance(tmp_path, monkeypatch):
    stack = tmp_path / "stack.json"
    stack.write_bytes(b'{"stack":"test"}')
    database = type(
        "Database",
        (),
        {"slice_ref": type("SliceRef", (), {"stack_path": stack})()},
    )()
    opened = {}
    built = {}

    class PerfDatabase:
        @staticmethod
        def open_fidb(*args, **kwargs):
            opened["arguments"] = args
            opened["keywords"] = kwargs
            return database

    provider = _Provider()
    provider.desc = type("Description", (), {"geometry": {}})()
    sdk = ModuleType("infercast.sdk")
    sdk.PerfDatabase = PerfDatabase

    def build_umd_static_model(*args, **kwargs):
        built["arguments"] = args
        built["keywords"] = kwargs
        return provider

    sdk.build_umd_static_model = build_umd_static_model
    package = ModuleType("infercast")
    package.__path__ = []
    package.__version__ = "0.1.0"
    package.__revision__ = "a" * 40
    monkeypatch.setitem(sys.modules, "infercast", package)
    monkeypatch.setitem(sys.modules, "infercast.sdk", sdk)

    def make_predictor():
        return InferCastTimePredictor(
            ModelInfo(torch_dtype="bfloat16"),
            AcceleratorInfo(
                name="MI350X",
                vendor="AMD",
                hbm_capacity_gb=1,
                hbm_bandwidth_gb=1,
            ),
            SchedulerConfig(
                backend_name="sglang",
                backend_version="0.5.17",
                kv_cache_data_type=DataType.FP8,
            ),
            model_id="Qwen/Qwen3-32B-FP8",
            system="mi350x",
            systems_root=str(tmp_path / "systems"),
            attn_kernel_impl="cuda_graph",
            attn_dtype="bfloat16",
            kv_cache_dtype="fp8",
            model_revision="c" * 40,
            provider_revision="a" * 40,
        )

    predictor = make_predictor()
    metrics = predictor.get_metrics()["infercast"]
    assert opened["arguments"] == ("mi350x", "sglang", "0.5.17")
    assert built["keywords"]["revision"] == "c" * 40
    assert metrics["model_revision"] == "c" * 40
    assert metrics["stack_digest"] == hashlib.sha256(stack.read_bytes()).hexdigest()
    assert "systems_root" not in metrics
    assert "database_path" not in metrics

    del package.__revision__
    with pytest.raises(PredictorError) as exc_info:
        make_predictor()
    assert exc_info.value.code == "incompatible_runtime"


def test_loaded_model_geometry_must_match_provider():
    provider = _Provider()
    provider.desc = type("Description", (), {"geometry": {"hidden": 2}})()
    predictor = _predictor(provider)
    with pytest.raises(PredictorError) as exc_info:
        predictor._validate_model_geometry(ModelInfo(hidden_size=1))
    assert exc_info.value.code == "incompatible_runtime"


def test_metrics_reset_only_call_counters():
    predictor = _predictor(_Provider())
    predictor.predict_infer_time(_batch("MIXED", (1, 0), (8, 0)))
    before = predictor.get_metrics()["infercast"]
    predictor.reset_metrics()
    after = predictor.get_metrics()["infercast"]
    assert before["model_id"] == after["model_id"]
    assert before["stack_digest"] == after["stack_digest"]
    assert after["calls"] == {
        "total": 0,
        "context": 0,
        "mixed": 0,
        "decode": 0,
    }


def test_failed_prediction_does_not_increment_success_counters():
    predictor = _predictor(_Provider(error=RuntimeError("boom")))
    with pytest.raises(PredictorError):
        predictor.predict_infer_time(_batch("EXTEND", (1, 16)))
    assert predictor.get_metrics()["infercast"]["calls"] == {
        "total": 0,
        "context": 0,
        "mixed": 0,
        "decode": 0,
    }
