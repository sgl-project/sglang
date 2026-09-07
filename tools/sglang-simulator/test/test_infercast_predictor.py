import hashlib
import math
import sys
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
    InferCastTimePredictor,
    milliseconds_to_seconds,
    reduce_batch,
    validate_topology,
)


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

    def estimate_extend_forward_ms(self, **arguments):
        return self._call("estimate_extend_forward_ms", arguments)


def _predictor(provider, *, model=None, config=None, revision="a" * 40):
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
        attn_dtype="bfloat16",
        kv_cache_dtype="fp8",
        provider_revision=revision,
        _provider=provider,
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


def test_production_binding_and_portable_provenance(tmp_path, monkeypatch):
    stack = tmp_path / "stack.json"
    stack.write_bytes(b'{"stack":"test"}')
    database = type(
        "Database",
        (),
        {"slice_ref": type("SliceRef", (), {"stack_path": stack})()},
    )()
    opened = {}

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
    sdk.build_umd_static_model = lambda *args, **kwargs: provider
    package = ModuleType("infercast")
    package.__path__ = []
    package.__version__ = "0.1.0"
    package.__revision__ = "a" * 40
    monkeypatch.setitem(sys.modules, "infercast", package)
    monkeypatch.setitem(sys.modules, "infercast.sdk", sdk)

    predictor = InferCastTimePredictor(
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
        provider_revision="a" * 40,
    )
    metrics = predictor.get_metrics()["infercast"]
    assert opened["arguments"] == ("mi350x", "sglang", "0.5.17")
    assert metrics["stack_digest"] == hashlib.sha256(stack.read_bytes()).hexdigest()
    assert "systems_root" not in metrics
    assert "database_path" not in metrics


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
