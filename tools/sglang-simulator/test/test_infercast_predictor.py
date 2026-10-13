import copy
import hashlib
import json
import math
import sys
from dataclasses import replace
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest
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
    PROVIDER_CONTRACT,
    InferCastTimePredictor,
    _validate_batch,
    milliseconds_to_seconds,
    validate_topology,
)

try:
    from infercast.sdk import (
        DecodeExecutionProfile,
        ExtendExecutionProfile,
        ExtendRequestShape,
    )
except ImportError:
    DecodeExecutionProfile = ExtendExecutionProfile = ExtendRequestShape = None

_MODEL_ID = "deepseek-ai/DeepSeek-R1"
_FIXTURE = json.loads(
    (Path(__file__).parent / "fixtures" / "infercast_provider.json").read_text(
        encoding="utf-8"
    )
)
_REALIZATION_PROFILE = _FIXTURE["execution_profile"]
_DECODE_PROFILE = _FIXTURE["decode_execution_profile"]


class _ExecutionProfile:
    def __init__(self, payload):
        self.payload = dict(payload)

    @classmethod
    def from_dict(cls, payload):
        if not isinstance(payload, dict):
            raise TypeError("execution profile must be a mapping")
        return cls(payload)

    def validate_for_kernel_impl(self, kernel_impl):
        if kernel_impl != "cuda_graph":
            raise ValueError("profile requires cuda_graph")

    @property
    def kv_page_size(self):
        return self.payload["kv_page_size"]

    def to_dict(self):
        return dict(self.payload)


class _DecodeExecutionProfile(_ExecutionProfile):
    @property
    def kernel_impl(self):
        return (
            "eager"
            if self.payload["decode_graph_backend"] == "disabled"
            else "cuda_graph"
        )


def _batch(mode, *requests):
    return ScheduleBatch(
        [ScheduleRequest(extend, prefix) for extend, prefix in requests],
        forward_mode=mode,
    )


@pytest.mark.parametrize(
    "batch,code",
    [
        (_batch("DECODE"), "invalid_batch"),
        (_batch("EXTEND", (0, 0)), "invalid_batch"),
        (_batch("EXTEND", (1, -1)), "invalid_batch"),
        (_batch("EXTEND", (True, 0)), "invalid_batch"),
        (_batch("EXTEND", (1, 0.5)), "invalid_batch"),
        (_batch("DECODE", (2, 10)), "invalid_batch"),
        (_batch("DECODE", (1, 0)), "invalid_batch"),
        (_batch("TARGET_VERIFY", (1, 10)), "unsupported_forward_mode"),
        (_batch("FUTURE_MODE", (1, 10)), "unsupported_forward_mode"),
    ],
)
def test_provider_rejects_invalid_batches(batch, code):
    with pytest.raises(PredictorError) as exc_info:
        _validate_batch(batch)
    assert exc_info.value.code == code


@pytest.mark.parametrize("value", [True, False, 0, -1, math.nan, math.inf, "invalid"])
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
        self.desc = type(
            "Description",
            (),
            {
                "geometry": {
                    "hidden_size": 7168,
                    "num_layers": 61,
                    "num_mla_heads": 128,
                    "kv_lora_rank": 512,
                    "qk_rope_head_dim": 64,
                }
            },
        )()

    def _call(self, method, arguments):
        self.calls.append((method, arguments))
        if self.error:
            raise self.error
        return self.value

    def estimate_profiled_decode_forward_ms(self, **arguments):
        return self._call("estimate_profiled_decode_forward_ms", arguments)

    def estimate_ragged_extend_forward_ms(self, **arguments):
        return self._call("estimate_ragged_extend_forward_ms", arguments)


@pytest.fixture(autouse=True)
def sdk_double(tmp_path, monkeypatch):
    """Patch public SDK entry points; production construction stays unchanged."""
    stack = tmp_path / "stack.json"
    stack.write_bytes(b'{"stack":"test"}')
    module = ModuleType("infercast.sdk")
    module.database = SimpleNamespace(slice_ref=SimpleNamespace(stack_path=stack))
    module.opened, module.built = [], []

    def open_fidb(*args, **kwargs):
        module.opened.append((args, kwargs))
        return module.database

    def build(*args, **kwargs):
        module.built.append((args, kwargs))
        return module.provider

    module.PerfDatabase = SimpleNamespace(open_fidb=open_fidb)
    module.build_umd_static_model = build
    module.ExtendRequestShape = lambda **values: values
    module.ExtendExecutionProfile = _ExecutionProfile
    module.DecodeExecutionProfile = _DecodeExecutionProfile
    package = ModuleType("infercast")
    package.__path__ = []
    package.__version__, package.__revision__ = "0.1.0", "a" * 40
    monkeypatch.setitem(sys.modules, "infercast", package)
    monkeypatch.setitem(sys.modules, "infercast.sdk", module)
    return module


def _predictor(
    provider,
    *,
    model=None,
    config=None,
    model_revision="c" * 40,
    revision="a" * 40,
    **extra,
):
    sys.modules["infercast.sdk"].provider = provider
    options = {
        "model_id": _MODEL_ID,
        "system": "mi350x",
        "systems_root": "/unused",
        "attn_kernel_impl": "cuda_graph",
        "attn_dtype": "bfloat16",
        "kv_cache_dtype": "fp8",
        "model_revision": model_revision,
        "provider_revision": revision,
        "execution_profile": _REALIZATION_PROFILE,
        "decode_execution_profile": _DECODE_PROFILE,
    }
    options.update(extra)
    return InferCastTimePredictor(
        model
        or ModelInfo(
            attention_arch="MLA",
            hidden_size=7168,
            num_hidden_layers=61,
            num_attention_heads=128,
            kv_lora_rank=512,
            qk_rope_head_dim=64,
            torch_dtype="bfloat16",
        ),
        AcceleratorInfo(
            name="MI350X",
            vendor="AMD",
            hbm_capacity_gb=1,
            hbm_bandwidth_gb=1,
        ),
        config or SchedulerConfig(backend_name="sglang", backend_version="0.5.17"),
        **options,
    )


@pytest.mark.parametrize("field", ["contract_version", "reduction_policy", "_provider"])
def test_predictor_rejects_unknown_configuration_fields(field):
    with pytest.raises(PredictorError) as exc_info:
        _predictor(_Provider(), **{field: "obsolete"})

    assert exc_info.value.code == "provider_initialization_failed"
    assert exc_info.value.details["unsupported_fields"] == [field]


@pytest.mark.parametrize("model_id", ["Qwen/Qwen3-32B-FP8", "deepseek-ai/DeepSeek-V3"])
def test_unsupported_model_rejected_before_provider_use(model_id):
    provider = _Provider()
    with pytest.raises(PredictorError) as exc_info:
        _predictor(provider, model_id=model_id)
    assert exc_info.value.code == "unsupported_model"
    assert exc_info.value.details["supported_models"] == [_MODEL_ID]
    assert provider.calls == []


@pytest.mark.parametrize(
    "field,code",
    [
        ("execution_profile", "invalid_execution_profile"),
        ("decode_execution_profile", "invalid_decode_execution_profile"),
    ],
)
def test_execution_profiles_are_required(field, code):
    with pytest.raises(PredictorError) as exc_info:
        _predictor(_Provider(), **{field: None})
    assert exc_info.value.code == code


@pytest.mark.parametrize(
    "field,code",
    [
        ("execution_profile", "invalid_execution_profile"),
        ("decode_execution_profile", "invalid_decode_execution_profile"),
    ],
)
def test_profile_page_size_must_match_known_scheduler_page_size(field, code):
    payload = copy.deepcopy(
        _REALIZATION_PROFILE if field == "execution_profile" else _DECODE_PROFILE
    )
    payload["kv_page_size"] = 64
    with pytest.raises(PredictorError, match="scheduler page_size=1") as exc_info:
        _predictor(
            _Provider(),
            config=SchedulerConfig(page_size=1, backend_version="0.5.17"),
            **{field: payload},
        )
    assert exc_info.value.code == code


def test_decode_passes_ordered_histories_and_profile_once():
    provider = _Provider()
    predictor = _predictor(
        provider,
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
    assert metrics["contract"] == PROVIDER_CONTRACT
    assert metrics["decode_execution_profile"] == _DECODE_PROFILE
    assert metrics["runtime"]["attn_kernel_impl"] == "cuda_graph"
    assert metrics["runtime"]["decode_attn_kernel_impl"] == "cuda_graph"


def test_portable_vectors_preserve_mode_order_and_lengths():
    fixture = json.loads(
        (Path(__file__).parent / "fixtures" / "infercast_provider.json").read_text(
            encoding="utf-8"
        )
    )
    assert fixture["schema_version"] == 1
    assert fixture["contract"] == PROVIDER_CONTRACT
    for case in fixture["cases"]:
        provider = _Provider()
        predictor = _predictor(provider)
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
    "infercast_code,expected_code",
    [
        ("perf_data_not_available", "data_unavailable"),
        ("prefix_prediction_not_supported", "unsupported_model"),
        ("invalid_execution_profile", "invalid_execution_profile"),
        ("unsupported_execution_profile", "unsupported_execution_profile"),
        ("outside_calibrated_domain", "outside_calibrated_domain"),
        ("invalid_decode_execution_profile", "invalid_decode_execution_profile"),
        (
            "unsupported_decode_execution_profile",
            "unsupported_decode_execution_profile",
        ),
        ("decode_prediction_outside_domain", "decode_prediction_outside_domain"),
        ("decode_component_realization_mismatch", "data_unavailable"),
        (None, "prediction_failed"),
    ],
)
def test_provider_errors_preserve_details_without_counting_success(
    infercast_code,
    expected_code,
):
    error = RuntimeError("missing realization")
    error.code, error.details = infercast_code, {"op": "mla"}
    predictor = _predictor(_Provider(error=error))
    with pytest.raises(PredictorError) as exc_info:
        predictor.predict_infer_time(_batch("DECODE", (1, 128)))
    assert exc_info.value.code == expected_code
    assert exc_info.value.details["infercast_code"] == infercast_code
    assert exc_info.value.details["infercast_details"] == {"op": "mla"}
    assert predictor.get_metrics()["infercast"]["calls"]["total"] == 0


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


def test_production_binding_and_portable_provenance(sdk_double):
    predictor = _predictor(_Provider())
    assert predictor.requires_prepared_batch
    assert sdk_double.opened[0][0] == ("mi350x", "sglang", "0.5.17")
    assert sdk_double.opened[0][1]["database_mode"] == "SILICON"
    assert sdk_double.built[0][0] == (sdk_double.database, _MODEL_ID)
    assert sdk_double.built[0][1] == {"backend": "sglang", "revision": "c" * 40}
    metrics = predictor.get_metrics()["infercast"]
    stack = sdk_double.database.slice_ref.stack_path
    assert metrics["model_revision"] == "c" * 40
    assert metrics["provider_revision"] == "a" * 40
    assert metrics["stack_digest"] == hashlib.sha256(stack.read_bytes()).hexdigest()
    assert "systems_root" not in metrics and "database_path" not in metrics


@pytest.mark.parametrize("actual_revision", [None, "main", "b" * 40])
def test_installed_revision_must_match_before_database_access(
    sdk_double, actual_revision
):
    sys.modules["infercast"].__revision__ = actual_revision
    with pytest.raises(PredictorError) as exc_info:
        _predictor(_Provider())
    assert exc_info.value.code == "incompatible_runtime"
    assert sdk_double.opened == [] and sdk_double.built == []


@pytest.mark.parametrize(
    "field",
    ["hidden_size", "num_layers", "num_mla_heads", "kv_lora_rank", "qk_rope_head_dim"],
)
@pytest.mark.parametrize("mutation", ["different", "missing"])
def test_model_geometry_matches_including_mla_kv_layout(field, mutation):
    provider = _Provider()
    if mutation == "different":
        provider.desc.geometry[field] += 1
    else:
        del provider.desc.geometry[field]
    with pytest.raises(PredictorError) as exc_info:
        _predictor(provider)
    assert exc_info.value.code == "incompatible_runtime"
    assert provider.calls == []


@pytest.mark.parametrize(
    "field,value",
    [
        ("attention_arch", "MHA"),
        ("attention_arch", None),
        ("hidden_size", None),
        ("num_hidden_layers", None),
        ("num_attention_heads", None),
        ("kv_lora_rank", None),
        ("qk_rope_head_dim", None),
    ],
)
def test_loaded_model_requires_complete_mla_geometry(field, value):
    model = ModelInfo(
        attention_arch="MLA",
        hidden_size=7168,
        num_hidden_layers=61,
        num_attention_heads=128,
        kv_lora_rank=512,
        qk_rope_head_dim=64,
        torch_dtype="bfloat16",
    )
    setattr(model, field, value)
    provider = _Provider()
    with pytest.raises(PredictorError) as exc_info:
        _predictor(provider, model=model)
    assert exc_info.value.code == "incompatible_runtime"
    assert exc_info.value.details["mismatches"][field]["sglang"] == value
    assert provider.calls == []


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


@pytest.mark.parametrize("value", [True, 0, -1, math.nan, math.inf])
def test_invalid_provider_output_does_not_increment_success_counters(value):
    predictor = _predictor(_Provider(value=value))
    with pytest.raises(PredictorError) as exc_info:
        predictor.predict_infer_time(_batch("EXTEND", (1, 0)))
    assert exc_info.value.code == "invalid_provider_output"
    assert predictor.get_metrics()["infercast"]["calls"]["total"] == 0


@pytest.fixture
def canonical_predictor(sdk_double):
    if ExtendRequestShape is None:
        pytest.skip("optional InferCast contract tests")
    sdk_double.ExtendRequestShape = ExtendRequestShape
    sdk_double.ExtendExecutionProfile = ExtendExecutionProfile
    sdk_double.DecodeExecutionProfile = DecodeExecutionProfile
    return _predictor


@pytest.mark.parametrize("mode", ["EXTEND", "MIXED"])
def test_real_sdk_preserves_ordered_context_and_exact_profiles(
    canonical_predictor, mode
):
    provider = _Provider()
    predictor = canonical_predictor(
        provider,
        config=SchedulerConfig(
            tp_size=8, ep_size=2, page_size=1, backend_version="0.5.17"
        ),
    )
    predictor.predict_infer_time(_batch(mode, (1, 8194), (127, 66)))

    assert len(provider.calls) == 1
    method, arguments = provider.calls[0]
    assert method == "estimate_ragged_extend_forward_ms"
    assert tuple(shape.to_dict() for shape in arguments["requests"]) == (
        {"extend_len": 1, "prefix_len": 8194},
        {"extend_len": 127, "prefix_len": 66},
    )
    assert arguments["execution_profile"].to_dict() == _REALIZATION_PROFILE
    assert arguments["tp_size"] == 8
    assert arguments["moe_tp_size"] == 4
    assert arguments["moe_ep_size"] == 2
    metrics = predictor.get_metrics()["infercast"]
    assert metrics["execution_profile"] == _REALIZATION_PROFILE
    assert metrics["decode_execution_profile"] == _DECODE_PROFILE


def test_real_sdk_decode_profile_is_independent_of_context(canonical_predictor):
    eager = DecodeExecutionProfile(
        decode_graph_backend="disabled",
        graph_capture_batches=(),
        graph_batch_policy="disabled",
        fallback_policy="reject",
        kv_page_size=1,
        runtime_compatibility_id="test-eager-decode",
        full_forward_realization="test-whole-forward-eager",
        component_realization="test-mla-eager",
        runtime_compatibility_manifest_sha256="e" * 64,
    )
    provider = _Provider()
    predictor = canonical_predictor(provider, decode_execution_profile=eager.to_dict())
    predictor.predict_infer_time(_batch("DECODE", (1, 513), (1, 511)))

    assert len(provider.calls) == 1
    method, arguments = provider.calls[0]
    assert method == "estimate_profiled_decode_forward_ms"
    assert arguments["history_lengths"] == (513, 511)
    assert arguments["execution_profile"] == eager
    assert "attn_kernel_impl" not in arguments
    metrics = predictor.get_metrics()["infercast"]
    assert metrics["runtime"]["attn_kernel_impl"] == "cuda_graph"
    assert metrics["runtime"]["decode_attn_kernel_impl"] == "eager"


@pytest.mark.parametrize(
    "field,code",
    [
        ("execution_profile", "invalid_execution_profile"),
        ("decode_execution_profile", "invalid_decode_execution_profile"),
    ],
)
@pytest.mark.parametrize("mutation", ["digest", "unknown_field", "missing_field"])
def test_real_sdk_rejects_tampered_or_malformed_profiles(
    canonical_predictor, field, code, mutation
):
    payload = copy.deepcopy(
        _REALIZATION_PROFILE if field == "execution_profile" else _DECODE_PROFILE
    )
    if mutation == "digest":
        payload["kv_page_size"] = 64
    elif mutation == "unknown_field":
        payload["reduction_policy"] = "obsolete"
    else:
        del payload["runtime_compatibility_manifest_sha256"]
    provider = _Provider()
    with pytest.raises(PredictorError) as exc_info:
        canonical_predictor(provider, **{field: payload})
    assert exc_info.value.code == code
    assert provider.calls == []


def test_real_sdk_rejects_conflicting_decode_implementation(canonical_predictor):
    with pytest.raises(PredictorError) as exc_info:
        canonical_predictor(_Provider(), decode_attn_kernel_impl="eager")
    assert exc_info.value.code == "invalid_decode_execution_profile"


@pytest.mark.parametrize(
    "field,profile_class,code",
    [
        ("execution_profile", "ExtendExecutionProfile", "invalid_execution_profile"),
        (
            "decode_execution_profile",
            "DecodeExecutionProfile",
            "invalid_decode_execution_profile",
        ),
    ],
)
def test_real_sdk_rejects_valid_profile_for_another_scheduler_page_size(
    canonical_predictor, field, profile_class, code
):
    payload = _REALIZATION_PROFILE if field == "execution_profile" else _DECODE_PROFILE
    profile = replace(globals()[profile_class].from_dict(payload), kv_page_size=64)
    provider = _Provider()
    with pytest.raises(PredictorError, match="scheduler page_size=1") as exc_info:
        canonical_predictor(
            provider,
            config=SchedulerConfig(page_size=1, backend_version="0.5.17"),
            **{field: profile.to_dict()},
        )
    assert exc_info.value.code == code
    assert provider.calls == []
