# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest

from sglang.multimodal_gen.configs.pipeline_configs.base import PipelineConfig
from sglang.multimodal_gen.configs.pipeline_configs.wan import WanT2V480PConfig
from sglang.multimodal_gen.runtime.server_args import ServerArgs
from sglang.multimodal_gen.runtime.weight_cache.client import (
    PROTOCOL,
    WeightCacheClient,
    validate_response,
)
from sglang.multimodal_gen.runtime.weight_cache.guards import (
    reject_cached_weight_mutation,
)
from sglang.multimodal_gen.runtime.weight_cache.plan import (
    CacheCompatibilityPlan,
    plan_diff,
)


def make_args(**overrides):
    with (
        patch.object(ServerArgs, "_adjust_network_ports"),
        patch.object(PipelineConfig, "from_kwargs", return_value=WanT2V480PConfig()),
        patch(
            "sglang.multimodal_gen.runtime.platforms.current_platform.is_cuda",
            return_value=True,
        ),
    ):
        return ServerArgs.from_dict(
            {
                "model_path": "Wan-AI/Wan2.1-T2V-1.3B-Diffusers",
                "weight_cache_mode": "client",
                "performance_mode": "manual",
                **overrides,
            }
        )


def test_cache_pins_resident_and_raw_off_variant_is_independent():
    args = make_args()
    assert args.residency_mode("transformer") == "resident"
    assert not args.should_use_fsdp_for_component("transformer")
    off = args.resolve_variant(weight_cache_mode="off")
    assert off.weight_cache_mode == "off"
    assert "transformer" not in off._required_resident_components
    assert "transformer" in args._required_resident_components


@pytest.mark.parametrize(
    "overrides",
    [
        {"component_residency": {"dit": "component_offload"}},
        {"dit_cpu_offload": True},
        {"dit_layerwise_offload": True},
        {"use_fsdp_inference": True},
        {"weight_cache_components": ["vae"]},
        {"num_gpus": 2},
        {"weight_cache_fallback": "disk"},
        {"weight_cache_timeout": 0},
        {"weight_cache_max_deliveries": -1},
        {"lora_path": "/missing/adapter"},
    ],
)
def test_admission_rejects_before_worker_launch(overrides):
    with pytest.raises(ValueError):
        make_args(**overrides)


def test_plan_is_immutable_and_diff_reports_nested_field():
    fields = {"component": {"dtype": "bf16"}}
    plan = CacheCompatibilityPlan.from_fields(**fields)
    original = plan.digest
    fields["component"]["dtype"] = "fp16"
    returned = plan.to_dict()
    returned["component"]["dtype"] = "fp32"
    assert plan.digest == original
    assert plan_diff(plan.to_dict(), returned) == {
        "component.dtype": {"expected": "bf16", "actual": "fp32"}
    }


@pytest.mark.parametrize(
    "response",
    [
        {},
        {**PROTOCOL, "family": "srt", "status": "ok"},
        {**PROTOCOL, "cache_abi": -1, "status": "ok"},
        {**PROTOCOL, "status": "error"},
    ],
)
def test_protocol_fails_closed(response):
    with pytest.raises((ValueError, RuntimeError)):
        validate_response(response)


def test_missing_socket_is_error_not_fallback(tmp_path):
    args = SimpleNamespace(
        weight_cache_socket=str(tmp_path / "missing.sock"), weight_cache_timeout=1
    )
    plan = CacheCompatibilityPlan.from_fields(rank={"device_uuid": "test"})
    with pytest.raises(FileNotFoundError):
        with WeightCacheClient(plan, args):
            pytest.fail("Missing socket admitted")


def test_cache_mutation_guard_checks_imported_state_even_if_args_changed():
    pipeline = SimpleNamespace(
        server_args=SimpleNamespace(weight_cache_mode="off"),
        modules={"transformer": SimpleNamespace(_weight_cache_importer=object())},
    )
    with pytest.raises(ValueError, match="shared"):
        reject_cached_weight_mutation(pipeline, "LoRA")


def test_preflight_failure_does_not_start_worker():
    from sglang.multimodal_gen.runtime import launch_server

    args = make_args()
    with (
        patch(
            "sglang.multimodal_gen.runtime.weight_cache.preflight.preflight",
            side_effect=RuntimeError("missing owner"),
        ),
        patch.object(launch_server.mp, "get_context") as context,
    ):
        with pytest.raises(RuntimeError, match="missing owner"):
            launch_server.launch_server(args, launch_http_server=False)
    context.assert_not_called()


def test_physical_gpu_compatibility_does_not_include_local_ordinal():
    from sglang.multimodal_gen.runtime.weight_cache import identity

    with (
        patch.object(identity, "checkpoint_identity", return_value={}),
        patch.object(identity, "environment_identity", return_value={}),
        patch.object(
            identity.current_platform, "get_device_uuid", return_value="same-gpu"
        ),
    ):
        prepared = SimpleNamespace(
            pipeline_cls=SimpleNamespace(__name__="WanPipeline"),
            transformer=None,
            adapter=SimpleNamespace(fingerprint_fields=Mock(return_value={})),
        )
        a = identity.compatibility_plan(
            prepared, SimpleNamespace(gpu_ids=None, base_gpu_id=0)
        )
        b = identity.compatibility_plan(
            prepared, SimpleNamespace(gpu_ids=None, base_gpu_id=1)
        )
    assert a == b


def test_manifest_query_does_not_export_or_register_a_consumer():
    from sglang.multimodal_gen.runtime.weight_cache.daemon import (
        DiffusionWeightCacheDaemon,
    )
    from sglang.weight_cache_common.liveness import ProcessIdentity
    from sglang.weight_cache_common.transport import ExportGeneration

    owner = object.__new__(DiffusionWeightCacheDaemon)
    owner.stopping = False
    owner.consumers = set()
    owner.plan = CacheCompatibilityPlan.from_fields(component={})
    owner.exporter = Mock()
    owner.exporter.stats.return_value = {"budget_exhausted": False}
    owner.exporter.generation = ExportGeneration(
        ProcessIdentity(123, 456), "nonce", "digest", "gpu", "torch"
    )
    owner.exporter.manifest.to_dict.return_value = {}
    request = {
        **PROTOCOL,
        "type": "query_manifest",
        "compatibility": owner.plan.to_dict(),
    }
    result = owner._request(request, Mock())
    assert result["compatibility"] == owner.plan.to_dict()
    assert owner.consumers == set()
    owner.exporter.export.assert_not_called()
    with pytest.raises(ValueError, match="compatibility"):
        owner._request({**request, "compatibility": {}}, Mock())
    owner.exporter.export.assert_not_called()


def test_explicit_argv_is_not_lost_when_sys_argv_is_empty():
    from sglang.multimodal_gen.runtime.server_args import prepare_server_args

    with (
        patch("sys.argv", ["program"]),
        patch.object(PipelineConfig, "from_kwargs", return_value=WanT2V480PConfig()),
        patch.object(ServerArgs, "_adjust_network_ports"),
        patch(
            "sglang.multimodal_gen.runtime.platforms.current_platform.is_cuda",
            return_value=True,
        ),
    ):
        args = prepare_server_args(
            [
                "--model-path",
                "/fake",
                "--weight-cache-mode",
                "client",
                "--performance-mode",
                "manual",
            ]
        )
    assert args.model_path == "/fake"
    assert args.weight_cache_mode == "client"


@pytest.mark.parametrize(
    "overrides",
    [
        {"dit_cpu_offload": True},
        {"dit_layerwise_offload": True},
        {"use_fsdp_inference": True},
        {"cpu_offload_components": ["dit"]},
        {"layerwise_offload_components": ["dit"]},
        {"layerwise_offload_components": ["all"]},
        {"component_residency": {"dit": "resident"}, "dit_cpu_offload": True},
        {
            "component_residency": {"dit": "resident"},
            "layerwise_offload_components": ["dit"],
        },
    ],
)
def test_direct_dataclass_conflicts_cannot_be_erased_by_cache_pins(overrides):
    with pytest.raises(ValueError, match="Weight cache"):
        ServerArgs(
            model_path="/fake",
            pipeline_config=WanT2V480PConfig(),
            weight_cache_mode="client",
            performance_mode="manual",
            **overrides,
        )


def test_daemon_stop_interrupts_idle_control_connection():
    import socket

    from sglang.multimodal_gen.runtime.weight_cache.daemon import (
        DiffusionWeightCacheDaemon,
    )

    owner = object.__new__(DiffusionWeightCacheDaemon)
    left, right = socket.socketpair()
    try:
        owner._connection = left
        owner.stopping = False
        owner.stop()
        assert owner.stopping
        assert right.recv(1) == b""
    finally:
        left.close()
        right.close()


def test_explicit_environment_capability_does_not_probe_current_cuda_device():
    from sglang.srt.platforms import current_platform as srt_platform
    from sglang.srt.weight_cache.protocol import compute_env_stamp

    with patch.object(srt_platform, "get_device_capability") as capability:
        assert compute_env_stamp(device_capability="9.0")["device_capability"] == "9.0"
    capability.assert_not_called()


@pytest.fixture
def prepared_wan(tmp_path):
    import json

    from safetensors.torch import save_file

    from sglang.multimodal_gen.runtime.models.registry import ModelRegistry
    from sglang.multimodal_gen.runtime.pipelines.wan_pipeline import WanPipeline
    from sglang.multimodal_gen.runtime.pipelines_core.prepare import prepare_pipeline
    from sglang.multimodal_gen.runtime.weight_cache.adapters.dit_wan import (
        EXPECTED_CONFIG,
    )

    index = {
        "_class_name": "WanPipeline",
        "transformer": ["diffusers", "WanTransformer3DModel"],
        "text_encoder": ["transformers", "UMT5EncoderModel"],
        "tokenizer": ["transformers", "T5TokenizerFast"],
        "vae": ["diffusers", "AutoencoderKLWan"],
        "scheduler": ["diffusers", "UniPCMultistepScheduler"],
    }
    (tmp_path / "model_index.json").write_text(json.dumps(index))
    component = tmp_path / "transformer"
    component.mkdir()
    (component / "config.json").write_text(
        json.dumps({"_class_name": "WanTransformer3DModel", **EXPECTED_CONFIG})
    )
    # Valid metadata but no tensors: preparation may inspect quantization headers,
    # but cannot construct or populate the model from this checkpoint.
    save_file({}, component / "diffusion_pytorch_model.safetensors")
    # Existing lazy kernel imports probe CUDA (e.g. norm_triton autotuning).
    # This test isolates preparation after imports, not fresh-process purity.
    ModelRegistry.resolve_model_cls("WanTransformer3DModel")
    args = make_args(model_path=str(tmp_path))
    with (
        patch(
            "sglang.multimodal_gen.runtime.pipelines_core.prepare.maybe_download_model",
            return_value=str(tmp_path),
        ),
        patch(
            "sglang.multimodal_gen.runtime.platforms.current_platform.is_cuda",
            return_value=True,
        ),
    ):
        yield args, WanPipeline, prepare_pipeline


def test_post_import_preparation_constructs_no_modules_or_cuda_state(prepared_wan):
    import torch

    from sglang.multimodal_gen.runtime.loader.component_loaders import (
        transformer_loader,
    )

    args, pipeline, prepare = prepared_wan
    with (
        patch.object(torch.nn.Module, "__init__", side_effect=AssertionError("module")),
        patch.object(torch.cuda, "_lazy_init", side_effect=AssertionError("CUDA")),
        patch.object(
            transformer_loader,
            "get_local_torch_device",
            side_effect=AssertionError("live rank"),
        ),
    ):
        prepared = prepare(pipeline, args, required=True)
    assert prepared is not None
    assert len(prepared.specs) == 5
    assert args.model_paths == {}


def test_uncached_placement_changes_execution_not_cache_fingerprint(prepared_wan):
    from sglang.multimodal_gen.runtime.weight_cache.adapters.dit_wan import (
        fingerprint_fields,
    )

    args, pipeline, prepare = prepared_wan
    a = prepare(pipeline, args, required=True)
    other = make_args(
        model_path=args.model_path, component_residency={"vae": "component_offload"}
    )
    b = prepare(pipeline, other, required=True)
    assert a.execution_plan != b.execution_plan
    assert fingerprint_fields(a.transformer) == fingerprint_fields(b.transformer)
    ordinary = prepare(pipeline, args.resolve_variant(weight_cache_mode="off"))
    assert fingerprint_fields(a.transformer) == fingerprint_fields(ordinary.transformer)
    assert a.specs == ordinary.specs


@pytest.mark.parametrize(
    "variant",
    [
        "same_named_class",
        "config",
        "quant",
        "dtype",
        "attention",
        "attention_config",
        "fsdp",
        "tp",
    ],
)
def test_resolved_adapter_rejects_unverified_variants(prepared_wan, variant):
    import torch

    from sglang.multimodal_gen.runtime.weight_cache.adapters.dit_wan import (
        validate_supported,
    )

    args, pipeline, prepare = prepared_wan
    recipe = prepare(pipeline, args, required=True).transformer.thaw()
    attention = "fa"
    if variant == "same_named_class":
        recipe.model_cls = type("WanTransformer3DModel", (), {})
    elif variant == "config":
        recipe.init_params["hf_config"]["unknown_layout"] = True
    elif variant == "quant":
        recipe.quant_spec.post_load_hooks.append(object())
    elif variant == "dtype":
        from dataclasses import replace

        recipe.quant_spec = replace(recipe.quant_spec, param_dtype=torch.float16)
    elif variant == "attention":
        attention = "torch_sdpa"
    elif variant == "attention_config":
        recipe.server_args.attention_backend_config = {"custom": True}
    elif variant == "fsdp":
        recipe.component_starts_on_cpu = True
    elif variant == "tp":
        recipe.server_args.tp_size = 2
    frozen = Mock()
    frozen.thaw.return_value = recipe
    with pytest.raises(ValueError):
        validate_supported(frozen, pipeline_name="WanPipeline", attention=attention)


def test_custom_loader_is_not_silently_bypassed(prepared_wan):
    args, pipeline, prepare = prepared_wan
    with patch.object(pipeline, "component_loaders", {"transformer": object}):
        with pytest.raises(ValueError, match="custom transformer loader"):
            prepare(pipeline, args, required=True)
        assert prepare(pipeline, args.resolve_variant(weight_cache_mode="off")) is None
