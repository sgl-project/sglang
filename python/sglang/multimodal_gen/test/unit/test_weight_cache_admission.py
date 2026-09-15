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
        patch.object(identity, "fingerprint_fields", return_value={}),
        patch.object(identity, "checkpoint_identity", return_value={}),
        patch.object(identity, "environment_identity", return_value={}),
        patch.object(
            identity.current_platform, "get_device_uuid", return_value="same-gpu"
        ),
    ):
        prepared = SimpleNamespace(
            pipeline_cls=SimpleNamespace(__name__="WanPipeline"), transformer=None
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
