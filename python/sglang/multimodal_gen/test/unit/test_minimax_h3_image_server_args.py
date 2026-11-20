# SPDX-License-Identifier: Apache-2.0
"""Real ServerArgs normalization with simulated CUDA capacity, never GPU execution."""

import pytest

from sglang.multimodal_gen.configs.pipeline_configs import (
    minimax_h3_image as image_config,
)
from sglang.multimodal_gen.configs.pipeline_configs.minimax_h3 import (
    MiniMaxH3PipelineConfig,
)
from sglang.multimodal_gen.configs.pipeline_configs.minimax_h3_image import (
    MiniMaxH3ImagePipelineConfig,
)
from sglang.multimodal_gen.configs.pipeline_configs.model_deployment_config import (
    ModelDeploymentConfig,
)
from sglang.multimodal_gen.configs.task_type import ModelTaskType
from sglang.multimodal_gen.runtime.platforms.interface import Platform, PlatformEnum
from sglang.multimodal_gen.runtime.server_args import ServerArgs
from sglang.multimodal_gen.runtime.server_args import auto_tune as auto_tune_module
from sglang.multimodal_gen.runtime.server_args import server_args as args_module
from sglang.multimodal_gen.test.unit.test_minimax_h3_image_pipeline_config import (
    write_model_dir,
)


class SimulatedCUDA(Platform):
    _enum = PlatformEnum.CUDA
    device_name = "CPU-test-simulated-CUDA"
    device_type = "cuda"

    def __init__(self, memory_gib, shared_memory=False):
        self.memory_gib = memory_gib
        self.shared_memory = shared_memory

    def get_device_total_memory(self):
        return int(self.memory_gib * 1024**3)

    def get_available_gpu_memory(self, device_id=0, empty_cache=False):
        return self.memory_gib

    def device_shares_host_memory(self):
        return self.shared_memory


@pytest.fixture
def make_server(monkeypatch, tmp_path):
    def make(
        memory_gib=80, *, config=None, device_names=None, shared_memory=False, **kwargs
    ):
        platform = SimulatedCUDA(memory_gib, shared_memory=shared_memory)
        monkeypatch.setattr(args_module, "current_platform", platform)
        monkeypatch.setattr(auto_tune_module, "current_platform", platform)
        monkeypatch.setattr(image_config, "current_platform", platform)
        monkeypatch.setattr(
            image_config.torch.cuda,
            "get_device_name",
            lambda index: device_names[index] if device_names else "Simulated CUDA",
        )
        return ServerArgs(
            model_path=str(tmp_path / "not-a-model-no-weights"),
            pipeline_config=MiniMaxH3ImagePipelineConfig()
            if config is None
            else config,
            log_level="error",
            output_path=str(tmp_path / "outputs"),
            input_save_path=str(tmp_path / "inputs"),
            **kwargs,
        )

    return make


@pytest.mark.parametrize(
    "device_name,memory_gib",
    [
        ("NVIDIA B300 SXM6 AC", 268),
        ("NVIDIA H200", 140),
        ("NVIDIA RTX PRO 6000 Blackwell Server Edition", 95),
    ],
)
def test_online_adaln_default_and_explicit_overrides(
    make_server, device_name, memory_gib
):
    hardware = {"memory_gib": memory_gib, "device_names": [device_name]}
    args = make_server(**hardware)
    assert args.minimax_h3_adaln_online and args.minimax_h3_adaln_plan_width == 2
    disabled = make_server(**hardware, minimax_h3_adaln_online=False)
    assert not disabled.minimax_h3_adaln_online
    wider = make_server(**hardware, minimax_h3_adaln_plan_width=4)
    assert wider.minimax_h3_adaln_online and wider.minimax_h3_adaln_plan_width == 4
    other = make_server(device_names=["NVIDIA H100"])
    assert not other.minimax_h3_adaln_online
    ada = make_server(48, device_names=["NVIDIA RTX 6000 Ada Generation"])
    assert not ada.minimax_h3_adaln_online
    mixed = make_server(num_gpus=2, device_names=["NVIDIA B300 SXM6 AC", "NVIDIA H200"])
    assert not mixed.minimax_h3_adaln_online


def test_online_adaln_is_explicitly_available_on_other_devices(make_server):
    args = make_server(minimax_h3_adaln_online=True)
    assert args.minimax_h3_adaln_online and args.minimax_h3_adaln_plan_width == 2


@pytest.mark.parametrize(
    "overrides",
    [
        {"enable_breakable_cuda_graph": True},
        {"enable_torch_compile": True},
        {"enable_torch_compile": True, "regional_compile": True},
    ],
)
def test_image_admits_native_graph_and_compile_runners(make_server, overrides):
    args = make_server(component_residency="all=resident", **overrides)
    for key, value in overrides.items():
        assert getattr(args, key) == value


@pytest.mark.parametrize("method", ["fp8", "mxfp8"])
@pytest.mark.parametrize("component", [None, "transformer", "text_encoder"])
def test_image_online_quantization_preserves_native_precision_defaults(
    make_server, method, component
):
    override = (
        {"quantization": method}
        if component is None
        else {"component_quantizations": {component: method}}
    )
    args = make_server(minimax_h3_adaln_online=True, **override)
    assert args.minimax_h3_adaln_online
    assert args.pipeline_config.vae_precision == "fp32"
    assert args.pipeline_config.vae_decode_precision == "bf16"
    assert args.quantization == (method if component is None else None)
    assert args.component_quantizations == (
        {} if component is None else {component: method}
    )


def test_gb10_residency_requires_available_shared_memory_and_online_adaln(make_server):
    hardware = dict(device_names=["NVIDIA GB10"], shared_memory=True)
    resident = make_server(109, **hardware)
    assert resident.minimax_h3_adaln_online
    assert not resident.dit_cpu_offload and not resident.text_encoder_cpu_offload
    below_threshold = make_server(105, **hardware)
    assert below_threshold.text_encoder_cpu_offload
    with pytest.raises(ValueError, match="resident GB10 requires online AdaLN"):
        make_server(109, **hardware, minimax_h3_adaln_online=False)
    offload = make_server(109, **hardware, component_residency="all=component-offload")
    assert offload.dit_cpu_offload and offload.text_encoder_cpu_offload
    layerwise = make_server(
        109,
        **hardware,
        component_residency="all=layerwise-offload",
        minimax_h3_adaln_online=False,
    )
    assert layerwise.residency_mode("transformer") == "layerwise-offload"
    discrete = make_server(109, device_names=["NVIDIA GB10"])
    assert discrete.text_encoder_cpu_offload


@pytest.mark.parametrize("tasks", [["t2i"], ["t2i", "edit"]])
@pytest.mark.parametrize("spare_slots", [-1, 0, 1])
@pytest.mark.parametrize("host_gb", [None, 0.0, 8.0])
def test_host_tier_default_requires_capacity_for_the_whole_release(
    make_server, monkeypatch, tmp_path, tasks, spare_slots, host_gb
):
    """A fixed grid has at most steps × tasks plans, independent of request shape."""
    model_path = write_model_dir(tmp_path / "release", tasks)
    config = MiniMaxH3ImagePipelineConfig(model_path=model_path)
    capacity = config.release_pdd_steps() * len(tasks) + spare_slots
    monkeypatch.setenv("SGLANG_DIFFUSION_MINIMAX_H3_ADALN_GPU_PLANS", str(capacity))
    args = make_server(
        config=config,
        minimax_h3_adaln_online=True,
        minimax_h3_adaln_host_cache_gb=host_gb,
    )
    expected = 0.0 if host_gb is None and spare_slots >= 0 else host_gb
    assert args.minimax_h3_adaln_host_cache_gb == expected


def test_edit_online_adaln_rejects_narrow_plan_and_fp32_rebuild(
    make_server, monkeypatch
):
    config = MiniMaxH3ImagePipelineConfig()
    monkeypatch.setattr(
        config,
        "get_supported_task_types",
        lambda: (ModelTaskType.T2I, ModelTaskType.I2I),
    )
    with pytest.raises(ValueError, match="plan width >= 2"):
        make_server(
            config=config, minimax_h3_adaln_online=True, minimax_h3_adaln_plan_width=1
        )
    monkeypatch.setenv("SGLANG_DIFFUSION_MINIMAX_H3_ADALN_FP32", "1")
    with pytest.raises(ValueError, match="match precision"):
        make_server(minimax_h3_adaln_online=True)


@pytest.mark.parametrize("mode", ["auto", "manual", "speed", "memory"])
def test_image_server_modes_keep_offload_parallel_and_compile_off(make_server, mode):
    args = make_server(140, performance_mode=mode)
    assert not args.has_layerwise_offload_components()
    assert not args.use_fsdp_inference
    assert not args.enable_torch_compile
    assert not args.enable_cfg_parallel
    for name in (
        "num_gpus",
        "tp_size",
        "sp_degree",
        "ulysses_degree",
        "ring_degree",
        "dp_size",
        "cfg_parallel_degree",
    ):
        assert getattr(args, name) == 1
    # FA is requested through the generic explicit-request path the loader enforces.
    assert args.requested_component_attention_backend("transformer") == "fa"


def _layout(args):
    return {
        "tp": args.tp_size,
        "sp": args.sp_degree,
        "ulysses": args.ulysses_degree,
        "ring": args.ring_degree,
        "kv_gather": args.kv_gather_degree,
        "cfg": args.cfg_parallel_degree,
        "cfg_on": args.enable_cfg_parallel,
    }


@pytest.mark.parametrize("memory_gib", [130, 140, 280])
def test_large_memory_residency_keeps_explicit_offload_overrides(
    make_server, memory_gib
):
    resident = make_server(memory_gib)
    assert not resident.text_encoder_cpu_offload
    assert not resident.vae_cpu_offload
    offloaded = make_server(memory_gib, text_encoder_cpu_offload=True)
    assert offloaded.text_encoder_cpu_offload
    below_threshold = make_server(129)
    assert below_threshold.text_encoder_cpu_offload


@pytest.mark.parametrize("num_gpus", [2, 4, 8])
def test_default_multi_gpu_layout_is_ulysses_over_all_gpus(make_server, num_gpus):
    # No CFG parallel: every GPU goes to sequence parallelism. sp_degree=2
    # arrives as the auto K/V-gather alias (ulysses_degree follows it), which
    # the H3 DiT runs as its Ulysses exchange.
    args = make_server(140, num_gpus=num_gpus)
    assert not args.use_fsdp_inference
    assert _layout(args) == {
        "tp": 1,
        "sp": num_gpus,
        "ulysses": num_gpus,
        "ring": 1,
        "kv_gather": 2 if num_gpus == 2 else 1,
        "cfg": 1,
        "cfg_on": False,
    }
    assert args.sp_split_auto == (num_gpus == 2)


@pytest.mark.parametrize(
    "kwargs,expected",
    [
        (
            {"num_gpus": 4, "ulysses_degree": 4},
            {"tp": 1, "sp": 4, "ulysses": 4, "ring": 1, "kv_gather": 1, "cfg": 1},
        ),
        (
            {"num_gpus": 2, "tp_size": 2},
            {"tp": 2, "sp": 1, "ulysses": 1, "ring": 1, "kv_gather": 1, "cfg": 1},
        ),
        (
            {"num_gpus": 4, "tp_size": 2, "ulysses_degree": 2},
            {"tp": 2, "sp": 2, "ulysses": 2, "ring": 1, "kv_gather": 1, "cfg": 1},
        ),
    ],
)
def test_explicit_parallel_flags_normalize_to_the_requested_layout(
    make_server, kwargs, expected
):
    args = make_server(140, **kwargs)
    assert _layout(args) == {**expected, "cfg_on": False}


@pytest.mark.parametrize(
    "kwargs,match",
    [
        ({"num_gpus": 2, "cfg_parallel_degree": 2}, "does not support CFG parallel"),
        ({"num_gpus": 2, "kv_gather_degree": 2}, "K/V-gather"),
        ({"num_gpus": 2, "dp_size": 2}, "dp_size=1"),
        ({"num_gpus": 2, "use_fsdp_inference": True}, "fsdp"),
    ],
)
def test_unsupported_image_layouts_fail_at_server_args(make_server, kwargs, match):
    with pytest.raises(ValueError, match=match):
        make_server(140, **kwargs)


def test_image_auxiliary_policy_prevents_cpu_to_layerwise_replacement(make_server):
    args = make_server()
    args.text_encoder_cpu_offload = True
    args.image_encoder_cpu_offload = True
    args.vae_cpu_offload = True
    args.layerwise_offload_components = None
    auto_tune_module.ServerArgsAutoTuner(
        args
    ).maybe_replace_cpu_offloaded_components_with_layerwise()
    assert args.layerwise_offload_components is None
    assert args.text_encoder_cpu_offload and args.vae_cpu_offload


@pytest.mark.parametrize("component", ["transformer", "text_encoder", "video_vae"])
@pytest.mark.parametrize(
    "selector", ["component_residency", "layerwise_offload_components"]
)
def test_explicit_image_layerwise_placement_loads_on_cpu(
    make_server, component, selector
):
    value = (
        f"{component}=layerwise-offload"
        if selector == "component_residency"
        else [component]
    )
    args = make_server(24, **{selector: value})
    assert args.residency_mode(component) == "layerwise-offload"
    assert args.should_start_component_on_cpu(component)


def test_video_config_auto_offloads_auxiliary_components_layerwise(make_server):
    assert ModelDeploymentConfig().auto_auxiliary_layerwise_offload
    args = make_server(config=MiniMaxH3PipelineConfig())
    assert {"text_encoder", "image_encoder", "vae"}.issubset(
        args.layerwise_offload_components
    )
