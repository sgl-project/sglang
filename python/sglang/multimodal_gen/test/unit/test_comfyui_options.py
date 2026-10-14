# SPDX-License-Identifier: Apache-2.0
"""SGLDOptions / Cache-DiT node output, model-aware preflight, and request wiring."""

import os
from types import SimpleNamespace

import pytest

from sglang.multimodal_gen.apps.ComfyUI_SGLDiffusion.core.generator import (
    SGLDiffusionGenerator,
    _worker_env,
)
from sglang.multimodal_gen.apps.ComfyUI_SGLDiffusion.core.preflight import (
    check_model_options,
)
from sglang.multimodal_gen.apps.ComfyUI_SGLDiffusion.options import (
    SGLDOptions,
)


def _options(**values) -> dict:
    return SGLDOptions().create_options(**values)[0]


def test_defaults_leave_layout_and_attention_to_the_worker() -> None:
    options = _options()
    assert options == {
        "num_gpus": 1,
        "enable_torch_compile": False,
        "regional_compile": False,
        "enable_breakable_cuda_graph": False,
        "dit_layerwise_offload": False,
        "int8_kernel": "auto",
    }


def test_display_names_map_to_sglang_backends() -> None:
    options = _options(attention_backend="flash_attention", int8_gemm="sglang_jit")
    assert options["attention_backend"] == "fa"
    assert options["int8_kernel"] == "jit"


def test_manual_layout_derives_sp_and_checks_gpu_count() -> None:
    options = _options(num_gpus=4, parallel_mode="manual", tp_size=2, ulysses_degree=2)
    assert (options["tp_size"], options["sp_degree"], options["ring_degree"]) == (
        2,
        2,
        1,
    )
    with pytest.raises(ValueError, match="must equal"):
        _options(num_gpus=2, parallel_mode="manual")


def test_execution_mode_maps_to_exactly_one_server_flag() -> None:
    compiled = _options(execution_mode="torch_compile", regional_compile=True)
    assert compiled["enable_torch_compile"] and compiled["regional_compile"]
    graph = _options(execution_mode="cuda_graph", regional_compile=True)
    assert graph["enable_breakable_cuda_graph"]
    assert not graph["enable_torch_compile"] and not graph["regional_compile"]


@pytest.mark.parametrize(
    "model_type, values, native_dir, message",
    [
        (
            "flux",
            {"execution_mode": "cuda_graph"},
            False,
            "cuda_graph is not supported",
        ),
        (
            "minimax_h3",
            {"execution_mode": "cuda_graph", "enable_cache_dit": True},
            False,
            "enable_cache_dit",
        ),
        (
            "minimax_h3",
            {"execution_mode": "cuda_graph", "memory_mode": "layerwise_offload"},
            False,
            "layerwise offload",
        ),
        ("fast_h3", {"execution_mode": "torch_compile"}, True, "eager"),
        (
            "vdn_h3",
            {
                "num_gpus": 2,
                "parallel_mode": "manual",
                "ulysses_degree": 1,
                "ring_degree": 2,
            },
            True,
            "ring_degree=1",
        ),
        (
            "minimax_h3",
            {
                "num_gpus": 2,
                "parallel_mode": "manual",
                "ring_degree": 2,
                "attention_backend": "sage_attention2",
            },
            False,
            "requires attention_backend=fa",
        ),
        ("vdn_h3", {"attention_backend": "flash_attention"}, True, "does not apply"),
        (
            "qwen_image",
            {"attention_backend": "subblock_sparse_attn"},
            False,
            "does not apply",
        ),
        ("flux", {"quantization": "fp8"}, False, "native model directories"),
        ("lumina2", {"int8_gemm": "comfy_kitchen"}, False, "no effect"),
    ],
)
def test_preflight_rejects_what_the_model_cannot_run(
    model_type, values, native_dir, message
) -> None:
    with pytest.raises(ValueError, match=message):
        check_model_options(model_type, _options(**values), native_dir=native_dir)


@pytest.mark.parametrize(
    "model_type, values, native_dir",
    [
        ("minimax_h3", {"execution_mode": "cuda_graph"}, False),
        (
            "minimax_h3",
            {"enable_cache_dit": True, "attention_backend": "sage_attention2"},
            False,
        ),
        (
            "minimax_h3",
            {"num_gpus": 2, "parallel_mode": "manual", "ring_degree": 2},
            False,
        ),
        (
            "minimax_h3",
            {"int8_gemm": "comfy_kitchen"},
            False,
        ),
        ("fast_h3", {"attention_backend": "video_sparse_attn_h3"}, True),
        ("vdn_h3", {"quantization": "fp8", "enable_cache_dit": True}, True),
        ("qwen_image", {"execution_mode": "torch_compile"}, False),
    ],
)
def test_preflight_accepts_supported_combinations(
    model_type, values, native_dir
) -> None:
    check_model_options(model_type, _options(**values), native_dir=native_dir)


def test_native_models_are_not_selected_through_options() -> None:
    runtime = SGLDiffusionGenerator()
    runtime.init_generator = None
    with pytest.raises(ValueError, match="SGLDNativeModelLoader"):
        runtime.load_model("h3.safetensors", sgld_options={"model_type": "fast_h3"})


def test_rejected_options_keep_the_running_worker() -> None:
    """A config error used to surface only after the previous worker was killed."""
    from unittest.mock import Mock

    runtime = SGLDiffusionGenerator()
    runtime.get_comfyui_model = lambda *a: (SimpleNamespace(), None, "flux")
    runtime.close_generator = Mock()
    runtime.init_generator = Mock(side_effect=AssertionError("worker must not start"))
    with pytest.raises(ValueError, match="cuda_graph is not supported"):
        runtime.load_model(
            "flux.safetensors", sgld_options=_options(execution_mode="cuda_graph")
        )
    runtime.close_generator.assert_not_called()


def test_breakable_cuda_graph_keeps_comfyui_warmup_off() -> None:
    """BCG forced a synthetic server warmup, which the ComfyUI step stage cannot
    run (no sampler tensors), so every cuda_graph worker failed at startup."""
    from sglang.multimodal_gen.runtime.server_args import ServerArgs

    args = ServerArgs.from_kwargs(
        model_path="h3.safetensors",
        pipeline_class_name="MiniMaxH3Pipeline",
        model_id="MiniMax-H3",
        comfyui_mode=True,
        enable_breakable_cuda_graph=True,
        warmup_mode="off",
    )
    assert args.warmup_mode == "off"
    assert args.warmup_resolutions is None


def test_worker_env_is_restored_after_the_worker_starts() -> None:
    key = "SGLANG_DIFFUSION_CONVROT_INT8_BACKEND"
    before = os.environ.get(key)
    with _worker_env({key: "comfy_kitchen"}):
        assert os.environ[key] == "comfy_kitchen"
    assert os.environ.get(key) == before
