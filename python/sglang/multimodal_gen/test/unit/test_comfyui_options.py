# SPDX-License-Identifier: Apache-2.0
"""SGLDOptions output, model-aware preflight, and request wiring."""

from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from sglang.multimodal_gen.apps.ComfyUI_SGLDiffusion.core.generator import (
    SGLDiffusionGenerator,
    _apply_cache_mode,
)
from sglang.multimodal_gen.apps.ComfyUI_SGLDiffusion.core.preflight import (
    check_model_options,
)
from sglang.multimodal_gen.apps.ComfyUI_SGLDiffusion.options import SGLDOptions


def _options(**values) -> dict:
    return SGLDOptions().create_options(**values)[0]


def test_cache_mode_reaches_the_h3_request() -> None:
    from sglang.multimodal_gen.apps.ComfyUI_SGLDiffusion.executors.minimax_h3 import (
        MiniMaxH3Executor,
    )
    from sglang.multimodal_gen.runtime.cache.cache_dit_integration import (
        resolve_cache_dit_request_overrides,
    )

    assert "cache_dit_params" not in _options(cache_mode="cache_dit")
    options = _options(cache_mode="cache_dit", cache_dit_tuning="manual")
    assert resolve_cache_dit_request_overrides(options["cache_dit_params"])

    executor = object.__new__(MiniMaxH3Executor)
    torch.nn.Module.__init__(executor)
    packed = SimpleNamespace(guidance_scale=1.0, height=512, width=512)
    for values, cache_dit, spectrum in (
        ({"cache_mode": "cache_dit", "cache_dit_tuning": "manual"}, True, False),
        ({"cache_mode": "spectrum"}, False, True),
    ):
        _apply_cache_mode(executor, _options(**values))
        request = executor._sampling_params_kwargs(packed, torch.tensor(1.0))
        assert request["enable_cache_dit"] is cache_dit
        assert request.get("enable_spectrum", False) is spectrum
        assert ("cache_dit_params" in request) is cache_dit


def test_manual_layout_derives_sp_and_checks_gpu_count() -> None:
    options = _options(num_gpus=4, parallel_mode="manual", tp_size=2, ulysses_degree=2)
    assert (options["tp_size"], options["sp_degree"]) == (2, 2)
    with pytest.raises(ValueError, match="must equal"):
        _options(num_gpus=2, parallel_mode="manual")


_RING2 = {"num_gpus": 2, "parallel_mode": "manual", "ring_degree": 2}
_GRAPH = {"execution_mode": "cuda_graph"}


@pytest.mark.parametrize(
    "model_type, values, native_dir, message",
    [
        ("flux", _GRAPH, False, "cuda_graph is not"),
        ("minimax_h3", {**_GRAPH, "cache_mode": "spectrum"}, False, "cache_mode=off"),
        (
            "minimax_h3",
            {**_GRAPH, "memory_mode": "layerwise_offload"},
            False,
            "offload",
        ),
        ("fast_h3", {"execution_mode": "torch_compile"}, True, "eager"),
        ("vdn_h3", _RING2, True, "ring_degree=1"),
        (
            "minimax_h3",
            {**_RING2, "attention_backend": "sage_attention2"},
            False,
            "=fa",
        ),
        ("vdn_h3", {"attention_backend": "flash_attention"}, True, "does not apply"),
        ("qwen_image", {"attention_backend": "sol_attention"}, False, "does not apply"),
        ("flux", {"quantization": "fp8"}, False, "native model directory"),
        ("qwen_image21", {"quantization": "mxfp8"}, False, "pre-quantized"),
        ("qwen_image21", {"quantization": "fp8"}, False, "pre-quantized"),
        ("lumina2", {"int8_gemm": "comfy_kitchen"}, False, "no effect"),
        ("qwen_image", {"cache_mode": "cache_dit"}, False, "cache_dit is not"),
        ("minimax_h3", {"cache_mode": "teacache"}, False, "teacache is not"),
        ("minimax_h3", {"num_gpus": 2, "cache_mode": "spectrum"}, False, "degree 1"),
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
        ("minimax_h3", _GRAPH, False),
        ("minimax_h3", _RING2, False),
        ("minimax_h3", {"int8_gemm": "comfy_kitchen", "cache_mode": "spectrum"}, False),
        ("fast_h3", {"attention_backend": "video_sparse_attn_h3"}, True),
        ("vdn_h3", {"quantization": "fp8", "cache_mode": "cache_dit"}, True),
        ("qwen_image21", {"quantization": "convrot_int8"}, False),
        ("qwen_image21", {"int8_gemm": "sglang_jit"}, False),
    ],
)
def test_preflight_accepts_supported_combinations(
    model_type, values, native_dir
) -> None:
    check_model_options(model_type, _options(**values), native_dir=native_dir)


def test_native_models_are_not_selected_through_options() -> None:
    with pytest.raises(ValueError, match="SGLDNativeModelLoader"):
        SGLDiffusionGenerator().load_model(
            "h3.safetensors", sgld_options={"model_type": "fast_h3"}
        )


def test_rejected_options_keep_the_running_worker() -> None:
    """A config error used to surface only after the previous worker was killed."""
    runtime = SGLDiffusionGenerator()
    runtime.get_comfyui_model = lambda *a: (SimpleNamespace(), None, "flux")
    runtime.close_generator = Mock()
    runtime.init_generator = Mock(side_effect=AssertionError("worker must not start"))
    with pytest.raises(ValueError, match="cuda_graph is not supported"):
        runtime.load_model("flux.safetensors", sgld_options=_options(**_GRAPH))
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
    assert (args.warmup_mode, args.warmup_resolutions) == ("off", None)
