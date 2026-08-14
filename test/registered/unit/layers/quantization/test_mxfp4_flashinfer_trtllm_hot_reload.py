"""Online reload contract for the FlashInfer TRT-LLM MXFP4 MoE layout."""

from __future__ import annotations

import pytest
import torch
from torch.nn import Module

from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=60, stage="base-b", runner_config="1-gpu-large")

if not torch.cuda.is_available():
    pytest.skip("CUDA required", allow_module_level=True)

from sglang.srt.layers.quantization.mxfp4_flashinfer_trtllm_moe import (
    Mxfp4FlashinferTrtllmMoEMethod,
)
from sglang.srt.model_loader.loader import postprocess_weight, restore_weight
from sglang.srt.utils import is_sm100_supported

if not is_sm100_supported():
    pytest.skip("FlashInfer TRT-LLM MXFP4 MoE requires SM100+", allow_module_level=True)


EXPERT_PARAMS = (
    "w13_weight",
    "w2_weight",
    "w13_weight_scale_inv",
    "w2_weight_scale_inv",
)
NUM_EXPERTS = 2
HIDDEN_SIZE = 128
INTERMEDIATE_SIZE = 64


class _NoOpFp8:
    def process_weights_after_loading(self, layer: Module) -> None:
        pass


def _build_layer() -> Module:
    layer = Module().to("cuda")
    layer.num_local_experts = NUM_EXPERTS
    method = Mxfp4FlashinferTrtllmMoEMethod.__new__(Mxfp4FlashinferTrtllmMoEMethod)
    method._fp8 = _NoOpFp8()
    method.prefix = "test"
    method._kernel_layout = {}
    layer.quant_method = method
    method.create_weights(
        layer,
        NUM_EXPERTS,
        HIDDEN_SIZE,
        INTERMEDIATE_SIZE,
        torch.bfloat16,
        alloc_device="cuda",
        weight_loader=lambda *args, **kwargs: None,
    )
    return layer


def _load_weights(layer: Module, seed: int) -> None:
    generator = torch.Generator(device="cuda").manual_seed(seed)
    for name, param in layer.named_parameters():
        if "scale" in name:
            encoded = torch.randint(
                100,
                140,
                param.shape,
                generator=generator,
                device="cuda",
                dtype=torch.uint8,
            )
            param.data.copy_(encoded.view(torch.float8_e8m0fnu))
        else:
            param.data.copy_(
                torch.randint(
                    -128,
                    127,
                    param.shape,
                    generator=generator,
                    device="cuda",
                    dtype=torch.int8,
                )
            )


def _layout(layer: Module) -> dict[str, tuple[tuple[int, ...], torch.dtype]]:
    return {
        name: (tuple(param.shape), param.dtype)
        for name, param in layer.named_parameters()
    }


def test_hot_reload_matches_fresh_load_and_preserves_kernel_addresses():
    layer = _build_layer()
    load_layout = _layout(layer)

    _load_weights(layer, seed=0)
    postprocess_weight(layer, torch.device("cuda"))
    initial = {name: param.detach().clone() for name, param in layer.named_parameters()}
    addresses = {name: param.data_ptr() for name, param in layer.named_parameters()}

    reference = _build_layer()
    _load_weights(reference, seed=1)
    postprocess_weight(reference, torch.device("cuda"))
    expected = {
        name: param.detach().clone() for name, param in reference.named_parameters()
    }

    restore_weight(layer, torch.device("cuda"))
    assert _layout(layer) == load_layout
    for name in EXPERT_PARAMS:
        assert hasattr(getattr(layer, name), "weight_loader")

    _load_weights(layer, seed=1)
    postprocess_weight(layer, torch.device("cuda"))

    for name, param in layer.named_parameters():
        assert param.data_ptr() == addresses[name]
        assert not torch.equal(initial[name].view(torch.uint8), param.view(torch.uint8))
        assert torch.equal(expected[name].view(torch.uint8), param.view(torch.uint8))
