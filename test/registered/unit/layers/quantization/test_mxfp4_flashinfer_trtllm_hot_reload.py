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


LOAD_PARAMS = (
    "w13_weight",
    "w2_weight",
    "w13_weight_scale_inv",
    "w2_weight_scale_inv",
)
KERNEL_TENSORS = (
    "w13_weight",
    "w2_weight",
    "w13_weight_scale_inv_shuffled",
    "w2_weight_scale_inv_shuffled",
)
NUM_EXPERTS = 2
HIDDEN_SIZE = 128
INTERMEDIATE_SIZE = 64


class _NoOpFp8:
    """Stand-in for the wrapped ``Fp8MoEMethod``.

    On the FlashInfer MXFP4 backend the real ``process_weights_after_loading``
    only re-views the FP4 payloads as ``int8`` and returns before any scale
    transform, so running it a second time is a no-op and this test can leave
    it out. If that branch ever starts transforming weights or scales, the
    reload contract below must be checked with the real method instead.
    """

    def process_weights_after_loading(self, layer: Module) -> None:
        pass


def _build_layer() -> Module:
    layer = Module().to("cuda")
    layer.num_local_experts = NUM_EXPERTS
    method = Mxfp4FlashinferTrtllmMoEMethod.__new__(Mxfp4FlashinferTrtllmMoEMethod)
    method._fp8 = _NoOpFp8()
    method.prefix = "test"
    layer.quant_method = method
    with torch.device("cuda"):
        method.create_weights(
            layer,
            NUM_EXPERTS,
            HIDDEN_SIZE,
            INTERMEDIATE_SIZE,
            torch.bfloat16,
            weight_loader=lambda *args, **kwargs: None,
        )
    return layer


def _load_weights(layer: Module, seed: int) -> None:
    generator = torch.Generator(device="cuda").manual_seed(seed)
    for name in LOAD_PARAMS:
        param = getattr(layer, name)
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
        name: (tuple(getattr(layer, name).shape), getattr(layer, name).dtype)
        for name in LOAD_PARAMS
    }


def _kernel_tensors(layer: Module) -> dict[str, torch.Tensor]:
    return {name: getattr(layer, name) for name in KERNEL_TENSORS}


def test_hot_reload_matches_fresh_load_and_preserves_kernel_addresses():
    layer = _build_layer()
    load_layout = _layout(layer)
    load_params = {name: getattr(layer, name) for name in LOAD_PARAMS}

    _load_weights(layer, seed=0)
    postprocess_weight(layer, torch.device("cuda"))
    initial = {
        name: tensor.detach().clone() for name, tensor in _kernel_tensors(layer).items()
    }
    addresses = {
        name: tensor.data_ptr() for name, tensor in _kernel_tensors(layer).items()
    }

    reference = _build_layer()
    _load_weights(reference, seed=1)
    postprocess_weight(reference, torch.device("cuda"))
    expected = {
        name: tensor.detach().clone()
        for name, tensor in _kernel_tensors(reference).items()
    }

    # The loader hooks bracket every update session; neither may disturb the
    # checkpoint-layout parameters ``load_weights`` writes into.
    restore_weight(layer, torch.device("cuda"))
    assert _layout(layer) == load_layout
    for name in LOAD_PARAMS:
        assert getattr(layer, name) is load_params[name]
        assert hasattr(getattr(layer, name), "weight_loader")

    _load_weights(layer, seed=1)
    postprocess_weight(layer, torch.device("cuda"))

    for name in LOAD_PARAMS:
        assert getattr(layer, name) is load_params[name]
    for name, tensor in _kernel_tensors(layer).items():
        assert tensor.data_ptr() == addresses[name]
        assert not torch.equal(
            initial[name].view(torch.uint8), tensor.view(torch.uint8)
        )
        assert torch.equal(expected[name].view(torch.uint8), tensor.view(torch.uint8))
