"""Online reload contract for the FlashInfer TRT-LLM MXFP4 MoE layout."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch
from torch.nn import Module

from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=60, stage="base-b-kernel-unit", runner_config="4-gpu-b200")

if not torch.cuda.is_available():
    pytest.skip("CUDA required", allow_module_level=True)

from sglang.srt.layers.quantization.mxfp4_flashinfer_trtllm_moe import (
    Mxfp4FlashinferTrtllmMoEMethod,
)
from sglang.srt.model_loader.loader import postprocess_weight, restore_weight
from sglang.srt.utils import is_sm100_supported
from sglang.test.test_utils import CustomTestCase

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


def _build_layer(
    *,
    num_experts=NUM_EXPERTS,
    hidden_size=HIDDEN_SIZE,
    intermediate_size=INTERMEDIATE_SIZE,
) -> Module:
    layer = Module().to("cuda")
    layer.num_local_experts = num_experts
    layer.num_experts = num_experts
    layer.moe_ep_rank = 0
    method = Mxfp4FlashinferTrtllmMoEMethod.__new__(Mxfp4FlashinferTrtllmMoEMethod)
    method._fp8 = _NoOpFp8()
    method.prefix = "test"
    method.flashinfer_mxfp4_moe_precision = "default"
    layer.quant_method = method
    with torch.device("cuda"):
        method.create_weights(
            layer,
            num_experts,
            hidden_size,
            intermediate_size,
            torch.bfloat16,
            weight_loader=lambda *args, **kwargs: None,
        )
    method.create_moe_runner(layer, SimpleNamespace(swiglu_limit=10.0))
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


class TestMxfp4KernelForward(CustomTestCase):
    @torch.inference_mode()
    def test_mxfp8_forward_accepts_int8_load_parameters(self):
        """Signed checkpoint bytes must remain loadable across real MoE forwards."""
        from sglang.srt.layers.moe.token_dispatcher import StandardDispatchOutput
        from sglang.srt.layers.moe.token_dispatcher.flashinfer import (
            FlashinferCombineInput,
            FlashinferDispatchOutput,
        )
        from sglang.srt.layers.moe.topk import StandardTopKOutput
        from sglang.srt.layers.quantization import mxfp4_flashinfer_trtllm_moe

        layer = _build_layer(num_experts=32, hidden_size=4096, intermediate_size=2048)
        generator = torch.Generator(device="cuda").manual_seed(0)
        hidden_states = torch.randn(
            8, 4096, generator=generator, device="cuda", dtype=torch.bfloat16
        )
        dispatch = StandardDispatchOutput(
            hidden_states=hidden_states,
            hidden_states_scale=None,
            topk_output=StandardTopKOutput(
                topk_weights=torch.full(
                    (8, 6), 1 / 6, device="cuda", dtype=torch.float32
                ),
                topk_ids=torch.arange(6, device="cuda", dtype=torch.int32).repeat(8, 1),
                router_logits=None,
            ),
        )
        # A2A reserves padding slots for unequal source token counts.
        dispatch.topk_output.topk_ids[-1].fill_(-1)
        dispatch.topk_output.topk_weights[-1].zero_()
        a2a_dispatch = FlashinferDispatchOutput(
            hidden_states=dispatch.hidden_states,
            hidden_states_scale=None,
            topk_output=dispatch.topk_output,
        )
        outputs = []
        # A local kernel call needs no distributed symmetric-memory allocator.
        with (
            patch.object(
                mxfp4_flashinfer_trtllm_moe, "get_tp_group", return_value=None
            ),
            patch.object(
                mxfp4_flashinfer_trtllm_moe,
                "is_allocation_symmetric",
                return_value=False,
            ),
        ):
            for packed_byte in (-86, 34):
                if outputs:
                    restore_weight(layer, torch.device("cuda"))
                for name in LOAD_PARAMS:
                    getattr(layer, name).fill_(
                        1 / 32 if "scale" in name else packed_byte
                    )
                postprocess_weight(layer, torch.device("cuda"))
                output = layer.quant_method.apply(layer, dispatch).hidden_states
                torch.cuda.synchronize()
                self.assertEqual(output.shape, hidden_states.shape)
                self.assertEqual(output.dtype, torch.bfloat16)
                self.assertTrue(torch.isfinite(output).all().item())
                self.assertGreater(torch.count_nonzero(output).item(), 0)
                self.assertEqual(torch.count_nonzero(output[-1]).item(), 0)
                a2a_result = layer.quant_method.apply(layer, a2a_dispatch)
                self.assertIsInstance(a2a_result, FlashinferCombineInput)
                torch.testing.assert_close(
                    a2a_result.hidden_states, output, rtol=0, atol=0
                )
                self.assertEqual(layer.w13_weight.dtype, torch.int8)
                self.assertEqual(layer.w2_weight.dtype, torch.int8)
                outputs.append(output.clone())
        self.assertFalse(torch.equal(*outputs))


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__]))
