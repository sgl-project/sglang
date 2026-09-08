import sys
from types import ModuleType, SimpleNamespace
from unittest.mock import patch

import pytest
import torch

from sglang.srt.layers import zero_copy_context
from sglang.srt.layers.moe.token_dispatcher.standard import StandardDispatchOutput
from sglang.srt.layers.moe.topk import StandardTopKOutput
from sglang.test.ci.ci_register import register_amd_ci

register_amd_ci(est_time=5, stage="stage-b", runner_config="1-gpu-small-amd")

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires a GPU")


def test_situ_routed_moe_returns_published_output_buffer():
    # Import mxfp4 before flashinfer_trtllm to avoid the pre-existing
    # compressed_tensors circular import.
    # isort: off
    from sglang.srt.layers.quantization import mxfp4 as mxfp4_module
    from sglang.srt.layers.moe.moe_runner import (
        flashinfer_trtllm as flashinfer_trtllm_module,
    )
    from sglang.srt.layers.moe.moe_runner.base import MoeRunnerConfig
    from sglang.srt.layers.moe.moe_runner.flashinfer_trtllm import (
        FlashInferTrtllmGenMxfp4MoeQuantInfo,
        _fused_experts_flashinfer_mxfp4_sm100_trtllm_gen,
    )
    # isort: on

    tokens, hidden, top_k = 3, 128, 2
    x = torch.randn(tokens, hidden, dtype=torch.bfloat16, device="cuda")
    x_quant = torch.zeros(tokens, hidden, dtype=torch.uint8, device="cuda")
    x_scale = torch.zeros(tokens, hidden // 32, dtype=torch.uint8, device="cuda")
    packed_topk = torch.zeros(tokens, top_k, dtype=torch.int32, device="cuda")
    topk_output = StandardTopKOutput(
        topk_weights=torch.full(
            (tokens, top_k), 0.5, dtype=torch.float32, device="cuda"
        ),
        topk_ids=torch.zeros(tokens, top_k, dtype=torch.int32, device="cuda"),
        router_logits=torch.empty(tokens, 0, dtype=torch.float32, device="cuda"),
    )
    dispatch_output = StandardDispatchOutput(
        hidden_states=x,
        hidden_states_scale=None,
        topk_output=topk_output,
    )

    dummy = torch.empty(1, dtype=torch.uint8, device="cuda")
    quant_info = FlashInferTrtllmGenMxfp4MoeQuantInfo(
        w13_weight=dummy,
        w2_weight=dummy,
        w13_weight_scale=dummy,
        w2_weight_scale=dummy,
        w13_weight_bias=dummy,
        w2_weight_bias=dummy,
        gemm1_alpha=dummy,
        gemm1_beta=dummy,
        gemm1_clamp_limit=dummy,
        global_num_experts=1,
        local_expert_offset=0,
        local_num_experts=1,
        intermediate_size_per_partition=128,
        hidden_size=hidden,
        flashinfer_mxfp4_moe_precision="default",
    )
    expected = (
        torch.arange(tokens * hidden, dtype=torch.float32, device="cuda")
        .reshape(tokens, hidden)
        .to(torch.bfloat16)
    )
    returned_ptr = None

    def fake_routed_moe(**kwargs):
        nonlocal returned_ptr
        kwargs["output"].copy_(expected)
        ffi_result = kwargs["output"].clone()
        returned_ptr = ffi_result.data_ptr()
        return ffi_result

    flashinfer = ModuleType("flashinfer")
    flashinfer.__path__ = []
    flashinfer.trtllm_fp4_block_scale_moe = None
    fused_moe = ModuleType("flashinfer.fused_moe")
    fused_moe.trtllm_fp4_block_scale_routed_moe = fake_routed_moe
    tllm_enums = ModuleType("flashinfer.tllm_enums")
    tllm_enums.RoutingMethodType = SimpleNamespace(TopK=SimpleNamespace(value=0))
    tllm_enums.ActivationType = SimpleNamespace(Situ=SimpleNamespace(value=0))

    latent = torch.empty_like(x)
    with (
        patch.object(
            mxfp4_module,
            "_prepare_flashinfer_mxfp8_activations",
            return_value=(x, packed_topk, x_quant, x_scale),
        ),
        patch.dict(
            "sys.modules",
            {
                "flashinfer": flashinfer,
                "flashinfer.fused_moe": fused_moe,
                "flashinfer.tllm_enums": tllm_enums,
            },
        ),
        patch.object(
            flashinfer_trtllm_module, "trtllm_moe_enable_pdl", return_value=False
        ),
        zero_copy_context.set_moe_output(latent),
    ):
        combine_input = _fused_experts_flashinfer_mxfp4_sm100_trtllm_gen(
            dispatch_output,
            quant_info,
            MoeRunnerConfig(activation="situ"),
        )

    assert returned_ptr is not None
    assert returned_ptr != latent.data_ptr()
    assert combine_input.hidden_states.data_ptr() == latent.data_ptr()
    torch.testing.assert_close(combine_input.hidden_states, expected, rtol=0, atol=0)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
