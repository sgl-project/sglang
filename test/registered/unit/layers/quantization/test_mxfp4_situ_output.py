import sys
from types import SimpleNamespace
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
    from sglang.srt.layers.moe import route_quant_handoff
    from sglang.srt.layers.quantization import mxfp4 as mxfp4_module
    from sglang.srt.layers.quantization.mxfp4 import Mxfp4MoEMethod

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

    method = Mxfp4MoEMethod.__new__(Mxfp4MoEMethod)
    method.use_deep_gemm = False
    method.use_marlin = False
    method.use_flashinfer = True
    method._fi_kernel = None
    method.flashinfer_mxfp4_moe_precision = "default"
    method.hidden_size = hidden
    method.intermediate_size_per_partition = 128
    method.moe_runner_config = SimpleNamespace(activation="situ")

    dummy = torch.empty(1, dtype=torch.uint8, device="cuda")
    layer = SimpleNamespace(
        moe_ep_rank=0,
        num_local_experts=1,
        num_experts=1,
        w13_weight=dummy,
        w13_weight_scale=dummy,
        gemm1_alpha=None,
        gemm1_clamp_limit=None,
        w2_weight=dummy,
        w2_weight_scale=dummy,
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

    latent = torch.empty_like(x)
    with (
        patch.object(
            route_quant_handoff,
            "take",
            return_value=(packed_topk, x_quant, x_scale),
        ),
        patch(
            "sglang.srt.layers.quantization.mxfp4.trtllm_fp4_block_scale_routed_moe",
            side_effect=fake_routed_moe,
            create=True,
        ),
        patch.object(
            mxfp4_module,
            "RoutingMethodType",
            SimpleNamespace(TopK=SimpleNamespace(value=0)),
            create=True,
        ),
        patch.object(
            mxfp4_module,
            "ActivationType",
            SimpleNamespace(Situ=SimpleNamespace(value=0)),
            create=True,
        ),
        zero_copy_context.set_moe_output(latent),
    ):
        combine_input = method.apply(layer, dispatch_output)

    assert returned_ptr is not None
    assert returned_ptr != latent.data_ptr()
    assert combine_input.hidden_states.data_ptr() == latent.data_ptr()
    torch.testing.assert_close(combine_input.hidden_states, expected, rtol=0, atol=0)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
