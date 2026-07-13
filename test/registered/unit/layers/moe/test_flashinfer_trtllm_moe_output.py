import sys

import pytest
import torch

from sglang.srt.layers.moe.flashinfer_trtllm_moe import (
    trtllm_fp8_block_scale_moe_out_wrapper,
    trtllm_fp8_block_scale_moe_wrapper,
    trtllm_fp8_block_scale_routed_moe_out_wrapper,
    trtllm_fp8_block_scale_routed_moe_wrapper,
)
from sglang.srt.utils import is_sm100_supported
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=5, stage="base-b", runner_config="1-gpu-small")
register_cuda_ci(est_time=5, stage="base-b", runner_config="4-gpu-b200")

requires_sm100 = pytest.mark.skipif(
    not is_sm100_supported(),
    reason="FlashInfer TRTLLM-gen FP8 MoE kernels are SM100-only.",
)


def _random_fp8(shape, generator):
    return (
        torch.randn(shape, device="cuda", dtype=torch.float32, generator=generator)
        .clamp(-1, 1)
        .to(torch.float8_e4m3fn)
    )


def _make_fp8_moe_kwargs(seq_len, num_experts):
    hidden_size = 128
    intermediate_size = 128
    gen = torch.Generator(device="cuda").manual_seed(0)
    return dict(
        hidden_states=_random_fp8((seq_len, hidden_size), gen),
        hidden_states_scale=torch.ones(
            (hidden_size // 128, seq_len), device="cuda", dtype=torch.float32
        ),
        gemm1_weights=_random_fp8(
            (num_experts, 2 * intermediate_size, hidden_size), gen
        ),
        gemm1_weights_scale=torch.ones(
            (num_experts, 2 * intermediate_size // 128, hidden_size // 128),
            device="cuda",
            dtype=torch.float32,
        ),
        gemm2_weights=_random_fp8((num_experts, hidden_size, intermediate_size), gen),
        gemm2_weights_scale=torch.ones(
            (num_experts, hidden_size // 128, intermediate_size // 128),
            device="cuda",
            dtype=torch.float32,
        ),
        num_experts=num_experts,
        intermediate_size=intermediate_size,
        local_expert_offset=0,
        local_num_experts=num_experts,
        routed_scaling_factor=1.0,
        tune_max_num_tokens=seq_len,
    )


@requires_sm100
def test_routed_fp8_wrapper_writes_real_flashinfer_output_tensor():
    seq_len = 4
    num_experts = 4
    top_k = 1
    topk_ids = (
        torch.arange(seq_len, device="cuda", dtype=torch.int32).reshape(seq_len, 1)
        % num_experts
    )
    topk_weights = torch.ones((seq_len, top_k), device="cuda", dtype=torch.float32)
    packed_topk_ids = (topk_ids << 16) | topk_weights.to(torch.bfloat16).view(
        torch.int16
    ).to(torch.int32)

    kwargs = dict(
        **_make_fp8_moe_kwargs(seq_len, num_experts),
        topk_ids=packed_topk_ids,
        routing_bias=None,
        top_k=top_k,
        n_group=None,
        topk_group=None,
    )

    ref = trtllm_fp8_block_scale_routed_moe_wrapper(**kwargs)
    assert ref is not None
    torch.cuda.synchronize()
    output = torch.full_like(ref, float("nan"))

    result = trtllm_fp8_block_scale_routed_moe_out_wrapper(
        **kwargs,
        output=output,
    )
    torch.cuda.synchronize()

    assert result is None
    assert torch.isfinite(output).all().item()
    torch.testing.assert_close(output, ref, rtol=0, atol=0)


@requires_sm100
def test_non_routed_fp8_wrapper_writes_real_flashinfer_output_tensor():
    seq_len = 8
    num_experts = 8
    gen = torch.Generator(device="cuda").manual_seed(1)
    kwargs = dict(
        **_make_fp8_moe_kwargs(seq_len, num_experts),
        routing_logits=torch.randn(
            (seq_len, num_experts),
            device="cuda",
            dtype=torch.float32,
            generator=gen,
        ),
        routing_bias=torch.zeros(num_experts, device="cuda", dtype=torch.float32),
        top_k=2,
        n_group=2,
        topk_group=1,
        routing_method_type=2,
    )

    ref = trtllm_fp8_block_scale_moe_wrapper(**kwargs)
    torch.cuda.synchronize()
    output = torch.full_like(ref, float("nan"))

    result = trtllm_fp8_block_scale_moe_out_wrapper(
        **kwargs,
        output=output,
    )
    torch.cuda.synchronize()

    assert result is None
    assert torch.isfinite(output).all().item()
    torch.testing.assert_close(output, ref, rtol=0, atol=0)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
