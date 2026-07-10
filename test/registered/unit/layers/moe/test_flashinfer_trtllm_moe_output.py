import inspect
import sys

import pytest
import torch

from sglang.srt.layers.moe.flashinfer_trtllm_moe import (
    trtllm_fp8_block_scale_routed_moe_wrapper,
)
from sglang.srt.layers.moe.moe_runner.base import (
    finalize_moe_output_copy_add,
    maybe_moe_output_copy_add,
    moe_output_copy_add_ctx,
)
from sglang.srt.utils import is_sm100_supported
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=5, stage="base-b", runner_config="1-gpu-small")
register_cuda_ci(est_time=5, stage="base-b", runner_config="4-gpu-b200")

requires_cuda = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA is required for FlashInfer MoE tests."
)
requires_sm100 = pytest.mark.skipif(
    not is_sm100_supported(),
    reason="FlashInfer TRTLLM-gen routed FP8 MoE kernels are SM100-only.",
)


@requires_cuda
def test_moe_output_copy_add_folds_shared_output_on_alt_stream():
    current_stream = torch.cuda.current_stream()
    alt_stream = torch.cuda.Stream()

    shape = (8, 128)
    routed_source = torch.arange(
        shape[0] * shape[1], device="cuda", dtype=torch.float32
    )
    routed_source = routed_source.reshape(shape).to(torch.bfloat16)
    routed_output = torch.empty_like(routed_source)
    shared_source = torch.full(shape, 3.0, device="cuda", dtype=torch.bfloat16)
    shared_output = torch.empty_like(routed_source)
    out = torch.empty_like(routed_source)

    # Match production command ordering: routed MoE is issued on the current
    # stream before shared experts are issued on the alternate stream.
    alt_stream.wait_stream(current_stream)
    routed_output.copy_(routed_source)

    with moe_output_copy_add_ctx() as state:
        result = maybe_moe_output_copy_add(routed_output, out)
        assert result is out
        assert not state.consumed

    with torch.cuda.stream(alt_stream):
        shared_output.copy_(shared_source)

    current_stream.wait_stream(alt_stream)
    assert finalize_moe_output_copy_add(state, shared_output)
    torch.testing.assert_close(out, routed_source + shared_source, rtol=0, atol=0)

    fallback_out = torch.empty_like(routed_source)
    result = maybe_moe_output_copy_add(routed_output, fallback_out)
    assert result is fallback_out
    torch.testing.assert_close(fallback_out, routed_source, rtol=0, atol=0)


@requires_sm100
def test_routed_fp8_wrapper_writes_real_flashinfer_output_tensor():
    fused_moe = pytest.importorskip("flashinfer.fused_moe")
    if (
        "output"
        not in inspect.signature(fused_moe.trtllm_fp8_block_scale_routed_moe).parameters
    ):
        pytest.skip("FlashInfer routed FP8 MoE does not expose output=.")

    seq_len = 4
    hidden_size = 128
    intermediate_size = 128
    num_experts = 4
    top_k = 1

    gen = torch.Generator(device="cuda").manual_seed(0)
    hidden_states = (
        torch.randn(
            (seq_len, hidden_size),
            device="cuda",
            dtype=torch.float32,
            generator=gen,
        )
        .clamp(-1, 1)
        .to(torch.float8_e4m3fn)
    )
    hidden_states_scale = torch.ones(
        (hidden_size // 128, seq_len), device="cuda", dtype=torch.float32
    )
    gemm1_weights = (
        torch.randn(
            (num_experts, 2 * intermediate_size, hidden_size),
            device="cuda",
            dtype=torch.float32,
            generator=gen,
        )
        .clamp(-1, 1)
        .to(torch.float8_e4m3fn)
    )
    gemm1_weights_scale = torch.ones(
        (num_experts, (2 * intermediate_size) // 128, hidden_size // 128),
        device="cuda",
        dtype=torch.float32,
    )
    gemm2_weights = (
        torch.randn(
            (num_experts, hidden_size, intermediate_size),
            device="cuda",
            dtype=torch.float32,
            generator=gen,
        )
        .clamp(-1, 1)
        .to(torch.float8_e4m3fn)
    )
    gemm2_weights_scale = torch.ones(
        (num_experts, hidden_size // 128, intermediate_size // 128),
        device="cuda",
        dtype=torch.float32,
    )
    topk_ids = (
        torch.arange(seq_len, device="cuda", dtype=torch.int32).reshape(seq_len, 1)
        % num_experts
    )
    topk_weights = torch.ones((seq_len, top_k), device="cuda", dtype=torch.float32)
    packed_topk_ids = (topk_ids << 16) | topk_weights.to(torch.bfloat16).view(
        torch.int16
    ).to(torch.int32)

    kwargs = dict(
        topk_ids=packed_topk_ids,
        routing_bias=None,
        hidden_states=hidden_states,
        hidden_states_scale=hidden_states_scale,
        gemm1_weights=gemm1_weights,
        gemm1_weights_scale=gemm1_weights_scale,
        gemm2_weights=gemm2_weights,
        gemm2_weights_scale=gemm2_weights_scale,
        num_experts=num_experts,
        top_k=top_k,
        n_group=None,
        topk_group=None,
        intermediate_size=intermediate_size,
        local_expert_offset=0,
        local_num_experts=num_experts,
        routed_scaling_factor=1.0,
        tune_max_num_tokens=seq_len,
    )

    ref = trtllm_fp8_block_scale_routed_moe_wrapper(**kwargs)
    assert ref is not None
    torch.cuda.synchronize()
    output = torch.full_like(ref, float("nan"))

    result = trtllm_fp8_block_scale_routed_moe_wrapper(
        **kwargs,
        output=output,
    )
    torch.cuda.synchronize()

    assert result is None
    assert torch.isfinite(output).all().item()
    torch.testing.assert_close(output, ref, rtol=0, atol=0)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
