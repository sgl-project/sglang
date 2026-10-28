"""Actual CUDA replay of Clover convolution/GDN/coreset with refreshed inputs."""

import pytest
import torch

from sglang.kernels.ops.attention.clover_gdn_triton import (
    clover_local_memory_triton,
)
from sglang.kernels.ops.mamba.clover_conv_triton import (
    clover_local_memory_conv,
)
from sglang.srt.models.clover_moe import block_coreset
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=15, stage="base-b-kernel-unit", runner_config="1-gpu-large")


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_preview_graph_refreshes_history_indices_valid_mask_and_coreset():
    torch.manual_seed(57)
    width = 32
    bs, heads, dim = 3, 2, 128
    raw = torch.randn(bs * width, 3 * heads * dim, device="cuda", dtype=torch.bfloat16)
    conv = torch.randn(5, raw.shape[1], 3, device="cuda", dtype=torch.bfloat16) * 0.1
    state = torch.randn(5, heads, dim, dim, device="cuda") * 0.05
    weights = torch.randn(raw.shape[1], 4, device="cuda", dtype=torch.bfloat16) * 0.1
    right = torch.randn(raw.shape[1], 1, 3, device="cuda", dtype=torch.bfloat16) * 0.1
    slots = torch.tensor([1, 2, -1], device="cuda", dtype=torch.int32)
    valid = torch.ones(bs * width, device="cuda", dtype=torch.bool)
    valid[-width:] = False
    router = torch.randn(bs * width, 256, device="cuda")
    g = -torch.rand(1, bs * width, heads, device="cuda")
    conv_kw = dict(
        clean_conv_states=conv,
        cache_indices=slots,
        conv_weights=weights,
        bias=None,
        activation="silu",
        block_size=width,
        active_valid_mask=valid,
        conv_mode="independent_symmetric",
        conv_right_weight=right,
    )
    gdn_kw = dict(
        clean_ssm_states=state,
        cache_indices=slots,
        active_valid_mask=valid,
        scale=dim**-0.5,
        softmax_logit_scale=torch.full((heads,), 10.0, device="cuda"),
        softmax_history_bias=torch.full((heads,), 5.0, device="cuda"),
        softmax_history_c=torch.ones(heads, device="cuda"),
    )

    def preview():
        convolved = clover_local_memory_conv(raw, **conv_kw)
        q, k, v = [
            part.reshape(1, bs * width, heads, dim)
            for part in convolved.chunk(3, dim=-1)
        ]
        return (
            convolved,
            clover_local_memory_triton(q, k, v, g, **gdn_kw),
            block_coreset(router, width, 8, 0.7),
        )

    before = conv.clone(), state.clone()
    preview()  # compile before capture
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured = preview()
    for current, original in zip((conv, state), before):
        torch.testing.assert_close(current, original, atol=0, rtol=0)
    pointers = [value.data_ptr() for value in captured]
    for step, new_slots in enumerate(([2, 1, -1], [4, -1, -1], [3, 4, 1])):
        slots.copy_(torch.tensor(new_slots, device="cuda", dtype=torch.int32))
        valid.copy_((slots >= 0).repeat_interleave(width))
        valid[width - 1] = False  # special tail inside a live row
        raw.normal_()
        router.normal_()
        # Represents a causal commit/radix restoration between previews.
        conv[4].fill_(0.2 + step * 0.1)
        state[4].fill_(0.02 + step * 0.01)
        before = conv.clone(), state.clone()
        graph.replay()
        eager = preview()
        for got, want in zip(captured, eager):
            torch.testing.assert_close(got, want)
        for current, original in zip((conv, state), before):
            torch.testing.assert_close(current, original, atol=0, rtol=0)
        assert pointers == [value.data_ptr() for value in captured]


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__]))
