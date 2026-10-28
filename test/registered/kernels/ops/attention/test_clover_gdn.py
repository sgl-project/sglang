"""Clover GDN preview and native chunk checkpoint numerical contracts."""

import math

import pytest
import torch

from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=15, stage="base-b-kernel-unit", runner_config="1-gpu-large")


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("block_size", [2, 32])
def test_preview_kernel_parity_readonly_and_graph(block_size):
    from sglang.kernels.ops.attention.clover_gdn_triton import (
        clover_local_memory_triton,
    )

    torch.manual_seed(123)
    q = torch.randn(1, block_size * 2, 2, 128, device="cuda", dtype=torch.bfloat16)
    k = torch.randn_like(q)
    v = torch.randn(1, block_size * 2, 4, 128, device="cuda", dtype=torch.bfloat16)
    g = -torch.rand(1, block_size * 2, 4, device="cuda")
    state = torch.randn(3, 4, 128, 128, device="cuda") * 0.05
    indices = torch.tensor([1, -1], device="cuda", dtype=torch.int32)
    kw = dict(
        clean_ssm_states=state,
        cache_indices=indices,
        active_valid_mask=torch.ones(block_size * 2, device="cuda", dtype=torch.bool),
        scale=128**-0.5,
        softmax_logit_scale=torch.full((4,), 10.0, device="cuda"),
        softmax_history_bias=torch.full((4,), 5.0, device="cuda"),
        softmax_history_c=torch.ones(4, device="cuda"),
    )
    kw["active_valid_mask"][block_size:] = False
    before = state.clone()
    # Only the first row is live. Combine its local softmax with the history bucket.
    qf = q[0, :block_size].float().repeat_interleave(2, dim=1)
    kf = k[0, :block_size].float().repeat_interleave(2, dim=1)
    qf = qf * torch.rsqrt(qf.square().sum(-1, keepdim=True) + 1e-6)
    kf = kf * torch.rsqrt(kf.square().sum(-1, keepdim=True) + 1e-6)
    local_logits = torch.einsum("lhk,shk->hls", qf, kf) * 10
    history_logits = torch.full_like(
        local_logits[..., :1], 5 + math.log(block_size / 32)
    )
    weights = torch.cat((local_logits, history_logits), dim=-1).softmax(-1)
    local = torch.einsum("hls,shv->lhv", weights[..., :-1], v[0, :block_size].float())
    history = torch.einsum("lhk,hvk->lhv", qf * 128**-0.5, state[1])
    history = history * g[0, :block_size].exp()[..., None]
    expected = torch.zeros_like(v)
    expected[0, :block_size] = (local + weights[..., -1].T[..., None] * history).to(
        v.dtype
    )
    actual = clover_local_memory_triton(q, k, v, g, **kw)
    torch.testing.assert_close(actual.float(), expected.float(), atol=0.04, rtol=0.04)
    # Reduced-precision and non-contiguous scalar views must match FP32 inputs.
    for dtype in (torch.bfloat16, torch.float16):
        for name in (
            "softmax_logit_scale",
            "softmax_history_bias",
            "softmax_history_c",
        ):
            kw[name] = torch.randn(8, device="cuda", dtype=dtype)[::2]
        fp32_kw = {
            name: value.float() if name.startswith("softmax_") else value
            for name, value in kw.items()
        }
        torch.testing.assert_close(
            clover_local_memory_triton(q, k, v, g, **kw),
            clover_local_memory_triton(q, k, v, g, **fp32_kw),
            rtol=0,
            atol=0,
        )
    graph = torch.cuda.CUDAGraph()
    torch.cuda.synchronize()
    with torch.cuda.graph(graph):
        captured = clover_local_memory_triton(q, k, v, g, **kw)
    q.mul_(0.5)
    graph.replay()
    torch.testing.assert_close(captured, clover_local_memory_triton(q, k, v, g, **kw))
    torch.testing.assert_close(state, before, atol=0, rtol=0)


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__]))
