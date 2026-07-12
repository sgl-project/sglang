import pytest
import torch

from sglang.srt.layers.moe.ep_moe.kernels import (
    post_reorder_deepgemm,
    post_reorder_triton_kernel,
)
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=30, stage="base-b-kernel-unit", runner_config="1-gpu-large")
register_cuda_ci(est_time=30, stage="base-b-kernel-unit", runner_config="4-gpu-b200")

dev = "cuda"


def _build(
    num_tokens,
    hidden,
    num_routed=128,
    top_k_routed=4,
    with_shared=True,
    pad_frac=0.0,
    seed=0,
):
    num_experts = num_routed + (1 if with_shared else 0)
    top_k = top_k_routed + (1 if with_shared else 0)
    m_max = (num_tokens // 256 + 1) * 256
    g = torch.Generator(device="cpu").manual_seed(seed)
    down_output = torch.randn(
        num_experts * m_max, hidden, dtype=torch.bfloat16, device=dev
    )
    topk_ids = torch.full((num_tokens, top_k), -1, dtype=torch.int32, device=dev)
    src2dst = torch.full((num_tokens, top_k), -1, dtype=torch.int32, device=dev)
    topk_weights = torch.rand(num_tokens, top_k, dtype=torch.float32, device=dev)
    counts = [0] * num_experts
    for t in range(num_tokens):
        experts = torch.randperm(num_routed, generator=g)[:top_k_routed].tolist()
        if with_shared:
            experts = experts + [num_routed]
        for slot, e in enumerate(experts):
            if (
                pad_frac > 0
                and slot < top_k - 1
                and torch.rand(1, generator=g).item() < pad_frac
            ):
                continue
            src2dst[t, slot] = e * m_max + counts[e]
            counts[e] += 1
            topk_ids[t, slot] = e
    return down_output, src2dst, topk_ids, topk_weights, top_k


def _ref(down_output, src2dst, topk_ids, topk_weights, top_k, hidden, rsf):
    num_tokens = topk_ids.shape[0]
    out = torch.zeros(num_tokens, hidden, dtype=torch.float32, device=dev)
    do = down_output.float()
    for slot in range(top_k):
        valid = topk_ids[:, slot] >= 0
        dst = src2dst[:, slot].long().clamp_min(0)
        out[valid] += (do[dst] * topk_weights[:, slot, None])[valid]
    return out * rsf


@pytest.mark.parametrize("num_tokens", [1, 8, 128, 1024, 4096])
@pytest.mark.parametrize("pad_frac", [0.0, 0.3])
@pytest.mark.parametrize("rsf", [1.0, 2.0])
def test_post_reorder_deepgemm(num_tokens, pad_frac, rsf):
    hidden = 6144
    do, s2d, tids, tw, tk = _build(
        num_tokens, hidden, pad_frac=pad_frac, seed=num_tokens
    )
    ref = _ref(do, s2d, tids, tw, tk, hidden, rsf)

    new = torch.empty(num_tokens, hidden, dtype=torch.bfloat16, device=dev)
    post_reorder_deepgemm(do, new, s2d, tids, tw, tk, num_tokens, hidden, rsf)

    old = torch.empty(num_tokens, hidden, dtype=torch.bfloat16, device=dev)
    post_reorder_triton_kernel[(num_tokens,)](
        do, old, s2d, tids, tw, tk, hidden, BLOCK_SIZE=512
    )
    old *= rsf

    torch.testing.assert_close(new.float(), ref, rtol=2e-2, atol=2e-2)
    torch.testing.assert_close(new.float(), old.float(), rtol=5e-2, atol=0.5)


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__, "-v", "-s"]))
