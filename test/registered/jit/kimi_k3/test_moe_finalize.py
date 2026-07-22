"""K3 MoE finalize (top-16 weighted unpermute) correctness.

Pinned against a pure-torch fp32 ascending-k reference, which is bit-identical
to the kernel (the bf16 product is exact in fp32; both round once per add).
Covers the production-like permuted layout (per-expert tile padding), dropped
(-1) slots, non-K3 hidden sizes, and the trtllm-fork baseline kernel
(moe_finalize_fuse_shared) for cross-implementation agreement.
"""

import unittest

import torch
from sglang.jit_kernel.kimi_k3 import moe_finalize
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=60, stage="base-b-kernel-unit", runner_config="1-gpu-large")
register_cuda_ci(est_time=60, stage="base-b-kernel-unit", runner_config="4-gpu-b200")

TOPK = 16


def build_permuted_layout(
    num_tokens: int,
    hidden: int,
    num_experts: int = 896,
    tile: int = 8,
    drop_prob: float = 0.0,
    device: str = "cuda",
    seed: int = 0,
):
    """Emulate the trtllm-gen permuted gemm2 layout: rows grouped by expert,
    each expert's token segment padded to a tile multiple; expanded slot
    (t, k) -> row base[expert] + arrival order. Returns (gemm2_out, idx, w)."""
    gen = torch.Generator(device="cpu").manual_seed(seed)
    topk_ids = torch.stack(
        [torch.randperm(num_experts, generator=gen)[:TOPK] for _ in range(num_tokens)]
    )
    counts = torch.bincount(topk_ids.flatten(), minlength=num_experts)
    padded = (counts + tile - 1) // tile * tile
    bases = torch.cumsum(padded, 0) - padded
    fill = torch.zeros(num_experts, dtype=torch.long)
    idx = torch.empty(num_tokens * TOPK, dtype=torch.int32)
    for i, e in enumerate(topk_ids.flatten().tolist()):
        idx[i] = bases[e] + fill[e]
        fill[e] += 1
    if drop_prob > 0:
        dropped = torch.rand(num_tokens * TOPK, generator=gen) < drop_prob
        idx[dropped] = -1
    num_rows = int(padded.sum())
    gemm2_out = torch.randn(num_rows, hidden, dtype=torch.bfloat16, device=device)
    weights = torch.rand(
        num_tokens, TOPK, generator=gen
    ).to(device=device, dtype=torch.bfloat16)
    return gemm2_out, idx.to(device), weights


def ref_finalize(
    gemm2_out: torch.Tensor, idx: torch.Tensor, weights: torch.Tensor
) -> torch.Tensor:
    num_tokens = weights.shape[0]
    idx2 = idx.view(num_tokens, TOPK).long()
    acc = torch.zeros(
        num_tokens, gemm2_out.shape[1], dtype=torch.float32, device=gemm2_out.device
    )
    for k in range(TOPK):
        valid = idx2[:, k] >= 0
        rows = gemm2_out[idx2[:, k].clamp(min=0)].float()
        acc += torch.where(
            valid[:, None], weights[:, k, None].float() * rows, torch.zeros_like(rows)
        )
    return acc.to(torch.bfloat16)


class TestK3MoeFinalize(CustomTestCase):
    def _check(self, num_tokens: int, hidden: int, drop_prob: float = 0.0, seed: int = 0):
        gemm2_out, idx, weights = build_permuted_layout(
            num_tokens, hidden, drop_prob=drop_prob, seed=seed
        )
        out = moe_finalize(gemm2_out, idx, weights)
        ref = ref_finalize(gemm2_out, idx, weights)
        self.assertTrue(
            bool((out.view(torch.uint16) == ref.view(torch.uint16)).all()),
            f"T={num_tokens} H={hidden} drop={drop_prob}: kernel != fp32 reference",
        )
        return gemm2_out, idx, weights, out

    def test_k3_shapes(self):
        for num_tokens in (1, 2, 8, 64, 300):
            self._check(num_tokens, hidden=3584, seed=num_tokens)

    def test_free_hidden(self):
        # hidden only needs to be a multiple of 8
        for hidden in (8, 72, 512, 1032, 7168):
            self._check(num_tokens=16, hidden=hidden, seed=hidden)

    def test_dropped_slots(self):
        self._check(num_tokens=64, hidden=3584, drop_prob=0.2, seed=7)

    def test_out_param(self):
        gemm2_out, idx, weights = build_permuted_layout(32, 3584, seed=11)
        out = torch.empty(32, 3584, dtype=torch.bfloat16, device="cuda")
        result = moe_finalize(gemm2_out, idx, weights, out=out)
        self.assertIs(result, out)
        ref = ref_finalize(gemm2_out, idx, weights)
        self.assertTrue(bool((out.view(torch.uint16) == ref.view(torch.uint16)).all()))

    def test_matches_trtllm_fork_kernel(self):
        from sglang.jit_kernel.moe_finalize_fuse_shared import moe_finalize_fuse_shared
        from sglang.jit_kernel.utils import is_arch_support_pdl

        gemm2_out, idx, weights, out = self._check(64, 3584, seed=42)
        baseline = moe_finalize_fuse_shared(
            gemm2_out, idx, weights, None, TOPK, enable_pdl=is_arch_support_pdl()
        )
        self.assertTrue(
            bool((out.view(torch.uint16) == baseline.view(torch.uint16)).all()),
            "kernel disagrees with the trtllm-fork finalize",
        )


if __name__ == "__main__":
    unittest.main()
