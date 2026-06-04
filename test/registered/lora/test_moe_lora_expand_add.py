"""Unit test for the rank-specialized LoRA-B expand-add kernel
(`_moe_lora_expand_add_kernel` in
`sglang/srt/lora/trtllm_moe/specialized_expand.py`), scoped to the **down-proj** GEMM.

The down-proj caller (`lora_dispatch.py`, `moe_overlap.py`) drives this kernel with
`mul_routed_weight=True` + `fuse_sum_all_reduce=True`: each of a token's `top_k`
per-expert deltas is scaled by its routing weight and atomic-added into the single
per-token output row. This test checks that fused variant against a torch reference for
the qwen3.5-35b local-EP shape (64 experts, N=2048, rank=16, top_k=8).

It also guards the routing/tiling block-size contract: `_invoke_moe_lora_expand_add`
tiles `expert_ids` with `config["BLOCK_SIZE_M"]` (one entry per M-block), so the routing
buffers MUST be aligned with the same block. The test sweeps block_m {16,32,64} so any
hardcoded mismatch overruns `expert_ids` -> CUDA illegal memory access (the same class
of bug as the shrink f2adddd regression).

Companion micro-benchmark: `benchmark/kernels/lora_moe_expand/bench_expand_add_down.py`.

Usage:
    python -m pytest test/registered/lora/test_moe_lora_expand_add.py -v
"""

import unittest

import torch
import triton

from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=15, stage="base-b", runner_config="1-gpu-small")

from sglang.srt.layers.moe.moe_runner.triton_utils.moe_align_block_size import (
    moe_align_block_size,
)
from sglang.srt.lora.triton_ops.virtual_experts import (
    _fused_virtual_topk_ids,
    fused_sanitize_expert_ids,
)
from sglang.srt.lora.trtllm_moe.specialized_expand import _invoke_moe_lora_expand_add


def _make_inputs(bs, num_experts, top_k, n, rank, dtype, device):
    torch.manual_seed(0)
    topk_ids = torch.stack(
        [torch.randperm(num_experts, device=device)[:top_k] for _ in range(bs)]
    ).to(torch.int32)
    topk_weights = torch.rand(bs, top_k, device=device, dtype=torch.float32) * 0.9 + 0.1
    token_lora_mapping = torch.zeros(bs, device=device, dtype=torch.int32)
    intermediate = torch.randn(bs * top_k, rank, device=device, dtype=dtype) * 0.1
    lora_b = torch.randn(1, num_experts, n, rank, device=device, dtype=dtype) * 0.1
    return topk_ids, topk_weights, token_lora_mapping, intermediate, lora_b


def _build_routing(topk_ids, token_lora_mapping, num_experts, block_m):
    """Single-adapter virtual-expert routing tiled at ``block_m`` (mirrors
    `_get_routing` in virtual_experts.py: virtual ids -> align -> tight trim ->
    sanitize)."""
    virtual_topk_ids, _, virtual_num_experts = _fused_virtual_topk_ids(
        topk_ids, token_lora_mapping, num_experts, shared_outer=False, max_loras=1
    )
    sorted_token_ids, expert_ids, num_tokens_post_padded = moe_align_block_size(
        virtual_topk_ids, block_m, virtual_num_experts
    )
    num_tokens = topk_ids.numel()
    max_nonempty = min(num_tokens, virtual_num_experts)
    tight = triton.cdiv(num_tokens + max_nonempty * (block_m - 1), block_m) * block_m
    return (
        sorted_token_ids[:tight],
        fused_sanitize_expert_ids(expert_ids[: tight // block_m], virtual_num_experts),
        num_tokens_post_padded,
    )


def _expand(intermediate, lora_b, topk_ids, topk_weights, routing, block_m):
    sorted_token_ids, expert_ids, num_tokens_post_padded = routing
    lora_b_virtual = lora_b.reshape(lora_b.shape[0] * lora_b.shape[1], *lora_b.shape[2:])
    n = lora_b.shape[2]
    bs = topk_ids.shape[0]
    # fuse_sum_all_reduce atomic-adds the top_k deltas into one row -> zero each call.
    output = torch.zeros(bs, n, dtype=intermediate.dtype, device=intermediate.device)
    config = {"BLOCK_SIZE_M": block_m, "BLOCK_SIZE_N": 64, "GROUP_SIZE_M": 1, "num_warps": 4}
    _invoke_moe_lora_expand_add(
        intermediate,
        lora_b_virtual,
        output,
        topk_weights.reshape(-1),
        topk_ids,
        sorted_token_ids,
        expert_ids,
        num_tokens_post_padded,
        config,
        mul_routed_weight=True,
        fuse_sum_all_reduce=True,
    )
    return output


def _ref_expand(intermediate, lora_b, topk_ids, topk_weights):
    bs, top_k = topk_ids.shape
    n = lora_b.shape[2]
    b = lora_b[0].float()
    inter = intermediate.float()
    out = torch.zeros(bs, n, device=intermediate.device, dtype=torch.float32)
    for m in range(bs):
        for k in range(top_k):
            e = int(topk_ids[m, k].item())
            vt = m * top_k + k
            out[m] += (inter[vt] @ b[e].t()) * float(topk_weights[m, k].item())
    return out


class TestMoeLoraExpandAddDown(CustomTestCase):
    """Correctness of the fused (routed-weight + sum-all-reduce) down-proj expand-add."""

    NUM_EXPERTS = 64
    TOP_K = 8
    N = 2048
    RANK = 16
    # fp32 reference vs per-expert bf16 atomic-add accumulation -> generous abs tol.
    TOL = 5e-2

    @classmethod
    def setUpClass(cls):
        if not torch.cuda.is_available():
            raise unittest.SkipTest("CUDA required")
        cls.device = "cuda:0"

    def _check(self, bs, block_m):
        topk_ids, tkw, tlm, inter, lb = _make_inputs(
            bs, self.NUM_EXPERTS, self.TOP_K, self.N, self.RANK,
            torch.bfloat16, self.device,
        )
        routing = _build_routing(topk_ids, tlm, self.NUM_EXPERTS, block_m)
        out = _expand(inter, lb, topk_ids, tkw, routing, block_m).float()
        ref = _ref_expand(inter, lb, topk_ids, tkw)
        err = float((out - ref).abs().max().item())
        self.assertLessEqual(
            err, self.TOL, f"bs={bs} block_m={block_m} max_abs_err={err:.4e}"
        )

    def test_p0_bs64_rank16(self):
        """P0 scope: bs=64, rank=16, production-tuned block_m=64."""
        self._check(bs=64, block_m=64)

    def test_block_m_contract_sweep(self):
        """Routing must be tiled with the same block the launcher uses. Sweeping block_m
        catches any hardcoded-block mismatch (expert_ids overrun -> IMA)."""
        for bs in (16, 64):
            for block_m in (16, 32, 64):
                with self.subTest(bs=bs, block_m=block_m):
                    self._check(bs, block_m)


if __name__ == "__main__":
    unittest.main()
