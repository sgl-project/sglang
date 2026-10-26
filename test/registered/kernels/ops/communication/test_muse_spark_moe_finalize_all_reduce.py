"""Muse Spark TRT-LLM MoE finalize (unpermute + post-expert norm + weighted sum).

Single-GPU: the CuTeDSL kernel without the fused all-reduce against the torch
reference.
"""

import pytest
import torch

from sglang.kernels.ops.communication import muse_spark_moe_finalize_all_reduce_cutedsl
from sglang.kernels.ops.communication.muse_spark_moe_finalize_all_reduce import (
    FusedPostExpertNormWeightedSumTRTLLM,
)
from sglang.kernels.spec import KernelBackend
from sglang.test.ci.ci_register import register_cuda_ci

# The SM100 pool has no single-GPU runner_config.
register_cuda_ci(est_time=60, stage="base-b-kernel-unit", runner_config="4-gpu-b200")

pytestmark = pytest.mark.skipif(
    not muse_spark_moe_finalize_all_reduce_cutedsl.HAS_CUTEDSL,
    reason="needs the CuTeDSL runtime",
)


@pytest.mark.parametrize("num_tokens", [1, 20, 100, 300])
@pytest.mark.parametrize("weight_first", [True, False])
@pytest.mark.parametrize("weights_dtype", [torch.bfloat16, torch.float32])
def test_finalize_matches_native(num_tokens, weight_first, weights_dtype):
    torch.manual_seed(num_tokens)
    topk, hidden, padded = 8, 4096, 4096 + 64
    op = FusedPostExpertNormWeightedSumTRTLLM(
        topk=topk, hidden_dim=hidden, eps=1e-6, weight_first=weight_first
    )
    num_slots = num_tokens * topk
    gemm2_output = torch.randn(num_slots, padded, device="cuda", dtype=torch.bfloat16)
    expert_weights = torch.rand(num_tokens, topk, device="cuda").to(weights_dtype)
    permuted = torch.randperm(num_slots, device="cuda", dtype=torch.int32)
    # -1 marks an expert slot owned by another EP rank.
    permuted[torch.rand(num_slots, device="cuda") < 0.2] = -1
    norm_weight = 1.0 + 0.1 * torch.randn(hidden, device="cuda", dtype=torch.bfloat16)

    def run(backend):
        out = torch.empty(num_tokens, hidden, device="cuda", dtype=torch.bfloat16)
        op(gemm2_output, expert_weights, permuted, norm_weight, out, backend=backend)
        return out

    # The reference adds each expert's bf16-rounded contribution into the bf16
    # output; the kernel accumulates in fp32 and rounds once.
    torch.testing.assert_close(
        run(KernelBackend.CUTE_DSL), run(KernelBackend.TORCH), rtol=2e-2, atol=1e-1
    )


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__, "-v"]))
