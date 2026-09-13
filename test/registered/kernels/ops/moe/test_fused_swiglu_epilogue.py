"""Correctness of the SwiGLU-in-the-up-GEMM-epilogue MoE fast path.

Two claims are invisible at the call site and would break silently under an
innocuous-looking rewrite:

1. Interleaving W13 rows leaves every up-GEMM output column unchanged -- the
   permute only decides which column a gate/up pair lands in.
2. The epilogue reproduces the `silu_and_mul` it replaces bit for bit (that
   kernel keeps silu at float until the multiply; rounding to bf16 first
   double-rounds and diverges on many inputs).

Hence bitwise assertions: a tolerance would accept exactly the errors these
tests exist to catch.
"""

import sys

import pytest
import torch

from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=12, stage="base-b-kernel-unit", runner_config="1-gpu-large")

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


@pytest.fixture(scope="module", autouse=True)
def _runtime_scaffolding():
    """`fused_experts` needs global server args and a TP group.

    It reads server args for the fused-sum-all-reduce switch, and allocates
    its output under ``use_symmetric_memory(get_tp_group(), ...)`` even when
    symmetric allocation is off. Single rank, gloo, TP=EP=PP=1.
    """
    import os

    from sglang.srt.distributed.parallel_state import (
        init_distributed_environment,
        initialize_model_parallel,
        model_parallel_is_initialized,
    )
    from sglang.srt.server_args import ServerArgs, set_global_server_args_for_scheduler

    set_global_server_args_for_scheduler(ServerArgs(model_path="dummy"))

    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29641")
    os.environ.setdefault("RANK", "0")
    os.environ.setdefault("WORLD_SIZE", "1")
    os.environ.setdefault("LOCAL_RANK", "0")
    if not torch.distributed.is_initialized():
        init_distributed_environment(world_size=1, rank=0, local_rank=0, backend="gloo")
    if not model_parallel_is_initialized():
        initialize_model_parallel(
            tensor_model_parallel_size=1,
            expert_model_parallel_size=1,
            pipeline_model_parallel_size=1,
            backend="gloo",
        )


def _interleave_w13_rows(w13: torch.Tensor) -> torch.Tensor:
    """Reproduce the load-time permute: [gate; up] -> [gate0, up0, gate1, ...]."""
    inter = w13.shape[1] // 2
    idx = torch.empty(w13.shape[1], dtype=torch.long, device=w13.device)
    idx[0::2] = torch.arange(0, inter, device=w13.device)
    idx[1::2] = torch.arange(inter, 2 * inter, device=w13.device)
    return w13[:, idx].contiguous()


@pytest.mark.parametrize(
    "num_tokens,hidden,inter,num_experts,topk",
    [
        (1, 256, 128, 8, 2),  # bs=1 decode, the shape this path exists for
        (13, 512, 256, 16, 4),  # ragged token count, forces the BLOCK_M tail
    ],
)
def test_fused_matches_unfused_bitwise(num_tokens, hidden, inter, num_experts, topk):
    """The fused epilogue reproduces the standalone activation path exactly.

    This is the production contract: flipping the flag must not move a single
    bit of the MoE output. It exercises claims 1 and 2 together through the
    real `fused_experts` entry point rather than a hand-rolled harness.
    """
    from sglang.srt.layers.moe.moe_runner.base import MoeRunnerConfig
    from sglang.srt.layers.moe.moe_runner.triton_utils.fused_moe import fused_experts
    from sglang.srt.layers.moe.topk import StandardTopKOutput

    torch.manual_seed(0)
    dtype = torch.bfloat16
    x = torch.randn(num_tokens, hidden, dtype=dtype, device="cuda")
    w13 = torch.randn(num_experts, 2 * inter, hidden, dtype=dtype, device="cuda") / 16
    w2 = torch.randn(num_experts, hidden, inter, dtype=dtype, device="cuda") / 16

    router_logits = torch.randn(num_tokens, num_experts, dtype=dtype, device="cuda")
    topk_weights, topk_ids = torch.topk(router_logits.float(), topk, dim=-1)
    topk_weights = torch.softmax(topk_weights, dim=-1)
    topk_ids = topk_ids.to(torch.int32)

    def _run(w1, fuse):
        topk_output = StandardTopKOutput(
            topk_weights=topk_weights.clone(),
            topk_ids=topk_ids.clone(),
            router_logits=router_logits,
        )
        config = MoeRunnerConfig(
            num_experts=num_experts,
            top_k=topk,
            hidden_size=hidden,
            intermediate_size_per_partition=inter,
            params_dtype=dtype,
            activation="silu",
            inplace=False,
        )
        return fused_experts(
            x.clone(),
            w1,
            w2,
            topk_output,
            config,
            fuse_swiglu_interleaved=fuse,
        )

    ref = _run(w13, False)
    got = _run(_interleave_w13_rows(w13), True)

    mismatch = (got.view(torch.int16) != ref.view(torch.int16)).sum().item()
    assert mismatch == 0, f"{mismatch}/{ref.numel()} output elements differ"


def test_epilogue_matches_silu_and_mul_bitwise():
    """The in-register activation is bit-identical to the kernel it replaces.

    Guards the fast-math instruction replication (`ex2.approx.ftz` for
    `__expf`, `div.approx.ftz` for `__fdividef`) and the single final rounding
    to bf16. Rewriting this as `tl.sigmoid`, or casting silu to bf16 before the
    multiply, passes any tolerance check and fails here.

    The reference is pinned to JIT — the backend this fusion replaces on CUDA.
    Left on auto-dispatch, a fallback to the `forward_native` torch reference
    (accurate sigmoid) would fail the comparison for an unrelated reason.
    """
    import triton
    import triton.language as tl

    import sglang.kernels as K
    from sglang.kernels.ops.activation.activation import silu_and_mul
    from sglang.kernels.spec import KernelBackend

    @triton.jit
    def _epilogue_only(x_ptr, out_ptr, N: tl.constexpr, BLOCK: tl.constexpr):
        offs = tl.arange(0, BLOCK)
        acc = tl.load(x_ptr + offs, mask=offs < N, other=0.0)
        gate_b, up_b = tl.split(tl.reshape(acc, (BLOCK // 2, 2)))
        gate_f = gate_b.to(tl.float32)
        exp_neg = tl.inline_asm_elementwise(
            "{ mul.ftz.f32 $0, $1, 0fBFB8AA3B; ex2.approx.ftz.f32 $0, $0; }",
            "=f,f",
            [gate_f],
            dtype=tl.float32,
            is_pure=True,
            pack=1,
        )
        silu_f = tl.inline_asm_elementwise(
            "div.approx.ftz.f32 $0, $1, $2;",
            "=f,f,f",
            [gate_f, 1.0 + exp_neg],
            dtype=tl.float32,
            is_pure=True,
            pack=1,
        )
        out = (silu_f * up_b.to(tl.float32)).to(acc.dtype)
        offs_h = tl.arange(0, BLOCK // 2)
        tl.store(out_ptr + offs_h, out, mask=offs_h < N // 2)

    torch.manual_seed(0)
    inter = 512
    # Ordinary range plus the saturating tails and signed zeros, where an
    # approx/ftz instruction and a libm-style sigmoid are most likely to part.
    tails = torch.tensor(
        [0.0, -0.0, 1e-8, -1e-8, 60.0, -60.0, 1e4, -1e4],
        dtype=torch.float32,
        device="cuda",
    )
    gate = torch.cat(
        [torch.randn(inter - tails.numel(), device="cuda") * 6.0, tails]
    ).to(torch.bfloat16)
    up = (torch.randn(inter, device="cuda") * 3.0).to(torch.bfloat16)

    ref = torch.empty(1, inter, dtype=torch.bfloat16, device="cuda")
    K.set_fused_op_backend(KernelBackend.JIT)
    try:
        silu_and_mul(torch.cat([gate, up]).unsqueeze(0), ref)
    finally:
        K.set_fused_op_backend(None)

    interleaved = torch.stack([gate, up], dim=-1).reshape(-1).contiguous()
    got = torch.empty(inter, dtype=torch.bfloat16, device="cuda")
    _epilogue_only[(1,)](interleaved, got, 2 * inter, BLOCK=2 * inter)

    mismatch = (got.view(torch.int16) != ref[0].view(torch.int16)).sum().item()
    assert mismatch == 0, f"{mismatch}/{inter} elements differ from silu_and_mul"


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
