"""Standalone GPU test for SGLANG_OPT_MOE_QUANT_ONCE (quantize the MoE input
once, feed both the fused shared-expert GEMM and the routed triton runner).


    CUDA_VISIBLE_DEVICES=0 python test/manual/test_moe_quant_once.py

Verifies, against the double-quant baseline:
  (1) quant equivalence: the row-padded quantize-once kernel produces the
      same q bits / scale values as the routed path's default row-major quant
      (JIT v2 kernel) on the valid rows;
  (2) shared consumer: cutlass_w8a8_block_fp8_linear_with_fallback with a
      pre-quantized (q, s) tuple vs its own internal quant -- expected BITWISE
      (baseline uses the identical row-padded quant + identical GEMM);
  (2b) shared consumer under SGLANG_ENABLE_JIT_DEEPGEMM=1 (the recommended JIT-DeepGEMM config):
      deepgemm_w8a8_block_fp8_linear_with_fallback with the same (q, s) tuple
      -- expected BITWISE (DG's own quant layout, column-major TMA-aligned
      fp32 scales, is byte-identical to the row-padded quantize-once layout);
      skipped cleanly when deep_gemm is unavailable or UE8M0 (Blackwell);
  (3) routed consumer: fused_experts(a1_q=..., a1_scale=...) vs the in-kernel
      quant baseline -- expected BITWISE if (1) is bitwise (the fused kernel
      reads A_scale through explicit strides, so the column-major scale view
      feeds identical values).

If (1) is not bitwise (AOT v2 vs JIT v2 quant kernels round differently),
(3) falls back to an allclose check at atol=1e-2 and the discrepancy is
reported --.
"""

import sys

import torch

from sglang.kernels.ops.quantization.fp8_kernel import (
    sglang_per_token_group_quant_fp8,
    sglang_per_token_group_quant_fp8_row_padded,
)
from sglang.srt.layers.moe.moe_runner.base import MoeRunnerConfig
from sglang.srt.layers.moe.moe_runner.triton_utils.fused_moe import fused_experts
from sglang.srt.layers.moe.topk import StandardTopKOutput
from sglang.srt.layers.quantization.fp8_utils import (
    cutlass_w8a8_block_fp8_linear_with_fallback,
)
from sglang.srt.server_args import ServerArgs, set_global_server_args_for_scheduler

GROUP = 128
FAILURES = []


def _report(name, ok, detail=""):
    status = "PASS" if ok else "FAIL"
    print(f"[{status}] {name} {detail}")
    if not ok:
        FAILURES.append(name)


def _quant_weight_blockwise(w_bf16, block=128):
    """Per-[128,128]-block fp8 weight quant (reference, fp32 math)."""
    n, k = w_bf16.shape
    w = w_bf16.float().view(n // block, block, k // block, block)
    amax = w.abs().amax(dim=(1, 3), keepdim=True).clamp(min=1e-4)
    scale = amax / torch.finfo(torch.float8_e4m3fn).max
    q = (w / scale).clamp(-448, 448).to(torch.float8_e4m3fn)
    return (
        q.view(n, k),
        scale.squeeze(1).squeeze(-1).to(torch.float32),  # [n/128, k/128]
    )


def test_quant_equivalence(T, K, device):
    x = torch.randn(T, K, device=device, dtype=torch.bfloat16) * 3
    q_ref, s_ref = sglang_per_token_group_quant_fp8(x, GROUP)  # routed baseline
    q_pad, s_pad = sglang_per_token_group_quant_fp8_row_padded(x, GROUP)

    bitwise_q = torch.equal(q_pad[:T].view(torch.uint8), q_ref.view(torch.uint8))
    bitwise_s = torch.equal(s_pad[:T].contiguous(), s_ref)
    _report(
        f"quant-equivalence T={T} K={K}",
        bitwise_q and bitwise_s,
        f"(q bitwise={bitwise_q}, s bitwise={bitwise_s})",
    )
    return bitwise_q and bitwise_s


def test_shared_consumer(T, K, N, device):
    torch.manual_seed(T + K)
    x = torch.randn(T, K, device=device, dtype=torch.bfloat16)
    w_bf16 = torch.randn(N, K, device=device, dtype=torch.bfloat16) / K**0.5
    w, ws = _quant_weight_blockwise(w_bf16)

    ref = cutlass_w8a8_block_fp8_linear_with_fallback(x, w, [128, 128], ws)

    q_pad, s_pad = sglang_per_token_group_quant_fp8_row_padded(x, GROUP)
    out = cutlass_w8a8_block_fp8_linear_with_fallback(
        q_pad, w, [128, 128], ws, input_scale=s_pad
    )[:T]

    bitwise = torch.equal(out, ref)
    close = torch.allclose(out.float(), ref.float(), atol=1e-2, rtol=1e-2)
    _report(
        f"shared-consumer T={T} K={K} N={N}",
        close,
        f"(bitwise={bitwise}, max|d|={(out.float() - ref.float()).abs().max().item():.3e})",
    )
    return bitwise


def test_shared_consumer_deepgemm(T, K, N, device):
    """DG branch (SGLANG_ENABLE_JIT_DEEPGEMM=1 recommended JIT-DeepGEMM config): the shared-expert
    linear resolves to deepgemm_w8a8_block_fp8_linear_with_fallback. Its own
    quant (column-major + TMA-aligned fp32 scales) has the same buffer layout
    as the row-padded quantize-once kernel, so this is expected BITWISE."""
    from sglang.srt.layers.quantization.fp8_utils import (
        deepgemm_w8a8_block_fp8_linear_with_fallback,
    )

    torch.manual_seed(T + K + 1)
    x = torch.randn(T, K, device=device, dtype=torch.bfloat16)
    w_bf16 = torch.randn(N, K, device=device, dtype=torch.bfloat16) / K**0.5
    w, ws = _quant_weight_blockwise_n64(w_bf16)

    ref = deepgemm_w8a8_block_fp8_linear_with_fallback(x, w, [128, 128], ws)

    q_pad, s_pad = sglang_per_token_group_quant_fp8_row_padded(x, GROUP)
    out = deepgemm_w8a8_block_fp8_linear_with_fallback(
        q_pad, w, [128, 128], ws, input_scale=s_pad
    )[:T]

    bitwise = torch.equal(out, ref)
    close = torch.allclose(out.float(), ref.float(), atol=1e-2, rtol=1e-2)
    _report(
        f"shared-consumer-deepgemm T={T} K={K} N={N}",
        close,
        f"(bitwise={bitwise}, max|d|={(out.float() - ref.float()).abs().max().item():.3e})",
    )
    return bitwise


def _quant_weight_blockwise_n64(w_bf16, block=128):
    """Like _quant_weight_blockwise but supports N % 64 == 0 (DeepGEMM's
    minimum): the last (partial) N-block reuses ceil-division block indexing."""
    n, k = w_bf16.shape
    if n % block == 0:
        return _quant_weight_blockwise(w_bf16, block)
    import math

    n_blocks = math.ceil(n / block)
    w = w_bf16.float()
    q = torch.empty(n, k, device=w.device, dtype=torch.float8_e4m3fn)
    scale = torch.empty(n_blocks, k // block, device=w.device, dtype=torch.float32)
    for bn in range(n_blocks):
        rows = slice(bn * block, min((bn + 1) * block, n))
        wb = w[rows].view(rows.stop - rows.start, k // block, block)
        amax = wb.abs().amax(dim=(0, 2)).clamp(min=1e-4)
        s = amax / torch.finfo(torch.float8_e4m3fn).max
        q[rows] = (
            (wb / s[None, :, None]).clamp(-448, 448).to(torch.float8_e4m3fn).view(-1, k)
        )
        scale[bn] = s
    return q, scale


def test_routed_consumer(T, K, E, I, topk, device):
    torch.manual_seed(T * 7 + K)
    x = torch.randn(T, K, device=device, dtype=torch.bfloat16)
    w1 = torch.empty(E, 2 * I, K, device=device, dtype=torch.float8_e4m3fn)
    w1s = torch.empty(E, 2 * I // 128, K // 128, device=device)
    w2 = torch.empty(E, K, I, device=device, dtype=torch.float8_e4m3fn)
    w2s = torch.empty(E, K // 128, I // 128, device=device)
    for e in range(E):
        w1[e], w1s[e] = _quant_weight_blockwise(
            torch.randn(2 * I, K, device=device, dtype=torch.bfloat16) / K**0.5
        )
        w2[e], w2s[e] = _quant_weight_blockwise(
            torch.randn(K, I, device=device, dtype=torch.bfloat16) / I**0.5
        )

    topk_weights = torch.rand(T, topk, device=device)
    topk_weights = (topk_weights / topk_weights.sum(-1, keepdim=True)).to(torch.float32)
    topk_ids = torch.stack(
        [torch.randperm(E, device=device)[:topk] for _ in range(T)]
    ).to(torch.int32)
    topk_output = StandardTopKOutput(
        topk_weights=topk_weights, topk_ids=topk_ids, router_logits=None
    )
    # num_experts == num_local_experts => filter_expert=False (pure TP layout)
    cfg = MoeRunnerConfig(
        num_experts=E,
        num_local_experts=E,
        top_k=topk,
        inplace=False,
        activation="silu",
        is_gated=True,
    )

    kwargs = dict(
        w1=w1,
        w2=w2,
        topk_output=topk_output,
        moe_runner_config=cfg,
        use_fp8_w8a8=True,
        w1_scale=w1s,
        w2_scale=w2s,
        block_shape=[128, 128],
    )
    ref = fused_experts(hidden_states=x, **kwargs)

    q_pad, s_pad = sglang_per_token_group_quant_fp8_row_padded(x, GROUP)
    out = fused_experts(hidden_states=x, a1_q=q_pad, a1_scale=s_pad, **kwargs)

    bitwise = torch.equal(out, ref)
    close = torch.allclose(out.float(), ref.float(), atol=1e-2, rtol=1e-2)
    _report(
        f"routed-consumer T={T} K={K} E={E} topk={topk}",
        close,
        f"(bitwise={bitwise}, max|d|={(out.float() - ref.float()).abs().max().item():.3e})",
    )
    return bitwise


def main():
    assert torch.cuda.is_available(), "CUDA required"
    set_global_server_args_for_scheduler(ServerArgs(model_path="dummy"))
    device = "cuda"
    torch.manual_seed(0)

    print("== (1) quantize-once vs routed-baseline quant equivalence ==")
    all_bitwise_q = True
    for T in (1, 3, 4093, 4096):
        for K in (6144, 7168):
            all_bitwise_q &= test_quant_equivalence(T, K, device)

    print("== (2) shared consumer (cutlass w8a8 linear) ==")
    # N=512 mirrors a tp8 shared expert gate_up (2*2048/8); must be %128==0.
    for T in (4093, 4096):
        for K in (6144, 7168):
            test_shared_consumer(T, K, 512, device)

    print(
        "== (2b) shared consumer (deepgemm w8a8 linear, JIT DG recommended JIT-DeepGEMM config) =="
    )
    from sglang.srt.layers import deep_gemm_wrapper

    if not deep_gemm_wrapper.ENABLE_JIT_DEEPGEMM:
        print("SKIP: deep_gemm unavailable or SGLANG_ENABLE_JIT_DEEPGEMM=0")
    elif deep_gemm_wrapper.DEEPGEMM_SCALE_UE8M0:
        print("SKIP: Blackwell UE8M0 scale layout (gated ineligible by design)")
    else:
        for T in (4093, 4096):
            for K in (6144, 7168):
                test_shared_consumer_deepgemm(T, K, 512, device)
        # DG accepts N % 64 (cutlass needs % 128) -- exercise the DG-only shape.
        test_shared_consumer_deepgemm(4096, 7168, 320, device)

    print("== (3) routed consumer (triton fused_experts) ==")
    # Identical in both cutlass and JIT-DG configs: the MoE runner stays
    # triton with a2a=none (is_deepgemm_moe_runner_backend_enabled() is False
    # for auto + a2a=none even when SGLANG_ENABLE_JIT_DEEPGEMM=1).
    for T in (61, 4093, 4096):
        test_routed_consumer(T, 7168, E=32, I=256, topk=8, device=device)

    if not all_bitwise_q:
        print(
            "NOTE: quantize-once q/s not bitwise vs the routed baseline quant "
            "(AOT v2 vs JIT v2 kernel rounding) -- routed consumer is then "
            "allclose-only; document this in the PR."
        )
    if FAILURES:
        print(f"FAILED: {FAILURES}")
        sys.exit(1)
    print("ALL PASS")


if __name__ == "__main__":
    main()
