# SPDX-License-Identifier: Apache-2.0
"""Unit test for the fused JIT op ``silu_and_mul_scaled_fp4_experts_quant_packed``
(introduced in PR #18612).

On the CUTLASS NVFP4 MoE intermediate, the op fuses the previous two-step path

    intermediate = silu_and_mul(c1)                       # SiLU(gate) * up
    fp4, sf      = scaled_fp4_experts_quant(intermediate) # NVFP4 expert quant

into a single kernel

    fp4, sf = silu_and_mul_scaled_fp4_experts_quant_packed(c1, ...)

This test compares the fused op against that exact unfused
``silu_and_mul`` + ``scaled_fp4_experts_quant`` path **with uneven expert offsets**:
experts deliberately receive very different token counts, including tiny experts and
experts whose row count is not a multiple of the 128-row block-scale padding. That is
precisely the regime that stresses the per-expert ``expert_offsets`` /
``blockscale_offsets`` indexing the fusion has to get right.

It follows the two existing siblings:
  * ``test_silu_and_mul_quantize_to_fp4_grouped`` (the grouped/masked variant) -- the
    unfused path is the reference, and the fused output must match it bit-exactly
    (packed FP4 nibbles + recovered block scales), and
  * ``test_nvfp4_blockwise_moe`` (the expert-offset variant) -- offsets are built from
    an explicit, non-uniform per-expert token list.

A high-precision ``F.silu(gate) * up`` check additionally grounds the unfused path so a
bug shared by both kernels cannot produce a false (vacuous) pass.

    pytest python/sglang/jit_kernel/tests/test_silu_and_mul_scaled_fp4_experts_quant_packed.py -v
"""

import sys

import pytest
import torch
import triton
from torch.nn import functional as F

from sglang.jit_kernel.activation import silu_and_mul
from sglang.jit_kernel.nvfp4 import (
    scaled_fp4_experts_quant,
    silu_and_mul_scaled_fp4_experts_quant_packed,
)
from sglang.test.ci.ci_register import register_cuda_ci

# The NVFP4 expert-quant kernels are Blackwell-only (sm100a), so this runs on
# the B200 unit suite.
register_cuda_ci(est_time=20, stage="base-b-kernel-unit", runner_config="4-gpu-b200")

FLOAT8_E4M3_MAX = 448.0
FLOAT4_E2M1_MAX = 6.0
BLOCK_SIZE = 16
kE2M1ToFloat = torch.tensor(
    [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0], dtype=torch.float32
)


def _nvfp4_supported() -> bool:
    return torch.cuda.is_available() and torch.cuda.get_device_capability() >= (10, 0)


def _round_up(x: int, y: int) -> int:
    return ((x + y - 1) // y) * y


# --------------------------------------------------------------------------- #
# Offset builders (mirror test_nvfp4_blockwise_moe.py).
#   expert_offsets:     cumulative *actual* per-expert rows           ([E+1] int32)
#   blockscale_offsets: cumulative rows padded up to 128 per expert   ([E+1] int32)
# A non-uniform ``m_per_expert`` makes both offset tensors uneven.
# --------------------------------------------------------------------------- #
def _build_expert_offsets(m_per_expert, device) -> torch.Tensor:
    offsets = [0]
    for m in m_per_expert:
        offsets.append(offsets[-1] + m)
    return torch.tensor(offsets, dtype=torch.int32, device=device)


def _build_blockscale_offsets(m_per_expert, device) -> torch.Tensor:
    offsets = [0]
    for m in m_per_expert:
        offsets.append(offsets[-1] + _round_up(m, 128))
    return torch.tensor(offsets, dtype=torch.int32, device=device)


# --------------------------------------------------------------------------- #
# FP4 dequant / scale-recovery helpers (mirror test/registered/kernels/test_fp4_moe.py)
# --------------------------------------------------------------------------- #
def break_fp4_bytes(a: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    assert a.dtype == torch.uint8
    m, n = a.shape
    a_flat = a.flatten()
    high = (a_flat & 0xF0) >> 4
    low = a_flat & 0x0F
    combined = torch.stack((low, high), dim=1).flatten()
    signs = (combined & 0x08).to(torch.bool)
    abs_vals = (combined & 0x07).to(torch.long)
    kE2M1 = kE2M1ToFloat.to(device=a.device)
    values = kE2M1[abs_vals] * torch.where(signs, -1.0, 1.0)
    return values.reshape(m, n * 2).to(dtype=dtype)


def convert_swizzled_to_linear(
    a_sf_swizzled: torch.Tensor, m: int, k: int, block_size: int
) -> torch.Tensor:
    """De-swizzle one expert's block-scale region and drop the 128-row padding tail."""
    m_tiles = (m + 128 - 1) // 128
    f = block_size * 4
    k_tiles = (k + f - 1) // f
    tmp = torch.reshape(a_sf_swizzled, (1, m_tiles, k_tiles, 32, 4, 4))
    tmp = torch.permute(tmp, (0, 1, 4, 3, 2, 5))
    out = tmp.reshape(m_tiles * 128, k_tiles * f // block_size)
    return out[0:m, 0:k]


def dequantize_nvfp4_to_dtype(
    tensor_fp4: torch.Tensor,
    tensor_sf: torch.Tensor,
    global_scale: torch.Tensor,
    dtype: torch.dtype,
    device: torch.device,
    block_size: int = 16,
) -> torch.Tensor:
    """Dequantize one expert's packed FP4 (m, k//2) + swizzled block scales."""
    assert tensor_fp4.dtype == torch.uint8
    m, packed_k = tensor_fp4.shape
    k = packed_k * 2
    tensor_f32 = break_fp4_bytes(tensor_fp4, dtype)
    tensor_f32 = tensor_f32.reshape(m, k // block_size, block_size)
    tensor_sf = tensor_sf.view(torch.float8_e4m3fn)
    tensor_sf = convert_swizzled_to_linear(tensor_sf, m, k, block_size)
    tensor_sf_dtype = tensor_sf.to(torch.float32) / global_scale
    out = (tensor_f32 * tensor_sf_dtype.unsqueeze(-1)).reshape(m, k)
    return out.to(dtype=dtype)


def _recover_block_scales(
    sf: torch.Tensor, s0: int, s1: int, m_e: int, n: int
) -> torch.Tensor:
    """De-swizzled, un-padded block scales (float32) for one expert's region."""
    block = sf[s0:s1].contiguous().view(torch.float8_e4m3fn)
    return convert_swizzled_to_linear(block, m_e, n, BLOCK_SIZE).to(torch.float32)


def _rel_l2(a: torch.Tensor, b: torch.Tensor) -> float:
    return (a.float() - b.float()).norm().item() / b.float().norm().clamp_min(
        1e-9
    ).item()


# --------------------------------------------------------------------------- #
# Uneven per-expert token counts. Each list is deliberately NON-uniform so that
# expert_offsets / blockscale_offsets are uneven, exercising:
#   * tiny experts (1, 5, 7 tokens),
#   * an exactly-128 expert (no padding),
#   * experts straddling the 128-row block-scale padding (130, 200, 384).
# --------------------------------------------------------------------------- #
UNEVEN_M_PER_EXPERT = [
    [33, 17, 48, 29],  # all < 128 (matches test_nvfp4_blockwise_moe)
    [1, 128, 200, 5, 64],  # tiny + exactly-128 + cross-128
    [130, 1, 384, 17, 96, 7],  # heavy skew, large dynamic range
]
NS = [256, 768]  # 768 == Qwen3-30B-A3B moe_intermediate_size
DTYPES = [torch.bfloat16, torch.float16]


@pytest.mark.skipif(
    not _nvfp4_supported(),
    reason="NVFP4 fused expert-quant kernel requires compute capability >= 10.0 (B200/SM100).",
)
@pytest.mark.parametrize("m_per_expert", UNEVEN_M_PER_EXPERT)
@pytest.mark.parametrize("n", NS)
@pytest.mark.parametrize("dtype", DTYPES)
@torch.inference_mode()
def test_fused_matches_unfused_uneven_offsets(m_per_expert, n, dtype):
    torch.manual_seed(0)
    device = torch.device("cuda")
    num_experts = len(m_per_expert)

    # --- uneven expert offsets ---
    expert_offsets = _build_expert_offsets(m_per_expert, device)
    blockscale_offsets = _build_blockscale_offsets(m_per_expert, device)
    total_m = int(expert_offsets[-1].item())
    counts = torch.tensor(m_per_expert)
    assert (
        counts.max() >= 2 * counts.min()
    ), "expert offsets must be uneven for this test"

    # gate+up concatenated input (m, 2n); /5 keeps values in a sane FP4 range.
    c1 = torch.randn((total_m, 2 * n), dtype=dtype, device=device) / 5.0
    gate, up = c1[:, :n].float(), c1[:, n:].float()
    ref = F.silu(gate) * up  # high-precision SiLU(gate) * up, (total_m, n) fp32

    # Per-expert global scale, exactly like cutlass_moe builds a2_gscale.
    gscale = torch.empty(num_experts, dtype=torch.float32, device=device)
    for e in range(num_experts):
        r0, r1 = int(expert_offsets[e]), int(expert_offsets[e + 1])
        amax = ref[r0:r1].abs().max().clamp_min(1e-6)
        gscale[e] = FLOAT8_E4M3_MAX * FLOAT4_E2M1_MAX / amax

    # topk only gates the buffer-size assertion in the wrapper; the per-expert
    # layout is driven entirely by the offsets. Both paths use the same value.
    topk = 1

    # ---- fused (new op) ----
    fused_fp4, fused_sf = silu_and_mul_scaled_fp4_experts_quant_packed(
        c1, gscale, expert_offsets, blockscale_offsets, topk
    )

    # ---- unfused (the exact path the op replaced) ----
    intermediate = torch.empty((total_m, n), dtype=dtype, device=device)
    silu_and_mul(c1, intermediate)
    unf_fp4, unf_sf = scaled_fp4_experts_quant(
        intermediate, gscale, expert_offsets, blockscale_offsets, topk
    )

    assert fused_fp4.shape == unf_fp4.shape == (total_m, n // 2)

    # Per-expert, bit-exact comparison honoring the uneven offsets.
    for e in range(num_experts):
        r0, r1 = int(expert_offsets[e]), int(expert_offsets[e + 1])
        s0, s1 = int(blockscale_offsets[e]), int(blockscale_offsets[e + 1])
        m_e = r1 - r0

        # (1) Packed FP4 nibbles are identical: same NVFP4 quantizer, same fused
        # SiLU(gate)*up rounded to the storage dtype before quantization.
        torch.testing.assert_close(
            fused_fp4[r0:r1], unf_fp4[r0:r1], msg=f"FP4 bytes differ for expert {e}"
        )

        # (2) Recovered (de-swizzled, un-padded) block scales are identical.
        torch.testing.assert_close(
            _recover_block_scales(fused_sf, s0, s1, m_e, n),
            _recover_block_scales(unf_sf, s0, s1, m_e, n),
            msg=f"block scales differ for expert {e}",
        )

        # (3) Grounding: the unfused path really reproduces SiLU(gate)*up within FP4
        # error, so (1)/(2) cannot pass vacuously on a bug shared by both kernels.
        deq = dequantize_nvfp4_to_dtype(
            unf_fp4[r0:r1].contiguous(),
            unf_sf[s0:s1].contiguous(),
            gscale[e],
            dtype,
            device,
            BLOCK_SIZE,
        )
        assert (
            _rel_l2(deq, ref[r0:r1]) < 0.2
        ), f"expert {e}: unfused dequant does not match SiLU(gate)*up reference"


# --------------------------------------------------------------------------- #
# Performance. The fusion removes, on the MoE down-projection input, one
# intermediate buffer allocation, one extra kernel launch, and a full HBM
# round-trip of the SiLU(gate)*up result. The speedup is measured under CUDA
# graphs -- the steady-state GPU memory-traffic saving, matching how SGLang
# executes graphed decode (the credible, low-noise number; eager wall-clock is
# dominated by launch/dispatch overhead and is too noisy to assert on). The
# assert is only a conservative regression floor; the printed speedup is the real
# result. Mirrors test_cutedsl_gdn_performance, in the same kernel unit suite.
#
# Tokens are spread evenly across experts here (the representative throughput
# case) at realistic Qwen3-30B-A3B MoE dims (n=768, 128 experts), swept from a
# decode batch up to a prefill chunk; the uneven-offset corner cases are covered
# by the correctness test above.
# --------------------------------------------------------------------------- #
PERF_SHAPES = [
    (1024, 768, 128),
    (4096, 768, 128),
    (16384, 768, 128),
]


def _even_offsets(total_tokens, num_experts, device):
    base, rem = divmod(total_tokens, num_experts)
    m_per_expert = [base + (1 if i < rem else 0) for i in range(num_experts)]
    return (
        _build_expert_offsets(m_per_expert, device),
        _build_blockscale_offsets(m_per_expert, device),
    )


@pytest.mark.skipif(
    not _nvfp4_supported(),
    reason="NVFP4 fused expert-quant kernel requires compute capability >= 10.0 (B200/SM100).",
)
@pytest.mark.parametrize("total_tokens,n,num_experts", PERF_SHAPES)
@torch.inference_mode()
def test_fused_perf_not_regressed(total_tokens, n, num_experts):
    device = torch.device("cuda")
    dtype = torch.bfloat16
    expert_offsets, blockscale_offsets = _even_offsets(
        total_tokens, num_experts, device
    )
    c1 = torch.randn((total_tokens, 2 * n), dtype=dtype, device=device) / 5.0
    gscale = torch.empty(num_experts, dtype=torch.float32, device=device)
    for e in range(num_experts):
        r0, r1 = int(expert_offsets[e]), int(expert_offsets[e + 1])
        amax = c1[r0:r1].abs().max().to(torch.float32).clamp_min(1e-6)
        gscale[e] = FLOAT8_E4M3_MAX * FLOAT4_E2M1_MAX / amax
    topk = 1

    def fused():
        silu_and_mul_scaled_fp4_experts_quant_packed(
            c1, gscale, expert_offsets, blockscale_offsets, topk
        )

    def unfused():
        # The exact path the op replaced: alloc the intermediate, SiLU*mul into
        # it, then quantize it -- one extra buffer + kernel + HBM round-trip.
        intermediate = torch.empty((total_tokens, n), dtype=dtype, device=device)
        silu_and_mul(c1, intermediate)
        scaled_fp4_experts_quant(
            intermediate, gscale, expert_offsets, blockscale_offsets, topk
        )

    g_f = triton.testing.do_bench_cudagraph(fused)  # ms, median
    g_u = triton.testing.do_bench_cudagraph(unfused)
    cuda_graph_speedup = g_u / g_f
    print(
        f"\n  [PERF] tokens={total_tokens:>6} n={n} E={num_experts}:  "
        f"unfused {g_u * 1e3:6.1f}us  fused {g_f * 1e3:6.1f}us  "
        f"cuda-graph speedup = {cuda_graph_speedup:.2f}x"
    )
    # Regression guard only: the fusion must not make this op slower. The actual
    # win carries a wide margin over this floor, so shared-runner noise cannot
    # flake it.
    assert (
        cuda_graph_speedup >= 1.05
    ), f"fused regressed under cuda-graph: {cuda_graph_speedup:.2f}x"


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v", "-s"]))
