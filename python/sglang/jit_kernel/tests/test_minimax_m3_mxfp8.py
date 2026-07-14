# SPDX-License-Identifier: Apache-2.0
"""Reference-vs-fused unit tests for the MiniMax-M3 ROCm native MXFP8 ops.

Each fused kernel has a slow PyTorch / dequant-to-bf16 reference; these assert
the two agree within tolerance:

  * Fused MXFP8 activation quant (Triton)        -> torch reference
  * Native MXFP8 linear (tl.dot_scaled)          -> dequant-to-bf16 @ matmul
  * Native MXFP8 MoE (dot_scaled grouped GEMM)   -> dequant-to-bf16 MoE math

ROCm-only. The pure quant test runs on any ROCm arch; the native MXFP8
``dot_scaled`` linear/MoE tests are gated to CDNA4 gfx95x (the hardware
microscaling matrix cores) -- gfx942 has no native ``dot_scaled`` MX path.

Run:  pytest python/sglang/jit_kernel/tests/test_minimax_m3_mxfp8.py -v
"""

import pytest
import torch

from sglang.srt.utils import is_hip

if not is_hip():
    pytest.skip(
        "MiniMax-M3 native MXFP8 ops are the ROCm path.", allow_module_level=True
    )
if not torch.cuda.is_available():
    pytest.skip("Requires a GPU.", allow_module_level=True)

from sglang.kernels.ops.quantization.mxfp8_amd_gfx95 import (  # noqa: E402
    _mxfp8_dot_scaled_linear,
    _mxfp8_e4m3_quantize_torch,
    _mxfp8_e4m3_quantize_triton,
    dequant_mxfp8_to_bf16,
)

DEVICE = "cuda"


def _gcn_arch() -> str:
    try:
        return torch.cuda.get_device_properties(0).gcnArchName
    except Exception:  # pragma: no cover - no device / non-AMD
        return ""


requires_gfx950 = pytest.mark.skipif(
    "gfx95" not in _gcn_arch(),
    reason="native MXFP8 dot_scaled is a CDNA4 (gfx95x) feature; "
    "gfx942 has no native dot_scaled MX path.",
)


def _relerr(a: torch.Tensor, b: torch.Tensor) -> float:
    a = a.float()
    b = b.float()
    return ((a - b).norm() / (b.norm() + 1e-8)).item()


# --------------------------------------------------------------------------- #
# Fused MXFP8 activation quant (Triton vs torch reference)
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("shape", [(64, 4096), (1, 6144), (333, 2048)])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@torch.inference_mode()
def test_mxfp8_quant_triton_matches_torch(shape, dtype):
    torch.manual_seed(0)
    x = torch.randn(*shape, device=DEVICE, dtype=dtype)
    xq_t, s_t = _mxfp8_e4m3_quantize_torch(x)
    xq_k, s_k = _mxfp8_e4m3_quantize_triton(x)
    assert s_k.shape == s_t.shape == (shape[0], shape[1] // 32)
    # E8M0 block exponents share the round-up ceil(log2(amax/e4m3_max))+127
    # algorithm; allow at most a 1-step difference at exact powers of two.
    assert (s_k.int() - s_t.int()).abs().max().item() <= 1
    # Dequantized values agree to fp8 granularity.
    deq_t = dequant_mxfp8_to_bf16(xq_t, s_t)
    deq_k = dequant_mxfp8_to_bf16(xq_k, s_k)
    assert _relerr(deq_k, deq_t) < 1e-2


@pytest.mark.parametrize("m,inter", [(8, 512), (65, 2048)])
@torch.inference_mode()
def test_minimax_swiglu_mxfp8_quant_matches_unfused_fp32(m, inter):
    # The fused swiglu+quant kernel keeps the activation in fp32 through the
    # E8M0 scale selection (no bf16 round-trip; matches the vLLM/ame kernel), so
    # the reference is the unfused fp32 swiglu followed by MXFP8 quant. Not
    # bit-identical because the reference quant runs in torch vs the fused triton
    # path, but numerically equivalent (tight relerr, scales agree within 1 ulp).
    from sglang.jit_kernel.minimax_m3 import (
        swiglu_oai_mxfp8_quant,
        swiglu_oai_split,
    )
    from sglang.kernels.ops.quantization.mxfp8_amd_gfx95 import mxfp8_e4m3_quantize

    torch.manual_seed(0)
    alpha, beta, limit = 1.702, 1.0, 7.0
    gate_up = torch.randn(m, 2 * inter, device=DEVICE, dtype=torch.bfloat16) * 0.5

    act = swiglu_oai_split(
        gate_up, alpha=alpha, beta=beta, limit=limit, out_dtype=torch.float32
    )
    q_ref, s_ref = mxfp8_e4m3_quantize(act)
    q, s = swiglu_oai_mxfp8_quant(gate_up, alpha=alpha, beta=beta, limit=limit)

    assert q.shape == q_ref.shape
    assert s.shape == s_ref.shape
    # E8M0 block scales agree within one exponent step (last-bit amax differences).
    assert (s.int() - s_ref.int()).abs().max().item() <= 1
    assert (
        _relerr(dequant_mxfp8_to_bf16(q, s), dequant_mxfp8_to_bf16(q_ref, s_ref)) < 1e-2
    )


# --------------------------------------------------------------------------- #
# Native MXFP8 linear (dot_scaled) vs dequant-to-bf16 matmul
# --------------------------------------------------------------------------- #
@requires_gfx950
@pytest.mark.parametrize("m,n,k", [(64, 256, 128), (37, 512, 256), (1, 6144, 4096)])
@torch.inference_mode()
def test_mxfp8_native_linear(m, n, k):
    torch.manual_seed(0)
    w_bf16 = torch.randn(n, k, device=DEVICE, dtype=torch.bfloat16) * 0.1
    w_fp8, w_scale = _mxfp8_e4m3_quantize_torch(w_bf16)
    x = torch.randn(m, k, device=DEVICE, dtype=torch.bfloat16) * 0.5

    got = _mxfp8_dot_scaled_linear(x, w_fp8, w_scale)
    # Reference consumes the SAME quantized weights (isolates activation-quant
    # noise) -> dequant to bf16, plain matmul.
    w_deq = dequant_mxfp8_to_bf16(w_fp8, w_scale)
    ref = torch.nn.functional.linear(x, w_deq).to(x.dtype)
    assert got.shape == (m, n)
    assert _relerr(got, ref) < 5e-2


# --------------------------------------------------------------------------- #
# Native MXFP8 MoE (dot_scaled grouped GEMM) vs dequant-to-bf16 MoE math
# --------------------------------------------------------------------------- #
def _ref_moe(x, w13, w2, topk_weights, topk_ids, alpha, beta, limit):
    T, H = x.shape
    inter = w2.shape[-1]
    top_k = topk_ids.shape[1]
    out = torch.zeros(T, H, device=x.device, dtype=torch.float32)
    for t in range(T):
        for j in range(top_k):
            e = int(topk_ids[t, j].item())
            if e < 0 or e >= w13.shape[0]:
                continue
            g1 = x[t].float() @ w13[e].float().T  # [2I]
            gate = g1[:inter]
            up = g1[inter:]
            if limit is not None:
                gate = gate.clamp(max=limit)
                up = up.clamp(min=-limit, max=limit)
            act = gate * torch.sigmoid(alpha * gate) * (up + beta)
            g2 = act @ w2[e].float().T  # [H]
            out[t] += topk_weights[t, j].float() * g2
    return out.to(x.dtype)


@requires_gfx950
@pytest.mark.parametrize(
    "T,H,inter,E,top_k", [(8, 256, 512, 8, 2), (1, 512, 256, 16, 4)]
)
@torch.inference_mode()
def test_mxfp8_native_moe(T, H, inter, E, top_k):
    from sglang.kernels.ops.moe.mxfp8_moe_amd_gfx95 import (
        fused_moe_mxfp8_native,
    )

    torch.manual_seed(0)
    alpha, beta, limit = 1.702, 1.0, 7.0
    w13_bf16 = torch.randn(E, 2 * inter, H, device=DEVICE, dtype=torch.bfloat16) * 0.1
    w2_bf16 = torch.randn(E, H, inter, device=DEVICE, dtype=torch.bfloat16) * 0.1
    w13_fp8, w13_scale = _mxfp8_e4m3_quantize_torch(w13_bf16)
    w2_fp8, w2_scale = _mxfp8_e4m3_quantize_torch(w2_bf16)

    x = torch.randn(T, H, device=DEVICE, dtype=torch.bfloat16) * 0.5
    logits = torch.randn(T, E, device=DEVICE, dtype=torch.float32)
    topk_weights, topk_ids = logits.softmax(dim=-1).topk(top_k, dim=-1)
    topk_weights = topk_weights.to(torch.float32)
    topk_ids = topk_ids.to(torch.int32)

    got = fused_moe_mxfp8_native(
        x,
        w13_fp8,
        w13_scale,
        w2_fp8,
        w2_scale,
        topk_weights,
        topk_ids,
        alpha=alpha,
        beta=beta,
        limit=limit,
    )
    # Reference consumes the dequantized weights (same bits the kernel reads).
    w13_deq = dequant_mxfp8_to_bf16(w13_fp8, w13_scale)
    w2_deq = dequant_mxfp8_to_bf16(w2_fp8, w2_scale)
    ref = _ref_moe(x, w13_deq, w2_deq, topk_weights, topk_ids, alpha, beta, limit)
    assert got.shape == (T, H)
    assert _relerr(got, ref) < 5e-2


@requires_gfx950
@torch.inference_mode()
def test_mxfp8_native_moe_ep_expert_map_filters_non_local_routes():
    from sglang.kernels.ops.moe.mxfp8_moe_amd_gfx95 import (
        fused_moe_mxfp8_native,
    )

    torch.manual_seed(0)
    T, H, inter = 4, 256, 512
    local_E = 3
    alpha, beta, limit = 1.702, 1.0, 7.0

    w13_bf16 = (
        torch.randn(local_E, 2 * inter, H, device=DEVICE, dtype=torch.bfloat16) * 0.1
    )
    w2_bf16 = torch.randn(local_E, H, inter, device=DEVICE, dtype=torch.bfloat16) * 0.1
    w13_fp8, w13_scale = _mxfp8_e4m3_quantize_torch(w13_bf16)
    w2_fp8, w2_scale = _mxfp8_e4m3_quantize_torch(w2_bf16)

    x = torch.randn(T, H, device=DEVICE, dtype=torch.bfloat16) * 0.5
    topk_ids_global = torch.tensor(
        [[0, 1, 4], [2, 3, 5], [4, 0, 3], [5, 1, 2]],
        device=DEVICE,
        dtype=torch.int32,
    )
    topk_weights = torch.tensor(
        [
            [0.50, 0.25, 0.25],
            [0.40, 0.30, 0.30],
            [0.70, 0.20, 0.10],
            [0.60, 0.30, 0.10],
        ],
        device=DEVICE,
        dtype=torch.float32,
    )
    expert_map = torch.tensor([0, -1, 1, -1, 2, -1], device=DEVICE, dtype=torch.int32)

    got = fused_moe_mxfp8_native(
        x,
        w13_fp8,
        w13_scale,
        w2_fp8,
        w2_scale,
        topk_weights,
        topk_ids_global,
        alpha=alpha,
        beta=beta,
        limit=limit,
        expert_map=expert_map,
    )

    topk_ids_local = expert_map[topk_ids_global.long()]
    w13_deq = dequant_mxfp8_to_bf16(w13_fp8, w13_scale)
    w2_deq = dequant_mxfp8_to_bf16(w2_fp8, w2_scale)
    ref = _ref_moe(x, w13_deq, w2_deq, topk_weights, topk_ids_local, alpha, beta, limit)
    assert got.shape == (T, H)
    assert _relerr(got, ref) < 5e-2


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__, "-v"]))
