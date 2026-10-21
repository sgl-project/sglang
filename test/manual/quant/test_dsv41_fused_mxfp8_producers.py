import sys
from types import SimpleNamespace

import deep_gemm
import torch
from flashinfer import block_scale_interleave

from sglang.kernels.ops.layernorm.mhc_mega import mhc_mega_boundary
from sglang.srt.layers.attention.deepseek_v4_mega_backend import quantize_wo_a
from sglang.srt.layers.quantization.fp8_utils import (
    Mxfp8DenseGemmBackend,
    block_fp8_scale_to_mxfp8_e8m0,
    deepgemm_mxfp8_prequantized_linear,
    flashinfer_mxfp8_blockscaled_linear,
    prepare_deepgemm_mxfp8_input_path,
)
from sglang.srt.layers.quantization.mxfp8_input import Mxfp8DeepGemmInput

torch.manual_seed(0)
dev = "cuda"


def block_fp8_layer(n, k, std=0.02):
    w = torch.randn(n, k, device=dev) * std
    blk = w.view(n // 32, 32, k // 32, 32).abs().amax(dim=(1, 3)).clamp(min=1e-8)
    sf = torch.exp2(torch.ceil(torch.log2(blk / 448.0)))
    wq = (
        (w.view(n // 32, 32, k // 32, 32) / sf[:, None, :, None])
        .view(n, k)
        .to(torch.float8_e4m3fn)
    )
    method = SimpleNamespace(
        mxfp8_dense_backend=Mxfp8DenseGemmBackend.FLASHINFER_CUTEDSL,
        use_mxfp8=False,
        block_fp8_as_mxfp8=True,
        weight_block_size=[32, 32],
    )
    layer = SimpleNamespace(
        weight=wq,
        weight_scale_inv=SimpleNamespace(data=sf),
        quant_method=method,
        block_fp8_mxfp8_ready=True,
    )
    u8 = block_fp8_scale_to_mxfp8_e8m0(sf, (n, k), [32, 32])
    layer.weight_scale_inv_swizzled = block_scale_interleave(
        u8.contiguous()
    ).contiguous()
    assert prepare_deepgemm_mxfp8_input_path(layer)
    w_deq = (wq.float().view(n // 32, 32, k // 32, 32) * sf[:, None, :, None]).view(
        n, k
    )
    return layer, w_deq


def fi_linear(layer, x):
    return flashinfer_mxfp8_blockscaled_linear(
        x,
        layer.weight,
        layer.weight_scale_inv_swizzled,
        backend="cute-dsl",
        pin_tactic=True,
    )


def rel(a, b):
    return ((a.float() - b.float()).norm() / b.float().norm()).item()


def check_wqkv_a(m, H=5120, hc=4):
    layer, w_deq = block_fp8_layer(1792, H)
    x = torch.randn(m, H, device=dev, dtype=torch.bfloat16)
    residual = torch.randn(m, hc, H, device=dev, dtype=torch.bfloat16)
    pre = torch.rand(m, hc, device=dev) * 0.5
    post = torch.rand(m, hc, device=dev)
    comb = torch.rand(m, hc, hc, device=dev).softmax(-1)
    mix = (2 + hc) * hc
    fn = torch.randn(mix, hc * H, device=dev) * 0.01
    scale = torch.rand(3, device=dev)
    base = torch.randn(mix, device=dev) * 0.1
    weight = (torch.rand(H, device=dev) + 0.5).to(torch.bfloat16)
    args = (
        x,
        residual,
        pre,
        post,
        comb,
        fn,
        scale,
        base,
        weight,
        1e-6,
        1e-6,
        1e-20,
        20,
    )
    upd0, y0, st0 = mhc_mega_boundary(*args)
    upd1, y1, st1, (yq, ysf) = mhc_mega_boundary(*args, emit_mxfp8=True)
    assert torch.equal(upd0, upd1) and torch.equal(y0, y1), (
        "bf16 outputs changed with emit_mxfp8"
    )
    assert all(torch.equal(a, b) for a, b in zip(st0, st1))
    cur = fi_linear(layer, y0)
    new = deepgemm_mxfp8_prequantized_linear(layer, Mxfp8DeepGemmInput(yq, ysf))
    ref = y0.float() @ w_deq.t()
    print(
        f"wqkv_a M={m}: rel(new, current)={rel(new, cur):.2e}  rel(current, fp32 ref)={rel(cur, ref):.2e}"
        f"  rel(new, fp32 ref)={rel(new, ref):.2e}  bitwise-equal={torch.equal(new, cur)}"
    )
    return rel(new, cur)


def check_wo_b(m, G=8, R=1024, D=4096, H=5120):
    layer, w_deq = block_fp8_layer(H, G * R, std=0.01)
    wo_a = torch.randn(G, R, D, device=dev, dtype=torch.bfloat16) * 0.02
    wq, wsf = quantize_wo_a(wo_a)
    o = torch.randn(m, G, D, device=dev, dtype=torch.bfloat16)
    from deep_gemm.utils.layout import get_mn_major_tma_aligned_packed_ue8m0_tensor
    from deep_gemm.utils.math import per_token_cast_to_fp8

    oq, osf = per_token_cast_to_fp8(o.view(m, G * D), use_ue8m0=True, gran_k=32)
    osf = get_mn_major_tma_aligned_packed_ue8m0_tensor(osf)
    oq = oq.view(m, G, D)
    osf = osf.view(m, G, D // 128)
    o_bf = torch.empty(m, G, R, device=dev, dtype=torch.bfloat16)
    deep_gemm.fp8_einsum("bhr,hdr->bhd", (oq, osf), (wq, wsf), o_bf, recipe=(1, 1, 32))
    cur = fi_linear(layer, o_bf.view(m, G * R))
    o_q = torch.empty(m, G, R, device=dev, dtype=torch.float8_e4m3fn)
    o_sf = torch.empty_strided(
        (m, G * R // 128),
        (1, deep_gemm.get_tma_aligned_size(m, 4)),
        dtype=torch.int32,
        device=dev,
    )
    deep_gemm.fp8_einsum(
        "bhr,hdr->bhd", (oq, osf), (wq, wsf), (o_q, o_sf), recipe=(1, 1, 32)
    )
    new = deepgemm_mxfp8_prequantized_linear(
        layer, Mxfp8DeepGemmInput(o_q.view(m, -1), o_sf)
    )
    ref = o_bf.view(m, -1).float() @ w_deq.t()
    q_ref, sf_ref = per_token_cast_to_fp8(o_bf.view(m, -1), use_ue8m0=True, gran_k=32)
    sf_ref = get_mn_major_tma_aligned_packed_ue8m0_tensor(sf_ref)
    code_match = (o_q.view(m, -1).float() == q_ref.float()).float().mean().item()
    sf_match = (o_sf == sf_ref).float().mean().item()
    print(
        f"wo_b   M={m}: rel(new, current)={rel(new, cur):.2e}  rel(current, fp32 ref)={rel(cur, ref):.2e}"
        f"  rel(new, fp32 ref)={rel(new, ref):.2e}  einsum fp8 codes == quant(bf16 out): {code_match:.4f}, scales {sf_match:.4f}"
    )
    return rel(new, cur)


worst = 0.0
for m in [int(v) for v in (sys.argv[1:] or ["1", "4", "37", "256", "1024", "2048"])]:
    worst = max(worst, check_wqkv_a(m), check_wo_b(m))
print(f"worst rel(new, current) = {worst:.2e}")
assert worst < 2e-2, "fused path drifts beyond MXFP8 tolerance"
print("PASS")
