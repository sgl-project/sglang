"""MI350X parity and timing of the vendored gfx950 TLX ops against the kernels they replace in ROCm Muse Spark.

Shapes are Muse-Spark-Open at TP8/EP8 per rank: 16 query and 2 KV heads of 64 (QKV + gate projection 2304 wide).
Widths are decode / verify CUDA-graph batch sizes plus prefill chunks.

Every op is timed with ``marker.do_bench`` (HIP graph, cold L2: inputs rotated past L2 and L2 flushed per replay)
next to the time its bytes take at peak HBM bandwidth (``--hbm-tbps``, MI350X 8 TB/s), and checked against the
incumbent: bitwise where the contracts promise it, otherwise the mismatch fraction and max error are printed.
``rmsnorm_gate`` and ``qk_norm_rope`` are plain Triton and run on stock ROCm Triton; ``paged_decode`` uses TLX
primitives and needs Triton with the uTLX plugin (SGLang's ``BUILD_UTLX=1`` ROCm image).

    python test/manual/kernels/bench_tlx_gfx950_muse_ops.py [--ops rmsnorm_gate,qk_norm_rope,paged_decode]
"""

import argparse
import json

import torch

from sglang.kernels.jit.benchmark import marker

DEVICE = "cuda"
DECODE_ROWS = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512]
PREFILL_ROWS = [2048, 8192]
HQ, HK, D, PAGE = 16, 2, 64, 16


def _time_us(fn, args, clone=None):
    result = marker.do_bench(
        fn,
        input_args=args,
        graph_clone_args=clone,
        metrics=(0.5,),
        disable_log_bandwidth=True,
    )
    return result.times[0] * 1e6


def _nbytes(*tensors):
    return sum(t.numel() * t.element_size() for t in tensors if t is not None)


def _mismatch(a, b):
    a, b = a.float(), b.float()
    diff = (a - b).abs()
    return {
        "mismatch_frac": (diff != 0).float().mean().item(),
        "max_abs": diff.max().item() if diff.numel() else 0.0,
    }


def _bitwise(a, b):
    return bool(torch.equal(a.view(torch.uint8), b.view(torch.uint8)))


def _row(op, rows, impls, bound_us, check):
    best = min(impls, key=impls.get)
    record = {
        "op": op,
        "rows": rows,
        "bound_us": round(bound_us, 2),
        **{f"{name}_us": round(t, 2) for name, t in impls.items()},
        "fastest": best,
        **check,
    }
    print(json.dumps(record), flush=True)
    return record


def bench_rmsnorm_gate(args):
    from sglang.kernels.ops.layernorm.tlx import rmsnorm_gate, rmsnorm_gate_supported
    from sglang.srt.rocm_muse_spark.fused_norm_gate import FusedRMSNormGate
    from sglang.srt.rocm_muse_spark.layers import MuseSpark1xRMSNorm

    eps = 1e-6
    incumbent = FusedRMSNormGate(
        MuseSpark1xRMSNorm(D, eps=eps, has_weight=False), use_o_gate=True, head_dim=D
    )
    width = (2 * HQ + 2 * HK) * D
    for rows in DECODE_ROWS + PREFILL_ROWS:
        x = torch.randn(rows, HQ * D, device=DEVICE, dtype=torch.bfloat16)
        qkv = torch.randn(rows, width, device=DEVICE, dtype=torch.bfloat16)
        gate = qkv[:, (HQ + 2 * HK) * D :]
        assert rmsnorm_gate_supported(x, gate)
        impls = {
            "triton": _time_us(incumbent.forward_triton, (x, gate), clone=[0, 1]),
            "tlx": _time_us(
                lambda x, g: rmsnorm_gate(x, g, eps), (x, gate), clone=[0, 1]
            ),
        }
        check = {
            "bitwise_vs_triton": _bitwise(
                rmsnorm_gate(x, gate, eps), incumbent.forward_triton(x, gate)
            )
        }
        _row("rmsnorm_gate[16x64]", rows, impls, _nbytes(x, x, x) / args.hbm, check)


def _qk_inputs(rows, swa):
    width = (2 * HQ + 2 * HK) * D
    qkv = torch.randn(rows, width, device=DEVICE, dtype=torch.bfloat16)
    q, k, v, _ = qkv.split([HQ * D, HK * D, HK * D, HQ * D], dim=-1)
    num_blocks = max(64, 4 * rows // PAGE + 8)
    cache = torch.zeros(
        num_blocks, PAGE, HK, D, device=DEVICE, dtype=torch.float8_e4m3fn
    )
    slots = torch.randperm(num_blocks * PAGE, device=DEVICE)[:rows].to(torch.int64)
    remap = (
        torch.randperm(num_blocks * PAGE, device=DEVICE).to(torch.int64)
        if swa
        else None
    )
    positions = torch.randint(0, 32768, (rows,), device=DEVICE, dtype=torch.int64)
    inv = 1.0 / (10000 ** (torch.arange(0, D, 2, device=DEVICE).float() / D))
    angles = torch.arange(32768, device=DEVICE).float()[:, None] * inv[None, :]
    cos_sin = torch.cat([angles.cos(), angles.sin()], dim=-1).contiguous()
    k_scale = torch.tensor([0.05], device=DEVICE)
    v_scale = torch.tensor([0.04], device=DEVICE)
    return (
        q,
        k,
        v,
        cache,
        cache.clone(),
        slots,
        remap,
        positions,
        cos_sin,
        k_scale,
        v_scale,
    )


def bench_qk_norm_rope(args):
    from sglang.kernels.ops.attention.qk_norm_rope_cache import qk_norm_rope_cache
    from sglang.kernels.ops.attention.tlx import (
        qk_norm_rope_kv_cache,
        qk_norm_rope_kv_cache_supported,
    )

    eps, q_scale = 1e-6, 0.125
    for swa in (False, True):
        for rows in DECODE_ROWS + PREFILL_ROWS:
            q, k, v, kc, vc, slots, remap, pos, cos_sin, ks, vs = _qk_inputs(rows, swa)
            assert qk_norm_rope_kv_cache_supported(q, k, v, kc)

            def triton_fn(q, k, v, kc, vc):
                return qk_norm_rope_cache(
                    q, k, v, pos, cos_sin, kc.view(-1, HK, D), vc.view(-1, HK, D),
                    slots, remap, eps, q_scale, ks, vs,
                )[0]  # fmt: skip

            def tlx_fn(q, k, v, kc, vc):
                return qk_norm_rope_kv_cache(
                    q, k, v, kc, vc, slots, remap, pos, cos_sin, eps, q_scale, ks, vs
                )

            impls = {
                "triton": _time_us(triton_fn, (q, k, v, kc, vc), clone=[0, 1, 2]),
                "tlx": _time_us(tlx_fn, (q, k, v, kc, vc), clone=[0, 1, 2]),
            }
            caches = [(kc.clone().zero_(), vc.clone().zero_()) for _ in range(2)]
            q_tri = triton_fn(q, k, v, *caches[0])
            q_tlx = tlx_fn(q, k, v, *caches[1])
            check = {
                "q": _mismatch(q_tlx, q_tri),
                "k_cache_bitwise": _bitwise(caches[1][0], caches[0][0]),
                "v_cache_bitwise": _bitwise(caches[1][1], caches[0][1]),
            }
            moved = _nbytes(q, k, v, q) + 2 * rows * HK * D
            name = "qk_norm_rope_kv_cache" + ("[swa]" if swa else "")
            _row(name, rows, impls, moved / args.hbm, check)


PAGED_DECODE_ROWS = [1, 2, 4, 8, 16, 32, 64, 128]
CONTEXT, WINDOW, MAX_CONTEXT = 8704, 2047, 32768


def _decode_reference(q, kc, vc, bt, ctx, window, gate, scale, eps):
    """FP32 attention over the visible keys, then the weightless per-head RMSNorm times sigmoid(gate)."""
    rows = q.shape[0]
    lo = max(0, ctx - window) if window else 0
    pages = bt[:rows, : (ctx + PAGE - 1) // PAGE].long()
    k = kc[pages].float().reshape(rows, -1, HK, D)[:, lo:ctx]
    v = vc[pages].float().reshape(rows, -1, HK, D)[:, lo:ctx]
    qf = q.float().view(rows, HK, HQ // HK, D)
    p = torch.softmax(torch.einsum("bgqd,bkgd->bgqk", qf, k) * scale, dim=-1)
    o = torch.einsum("bgqk,bkgd->bgqd", p, v).reshape(rows, HQ, D)
    o = o * torch.rsqrt(o.pow(2).mean(-1, keepdim=True) + eps)
    return (o.reshape(rows, HQ * D) * torch.sigmoid(gate.float())).to(torch.bfloat16)


def bench_paged_decode(args):
    """One-token decode + the attention-output norm-gate, as aiter_backend.forward_decode and the model run it:
    up to 8 rows muse_decode_splitkv with the split-K merge folded into the norm-gate; above 8 rows the FP8 query
    quant, aiter unified attention and the Triton norm-gate. TLX takes the BF16 query and folds the norm-gate into
    its reduce."""
    from aiter.ops.triton.attention.unified_attention import unified_attention

    from sglang.kernels.ops.attention.swa_verify_splitkv import muse_decode_splitkv
    from sglang.kernels.ops.attention.tlx import (
        paged_attention_decode,
        paged_attention_decode_supported,
    )
    from sglang.kernels.ops.quantization.fp8_kernel import scaled_fp8_quant
    from sglang.srt.rocm_muse_spark.fused_norm_gate import FusedRMSNormGate
    from sglang.srt.rocm_muse_spark.layers import MuseSpark1xRMSNorm

    eps, scale, fp8 = 1e-6, D**-0.5, torch.float8_e4m3fn
    norm_gate = FusedRMSNormGate(
        MuseSpark1xRMSNorm(D, eps=eps, has_weight=False), use_o_gate=True, head_dim=D
    )
    one = torch.ones(1, device=DEVICE)
    for window in (WINDOW, 0):
        for rows in PAGED_DECODE_ROWS:
            pages = (CONTEXT + PAGE - 1) // PAGE
            blocks = rows * pages
            kc = (torch.randn(blocks, PAGE, HK, D, device=DEVICE) * 0.5).to(fp8)
            vc = (torch.randn(blocks, PAGE, HK, D, device=DEVICE) * 0.5).to(fp8)
            # CUDA-graph decode passes the whole page-table buffer: more rows than the batch, MAX_CONTEXT wide.
            bt = torch.zeros(
                rows + 8, MAX_CONTEXT // PAGE, dtype=torch.int32, device=DEVICE
            )
            perm = torch.randperm(blocks, device=DEVICE).to(torch.int32)
            bt[:rows, :pages] = perm.view(rows, pages)
            seq = torch.full((rows,), CONTEXT, dtype=torch.int64, device=DEVICE)
            cu = torch.arange(rows + 1, dtype=torch.int32, device=DEVICE)
            q = torch.randn(rows, HQ * D, device=DEVICE, dtype=torch.bfloat16)
            gate = torch.randn(rows, HQ * D, device=DEVICE, dtype=torch.bfloat16)
            out = torch.empty_like(q)
            assert paged_attention_decode_supported(q.view(rows, HQ, D), kc, vc, bt)

            if rows <= 8:

                def incumbent(q, gate):
                    state = muse_decode_splitkv(
                        q.view(rows, HQ, D), kc, vc, out.view(rows, HQ, D), cu, seq,
                        bt, scale, window, one, one, defer_reduce=True,
                    )  # fmt: skip
                    return norm_gate.forward_verify_partials((*state, one), out, gate)

            else:
                window_size = (window - 1, 0) if window else (-1, -1)

                def incumbent(q, gate):
                    qf, _ = scaled_fp8_quant(q, one)
                    unified_attention(
                        q=qf.view(rows, HQ, D), k=kc, v=vc, out=out.view(rows, HQ, D),
                        cu_seqlens_q=cu, max_seqlen_q=1, seqused_k=seq, max_seqlen_k=MAX_CONTEXT,
                        softmax_scale=scale, causal=True, window_size=window_size, block_table=bt,
                        softcap=0, q_descale=one, k_descale=one, v_descale=one,
                    )  # fmt: skip
                    return norm_gate.forward_triton(out, gate)

            def tlx(q, gate):
                o = torch.empty_like(q)
                paged_attention_decode(
                    q.view(rows, HQ, D), kc, vc, o.view(rows, HQ, D), seq, bt, scale,
                    window, MAX_CONTEXT, one, one, out_gate=gate, out_norm_eps=eps,
                )  # fmt: skip
                return o

            impls = {
                "incumbent": _time_us(incumbent, (q, gate), clone=[0, 1]),
                "tlx": _time_us(tlx, (q, gate), clone=[0, 1]),
            }
            ref = _decode_reference(q, kc, vc, bt, CONTEXT, window, gate, scale, eps)
            check = {
                "tlx_vs_fp32_ref": _mismatch(tlx(q, gate), ref)["max_abs"],
                "incumbent_vs_fp32_ref": _mismatch(incumbent(q, gate), ref)["max_abs"],
            }
            visible = min(CONTEXT, window) if window else CONTEXT
            moved = 2 * rows * visible * HK * D + _nbytes(q, gate, q)
            name = "paged_decode[" + ("window" if window else "full") + "]"
            _row(name, rows, impls, moved / args.hbm, check)


OPS = {
    "rmsnorm_gate": bench_rmsnorm_gate,
    "qk_norm_rope": bench_qk_norm_rope,
    "paged_decode": bench_paged_decode,
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ops", default=",".join(OPS))
    parser.add_argument("--hbm-tbps", type=float, default=8.0)
    args = parser.parse_args()
    args.hbm = args.hbm_tbps * 1e12 / 1e6  # bytes per microsecond

    torch.manual_seed(0)
    print(
        json.dumps({"device": torch.cuda.get_device_name(), "torch": torch.__version__})
    )
    for op in args.ops.split(","):
        OPS[op](args)


if __name__ == "__main__":
    main()
