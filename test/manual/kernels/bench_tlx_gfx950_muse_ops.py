"""MI350X parity and timing of the vendored gfx950 TLX ops against the kernels they replace in ROCm Muse Spark.

Shapes are Muse-Spark-Open at TP8/EP8 per rank: 16 query and 2 KV heads of 64 (QKV + gate projection 2304 wide).
Widths are decode / verify CUDA-graph batch sizes plus prefill chunks.

Every op is timed with ``marker.do_bench`` (HIP graph, cold L2: inputs rotated past L2 and L2 flushed per replay)
next to the time its bytes take at peak HBM bandwidth (``--hbm-tbps``, MI350X 8 TB/s), and checked against the
incumbent: bitwise where the contracts promise it, otherwise the mismatch fraction and max error are printed.
Both kernels are plain Triton, so stock ROCm Triton runs this script.

    python test/manual/kernels/bench_tlx_gfx950_muse_ops.py [--ops rmsnorm_gate,qk_norm_rope]
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


OPS = {
    "rmsnorm_gate": bench_rmsnorm_gate,
    "qk_norm_rope": bench_qk_norm_rope,
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
