"""Microbenchmark: Triton ``_fwd_grouped_kernel_stage1`` vs. the CUDA/HIP port.

Compares the Triton MLA decode stage-1 kernel (via ``_decode_grouped_att_m_fwd``)
against the hand-written CUDA/HIP kernel in
``sgl-kernel/csrc/attention/decode_grouped_attention_mla_stage1.cu``.

The CUDA kernel is JIT-built on first run via torch.utils.cpp_extension.load
(works on both NVIDIA and ROCm/MI3xx -- torch hipifies the sources on ROCm).

Regime: DeepSeek-style MLA (Lk=576 = 512 nope + 64 rope, Lv=512, H_KV=1, has_mla).

Run (on a GPU box):
    python benchmark/kernels/decoding_attention_triton/bench_stage1_triton_vs_cuda.py
    python .../bench_stage1_triton_vs_cuda.py --batch 32 --seq-len 8192 --check-only
"""

import argparse
import os

import torch
from torch.utils.cpp_extension import load

from sglang.kernels.ops.attention.decode_attention import _decode_grouped_att_m_fwd

# --- MLA constants (DeepSeek-V3 style) ---
KV_LORA_RANK = 512  # == Lv
QK_ROPE_HEAD_DIM = 64
LK = KV_LORA_RANK + QK_ROPE_HEAD_DIM  # 576
LV = KV_LORA_RANK  # 512

_HERE = os.path.dirname(os.path.abspath(__file__))


# The kernel uses gfx950 (CDNA4 / MI350) wide bf16 MFMA intrinsics, so build for
# that arch only. Override with SGL_STAGE1_ARCH if needed (e.g. "gfx942").
_ARCH = os.environ.get("SGL_STAGE1_ARCH", "gfx950")

# torch.utils.cpp_extension emits one --offload-arch per arch PyTorch was built
# for (e.g. gfx942;gfx950) UNLESS PYTORCH_ROCM_ARCH is set. Pin it so hipcc gets
# only our target arch and doesn't also codegen gfx942. Must be set before load().
os.environ["PYTORCH_ROCM_ARCH"] = _ARCH


def _build_cuda_ext(verbose: bool):
    """JIT-compile the HIP stage-1 kernel + torch binding for gfx950 only."""
    return load(
        name="sgl_stage1_mla_bench",
        sources=[
            os.path.join(_HERE, "_stage1_mla_binding.cu"),
            os.path.join(
                _HERE,
                "../../../sgl-kernel/csrc/attention/"
                "decode_grouped_attention_mla_stage1_fp8.cu",
            ),
        ],
        extra_cuda_cflags=["-O3"],
        verbose=verbose,
    )


def _time_kernel(fn, *, warmup: int, iters: int) -> float:
    """Mean latency in microseconds via CUDA events."""
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()

    start = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    end = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    for i in range(iters):
        start[i].record()
        fn()
        end[i].record()
    torch.cuda.synchronize()
    return (sum(s.elapsed_time(e) for s, e in zip(start, end)) / iters) * 1e3


def _make_inputs(*, batch, seq_len, h_q, max_kv_splits, num_kv_splits, device,
                 kv_dtype="bf16"):
    dtype = torch.bfloat16
    h_kv = 1
    total_tokens = batch * seq_len
    sm_scale = 1.0 / (LK**0.5)

    q = torch.randn(batch, h_q, LK, dtype=dtype, device=device)
    k_buffer = torch.randn(total_tokens, h_kv, LK, dtype=dtype, device=device)
    # fp8 KV cache: store K as e4m3 and hand the SAME fp8 buffer to both Triton
    # and the cuda kernel, so both run the identical fp8 recipe (q->fp8, fp8 dot,
    # p->fp8). k_scale = 1 here (randn is within e4m3 range); a real cache folds
    # k_scale into sm_scale, which both this kernel and Triton do.
    if kv_dtype == "fp8":
        k_buffer = k_buffer.to(torch.float8_e4m3fn)
    v_buffer = k_buffer[..., :LV]  # only used for strides/Lv (has_mla derives V)

    b_seq_len = torch.full((batch,), seq_len, dtype=torch.int64, device=device)
    kv_indptr = torch.zeros((batch + 1,), dtype=torch.int32, device=device)
    kv_indptr[1:] = torch.cumsum(b_seq_len, dim=0)
    kv_indices = torch.arange(total_tokens, dtype=torch.int32, device=device)
    num_kv_splits_t = torch.full(
        (batch,), num_kv_splits, dtype=torch.int32, device=device
    )

    return dict(
        q=q,
        k_buffer=k_buffer,
        v_buffer=v_buffer,
        kv_indptr=kv_indptr,
        kv_indices=kv_indices,
        num_kv_splits=num_kv_splits_t,
        sm_scale=sm_scale,
        max_kv_splits=max_kv_splits,
        h_q=h_q,
        total_tokens=total_tokens,
    )


def _empty_outputs(*, batch, h_q, max_kv_splits, device):
    # Zero-inited so inactive KV-splits (skipped identically by both kernels)
    # compare equal instead of comparing uninitialized memory.
    att_out = torch.zeros(
        (batch, h_q, max_kv_splits, LV), dtype=torch.float32, device=device
    )
    att_lse = torch.zeros(
        (batch, h_q, max_kv_splits), dtype=torch.float32, device=device
    )
    return att_out, att_lse


def _run_triton(inp, att_out, att_lse):
    _decode_grouped_att_m_fwd(
        inp["q"],
        inp["k_buffer"],
        inp["v_buffer"],
        att_out,
        att_lse,
        inp["kv_indptr"],
        inp["kv_indices"],
        inp["num_kv_splits"],
        inp["max_kv_splits"],
        inp["sm_scale"],
        logit_cap=0.0,
        has_mla=True,
    )


def _run_cuda(ext, inp, att_out, att_lse):
    fn = (ext.stage1_mla_fp8 if inp["k_buffer"].dtype == torch.float8_e4m3fn
          else ext.stage1_mla)
    fn(
        inp["q"], inp["k_buffer"], att_out, att_lse, inp["kv_indptr"],
        inp["kv_indices"], inp["num_kv_splits"], inp["sm_scale"],
        inp["max_kv_splits"],
    )


def _check(inp, batch, h_q, max_kv_splits, device):
    """Numerical parity on a single launch. Returns (out_max_err, lse_max_err)."""
    ext = _build_cuda_ext(verbose=True)

    out_t, lse_t = _empty_outputs(
        batch=batch, h_q=h_q, max_kv_splits=max_kv_splits, device=device
    )
    out_c, lse_c = _empty_outputs(
        batch=batch, h_q=h_q, max_kv_splits=max_kv_splits, device=device
    )
    _run_triton(inp, out_t, lse_t)
    _run_cuda(ext, inp, out_c, lse_c)
    torch.cuda.synchronize()

    out_err = (out_t - out_c).abs().max().item()
    lse_err = (lse_t - lse_c).abs().max().item()
    return ext, out_err, lse_err


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--batch", type=int, default=64)
    p.add_argument("--seq-len", type=int, default=4096)
    p.add_argument("--h-q", type=int, default=128, help="q head count (TP=1 DeepSeek)")
    p.add_argument("--max-kv-splits", type=int, default=8)
    p.add_argument("--num-kv-splits", type=int, default=4)
    p.add_argument("--warmup", type=int, default=20)
    p.add_argument("--iters", type=int, default=100)
    p.add_argument("--check-only", action="store_true", help="parity check, no timing")
    p.add_argument("--kv-dtype", choices=["bf16", "fp8"], default="bf16",
                   help="KV cache dtype (fp8 = e4m3, half the KV memory + HBM)")
    args = p.parse_args()
    assert args.num_kv_splits <= args.max_kv_splits, (
        f"num_kv_splits ({args.num_kv_splits}) > max_kv_splits "
        f"({args.max_kv_splits}): only the first max_kv_splits*kv_len_per_split "
        f"tokens would be attended (grid.z = max_kv_splits)."
    )

    assert torch.cuda.is_available(), "GPU device required"
    device = "cuda"
    print(f"device: {torch.cuda.get_device_name(0)}")
    print(f"MLA: Lk={LK} (BLOCK_DMODEL=512 + BLOCK_DPE=64), Lv={LV}, H_KV=1")
    print(
        f"B={args.batch} S={args.seq_len} H_Q={args.h_q} "
        f"splits={args.num_kv_splits}/{args.max_kv_splits} kv_dtype={args.kv_dtype}\n"
    )

    inp = _make_inputs(
        batch=args.batch,
        seq_len=args.seq_len,
        h_q=args.h_q,
        max_kv_splits=args.max_kv_splits,
        num_kv_splits=args.num_kv_splits,
        device=device,
        kv_dtype=args.kv_dtype,
    )

    # --- Parity (also triggers the JIT build) ---
    ext, out_err, lse_err = _check(
        inp, args.batch, args.h_q, args.max_kv_splits, device
    )
    # bf16 inputs, fp32 accum: expect small absolute error.
    ok = out_err < 5e-2 and lse_err < 5e-2
    print(
        f"\nparity: max|att_out| err = {out_err:.3e}, "
        f"max|att_lse| err = {lse_err:.3e}  ->  {'PASS' if ok else 'MISMATCH'}"
    )
    if args.check_only:
        return

    # --- Timing ---
    out_t, lse_t = _empty_outputs(
        batch=args.batch, h_q=args.h_q, max_kv_splits=args.max_kv_splits, device=device
    )
    out_c, lse_c = _empty_outputs(
        batch=args.batch, h_q=args.h_q, max_kv_splits=args.max_kv_splits, device=device
    )

    t_triton = _time_kernel(
        lambda: _run_triton(inp, out_t, lse_t),
        warmup=args.warmup,
        iters=args.iters,
    )
    t_cuda = _time_kernel(
        lambda: _run_cuda(ext, inp, out_c, lse_c),
        warmup=args.warmup,
        iters=args.iters,
    )

    kv_bytes = inp["total_tokens"] * 1 * LK * inp["q"].element_size()

    def _line(name, us):
        bw = kv_bytes / (us * 1e-6) / 1e9
        return f"{name:<8} | {us:8.2f} us | KV read {kv_bytes / 1e6:7.1f} MB | {bw:7.1f} GB/s"

    print()
    print(_line("triton", t_triton))
    print(_line("cuda", t_cuda))
    print(f"\nspeedup (triton/cuda): {t_triton / t_cuda:.2f}x")


if __name__ == "__main__":
    main()
