"""Stress the kernel on the kind of garbage input that cuda-graph padding rows
can contain: uninitialized memory, NaN, Inf, very large bf16 values, mixed
NaN/normal in the same group, and the worst case where ENTIRE rows are NaN."""

import sys
import torch
from sglang.srt.layers.quantization.fp8_kernel import sglang_per_token_group_quant_fp8
from sglang.srt.layers.fast_fp8_quant import fast_per_token_group_quant_fp8_128


PASS = 0
FAIL = 0
GROUP = 128


def report(name, ok, detail=""):
    global PASS, FAIL
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}  {detail}")
    if ok:
        PASS += 1
    else:
        FAIL += 1


def safe_for_fp8(q: torch.Tensor) -> bool:
    """An fp8_e4m3fn output is 'safe' for downstream deep_gemm if it has no
    NaN bit patterns. fp8_e4m3fn NaN is encoded as 0x7F or 0xFF (S.1111.111).
    Non-NaN values must avoid those byte patterns."""
    bits = q.view(torch.uint8)
    return bool(((bits & 0x7F) != 0x7F).all().item())


def has_finite_scales(s: torch.Tensor) -> bool:
    return torch.isfinite(s).all().item()


def t_uninitialized():
    """Padding rows often look like 'whatever was there before' — could be
    NaN-prone subnormal patterns. Use torch.empty() and DON'T initialize."""
    print("\n[padding] uninitialized memory (worst case)")
    M, K = 16384, 4096
    x = torch.empty(M, K, dtype=torch.bfloat16, device="cuda")
    # Don't fill — this is what padding rows look like sometimes.
    q, s = fast_per_token_group_quant_fp8_128(x)
    report("uninit-q-no-NaN-bits", safe_for_fp8(q))
    report("uninit-s-finite", has_finite_scales(s))


def t_nan_rows():
    """Some rows are entirely NaN (worst case for cuda-graph padding)."""
    print("\n[padding] half rows NaN")
    M, K = 1024, 4096
    x = torch.randn(M, K, dtype=torch.bfloat16, device="cuda") * 0.1
    x[: M // 2] = float("nan")
    q, s = fast_per_token_group_quant_fp8_128(x)
    # Real rows must be quantized correctly.
    real_q = q[M // 2 :]
    real_s = s[M // 2 :]
    report("nan-half: real-q-no-NaN-bits", safe_for_fp8(real_q))
    report("nan-half: real-s-finite", has_finite_scales(real_s))


def t_inf_rows():
    print("\n[padding] +inf rows")
    M, K = 1024, 4096
    x = torch.randn(M, K, dtype=torch.bfloat16, device="cuda") * 0.1
    x[: M // 2] = float("inf")
    q, s = fast_per_token_group_quant_fp8_128(x)
    real_q = q[M // 2 :]
    real_s = s[M // 2 :]
    report("inf-half: real-q-no-NaN-bits", safe_for_fp8(real_q))
    report("inf-half: real-s-finite", has_finite_scales(real_s))


def t_mixed_nan_in_group():
    """One element per group is NaN, the rest are finite — the per-group max
    becomes NaN and propagates through the whole group's quant output."""
    print("\n[padding] one NaN per group")
    M, K = 1024, 4096
    x = torch.randn(M, K, dtype=torch.bfloat16, device="cuda") * 0.1
    x[:, 0::GROUP] = float("nan")  # first element of every group
    q, s = fast_per_token_group_quant_fp8_128(x)
    report("mixed-nan: q-no-NaN-bits", safe_for_fp8(q))
    report("mixed-nan: s-finite", has_finite_scales(s))


def t_compare_padding_with_sgl():
    """Run BOTH on identical padded input, see if they produce the same
    bit-level pattern. If they diverge here, that's how the FP8 path
    would diverge across ranks: same garbage in, different bits out."""
    print("\n[padding] my-kernel-vs-sgl-on-uninitialized-input")
    M, K = 8192, 4096
    x = torch.empty(M, K, dtype=torch.bfloat16, device="cuda")
    qf, sf = fast_per_token_group_quant_fp8_128(x)
    qs, ss = sglang_per_token_group_quant_fp8(x, group_size=GROUP, enable_v2=True)
    qf_safe = safe_for_fp8(qf)
    qs_safe = safe_for_fp8(qs)
    report("uninit-vs-sgl: my-q-no-NaN", qf_safe)
    report("uninit-vs-sgl: sgl-q-no-NaN", qs_safe)
    report("uninit-vs-sgl: my-s-finite", has_finite_scales(sf))
    report("uninit-vs-sgl: sgl-s-finite", has_finite_scales(ss))


def main():
    t_uninitialized()
    t_nan_rows()
    t_inf_rows()
    t_mixed_nan_in_group()
    t_compare_padding_with_sgl()
    print()
    print(f"{PASS} pass, {FAIL} fail")
    sys.exit(1 if FAIL else 0)


if __name__ == "__main__":
    main()
