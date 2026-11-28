"""Reproduce cuBLAS BF16 GEMM rounding on SM80 (A100/A800), and compare the Triton backend.

    python benchmark/kernels/gemm/repro_cublas_bf16_fp32_sm80.py [--device cuda:0]

For each shape (M, N, K) of x [M, K] @ w[N, K].T it prints:
- bf16 out: share of elements of ``torch.mm(x, w.T)`` (BF16 output) that differ from
  the FP64 product rounded to BF16, with PyTorch's
  ``allow_bf16_reduced_precision_reduction`` on (default) and off;
- fp32 out: the operand offsets (1-1024 elements) at which
  ``torch.mm(x, w.T, out_dtype=torch.float32)`` gives different bits;
- the same for ``linear_bf16_fp32_triton`` (``SGLANG_OPT_BF16_FP32_GEMM_ALGO=triton``);
- time per call of both FP32-output GEMMs inside a CUDA graph.

Shapes are DeepSeek-V4.1-Flash at TP4: LM head and DSpark draft head (N 32320), indexer
and compressor projections (N 512 / 384 / 128 / 32), router (N 384).
"""

import argparse

import torch

SHAPES = [
    (1, 32320, 5120),
    (5, 32320, 5120),
    (5, 32320, 256),
    (128, 512, 5120),
    (710, 512, 5120),
    (256, 384, 5120),
    (9, 384, 5120),
    (710, 128, 512),
    (4, 32, 5120),
    (3, 1, 5376),
    (65, 384, 5120),
]
# Element offsets: 8 bf16 = 16 bytes, 128 = 256 bytes (PyTorch allocations are 512-aligned).
OFFSETS = (1, 2, 4, 8, 16, 32, 64, 128, 256, 1024)


def shifted(t, offset):
    buf = torch.empty(t.numel() + offset, dtype=t.dtype, device=t.device)
    view = buf[offset:].view(t.shape)
    view.copy_(t)
    return view


def address_changes(fn, x, w):
    """Offsets (in elements, x then w) at which the result's bits change; "-" if none."""
    first = fn(x, w).clone()
    bad = [f"x+{o}" for o in OFFSETS if not torch.equal(fn(shifted(x, o), w), first)]
    bad += [f"w+{o}" for o in OFFSETS if not torch.equal(fn(x, shifted(w, o)), first)]
    return " ".join(bad) or "-"


def time_us(fn, x, w, iters=50):
    """Microseconds per call inside a CUDA graph of back-to-back calls."""
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3):
            fn(x, w)
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        for _ in range(iters):
            fn(x, w)
    graph.replay()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    graph.replay()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) * 1000 / iters


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    dev = torch.device(args.device)
    props = torch.cuda.get_device_properties(dev)
    print(
        f"{props.name} sm{props.major}{props.minor}, torch {torch.__version__}, "
        f"cuBLAS {torch.backends.cuda.preferred_blas_library()}, "
        f"CUDA {torch.version.cuda}"
    )
    from sglang.kernels.ops.gemm.bf16_fp32_triton import linear_bf16_fp32_triton

    def cublas_fp32(a, b):
        return torch.mm(a, b.T, out_dtype=torch.float32)

    print(
        "| M | N | K | bf16 out wrong (reduced-precision on) | (off) "
        "| cuBLAS fp32 bits change at | Triton bits change at | Triton max rel err "
        "| cuBLAS fp32 us | Triton us |"
    )
    print("|---|---|---|---|---|---|---|---|---|---|")
    matmul = torch.backends.cuda.matmul
    default_reduced = matmul.allow_bf16_reduced_precision_reduction
    for m, n, k in SHAPES:
        torch.manual_seed(0)
        x = torch.randn(m, k, device=dev).bfloat16()
        w = (torch.randn(n, k, device=dev) / k**0.5).bfloat16()
        exact = x.double() @ w.double().T
        want = exact.bfloat16()
        wrong = []
        for reduced in (True, False):
            matmul.allow_bf16_reduced_precision_reduction = reduced
            got = torch.mm(x, w.T)
            wrong.append(f"{(got != want).float().mean().item():.1%}")
        matmul.allow_bf16_reduced_precision_reduction = default_reduced
        tri = linear_bf16_fp32_triton(x, w)
        rel = ((tri.double() - exact).norm() / exact.norm()).item()
        print(
            f"| {m} | {n} | {k} | {wrong[0]} | {wrong[1]} "
            f"| {address_changes(cublas_fp32, x, w)} "
            f"| {address_changes(linear_bf16_fp32_triton, x, w)} | {rel:.1e} "
            f"| {time_us(cublas_fp32, x, w):.1f} "
            f"| {time_us(linear_bf16_fp32_triton, x, w):.1f} |"
        )


if __name__ == "__main__":
    main()
