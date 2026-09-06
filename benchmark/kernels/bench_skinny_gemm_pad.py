"""Where does cuBLAS fall off the skinny-GEMM cliff, and does row padding fix it?

Times a bf16 ``[M, K] x [K, N]`` GEMM for M = 1..16, plain and padded with
``sglang.srt.utils.skinny_gemm_pad.skinny_gemm_pad_rows``, over every
Qwen3.5-family GDN ``in_proj_ba`` shape (N = 2 * num_v_heads / TP, K = hidden).
Weights rotate through ~160 MB so the timing is HBM-resident, like a decode
step. A "cliff" row is one where the plain kernel is > 1.5x the padded one.

    python benchmark/kernels/bench_skinny_gemm_pad.py
    python benchmark/kernels/bench_skinny_gemm_pad.py --shapes 48:5120 96:5120
"""

import argparse

import torch
import torch.nn.functional as F

from sglang.srt.utils.skinny_gemm_pad import skinny_gemm_pad_rows

QWEN3_5_SHAPES = [
    ("27B tp1", 96, 5120),
    ("27B tp2", 48, 5120),
    ("35B-A3B tp1", 64, 2048),
    ("35B-A3B tp2", 32, 2048),
    ("122B tp1", 128, 4096),
    ("122B tp2", 64, 4096),
    ("122B tp4", 32, 4096),
    ("397B tp1", 128, 6144),
    ("397B tp2", 64, 6144),
    ("397B tp4", 32, 6144),
]


def time_us(fn, iters):
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        fn()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) * 1e3 / iters


def bench_shape(name, n, k, device, max_m):
    rotation = max(2, (160 << 20) // (n * k * 2))
    weights = [
        torch.randn(n, k, dtype=torch.bfloat16, device=device) * 0.02
        for _ in range(rotation)
    ]
    plain_us, padded_us, cliff = [], [], []
    for m in range(1, max_m + 1):
        x = torch.randn(m, k, dtype=torch.bfloat16, device=device)
        pad_to = skinny_gemm_pad_rows(m=m, n=n, k=k)
        counter = [0]

        def plain():
            w = weights[counter[0] % rotation]
            counter[0] += 1
            return F.linear(x, w)

        def padded():
            w = weights[counter[0] % rotation]
            counter[0] += 1
            if not pad_to:
                return F.linear(x, w)
            return F.linear(F.pad(x, (0, 0, 0, pad_to - m)), w)[:m]

        for fn in (plain, padded):
            for _ in range(rotation):
                fn()
        t_plain, t_padded = time_us(plain, 2 * rotation), time_us(padded, 2 * rotation)
        plain_us.append(t_plain)
        padded_us.append(t_padded)
        if t_plain > 1.5 * t_padded:
            cliff.append(m)
    pad_to = skinny_gemm_pad_rows(m=2, n=n, k=k)
    print(f"{name:>12} N={n:<4} K={k:<5} pad_to={pad_to:<3} cliff M={cliff}")
    print(f"{'':>12}   M     : " + " ".join(f"{m:5d}" for m in range(1, max_m + 1)))
    print(f"{'':>12}   plain : " + " ".join(f"{t:5.1f}" for t in plain_us))
    print(f"{'':>12}   padded: " + " ".join(f"{t:5.1f}" for t in padded_us), flush=True)
    torch.cuda.empty_cache()


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max-m", type=int, default=16)
    parser.add_argument(
        "--shapes",
        nargs="*",
        help="N:K pairs; default is every Qwen3.5-family in_proj_ba shape",
    )
    args = parser.parse_args()
    torch.manual_seed(0)
    shapes = QWEN3_5_SHAPES
    if args.shapes:
        shapes = [(s, *map(int, s.split(":"))) for s in args.shapes]
    major, minor = torch.cuda.get_device_capability(args.device)
    print(
        f"{torch.cuda.get_device_name(args.device)} sm{major}{minor}"
        f" torch {torch.__version__} cuda {torch.version.cuda}; times in us"
    )
    for name, n, k in shapes:
        bench_shape(name, n, k, args.device, args.max_m)


if __name__ == "__main__":
    main()
