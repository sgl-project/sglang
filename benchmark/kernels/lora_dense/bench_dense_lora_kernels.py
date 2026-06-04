"""Self-contained micro-benchmark + correctness check for the small dense LoRA Triton
kernels: ``_sgemm_lora_a_kernel`` (shrink), ``_sgemm_lora_b_kernel`` (expand-add) and
``_gate_up_lora_b_kernel`` (2-slice expand-add).

Shapes are the FULL set measured e2e on Qwen3.5-35B-A3B-FP8 tp4/ep4 decode bs64 with a
single rank-16 adapter (SHAPECAP capture 2026-06-04), one entry per distinct module
signature -- see ``SHAPES`` below (in_proj_qkvz / qkv_proj / o_proj / shared expert
gate_up+down / lm_head).

Dispatch note (measured e2e): in the merged single-adapter decode batch, production
``sgemm_lora_a_fwd`` takes the ``F.linear`` fast path (the Triton shrink kernel never
runs); the Triton kernels of record for decode are the three *_lora_b expands. The
bench therefore reports, for every lora_a shape, BOTH the production ``F.linear`` path
and the Triton kernel path (the latter for optimization reference).

Decode batch_info reproduces production: merged single segment (bs=1, seg_lens=[64]),
permutation (SORTED_BY_ADAPTER=True), uniform adapter slot 0 / rank 16 / scaling 2.0,
use_cuda_graph=True.

Benchmark methodology: rotate N auto-sized buffer groups (footprint = ``--l2-mult`` x
L2, default 4x) inside one CUDA graph timed by ``triton.testing.do_bench_cudagraph``;
reported time = graph time / N. This amortizes host launch overhead to ~0 and prevents
any input from being served out of L2 on its next use.

  python3 bench_dense_lora_kernels.py --mode bench
  python3 bench_dense_lora_kernels.py --mode bench --only lm_head.B
  python3 bench_dense_lora_kernels.py --mode correctness
  python3 bench_dense_lora_kernels.py --mode profile --only o_proj.B --iters 4  # for ncu
"""

from __future__ import annotations

import argparse
import math

import torch
import triton
import triton.testing

from sglang.srt.lora.triton_ops.gate_up_lora_b import gate_up_lora_b_fwd
from sglang.srt.lora.triton_ops.sgemm_lora_a import sgemm_lora_a_fwd
from sglang.srt.lora.triton_ops.sgemm_lora_b import sgemm_lora_b_fwd
from sglang.srt.lora.utils import LoRABatchInfo

# All distinct dense-LoRA kernel signatures measured e2e (qwen3.5-35b tp4/ep4 decode
# bs64, rank 16). "calls/step" is per rank per decode step, for context.
#   kernel=sgemm_a: weights [1, stack_num*rank, K], x [s, K] -> out [s, stack_num*rank]
#   kernel=sgemm_b: weights [1, N, rank], x [s, rank] -> base_output [s, N] (+=)
#   kernel=gate_up_b: weights [1, 2*output_dim, rank], x [s, 2*rank] -> base [s, 2*output_dim] (+=)
SHAPES = [
    ("in_proj_qkvz.A", "sgemm_a", {"K": 2048, "stack_num": 4}),  # 30 calls/step
    ("qkv_proj.A", "sgemm_a", {"K": 2048, "stack_num": 3}),  # 10 calls/step
    ("shared_gate_up.A", "sgemm_a", {"K": 2048, "stack_num": 2}),  # 40 calls/step
    ("o_proj.A", "sgemm_a", {"K": 1024, "stack_num": 1}),  # 40 calls/step
    ("shared_down.A", "sgemm_a", {"K": 128, "stack_num": 1}),  # 40 calls/step
    ("lm_head.A", "sgemm_a", {"K": 2048, "stack_num": 1}),  # 1 call/step
    ("o_proj+shared_down.B", "sgemm_b", {"N": 2048}),  # 80 calls/step
    ("lm_head.B", "sgemm_b", {"N": 62080}),  # 1 call/step (vocab 248320 / tp4)
    ("shared_gate_up.B", "gate_up_b", {"output_dim": 128}),  # 40 calls/step
]


def make_merged_decode_batch_info(
    s: int,
    rank: int,
    scaling: float,
    device,
    with_single_adapter: bool,
    shuffle_permutation: bool = False,
) -> LoRABatchInfo:
    """Production single-adapter decode batch info: one merged segment (bs=1) with a
    token permutation. ``with_single_adapter`` toggles sgemm_lora_a_fwd's dispatch:
    True -> production F.linear fast path, False -> the Triton kernel path."""
    permutation = torch.arange(s, dtype=torch.int32, device=device)
    if shuffle_permutation:
        permutation = permutation[torch.randperm(s, device=device)]
    return LoRABatchInfo(
        use_cuda_graph=True,
        bs=1,
        num_segments=1,
        seg_indptr=torch.tensor([0, s], dtype=torch.int64, device=device),
        weight_indices=torch.zeros(1, dtype=torch.int32, device=device),
        lora_ranks=torch.tensor([rank], dtype=torch.int64, device=device),
        scalings=torch.tensor([scaling], dtype=torch.float32, device=device),
        max_len=s,
        seg_lens=torch.tensor([s], dtype=torch.int64, device=device),
        permutation=permutation,
        uniform_weight_index=0,
        uniform_rank=rank,
        uniform_scaling=scaling,
        single_adapter=(0, rank) if with_single_adapter else None,
    )


def make_inputs(name, kernel, spec, s, rank, dtype, device, seed=0):
    """Returns (x, weights, base_output_or_None) for one SHAPES entry."""
    gen = torch.Generator(device=device).manual_seed(seed)

    def rnd(*shape):
        return torch.randn(*shape, generator=gen, device=device, dtype=dtype) * 0.1

    if kernel == "sgemm_a":
        x = rnd(s, spec["K"])
        weights = rnd(1, spec["stack_num"] * rank, spec["K"])
        return x, weights, None
    if kernel == "sgemm_b":
        x = rnd(s, rank)
        weights = rnd(1, spec["N"], rank)
        return x, weights, rnd(s, spec["N"])
    assert kernel == "gate_up_b"
    x = rnd(s, 2 * rank)
    weights = rnd(1, 2 * spec["output_dim"], rank)
    return x, weights, rnd(s, 2 * spec["output_dim"])


def make_call(kernel, spec, x, weights, base_output, batch_info):
    if kernel == "sgemm_a":
        return lambda: sgemm_lora_a_fwd(
            x, weights, batch_info, stack_num=spec["stack_num"]
        )
    if kernel == "sgemm_b":
        return lambda: sgemm_lora_b_fwd(x, weights, batch_info, base_output=base_output)
    assert kernel == "gate_up_b"
    return lambda: gate_up_lora_b_fwd(
        x, weights, batch_info, spec["output_dim"], base_output=base_output
    )


def ref_output(kernel, spec, x, weights, base_output, rank, scaling):
    """fp32 reference. sgemm_a has no scaling and writes its own output; the two
    expand kernels scale and add into base_output."""
    w = weights[0].float()
    if kernel == "sgemm_a":
        return x.float() @ w.t()
    if kernel == "sgemm_b":
        return base_output.float() + scaling * (x.float() @ w.t())
    assert kernel == "gate_up_b"
    out = base_output.float().clone()
    output_dim = spec["output_dim"]
    for i in range(2):
        lo, hi = i * output_dim, (i + 1) * output_dim
        xi = x[:, i * rank : (i + 1) * rank].float()
        out[:, lo:hi] += scaling * (xi @ w[lo:hi, :rank].t())
    return out


def group_bytes_of(kernel, spec, s, rank) -> int:
    if kernel == "sgemm_a":
        return 2 * (s * spec["K"] + spec["stack_num"] * rank * spec["K"] + s * spec["stack_num"] * rank)
    if kernel == "sgemm_b":
        return 2 * (s * rank + spec["N"] * rank + s * spec["N"])
    return 2 * (s * 2 * rank + 2 * spec["output_dim"] * rank + s * 2 * spec["output_dim"])


def auto_num_groups(
    group_bytes: int, l2_mult: float, min_groups: int, max_groups: int
) -> int:
    """Enough buffer groups that the rotation footprint is ``l2_mult`` x L2, so no
    group survives in L2 until its next use. Err on the high side: an optimized kernel
    that reads less memory needs MORE groups for the same eviction guarantee."""
    props = torch.cuda.get_device_properties(torch.cuda.current_device())
    l2_bytes = getattr(props, "L2_cache_size", 128 * 1024 * 1024)
    need = math.ceil(l2_bytes * l2_mult / max(group_bytes, 1))
    return max(min_groups, min(need, max_groups))


def bench_us_rotated(calls, rep_ms: int) -> float:
    """Capture all rotated calls in ONE CUDA graph via do_bench_cudagraph; per-call us =
    graph time / num_groups."""

    def fn():
        for call in calls:
            call()

    fn()  # eager warmup: triton JIT compile outside graph capture
    torch.cuda.synchronize()
    ms = triton.testing.do_bench_cudagraph(fn, rep=rep_ms)
    return float(ms) * 1e3 / len(calls)


def variants_for(name, kernel):
    """sgemm_a gets both dispatch paths; the expand kernels only have the Triton path."""
    if kernel == "sgemm_a":
        return [("triton", False), ("F.linear(prod)", True)]
    return [("triton", None)]


def run_correctness(args, shapes, dtype, device) -> None:
    failures = 0
    for shuffle in [False, True]:
        for name, kernel, spec in shapes:
            for variant, single_adapter in variants_for(name, kernel):
                bi = make_merged_decode_batch_info(
                    args.bs,
                    args.rank,
                    args.scaling,
                    device,
                    with_single_adapter=bool(single_adapter),
                    shuffle_permutation=shuffle,
                )
                x, weights, base = make_inputs(
                    name, kernel, spec, args.bs, args.rank, dtype, device
                )
                base_run = base.clone() if base is not None else None
                out = make_call(kernel, spec, x, weights, base_run, bi)()
                if kernel != "sgemm_a":
                    out = base_run
                ref = ref_output(kernel, spec, x, weights, base, args.rank, args.scaling)
                err = float((out.float() - ref).abs().max().item())
                rel = err / float(ref.abs().max().item() + 1e-9)
                # bf16 output quantization makes the achievable ABS error scale with the
                # output magnitude (large for the K=2048 shrinks), so accept either bound.
                ok = err <= args.tol or rel <= args.rtol
                failures += int(not ok)
                print(
                    f"{'PASS' if ok else 'FAIL'} {name:<22s} {variant:<14s} "
                    f"shuffled={int(shuffle)} max_abs_err={err:.4e} rel={rel:.2e}"
                )
    if failures:
        raise SystemExit(1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--mode", choices=["bench", "correctness", "profile"], default="bench"
    )
    ap.add_argument("--only", default=None, help="run a single SHAPES entry by name")
    ap.add_argument("--bs", type=int, default=64, help="decode batch size (1 tok/req)")
    ap.add_argument("--rank", type=int, default=16)
    ap.add_argument("--scaling", type=float, default=2.0)
    ap.add_argument(
        "--num-groups",
        type=int,
        default=0,
        help="rotated buffer groups; 0 = auto-size to --l2-mult x L2",
    )
    ap.add_argument("--l2-mult", type=float, default=4.0)
    ap.add_argument("--min-groups", type=int, default=32)
    ap.add_argument("--max-groups", type=int, default=1024)
    ap.add_argument("--rep-ms", type=int, default=100)
    ap.add_argument("--iters", type=int, default=4, help="profile-mode eager sweeps")
    ap.add_argument("--tol", type=float, default=5e-2)
    ap.add_argument("--rtol", type=float, default=1e-2)
    args = ap.parse_args()

    device = "cuda"
    dtype = torch.bfloat16
    shapes = [e for e in SHAPES if args.only is None or e[0] == args.only]
    assert shapes, f"--only {args.only!r} matches no SHAPES entry"

    if args.mode == "correctness":
        run_correctness(args, shapes, dtype, device)
        return

    s = args.bs
    for name, kernel, spec in shapes:
        for variant, single_adapter in variants_for(name, kernel):
            bi = make_merged_decode_batch_info(
                s, args.rank, args.scaling, device, with_single_adapter=bool(single_adapter)
            )
            group_bytes = group_bytes_of(kernel, spec, s, args.rank)
            num_groups = args.num_groups or auto_num_groups(
                group_bytes, args.l2_mult, args.min_groups, args.max_groups
            )
            groups = [
                make_inputs(name, kernel, spec, s, args.rank, dtype, device, seed=g)
                for g in range(num_groups)
            ]
            calls = [
                make_call(kernel, spec, x, w, base, bi) for x, w, base in groups
            ]

            if args.mode == "profile":
                for _ in range(2):
                    calls[0]()
                torch.cuda.synchronize()
                for _ in range(args.iters):
                    for call in calls:
                        call()
                torch.cuda.synchronize()
                print(f"PROFILE {name} [{variant}]: {args.iters} x {num_groups} groups")
                continue

            us = bench_us_rotated(calls, args.rep_ms)
            dims = " ".join(f"{k}={v}" for k, v in spec.items())
            print(
                f"BENCH {name:<22s} [{variant:<14s}] s={s} r={args.rank} {dims:<22s} "
                f"groups={num_groups} ({group_bytes * num_groups / 1e6:.0f} MB rotated): "
                f"{us:.2f} us"
            )


if __name__ == "__main__":
    main()
