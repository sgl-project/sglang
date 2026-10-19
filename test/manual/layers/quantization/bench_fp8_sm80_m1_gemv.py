# SPDX-License-Identifier: Apache-2.0
"""SM80 (A100) M=1 FP8-E4M3 group-32 GEMV vs the production gptq_marlin path.

Run on the GPU host (JIT-builds the C++ Marlin baseline on first use):
    PYTHONPATH=python MAX_JOBS=1 python3 \\
        test/manual/layers/quantization/bench_fp8_sm80_m1_gemv.py
Stdout is one JSON object per line; the last JSON line is a summary and the
final plain marker is SM80_M1_GEMV_{PASS,FAIL}.

WHY
---
The SM80 dense FP8 Marlin path reaches only ~20-26% of A100 HBM peak on these
M=1 decode shapes, so the question is whether a purpose-built memory-bound M=1
GEMV can stream the *same raw bytes* faster.  This file answers that with
measurements, and also shows where the answer is structurally capped:

  * N=1792 and N=576 have very little output parallelism.  A design with one
    program per N-block can only field N/BLOCK_N thread blocks; with BLOCK_N=64
    that is 28 (resp. 18) blocks on a 108-SM part, i.e. ~1 warp per SM, which is
    latency- not bandwidth-bound.  That is the most likely reason Marlin sits
    near 20% of peak on (1792,5120) and (576,5120).  The K-split configs below
    exist to restore occupancy for exactly those shapes.
  * The small-K shapes ((4096,1280), (5120,1024), (5120,320)) are latency and
    launch bound, so the interesting number there is effective GB/s.

NUMERICAL CONTRACT REPRODUCED HERE (Marlin's BF16/FP8 group-32 arithmetic)
-------------------------------------------------------------------------
For output column n and k-group g = k // 32:
    w_eff[n, k] = bf16( bf16(fp8_e4m3_to_bf16(raw[n, k])) * bf16(scale[n, g]) )
    out[n]      = bf16( sum_k fp32( w_eff[n, k] * x[k] ) )
Both operand roundings are BF16, there is exactly one BF16 rounding of the
product, the accumulation is FP32 and the result is stored as BF16.  No FP16, no
TF32, and the scale is never fused into the accumulator.

The kernel reads the RAW row-major FP8 weight (N, K) as uint8 and the BF16
scales (N, K/32) contiguous: no repacking, no Marlin layout, no predecoded copy.
The Marlin baseline is fed the checkpoint-style block scale (N/32, K/32) through
the repo helpers (marlin_permute_scales + fp8_fused_exponent_bias_into_scales).

FP8 -> BF16 DECODE AND THE EXPONENT-BIAS FOLD (why this is bit-exact)
--------------------------------------------------------------------
Triton has no native sm80 fp8 cast, so the kernel rebuilds the BF16 bit pattern:
    bits = ((raw << 8) & 0x8000) | ((raw << 4) & 0x07F0)
which is bit-exactly fp8_value * 2**-120 for every finite E4M3 code.  (fp8's
exponent bias 7 and bf16's 127 differ by exactly 120, and the subnormal
encodings of both formats use exponent field 0, so the same statement holds for
subnormals; fp8's smallest subnormal 2**-9 becomes the exactly representable
bf16 subnormal 2**-129.)  That identity is verified at runtime against torch's
own fp8 cast.

The kernel then multiplies the payload by the scale *folded with the reciprocal
bias*, exactly like the production Marlin kernel (dequant.h BIAS_OFFSET plus
fp8_fused_exponent_bias_into_scales):
    wk = bf16( (v * 2**-120) * (scale * 2**120) )
The fold is a pure power-of-two rescale of each bf16 factor, so it changes
neither rounding: bf16(scale * 2**120) is exact unless it overflows to inf
(checked below), and the product is formed at the true magnitude, giving
bit-identical results to bf16(v) * bf16(scale).  It is done this way for two
concrete sm80 reasons:
  * the per-element arithmetic stays on the bf16 unit (one bf16 multiply).  A
    literal per-element "* 2**120" makes every payload below 2**-6 a bf16
    subnormal, and Triton compiles fp32 arithmetic with __CUDA_FTZ=1, so any
    fp32 promotion of that rescale would flush the payload to zero;
  * the folded weight value is a normal bf16 for every representable
    (fp8 x scale) product, so the FP32 accumulation never sees a subnormal
    operand either.
The scale fold is hoisted: it costs one extra bf16 multiply per (column, group),
i.e. 1/32 of the elements, not per weight element.

K-SPLIT REDUCTION ORDER (explicit note, as required for a split-K design)
------------------------------------------------------------------------
SPLIT=1 configs sum all of K inside one program: a single sequential FP32
accumulation over ascending k, reduced across the accumulator tile at the end.
SPLIT>1 configs compute, for s = 0..SPLIT-1, an FP32 prefix sum over the
contiguous chunk [s*K/SPLIT, (s+1)*K/SPLIT) and then add those SPLIT partials in
ascending s with an FP32 add.  That is *not* bit-identical to the SPLIT=1 order
nor to Marlin's order: it is a different FP32 association of the same terms.  It
still satisfies the contract (FP32 accumulator, final BF16 cast), so every check
below is against a float64 oracle with a tolerance and the per-route bitwise
agreement with Marlin is *reported*, never claimed.  SPLIT>1 also costs one
extra tiny kernel launch per call.

BENCHMARK PROTOCOL
------------------
32 distinct cold weight rotations per shape (total working set well above the
40MiB L2), two warmup passes, one CUDA graph per rotation captured on a
dedicated non-default stream (capturing on the default stream raises "CUDA
graphs must be captured on a non-default stream"), timing order alternated
between rounds, median reported, plus effective GB/s (raw FP8 bytes + BF16 scale
bytes; the same numerator for every variant) and projected ms saved per 40
calls.  Graphs are replayed after mutating x, the scales and the raw/packed
weights to prove the captured launch re-reads memory instead of replaying a
constant.

WHAT THIS FILE DOES NOT CLAIM
-----------------------------
Only five M=1 shapes, only an isolated dense projection, only cold-weight
streaming.  Every number must come from a run on the actual A100: nothing here
is a measured result by itself.
"""
from __future__ import annotations

import argparse
import json
import os
import statistics

import torch
import triton
import triton.language as tl

from sgl_kernel.scalar_type import scalar_types
from sglang.kernels.ops.gemm.gptq_marlin import gptq_marlin_gemm
from sglang.kernels.ops.quantization.gptq_marlin_repack import gptq_marlin_repack
from sglang.srt.layers.quantization.marlin_utils import (
    marlin_make_workspace,
    marlin_permute_scales,
)
from sglang.srt.layers.quantization.marlin_utils_fp8 import (
    fp8_fused_exponent_bias_into_scales,
    pack_fp8_to_int32,
)

GROUP = 32
EXPERIMENTAL_FLAG = "SGLANG_DSV41_SM80_EXPERIMENTAL"
TILE_FLAG = "SGLANG_DSV41_SM80_DENSE_TILE"
SHAPES = ((1792, 5120), (4096, 1280), (5120, 1024), (576, 5120), (5120, 320))
# Shapes eligible for the experimental dense tile in gptq_marlin.cuh.
TILE_ELIGIBLE = ((1792, 5120), (4096, 1280), (5120, 1024))
L2_BYTES = 40 * 2**20
A100_HBM_PEAK_GB_S = 2039.0
ORACLE_ROUTES = (0, 1, 2, 3)
MUTATIONS = ("x_neg", "scale_half", "weight_sign")
REDUCE_BLOCK = 1024
# 0x80808080 as int32: flips the sign bit of all four packed FP8 bytes at once,
# turning every repacked Marlin weight into its exact negation.
PACKED_SIGN_MASK = -2139062144

# (BLOCK_N, UNROLL groups of 32 per step, K SPLIT, num_warps, num_stages),
# named <n block>_u<k unroll>_s<split>_w<warps>.  num_stages is exposed because
# the task asks for it, but this kernel has no tl.dot: Triton's software
# pipeliner only stages loads through shared memory for dot loops, so the knob
# is expected to be inert here and cross-iteration memory parallelism comes from
# UNROLL instead.
GEMV_CONFIGS = (
    ("n32_u1_s1_w1", (32, 1, 1, 1, 1)),
    ("n32_u2_s5_w1", (32, 2, 5, 1, 2)),
    ("n64_u2_s4_w2", (64, 2, 4, 2, 2)),
    ("n32_u4_s8_w1", (32, 4, 8, 1, 2)),
    ("n64_u4_s8_w2", (64, 4, 8, 2, 2)),
    ("n32_u4_s16_w1", (32, 4, 16, 1, 3)),
    ("n16_u8_s8_w1", (16, 8, 8, 1, 3)),
    ("n128_u4_s1_w4", (128, 4, 1, 4, 2)),
)

FAILURES: list[str] = []


def check(condition: bool, message: str) -> bool:
    if not condition:
        FAILURES.append(message)
        print(json.dumps({"kind": "FAIL", "message": message}), flush=True)
    return condition


# ---------------------------------------------------------------------------
# Kernels
# ---------------------------------------------------------------------------


@triton.jit
def fp8_m1_gemv_kernel(
    w_ptr,  # uint8 (N, K) raw E4M3 bytes, row major
    s_ptr,  # bf16  (N, K // 32) plain model scales (NOT the 2**120 folded ones)
    x_ptr,  # bf16  (K,) activation vector
    out_ptr,  # bf16 (N,) result, written when SPLIT == 1
    part_ptr,  # fp32 (SPLIT, N) partials, written when SPLIT > 1
    N,
    K,
    KS,  # K // 32
    SPLIT: tl.constexpr,
    BLOCK_N: tl.constexpr,
    GROUP: tl.constexpr,
    UNROLL: tl.constexpr,
    EVEN_N: tl.constexpr,
    EVEN_K: tl.constexpr,
):
    """One program owns BLOCK_N consecutive output columns of one K chunk and
    streams it with GROUP(32)-wide loads that vectorize to >=16 bytes.

    The activation is read once per K element per program (each chunk is loaded
    once and broadcast over the program's BLOCK_N columns); the scale is read
    once per (column, k group), never per weight element.  Weights are streamed
    with evict_first because nothing in the block is reused.
    """
    pid_n = tl.program_id(0)
    pid_s = tl.program_id(1)
    n_offs = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    n_mask = n_offs < N
    k_begin = (K // SPLIT) * pid_s
    k_end = tl.minimum(k_begin + (K // SPLIT), K)
    # Exact bf16 constant; a tensor (not a python float) so the multiply below
    # cannot be promoted to fp32.
    two120 = tl.full([1], 2.0**120, dtype=tl.bfloat16)

    acc = tl.zeros((BLOCK_N, GROUP), dtype=tl.float32)
    for k0 in range(k_begin, k_end, GROUP * UNROLL):
        for u in tl.static_range(UNROLL):
            kg = k0 + u * GROUP
            k_offs = kg + tl.arange(0, GROUP)
            w_off = n_offs[:, None] * K + k_offs[None, :]
            # kg may pass k_end inside the unrolled tail, so clamp the group
            # index: the clamped scale is multiplied by a zeroed weight.
            s_off = n_offs * KS + tl.minimum(kg // GROUP, KS - 1)
            if EVEN_N and EVEN_K:
                xk = tl.load(x_ptr + k_offs)
                raw = tl.load(w_ptr + w_off, eviction_policy="evict_first")
                sc = tl.load(s_ptr + s_off)
            elif EVEN_N:
                km = k_offs < k_end
                xk = tl.load(x_ptr + k_offs, mask=km, other=0.0)
                raw = tl.load(
                    w_ptr + w_off,
                    mask=km[None, :],
                    other=0,
                    eviction_policy="evict_first",
                )
                sc = tl.load(s_ptr + s_off)
            elif EVEN_K:
                nm = n_mask[:, None]
                xk = tl.load(x_ptr + k_offs)
                raw = tl.load(
                    w_ptr + w_off, mask=nm, other=0, eviction_policy="evict_first"
                )
                sc = tl.load(s_ptr + s_off, mask=n_mask, other=0.0)
            else:
                km = k_offs < k_end
                m2 = n_mask[:, None] & km[None, :]
                xk = tl.load(x_ptr + k_offs, mask=km, other=0.0)
                raw = tl.load(
                    w_ptr + w_off, mask=m2, other=0, eviction_policy="evict_first"
                )
                sc = tl.load(s_ptr + s_off, mask=n_mask, other=0.0)
            # Bit-exact E4M3 -> BF16 payload (fp8_value * 2**-120), then the
            # contract BF16 multiply with the bias-folded scale.
            b = raw.to(tl.int32)
            bits = ((b << 8) & 0x8000) | ((b << 4) & 0x07F0)
            wv = bits.to(tl.int16).to(tl.bfloat16, bitcast=True)
            wk = (wv * (sc * two120)[:, None]).to(tl.bfloat16)
            acc += wk.to(tl.float32) * xk.to(tl.float32)[None, :]
    row = tl.sum(acc, axis=1)
    if SPLIT == 1:
        tl.store(out_ptr + n_offs, row.to(tl.bfloat16), mask=n_mask)
    else:
        tl.store(part_ptr + pid_s * N + n_offs, row, mask=n_mask)


@triton.jit
def fp8_m1_gemv_reduce_kernel(
    part_ptr,  # fp32 (SPLIT, N)
    out_ptr,  # bf16 (N,)
    N,
    SPLIT: tl.constexpr,
    BLOCK: tl.constexpr,
):
    """Deterministic FP32 combine of the K-split partials in ascending s."""
    pid = tl.program_id(0)
    n_offs = pid * BLOCK + tl.arange(0, BLOCK)
    m = n_offs < N
    acc = tl.zeros((BLOCK,), dtype=tl.float32)
    for s in tl.static_range(SPLIT):
        acc += tl.load(part_ptr + s * N + n_offs, mask=m, other=0.0)
    tl.store(out_ptr + n_offs, acc.to(tl.bfloat16), mask=m)


# ---------------------------------------------------------------------------
# Host helpers: exact decode mirror, float64 oracle, fixtures, launcher
# ---------------------------------------------------------------------------


def fp8_payload_to_bf16(raw: torch.Tensor) -> torch.Tensor:
    """Torch mirror of the kernel decode: fp8_value * 2**-120, bf16."""
    b = raw.view(torch.uint8).to(torch.int32)
    bits = ((b << 8) & 0x8000) | ((b << 4) & 0x07F0)
    return bits.to(torch.int16).view(torch.bfloat16)


def oracle_ref(w8: torch.Tensor, sn: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """float64 CPU oracle with the explicit BF16 intermediate rounding.

    w8: (N, K) uint8 raw bytes; sn: (N, K // 32) bf16 scales; x: (K,) bf16.
    Returns (1, N) bf16.
    """
    n, k = w8.shape
    ks = sn.shape[1]
    assert k == ks * GROUP, (k, ks)
    fp8_value = fp8_payload_to_bf16(w8).double() * float(2.0**120)
    scale = sn.double()[:, :, None].expand(n, ks, GROUP).reshape(n, k)
    # bf16 x bf16 needs at most 16 mantissa bits, so the float64 product is
    # exact and this single .to(bfloat16) is the one contract rounding.
    w_eff = (fp8_value * scale).to(torch.bfloat16).double()
    return (w_eff @ x.double()).to(torch.bfloat16).unsqueeze(0)


def metrics(got: torch.Tensor, ref: torch.Tensor) -> dict:
    diff = got.float() - ref.float()
    return {
        "max_abs": diff.abs().max().item(),
        "relative_rms": (
            diff.square().mean().sqrt()
            / ref.float().square().mean().sqrt().clamp_min(1e-30)
        ).item(),
        "bitwise_equal": bool(torch.equal(got.view(torch.int16), ref.view(torch.int16))),
    }


def close_ok(got: torch.Tensor, ref: torch.Tensor, rtol: float = 1.5e-2, atol: float = 4e-3) -> bool:
    return bool(torch.allclose(got.float(), ref.float(), rtol=rtol, atol=atol))


def make_weight(n: int, k: int, route: int, device) -> torch.Tensor:
    """Route 1 covers the full finite E4M3 codebook; the others are realistic
    Gaussian-ish, exact-cancellation and wider-range signed FP8 patterns."""
    if route == 1:
        codes = torch.arange(256, dtype=torch.uint8)
        codes = codes[(codes & 0x7F) != 0x7F]  # 0x7F / 0xFF are NaN, drop them
        idx = torch.arange(n * k, device=device, dtype=torch.int64) % codes.numel()
        flat = codes.to(device)[idx][torch.randperm(n * k, device=device)]
        return flat.view(n, k).view(torch.float8_e4m3fn).contiguous()
    w = (torch.randn(n, k, device=device) * 0.25).to(torch.float8_e4m3fn)
    if route == 2:
        signs = torch.where(torch.arange(k, device=device) % 2 == 0, 1.0, -1.0)
        w = (w.float().abs() * signs).to(torch.float8_e4m3fn)
    elif route == 3:
        w = (torch.randn(n, k, device=device) * 2.0).to(torch.float8_e4m3fn)
    return w.contiguous()


def make_scales(n: int, k: int, route: int, device) -> torch.Tensor:
    """Checkpoint-style block scales (N // 32, K // 32) mixing exact powers of
    two with arbitrary finite BF16 values."""
    shape = (n // 32, k // GROUP)
    if route % 4 == 0:
        sf = torch.exp2(torch.randint(-6, 1, shape, device=device).float()).to(torch.bfloat16)
    elif route % 4 == 1:
        sf = (0.01 + torch.rand(shape, device=device) * 1.99).to(torch.bfloat16)
    elif route % 4 == 2:
        e = torch.randint(-8, 4, shape, device=device).float()
        sf = (torch.exp2(e) * (1.0 + torch.rand(shape, device=device))).to(torch.bfloat16)
    else:
        sf = (torch.rand(shape, device=device) * 0.02).to(torch.bfloat16)
    return sf


def make_x(k: int, route: int, device) -> torch.Tensor:
    x = (torch.randn(k, device=device) * 0.05).to(torch.bfloat16)
    if route == 1:
        x.mul_(0.01)
    elif route == 2:
        x.fill_(0.125)
    return x


def config_skip_reason(cfg, n: int, k: int) -> str | None:
    block_n, unroll, split, _, _ = cfg
    if block_n > n:
        return f"BLOCK_N {block_n} exceeds N {n}"
    if split > 1 and k % split != 0:
        return f"K {k} is not divisible by SPLIT {split}"
    if k % (GROUP * unroll) != 0 and (k // split) < GROUP * unroll:
        return f"K chunk {k // split} is smaller than the unrolled step {GROUP * unroll}"
    return None


def lane_waste_percent(cfg, k: int) -> float:
    """Share of issued K lanes discarded by the tail mask (0.0 on the fast path).

    Every config issues whole GROUP * UNROLL steps, so a chunk that is not a
    multiple of the step pays for the difference in masked lanes.
    """
    _, unroll, split, _, _ = cfg
    step = GROUP * unroll
    chunk = k // split if k % split == 0 else k
    issued = max(-(-chunk // step), 1) * step
    return 100.0 * (issued - chunk) / issued


def gemv_forward(cfg, w8, sn, x, out, part) -> None:
    """Launch the M1 GEMV (+ K-split combine) for one route.

    Every buffer is preallocated, so the whole call is CUDA-graph capturable.
    """
    block_n, unroll, split, warps, stages = cfg
    n = out.shape[1]
    k = x.numel()
    step = GROUP * unroll
    grid = (triton.cdiv(n, block_n), split)
    fp8_m1_gemv_kernel[grid](
        w8,
        sn,
        x,
        out,
        part,
        n,
        k,
        k // GROUP,
        SPLIT=split,
        BLOCK_N=block_n,
        GROUP=GROUP,
        UNROLL=unroll,
        # EVEN_* is the mask-free fast path: it is only sound when no masked
        # branch is ever needed, i.e. the K chunk of every split is step-aligned.
        EVEN_N=(n % block_n) == 0,
        EVEN_K=(k % step == 0) and (split == 1 or (k // split) % step == 0),
        num_warps=warps,
        num_stages=stages,
    )
    if split > 1:
        fp8_m1_gemv_reduce_kernel[(triton.cdiv(n, REDUCE_BLOCK),)](
            part, out, n, SPLIT=split, BLOCK=REDUCE_BLOCK, num_warps=4
        )


def capture_variant(call, out, routes: int):
    """Warm on a non-default stream, snapshot eager results, capture one graph
    per rotation, then verify that replay reproduces the eager snapshot."""
    for _ in range(2):
        for i in range(routes):
            call(i)
    torch.cuda.synchronize()
    snap = [o.cpu().clone() for o in out]

    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(2):
            for i in range(routes):
                call(i)
    stream.synchronize()

    graphs = []
    for i in range(routes):
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            call(i)
        graphs.append(graph)
    torch.cuda.current_stream().wait_stream(stream)

    for graph in graphs:
        graph.replay()
    torch.cuda.synchronize()
    replay_ok = all(
        torch.equal(a.view(torch.int16), b.view(torch.int16))
        for a, b in zip(snap, [o.cpu() for o in out])
    )
    return graphs, snap, replay_ok, stream


# ---------------------------------------------------------------------------
# Per-shape benchmark
# ---------------------------------------------------------------------------


def run_shape(n: int, k: int, args) -> None:
    device = torch.device("cuda")
    routes = args.routes
    ks = k // GROUP
    per_call_bytes = n * k + n * ks * 2
    mutations = () if args.no_mutations else MUTATIONS
    check(n % 32 == 0 and k % GROUP == 0, f"{n}x{k}: need N % 32 == 0 and K % 32 == 0")
    check(
        routes * per_call_bytes > L2_BYTES,
        f"{n}x{k}: {routes} rotations of {per_call_bytes} B do not exceed the 40MiB L2",
    )
    print(
        json.dumps(
            {
                "kind": "shape",
                "MNK": [1, n, k],
                "routes": routes,
                "rounds": args.rounds,
                "per_call_bytes": per_call_bytes,
                "rotation_MiB": routes * per_call_bytes / 2**20,
                "oracle_routes": list(ORACLE_ROUTES),
            }
        ),
        flush=True,
    )

    workspace = marlin_make_workspace(device)
    perm = torch.empty(0, dtype=torch.int32, device=device)
    dummy = torch.zeros(1, dtype=torch.float32, device=device)

    w8, sn, xs, packed, mscales = [], [], [], [], []
    decode_mismatch, fold_overflow, scale_absmax = 0, 0, 0.0
    codebook_codes = 0
    for route in range(routes):
        w = make_weight(n, k, route, device)
        sf = make_scales(n, k, route, device)
        x = make_x(k, route, device)
        if route == 1:
            # Fixture property, not an assumption: all 254 finite E4M3 codes.
            codebook_codes = int(torch.unique(w.view(torch.uint8)).numel())
            check(
                codebook_codes == 254,
                f"{n}x{k}: codebook fixture covers {codebook_codes} codes, expected 254",
            )
        # Independent check that the kernel's bit trick equals torch's own fp8
        # cast.  float64 here only to keep the check itself free of any FTZ.
        payload = fp8_payload_to_bf16(w).double() * float(2.0**120)
        if not torch.equal(payload, w.float().double()):
            decode_mismatch += 1
        del payload
        scale_absmax = max(scale_absmax, sf.abs().max().item())
        sn.append(sf.repeat_interleave(GROUP, 0).contiguous())
        xs.append(x)
        w8.append(w.view(torch.uint8))
        packed.append(
            gptq_marlin_repack(
                b_q_weight=pack_fp8_to_int32(w, False).T.contiguous(),
                perm=perm,
                size_k=k,
                size_n=n,
                num_bits=8,
            )
        )
        # Exact Marlin scale path: expand the checkpoint block scale to
        # (K/32, N), permute it with the weight permutation, then fold 2**120.
        expanded = sf.T.contiguous().repeat_interleave(GROUP, 1)
        fused = fp8_fused_exponent_bias_into_scales(
            marlin_permute_scales(s=expanded, size_k=k, size_n=n, group_size=GROUP)
        )
        if not bool(torch.isfinite(fused).all()):
            fold_overflow += 1
        mscales.append(fused)
    check(decode_mismatch == 0, f"{n}x{k}: fp8 decode trick mismatch on {decode_mismatch} routes")
    # The fold is exact only while scale * 2**120 stays finite in bf16.
    check(
        fold_overflow == 0 and scale_absmax * 2.0**120 < 3.39e38,
        f"{n}x{k}: scale fold overflow (max |scale| {scale_absmax})",
    )
    print(
        json.dumps(
            {
                "kind": "fixtures",
                "MNK": [1, n, k],
                "routes": routes,
                "decode_mismatch": decode_mismatch,
                "fold_overflow": fold_overflow,
                "scale_absmax": scale_absmax,
                "codebook_codes": codebook_codes,
            }
        ),
        flush=True,
    )

    marlin_out = [
        torch.empty((1, n), dtype=torch.bfloat16, device=device) for _ in range(routes)
    ]

    def marlin_call(i):
        return gptq_marlin_gemm(
            a=xs[i].unsqueeze(0),
            c=marlin_out[i],
            b_q_weight=packed[i],
            b_scales=mscales[i],
            global_scale=None,
            b_zeros=None,
            g_idx=None,
            perm=None,
            workspace=workspace,
            b_q_type=scalar_types.float8_e4m3fn,
            size_m=1,
            size_n=n,
            size_k=k,
            is_k_full=True,
            use_atomic_add=False,
            use_fp32_reduce=True,
            is_zp_float=False,
        )

    # Oracle references do not depend on the variant, so build them once.
    oracle_refs = {
        i: oracle_ref(w8[i].cpu(), sn[i].cpu(), xs[i].cpu()) for i in ORACLE_ROUTES
    }

    # Mutation references: raw weights/scales are mutated in place and the
    # oracle recomputed on the mutated data.  For the packed Marlin inputs the
    # equivalent in-place change is used (sign bit of every packed byte, or a
    # halved scale), so one reference serves both paths.
    saved = (xs[0].clone(), sn[0].clone(), w8[0].clone(), packed[0].clone(), mscales[0].clone())

    def mutate(which: str, revert: bool = False) -> None:
        if which == "x_neg":
            if revert:
                xs[0].copy_(saved[0])
            else:
                xs[0].neg_()
        elif which == "scale_half":
            if revert:
                sn[0].copy_(saved[1])
                mscales[0].copy_(saved[4])
            else:
                sn[0].mul_(0.5)
                mscales[0].mul_(0.5)
        else:
            if revert:
                w8[0].copy_(saved[2])
                packed[0].copy_(saved[3])
            else:
                w8[0].bitwise_xor_(0x80)
                packed[0].bitwise_xor_(PACKED_SIGN_MASK)

    mut_refs = {}
    for which in mutations:
        mutate(which)
        mut_refs[which] = oracle_ref(w8[0].cpu(), sn[0].cpu(), xs[0].cpu())
        mutate(which, revert=True)

    specs = [("marlin_default", "marlin", "default", None)]
    specs.append(("marlin_n64", "marlin", "n64", None))
    specs += [(f"gemv:{name}", "gemv", None, cfg) for name, cfg in GEMV_CONFIGS]

    results = {}
    for name, kind, tile, cfg in specs:
        if args.variants and name not in args.variants:
            continue
        if kind == "marlin":
            if tile == "n64" and (n, k) not in TILE_ELIGIBLE:
                print(
                    json.dumps(
                        {
                            "kind": "skip",
                            "MNK": [1, n, k],
                            "variant": name,
                            "reason": "not eligible for the experimental dense tile",
                        }
                    ),
                    flush=True,
                )
                continue
            os.environ[TILE_FLAG] = tile
            out, call = marlin_out, marlin_call
        else:
            os.environ.pop(TILE_FLAG, None)
            reason = config_skip_reason(cfg, n, k)
            if reason is not None:
                print(
                    json.dumps(
                        {
                            "kind": "skip",
                            "MNK": [1, n, k],
                            "variant": name,
                            "reason": reason,
                        }
                    ),
                    flush=True,
                )
                continue
            block_n, unroll, split, warps, stages = cfg
            part = (
                torch.empty((split, n), dtype=torch.float32, device=device)
                if split > 1
                else dummy
            )
            out = [
                torch.empty((1, n), dtype=torch.bfloat16, device=device)
                for _ in range(routes)
            ]

            def call(i, _cfg=cfg, _out=out, _part=part):
                gemv_forward(_cfg, w8[i], sn[i], xs[i], _out[i], _part)

            print(
                json.dumps(
                    {
                        "kind": "config",
                        "MNK": [1, n, k],
                        "variant": name,
                        "BLOCK_N": block_n,
                        "UNROLL": unroll,
                        "SPLIT": split,
                        "num_warps": warps,
                        "num_stages": stages,
                        "grid": [triton.cdiv(n, block_n), split],
                        "K_chunk": k // split,
                        "masked_lane_percent": lane_waste_percent(cfg, k),
                        "mask_free": (n % block_n == 0)
                        and (k % (GROUP * unroll) == 0)
                        and (split == 1 or (k // split) % (GROUP * unroll) == 0),
                    }
                ),
                flush=True,
            )

        graphs, snap, replay_ok, stream = capture_variant(call, out, routes)
        results[name] = {
            "kind": kind,
            "out": out,
            "graphs": graphs,
            "snap": snap,
            "stream": stream,
        }
        print(
            json.dumps(
                {
                    "kind": "replay",
                    "MNK": [1, n, k],
                    "variant": name,
                    "graph_replay_bitwise_identical": replay_ok,
                }
            ),
            flush=True,
        )

        for i in ORACLE_ROUTES:
            got, ref = snap[i], oracle_refs[i]
            stat = metrics(got, ref)
            check(close_ok(got, ref), f"{n}x{k} {name} fixture {i}: oracle mismatch {stat}")
            print(
                json.dumps(
                    {"kind": "oracle", "MNK": [1, n, k], "variant": name, "fixture": i, **stat}
                ),
                flush=True,
            )

        base_snap = results.get("marlin_default", {}).get("snap")
        if base_snap is not None and name != "marlin_default":
            cross = [metrics(snap[i], base_snap[i]) for i in range(routes)]
            check(
                all(close_ok(snap[i], base_snap[i]) for i in range(routes)),
                f"{n}x{k} {name}: cross-check against marlin_default failed",
            )
            print(
                json.dumps(
                    {
                        "kind": "cross",
                        "MNK": [1, n, k],
                        "variant": name,
                        "routes": routes,
                        "max_abs": max(c["max_abs"] for c in cross),
                        "max_relative_rms": max(c["relative_rms"] for c in cross),
                        "bitwise_routes": sum(c["bitwise_equal"] for c in cross),
                    }
                ),
                flush=True,
            )

        for which in mutations:
            mutate(which)
            graphs[0].replay()
            torch.cuda.synchronize()
            got, ref = out[0].cpu(), mut_refs[which]
            stat = metrics(got, ref)
            check(close_ok(got, ref), f"{n}x{k} {name} mutation {which}: {stat}")
            print(
                json.dumps(
                    {
                        "kind": "mutation",
                        "MNK": [1, n, k],
                        "variant": name,
                        "mutation": which,
                        "route": 0,
                        **stat,
                    }
                ),
                flush=True,
            )
            mutate(which, revert=True)

    names = list(results)
    if not names:
        check(False, f"{n}x{k}: no variant was runnable")
        return
    if "marlin_default" not in names:
        print(
            json.dumps(
                {
                    "kind": "warn",
                    "MNK": [1, n, k],
                    "message": "marlin_default not selected; speedups use the first variant",
                }
            ),
            flush=True,
        )

    for name in names:
        for _ in range(3):
            for graph in results[name]["graphs"]:
                graph.replay()
    torch.cuda.synchronize()

    samples = {name: [] for name in names}
    for round_id in range(args.rounds):
        order = names if round_id % 2 == 0 else list(reversed(names))
        for name in order:
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            for graph in results[name]["graphs"]:
                graph.replay()
            end.record()
            end.synchronize()
            samples[name].append(start.elapsed_time(end) * 1000 / routes)

    base_name = "marlin_default" if "marlin_default" in names else names[0]
    base = statistics.median(samples[base_name])
    for name in names:
        median = statistics.median(samples[name])
        gbs = per_call_bytes / (median * 1e-6) / 1e9
        print(
            json.dumps(
                {
                    "kind": "benchmark",
                    "MNK": [1, n, k],
                    "variant": name,
                    "kernel": results[name]["kind"],
                    "baseline": base_name,
                    "median_us": median,
                    "speedup_vs_baseline": base / median,
                    "effective_GB/s": gbs,
                    "percent_of_2039_GB/s": 100 * gbs / A100_HBM_PEAK_GB_S,
                    "projected_40_calls_saved_ms": (base - median) * 40 / 1000,
                    "samples_us": samples[name],
                }
            ),
            flush=True,
        )

    os.environ.pop(TILE_FLAG, None)
    results.clear()
    del results, w8, sn, xs, packed, mscales, marlin_out, graphs, snap
    torch.cuda.empty_cache()


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--routes", type=int, default=32, help="cold weight rotations")
    parser.add_argument("--rounds", type=int, default=20, help="timing rounds")
    parser.add_argument(
        "--shapes",
        default=",".join(f"{n}x{k}" for n, k in SHAPES),
        help="comma separated NxK list (N is the output/weight-row dimension)",
    )
    parser.add_argument(
        "--variants", default="", help="comma separated subset of variant names"
    )
    parser.add_argument(
        "--no-mutations", action="store_true", help="skip the in-place mutation checks"
    )
    args = parser.parse_args()
    args.variants = tuple(v for v in args.variants.split(",") if v)
    shapes = tuple(tuple(int(v) for v in s.split("x")) for s in args.shapes.split(","))

    assert torch.cuda.is_available(), "needs a CUDA GPU"
    assert torch.version.hip is None, "SM80 CUDA only"
    assert torch.cuda.get_device_capability() == (8, 0), torch.cuda.get_device_capability()
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    os.environ[EXPERIMENTAL_FLAG] = "1"
    torch.manual_seed(41202)
    check(args.routes >= 32, "at least 32 cold rotations are required")
    check(args.rounds >= 2, "at least 2 timing rounds are required")

    props = torch.cuda.get_device_properties(0)
    print(
        json.dumps(
            {
                "kind": "device",
                "name": props.name,
                "sms": props.multi_processor_count,
                "L2_MiB": getattr(props, "L2_cache_size", 0) / 2**20,
                "torch": torch.__version__,
                "triton": triton.__version__,
                "routes": args.routes,
                "rounds": args.rounds,
                "configs": {
                    name: dict(
                        zip(
                            ("BLOCK_N", "UNROLL", "SPLIT", "num_warps", "num_stages"),
                            cfg,
                        )
                    )
                    for name, cfg in GEMV_CONFIGS
                },
            }
        ),
        flush=True,
    )

    for n, k in shapes:
        run_shape(n, k, args)

    os.environ.pop(EXPERIMENTAL_FLAG, None)
    os.environ.pop(TILE_FLAG, None)
    print(json.dumps({"kind": "summary", "failures": FAILURES}), flush=True)
    print("SM80_M1_GEMV_" + ("PASS" if not FAILURES else "FAIL"), flush=True)
    raise SystemExit(0 if not FAILURES else 1)


if __name__ == "__main__":
    main()
