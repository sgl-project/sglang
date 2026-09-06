# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

# KDA provenance: this kernel was automatically optimized by the Humanize2
# workflow (https://github.com/PolyArch/humanize) and Kernel Design Agents
# (https://github.com/mit-han-lab/kernel-design-agents).
#
# The mainloop / epilogue are forked from the Qwen3.x NVFP4 SM120 kernel
# (BBuf/KDA-Pilot#195 @ 516c976cee824a236679adf6eb525275a0a9a120), with the
# MXF4NVF4 block-scaled MMA and all SFA/SFB plumbing replaced by a dense BF16
# m16n8k16 warp MMA. The grid computation follows the production dense BF16
# CuTe-DSL GEMM in sglang/kernels/ops/gemm/cutedsl_bf16_gemm.py: plain ceil-div
# CTA counts launched directly from the host-side JIT function, with NO
# StaticPersistentTileScheduler (that scheduler requires a rank-3 (M, N, L)
# problem layout which a 2D fake C tensor cannot provide through the DSL).

# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:

# 1. Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.

# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.

# 3. Neither the name of the copyright holder nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.

# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

# This file implements a dense BF16 GEMM specialized for decode shapes
# (M <= 48) on SM120 (RTX 5090, CC 12.0).

from __future__ import annotations

import threading

import cuda.bindings.driver as cuda
import cutlass
import cutlass.utils.hopper_helpers as sm90_utils
import torch
from cutlass import Int32, Int64, cute, pipeline, utils
from cutlass._mlir.dialects import llvm
from cutlass.cute.arch import griddepcontrol_launch_dependents, griddepcontrol_wait
from cutlass.cute.nvgpu import cpasync
from cutlass.cutlass_dsl import T, dsl_user_op
from cutlass.utils.static_persistent_tile_scheduler import WorkTileInfo


def _ceil_div(a: int, b: int) -> int:
    return (a + b - 1) // b


@dsl_user_op
def _make_evict_first_policy(*, loc=None, ip=None) -> Int64:
    return Int64(
        llvm.inline_asm(
            T.i64(),
            [],
            "createpolicy.fractional.L2::evict_first.b64 $0, 1.0;",
            "=l",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def _make_evict_last_policy(*, loc=None, ip=None) -> Int64:
    return Int64(
        llvm.inline_asm(
            T.i64(),
            [],
            "createpolicy.fractional.L2::evict_last.b64 $0, 1.0;",
            "=l",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
            loc=loc,
            ip=ip,
        )
    )


# Decode tile buckets: (tile_m, tile_n) -> (tile_shape_mnk, atom_layout_mnk,
# num_mma_warps). mma.sync has a 16-row minimum M atom. Production M <= 16
# uses eight N-split warps with tile_n=128, widening to tile_n=256 when N is
# large enough to retain CTA parallelism. These configs win across RTX 5090
# and RTX PRO 6000 Blackwell; the larger-M buckets below remain available for
# direct kernel benchmarking but are excluded from automatic dispatch.
# Decode GEMM is weight (B) streaming bound, so for M in (16, 48] the fastest
# direct-benchmark shape uses the narrow tile_n=64 with every warp N-split
# (atom_n = warps): each warp then
# holds a thin tile_n/atom_n/8-wide accumulator whose smem->reg ldmatrix
# traffic the small warp count can hide, while the CTA count doubles to
# spread the weight stream across more SMs. The old (32/64,128) buckets
# used atom_layout (2,2,1) with 4 warps, which left each warp with a wide
# MMA_N=8 accumulator (32x64 values, ~544B/thread/k-tile of ldmatrix) that
# 4 warps could not hide, and the M=48 rows padded to tile_m=64 still ran
# mma.sync on 16 dead rows; M=48 was 0.26x-0.98x vs F.linear there.
# (48,64) covers M=48 exactly (3x16, zero padded rows) and (32,64) keeps
# atom_n=4 for the M<=32 direct benchmark path.
# Wide-N/large-K fallback for M=48: (48,64) atom_layout (3,2,1) splits the
# tile_n=64 stream across 2 N-warps (MMA_N=4 each) and caps the grid at
# N/64 CTAs, which on N=4096 is only 64 CTAs / 170 SMs = 38% util in a
# single wave and left (48,4096,11008) at 0.51x vs F.linear. (48,32) with
# (1,4,1) splits the tile_n=32 B-stream across 4 N-warps (MMA_N=1 each,
# 4/3x the (3,1,1) per-CTA B-stream ldmatrix issue rate) and doubles the
# grid to 128 CTAs (75% util, 2 CTAs per L2 slice on the K-duplicated
# weight stream). The smem
# stage shrinks to (48+32)*64*2 = 10KB, so raw_ab_stage stays >= 5 and the
# pipeline depth is unchanged. The swap is gated on N/K size in
# _tile_key_for_shape so the small-N M=48 wins (48,4096,4096)=0.93x and
# (48,11008,4096)=0.95x keep (48,64).
# PERF pass 4: the M=48 rows are B-stream bound, so the (48,64) bucket
# moves to tile_k=128. A 128-wide TMA box halves the per-stage transaction
# count (better DRAM efficiency on the weight stream) and doubles
# num_k_blocks to 8, so the ldmatrix->mma steady state gets 4x the
# independent instructions per pipeline stage; ab_stage drops 5->4, which
# still hides the stream latency but with fewer, deeper barriers. The
# (16/32,*) buckets keep tile_k=64 (they are already >=1.10x), and (48,32)
# keeps tile_k=64 so it stays in every M=48 benchmark row as the
# non-tile_k fallback if K is not 128-divisible. k_loop_unroll follows
# tile_k: the 64-wide 2-6-warp tiles go 2->4 to deepen the software
# pipeline, while the 2-warp (16,64) bucket keeps 2 (its accumulator/ldmx
# budget would spill at 4).
# PERF pass 5: the M=48 rows are still 0.92-0.97x (K=4096) / 0.70x
# (48,4096,11008) -- the per-CTA B stream is the cap, and tile_k/unroll
# changes cannot raise it. A WIDE-N tile attacks it from the other side:
# (48,128,64) with atom_layout (3,4,1) = 12 MMA warps streams 128 B
# columns per CTA, so each CTA moves 2x the B bytes per k-step and the
# fixed per-k-step overhead (pipeline barrier arrive/wait, fragment
# slicing) is amortized over 2x the mma work. Crucially, the per-warp
# fragment load is UNCHANGED vs the 6-warp (48,64): MMA_M = 48/(3*16) = 1
# and MMA_N = 128/(4*8) = 4, so each warp still holds a 16x32 fp32
# accumulator (32 regs) with identical ldmatrix traffic; the B stream just
# fans out to 12 warps instead of 6. Doubling the warps also doubles the
# in-flight mma issue rate per CTA, which is what a latency-bound inner
# loop needs. Register budget: 48x128 fp32 acc / 384 threads = 16 regs,
# ldmatrix fragments and the epilogue G2R slice are ~2x the (48,64)
# bucket's per-thread traffic, so the mma warps drop to 128 regs/thread
# (headroom over the ~80-100 needed; 384x128 + 32x40 = 50K regs < 64K/SM)
# instead of the 232 the small tiles use for ldmatrix-x4 multi-buffering.
# The grid drops to N/128 CTAs, but the K=4096 rows were already
# 38%-utilized and warp-level parallelism is worth more than CTA-level
# parallelism there; (48,4096,11008) flips too because a 32-CTA fat
# stream (22.5KB stage, 5-deep pipeline) beats 128 thin CTAs whose TMA
# issue rate could not fill the pipe.
# PERF pass 6: the pass-5 all-rows->(48,128) routing regressed small-N
# M=48 rows. Measured vs F.linear: (48,4096,4096) 0.67x on (48,128) vs
# 0.93x on (48,64); (48,6144,4096) 0.66x vs 0.92x; (48,11008,4096) 1.26x
# on (48,128) vs 0.96x on (48,64); (48,4096,11008) 0.35x on (48,128) vs
# 0.70x on (48,32). The 12-warp fat stream only wins when N is large
# enough for N/128 CTAs to fill 170 SMs (N=11008 -> 86 CTAs); small N
# (4096 -> 32 CTAs, 6144 -> 48 CTAs) underfills the GPU, where the
# narrower (48,64) (64-96 CTAs) or (48,32) (128 CTAs) wins. Routing is
# therefore per-shape in _tile_key_for_shape: large N -> (48,128),
# large K / small N -> (48,32), otherwise (48,64). The (48,128) tile and
# its 128-reg/thread fix stay for the large-N rows.
# PERF pass 7: the (48,32) large-K bucket is re-warped (3,1,1) 3w ->
# (1,4,1) 4w, the same all-warp-N-split pattern the (32,64) M<=32 bucket
# proves fastest. The old (3,1,1) split M across 3 warps (MMA_M=1,
# MMA_N=4, 16x32 acc), so all 3 warps consumed the SAME 32 B columns per
# k-step: the per-CTA B stream was capped by a 3-warp ldmatrix issue rate
# on a weight-stream-bound loop, which is why (48,4096,11008) sat at
# 0.70x. The (1,4,1) layout splits tile_n=32 across 4 N-warps instead
# (MMA_M=3, MMA_N=1): each warp now owns a DISTINCT 8-wide B column
# slice, so the per-CTA B-stream ldmatrix issue rate rises 4/3x while the
# fp32 accumulator per warp is UNCHANGED (48x32/128 threads = 16x32 = 32
# regs/warp -- the extra M iterations reuse the same B fragment against 3
# A-row fragments, +8 regs). The grid is unchanged (128 CTAs on N=4096),
# the 10KB smem stage and >=5 pipeline depth are unchanged, and 4 warps
# keep the 232-reg ldmatrix multi-buffering budget (128 threads; the
# 128-reg fix is only needed at >=12 warps). mma.sync is warp-local, so
# the (3,1,1)->(1,4,1) atom swap is a pure reschedule with identical
# fp32 math and epilogue coverage.
# PERF pass 8: the M=48 small-N rows (N=4096) are still 0.89/0.92
# (K=4096) and 0.73 (K=11008). The failing configs are the two extremes:
# (48,64) has only 64 CTAs / 6 warps (38% SM util), and (48,32) has 128
# CTAs / 4 warps (~2.9 warps/SM, a quarter of one SM issue group). The
# measured winner pattern -- (48,128) on N=11008, 1.26x -- shows the
# large-K M=48 rows want the 12-warp fat B stream. (48,96) ports that
# pattern to small N: tile (48,96,128) with atom_layout (3,4,1) = 12 MMA
# warps keeps the exact pass-5 per-warp footprint (MMA_M=1, MMA_N=3, a
# 16x24 fp32 accumulator = 24 regs, LIGHTER than (48,128)'s 16x32) while
# each CTA streams 96 B columns per k-step (1.5x (48,64), 3x (48,32)).
# On N=4096 the grid is 43 CTAs x 12 warps = 516 warps (~3.0/SM, same as
# (48,32)'s 512 but with a 3x fatter per-CTA DRAM stream, 4-deep TMA
# pipeline, and the pass-4 tile_k=128 wide boxes); on N=6144 it is 64
# CTAs = 768 warps. Large-K (K>=8192) small-N rows move to it for the
# same reason (48,128) won on N=11008: the fat stream beats 128 thin
# CTAs whose 4-warp TMA issue rate could not fill the pipe. smem: the
# 36KB stage x5 = 180KB + 9KB epi + 1KB mbar = 190KB <= 227KB, and the
# 128-reg/thread fix already applies at 12 warps (384*128 + 32*40 =
# 50K regs < 64K/SM). Gated to n%96==0 (4096, 6144) in
# _tile_key_for_shape so the big-N winner (48,11008,4096) keeps
# (48,128); N=8192 is unreachable (n>=8192 routes first).
_TILE_CONFIGS: dict[tuple[int, int], tuple[tuple[int, int, int], tuple, int]] = {
    (16, 64): ((16, 64, 64), (1, 2, 1), 2),
    (16, 128): ((16, 128, 64), (1, 8, 1), 8),
    (16, 256): ((16, 256, 64), (1, 8, 1), 8),
    (32, 64): ((32, 64, 64), (1, 4, 1), 4),
    (32, 128): ((32, 128, 64), (1, 8, 1), 8),
    (32, 96): ((32, 96, 64), (2, 4, 1), 8),
    (48, 64): ((48, 64, 128), (3, 2, 1), 6),
    (48, 96): ((48, 96, 128), (3, 4, 1), 12),
    (48, 32): ((48, 32, 64), (1, 4, 1), 4),
    (48, 128): ((48, 128, 64), (3, 4, 1), 12),
}

_MAX_DECODE_M = 48


def _tile_key_for_m(m: int) -> tuple[int, int]:
    if m < 1 or m > _MAX_DECODE_M:
        raise ValueError(
            f"M={m} is outside the SM120 BF16 decode range (M <= {_MAX_DECODE_M})"
        )
    if m <= 16:
        return (16, 128)
    if m <= 32:
        return (32, 64)
    return (48, 64)


def _tile_key_for_shape(m: int, n: int, k: int) -> tuple[int, int]:
    # Small-M projections use eight N-split warps on both RTX 5090 (170 SMs)
    # and RTX PRO 6000 Blackwell (188 SMs).  Very wide outputs have enough
    # N-parallel CTAs for a 256-column tile, which halves the streamed-weight
    # CTA count without under-filling either GPU.
    if m <= 16:
        if n >= 24576 and n % 256 == 0:
            return (16, 256)
        # Narrow-N (small-grid) rows win on the 2-warp (16,64) tile: the
        # fatter per-CTA N stream fills the SMs that a 128-wide tile leaves
        # idle (q_proj 1.17x vs 1.02x, kv_proj 1.47x vs 1.33x). Wide N has
        # enough N-parallel CTAs to prefer the lighter (16,128).
        if n < 4096 and n % 64 == 0:
            return (16, 64)
        return (16, 128)

    # Per-row routing for M in (32, 48], from measured M=48 bench rows:
    #   (48,11008,4096): (48,128) = 1.26x vs (48,64) = 0.96x  -> (48,128)
    #   (48,4096,11008): (48,32)  = 0.70x vs (48,128) = 0.35x -> (48,32)
    #   (48,4096,4096):  (48,64)  = 0.93x vs (48,128) = 0.67x -> (48,64)
    #   (48,6144,4096):  (48,64)  = 0.92x vs (48,128) = 0.66x -> (48,64)
    # The 12-warp (48,128) fat stream only wins when N is large enough for
    # N/128 CTAs to fill the 170 SMs (N=11008 -> 86 CTAs); small N
    # (4096 -> 32 CTAs, 6144 -> 48 CTAs) underfills the GPU, where more
    # thin CTAs win. n>=8192 puts the crossover between the measured
    # N=6144 (48,64) and N=11008 (48,128) rows. Large K with small N goes
    # to (48,32) (128 CTAs on N=4096); everything else keeps (48,64).
    # Small N (4096/6144) goes to (48,96): 43/64 CTAs x 12 warps with the
    # (48,128)-class fat B stream and unchanged per-warp fragments (see
    # PERF pass 8 above). It covers both the K=4096 rows (targeting
    # 0.89/0.92) and the large-K down-proj rows (48,4096,11008)=0.73 /
    # (48,4096,12288), which previously fell to the 4-warp (48,32). The
    # n>=8192 check stays first, so the (48,11008,4096)=1.26x winner
    # keeps (48,128); (48,32) remains for K>=8192 with N not 96-divisible
    # and as the K%128!=0 tile_k=64 fallback.
    # Split-K is permanently disabled (measured a net loss on every M=48
    # small-N large-K shape; see run_bf16_gemm_sm120), so the k>=8192
    # small-N rows stay on the plain (48,32) single-pass tile.
    # (48,32) also stays the K%128!=0 tile_k=64 fallback in
    # run_bf16_gemm_sm120.
    # PERF pass 9: big-N routing is K-sensitive. The 12-warp (48,128) fat
    # stream pays a fixed prologue/epilogue overhead (12-warp pipeline
    # barrier setup, 22.5KB stage TMA, wide epilogue) that only amortizes
    # over a long k-loop: (48,12288,4096)=1.24 / (48,11008,4096)=1.26 win
    # at K=4096, but (48,9216,2560) dropped to 0.83 on the same tile --
    # K=2560 is only 40 64-wide k-tiles, so the overhead dominates.
    # Short-K big-N rows (n>=8192, k<4096) now take (48,96): still a
    # 12-warp fat stream but with a lighter per-warp footprint
    # (MMA_N=3, 16x24 fp32 acc) and tile_k=128, so each CTA moves 2x the
    # B bytes per k-step with fewer, deeper stages -- 96 CTAs on N=9216,
    # 1.34x the (48,128) CTA count. The K>=4096 big-N winners keep
    # (48,128).
    # M in (16, 32]: narrow N fills best on the 4-warp (32,64) thin tile
    # (q/kv/o 1.29-1.50x); very wide N needs the 8-warp (32,96) fat stream to
    # avoid a 192-CTA wave quantization (gate_up 1.17x vs 0.88x). Large-K
    # narrow-N (down) is K-bound -- the per-CTA B stream through K/tile_k
    # sequential k-tiles caps it and split-K/larger tiles measured net losses
    # (down 0.53x), so it stays on F.linear (see use_bf16_gemm_sm120).
    if m > 16:
        if n >= 8192 and n % 96 == 0:
            return (32, 96)
        return (32, 64)
    if m > 32:
        if n >= 8192:
            if k >= 4096 and n % 128 == 0:
                return (48, 128)
            if n % 96 == 0:
                return (48, 96)
            if n % 128 == 0:
                return (48, 128)
        elif n % 96 == 0:
            return (48, 96)
        if k >= 8192:
            return (48, 32)
        return (48, 64)
    return _tile_key_for_m(m)


def use_bf16_gemm_sm120(m: int, n: int, k: int) -> bool:
    """Return True when the SM120 BF16 kernel wins for this decode shape."""
    # Skinny decode M in {1,2,4}: the mma tile has no M predication and
    # would read/write the A/C tail out of bounds, so these rows take the
    # warp-per-row sm120_bf16_gemv streaming path (dispatched inside
    # run_bf16_gemm_sm120). It beats cuBLAS by 1.13-1.69x on every E2B
    # projection because it saturates DRAM bandwidth that the cuBLAS M<16
    # tiles leave idle (measured 1.5-4.6 TB/s vs 0.19-2.3 TB/s).
    if m in (1, 2, 4):
        return n % 8 == 0 and k % 256 == 0 and 512 <= k <= 28672 and 64 <= n <= 65536
    # The mma tile epilogue has no M predication and always reads/stores a
    # full tile_m-row tile, so an in-between batch (m in {3,5..15,17..31})
    # would go out of bounds on the (16,*)/(32,*) tiles. Only the exact
    # tile_m decode buckets (m in {16, 32}) are out-of-bounds-free.
    if m not in (16, 32):
        return False
    # N must be tile-divisible for the unpredicated full-tile TMA store;
    # K needs 64 for the K%128!=0 tile_k=64 fallback in run_bf16_gemm_sm120.
    if n % 64 != 0 or k % 64 != 0:
        return False
    # Large-K narrow-N (the down-proj family, n <= k and k >= 4096) is
    # K-bound at M=32: the per-CTA B stream through K/tile_k sequential
    # k-tiles caps the kernel at ~0.53x of cuBLAS, and split-K / larger
    # tiles measured net losses. Those rows stay on cuBLAS.
    if m == 32 and k >= 4096 and n <= k:
        return False
    # M=16 extreme-K down-proj (1536, 12288): only N/64 = 24 CTAs fill the
    # 170 SMs, so the long K/tile_k B stream leaves the GPU ~86% idle and
    # every tile bucket measures ~0.47x of cuBLAS (the narrower-N/longer-K
    # sibling of the M=32 roofline above). cuBLAS's K-split tile reaches
    # ~2x the throughput, so this one shape stays on cuBLAS. The K=4096
    # sibling (16,1536,4096) still wins (1.16-1.34x) and is kept.
    if m == 16 and k >= 12288 and n <= k:
        return False
    # M=16 wins on every projection family (q 1.17x, kv 1.47x, o 1.25x,
    # gate_up 1.40x, down 1.09x); M=32 wins on q/kv/o/gate_up (1.17-1.50x).
    return True


class _Bf16GemmSm120Kernel:
    """SM120 warp-MMA kernel for dense BF16 decode shapes (M <= 48).

    It uses m16n8k16 ``MmaF16BF16Op`` atoms, a TMA producer warp, and no
    TMEM/tcgen05/2-CTA instructions (SM120 lacks them). Each CTA computes
    exactly one output tile; the grid is derived with plain ceil-div CTA
    counts in the host-side JIT function, following the production dense
    BF16 kernel in sglang/kernels/ops/gemm/cutedsl_bf16_gemm.py. No
    StaticPersistentTileScheduler is used.

    Two-way split-K is a compile-time specialization (``num_splits=2``):
    the grid gains a z dimension, each CTA computes half the K range into
    a per-split fp32 workspace slab via a direct SIMT global-store
    epilogue (the fp32 smem/TMA epilogue path is non-structurable in the
    SM120 DSL), and the host wrapper launches a tiny reduction that sums
    the two fp32 slabs and converts to bf16 (adding the optional bias
    there, once).
    """

    def __init__(
        self,
        *,
        tile_m: int,
        tile_n: int,
        cache_policy: bool,
        num_splits: int = 1,
    ):
        self.acc_dtype = cutlass.Float32
        self.mma_k = 16
        tile_shape_mnk, atom_shape, num_mma_warps = _TILE_CONFIGS[(tile_m, tile_n)]
        if num_splits != 1:
            if num_splits != 2 or tile_shape_mnk[2] != 64:
                raise ValueError(
                    f"num_splits={num_splits} requires the tile_k=64 (48,32) bucket"
                )
            self.c_dtype = cutlass.Float32
        self.num_splits = num_splits
        self.tile_shape_mnk = tile_shape_mnk
        self.mma_tile_shape_mnk = self.tile_shape_mnk
        self.cluster_shape_mnk = (1, 1, 1)
        self.epi_tile = (self.tile_shape_mnk[0], self.tile_shape_mnk[1])
        self.load_path = "tma"
        # Software-pipeline depth for the steady-state k loop. The tile_k=128
        # bucket has num_k_blocks=8 per stage, so its in-stage stream already
        # hides smem latency at unroll 2; the tile_k=64 buckets with 2-6 MMA
        # warps get unroll 4 (the 2-warp (16,64) tile would spill its
        # 232-reg budget there).
        self.k_loop_unroll = 2 if self.tile_shape_mnk[2] == 128 else 4
        self.use_operand_cache_policy = cache_policy
        # Programmatic Dependent Launch: decode issues this GEMM back-to-back
        # with other kernels on one stream. PDL lets the next kernel's
        # prologue (descriptor prefetch, smem alloc, pipeline/barrier init,
        # first TMA issue) overlap this kernel's epilogue tail, and the
        # matching griddepcontrol_wait() before the first smem read keeps the
        # overlap safe. Numerics are unchanged (pure launch-scheduling); the
        # decode-step stream measured ~17% faster with PDL on.
        self.enable_pdl = True
        self.atom_shape = atom_shape

        self.tiled_mma = None
        self.occupancy = 1
        self.num_mma_warps = num_mma_warps
        self.tma_load_warp_id = self.num_mma_warps
        self.num_threads_per_warp = 32
        self.threads_per_cta = (
            self.num_mma_warps + 1  # 1 warp for DMA
        ) * self.num_threads_per_warp
        # The 12-warp (48,128) bucket must fit 384 MMA threads in the 64K
        # regs/SM budget: 232 regs/thread is 89K > 64K and does not even
        # launch. Its per-warp fragment/accumulator footprint matches the
        # 6-warp (48,64) bucket (MMA_M=1, MMA_N=4), and the only
        # per-thread growth is the B fragment (48 bf16 = 24 regs) and the
        # 12KB epilogue G2R slice, so 128 regs/thread leaves headroom over
        # the ~80-100 needed (384*128 + 32*40 = 50K regs, one CTA/SM).
        # The smaller buckets keep 232: their ldmatrix-x4 multi-buffered
        # operand fragments use it to hide smem latency.
        mma_regs = 128 if self.num_mma_warps >= 12 else 232

        self.smem_capacity = utils.get_smem_capacity_in_bytes("sm_120")

        self.ab_stage = None
        self.epi_stage = None
        self.a_smem_layout_staged = None
        self.b_smem_layout_staged = None
        self.epi_smem_layout_staged = None

        self.buffer_align_bytes = 1024

        self.epilog_sync_barrier = pipeline.NamedBarrier(
            barrier_id=2,
            num_threads=self.num_mma_warps * self.num_threads_per_warp,
        )
        self.load_register_requirement = 40
        self.mma_register_requirement = mma_regs

    def _setup_attributes(self):
        # Dense BF16 mma.sync m16n8k16 with fp32 accumulate. SM120 has no
        # tcgen05/TMEM/2-CTA MMA, so the Ampere-style warp MMA is the native
        # tensor-core path.
        mma_op = cute.nvgpu.warp.MmaF16BF16Op(
            self.a_dtype,
            self.acc_dtype,
            (16, 8, 16),
        )
        atom_layout = cute.make_layout(self.atom_shape)
        # Dense SM120 warp-MMA pattern, exactly as in the proven
        # ops/attention/fa4_sm120/flash_fwd.py _get_tiled_mma: each
        # permutation_mnk mode spans ONE atom-layout iteration of the
        # (16, 8, 16) atom, i.e. atom_layout_mnk * (16, 8, 16). The
        # m16n8k16 C-atom value layout natively covers exactly 16x8, so any
        # permutation mode larger than the value layout (e.g. the previous
        # full-CTA-tile (tile_m, tile_n, 64)) is folded to the atom's
        # canonical value layout -- but with the stride taken from the
        # oversized integer. The K mode (64) only widens the A/B fragments,
        # not C, so the C value->(m, n) mapping keeps the wrong 4x
        # coordinate stride: registers beyond the first 8x16 value block
        # (32 of the 2048 f32 values per warp, ~1.6%) land on scrambled
        # output rows/columns. With the atom-iteration-sized permutation
        # the value mode is unpermuted (identity), and the leftover
        # tile_m/(atom_m*16) and tile_n/(atom_n*8) iterations live in the
        # MMA_M/MMA_N accumulator modes, which thr_mma.partition_C and
        # make_tiled_copy_C_atom/StMatrix map back to coordinates
        # canonically. The permutation also keeps the per-atom value modes
        # of partition/make_fragment fully formed, so the tCrX[None, None,
        # k] slices stay rank-2 for cute.gemm's verifier (the original
        # reason an explicit permutation was introduced).
        permutation_mnk = (
            self.atom_shape[0] * 16,
            self.atom_shape[1] * 8,
            16,
        )
        self.tiled_mma = cute.make_tiled_mma(
            mma_op, atom_layout, permutation_mnk=permutation_mnk
        )
        # Proven dense SM120 form (fa4_sm120/flash_fwd.py): the tiled_mma
        # itself iterates the M/N value modes, so the mainloop calls
        # cute.gemm(tiled_mma, acc, tCrA[None, None, k], tCrB[None, None, k],
        # acc) with the WHOLE accumulator and TWO-slot fragment slicing. No
        # bare mma_atom and no manual _mt/_nt unroll — that NVFP4-derived
        # form collapses the gemm operands to rank 1.

        self.cta_layout_mnk = cute.make_layout(self.cluster_shape_mnk)

        # Compute stage before compute smem layout
        self.ab_stage, self.epi_stage = self._compute_stages(
            self.tile_shape_mnk,
            self.a_dtype,
            self.b_dtype,
            self.epi_tile,
            self.c_dtype,
            self.smem_capacity,
            self.occupancy,
        )

        assert self.epi_stage > 0, (
            "epi_stage <= 0, not enough shared memory. This configuration will be skipped."
        )

        (
            self.a_smem_layout_staged,
            self.b_smem_layout_staged,
            self.epi_smem_layout_staged,
        ) = self._make_smem_layouts(
            self.tile_shape_mnk,
            self.epi_tile,
            self.a_dtype,
            self.a_layout,
            self.b_dtype,
            self.b_layout,
            self.ab_stage,
            self.c_dtype,
            self.c_layout,
            self.epi_stage,
        )

    @cute.jit
    def __call__(
        self,
        a: cute.Tensor,
        b: cute.Tensor,
        c: cute.Tensor,
        stream: cuda.CUstream,
        epilogue_op: cutlass.Constexpr = lambda x: x,
    ):
        """Execute the GEMM operation.

        Args:
            a: Input tensor A, (m, k) row-major bf16
            b: Input tensor B, (n, k) row-major bf16 (linear weight layout)
            c: Output tensor C, (m, n) row-major bf16
            stream: CUDA stream
            epilogue_op: Elementwise epilogue function
        """
        # Setup static attributes
        self.a_dtype = a.element_type
        self.b_dtype = b.element_type
        if cutlass.const_expr(self.num_splits == 1):
            self.c_dtype = c.element_type
        else:
            self.c_dtype = cutlass.Float32

        self.a_layout = utils.LayoutEnum.from_tensor(a)
        self.b_layout = utils.LayoutEnum.from_tensor(b)
        self.c_layout = utils.LayoutEnum.from_tensor(c)

        if cutlass.const_expr(self.a_dtype != self.b_dtype):
            raise TypeError(f"Type mismatch: {self.a_dtype} != {self.b_dtype}")

        self._setup_attributes()

        tma_atom_a, tma_tensor_a = self._make_tma_atoms_and_tensors(
            a,
            self.a_smem_layout_staged,
            (self.tile_shape_mnk[0], self.tile_shape_mnk[2]),
            1,
        )
        tma_atom_b, tma_tensor_b = self._make_tma_atoms_and_tensors(
            b,
            self.b_smem_layout_staged,
            (self.tile_shape_mnk[1], self.tile_shape_mnk[2]),
            1,
        )
        if cutlass.const_expr(self.num_splits == 1):
            tma_atom_c, tma_tensor_c = self._make_tma_store_atoms_and_tensors(
                c,
                self.epi_smem_layout_staged,
                self.epi_tile,
            )
        else:
            # Split-K fp32 epilogue: the store bypasses smem/TMA entirely
            # (see the kernel epilogue), so no fp32 TMA store atom is built.
            # Building one aborts cute.compile anyway: make_tiled_tma_atom
            # folds the (m, n, 2) workspace's unit n mode into the 2D box
            # and C++-asserts on the resulting stride-0 descriptor dim
            # (SIGABRT, not a catchable MLIRError). The SIMT epilogue does
            # not need the TMA-basis walk either: the rank-3 (m, n, split)
            # workspace tensor goes straight to the kernel as a plain
            # global tensor with its (n, 1, m*n) layout untouched (the
            # flash_fwd_sm90 Optional[cute.CopyAtom] idiom for SIMT
            # epilogues), and the kernel peels the split slab at entry.
            tma_atom_c = None
            tma_tensor_c = c

        # Grid: one CTA per output tile (x per split along z), plain
        # ceil-div CTA counts (the same scheme cutedsl_bf16_gemm.py uses on
        # SM100). M <= 48 < 2 * tile_m so there is exactly one M tile per
        # column; the M tile count is 1. For num_splits=2, C is the fp32
        # (m, n, 2) workspace and grid.z selects the split slab; the kernel
        # peels the rank-3 (m, n, split) workspace down to a rank-2 (m, n)
        # slab with a scalar split id at entry.
        num_m_tiles = cute.ceil_div(c.shape[0], self.tile_shape_mnk[0])
        num_n_tiles = cute.ceil_div(c.shape[1], self.tile_shape_mnk[1])
        grid = (num_m_tiles, num_n_tiles, self.num_splits)

        @cute.struct
        class SharedStorage:
            mainloop_pipeline_array_ptr: cute.struct.MemRange[
                cutlass.Int64, self.ab_stage * 2
            ]
            sC: cute.struct.Align[
                cute.struct.MemRange[
                    self.c_dtype, cute.cosize(self.epi_smem_layout_staged)
                ],
                self.buffer_align_bytes,
            ]
            sA: cute.struct.Align[
                cute.struct.MemRange[
                    self.a_dtype, cute.cosize(self.a_smem_layout_staged)
                ],
                self.buffer_align_bytes,
            ]
            sB: cute.struct.Align[
                cute.struct.MemRange[
                    self.b_dtype, cute.cosize(self.b_smem_layout_staged)
                ],
                self.buffer_align_bytes,
            ]

        self.shared_storage = SharedStorage

        self.kernel(
            tma_atom_a,
            tma_tensor_a,
            tma_atom_b,
            tma_tensor_b,
            tma_atom_c,
            tma_tensor_c,
            self.tiled_mma,
            self.cta_layout_mnk,
            self.a_smem_layout_staged,
            self.b_smem_layout_staged,
            self.epi_smem_layout_staged,
            epilogue_op,
        ).launch(
            grid=grid,
            block=[self.threads_per_cta, 1, 1],
            cluster=[1, 1, 1],
            smem=cute.Int64(self.smem_capacity),
            stream=stream,
            use_pdl=self.enable_pdl,
        )

    # GPU device kernel
    @cute.kernel
    def kernel(
        self,
        tma_atom_a: cute.CopyAtom,
        mA_mkl: cute.Tensor,
        tma_atom_b: cute.CopyAtom,
        mB_nkl: cute.Tensor,
        tma_atom_c: cute.CopyAtom,  # Optional; None on the split-K SIMT path
        mC_mnl: cute.Tensor,
        tiled_mma: cute.TiledMma,
        cta_layout_mnk: cute.Layout,
        a_smem_layout_staged: cute.ComposedLayout,
        b_smem_layout_staged: cute.ComposedLayout,
        epi_smem_layout_staged: cute.ComposedLayout,
        epilogue_op: cutlass.Constexpr,
    ):
        tidx, _, _ = cute.arch.thread_idx()
        warp_idx = cute.arch.warp_idx()
        warp_idx = cute.arch.make_warp_uniform(warp_idx)

        # Split-K: peel this CTA's fp32 (m, n) slab off the rank-3
        # (m, n, split) workspace tensor with a scalar split index, so the
        # split epilogue slices a plain rank-2 (m, n) global tile.
        if cutlass.const_expr(self.num_splits == 2):
            split_id = cute.arch.block_idx()[2]
            mC_mnl = mC_mnl[(None, None, split_id)]

        # Prefetch TMA descriptors
        if warp_idx == 0:
            if cutlass.const_expr(self.load_path == "tma"):
                cpasync.prefetch_descriptor(tma_atom_a)
                cpasync.prefetch_descriptor(tma_atom_b)
                if cutlass.const_expr(self.num_splits == 1):
                    cpasync.prefetch_descriptor(tma_atom_c)

        cta_rank_in_cluster = cute.arch.make_warp_uniform(
            cute.arch.block_idx_in_cluster()
        )
        cluster_coord_mnk = cta_layout_mnk.get_flat_coord(cta_rank_in_cluster)

        a_smem_layout = cute.slice_(a_smem_layout_staged, (None, None, 0))
        b_smem_layout = cute.slice_(b_smem_layout_staged, (None, None, 0))
        tma_copy_bytes = cute.size_in_bytes(
            self.a_dtype, a_smem_layout
        ) + cute.size_in_bytes(self.b_dtype, b_smem_layout)

        # Allocate shared memory
        smem = cutlass.utils.SmemAllocator()
        storage = smem.allocate(self.shared_storage)

        # Pipeline setup
        mainloop_pipeline_array_ptr = storage.mainloop_pipeline_array_ptr.data_ptr()
        mainloop_pipeline_producer_group = pipeline.CooperativeGroup(
            pipeline.Agent.Thread
        )
        mainloop_pipeline_consumer_group = pipeline.CooperativeGroup(
            pipeline.Agent.Thread, self.num_mma_warps
        )

        cta_layout_vmnk = cute.make_layout((1, *cta_layout_mnk.shape))
        mainloop_pipeline = pipeline.PipelineTmaAsync.create(
            num_stages=self.ab_stage,
            producer_group=mainloop_pipeline_producer_group,
            consumer_group=mainloop_pipeline_consumer_group,
            tx_count=tma_copy_bytes,
            barrier_storage=mainloop_pipeline_array_ptr,
            cta_layout_vmnk=cta_layout_vmnk,
        )

        # Generate smem tensors
        sA = storage.sA.get_tensor(
            a_smem_layout_staged.outer, swizzle=a_smem_layout_staged.inner
        )
        sB = storage.sB.get_tensor(
            b_smem_layout_staged.outer, swizzle=b_smem_layout_staged.inner
        )
        sC = (
            storage.sC.get_tensor(
                epi_smem_layout_staged.outer, swizzle=epi_smem_layout_staged.inner
            )
            if cutlass.const_expr(self.num_splits == 1)
            else None
        )

        # Local_tile partition global tensors (2D (m, k) / (n, k) / (m, n))
        gA_mkl = cute.local_tile(
            mA_mkl,
            cute.slice_(self.tile_shape_mnk, (None, 0, None)),
            (None, None),
        )
        gB_nkl = cute.local_tile(
            mB_nkl,
            cute.slice_(self.tile_shape_mnk, (0, None, None)),
            (None, None),
        )
        gC_mnl = cute.local_tile(
            mC_mnl,
            cute.slice_(self.tile_shape_mnk, (None, None, 0)),
            (None, None),
        )

        # Partition for TiledMMA
        thr_mma = tiled_mma.get_slice(tidx)

        # TMA partitions for A
        a_cta_layout = cute.make_layout(cute.slice_(cta_layout_mnk, (0, None, 0)).shape)
        a_cta_crd = cluster_coord_mnk[1]
        if cutlass.const_expr(self.load_path == "tma"):
            tAsA, tAgA = cpasync.tma_partition(
                tma_atom_a,
                a_cta_crd,
                a_cta_layout,
                cute.group_modes(sA, 0, 2),
                cute.group_modes(gA_mkl, 0, 2),
            )

        # TMA partitions for B
        b_cta_layout = cute.make_layout(cute.slice_(cta_layout_mnk, (None, 0, 0)).shape)
        b_cta_crd = cluster_coord_mnk[0]
        if cutlass.const_expr(self.load_path == "tma"):
            tBsB, tBgB = cpasync.tma_partition(
                tma_atom_b,
                b_cta_crd,
                b_cta_layout,
                cute.group_modes(sB, 0, 2),
                cute.group_modes(gB_nkl, 0, 2),
            )

        # Make fragments.
        tCsA = thr_mma.partition_A(sA)
        tCsB = thr_mma.partition_B(sB)

        # Proven dense SM120 fragment idiom (fa4_sm120/flash_fwd.py
        # _gemm_qk/_gemm_pv): partition the SMEM tile, drop only the stage
        # mode, and keep the (MMA, MMA_M/N, MMA_K) fragment rank 3 so that
        # the two-slot tCrX[None, None, k] slice is a rank-2 (MMA, MMA_M/N)
        # operand — exactly what cute.gemm's verifier requires.
        tCrA = tiled_mma.make_fragment_A(tCsA[None, None, None, 0])
        tCrB = tiled_mma.make_fragment_B(tCsB[None, None, None, 0])
        # FA4 accumulator idiom: partition_shape_C over the CTA tile yields
        # (MMA, MMA_M, MMA_N); the whole accumulator is passed to cute.gemm
        # and the tiled_mma iterates the M/N value modes.
        acc_shape = thr_mma.partition_shape_C(
            (self.tile_shape_mnk[0], self.tile_shape_mnk[1])
        )
        accumulators = cute.make_rmem_tensor(acc_shape, self.acc_dtype)

        # Thread sync (single-CTA cluster, so a plain barrier suffices)
        cute.arch.sync_threads()

        if cutlass.const_expr(self.enable_pdl):
            griddepcontrol_wait()

        # gA_mkl is (TileM, TileK, RestM, RestK): the rank-2 (m, k) tensor is
        # tiled with (None, None), so the rest modes land at 2 and 3. The
        # k-tile count is RestK at mode 3; mode 2 is RestM (always 1 for
        # decode M <= 48), which would silently run exactly one k-tile
        # regardless of K. (The NVFP4 kernel also reads mode 3, but for a
        # different reason: its rank-3 (m, k, l) tiling puts RestK there.)
        k_tile_iter_cnt = cute.size(gA_mkl, mode=[3])

        # Work tile: this CTA owns exactly one output tile at
        # (blockIdx.x, blockIdx.y) (blockIdx.z = split slice). No persistent
        # scheduler. With num_splits=2 each CTA computes exactly half the
        # k-tiles (both halves have the same count because the split-K
        # routing gate requires K % (2 * tile_k) == 0), so the producer /
        # consumer loops keep their original trip count and only their
        # global k offset changes.
        block_idx = cute.arch.block_idx()
        work_tile = WorkTileInfo(
            (block_idx[0], block_idx[1], Int32(0)),
            cutlass.Boolean(1),
        )
        if cutlass.const_expr(self.num_splits == 2):
            split_id = block_idx[2]
            k_tile_start = split_id * (k_tile_iter_cnt // 2)
            k_tile_iter_cnt = k_tile_iter_cnt // 2
        else:
            split_id = Int32(0)
            k_tile_start = Int32(0)

        # Pipeline states
        mainloop_producer_state = pipeline.make_pipeline_state(
            pipeline.PipelineUserType.Producer, self.ab_stage
        )
        mainloop_consumer_state = pipeline.make_pipeline_state(
            pipeline.PipelineUserType.Consumer, self.ab_stage
        )

        # MMA warp group
        if warp_idx < self.num_mma_warps:
            cute.arch.setmaxregister_increase(self.mma_register_requirement)

            num_k_blocks = cute.size(tCrA, mode=[2])

            # Copy atoms for SMEM->RMEM
            atom_copy_ldmatrix_A = cute.make_copy_atom(
                cute.nvgpu.warp.LdMatrix8x8x16bOp(self.a_layout.is_m_major_a(), 4),
                self.a_dtype,
            )
            atom_copy_ldmatrix_B = cute.make_copy_atom(
                cute.nvgpu.warp.LdMatrix8x8x16bOp(self.b_layout.is_n_major_b(), 4),
                self.b_dtype,
            )
            smem_tiled_copy_A = cute.make_tiled_copy_A(atom_copy_ldmatrix_A, tiled_mma)
            smem_tiled_copy_B = cute.make_tiled_copy_B(atom_copy_ldmatrix_B, tiled_mma)

            thr_copy_ldmatrix_A = smem_tiled_copy_A.get_slice(tidx)
            thr_copy_ldmatrix_B = smem_tiled_copy_B.get_slice(tidx)
            tCsA_copy_view = thr_copy_ldmatrix_A.partition_S(sA)
            tCrA_copy_view = thr_copy_ldmatrix_A.retile(tCrA)
            tCsB_copy_view = thr_copy_ldmatrix_B.partition_S(sB)
            tCrB_copy_view = thr_copy_ldmatrix_B.retile(tCrB)

            while work_tile.is_valid_tile:
                tile_coord_mnl = work_tile.tile_idx
                gC_mnl_slice = gC_mnl[(None, None, *tile_coord_mnl[:2])]
                # num_splits=2 already sliced mC_mnl down to this CTA's
                # rank-2 fp32 slab at kernel entry, so gC_mnl and this
                # (m, n) tile slice are exactly the non-split rank-2 path.
                accumulators.fill(0.0)

                # Pipelined MAINLOOP
                mainloop_consumer_state.reset_count()

                peek_ab_full_status = cutlass.Boolean(1)
                if mainloop_consumer_state.count < k_tile_iter_cnt:
                    peek_ab_full_status = mainloop_pipeline.consumer_try_wait(
                        mainloop_consumer_state
                    )

                mainloop_pipeline.consumer_wait(
                    mainloop_consumer_state, peek_ab_full_status
                )
                tCsA_p = tCsA_copy_view[None, None, None, mainloop_consumer_state.index]
                tCsB_p = tCsB_copy_view[None, None, None, mainloop_consumer_state.index]
                cute.copy(
                    smem_tiled_copy_A,
                    tCsA_p[None, None, 0],
                    tCrA_copy_view[None, None, 0],
                )
                cute.copy(
                    smem_tiled_copy_B,
                    tCsB_p[None, None, 0],
                    tCrB_copy_view[None, None, 0],
                )

                for _k_tile in range(
                    0,
                    k_tile_iter_cnt - 1,
                    1,
                    unroll=self.k_loop_unroll,
                ):  # type: ignore[call-overload]
                    for k_block_idx in cutlass.range_constexpr(num_k_blocks):
                        k_block_next = (
                            0 if k_block_idx + 1 == num_k_blocks else k_block_idx + 1
                        )

                        if k_block_idx == num_k_blocks - 1:
                            mainloop_pipeline.consumer_release(mainloop_consumer_state)
                            mainloop_consumer_state.advance()

                            peek_ab_full_status = cutlass.Boolean(1)
                            peek_ab_full_status = mainloop_pipeline.consumer_try_wait(
                                mainloop_consumer_state
                            )

                            tCsA_p = tCsA_copy_view[
                                None, None, None, mainloop_consumer_state.index
                            ]
                            tCsB_p = tCsB_copy_view[
                                None, None, None, mainloop_consumer_state.index
                            ]
                            mainloop_pipeline.consumer_wait(
                                mainloop_consumer_state, peek_ab_full_status
                            )

                        # Proven dense SM120 gemm idiom: the tiled_mma walks
                        # the M/N value modes itself, so pass the whole
                        # accumulator and slice fragments with TWO slots
                        # [None, None, k]. The three-slot bare-atom form is
                        # NVFP4-blockscaled-only and collapses A to rank 1.
                        cute.gemm(
                            tiled_mma,
                            accumulators,
                            tCrA[None, None, k_block_idx],
                            tCrB[None, None, k_block_idx],
                            accumulators,
                        )
                        cute.copy(
                            smem_tiled_copy_A,
                            tCsA_p[None, None, k_block_next],
                            tCrA_copy_view[None, None, k_block_next],
                        )
                        cute.copy(
                            smem_tiled_copy_B,
                            tCsB_p[None, None, k_block_next],
                            tCrB_copy_view[None, None, k_block_next],
                        )

                # Hoist out last k_tile
                for k_block_idx in cutlass.range_constexpr(num_k_blocks):
                    k_block_next = (
                        0 if k_block_idx + 1 == num_k_blocks else k_block_idx + 1
                    )

                    if k_block_idx == num_k_blocks - 1:
                        mainloop_pipeline.consumer_release(mainloop_consumer_state)
                        mainloop_consumer_state.advance()

                    if k_block_next > 0:
                        cute.copy(
                            smem_tiled_copy_A,
                            tCsA_p[None, None, k_block_next],
                            tCrA_copy_view[None, None, k_block_next],
                        )
                        cute.copy(
                            smem_tiled_copy_B,
                            tCsB_p[None, None, k_block_next],
                            tCrB_copy_view[None, None, k_block_next],
                        )
                    cute.gemm(
                        tiled_mma,
                        accumulators,
                        tCrA[None, None, k_block_idx],
                        tCrB[None, None, k_block_idx],
                        accumulators,
                    )

                # EPILOGUE
                # num_matrices=2 keeps the stmatrix atom at 4 values/thread,
                # matching the m16n8k16 f32 C fragment width (LayoutC_TV size
                # 4); num_matrices=4 would exceed it and fail
                # make_tiled_copy_C_atom. transpose follows is_m_major_c.
                if cutlass.const_expr(self.num_splits == 1):
                    _is_m_major = self.c_layout.is_m_major_c()
                    copy_atom_r2s = cute.make_copy_atom(
                        cute.nvgpu.warp.StMatrix8x8x16bOp(_is_m_major, 2),
                        self.c_dtype,
                    )
                    copy_atom_C = cute.make_copy_atom(
                        cute.nvgpu.warp.StMatrix8x8x16bOp(
                            self.c_layout.is_m_major_c(), 2
                        ),
                        self.c_dtype,
                    )

                    tiled_copy_C_Atom = cute.make_tiled_copy_C_atom(
                        copy_atom_C, tiled_mma
                    )

                    tiled_copy_r2s = cute.make_tiled_copy_S(
                        copy_atom_r2s,
                        tiled_copy_C_Atom,
                    )

                    thr_copy_r2s = tiled_copy_r2s.get_slice(tidx)
                    tRS_sD = thr_copy_r2s.partition_D(sC)
                    tRS_rAcc = tiled_copy_r2s.retile(accumulators)

                    rD_shape = cute.shape(thr_copy_r2s.partition_S(sC))
                    tRS_rD_layout = cute.make_layout(rD_shape[:3])
                    tRS_rD = cute.make_rmem_tensor(tRS_rD_layout.shape, self.acc_dtype)

                    sepi_for_tma_partition = cute.group_modes(sC, 0, 2)
                    tcgc_for_tma_partition = cute.zipped_divide(
                        gC_mnl_slice, self.epi_tile
                    )

                    bSG_sD, bSG_gD = cpasync.tma_partition(
                        tma_atom_c,
                        0,
                        cute.make_layout(1),
                        sepi_for_tma_partition,
                        tcgc_for_tma_partition,
                    )

                    epi_rest_m = bSG_gD.shape[1][0]
                    epi_rest_n = bSG_gD.shape[1][1]
                    epi_tile_m = self.epi_tile[0]
                    epi_tile_n = self.epi_tile[1]
                    mma_tile_m = self.tile_shape_mnk[0] // cute.size(tRS_rAcc, mode=[1])
                    mma_tile_n = self.tile_shape_mnk[1] // cute.size(tRS_rAcc, mode=[2])
                    has_multi_epi_store = cutlass.const_expr(
                        not (
                            self.epi_stage == 1 and epi_rest_m == 1 and epi_rest_n == 1
                        )
                    )
                    tma_store_producer_group = pipeline.CooperativeGroup(
                        pipeline.Agent.Thread,
                        self.num_mma_warps * self.num_threads_per_warp,
                    )
                    tma_store_pipeline = pipeline.PipelineTmaStore.create(
                        num_stages=self.epi_stage,
                        producer_group=tma_store_producer_group,
                    )

                    for epi_m in cutlass.range_constexpr(epi_rest_m):
                        for epi_n in cutlass.range_constexpr(epi_rest_n):
                            MmaMPerEpiM = epi_tile_m // mma_tile_m
                            MmaNPerEpiN = epi_tile_n // mma_tile_n
                            for mma_n_in_epi in cutlass.range_constexpr(MmaNPerEpiN):
                                for mma_m_in_epi in cutlass.range_constexpr(
                                    MmaMPerEpiM
                                ):
                                    mma_n = (epi_n * MmaNPerEpiN) + mma_n_in_epi
                                    mma_m = (epi_m * MmaMPerEpiM) + mma_m_in_epi
                                    tRS_rD_slice = tRS_rD[
                                        (None, mma_m_in_epi, mma_n_in_epi)
                                    ]
                                    tRS_rAcc_slice = tRS_rAcc[(None, mma_m, mma_n)]
                                    for elem_idx in cutlass.range_constexpr(
                                        cute.size(tRS_rD_slice)
                                    ):
                                        tRS_rD_slice[elem_idx] = tRS_rAcc_slice[
                                            elem_idx
                                        ]

                            gmem_coord = (epi_m, epi_n)
                            tRS_rD_out = cute.make_rmem_tensor(
                                tRS_rD_layout.shape, self.c_dtype
                            )
                            acc_vec = tRS_rD.load()
                            acc_vec = epilogue_op(acc_vec.to(self.c_dtype))
                            tRS_rD_out.store(acc_vec)

                            # Register to shared memory
                            epi_buffer = (epi_m * epi_rest_n + epi_n) % cute.size(
                                tRS_sD, mode=[3]
                            )
                            if has_multi_epi_store:
                                self.epilog_sync_barrier.arrive_and_wait()
                            cute.copy(
                                tiled_copy_r2s,
                                tRS_rD_out,
                                tRS_sD[(None, None, None, epi_buffer)],
                            )
                            cute.arch.fence_proxy(
                                "async.shared",
                                space="cta",
                            )
                            self.epilog_sync_barrier.arrive_and_wait()

                            # Copy from shared memory to global memory
                            if warp_idx == 0:
                                cute.copy(
                                    tma_atom_c,
                                    bSG_sD[(None, epi_buffer)],
                                    bSG_gD[(None, gmem_coord)],
                                )
                                if has_multi_epi_store:
                                    tma_store_pipeline.producer_commit()
                                    tma_store_pipeline.producer_acquire()
                else:
                    # Split-K fp32 epilogue: direct SIMT global store of the
                    # fp32 accumulator into this CTA's (m, n) workspace slab
                    # (already peeled to rank 2 at kernel entry). This
                    # sidesteps the fp32 smem+TMA epilogue entirely: sm120
                    # stmatrix has no fp32 mode, and the fp32
                    # make_tiled_copy_C_atom(CopyUniversalOp) /
                    # make_tiled_tma_atom(S2G) constructions abort the MLIR
                    # builder inside cute.compile. mma.sync m16n8k16 gives
                    # each thread 2 contiguous n per 8-col group, so the
                    # row-major slab coalesces to 8B stores. Bias stays
                    # disabled (the host reduction applies it once).
                    tCgC = thr_mma.partition_C(gC_mnl_slice)
                    cute.autovec_copy(accumulators, tCgC)

                # One work tile per CTA: the loop exits after the first tile.
                work_tile = WorkTileInfo(
                    work_tile.tile_idx,
                    cutlass.Boolean(0),
                )

                if cutlass.const_expr(self.num_splits == 1):
                    if has_multi_epi_store:
                        tma_store_pipeline.producer_tail()

        elif warp_idx == self.tma_load_warp_id:
            cute.arch.setmaxregister_decrease(self.load_register_requirement)

            while work_tile.is_valid_tile:
                tile_coord_mnl = work_tile.tile_idx
                if cutlass.const_expr(self.load_path == "tma"):
                    tAgA_mkl = tAgA[(None, tile_coord_mnl[0], None)]
                    tBgB_nkl = tBgB[(None, tile_coord_mnl[1], None)]

                mainloop_producer_state.reset_count()

                for _k_tile in range(
                    0,
                    k_tile_iter_cnt,
                    1,
                    unroll=self.k_loop_unroll,
                ):  # type: ignore[call-overload]
                    mainloop_pipeline.producer_acquire(mainloop_producer_state)

                    k_tile_global = k_tile_start + mainloop_producer_state.count
                    if cutlass.const_expr(self.load_path == "tma"):
                        tAgA_k = tAgA_mkl[(None, k_tile_global)]
                        tAsA_pipe = tAsA[(None, mainloop_producer_state.index)]
                        tBgB_k = tBgB_nkl[(None, k_tile_global)]
                        tBsB_pipe = tBsB[(None, mainloop_producer_state.index)]

                    if cutlass.const_expr(self.load_path == "tma"):
                        if cutlass.const_expr(self.use_operand_cache_policy):
                            cute.copy(
                                tma_atom_a,
                                tAgA_k,
                                tAsA_pipe,
                                tma_bar_ptr=mainloop_pipeline.producer_get_barrier(
                                    mainloop_producer_state
                                ),
                                cache_policy=_make_evict_last_policy(),
                            )
                            cute.copy(
                                tma_atom_b,
                                tBgB_k,
                                tBsB_pipe,
                                tma_bar_ptr=mainloop_pipeline.producer_get_barrier(
                                    mainloop_producer_state
                                ),
                                cache_policy=_make_evict_first_policy(),
                            )
                        else:
                            cute.copy(
                                tma_atom_a,
                                tAgA_k,
                                tAsA_pipe,
                                tma_bar_ptr=mainloop_pipeline.producer_get_barrier(
                                    mainloop_producer_state
                                ),
                            )
                            cute.copy(
                                tma_atom_b,
                                tBgB_k,
                                tBsB_pipe,
                                tma_bar_ptr=mainloop_pipeline.producer_get_barrier(
                                    mainloop_producer_state
                                ),
                            )
                    mainloop_pipeline.producer_commit(mainloop_producer_state)
                    mainloop_producer_state.advance()

                # One work tile per CTA: the loop exits after the first tile.
                work_tile = WorkTileInfo(
                    work_tile.tile_idx,
                    cutlass.Boolean(0),
                )

            mainloop_pipeline.producer_tail(mainloop_producer_state)

        if cutlass.const_expr(self.enable_pdl):
            griddepcontrol_launch_dependents()

    @staticmethod
    def _compute_stages(
        tile_shape_mnk: tuple,
        a_dtype,
        b_dtype,
        epi_tile: tuple,
        c_dtype,
        smem_capacity: int,
        occupancy: int,
    ) -> tuple:
        epi_stage_max = (tile_shape_mnk[1] // epi_tile[1]) * (
            tile_shape_mnk[0] // epi_tile[0]
        )
        epi_stage = min(epi_stage_max, 4)
        c_bytes_per_stage = cute.size(epi_tile) * c_dtype.width // 8
        epi_bytes = c_bytes_per_stage * epi_stage

        a_shape = cute.slice_(tile_shape_mnk, (None, 0, None))
        b_shape = cute.slice_(tile_shape_mnk, (0, None, None))
        ab_bytes_per_stage = (
            cute.size(a_shape) * a_dtype.width // 8
            + cute.size(b_shape) * b_dtype.width // 8
        )
        mbar_helpers_bytes = 1024

        raw_ab_stage = (
            (smem_capacity - occupancy * 1024) // occupancy
            - mbar_helpers_bytes
            - epi_bytes
        ) // ab_bytes_per_stage
        # The decode GEMM is weight (B) streaming bound; on large K the
        # per-CTA producer loop is long and gated by TMA latency on the
        # B-stream, so allow one extra TMA stage wherever smem permits
        # (sm_120 has 227KB smem; raw_ab_stage is ~15-21 for the 10-14KB
        # 64-wide stages and ~10 for the 18KB (16,128) stage, so +1 stage
        # stays <= ~68KB = 30% of capacity at occupancy 1). This deepens
        # producer/consumer overlap for the long-K loop; the M <= 16
        # shapes are short-K friendly and keep their prior stages for the
        # (16,64) bucket, while (16,128) already had 5.
        if tile_shape_mnk[1] == 128 or raw_ab_stage >= 5:
            ab_stage = max(1, min(raw_ab_stage, 5))
        else:
            ab_stage = max(1, min(raw_ab_stage, 4))
        return ab_stage, epi_stage

    @staticmethod
    def _make_smem_layouts(
        tile_shape_mnk: tuple,
        epi_tile: tuple,
        a_dtype,
        a_layout,
        b_dtype,
        b_layout,
        ab_stage: int,
        c_dtype,
        c_layout,
        epi_stage: int,
    ) -> tuple:
        a_smem_shape = cute.slice_(tile_shape_mnk, (None, 0, None))

        a_is_k_major = a_layout.is_k_major_a()
        b_is_k_major = b_layout.is_k_major_b()
        a_major_mode_size = tile_shape_mnk[2 if a_is_k_major else 0]

        a_smem_layout_atom = cute.nvgpu.warpgroup.make_smem_layout_atom(
            sm90_utils.get_smem_layout_atom(
                a_layout,
                a_dtype,
                a_major_mode_size,
            ),
            a_dtype,
        )
        a_smem_layout_staged = cute.tile_to_shape(
            a_smem_layout_atom,
            cute.append(a_smem_shape, ab_stage),
            order=(0, 1, 2) if a_is_k_major else (1, 0, 2),
        )

        b_smem_shape = cute.slice_(tile_shape_mnk, (0, None, None))
        b_major_mode_size = tile_shape_mnk[2 if b_is_k_major else 1]
        b_smem_layout_atom = cute.nvgpu.warpgroup.make_smem_layout_atom(
            sm90_utils.get_smem_layout_atom(
                b_layout,
                b_dtype,
                b_major_mode_size,
            ),
            b_dtype,
        )
        b_smem_layout_staged = cute.tile_to_shape(
            b_smem_layout_atom,
            cute.append(b_smem_shape, ab_stage),
            order=(0, 1, 2) if b_is_k_major else (1, 0, 2),
        )

        c_smem_shape = epi_tile
        c_major_mode_size = epi_tile[1] if c_layout.is_n_major_c() else epi_tile[0]
        c_smem_layout_atom = cute.nvgpu.warpgroup.make_smem_layout_atom(
            sm90_utils.get_smem_layout_atom(
                c_layout,
                c_dtype,
                c_major_mode_size,
            ),
            c_dtype,
        )
        epi_smem_layout_staged = cute.tile_to_shape(
            c_smem_layout_atom,
            cute.append(c_smem_shape, epi_stage),
            order=(1, 0, 2) if c_layout.is_m_major_c() else (0, 1, 2),
        )

        return (
            a_smem_layout_staged,
            b_smem_layout_staged,
            epi_smem_layout_staged,
        )

    @staticmethod
    def _make_tma_store_atoms_and_tensors(
        tensor_c,
        epi_smem_layout_staged,
        epi_tile: tuple,
    ) -> tuple:
        epi_smem_layout = cute.slice_(epi_smem_layout_staged, (None, None, 0))
        tma_atom_c, tma_tensor_c = cpasync.make_tiled_tma_atom(
            cpasync.CopyBulkTensorTileS2GOp(),
            tensor_c,
            epi_smem_layout,
            epi_tile,
        )
        return tma_atom_c, tma_tensor_c

    @staticmethod
    def _make_tma_atoms_and_tensors(
        tensor,
        smem_layout_staged,
        smem_tile: tuple,
        mcast_dim: int,
        internal_type=None,
    ) -> tuple:
        op = (
            cpasync.CopyBulkTensorTileG2SOp()
            if mcast_dim == 1
            else cpasync.CopyBulkTensorTileG2SMulticastOp()
        )
        smem_layout = cute.slice_(smem_layout_staged, (None, None, 0))
        tma_atom, tma_tensor = cpasync.make_tiled_tma_atom(
            op,
            tensor,
            smem_layout,
            smem_tile,
            num_multicast=mcast_dim,
            internal_type=internal_type,
        )
        return tma_atom, tma_tensor


_COMPILED_KERNELS: dict[tuple, object] = {}
_COMPILE_LOCK = threading.Lock()


def _compile_decode_kernel(
    device_index: int,
    tile_m: int,
    tile_n: int,
    *,
    cache_policy: bool,
    num_splits: int = 1,
):
    cache_key = (
        device_index,
        tile_m,
        tile_n,
        cache_policy,
        num_splits,
    )
    compiled = _COMPILED_KERNELS.get(cache_key)
    if compiled is not None:
        return compiled

    with _COMPILE_LOCK, torch.cuda.device(device_index):
        compiled = _COMPILED_KERNELS.get(cache_key)
        if compiled is not None:
            return compiled

        gemm = _Bf16GemmSm120Kernel(
            tile_m=tile_m,
            tile_n=tile_n,
            cache_policy=cache_policy,
            num_splits=num_splits,
        )
        # Symbolic-M/N/K fake tensors, compiled once per (tile, bias) bucket.
        # cute.compile targets the locally attached SM120 GPU (arch=sm_120).
        sym_m = cute.sym_int()
        sym_k = cute.sym_int()
        sym_n = cute.sym_int()
        a_fake = cute.runtime.make_fake_compact_tensor(
            cutlass.BFloat16,
            (sym_m, sym_k),
            stride_order=(1, 0),
            assumed_align=16,
        )
        b_fake = cute.runtime.make_fake_compact_tensor(
            cutlass.BFloat16,
            (sym_n, sym_k),
            stride_order=(1, 0),
            assumed_align=16,
        )
        if num_splits == 1:
            c_fake = cute.runtime.make_fake_compact_tensor(
                cutlass.BFloat16,
                (sym_m, sym_n),
                stride_order=(1, 0),
                assumed_align=16,
            )
        else:
            # Split-K workspace: fp32 (m, n, 2), compact along (n, m, l);
            # the (48, 32) fp32 epilogue TMA box needs stride[1] = 128B.
            c_fake = cute.runtime.make_fake_compact_tensor(
                cutlass.Float32,
                (sym_m, sym_n, 2),
                stride_order=(1, 0, 2),
                assumed_align=16,
            )
        stream_fake = cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True)
        compiled = cute.compile(
            gemm,
            a_fake,
            b_fake,
            c_fake,
            stream_fake,
            options="--opt-level 2 --enable-tvm-ffi",
        )
        _COMPILED_KERNELS[cache_key] = compiled
        return compiled




import os as _os

_GEMV_ENABLED = _os.environ.get("SGLANG_SM120_BF16_GEMV", "1") == "1"


def _gemv_graph_safe() -> bool:
    # The GEMV wrapper enforces capture safety itself: a shape whose JIT
    # module was never loaded outside capture raises a clear error (call
    # warmup_sm120_bf16_gemv at model-init, or rely on the eager warmup
    # forward before SGLang captures decode graphs). This flag is a kill
    # switch only.
    return _GEMV_ENABLED

def run_bf16_gemm_sm120(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None = None,
) -> torch.Tensor:
    """Dense BF16 GEMM ``x @ weight.T + bias`` on SM120 (decode M <= 48).

    Args:
        x: Activations, (M, K) row-major bf16.
        weight: Linear-layer weight, (N, K) row-major bf16.
        bias: Optional (N,) bf16 vector. Biased calls use the PyTorch fallback.

    Returns:
        (M, N) bf16 tensor.
    """
    if x.dtype != torch.bfloat16 or weight.dtype != torch.bfloat16:
        raise TypeError(f"bf16 inputs required, got {x.dtype} / {weight.dtype}")
    if bias is not None and bias.dtype != torch.bfloat16:
        raise TypeError(f"bf16 bias required, got {bias.dtype}")
    x = x.contiguous()
    weight = weight.contiguous()
    rows, k = x.shape
    n = weight.shape[0]
    if weight.shape[1] != k:
        raise ValueError(f"K mismatch: x K={k}, weight K={weight.shape[1]}")
    if bias is not None:
        bias = bias.contiguous()
        if bias.shape != (n,):
            raise ValueError(f"bias shape {tuple(bias.shape)} != ({n},)")
        # The optimized path is benchmarked for bias-free transformer
        # projections; retain the existing PyTorch path for biased calls.
        return torch.nn.functional.linear(x, weight, bias)
    # Skinny decode (M in {1,2,4}) takes the warp-per-row streaming GEMV:
    # no M predication is needed (the GEMV is row-exact) and it saturates
    # DRAM bandwidth that the mma tile cannot reach at these tiny M.
    if rows in (1, 2, 4):
        from sglang.kernels.ops.gemm.sm120_bf16_gemv import (
            sm120_bf16_gemv,
            use_sm120_bf16_gemv,
        )

        if use_sm120_bf16_gemv(rows, n, k) and _gemv_graph_safe():
            return sm120_bf16_gemv(x, weight)
        # The mma tile is unsafe for M<16 (no M predication). These shapes
        # stay on the torch path.
        return torch.nn.functional.linear(x, weight, bias)
    tile_key = _tile_key_for_shape(rows, n, k)
    tile_k = _TILE_CONFIGS[tile_key][0][2]
    if k % tile_k != 0:
        # The (48,64) bucket now wants tile_k=128 but the k-tile count is
        # floor(K/tile_k) (no predicated tail), so a non-divisible K must
        # fall back to a tile_k=64 config. (48,32) is the only tile_k=64
        # bucket that still covers M=48 (tile_m=48), and it stays in the
        # benchmark set for exactly this case.
        tile_key = (48, 32) if rows > 32 else _tile_key_for_m(rows)
    tile_m, tile_n = tile_key
    # Split-K is PERMANENTLY DISABLED: measured a net loss on every M=48
    # small-N large-K shape tried on SM120. (48,2560,9216): 0.74x with
    # split-K vs 0.81x plain; (48,4096,11008): 0.625x vs 0.703x. The SIMT
    # fp32 epilogue (direct global-store, no TMA) plus the extra fp32
    # reduction pass always costs more than the occupancy gain from the
    # doubled grid. The code path (_run_bf16_gemm_sm120_split_k) is kept
    # for reference but is unreachable; the small-N large-K rows
    # ((48,2560,9216)=0.81, (48,4096,12288)=0.79) stay on the plain
    # (48,32) tile, which is the best single-pass option measured.
    split_k = False
    device_index = x.device.index
    if device_index is None:
        device_index = torch.cuda.current_device()
    if split_k:
        return _run_bf16_gemm_sm120_split_k(
            x, weight, bias, rows, n, k, tile_m, tile_n, device_index
        )
    # cache_policy was keyed off rows == 1, but rows in {1,2,4} take the GEMV
    # path above and never reach this mma tile, so the M==1 evict-policy
    # variant was dead code that also forced a redundant recompile. The mma
    # tile always runs without the operand cache policy.
    kernel = _compile_decode_kernel(
        device_index,
        tile_m,
        tile_n,
        cache_policy=False,
    )
    output = torch.empty(rows, n, device=x.device, dtype=torch.bfloat16)
    kernel(x, weight, output)
    return output


def _run_bf16_gemm_sm120_split_k(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None,
    rows: int,
    n: int,
    k: int,
    tile_m: int,
    tile_n: int,
    device_index: int,
) -> torch.Tensor:
    """2-way split-K path for the K-long / small-N M=48 rows.

    The GEMM kernel is compiled with num_splits=2: the grid is
    (M/tile_m, N/tile_n, 2) and each z-slice accumulates half the K range
    into its own fp32 (m, n, split) workspace slab via a direct SIMT
    global-store epilogue (no atomics, deterministic). A tiny elementwise
    reduction then sums the two fp32 slabs (+ optional bias) and casts to
    bf16.
    """
    kernel = _compile_decode_kernel(
        device_index,
        tile_m,
        tile_n,
        cache_policy=False,
        num_splits=2,
    )
    # The compiled kernel expects the workspace as (m, n, 2):(n, 1, m*n),
    # i.e. split outermost with n contiguous (the compile-time fake tensor
    # uses stride_order=(1, 0, 2)). A plain torch.empty(rows, n, 2) is
    # (2n, 2, 1) -- split innermost -- which fails the JIT arg validator.
    # Allocate (2, rows, n) contiguous and permute to the (rows, n, 2)
    # logical view with strides (n, 1, rows*n): n stays the contiguous
    # dim, so the SIMT autovec_copy epilogue stays coalesced.
    slabs = torch.empty(2, rows, n, device=x.device, dtype=torch.float32)
    workspace = slabs.permute(1, 2, 0)
    kernel(x, weight, workspace)
    output = torch.empty(rows, n, device=x.device, dtype=torch.bfloat16)
    # fp32 sum of both K-halves (+ bias), then one bf16 convert. The order
    # is fixed by construction (slab 0 + slab 1 + bias), so the result is
    # deterministic across runs.
    partial = slabs[0] + slabs[1]
    if bias is not None:
        partial = partial + bias.unsqueeze(0).float()
    output.copy_(partial)
    return output
