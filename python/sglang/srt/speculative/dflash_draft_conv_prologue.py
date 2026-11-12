# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the SGLang project
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Portions adapted from vLLM (https://github.com/vllm-project/vllm, Apache-2.0) at commit
# 642c18d64c4cb6db3ded8fbe1786718cd4005fcb (PR #61867):
# vllm/models/kimi_k3/nvidia/specialized_decode/cutedsl/gemm/splitk.py (SplitKGemm) and gemm/projection.py,
# which adapt FlashInfer v0.7.0.post1 flashinfer/gemm/kernels/dense_bf16_gemm_sm100_splitk.py (Apache-2.0).
# The same SplitKGemm ships in sgl-nvidia-mega as sgl_nvidia_mega/kimi_k3/cutedsl/gemm/splitk.py.
# FlashInfer NOTICE: FlashInfer / Copyright 2025-2026 NVIDIA / Copyright 2023-2026 FlashInfer community (https://flashinfer.ai/)
# Modified by NVIDIA in 2026: the draft sublayer input's grouped conv in the split-K GEMM prologue.
"""Default-off conv prologue (SGLANG_OPT_DFLASH_DRAFT_CONV_PROLOGUE): the grouped conv of a draft sublayer's input (side 0) runs inside
the next split-K GEMM (fused qkv_a, or gate_up / the fused gate_up + SiLU-mul) instead of as its own kernel.

Served: AR site -> conv kernel projection -> dflash_grouped_conv (Triton) -> qkv_a / gate_up. With this knob the GEMM's
activation TMA loads the RAW rows (the AR site's normed output) into its B stage and the transform threads convolve the
stage in place before the MMA reads it: per element out[r, c] = bf16(fma(c0, x[r, c], c1 * x[r-1, c])) with
c_t = float(base[t, c]) + float(delta[r, t, c // GROUP]) and x[r-1] = 0 at block position 0, the FMA form the served
Triton kernel compiles to (its PTX: mul.f32x2 of the tap-1 coefficient and the shifted row, then fma.rn.f32x2 of the
tap-0 coefficient and the row into it).
The N tile is one draft block (8 rows), so row r-1 of every row is in the same stage.

Roles (SplitKGemm's warps): the activation warp issues the B TMAs onto ``bar_raw`` instead of ``bar_full``; the four
epilogue warps (idle until the MMA ends) transform first: thread t owns 8 columns (one 16-byte chunk) of K tile
``t // 16`` (+ 8 per round) across all 8 rows, so no other thread reads what it overwrites; each owner stores, executes
fence.proxy.async.shared::cta (its generic SMEM stores -> the MMA's async-proxy reads) and arrives on ``bar_full``,
whose count is 1 (the weight warp's expect_tx arrival) + 16 (the tile's owners). The MMA waits on ``bar_full`` as before.
The conv coefficients: the base kernel (a parameter, static) is read before the PDL wait, the conv projection (written by
the previous kernel) after it.
"""

from __future__ import annotations

import logging
from functools import cache

import cuda.bindings.driver as _cuda
import cutlass
import cutlass.cute as cute
import cutlass.utils as utils
import cutlass.utils.blackwell_helpers as sm100_utils
import torch
from cutlass import Float32, Int32, Int64, Uint32
from cutlass._mlir.dialects import llvm
from cutlass.cute import experimental as cute_ext
from cutlass.cute.runtime import make_fake_stream, make_fake_tensor
from cutlass.cutlass_dsl import CompileCallable, dsl_user_op
from sgl_nvidia_mega.kimi_k3.cutedsl.common.host import architecture, stream
from sgl_nvidia_mega.kimi_k3.cutedsl.common.primitives import bf16, bits_float, fma, load_words, pack_pair, store_words
from sgl_nvidia_mega.kimi_k3.cutedsl.gemm import splitk as _splitk
from sgl_nvidia_mega.kimi_k3.cutedsl.gemm.projection import PreparedProjection, matrix_view, set_oversized, smem_overflow
from sgl_nvidia_mega.kimi_k3.cutedsl.gemm.splitk import SplitKGemm, SplitKTactic

from sglang.srt.speculative.dflash_draft_ar_site import GROUP, GROUPS, HIDDEN, TAPS

logger = logging.getLogger(__name__)

_DSL_COMPILE = CompileCallable()
ROWS = 8  # one B1 draft block = the N tile
CHUNK = 8  # BF16 columns per 16-byte chunk = per transform thread and row
CHUNKS_PER_TILE = 128 // CHUNK  # 16 owners per K tile
TRANSFORM_THREADS = 128  # the epilogue warps 4-7
SLOTS = TRANSFORM_THREADS // CHUNKS_PER_TILE  # 8 owner slots of 16 threads (one K tile each at a time)


@dsl_user_op
def _fence_proxy_async_shared(*, loc=None, ip=None):
    """This thread's generic SMEM stores -> later async-proxy (tcgen05.mma) reads of them, ordered by the mbarrier
    arrive / wait that follows (PTX 9.7.15.4 fence.proxy; CUTLASS's fence_view_async_shared)."""
    llvm.inline_asm(
        None, [], "fence.proxy.async.shared::cta;", "~{memory}",
        has_side_effects=True, is_align_stack=False, asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc, ip=ip,
    )  # fmt: skip


def _chunk_offset(row, col):
    """Element offset of the 16-byte chunk at (row, col) (col a multiple of 8) in one B stage: K-major, 128-byte
    swizzle, atoms of 8 rows x 64 columns stacked along K (the activation TMA's layout for an 8-row N tile): the chunk
    index inside a 128-byte row XOR the row."""
    return (col // 64) * (ROWS * 64) + row * 64 + (((col % 64) // CHUNK) ^ row) * CHUNK


class ConvPrologueSplitKGemm(SplitKGemm):
    """``SplitKGemm`` whose B operand is the raw draft rows, convolved in SMEM before the MMA (module doc). ``k_tiles``:
    K tiles per split (K / 128 / split), fixed per weight. ``fenced_trigger`` must be 0 (the transform uses warp 3's
    neighbours only, but mode 2's warp-3 trigger is not needed by the GEMMs this serves)."""

    def __init__(self, *, tactic: SplitKTactic, k_tiles: int, act_fuse: bool = False, oversized: bool = False):
        super().__init__(tactic=tactic, fenced_trigger=0, proxy_fences=True, act_fuse=act_fuse, oversized=oversized)
        assert self.cta_n == ROWS, ("the N tile must be one draft block", self.cta_n)
        self.k_tiles = int(k_tiles)
        stages = self.num_ab_stage
        self.passes = -(-self.k_tiles // stages)  # passes over the B stages
        self.stage_sets = -(-stages // SLOTS)  # stages per owner slot

    @cute.experimental.jit
    def __call__(
        self, a: cute.Tensor, b: cute.Tensor, c: cute.Tensor, bias: cute.Tensor,
        a_map: cutlass.Int64, proj: cutlass.Int64, base: cutlass.Int64, stream: _cuda.CUstream,
    ):  # fmt: skip
        m_rows = a.layout.shape[0] if cutlass.const_expr(self.act_fuse) else c.layout.shape[0]
        self.kernel(a, b, c, bias, a_map, proj, base).launch(
            grid=(
                cute.ceil_div(m_rows, self.cta_m),
                cute.ceil_div(c.layout.shape[1], self.cta_n) * self.split_k,
                c.layout.shape[2],
            ),
            block=(self.threads_per_cta, 1, 1),
            cluster=self.cluster_shape,
            smem=cute.Int64(self.smem_alloc),
            stream=stream,
            use_pdl=True,
        )

    @cute.experimental.kernel
    def kernel(
        self, mA: cute.Tensor, mB: cute.Tensor, mC: cute.Tensor, mBias: cute.Tensor,
        a_map: cutlass.Int64, proj: cutlass.Int64, base: cutlass.Int64,
    ):  # fmt: skip
        """``SplitKGemm.kernel`` with ``bar_raw`` and the transform (module doc). ``proj``: the conv kernel projection
        [rows, 2 * TAPS * GROUPS] BF16 (side 0 = the first TAPS * GROUPS columns); ``base``: the side-0 base kernel
        [TAPS, HIDDEN] BF16."""
        stages = self.num_ab_stage
        ab_dtype = mA.element_type
        majors = (utils.LayoutEnum.from_tensor(t).mma_major_mode() for t in (mA, mB))
        tiled_mma = sm100_utils.make_trivial_tiled_mma(
            ab_dtype, ab_dtype, *majors, self.acc_dtype, self.cta_group,
            self.mma_tiler_mn,
        )  # fmt: skip

        mnk_tiler = (self.mma_tiler_mn[0], self.mma_tiler_mn[1], self.cta_k)
        block_idx = cute.arch.block_idx()
        bidx = block_idx[0]
        split_rank = cute.arch.make_warp_uniform(cute.arch.block_idx_in_cluster())
        n_idx = block_idx[1] // self.split_k
        l_idx = block_idx[2]
        warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())

        shape = (tiled_mma, mnk_tiler, ab_dtype, stages)
        sA = _splitk._smem(ab_dtype, None, 1024, sm100_utils.make_smem_layout_a(*shape))
        sB = _splitk._smem(ab_dtype, None, 1024, sm100_utils.make_smem_layout_b(*shape))
        acc_layout = cute_ext.make_tmem_layout_acc(
            tiled_mma, self.mma_tiler_mn, acc_stage=1
        )
        bar_full = _splitk._smem(cutlass.Int64, stages, 8).iterator
        bar_empty = _splitk._smem(cutlass.Int64, stages, 8).iterator
        bar_raw = _splitk._smem(cutlass.Int64, stages, 8).iterator  # conv prologue: the raw B stage landed
        bar_tma_epilog = _splitk._smem(cutlass.Int64, 1, 8).iterator  # FlashInfer's bias
        bar_mma_epilog = _splitk._smem(cutlass.Int64, 1, 8).iterator
        bar_tmem_alloc = _splitk._smem(cutlass.Int64, 1, 8).iterator
        tmem_base_ptr = _splitk._smem(cutlass.Int32, 1, 4).iterator
        mailbox, bar_reduce = sA, bar_mma_epilog  # split 1: compile-time dead
        if cutlass.const_expr(self.split_k > 1):
            mailbox = _splitk._smem(cutlass.Float32, self.mailbox_elements, 128)
            bar_reduce = _splitk._smem(cutlass.Int64, 1, 8).iterator
        stage_d = sA  # act_fuse only: the owner's BF16 tile, (mma_m, mma_n) M-major
        if cutlass.const_expr(self.act_fuse):
            stage_d = _splitk._smem(
                cutlass.BFloat16, None, 16,
                cute.make_layout((self.cta_m, self.cta_n), stride=(1, self.cta_m)),
            )  # fmt: skip

        if warp_idx == 0:
            with cute.arch.elect_one():
                for i in range(stages):
                    # conv prologue: the weight warp's expect_tx arrival + one arrival per transform owner of the tile.
                    cute.arch.mbarrier_init(bar_full + i, 1 + CHUNKS_PER_TILE)
                    cute.arch.mbarrier_init(bar_empty + i, 1)
                    cute.arch.mbarrier_init(bar_raw + i, 1)
                cute.arch.mbarrier_init(bar_tma_epilog, 32)
                cute.arch.mbarrier_init(bar_mma_epilog, 1)
                cute.arch.mbarrier_init(bar_tmem_alloc, 160)
                if cutlass.const_expr(self.split_k > 1):
                    cute.arch.mbarrier_init(bar_reduce, 1)

        cute.arch.mbarrier_init_fence()
        if cutlass.const_expr(self.split_k > 1):
            cute.arch.cluster_arrive_relaxed()
        else:
            cute.arch.barrier()

        k_tile_count = cute.size(mA, mode=[1]) // self.cta_k // self.split_k
        k_tile_start = split_rank * k_tile_count

        if cutlass.const_expr(self.split_k > 1):
            cute.arch.cluster_wait()

        tiles = (k_tile_start, k_tile_count)
        if warp_idx == 0:
            if cutlass.const_expr(self.act_fuse):
                half = self.cta_m // 2
                up0 = cute.size(mA, mode=[0]) // 2 + bidx * half
                self.weight_warp_pair(bar_full, bar_empty, sA, a_map, bidx * half, up0, *tiles)
            else:
                self.weight_warp(bar_full, bar_empty, sA, a_map, bidx * self.cta_m, *tiles)
        elif warp_idx == 1:
            b_tile = cute.local_tile(mB, (self.cta_n, self.cta_k), (n_idx, None, l_idx))
            b_map = cute_ext.get_cta_v_map_ab(mB, mnk_tiler, tiled_mma, "B")
            # conv prologue: the raw stage lands on bar_raw (same TMA, fences and expect_tx as the served activation warp).
            self.activation_warp(bar_raw, bar_empty, b_tile, sB, b_map, *tiles)
        elif warp_idx == 2:
            k_blocks = self.cta_k // cute.size(tiled_mma.shape_mnk, mode=[2])
            self.mma_warp(
                bar_full, bar_empty, bar_mma_epilog, bar_tmem_alloc, tiled_mma, sA, sB,
                tmem_base_ptr, acc_layout, k_blocks, k_tile_count,
            )  # fmt: skip
        elif warp_idx >= 4:
            t = cute.arch.thread_idx()[0] - 128
            slot = t // CHUNKS_PER_TILE
            chunk = t - slot * CHUNKS_PER_TILE
            col = chunk * CHUNK  # this thread's columns inside a K tile
            rows = mB.layout.shape[0]
            # Owners by STAGE, not by tile: slot s owns stages s, s + SLOTS, ... and every pass over them, visited in tile
            # order (pass outer), so the thread that waits on bar_raw[stage] for pass q has itself waited for pass q - 1:
            # a parity wait is unambiguous only then (with fewer stages than K tiles, a fresh owner of pass 1 would pass
            # a parity-1 wait on a barrier still in phase 0 at once: a launch failure at the (128, 8, 7, 5) gate_up tactic).
            # Base coefficients (a parameter) before the wait: per (pass, stage set), tap, this thread's 8 columns.
            base_words = []
            for r in cutlass.range_constexpr(self.passes * self.stage_sets):
                tile_b = cutlass.min((r // self.stage_sets) * stages + slot + (r % self.stage_sets) * SLOTS, self.k_tiles - 1)
                k0 = (k_tile_start + tile_b) * self.cta_k + col
                for tap in cutlass.range_constexpr(TAPS):
                    base_words.append(load_words(base + Int64(tap * HIDDEN + k0) * 2, count=4))
            cute.arch.griddepcontrol_wait()  # the conv projection is written by the previous kernel
            for r in cutlass.range_constexpr(self.passes * self.stage_sets):
                stage = slot + (r % self.stage_sets) * SLOTS
                tile = (r // self.stage_sets) * stages + stage
                # stage < stages and tile < k_tiles as one comparison (the bound is a compile-time int).
                if stage < min(stages, self.k_tiles - (r // self.stage_sets) * stages):
                    group = (k_tile_start + tile) * (self.cta_k // GROUP) + col // GROUP
                    delta = []  # [row][tap], FP32 of the BF16 projection value
                    for row in cutlass.range_constexpr(ROWS):
                        rg = cutlass.min(n_idx * self.cta_n + row, rows - 1)
                        for tap in cutlass.range_constexpr(TAPS):
                            index = rg * (2 * TAPS * GROUPS) + tap * GROUPS + group
                            word = load_words(proj + Int64(index // 2) * 4, count=1, mutable=True)[0]
                            delta.append(bits_float((word >> (Uint32(index % 2) * Uint32(16))) << Uint32(16)))
                    cute.arch.mbarrier_wait(bar_raw + stage, (r // self.stage_sets) % 2)
                    s_stage = sB[None, None, None, stage].iterator
                    raw = []
                    for row in cutlass.range_constexpr(ROWS):
                        raw.append(load_words(s_stage + _chunk_offset(row, col), count=4, shared=True, mutable=True))
                    for row in cutlass.range_constexpr(ROWS):
                        words = []
                        for k in cutlass.range_constexpr(CHUNK // 2):
                            pair = []
                            for h in (False, True):
                                c0 = bf16(base_words[2 * r][k], h) + delta[row * TAPS]
                                c1 = bf16(base_words[2 * r + 1][k], h) + delta[row * TAPS + 1]
                                prev = Float32(0.0)
                                if cutlass.const_expr(row > 0):  # position 0 of the block: x[r-1] = 0
                                    prev = bf16(raw[row - 1][k], h)
                                pair.append(fma(c0, bf16(raw[row][k], h), c1 * prev))
                            words.append(pack_pair(pair[0], pair[1], policy=None))
                        store_words(s_stage + _chunk_offset(row, col), tuple(words), shared=True)
                    _fence_proxy_async_shared()
                    cute.arch.mbarrier_arrive(bar_full + stage)
            if cutlass.const_expr(self.act_fuse):
                gtile = cute.local_tile(mC, (self.cta_m // 2, self.cta_n), (bidx, n_idx, l_idx))
                rows_left = cute.size(mC, mode=[1]) - n_idx * self.cta_n
                self.epilog_warp_act(
                    bar_mma_epilog, bar_tmem_alloc, tmem_base_ptr, acc_layout, gtile,
                    cute.arch.thread_idx()[0] - 128, mC.element_type,
                    utils.LayoutEnum.from_tensor(mC), mailbox, bar_reduce, split_rank, stage_d, rows_left,
                )  # fmt: skip
            else:
                gtile = cute.local_tile(mC, (self.cta_m, self.cta_n), (bidx, n_idx, l_idx))
                self.epilog_warp(
                    bar_mma_epilog, bar_tmem_alloc, tmem_base_ptr, acc_layout, gtile,
                    cute.arch.thread_idx()[0] - 128, mC.element_type,
                    utils.LayoutEnum.from_tensor(mC), mailbox, bar_reduce, split_rank,
                )  # fmt: skip


def _fake(leading, dtype=cutlass.BFloat16):
    shape = tuple(cute.sym_int() for _ in range(3))
    stride = (1, cute.sym_int(64), 0) if leading == 0 else (cute.sym_int(64), 1, 0)
    return make_fake_tensor(dtype, shape, stride, assumed_align=16)


@cache
def compile_conv_prologue(tactic, k_tiles: int, act_fuse: bool, oversized: bool, arch: str):
    kernel = ConvPrologueSplitKGemm(tactic=SplitKTactic(*tactic), k_tiles=k_tiles, act_fuse=act_fuse, oversized=oversized)
    y = _fake(0)
    args = (_fake(1), _fake(1), y, y, Int64(0), Int64(0), Int64(0), make_fake_stream())
    compiled = _DSL_COMPILE(kernel, *args, options=f"--gpu-arch={arch}")
    overflow = smem_overflow(compiled)
    if overflow is not None:  # SMEM overflow check, as compile_projection
        raise ValueError(f"conv prologue tactic {tuple(tactic)} uses more shared memory than its launch allocates ({overflow} B)")
    return compiled


class DraftConvPrologueProjection:
    """Conv-prologue form of one draft weight [N, HIDDEN] (fused qkv_a, gate_up, or the merged gate_up with ``act_fuse``) for
    1..ROWS rows at ``tactic`` (the served <= 8-row tactic or its OV form: same tiles, split and sum)."""

    def __init__(self, weight: torch.Tensor, tactic, block_size: int, act_fuse: bool = False, oversized: bool = False):
        n, k = weight.shape
        if k != HIDDEN or weight.dtype != torch.bfloat16 or tactic[1] != ROWS or block_size != ROWS:
            raise ValueError(f"conv prologue: weight {tuple(weight.shape)} {weight.dtype}, tactic {tactic}, block {block_size}")
        if k % (128 * tactic[2]):
            raise ValueError(f"conv prologue: tactic {tactic} does not divide K={k}")
        self.k, self.tactic, self.act_fuse = k, tuple(tactic), bool(act_fuse)
        self.n = n // 2 if act_fuse else n
        # Only the weight checks, the weight view and its TMA map: the kernel is defined here.
        self._prepared = PreparedProjection(weight, {ROWS: self.tactic}, proxy_fences=True, act_fuse=act_fuse)
        self._extra = self._prepared.weight_map(self.tactic[0] // 2 if act_fuse else self.tactic[0])
        k_tiles = k // 128 // self.tactic[2]
        with torch.accelerator.device_index(weight.device.index):
            self._code = compile_conv_prologue(self.tactic, k_tiles, self.act_fuse, bool(oversized), architecture(weight.device))
            if oversized:
                set_oversized(self._code, weight.device)

    def covers(self, x: torch.Tensor) -> bool:
        return (
            x.dim() == 2
            and 0 < x.shape[0] <= ROWS
            and x.shape[1] == self.k
            and x.dtype == torch.bfloat16
            and x.stride(1) == 1
            and x.stride(0) % 8 == 0
            and x.data_ptr() % 16 == 0
        )

    def run(self, x: torch.Tensor, projection: torch.Tensor, base: torch.Tensor, out=None) -> torch.Tensor:
        """GEMM of conv side 0 of the raw rows ``x``: ``projection`` the conv kernel projection [rows, 2 * TAPS * GROUPS],
        ``base`` the side-0 base kernel [TAPS, HIDDEN]."""
        rows = x.shape[0]
        if projection.shape != (rows, 2 * TAPS * GROUPS) or not projection.is_contiguous() or projection.dtype != torch.bfloat16:
            raise ValueError(f"conv prologue: conv projection {tuple(projection.shape)} {projection.dtype}")
        if base.shape != (TAPS, HIDDEN) or not base.is_contiguous() or base.dtype != torch.bfloat16:
            raise ValueError(f"conv prologue: base kernel {tuple(base.shape)} {base.dtype}")
        if out is None:
            out = x.new_empty((rows, self.n))
        y = matrix_view(out, output=True)
        self._code(
            self._prepared.weight_view, matrix_view(x), y, y, *self._extra, Int64(projection.data_ptr()),
            Int64(base.data_ptr()), stream(x.device),
        )  # fmt: skip
        return out


def take_pending(conv, x: torch.Tensor):
    """The conv projection that ``conv.prepare_projection`` left for its consumer GEMM, if any; ``x`` must be the raw
    rows it returned (anything else would be a GEMM of unconvolved rows: raise)."""
    pending = getattr(conv, "conv_pending", None)
    if pending is None:
        return None
    raw, projection = pending
    if x is not raw and (x.data_ptr(), x.shape, x.stride()) != (raw.data_ptr(), raw.shape, raw.stride()):
        raise RuntimeError("conv prologue: the conv's consumer GEMM got another tensor than the raw rows prepare_projection returned")
    conv.conv_pending = None
    return projection


class DraftConvPrologueLinearMethod:
    """quant_method of fused qkv_a / gate_up under the conv prologue: a call carrying its conv's pending raw rows runs
    the conv-prologue GEMM; every other call goes to ``inner`` (the served split-K method)."""

    def __init__(self, inner, conv, d2: DraftConvPrologueProjection):
        self.inner, self.conv, self.d2 = inner, conv, d2

    def __getattr__(self, name):
        return getattr(self.inner, name)

    def apply(self, layer, x, bias=None):
        projection = take_pending(self.conv, x)
        if projection is None:
            return self.inner.apply(layer, x, bias)
        if bias is not None or not self.d2.covers(x):
            raise RuntimeError(f"conv prologue: pending raw rows {tuple(x.shape)} not covered by the conv-prologue GEMM")
        return self.d2.run(x, projection, self.conv.base_kernel.detach()[0])

    def apply_into(self, layer, x, output, bias=None):
        if getattr(self.conv, "conv_pending", None) is not None:
            raise RuntimeError("conv prologue: apply_into with pending raw rows (not routed)")
        return self.inner.apply_into(layer, x, output, bias)


class DraftConvPrologueFused:
    """DFlashMLP.split_k_gate_up_act under the conv prologue (the fused gate_up + SiLU-mul): covers / run like the served projection."""

    def __init__(self, inner, conv, d2: DraftConvPrologueProjection):
        self.inner, self.conv, self.d2 = inner, conv, d2

    def covers(self, x):
        return self.inner.covers(x)

    def run(self, x, out=None):
        projection = take_pending(self.conv, x)
        if projection is None:
            return self.inner.run(x, out)
        return self.d2.run(x, projection, self.conv.base_kernel.detach()[0], out)

    def tensor_maps(self):
        return self.inner.tensor_maps()


def install_draft_conv_prologue(draft_model) -> int:
    """SGLANG_OPT_DFLASH_DRAFT_CONV_PROLOGUE (with SGLANG_OPT_DFLASH_DRAFT_SPLITK and the draft AR sites): each layer's
    attention conv side 0 runs inside fused qkv_a and the MLP conv side 0 inside gate_up (or the fused gate_up + SiLU-mul),
    at the <= 8-row tactic the served projection uses (its OV form with SGLANG_OPT_DFLASH_DRAFT_SPLITK_OV). With
    SGLANG_OPT_DFLASH_DRAFT_CONV_PROLOGUE_MLP_ONLY only the MLP conv is routed; the attention conv and qkv_a stay served.
    Returns how many GEMMs were routed."""
    from sglang.srt.environ import envs
    from sglang.srt.speculative.dflash_draft_splitk import OV_TACTICS, SPLITK_TACTICS, DraftSplitKLinearMethod

    use_ov = envs.SGLANG_OPT_DFLASH_DRAFT_SPLITK_OV.get()
    mlp_only = envs.SGLANG_OPT_DFLASH_DRAFT_CONV_PROLOGUE_MLP_ONLY.get()

    def build(weight, conv, act_fuse=False):
        shape = tuple(weight.shape)
        tactic = OV_TACTICS.get(shape) if use_ov else None
        oversized = tactic is not None
        tactic = tactic or SPLITK_TACTICS.get(shape)
        if tactic is None:
            raise ValueError(f"SGLANG_OPT_DFLASH_DRAFT_CONV_PROLOGUE: no <= 8-row tactic for {shape}")
        return DraftConvPrologueProjection(weight.detach(), tactic, conv.block_size, act_fuse=act_fuse, oversized=oversized)

    installed = 0
    for layer in draft_model.layers:
        attn, mlp = layer.self_attn, layer.mlp
        if not mlp_only:
            qkv_a = attn.fused_qkv_a_proj_with_mqa
            if not isinstance(qkv_a.quant_method, DraftSplitKLinearMethod):
                raise ValueError("SGLANG_OPT_DFLASH_DRAFT_CONV_PROLOGUE needs SGLANG_OPT_DFLASH_DRAFT_SPLITK on qkv_a")
            conv = layer.attention_conv
            conv.conv_prologue = build(qkv_a.weight, conv)
            qkv_a.quant_method = DraftConvPrologueLinearMethod(qkv_a.quant_method, conv, conv.conv_prologue)
            installed += 1
        conv = layer.mlp_conv
        if mlp.split_k_gate_up_act is not None:
            conv.conv_prologue = build(mlp.gate_up_proj.weight, conv, act_fuse=True)
            mlp.split_k_gate_up_act = DraftConvPrologueFused(mlp.split_k_gate_up_act, conv, conv.conv_prologue)
        else:
            gate_up = mlp.gate_up_proj
            if not isinstance(gate_up.quant_method, DraftSplitKLinearMethod):
                raise ValueError("SGLANG_OPT_DFLASH_DRAFT_CONV_PROLOGUE needs SGLANG_OPT_DFLASH_DRAFT_SPLITK on gate_up")
            conv.conv_prologue = build(gate_up.weight, conv)
            gate_up.quant_method = DraftConvPrologueLinearMethod(gate_up.quant_method, conv, conv.conv_prologue)
        installed += 1
    logger.info(
        "DFLASH draft conv prologue: grouped conv side 0 inside %d GEMMs (OV %s, MLP only %s)", installed, use_ov, mlp_only
    )
    return installed
