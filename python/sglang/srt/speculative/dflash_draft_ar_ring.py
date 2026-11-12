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
# Modified by NVIDIA in 2026: ring o_proj / down_proj publishing into the DFlash draft all-reduce sites.
"""Default-off ring o_proj / down_proj for the DFlash draft all-reduce sites (SGLANG_OPT_DFLASH_DRAFT_AR_RING).

With SGLANG_OPT_DFLASH_DRAFT_AR_SITE, each site loads this rank's BF16 partial (the split-K o_proj / down_proj output),
convolves it (grouped conv side 1) and multicasts the result into slot ``rank`` of every rank's mailbox, then polls.
With this knob the projection does the first half in its epilogue instead of storing the partial: the owner CTA casts
its FP32 tile to BF16 (the bits the projection would have stored), stages the tile in shared memory, and each epilogue
thread convolves one row's 8 columns with the row before it in the same draft block (both are in the tile) and
multicasts the result into the slot. The site then only polls (DraftArSite ``polled``). The contribution bits are the
same as the site computes from the stored partial, so the site's outputs are bitwise those of the unfused pair.

The form (the row-parallel projection's epilogue publishes into the sites' Lamport slots; the sites read no partial) is
based on earlier NVIDIA work, on the megakernel plugin's SplitKGemm (vLLM's, see the header),
with this drafter's grouped conv folded into the publish.
"""

from __future__ import annotations

import logging
from contextlib import contextmanager
from functools import cache
from typing import Optional

import cuda.bindings.driver as _cuda
import cutlass
import cutlass.cute as cute
import cutlass.utils as utils
import cutlass.utils.blackwell_helpers as sm100_utils
import torch
from cutlass import Float32, Int32, Int64, Uint32
from cutlass.cute import experimental as cute_ext
from cutlass.cute.runtime import make_fake_stream, make_fake_tensor
from cutlass.cutlass_dsl import CompileCallable
from sgl_nvidia_mega.kimi_k3.cutedsl.common.host import architecture, stream
from sgl_nvidia_mega.kimi_k3.cutedsl.common.primitives import (
    bf16,
    bits_float,
    fma,
    load_words,
    multicast_words,
    pack_pair,
)
from sgl_nvidia_mega.kimi_k3.cutedsl.gemm import splitk as _splitk
from sgl_nvidia_mega.kimi_k3.cutedsl.gemm.projection import PreparedProjection, matrix_view, smem_overflow
from sgl_nvidia_mega.kimi_k3.cutedsl.gemm.splitk import SplitKGemm, SplitKTactic

from sglang.srt.speculative.dflash_draft_ar_site import GROUP, GROUPS, HIDDEN, TAPS
from sglang.srt.speculative.dflash_draft_splitk import DraftSplitKLinearMethod

logger = logging.getLogger(__name__)

_DSL_COMPILE = CompileCallable()
ROWS = 8  # one B1 draft block: the rows the served split-K o_proj / down_proj tactics cover


class RingConvSplitKGemm(SplitKGemm):
    """``SplitKGemm`` (split 1, no act_fuse) whose owner multicasts its conv-folded BF16 rows into a ring slot instead of
    storing C. ``block_size``: the draft block (conv position = row % block_size); it must divide the N tile so a row's
    predecessor in its block is in the same tile."""

    def __init__(self, *, tactic: SplitKTactic, block_size: int, fenced_trigger: int = 0, early_coef: bool = False):
        # fenced_trigger 1 (SGLANG_OPT_DFLASH_DRAFT_AR_DEFERRED): the activation warp releases the dependents right after
        # its wait, behind fence.acq_rel.gpu (SplitKGemm's mode 1; the trigger_after_wait form).
        # early_coef (SGLANG_OPT_DFLASH_DRAFT_RING_EARLY_COEF): the epilogue threads wait and load the conv coefficients
        # while the main loop runs instead of after the tile is staged.
        super().__init__(tactic=tactic, fenced_trigger=fenced_trigger, proxy_fences=True)
        assert self.split_k == 1, "ring projections use one K slice (the served o_proj / down_proj tactics)"
        assert self.cta_n % block_size == 0, (self.cta_n, block_size)
        assert self.cta_n * self.cta_m // 8 <= self.epilog_threads, "one 8-column fragment per epilogue thread"
        self.block_size = block_size
        self.early_coef = bool(early_coef)

    @cute.experimental.jit
    def __call__(
        self, a: cute.Tensor, b: cute.Tensor, c: cute.Tensor, bias: cute.Tensor,
        a_map: cutlass.Int64, proj: cutlass.Int64, base: cutlass.Int64, ring: cutlass.Int64,
        stream: _cuda.CUstream,
    ):  # fmt: skip
        self.kernel(a, b, c, bias, a_map, proj, base, ring).launch(
            grid=(
                cute.ceil_div(c.layout.shape[0], self.cta_m),
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
        a_map: cutlass.Int64, proj: cutlass.Int64, base: cutlass.Int64, ring: cutlass.Int64,
    ):  # fmt: skip
        """``SplitKGemm.kernel`` at split 1 with the owner's BF16 tile staged in shared memory, then the publish.
        ``proj``: the conv kernel projection [rows, 2 * TAPS * GROUPS] (BF16); ``base``: the side-1 base kernel
        [TAPS, HIDDEN] (BF16); ``ring``: the multicast address of this rank's slot [CAP, HIDDEN] (BF16)."""
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
        bar_tma_epilog = _splitk._smem(cutlass.Int64, 1, 8).iterator  # FlashInfer's bias
        bar_mma_epilog = _splitk._smem(cutlass.Int64, 1, 8).iterator
        bar_tmem_alloc = _splitk._smem(cutlass.Int64, 1, 8).iterator
        tmem_base_ptr = _splitk._smem(cutlass.Int32, 1, 4).iterator
        # The owner's BF16 tile, (mma_m, mma_n) M-major like C (as act_fuse stages it).
        stage_d = _splitk._smem(
            cutlass.BFloat16, None, 16,
            cute.make_layout((self.cta_m, self.cta_n), stride=(1, self.cta_m)),
        )  # fmt: skip

        if warp_idx == 0:
            with cute.arch.elect_one():
                for i in range(stages):
                    cute.arch.mbarrier_init(bar_full + i, 2)
                    cute.arch.mbarrier_init(bar_empty + i, 1)
                cute.arch.mbarrier_init(bar_tma_epilog, 32)
                cute.arch.mbarrier_init(bar_mma_epilog, 1)
                cute.arch.mbarrier_init(bar_tmem_alloc, 160)

        cute.arch.mbarrier_init_fence()
        cute.arch.barrier()

        k_tile_count = cute.size(mA, mode=[1]) // self.cta_k
        tiles = (Int32(0), k_tile_count)
        # Warp 3 is idle; warps 4-7 run the epilogue.
        if warp_idx == 0:
            self.weight_warp(bar_full, bar_empty, sA, a_map, bidx * self.cta_m, *tiles)
        elif warp_idx == 1:
            b_tile = cute.local_tile(mB, (self.cta_n, self.cta_k), (n_idx, None, l_idx))
            b_map = cute_ext.get_cta_v_map_ab(mB, mnk_tiler, tiled_mma, "B")
            self.activation_warp(bar_full, bar_empty, b_tile, sB, b_map, *tiles)
        elif warp_idx == 2:
            k_blocks = self.cta_k // cute.size(tiled_mma.shape_mnk, mode=[2])
            self.mma_warp(
                bar_full, bar_empty, bar_mma_epilog, bar_tmem_alloc, tiled_mma, sA, sB,
                tmem_base_ptr, acc_layout, k_blocks, k_tile_count,
            )  # fmt: skip
        elif warp_idx >= 4:
            epi_tid = cute.arch.thread_idx()[0] - 128
            chunks = self.cta_m // 8
            token = epi_tid // chunks
            chunk = epi_tid - token * chunks
            width, rows = mC.layout.shape[0], mC.layout.shape[1]
            n, m = n_idx * self.cta_n + token, bidx * self.cta_m + chunk * 8
            if cutlass.const_expr(self.early_coef):
                # The conv coefficients while the main loop runs: the conv projection is written by an earlier kernel of
                # the step, so this thread reads it after its own wait. Clamped indices keep idle threads' loads in bounds.
                cute.arch.griddepcontrol_wait()
                nc, mc = cutlass.min(n, rows - 1), cutlass.min(m, width - 8)
                group = mc // GROUP
                coef = []
                for tap in cutlass.range_constexpr(TAPS):
                    # coefficient(tap) = base[tap, c] + delta[row, tap, c // 16]; delta = proj[row, (TAPS + tap) * GROUPS
                    # + g], the side-1 half of the [rows, 2, TAPS, GROUPS] projection, as dflash_grouped_conv reads it.
                    index = nc * (2 * TAPS * GROUPS) + (TAPS + tap) * GROUPS + group
                    word = load_words(proj + Int64(index // 2) * 4, count=1, mutable=True)[0]
                    delta = bits_float((word >> (Uint32(index % 2) * Uint32(16))) << Uint32(16))
                    b = load_words(base + Int64(tap * HIDDEN + mc) * 2, count=4)
                    coef.append([bf16(b[k], h) + delta for k in range(4) for h in (False, True)])
            self.epilog_stage(
                bar_mma_epilog, bar_tmem_alloc, tmem_base_ptr, acc_layout, epi_tid, mC.element_type,
                utils.LayoutEnum.from_tensor(mC), stage_d,
            )  # fmt: skip
            if cutlass.const_expr(not self.early_coef):
                # The conv projection is written by an earlier kernel of the step: this thread reads it after its own wait.
                cute.arch.griddepcontrol_wait()
            # Each thread convolves one 8-column fragment of one row (DraftArSite.contribution's arithmetic on the staged
            # BF16 rows) and multicasts it into the ring slot.
            if token < self.cta_n and n < rows and m < width:
                source = stage_d.iterator + token * self.cta_m + chunk * 8
                cur = load_words(source, count=4, shared=True, mutable=True)
                y = [Float32(0)] * 8
                for k in cutlass.range_constexpr(4):
                    for h in (False, True):
                        y[2 * k + int(h)] = bf16(cur[k], h)
                if cutlass.const_expr(not self.early_coef):
                    # The same coefficients, loaded here (the pre-early-coefficient form; its own names, all local to
                    # this branch).
                    late_coef = []
                    late_group = m // GROUP
                    for late_tap in cutlass.range_constexpr(TAPS):
                        late_index = n * (2 * TAPS * GROUPS) + (TAPS + late_tap) * GROUPS + late_group
                        late_word = load_words(proj + Int64(late_index // 2) * 4, count=1, mutable=True)[0]
                        late_delta = bits_float((late_word >> (Uint32(late_index % 2) * Uint32(16))) << Uint32(16))
                        late_b = load_words(base + Int64(late_tap * HIDDEN + m) * 2, count=4)
                        late_coef.append([bf16(late_b[k], h) + late_delta for k in range(4) for h in (False, True)])
                prev = [Float32(0)] * 8
                if n % self.block_size > 0:  # the row before is in this tile: block_size divides the N tile
                    before_row = stage_d.iterator + (token - 1) * self.cta_m + chunk * 8
                    before = load_words(before_row, count=4, shared=True, mutable=True)
                    for k in cutlass.range_constexpr(4):
                        for h in (False, True):
                            prev[2 * k + int(h)] = bf16(before[k], h)
                if cutlass.const_expr(self.early_coef):
                    for e in cutlass.range_constexpr(8):
                        y[e] = fma(coef[1][e], prev[e], coef[0][e] * y[e])
                else:
                    for e in cutlass.range_constexpr(8):
                        y[e] = fma(late_coef[1][e], prev[e], late_coef[0][e] * y[e])
                words = tuple(pack_pair(y[2 * k], y[2 * k + 1], policy="latent") for k in range(4))
                multicast_words(ring + (Int64(n) * Int64(width) + Int64(m)) * 2, words, kind="bf16x2")

    @cute.experimental.jit
    def epilog_stage(
        self, bar_mma_epilog, bar_tmem_alloc, tmem_base_ptr, acc_layout: cutlass.Constexpr,
        epi_tid: cutlass.Int32, c_dtype: cutlass.Constexpr, d_layout: cutlass.Constexpr, stage_d: cute.Tensor,
    ):  # fmt: skip
        """``epilog_warp`` up to the owner's cast (the same TMEM load and cast), then the tile to shared memory and a
        sync of the 128 epilogue threads (named barrier 1): every staged value before any read of it."""
        cute.arch.mbarrier_arrive(bar_tmem_alloc)
        cute.arch.mbarrier_wait(bar_tmem_alloc, 0)
        _splitk._fence_tcgen05()

        acc_view = cute.make_tensor(
            cute.arch.retrieve_tmem_ptr(self.acc_dtype, 16, tmem_base_ptr),
            acc_layout,
        )[((None, None), 0, 0, 0)]
        epi_tile = (self.cta_m, self.cta_n)
        load_op = sm100_utils.get_tmem_load_op(
            (self.cta_m, self.cta_n, self.cta_k), d_layout, c_dtype, self.acc_dtype,
            epi_tile, False,
        )  # fmt: skip
        tiled_copy_t2r = cute.nvgpu.tcgen05.make_tmem_copy(load_op, acc_view)
        sD_epi = cute.flat_divide(stage_d, epi_tile)

        rmem_layout = cute_ext.make_t2r_rmem_layout(tiled_copy_t2r, sD_epi, epi_tid)
        rmem = cute.AddressSpace.rmem
        rAcc = cute_ext.allocate(self.acc_dtype, rmem, rmem_layout, alignment=32)
        rD = cute_ext.allocate(c_dtype, rmem, rmem_layout, alignment=32)
        thr_t2r = tiled_copy_t2r.get_slice(epi_tid)

        cute.arch.mbarrier_wait(bar_mma_epilog, 0)
        _splitk._fence_tcgen05()
        cute_ext.partition_and_copy(thr_t2r, acc_view, rAcc)
        cute.arch.fence_view_async_tmem_load()
        _splitk._fence_tcgen05(before=True)
        cute.arch.mbarrier_arrive(bar_tmem_alloc)

        rD.store(rAcc.load().to(c_dtype))
        cute_ext.partition_and_copy(thr_t2r, rD, sD_epi[None, None, 0, 0])
        cute.arch.barrier(barrier_id=1, number_of_threads=self.epilog_threads)


def _fake(leading, dtype=cutlass.BFloat16):
    shape = tuple(cute.sym_int() for _ in range(3))
    stride = (1, cute.sym_int(64), 0) if leading == 0 else (cute.sym_int(64), 1, 0)
    return make_fake_tensor(dtype, shape, stride, assumed_align=16)


@cache
def compile_ring(tactic, block_size: int, arch: str, fenced_trigger: int = 0, early_coef: bool = False):
    kernel = RingConvSplitKGemm(
        tactic=SplitKTactic(*tactic), block_size=block_size, fenced_trigger=fenced_trigger, early_coef=early_coef
    )
    y = _fake(0)
    args = (_fake(1), _fake(1), y, y, Int64(0), Int64(0), Int64(0), Int64(0), make_fake_stream())
    compiled = _DSL_COMPILE(kernel, *args, options=f"--gpu-arch={arch}")
    overflow = smem_overflow(compiled)
    if overflow is not None:  # SMEM overflow check, as compile_projection
        raise ValueError(f"ring tactic {tuple(tactic)} uses more shared memory than its launch allocates ({overflow} B)")
    return compiled


class DraftRingProjection:
    """The ring form of one draft row-parallel weight [HIDDEN, K] for 1..ROWS rows at the served split-K ``tactic``."""

    def __init__(
        self, weight: torch.Tensor, tactic, block_size: int, trigger_after_wait: bool = False,
        early_coef: Optional[bool] = None,
    ):  # fmt: skip
        from sglang.srt.environ import envs

        n, k = weight.shape
        if n != HIDDEN or weight.dtype != torch.bfloat16 or tactic[2] != 1:
            raise ValueError(f"ring projection: weight {tuple(weight.shape)} {weight.dtype}, tactic {tactic}")
        self.n, self.k, self.tactic = n, k, tuple(tactic)
        self.trigger_after_wait = bool(trigger_after_wait)  # the deferred sites' producer form
        # SGLANG_OPT_DFLASH_DRAFT_RING_EARLY_COEF: the conv coefficients load during the main loop (same bits).
        self.early_coef = envs.SGLANG_OPT_DFLASH_DRAFT_RING_EARLY_COEF.get() if early_coef is None else bool(early_coef)
        # Only the weight checks, the weight view and its TMA map: the kernel is defined here.
        self._prepared = PreparedProjection(weight, {ROWS: self.tactic}, proxy_fences=True)
        self._extra = self._prepared.weight_map(self.tactic[0])
        with torch.accelerator.device_index(weight.device.index):
            self._code = compile_ring(
                self.tactic, block_size, architecture(weight.device), int(self.trigger_after_wait), self.early_coef
            )

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

    def run(self, x: torch.Tensor, projection: torch.Tensor, base: torch.Tensor, ring: int) -> torch.Tensor:
        """Publish conv side 1 of x @ weight.T into the slot at ``ring``; returns an unwritten [rows, HIDDEN] tensor
        (it only shapes the grid)."""
        rows = x.shape[0]
        if projection.shape != (rows, 2 * TAPS * GROUPS) or not projection.is_contiguous():
            raise ValueError(f"ring projection: conv projection {tuple(projection.shape)}")
        if base.shape != (TAPS, HIDDEN) or not base.is_contiguous() or base.dtype != torch.bfloat16:
            raise ValueError(f"ring projection: base kernel {tuple(base.shape)} {base.dtype}")
        out = x.new_empty((rows, self.n))
        y = matrix_view(out, output=True)
        self._code(
            self._prepared.weight_view, matrix_view(x), y, y, *self._extra, Int64(projection.data_ptr()),
            Int64(base.data_ptr()), Int64(ring), stream(x.device),
        )  # fmt: skip
        return out


class DraftRingLinearMethod(DraftSplitKLinearMethod):
    """The draft o_proj (``parity`` 0) or down_proj (1) on its split-K projection: inside the sites' ring scope of the
    same parity, covered rows go through the ring projection instead (and the site is told to poll)."""

    def __init__(self, inner: DraftSplitKLinearMethod, ring: DraftRingProjection, sites, parity: int):
        super().__init__(inner.projection)
        self.ring, self.sites, self.parity = ring, sites, parity

    def apply(self, layer, x, bias=None):
        scope = self.sites.ring_pending
        if bias is None and scope is not None and scope[0] == self.parity and self.ring.covers(x):
            _, projection, conv = scope
            base = conv.base_kernel.detach()[1]
            out = self.ring.run(x, projection, base, self.sites.ring_address(self.parity))
            self.sites.ring_published(self.parity, x.shape[0], self.ring.trigger_after_wait)
            return out
        return super().apply(layer, x, bias)


def install_draft_ar_ring(draft_model, sites) -> int:
    """SGLANG_OPT_DFLASH_DRAFT_AR_RING: route each layer's o_proj / down_proj through its ring projection inside the
    sites' ring scopes; the projections keep the served split-K tactic of their shape (same tiles and sum)."""
    from sglang.srt.speculative.dflash_draft_splitk import SPLITK_TACTICS

    installed, early_coef = 0, None
    for layer in draft_model.layers:
        for linear, parity, conv in (
            (layer.self_attn.o_proj, 0, layer.attention_conv),
            (layer.mlp.down_proj, 1, layer.mlp_conv),
        ):
            if not isinstance(linear.quant_method, DraftSplitKLinearMethod):
                raise ValueError("SGLANG_OPT_DFLASH_DRAFT_AR_RING needs SGLANG_OPT_DFLASH_DRAFT_SPLITK (same tiles and sum)")
            tactic = SPLITK_TACTICS.get(tuple(linear.weight.shape))
            if tactic is None or tactic[2] != 1 or tactic[1] % conv.block_size:
                raise ValueError(
                    f"SGLANG_OPT_DFLASH_DRAFT_AR_RING: no split-1 tactic for {tuple(linear.weight.shape)} "
                    f"with block {conv.block_size}"
                )
            ring = DraftRingProjection(linear.weight.detach(), tactic, conv.block_size, trigger_after_wait=sites.deferred)
            linear.quant_method = DraftRingLinearMethod(linear.quant_method, ring, sites, parity)
            installed, early_coef = installed + 1, ring.early_coef
    sites.rings = True
    logger.info("DFLASH draft ring projections on %d linears (early coefficients %s)", installed, early_coef)
    return installed
