"""CuTeDSL port of the MoE LoRA-A "shrink" grouped GEMV.

    out[t, n] = sum_k hidden[t // top_k, k] * lora_a[expert(t), n, k]

Per expert token-block (BM=16 routed tokens), gather the 16 hidden rows + the
expert's [N, K] LoRA-A weight, stream K in BK tiles through a cp.async pipeline,
and run a warp m16n8k16 MMA (tokens in MMA-M, rank in MMA-N). Output rows are
scattered back to the routed positions.

Occupancy / latency hiding (this op is single-block latency-bound: the grid is
~68 m-blocks on a 148-SM GPU, so half the machine is idle and per-block latency
dominates). The output tile is only 16xrank, too small to tile M or N across more
than `rank/8` warps -- so we tile the MMA over K instead: the TiledMma atom layout
(1, n_tiles, k_split) puts `k_split` warps on the K mode, each reducing a distinct
K-slice (partition_A/B hand each warp its own slice automatically). Their partial
accumulators are summed once in shared memory in the epilogue. This is the CuTe
analogue of the CUDA pipe kernel's "8 warps, K split across warps" design (the
single biggest lever there): without it the 2-warp kernel is ~2x slower.

The token-row gather (sorted_token_ids // top_k) and the scatter epilogue are
hand-written -- they do not map to cute's contiguous tiled-copy model -- while
the inner GEMM uses cute's TiledMma + ldmatrix. CUDA reference:
csrc/lora/moe_lora_shrink_kernel.cu (moe_lora_shrink_pipe_kernel).
"""

import math

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import cutlass.utils as utils
import torch
from cutlass.cute.runtime import from_dlpack

BM = 16
WMMA_M, WMMA_N, WMMA_K = 16, 8, 16


class MoeLoraShrinkCute:
    def __init__(
        self,
        rank: int,
        bk: int = 256,
        stages: int = 4,
        k_split: int = 4,
        ab_dtype=cutlass.BFloat16,
        use_pdl: bool = True,
    ):
        assert rank in (16, 32)
        self.use_pdl = use_pdl
        self.rank = rank
        self.bk = bk
        self.stages = stages
        self.ab_dtype = ab_dtype
        self.acc_dtype = cutlass.Float32
        self.n_tiles = rank // WMMA_N  # MMA-N atoms (2 for r16, 4 for r32)
        # Latency-bound, single-block: the grid (~68 m-blocks) is smaller than the
        # SM count, so occupancy comes from MORE WARPS PER BLOCK, not more blocks.
        # We can't tile M (16=one m16 tile) or N (rank tiny) further, so we tile the
        # MMA over K: `k_split` warps each reduce a distinct slice of K, and their
        # partial accumulators are summed once in smem. This mirrors the CUDA pipe
        # kernel's "8 warps, K split across warps" design (the single biggest lever
        # there: 9.5us -> 4.6us). bk must be divisible by k_split*WMMA_K.
        assert bk % (k_split * WMMA_K) == 0
        self.k_split = k_split
        self.warps = self.n_tiles * k_split
        self.threads = self.warps * 32

    def _smem_layout(self, rows):
        # row-major [rows, bk] with 128B XOR swizzle on the K-contiguous segments
        copy_bits = 128
        seg = min(self.bk, 64)
        swizzle_bits = min(int(math.log2(seg * self.ab_dtype.width // copy_bits)), 3)
        atom = cute.make_composed_layout(
            cute.make_swizzle(swizzle_bits, 3, 3),
            0,
            cute.make_layout((8, seg), stride=(seg, 1)),
        )
        return cute.tile_to_shape(atom, (rows, self.bk, self.stages), (0, 1, 2))

    @cute.jit
    def __call__(
        self,
        mHidden: cute.Tensor,  # [num_tokens, K] bf16
        mLoraA: cute.Tensor,  # [E, N, K] bf16
        mOut: cute.Tensor,  # [num_valid, N] bf16
        mTok: cute.Tensor,  # [num_m_blocks * BM] int32
        mExpert: cute.Tensor,  # [num_m_blocks] int32
        mNpp: cute.Tensor,  # [1] int32
        stream: cuda.CUstream,
        top_k: cutlass.Constexpr,
        num_valid: cutlass.Int32,
        num_m_blocks: cutlass.Int32,
    ):
        K = mHidden.shape[1]

        op = cute.nvgpu.warp.MmaF16BF16Op(
            self.ab_dtype, self.acc_dtype, (WMMA_M, WMMA_N, WMMA_K)
        )
        # (M=1, N=n_tiles, K=k_split) atoms -> n_tiles*k_split warps. The K-mode
        # warps each accumulate a distinct K-slice (partition_A/B hand each its own
        # slice); their partials are reduced in smem in the epilogue.
        tiled_mma = cute.make_tiled_mma(op, (1, self.n_tiles, self.k_split))

        sA_layout = self._smem_layout(BM)
        sB_layout = self._smem_layout(self.rank)

        self.kernel(
            mHidden,
            mLoraA,
            mOut,
            mTok,
            mExpert,
            mNpp,
            tiled_mma,
            sA_layout,
            sB_layout,
            top_k,
            num_valid,
            K,
        ).launch(
            grid=[num_m_blocks, 1, 1],
            block=[self.threads, 1, 1],
            stream=stream,
            use_pdl=self.use_pdl,
        )

    @cute.kernel
    def kernel(
        self,
        mHidden: cute.Tensor,
        mLoraA: cute.Tensor,
        mOut: cute.Tensor,
        mTok: cute.Tensor,
        mExpert: cute.Tensor,
        mNpp: cute.Tensor,
        tiled_mma: cute.TiledMma,
        sA_layout: cute.ComposedLayout,
        sB_layout: cute.ComposedLayout,
        top_k: cutlass.Constexpr,
        num_valid: cutlass.Int32,
        K: cutlass.Constexpr,
    ):
        bidx, _, _ = cute.arch.block_idx()
        tidx, _, _ = cute.arch.thread_idx()

        # ---- shared storage (allocate unconditionally; cheap reservation) ----
        sC_layout = cute.make_layout((BM, self.rank))
        # Per-thread partial-acc reduction scratch: [threads, FRAG] fp32 (FRAG=4
        # for the m16n8 atom). The k_split warps sharing an (n,lane) sum here.
        sRed_layout = cute.make_layout((self.threads, 4))

        @cute.struct
        class Smem:
            a: cute.struct.Align[
                cute.struct.MemRange[self.ab_dtype, cute.cosize(sA_layout)], 16
            ]
            b: cute.struct.Align[
                cute.struct.MemRange[self.ab_dtype, cute.cosize(sB_layout)], 16
            ]
            c: cute.struct.Align[
                cute.struct.MemRange[self.acc_dtype, cute.cosize(sC_layout)], 16
            ]
            red: cute.struct.Align[
                cute.struct.MemRange[self.acc_dtype, cute.cosize(sRed_layout)], 16
            ]

        smem = utils.SmemAllocator()
        storage = smem.allocate(Smem.size_in_bytes(), byte_alignment=16)
        sA = Smem(storage).a.get_tensor(sA_layout)  # [BM, bk, stages]
        sB = Smem(storage).b.get_tensor(sB_layout)  # [rank, bk, stages]
        sC = Smem(storage).c.get_tensor(sC_layout)  # [BM, rank] fp32
        sRed = Smem(storage).red.get_tensor(sRed_layout)  # [threads, 4] fp32

        # PDL: this block's loads (hidden/lora_a/routing) don't depend on the prior
        # grid's *output*, so the wait only honors stream ordering; the win is letting
        # the next launch's grid setup overlap this grid's tail (released after the
        # K-loop via griddepcontrol_launch_dependents). All blocks must reach both.
        if self.use_pdl:
            cute.arch.griddepcontrol_wait()

        # Block-uniform early-out folded into a guard (no `return` in @cute.kernel).
        expert = mExpert[bidx]
        active = (bidx * BM < mNpp[0]) and (expert != cutlass.Int32(-1))
        if active:
            self._compute(
                mHidden,
                mLoraA,
                mOut,
                mTok,
                mNpp,
                tiled_mma,
                sA,
                sB,
                sC,
                sRed,
                top_k,
                num_valid,
                K,
                bidx,
                tidx,
                expert,
            )
        else:
            if self.use_pdl:
                cute.arch.griddepcontrol_launch_dependents()

    @cute.jit
    def _load_tile(self, g2s, rows, mHidden, gB, sA, sB, tidx, kt, stage):
        # Cheap warp/lane gather (mirrors the CUDA pipe kernel): warp g owns a
        # strided set of rows, lane owns a 128-bit K-vector. No per-element
        # divide/modulo or runtime view rebuild -- those integer ops dominated the
        # naive `cc//kvecs` gather (the kernel was issue-bound on them, so deeper
        # stages / bigger BK gave 0 improvement).
        kvecs = self.bk // 8  # 128-bit (8 bf16) vectors per row per tile
        kbase = kt * kvecs
        g = tidx // 32
        lane = tidx % 32
        nw = self.warps
        kv_iters = (kvecs + 31) // 32
        # A rows (gathered hidden rows; rows[m] < 0 -> padding, skip)
        a_iters = (BM + nw - 1) // nw
        for j in cutlass.range(a_iters, unroll_full=True):
            m = g + j * nw
            if m < BM:
                rr = rows[m]
                if rr >= 0:
                    for kj in cutlass.range(kv_iters, unroll_full=True):
                        kv = lane + kj * 32
                        if kv < kvecs:
                            src = cute.local_tile(
                                mHidden[rr, None], (8,), (kbase + kv,)
                            )
                            dst = cute.local_tile(sA[m, None, stage], (8,), (kv,))
                            cute.copy(g2s, src, dst)
        # B rows (contiguous for this expert)
        b_iters = (self.rank + nw - 1) // nw
        for j in cutlass.range(b_iters, unroll_full=True):
            n = g + j * nw
            if n < self.rank:
                for kj in cutlass.range(kv_iters, unroll_full=True):
                    kv = lane + kj * 32
                    if kv < kvecs:
                        src = cute.local_tile(gB[n, None], (8,), (kbase + kv,))
                        dst = cute.local_tile(sB[n, None, stage], (8,), (kv,))
                        cute.copy(g2s, src, dst)
        cute.arch.cp_async_commit_group()

    @cute.jit
    def _compute(
        self,
        mHidden,
        mLoraA,
        mOut,
        mTok,
        mNpp,
        tiled_mma,
        sA,
        sB,
        sC,
        sRed,
        top_k,
        num_valid,
        K,
        bidx,
        tidx,
        expert,
    ):
        # per-row gathered hidden-row index (-1 = padding)
        rows = cute.make_rmem_tensor(cute.make_layout(BM), cutlass.Int32)
        for m in cutlass.range(BM, unroll_full=True):
            t = mTok[bidx * BM + m]
            rows[m] = (t // top_k) if t < num_valid else cutlass.Int32(-1)

        gB = mLoraA[expert, None, None]  # [N, K]

        num_k_tiles = K // self.bk
        kvecs = self.bk // 8  # 128-bit (8 bf16) vectors per row per tile

        # cp.async copy atom (128-bit)
        g2s = cute.make_copy_atom(
            cute.nvgpu.cpasync.CopyG2SOp(
                cache_mode=cute.nvgpu.cpasync.LoadCacheMode.GLOBAL
            ),
            self.ab_dtype,
            num_bits_per_copy=128,
        )

        # prologue
        for s in cutlass.range(self.stages - 1, unroll_full=True):
            if s < num_k_tiles:
                self._load_tile(g2s, rows, mHidden, gB, sA, sB, tidx, s, s)

        thr_mma = tiled_mma.get_slice(tidx)
        tCsC = thr_mma.partition_C(sC)
        tCrC = tiled_mma.make_fragment_C(tCsC)
        tCrC.fill(0.0)

        ldA = cute.make_copy_atom(
            cute.nvgpu.warp.LdMatrix8x8x16bOp(False, 4), self.ab_dtype
        )
        ldB = cute.make_copy_atom(
            cute.nvgpu.warp.LdMatrix8x8x16bOp(False, 4), self.ab_dtype
        )
        tc_A = cute.make_tiled_copy_A(ldA, tiled_mma)
        tc_B = cute.make_tiled_copy_B(ldB, tiled_mma)
        thr_A = tc_A.get_slice(tidx)
        thr_B = tc_B.get_slice(tidx)

        read = 0
        write = self.stages - 1
        for kt in range(num_k_tiles):
            cute.arch.cp_async_wait_group(self.stages - 2)
            cute.arch.sync_threads()

            # Issue the next K-tile's loads BEFORE the MMA so cp.async overlaps the
            # tensor cores. The target buffer (`write`) is stages-1 ahead of `read`,
            # so it isn't the tile we're about to consume -> no second barrier needed
            # (the pipeline depth + cp_async_wait gate the WAR hazard on reuse).
            ld = kt + self.stages - 1
            if ld < num_k_tiles:
                self._load_tile(g2s, rows, mHidden, gB, sA, sB, tidx, ld, write)

            sA_r = sA[None, None, read]
            sB_r = sB[None, None, read]
            tCsA = thr_mma.partition_A(sA_r)
            tCsB = thr_mma.partition_B(sB_r)
            tCrA = tiled_mma.make_fragment_A(tCsA)
            tCrB = tiled_mma.make_fragment_B(tCsB)
            cute.copy(tc_A, thr_A.partition_S(sA_r), thr_A.retile(tCrA))
            cute.copy(tc_B, thr_B.partition_S(sB_r), thr_B.retile(tCrB))
            num_kb = cute.size(tCrA, mode=[2])
            for kb in cutlass.range(num_kb, unroll_full=True):
                cute.gemm(
                    tiled_mma, tCrC, tCrA[None, None, kb], tCrB[None, None, kb], tCrC
                )
            read = (read + 1) % self.stages
            write = (write + 1) % self.stages

        # K-loop done: release the next grid so its launch/preamble overlaps this
        # grid's reduction + store tail.
        if self.use_pdl:
            cute.arch.griddepcontrol_launch_dependents()

        # ---- epilogue ----
        cute.arch.cp_async_wait_group(0)
        cute.arch.sync_threads()

        # Cross-(K-warp) reduction. Each of the k_split warps holds a partial
        # accumulator over its K-slice; warps sharing an (n_tile, lane) own the
        # SAME logical C coords, so fragment element `i` is the same coord across
        # them and can be summed element-wise without knowing the coord mapping.
        # warp = n_idx + k_idx*n_tiles (atom layout (1,n_tiles,k_split), col-major).
        FRAG = 4
        warp = tidx // 32
        lane = tidx % 32
        n_idx = warp % self.n_tiles
        k_idx = warp // self.n_tiles
        for i in cutlass.range(FRAG, unroll_full=True):
            sRed[tidx, i] = tCrC[i]
        cute.arch.sync_threads()
        # k_idx==0 warps reduce their column of partials and write the m16n8 frag.
        if k_idx == 0:
            for i in cutlass.range(FRAG, unroll_full=True):
                acc = tCrC[i]
                for kk in cutlass.range(1, self.k_split, unroll_full=True):
                    acc = acc + sRed[(n_idx + kk * self.n_tiles) * 32 + lane, i]
                tCrC[i] = acc
            cute.autovec_copy(tCrC, tCsC)
        cute.arch.sync_threads()
        total = BM * self.rank
        for i in cutlass.range(total // self.threads + 1, unroll_full=True):
            ii = i * self.threads + tidx
            if ii < total:
                m = ii // self.rank
                n = ii % self.rank
                if rows[m] >= 0:
                    t = mTok[bidx * BM + m]
                    mOut[t, n] = sC[m, n].to(self.ab_dtype)


_CACHE = {}


def moe_lora_shrink_cute(
    out, hidden, lora_a, sorted_tok, expert_ids, npp, top_k, block_m=16
):
    assert block_m == BM
    rank = lora_a.shape[1]
    K = lora_a.shape[2]
    num_m_blocks = expert_ids.shape[0]
    num_valid = out.shape[0]
    stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)
    key = (rank, K, top_k)
    if key not in _CACHE:
        _CACHE[key] = cute.compile(
            MoeLoraShrinkCute(rank).__call__,
            from_dlpack(hidden, assumed_align=16),
            from_dlpack(lora_a, assumed_align=16),
            from_dlpack(out, assumed_align=16),
            from_dlpack(sorted_tok, assumed_align=4),
            from_dlpack(expert_ids, assumed_align=4),
            from_dlpack(npp, assumed_align=4),
            stream,
            top_k,
            cutlass.Int32(num_valid),
            cutlass.Int32(num_m_blocks),
        )
    _CACHE[key](
        from_dlpack(hidden, assumed_align=16),
        from_dlpack(lora_a, assumed_align=16),
        from_dlpack(out, assumed_align=16),
        from_dlpack(sorted_tok, assumed_align=4),
        from_dlpack(expert_ids, assumed_align=4),
        from_dlpack(npp, assumed_align=4),
        stream,
        cutlass.Int32(num_valid),
        cutlass.Int32(num_m_blocks),
    )
    return out


if __name__ == "__main__":
    import torch
    import triton

    from sglang.srt.layers.moe.moe_runner.triton_utils.moe_align_block_size import (
        moe_align_block_size,
    )
    from sglang.srt.lora.triton_ops.virtual_experts import (
        _fused_virtual_topk_ids,
        fused_sanitize_expert_ids,
    )

    E, TK, H, R = 64, 8, 2048, 16
    dev = "cuda"
    for bs in [1, 8, 16]:
        torch.manual_seed(0)
        topk = torch.stack([torch.randperm(E, device=dev)[:TK] for _ in range(bs)]).to(
            torch.int32
        )
        tlm = torch.zeros(bs, device=dev, dtype=torch.int32)
        hs = torch.randn(bs, H, device=dev, dtype=torch.bfloat16) * 0.1
        la = torch.randn(E, R, H, device=dev, dtype=torch.bfloat16) * 0.1
        vt, _, vne = _fused_virtual_topk_ids(topk, tlm, E, False, 1)
        sti, eid, npp = moe_align_block_size(vt, BM, vne)
        nt = topk.numel()
        tight = triton.cdiv(nt + min(nt, vne) * (BM - 1), BM) * BM
        sti = sti[:tight].contiguous()
        eid = fused_sanitize_expert_ids(eid[: tight // BM], vne)
        out = torch.zeros(bs * TK, R, device=dev, dtype=torch.bfloat16)
        moe_lora_shrink_cute(out, hs, la, sti, eid, npp, TK, BM)
        torch.cuda.synchronize()
        # reference
        ref = torch.zeros_like(out)
        for b in range(bs):
            for k in range(TK):
                t = b * TK + k
                e = topk[b, k].item()
                ref[t] = (hs[b].float() @ la[e].float().T).to(torch.bfloat16)
        diff = (out.float() - ref.float()).abs().max().item()
        print(f"bs={bs:3d}  max|Δ|={diff:.4e}  {'OK' if diff < 0.5 else 'MISMATCH'}")
