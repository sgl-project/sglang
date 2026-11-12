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
# Modified by NVIDIA in 2026: the act-fused gate_up on SK3B.
# The parent kernel SplitKTmemGemm64 is the megakernel plugin's gemm/splitk_tmem64.py (NVIDIA, Apache-2.0), based on earlier
# NVIDIA work.
# SGLANG_OPT_DFLASH_DRAFT_SPLITK_SK3B_ACT: the ACT-FUSED gate_up on SK3B (speculative/dflash_sk3b.py) through its hooks
# issue_a / owner_of / finish.
# WHAT: the drafter MLP's merged gate_up weight [2I, 7168] (gate rows 0..I-1, up rows I..2I-1, I = 896 at TP16) and C = SiLU(gate) * up [rows, I]
#   in ONE kernel, removing the served act_and_mul kernel at 96-128 rows.
#   (1) issue_a: CTA bidx's 128-row A tile = gate rows [64 bidx, 64 bidx + 64) in tile rows 0..63 and up rows [I + 64 bidx, ...) in 64..127, as two
#       {64, 64} boxes of a 64-row-box weight map (the plugin SplitKGemm act_fuse form: 64 rows x 128 B = 8 swizzle atoms, SW128 phase kept);
#       same 16 KB per stage, same expect_tx.
#   (2) owner_of: owner k of share s takes gate features [k s/2, (k+1) s/2) of the tile and the up rows of the SAME features (positions s/2..s-1),
#       so gate and up of one feature meet in one owner. A bijection of rows -> (owner, position): the reduce-scatter sums are untouched.
#   (3) finish: the owner's FP32 rank-order totals are rounded to BF16 exactly as the parent's finish (cvt.rn.bf16.f32 = the stock gate_up output),
#       staged in its tile [N tokens][share], then per (token, 8 features): gate / up BF16 -> FP32, the plugin's _silu_mul_fast (the served JIT
#       act_and_mul<bf16, kSiLU> --use_fast_math sequence), cvt.rn.bf16x2, one 16-byte store per 8 features of C [rows, I].
#   => bitwise to (the same GEMM form's output + the served act) BY CONSTRUCTION if _silu_mul_fast == the served act bitwise; checked on GPU.
# SGLANG_OPT_DFLASH_DRAFT_SPLITK_SK3B_PF: a policy with flags 512 also prefetches the next GEMM's weight (the MLP's down
#   projection) into L2 after the drain (dflash_sk3b.SplitKStream3b.prefetch_next); run(..., prefetch=weight). A cache hint: outputs unchanged.
# Sync: no new edge. The finish's tile / named barrier 3 (128 epilogue threads) are the parent's; the act reads only this CTA's staged tile after
#   that barrier (bar.sync orders shared memory among the participating threads, PTX 9.7.13.1), as in the parent.
from functools import cache

import cuda.bindings.driver as _cuda
import cutlass
import cutlass.cute as cute
import torch
from cutlass.cute.runtime import make_fake_stream, make_fake_tensor

from sglang.srt.speculative import dflash_sk3b as sk  # noqa: E402
from sgl_nvidia_mega.kimi_k3.cutedsl.common.host import architecture  # noqa: E402
from sgl_nvidia_mega.kimi_k3.cutedsl.common.host import stream as current_stream  # noqa: E402
from sgl_nvidia_mega.kimi_k3.cutedsl.common.primitives import at, bf16x2, named_barrier, store_words, unpack_bf16  # noqa: E402
from sgl_nvidia_mega.kimi_k3.cutedsl.gemm import splitk_tmem64 as tm  # noqa: E402
from sgl_nvidia_mega.kimi_k3.cutedsl.gemm.projection import _DSL_COMPILE, matrix_view, set_oversized, smem_overflow, weight_map  # noqa: E402
from sgl_nvidia_mega.kimi_k3.cutedsl.gemm.splitk import SplitKTactic, _silu_mul_fast  # noqa: E402


class ActFinish:
    """Mixin, placed BEFORE a TM64-family kernel class in the MRO. A = the merged gate_up weight [2 act_i, K] through a 64-row-box map;
    C = bf16(silu(gate) * up) [rows, act_i] (M-major view (act_i, rows, 1))."""

    def act_init(self, act_i):
        assert self.cta_m == 128 and self.share % 16 == 0, "128-row tiles; share / 2 features per owner in 16-byte rows"
        assert act_i % 64 == 0
        self.act_i = int(act_i)

    @cute.experimental.jit
    def __call__(
        self, a: cute.Tensor, b: cute.Tensor, c: cute.Tensor, bias: cute.Tensor,
        a_map: cutlass.Int64, stream: _cuda.CUstream,
    ):  # fmt: skip
        # grid x from A's rows (2 act_i / 128 tiles), not C's (act_i)
        self.kernel(a, b, c, bias, a_map).launch(
            grid=(
                cute.ceil_div(a.layout.shape[0], self.cta_m),
                cute.ceil_div(c.layout.shape[1], self.cta_n) * self.split_k,
                c.layout.shape[2],
            ),
            block=(self.threads_per_cta, 1, 1),
            cluster=self.cluster_shape,
            smem=cute.Int64(self.smem_bytes),
            stream=stream,
            use_pdl=True,
        )

    def issue_a(self, dst, barrier, a_map, k0, row0):
        """Gate rows [row0 / 2, + 64) into tile rows 0..63, up rows [act_i + row0 / 2, + 64) into 64..127 (two {64, 64} boxes)."""
        half = row0 // 2
        tm._tma_box(dst, barrier, a_map, k0, half)
        tm._tma_box(dst + 64 * 128, barrier, a_map, k0, half + self.act_i)

    def owner_of(self, row, row0):
        hs = self.share // 2
        j = row % 64
        return j // hs, (row // 64) * hs + j % hs

    @cute.experimental.jit
    def finish(
        self, epi_tid: cutlass.Int32, totals: cute.Tensor, tile: cutlass.Int32, out,
        row0: cutlass.Int32, col0: cutlass.Int32, l_idx: cutlass.Int32,
    ):  # fmt: skip
        """Owner rows [row0, row0 + share) = gate features f0 .. f0 + share / 2 - 1 (positions 0..) and their up rows (positions share / 2 ..),
        f0 = row0 / 2. BF16 staging as the parent's finish, then 8 features per (thread, item): silu-mul, one 16-byte store of C."""
        mC = out[0]
        share: cutlass.Constexpr = self.share
        hs: cutlass.Constexpr = self.share // 2
        rr, g = epi_tid % share, epi_tid // share
        width: cutlass.Constexpr = self.n * share // 128
        for t in cutlass.range_constexpr(width):
            tm._st_shared_bf16(tile + ((width * g + t) * share + rr) * 2, totals[t])
        named_barrier(3, 128)
        feat0 = row0 // 2
        columns = mC.layout.shape[1]
        chunks: cutlass.Constexpr = hs // 8
        items: cutlass.Constexpr = self.n * chunks
        for i in cutlass.range_constexpr((items + 127) // 128):
            idx = epi_tid + i * 128
            token = idx // chunks
            c = idx % chunks
            if (idx < items) & (col0 + token < columns):
                gw = tm._ld_shared_v4(tile + (token * share + 8 * c) * 2)
                uw = tm._ld_shared_v4(tile + (token * share + hs + 8 * c) * 2)
                words = []
                for q in cutlass.range_constexpr(4):
                    g_lo, g_hi = unpack_bf16(gw[q])
                    u_lo, u_hi = unpack_bf16(uw[q])
                    words.append(bf16x2(_silu_mul_fast(g_lo, u_lo), _silu_mul_fast(g_hi, u_hi)))
                store_words(at(mC, feat0 + 8 * c, col0 + token, l_idx), tuple(words))


class SK3BAct(ActFinish, sk.SplitKStream3b):
    def __init__(self, *, act_i, **kw):
        sk.SplitKStream3b.__init__(self, **kw)
        self.act_init(act_i)


@cache
def compile_sk3b_act(n, split, slots, wstages, owners, k, act_i, arch, flags=0):
    def fake(leading, dtype=cutlass.BFloat16):
        shape = tuple(cute.sym_int() for _ in range(3))
        stride = (1, cute.sym_int(64), 0) if leading == 0 else (cute.sym_int(64), 1, 0)
        return make_fake_tensor(dtype, shape, stride, assumed_align=16)

    y = fake(0)
    kw = dict(xlsu=bool(flags & 1), bulk=bool(flags & 2), ov=bool(flags & 4))
    if flags & 32:
        kw["dmb"] = True  # SK3B versions with the dedicated-mailbox reduce only
    if flags & 512:
        kw["pf"] = True  # SK3B_PF: next-weight L2 prefetch; the next weight's K-major view rides the bias slot
    kernel = SK3BAct(tactic=SplitKTactic(128, n, split, slots), k=k, wstages=wstages, owners=owners, act_i=act_i, **kw)
    args = (fake(1), fake(1), y, fake(1) if flags & 512 else y, cutlass.Int64(0), make_fake_stream())
    compiled = _DSL_COMPILE(kernel, *args, options=f"--gpu-arch={arch}")
    overflow = smem_overflow(compiled)
    if overflow is not None:
        raise ValueError(f"SK3B-act {(128, n, split, slots, wstages)} owners {owners} uses more SMEM than its launch allocates ({overflow} B)")
    return compiled, kernel.smem_bytes


class SK3BActProjection:
    """weight [2I, K] BF16 (gate rows then up rows); ``policies`` {rows: (N tile, split, x slots, wstages, owners[, flags])} as SK3BProjection;
    run(x [rows, K], out [rows, I]) -> bf16(silu(x W_gate^T) * (x W_up^T)) with SK3B's GEMM arithmetic."""

    def __init__(self, weight, policies):
        assert weight.dim() == 2 and weight.dtype == torch.bfloat16 and weight.stride(1) == 1 and weight.shape[0] % 128 == 0
        self.weight, self.device, self.policies = weight, weight.device, dict(policies)
        self.act_i = weight.shape[0] // 2
        self.weight_view = matrix_view(weight)
        self.kernels, self.map = {}, None

    def prepare(self):
        arch = architecture(self.device)
        k = self.weight.shape[1]
        with torch.accelerator.device_index(self.device.index):
            for rows, pol in self.policies.items():
                n, split, slots, wstages, owners = pol[:5]
                flags = int(pol[5]) if len(pol) > 5 else 0
                code, smem = compile_sk3b_act(n, split, slots, wstages, owners, k, self.act_i, arch, flags)
                sk.allow_nonportable(code, self.device)
                if smem > sk.NORMAL_SMEM:
                    set_oversized(code, self.device)
                self.kernels[rows] = code
            self.map = weight_map(self.weight, 64)
        return self

    def run(self, x, out, prefetch=None):
        b = x.shape[0]
        if b not in self.kernels or x.shape[1] != self.weight.shape[1] or out.shape != (b, self.act_i) or out.stride(1) != 1:
            raise ValueError(f"SK3B-act input {tuple(x.shape)} out {tuple(out.shape)}: prepared {sorted(self.kernels)}")
        y = matrix_view(out, output=True)
        bias = matrix_view(sk.prefetch_target(prefetch)) if sk.pf_policy(self.policies[b]) else y
        self.kernels[b](self.weight_view, matrix_view(x), y, bias, cutlass.Int64(self.map.data_ptr()), current_stream(self.device))
        return out
