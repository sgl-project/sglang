# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the SGLang project
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
"""SGLANG_OPT_DFLASH_DRAFT_PDL_GLUE: launched with PDL, the draft layer's Triton glue kernels (grouped conv, MLA fused KV
write, split-KV attention) read their inputs only after griddepcontrol.wait. Each runs behind a predecessor that releases
its dependents at once and writes their inputs DELAY_NS later, so a read issued before the wait sees the poisoned buffer;
the PDL launch must equal the plain launch bit for bit. A control kernel that reads before its wait must see the poison,
or the setup does not overlap the two kernels and the comparison would be blind. Every kernel is compiled before its race:
a first call's JIT compile delays the launch past the predecessor's write and hides the race."""

import unittest

import torch
import triton
import triton.language as tl

from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=60, stage="base-b", runner_config="1-gpu-large")

DEV = "cuda"
DELAY_NS = 50_000
RACES = 3
COPY_BLOCK = 4096
POISON = 0xFF  # every bf16 / fp8 byte pattern 0xFF.. is a NaN


@triton.jit
def _late_copy_kernel(src, dst, n, DELAY_NS: tl.constexpr, BLOCK: tl.constexpr):
    tl.extra.cuda.gdc_launch_dependents()
    t0 = tl.extra.cuda.globaltimer()
    while tl.extra.cuda.globaltimer() - t0 < DELAY_NS:
        pass
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    tl.store(dst + offs, tl.load(src + offs, mask=offs < n), mask=offs < n)


@triton.jit
def _early_read_kernel(src, dst, n, BLOCK: tl.constexpr):
    # Control: reads before its wait (the bug the glue kernels must not have).
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    x = tl.load(src + offs, mask=offs < n)
    tl.extra.cuda.gdc_wait()
    tl.store(dst + offs, x, mask=offs < n)


def _bytes(t):
    return t.view(torch.uint8).reshape(-1)


def _late_write(pairs):
    """One plain-launched predecessor per (src, dst), in order: each releases its dependents, then copies DELAY_NS
    later. Only the last pair is still unwritten when the next kernel starts, so pass the most-read input last."""
    for src, dst in pairs:
        dst.view(torch.uint8).fill_(POISON)
    for src, dst in pairs:
        s, d = _bytes(src), _bytes(dst)
        _late_copy_kernel[(triton.cdiv(s.numel(), COPY_BLOCK),)](s, d, s.numel(), DELAY_NS=DELAY_NS, BLOCK=COPY_BLOCK)


def _same(a, b):
    return torch.equal(_bytes(a.contiguous()), _bytes(b.contiguous()))


class TestDFlashDraftPDLGlue(unittest.TestCase):
    def setUp(self):
        self.gen = torch.Generator(device=DEV).manual_seed(7)

    def _randn(self, *shape, scale=1.0):
        return (torch.randn(*shape, generator=self.gen, device=DEV) * scale).bfloat16()

    def test_control_early_read_sees_poison(self):
        src = self._randn(8, 7168)
        dst, out = torch.empty_like(src), torch.empty_like(src)
        n = _bytes(dst).numel()

        def early_read():
            _early_read_kernel[(triton.cdiv(n, COPY_BLOCK),)](_bytes(dst), _bytes(out), n, BLOCK=COPY_BLOCK, launch_pdl=True)

        _late_write([(src, dst)])
        early_read()  # compiles both kernels
        torch.cuda.synchronize()
        for _ in range(RACES):
            _late_write([(src, dst)])
            early_read()
            torch.cuda.synchronize()
            self.assertFalse(_same(out, src), "the early read saw the written input: PDL did not overlap the kernels")

    def test_grouped_conv(self):
        from sglang.kernels.ops.speculative.dflash import dflash_grouped_conv

        for rows, block in ((8, 8), (16, 8), (3, 8)):
            hidden, delta = self._randn(rows, 7168), self._randn(rows, 2, 448, scale=0.1)
            base = self._randn(2, 7168, scale=0.3) + 1.0
            ref = dflash_grouped_conv(hidden, delta, base, block, 16, 2)
            dflash_grouped_conv(hidden, delta, base, block, 16, 2, pdl=True)  # compile
            h_late, d_late = torch.empty_like(hidden), torch.empty_like(delta)
            for race in range(RACES):
                _late_write([(delta, d_late), (hidden, h_late)])
                out = dflash_grouped_conv(h_late, d_late, base, block, 16, 2, pdl=True)
                torch.cuda.synchronize()
                self.assertTrue(_same(out, ref), f"rows {rows} race {race}")

    def test_mla_fused_kv_write(self):
        from sglang.kernels.ops.kvcache.mla_fused_kv_write import mla_concat_qk_store_fp8_kv

        t, h = 8, 6
        q_nope = self._randn(h, t, 512).transpose(0, 1)
        q_pe, k_nope, k_pe = self._randn(t, h, 64), self._randn(t, 1, 512), self._randn(t, 1, 64)
        loc = torch.randperm(4096, generator=self.gen, device=DEV)[:t]
        pos = torch.arange(9000, 9000 + t, device=DEV)
        inv = 1.0 / (50000.0 ** (torch.arange(0, 64, 2, device=DEV, dtype=torch.float32) / 64))
        fr = torch.arange(16384, device=DEV, dtype=torch.float32)[:, None] * inv[None, :]
        cos_sin = torch.cat([fr.cos(), fr.sin()], dim=-1).contiguous()
        rows_ref = torch.zeros(4096, 1, 576, dtype=torch.uint8, device=DEV)
        rows_pdl = rows_ref.clone()
        q_ref, k_ref = mla_concat_qk_store_fp8_kv(q_nope=q_nope, q_pe=q_pe, k_nope=k_nope, k_pe=k_pe, kv_rows=rows_ref,
                                                  loc=loc, positions=pos, cos_sin_cache=cos_sin)
        mla_concat_qk_store_fp8_kv(q_nope=q_nope, q_pe=q_pe, k_nope=k_nope, k_pe=k_pe, kv_rows=rows_pdl, loc=loc,
                                   positions=pos, cos_sin_cache=cos_sin, pdl=True)  # compile
        late = [torch.empty_like(x) for x in (q_pe, k_nope, k_pe)]
        for race in range(RACES):
            rows_pdl.zero_()
            _late_write([(k_nope, late[1]), (k_pe, late[2]), (q_pe, late[0])])
            q, k = mla_concat_qk_store_fp8_kv(q_nope=q_nope, q_pe=late[0], k_nope=late[1], k_pe=late[2], kv_rows=rows_pdl,
                                              loc=loc, positions=pos, cos_sin_cache=cos_sin, pdl=True)
            torch.cuda.synchronize()
            self.assertTrue(_same(q, q_ref) and _same(k, k_ref) and _same(rows_pdl, rows_ref), f"race {race}")

    def test_draft_splitkv_attention(self):
        from sglang.kernels.ops.attention.dflash_draft_splitkv import dflash_draft_splitkv_attention_fwd

        rows, heads, rail, slots = 8, 6, 1056, 1 << 14
        q, k_ext = self._randn(rows, heads, 576, scale=0.5), self._randn(rows, 1, 576, scale=0.5)
        v_ext = k_ext[..., :512].contiguous()
        kb = (torch.randn(slots, 1, 576, generator=self.gen, device=DEV) * 0.5).to(torch.float8_e4m3fn)
        idx = torch.randperm(slots, generator=self.gen, device=DEV)[:rail]
        qo = torch.tensor([0, rows], dtype=torch.int32, device=DEV)
        kv = torch.tensor([0, rail], dtype=torch.int32, device=DEV)

        def run(qq, kk, vv, **pdl):
            o = torch.empty(rows, heads, 512, dtype=torch.bfloat16, device=DEV)
            dflash_draft_splitkv_attention_fwd(qq, kk, vv, o, kb, kb[..., :512], qo, kv, idx, rows, 1.0, 1.0, 192 ** -0.5,
                                               max_kv_len=1087, sliding_window_size=4095, **pdl)
            return o

        ref = run(q, k_ext, v_ext)
        run(q, k_ext, v_ext, pdl=True, pdl_wait_first=True)  # compile
        late = [torch.empty_like(x) for x in (q, k_ext, v_ext)]
        for race in range(RACES):
            _late_write([(k_ext, late[1]), (v_ext, late[2]), (q, late[0])])
            out = run(*late, pdl=True, pdl_wait_first=True)
            torch.cuda.synchronize()
            self.assertTrue(_same(out, ref), f"race {race}")
        with self.assertRaises(ValueError):
            run(q, k_ext, v_ext, pdl_wait_first=True)


if __name__ == "__main__":
    unittest.main()
