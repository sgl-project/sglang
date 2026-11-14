"""The DSV4.1 OPUS sparse prefill path on gfx950: the -1 padded lists to CSR conversion, the
two-source OPUS attention against an fp32 reference, and the -1 filled raw top-k buffers it reads."""

import unittest

import torch

from sglang.srt.utils import is_gfx95_supported, is_hip
from sglang.test.ci.ci_register import register_amd_ci
from sglang.test.test_utils import CustomTestCase

register_amd_ci(est_time=20, stage="stage-b", runner_config="1-gpu-small-amd-mi35x")

D = 512
SCALE = D**-0.5
# about 3x the measured bf16 error of the kernel (~3e-3 of the output's max)
TOL = 1e-2
# the fp8 kernel against a reference over its own fp8 operands, so what is left is P's
# fp8 rounding: about 2.5x the worst measured over three seeds (3.2e-2; bf16 is 4e-3)
FP8_TOL = 8e-2


def _opus_available() -> bool:
    if not (is_hip() and is_gfx95_supported()):
        return False
    try:
        from aiter.ops.pa_sparse_prefill_opus import (  # noqa: F401
            pa_sparse_prefill_opus,
        )
    except ImportError:
        return False
    return True


def _fp8_opus_available() -> bool:
    if not _opus_available():
        return False
    from sglang.kernels.ops.attention.dsv4.opus_sparse_prefill_hip import (
        opus_fp8_available,
    )

    return opus_fp8_available()


def _mx_quant(x):
    """MXFP8 1x32 along the last dim, E8M0 putting each block's max in (224, 448]."""
    blk = x.float().reshape(*x.shape[:-1], -1, 32)
    amax = blk.abs().amax(-1, keepdim=True)
    e = torch.ceil(torch.log2(amax.clamp(min=1e-30) / 448.0))
    e = torch.where(amax == 0, torch.zeros_like(e), e)
    s = torch.exp2(e)
    return ((blk / s).to(torch.float8_e4m3fn).float() * s).reshape(x.shape)


def _decode_q_nope(q_nope):
    """The 448 NoPE values a packed fp8 query row carries."""
    vals = q_nope[..., :448].float().reshape(*q_nope.shape[:-1], 14, 32)
    e = q_nope.view(torch.uint8)[..., 448:462].float() - 127.0
    return (vals * torch.exp2(e)[..., None]).reshape(*q_nope.shape[:-1], 448)


def _padded_lists(rows, width, num_src_rows, gen, *, max_len=None):
    """-1 padded lists [rows, width] with per-row scanned lengths: valid entries, -1 holes inside
    the scanned prefix, and stale entries past it."""
    device = "cuda"
    max_len = width if max_len is None else max_len
    lens = torch.randint(0, max_len + 1, (rows,), generator=gen, device=device)
    idx = torch.randint(0, num_src_rows, (rows, width), generator=gen, device=device)
    holes = torch.rand(rows, width, generator=gen, device=device) < 0.2
    idx = torch.where(holes, -1, idx)
    return idx.to(torch.int32), lens.to(torch.int32)


def _csr_reference(indices, lens, row_start, row_end):
    out, indptr = [], [0]
    for row, n in zip(indices.tolist(), lens.tolist()):
        kept = [i - row_start for i in row[:n] if row_start <= i < row_end]
        out.extend(kept)
        indptr.append(indptr[-1] + len(kept))
    return out, indptr


def _attention_reference(q, sources, sink):
    """fp32 attention with sink: every query row over the union of its rows in each
    (keys, values, csr) source."""
    T, H, _ = q.shape
    out = torch.zeros(T, H, D, dtype=torch.float32, device=q.device)
    for t in range(T):
        keys, values = [], []
        for k_src, v_src, (indices, indptr) in sources:
            rows = indices[indptr[t] : indptr[t + 1]].long()
            keys.append(k_src[rows].float())
            values.append(v_src[rows].float())
        k, v = torch.cat(keys), torch.cat(values)
        s = q[t].float() @ k.T * SCALE
        logits = torch.cat([s, sink[:, None].float()], dim=1)
        p = torch.softmax(logits, dim=1)[:, :-1]
        out[t] = p @ v
    return out


@unittest.skipUnless(_opus_available(), "gfx950 with aiter's OPUS sparse prefill")
class TestCombinedToCsr(CustomTestCase):
    def test_matches_reference(self):
        from sglang.kernels.ops.attention.dsv4.opus_sparse_prefill_hip import (
            combined_to_csr,
        )

        gen = torch.Generator(device="cuda").manual_seed(0)
        for rows, width, row_start, row_end in [(37, 128, 0, 500), (64, 640, 100, 400)]:
            indices, lens = _padded_lists(rows, width, 600, gen)
            csr, indptr = combined_to_csr(indices, lens, row_start, row_end)
            ref, ref_indptr = _csr_reference(indices, lens, row_start, row_end)
            self.assertEqual(indptr.tolist(), ref_indptr)
            self.assertEqual(csr[: ref_indptr[-1]].tolist(), ref)


@unittest.skipUnless(_opus_available(), "gfx950 with aiter's OPUS sparse prefill")
class TestOpusSparsePrefill(CustomTestCase):
    def _case(self, heads, two_sources, gen):
        from sglang.kernels.ops.attention.dsv4.opus_sparse_prefill_hip import (
            combined_to_csr,
            opus_sparse_prefill,
        )

        T, n_c, n_s = 48, 700, 300
        q = (torch.randn(T, heads, D, generator=gen, device="cuda") * 0.5).to(
            torch.bfloat16
        )
        sink = torch.randn(heads, generator=gen, device="cuda") * 0.5
        swa_kv = (torch.randn(n_s, D, generator=gen, device="cuda") * 0.5).to(
            torch.bfloat16
        )
        swa_idx, swa_lens = _padded_lists(T, 128, n_s, gen)
        # every query row sees at least one key, as in prefill (its own SWA row)
        swa_idx[:, 0] = torch.arange(T, device="cuda", dtype=torch.int32) % n_s
        swa_lens = swa_lens.clamp(min=1)
        swa_csr = combined_to_csr(swa_idx, swa_lens, 0, n_s)
        if not two_sources:
            out = opus_sparse_prefill(q, swa_kv, swa_csr, sink, SCALE)
            ref = _attention_reference(q, [(swa_kv, swa_kv, swa_csr)], sink)
        else:
            c_kv = (torch.randn(n_c, D, generator=gen, device="cuda") * 0.5).to(
                torch.bfloat16
            )
            c_idx, c_lens = _padded_lists(T, 512, n_c, gen)
            # a row with no compressed history, like a request's first tokens
            c_lens[0] = 0
            c_csr = combined_to_csr(c_idx, c_lens, 0, n_c)
            out = opus_sparse_prefill(
                q, c_kv, c_csr, sink, SCALE, extend_kv=swa_kv, extend_csr=swa_csr
            )
            ref = _attention_reference(
                q, [(c_kv, c_kv, c_csr), (swa_kv, swa_kv, swa_csr)], sink
            )
        err = (out.float() - ref).abs().max().item() / ref.abs().max().item()
        self.assertLess(err, TOL, (heads, two_sources, err))

    def test_matches_fp32_reference(self):
        gen = torch.Generator(device="cuda").manual_seed(0)
        for heads in (32, 16):
            for two_sources in (True, False):
                self._case(heads, two_sources, gen)


@unittest.skipUnless(_fp8_opus_available(), "gfx950 with aiter's fp8 OPUS DMA path")
class TestQuantizeQOpusFp8(CustomTestCase):
    def test_packs_mxfp8_nope_and_keeps_rope(self):
        from sglang.kernels.ops.attention.dsv4.opus_sparse_prefill_hip import (
            quantize_q_opus_fp8,
        )

        gen = torch.Generator(device="cuda").manual_seed(0)
        q = (torch.randn(37, 16, D, generator=gen, device="cuda") * 0.5).to(
            torch.bfloat16
        )
        q[0, 0, :32] = 0  # an all-zero block
        q_nope, q_rope = quantize_q_opus_fp8(q)
        self.assertEqual(q_nope.shape, (37, 16, D))
        self.assertTrue(torch.equal(q_rope, q[..., 448:]))
        ref = _mx_quant(q[..., :448])
        dec = _decode_q_nope(q_nope)
        self.assertTrue(torch.isfinite(dec).all())
        # the two quantizers may round a value to neighbouring e4m3 codes: at most one
        # step, 2^-3 of the value at the top of a block, and almost never
        diff = (dec - ref).abs()
        top = q[..., :448].float().abs().max().item()
        self.assertLess(diff.max().item(), 2**-3 * top, diff.max().item())
        self.assertLess(diff.mean().item(), 1e-3 * top, diff.mean().item())


@unittest.skipUnless(_fp8_opus_available(), "gfx950 with aiter's fp8 OPUS DMA path")
class TestOpusSparsePrefillFp8(CustomTestCase):
    def _case(self, heads, two_sources, gen):
        from sglang.kernels.ops.attention.dsv4.opus_sparse_prefill_hip import (
            combined_to_csr,
            opus_sparse_prefill_fp8,
            quantize_q_opus_fp8,
        )

        def rows(n):
            """The DMA layout (512 unit-scale e4m3 values, RoPE in the last 64), a bf16
            RoPE drawn apart from the rows' own, so only QK^T reading it and P*V
            reading the rows matches the reference, and the keys QK^T sees."""
            x = torch.randn(n, D, generator=gen, device="cuda") * 0.5
            kv = x.to(torch.float8_e4m3fn)
            rope = (torch.randn(n, 64, generator=gen, device="cuda") * 0.5).to(
                torch.bfloat16
            )
            keys = torch.cat([kv[:, :448].float(), rope.float()], dim=-1)
            return kv, rope, keys

        T, n_c, n_s = 48, 700, 300
        q = (torch.randn(T, heads, D, generator=gen, device="cuda") * 0.5).to(
            torch.bfloat16
        )
        sink = torch.randn(heads, generator=gen, device="cuda") * 0.5
        q_nope, q_rope = quantize_q_opus_fp8(q)
        # what the kernel multiplies: the packed NoPE and the bf16 RoPE
        q_ref = torch.cat([_decode_q_nope(q_nope), q_rope.float()], dim=-1)

        swa_kv, swa_rope, swa_keys = rows(n_s)
        swa_idx, swa_lens = _padded_lists(T, 128, n_s, gen)
        swa_idx[:, 0] = torch.arange(T, device="cuda", dtype=torch.int32) % n_s
        swa_lens = swa_lens.clamp(min=1)
        swa_csr = combined_to_csr(swa_idx, swa_lens, 0, n_s)
        if not two_sources:
            out = opus_sparse_prefill_fp8(
                q_nope, q_rope, swa_kv, swa_rope, swa_csr, sink, SCALE
            )
            ref = _attention_reference(q_ref, [(swa_keys, swa_kv, swa_csr)], sink)
        else:
            c_kv, c_rope, c_keys = rows(n_c)
            c_idx, c_lens = _padded_lists(T, 512, n_c, gen)
            c_lens[0] = 0
            c_csr = combined_to_csr(c_idx, c_lens, 0, n_c)
            out = opus_sparse_prefill_fp8(
                q_nope,
                q_rope,
                c_kv,
                c_rope,
                c_csr,
                sink,
                SCALE,
                extend_kv=swa_kv,
                extend_kv_rope=swa_rope,
                extend_csr=swa_csr,
            )
            ref = _attention_reference(
                q_ref, [(c_keys, c_kv, c_csr), (swa_keys, swa_kv, swa_csr)], sink
            )
        err = (out.float() - ref).abs().max().item() / ref.abs().max().item()
        self.assertLess(err, FP8_TOL, (heads, two_sources, err))

    def test_matches_fp32_reference(self):
        gen = torch.Generator(device="cuda").manual_seed(0)
        for heads in (32, 16):
            for two_sources in (True, False):
                self._case(heads, two_sources, gen)


@unittest.skipUnless(_fp8_opus_available(), "gfx950 with aiter's fp8 OPUS DMA path")
class TestOpusFp8OverV4Cache(CustomTestCase):
    def test_fp8_gather_matches_bf16_dequant(self):
        """The fp8 path _forward_prefill_opus takes: SGLang's V4 cache store, the fp8
        gather, the fp8 kernel; against the bf16 dequant path over the same cache."""
        from sglang.kernels.ops.attention.dsv4.attn import fused_store_cache
        from sglang.kernels.ops.attention.dsv4.dequant_k_cache import (
            dequantize_k_cache_paged_dsv4,
            gather_dequant_requant_fp8_paged_dsv4,
        )
        from sglang.kernels.ops.attention.dsv4.kv_layout import KVLayout
        from sglang.kernels.ops.attention.dsv4.opus_sparse_prefill_hip import (
            combined_to_csr,
            opus_sparse_prefill,
            opus_sparse_prefill_fp8,
            quantize_q_opus_fp8,
        )

        gen = torch.Generator(device="cuda").manual_seed(0)
        page, slots, T, heads, n = 64, 1024, 48, 16, 300
        cache = torch.zeros(
            slots // page,
            KVLayout.V4.page_bytes(page),
            dtype=torch.uint8,
            device="cuda",
        )
        k = (torch.randn(slots, D, generator=gen, device="cuda") * 0.5).to(
            torch.bfloat16
        )
        loc = torch.randperm(slots, generator=gen, device="cuda").to(torch.int32)
        fused_store_cache(
            k, cache, loc, page_size=page, type="flashmla", layout=KVLayout.V4
        )
        token_ids = loc[:n]

        bf16_rows = dequantize_k_cache_paged_dsv4(
            cache, token_ids, page_size=page, layout=KVLayout.V4
        ).view(n, D)
        rope_rows = torch.empty(n, 64, dtype=torch.bfloat16, device="cuda")
        fp8_rows = gather_dequant_requant_fp8_paged_dsv4(
            cache, token_ids, page_size=page, rope_out=rope_rows
        ).view(n, D)
        # the RoPE the cache holds, untouched
        self.assertTrue(torch.equal(rope_rows, bf16_rows[:, 448:]))

        q = (torch.randn(T, heads, D, generator=gen, device="cuda") * 0.5).to(
            torch.bfloat16
        )
        sink = torch.randn(heads, generator=gen, device="cuda") * 0.5
        idx, lens = _padded_lists(T, 512, n, gen)
        idx[:, 0] = torch.arange(T, device="cuda", dtype=torch.int32) % n
        csr = combined_to_csr(idx, lens.clamp(min=1), 0, n)

        ref = _attention_reference(q, [(bf16_rows, bf16_rows, csr)], sink)
        out_bf16 = opus_sparse_prefill(q, bf16_rows, csr, sink, SCALE)
        q_nope, q_rope = quantize_q_opus_fp8(q)
        out_fp8 = opus_sparse_prefill_fp8(
            q_nope, q_rope, fp8_rows, rope_rows, csr, sink, SCALE
        )
        scale = ref.abs().max().item()
        err_bf16 = (out_bf16.float() - ref).abs().max().item() / scale
        err_fp8 = (out_fp8.float() - ref).abs().max().item() / scale
        self.assertLess(err_bf16, TOL, err_bf16)
        # here Q's NoPE and V's RoPE slice are quantized too, on top of P
        self.assertLess(err_fp8, FP8_TOL, err_fp8)


@unittest.skipUnless(is_hip() and is_gfx95_supported(), "gfx950")
class TestRawTopkBuffersAreMinusOneFilled(CustomTestCase):
    def test_fresh_buffers_hold_no_stale_rows(self):
        from sglang.srt.layers.attention.deepseek_v4_backend_hip_radix import (
            DSV4AttnMetadata,
        )

        rows, topk = 1024, 512
        # hand the caching allocator back blocks of valid-looking row indices
        junk = torch.full((rows * topk * 4,), 7, dtype=torch.int32, device="cuda")
        del junk

        meta = DSV4AttnMetadata.__new__(DSV4AttnMetadata)
        meta.index_topk = topk
        meta.low_ratios = (1, 2)
        lengths = torch.full((rows,), 3, dtype=torch.int32, device="cuda")
        meta.c4_topk_lengths_clamp1 = lengths
        meta.c4_topk_lengths_raw = lengths
        meta.c1_topk_lengths_clamp1 = lengths
        meta.c2_topk_lengths_clamp1 = lengths
        meta.init_flashmla_related(is_prefill=True)

        # identity rows write only their reachable columns; OPUS reads the rest as rows
        for ratio in (1, 2):
            raw = meta.sparse_raw_indices(ratio)
            self.assertTrue(bool((raw == -1).all()), ratio)


if __name__ == "__main__":
    unittest.main()
