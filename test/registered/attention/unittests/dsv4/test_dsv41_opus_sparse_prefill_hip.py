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


def _opus_available() -> bool:
    if not (is_hip() and is_gfx95_supported()):
        return False
    try:
        from aiter.ops.pa_sparse_prefill_opus import pa_sparse_prefill_opus  # noqa: F401
    except ImportError:
        return False
    return True


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
    """fp32 attention with sink: every query row over the union of its rows in each source."""
    T, H, _ = q.shape
    out = torch.zeros(T, H, D, dtype=torch.float32, device=q.device)
    for t in range(T):
        keys = []
        for kv, (indices, indptr) in sources:
            rows = indices[indptr[t] : indptr[t + 1]].long()
            keys.append(kv[rows].float())
        k = torch.cat(keys)
        s = q[t].float() @ k.T * SCALE
        logits = torch.cat([s, sink[:, None].float()], dim=1)
        p = torch.softmax(logits, dim=1)[:, :-1]
        out[t] = p @ k
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
        q = (torch.randn(T, heads, D, generator=gen, device="cuda") * 0.5).to(torch.bfloat16)
        sink = torch.randn(heads, generator=gen, device="cuda") * 0.5
        swa_kv = (torch.randn(n_s, D, generator=gen, device="cuda") * 0.5).to(torch.bfloat16)
        swa_idx, swa_lens = _padded_lists(T, 128, n_s, gen)
        # every query row sees at least one key, as in prefill (its own SWA row)
        swa_idx[:, 0] = torch.arange(T, device="cuda", dtype=torch.int32) % n_s
        swa_lens = swa_lens.clamp(min=1)
        swa_csr = combined_to_csr(swa_idx, swa_lens, 0, n_s)
        if not two_sources:
            out = opus_sparse_prefill(q, swa_kv, swa_csr, sink, SCALE)
            ref = _attention_reference(q, [(swa_kv, swa_csr)], sink)
        else:
            c_kv = (torch.randn(n_c, D, generator=gen, device="cuda") * 0.5).to(torch.bfloat16)
            c_idx, c_lens = _padded_lists(T, 512, n_c, gen)
            # a row with no compressed history, like a request's first tokens
            c_lens[0] = 0
            c_csr = combined_to_csr(c_idx, c_lens, 0, n_c)
            out = opus_sparse_prefill(
                q, c_kv, c_csr, sink, SCALE, extend_kv=swa_kv, extend_csr=swa_csr
            )
            ref = _attention_reference(q, [(c_kv, c_csr), (swa_kv, swa_csr)], sink)
        err = (out.float() - ref).abs().max().item() / ref.abs().max().item()
        self.assertLess(err, TOL, (heads, two_sources, err))

    def test_matches_fp32_reference(self):
        gen = torch.Generator(device="cuda").manual_seed(0)
        for heads in (32, 16):
            for two_sources in (True, False):
                self._case(heads, two_sources, gen)


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
