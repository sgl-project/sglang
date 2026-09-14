"""aiter's paged batch-prefill over prefix + chunk must match an fp32 reference."""

import unittest

import torch

from sglang.kernels.ops.attention.extend_attention import extend_attention_fwd
from sglang.srt.utils import is_hip
from sglang.test.ci.ci_register import register_amd_ci
from sglang.test.test_utils import CustomTestCase

register_amd_ci(est_time=60, suite="stage-b-test-1-gpu-small-amd-mi35x")

from sglang.srt.layers.attention.aiter_extend_long_prefix import (
    AiterLongPrefixExtend,
    build_paged_kv_indices,
)

_AITER = AiterLongPrefixExtend.try_create()
_AITER_OK = _AITER is not None

# observed on gfx950: aiter ~1e-3 (bf16 KV), 1e-3..1e-2 (fp8 KV); scaled fp8 Triton hits 5e-2
ATOL = {torch.bfloat16: 5e-3, torch.float8_e4m3fn: 3e-2}
REF_ROWS = 64


def _inputs(prefix_lens, extend_lens, h_q, h_kv, d, kv_dtype, device):
    B = len(prefix_lens)
    total_prefix = int(sum(prefix_lens))
    n_ext = int(sum(extend_lens))
    total = total_prefix + n_ext
    k_buffer = torch.randn(total, h_kv, d, device=device).to(kv_dtype)
    v_buffer = torch.randn(total, h_kv, d, device=device).to(kv_dtype)
    # scrambled cache locations for prefix and chunk alike
    perm = torch.randperm(total, device=device)
    kv_indices = perm[:total_prefix].to(torch.int64)
    out_cache_loc = perm[total_prefix:].to(torch.int64)
    kv_indptr = torch.zeros(B + 1, dtype=torch.int32, device=device)
    kv_indptr[1:] = torch.cumsum(
        torch.tensor(prefix_lens, dtype=torch.int32, device=device), 0
    )
    qo_indptr = torch.zeros(B + 1, dtype=torch.int32, device=device)
    qo_indptr[1:] = torch.cumsum(
        torch.tensor(extend_lens, dtype=torch.int32, device=device), 0
    )
    q = torch.randn(n_ext, h_q, d, dtype=torch.bfloat16, device=device)
    # the chunk's K/V as produced (bf16) and as stored in the cache
    k = k_buffer[out_cache_loc].to(torch.bfloat16)
    v = v_buffer[out_cache_loc].to(torch.bfloat16)
    return q, k, v, k_buffer, v_buffer, qo_indptr, kv_indptr, kv_indices, out_cache_loc


@unittest.skipIf(not torch.cuda.is_available() or not is_hip(), "ROCm GPU required")
@unittest.skipIf(not _AITER_OK, "aiter mha_batch_prefill required")
class TestExtendLongPrefixAiter(CustomTestCase):
    """A wrong page table, descale or causal offset shows as a mismatch on the last rows."""

    def _run(self, prefix_lens, extend_lens, h_q, h_kv, d, kv_dtype, kv_scale):
        device = "cuda"
        torch.manual_seed(0)
        (
            q,
            k,
            v,
            k_buffer,
            v_buffer,
            qo_indptr,
            kv_indptr,
            kv_indices,
            out_cache_loc,
        ) = _inputs(prefix_lens, extend_lens, h_q, h_kv, d, kv_dtype, device)
        B = len(prefix_lens)
        max_ext = max(extend_lens)
        sm_scale = d**-0.5
        k_scale = v_scale = kv_scale if kv_dtype != torch.bfloat16 else 1.0

        o_ref = torch.empty_like(q)
        extend_attention_fwd(
            q,
            k,
            v,
            o_ref,
            k_buffer,
            v_buffer,
            qo_indptr,
            kv_indptr,
            kv_indices,
            None,
            True,
            None,
            max_ext,
            k_scale,
            v_scale,
            sm_scale=sm_scale,
            extend_seq_lens_cpu=list(extend_lens),
        )

        extend_seq_lens = torch.tensor(extend_lens, dtype=torch.int32, device=device)
        extend_start_loc = qo_indptr[:B].clone()
        paged_indptr, pages = build_paged_kv_indices(
            kv_indptr, kv_indices, extend_start_loc, extend_seq_lens, out_cache_loc, B
        )
        self.assertEqual(paged_indptr.dtype, torch.int32)
        self.assertEqual(pages.dtype, torch.int32)
        # page table: prefix indices then chunk locations, per request
        for i in range(B):
            exp = torch.cat(
                [
                    kv_indices[kv_indptr[i] : kv_indptr[i + 1]],
                    out_cache_loc[qo_indptr[i] : qo_indptr[i + 1]],
                ]
            )
            torch.testing.assert_close(
                pages[paged_indptr[i] : paged_indptr[i + 1]].to(torch.int64), exp
            )

        o = torch.empty_like(q)
        _AITER.forward(
            q,
            o,
            k_buffer,
            v_buffer,
            qo_indptr,
            paged_indptr,
            pages,
            max_ext,
            max(p + e for p, e in zip(prefix_lens, extend_lens)),
            sm_scale,
            k_scale=None if kv_dtype == torch.bfloat16 else k_scale,
            v_scale=None if kv_dtype == torch.bfloat16 else v_scale,
        )
        # fp32 reference on the last REF_ROWS rows only: a full one is n_ext x h_q x context
        g = h_q // h_kv
        err_ref, err_tri = 0.0, 0.0
        for i in range(B):
            P, E = prefix_lens[i], extend_lens[i]
            r0 = max(0, E - REF_ROWS)
            rows = slice(int(qo_indptr[i]) + r0, int(qo_indptr[i + 1]))
            K = (
                torch.cat(
                    [
                        k_buffer[kv_indices[kv_indptr[i] : kv_indptr[i + 1]]],
                        k_buffer[out_cache_loc[qo_indptr[i] : qo_indptr[i + 1]]],
                    ]
                ).float()
                * k_scale
            )
            V = (
                torch.cat(
                    [
                        v_buffer[kv_indices[kv_indptr[i] : kv_indptr[i + 1]]],
                        v_buffer[out_cache_loc[qo_indptr[i] : qo_indptr[i + 1]]],
                    ]
                ).float()
                * v_scale
            )
            qs = q[rows].float()
            if kv_dtype != torch.bfloat16:
                qs = qs.to(kv_dtype).float()  # both kernels feed fp8 q to the fp8 dot
            s = torch.einsum("thd,nhd->thn", qs, K.repeat_interleave(g, dim=1))
            s *= sm_scale
            pos = torch.arange(r0, E, device=device) + P
            kpos = torch.arange(P + E, device=device)
            s.masked_fill_((kpos[None, :] > pos[:, None])[:, None, :], float("-inf"))
            ref = torch.einsum(
                "thn,nhd->thd", torch.softmax(s, -1), V.repeat_interleave(g, dim=1)
            )
            err_ref = max(err_ref, (o[rows].float() - ref).abs().max().item())
            err_tri = max(err_tri, (o_ref[rows].float() - ref).abs().max().item())
        self.assertLess(
            err_ref, ATOL[kv_dtype], f"aiter vs fp32 ref {err_ref} (triton {err_tri})"
        )

    def test_fp8_gqa16_ragged(self):
        self._run(
            [4096, 12000, 300], [2048, 1500, 7], 16, 1, 128, torch.float8_e4m3fn, 1.0
        )

    def test_fp8_scaled(self):
        self._run([9000], [3000], 16, 1, 128, torch.float8_e4m3fn, 0.7)

    def test_fp8_long_prefix_chunk(self):
        self._run([70000, 1000], [8192, 4096], 16, 1, 128, torch.float8_e4m3fn, 1.0)

    def test_bf16_gqa16(self):
        self._run([5000, 100], [1024, 1024], 16, 1, 128, torch.bfloat16, 1.0)

    def test_zero_prefix(self):
        self._run([0, 0], [2048, 512], 16, 1, 128, torch.float8_e4m3fn, 1.0)


if __name__ == "__main__":
    unittest.main()
