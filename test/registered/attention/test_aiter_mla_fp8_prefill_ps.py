"""Numerics of the aiter asm fp8 MLA prefill (PS metadata + mla_reduce_v1) on gfx950."""

import unittest
from types import SimpleNamespace

import torch

from sglang.srt.utils import is_hip
from sglang.test.ci.ci_register import register_amd_ci
from sglang.test.test_utils import CustomTestCase

register_amd_ci(est_time=60, suite="stage-b-test-1-gpu-small-amd-mi35x")

_RUNNABLE = is_hip() and torch.cuda.is_available()
if _RUNNABLE:
    from sglang.kernels.ops.quantization.fp8_kernel import fp8_dtype
    from sglang.srt.layers.attention.aiter_backend import (
        AiterAttnBackend,
        ForwardMetadata,
    )
    from sglang.srt.utils import is_gfx95_supported

    _RUNNABLE = is_gfx95_supported()

QK_NOPE, QK_ROPE, V_HEAD = 128, 64, 128
SCALE = (QK_NOPE + QK_ROPE) ** -0.5
# Kimi-K3 at TP8: no mla_reduce_v1 instance in the pinned AITER, so it pads to 16.
KIMI_K3_TP8_HEADS = 12


def _backend(num_heads, q_lens):
    backend = object.__new__(AiterAttnBackend)
    backend.device = torch.device("cuda")
    backend.input_dtype = torch.bfloat16
    backend._arange_buf = None
    backend.use_fp8_prefill_attn = True
    backend.fp8_prefill_num_head = backend.check_fp8_prefill_num_head(
        num_head=num_heads, num_kv_head=num_heads, v_head_dim=V_HEAD
    )
    backend.fp8_prefill_num_kv_head = backend.fp8_prefill_num_head
    qo_indptr = torch.zeros(len(q_lens) + 1, dtype=torch.int32)
    qo_indptr[1:] = torch.tensor(q_lens).cumsum(0)
    backend.forward_metadata = ForwardMetadata(
        kv_indptr=None,
        kv_indices=None,
        qo_indptr=qo_indptr.cuda(),
        kv_last_page_len=None,
        max_q_len=max(q_lens),
        max_kv_len=None,
    )
    return backend


def _layer(num_heads):
    return SimpleNamespace(tp_q_head_num=num_heads, v_head_dim=V_HEAD, scaling=SCALE)


def _qkv(num_heads, q_lens, kv_lens):
    q = torch.randn(sum(q_lens), num_heads, QK_NOPE + QK_ROPE, device="cuda")
    k_nope = torch.randn(sum(kv_lens), num_heads, QK_NOPE, device="cuda")
    k_pe = torch.randn(sum(kv_lens), 1, QK_ROPE, device="cuda")
    v = torch.randn(sum(kv_lens), num_heads, V_HEAD, device="cuda")
    return [t.to(torch.bfloat16) for t in (q, k_nope, k_pe, v)]


def _cat_k(k_nope, k_pe):
    return torch.cat([k_nope, k_pe.expand(-1, k_nope.shape[1], -1)], dim=-1)


def _reference(q, k, v, q_lens, kv_lens):
    """fp32 non-causal attention per request over the fp8-rounded inputs."""
    q, k, v = (t.to(fp8_dtype).float() for t in (q, k, v))
    out = torch.zeros(q.shape[0], q.shape[1], V_HEAD, device=q.device)
    lse = torch.full(q.shape[:2], float("-inf"), device=q.device)
    q_off = kv_off = 0
    for q_len, kv_len in zip(q_lens, kv_lens):
        if kv_len:
            qs = slice(q_off, q_off + q_len)
            ks = slice(kv_off, kv_off + kv_len)
            scores = torch.einsum("qhd,khd->hqk", q[qs], k[ks]) * SCALE
            lse[qs] = scores.logsumexp(-1).transpose(0, 1)
            out[qs] = torch.einsum("hqk,khd->qhd", scores.softmax(-1), v[ks])
        q_off += q_len
        kv_off += kv_len
    return out, lse


@unittest.skipUnless(_RUNNABLE, "requires gfx950 with AITER")
class TestAiterMlaFp8PrefillPs(CustomTestCase):
    def setUp(self):
        torch.manual_seed(0)

    def _assert_matches(self, out, lse, ref_out, ref_lse, rows):
        err = (out[rows].float() - ref_out[rows]).abs().max().item()
        self.assertLess(err, 0.03 * ref_out[rows].abs().max().item())
        lse_err = (lse[rows] - ref_lse[rows]).abs().max().item()
        self.assertLess(lse_err, 1e-2)

    def test_prefill_heads_match_reference(self):
        """Heads without an mla_reduce_v1 instance pad up to one;
        padding must not perturb the real heads' output or LSE, nor reach a missing instance."""
        q_lens, kv_lens = [300, 77], [517, 1200]
        for num_heads in (KIMI_K3_TP8_HEADS, 16):
            with self.subTest(num_heads=num_heads):
                backend = _backend(num_heads, q_lens)
                kv_indptr = torch.zeros(len(kv_lens) + 1, dtype=torch.int32)
                kv_indptr[1:] = torch.tensor(kv_lens).cumsum(0)
                ps = backend._build_prefill_ps_metadata(
                    qo_indptr=backend.forward_metadata.qo_indptr,
                    kv_indptr=kv_indptr.cuda(),
                    kv_lens_cpu=torch.tensor(kv_lens, dtype=torch.int32),
                    num_kv_tokens=sum(kv_lens),
                    max_q_len=max(q_lens),
                    is_causal=False,
                    need_lse=True,
                )
                q, k_nope, k_pe, v = _qkv(num_heads, q_lens, kv_lens)
                k = _cat_k(k_nope, k_pe)
                out, lse = backend._mla_fp8_prefill_attn_ps(
                    q, k, v, _layer(num_heads), ps
                )
                self.assertEqual(out.shape, (sum(q_lens), num_heads, V_HEAD))
                self.assertTrue(lse.is_contiguous())
                ref_out, ref_lse = _reference(q, k, v, q_lens, kv_lens)
                self._assert_matches(out, lse, ref_out, ref_lse, slice(None))


if __name__ == "__main__":
    unittest.main()
