"""Real-device XPU tests for the encoder-decoder varlen attention path.

Unlike test_encoder_decoder_varlen_gather.py (CPU, kernel mocked), these run the
actual sgl-kernel-xpu flash_attn_varlen_func on an XPU. They guard two things the
mocked test cannot:
  1. the dense gather + varlen kernel produce correct attention on hardware, and
  2. encoder_lens == 0 short-circuits to zeros WITHOUT faulting the device
     (an empty-KV varlen call is UR_RESULT_ERROR_DEVICE_LOST on XPU).
"""

import unittest

import torch

from sglang.srt.layers.attention.xpu_backend import XPUAttentionBackend
from sglang.test.ci.ci_register import register_xpu_ci
from sglang.test.test_utils import CustomTestCase

register_xpu_ci(est_time=15, suite="stage-b-test-1-gpu-xpu")


def _sdpa_ref(
    *, q, k_flat, v_flat, page_table, cache_seqlens, cu_seqlens_q, scale, causal
):
    """Per-request SDPA oracle: gather each request's valid slots straight from
    the page_table and attend. fp32 math for a stable bf16 comparison."""
    outs = []
    for i in range(page_table.shape[0]):
        qs, qe = int(cu_seqlens_q[i]), int(cu_seqlens_q[i + 1])
        sl = int(cache_seqlens[i])
        q_i = q[qs:qe].float()
        slots = page_table[i, :sl].long()
        k_i = k_flat.index_select(0, slots).float()
        v_i = v_flat.index_select(0, slots).float()
        scores = torch.einsum("qhd,khd->hqk", q_i, k_i) * scale
        lq, lk = qe - qs, sl
        if causal and lq > 0 and lk > 0:
            row = torch.arange(lq, device=q.device).unsqueeze(1)
            col = torch.arange(lk, device=q.device).unsqueeze(0)
            keep = col <= (lk - lq + row)
            scores = scores.masked_fill(~keep.unsqueeze(0), float("-inf"))
        probs = scores.softmax(dim=-1)
        outs.append(torch.einsum("hqk,khd->qhd", probs, v_i))
    return torch.cat(outs, dim=0)


@unittest.skipUnless(
    hasattr(torch, "xpu") and torch.xpu.is_available(), "requires an Intel XPU"
)
class TestXPUEncoderDecoderVarlen(CustomTestCase):
    # Whisper-large-v3 is MHA (num_kv_heads == num_heads), so use MHA here to keep
    # the reference exact; head_dim=64 satisfies the kernel's alignment.
    H, D, TOTAL_SLOTS = 8, 64, 64

    def setUp(self):
        torch.manual_seed(0)
        self.dev = torch.device("xpu")
        self.backend = XPUAttentionBackend.__new__(XPUAttentionBackend)
        self.backend.num_splits = 0
        self.backend.is_encoder_decoder = True
        # Deliberately scattered (non-page-aligned) slot indices: a paged kernel
        # would mis-read these; the dense gather must be alignment-agnostic.
        perm = torch.randperm(self.TOTAL_SLOTS)
        self.k_flat = torch.randn(
            self.TOTAL_SLOTS, self.H, self.D, dtype=torch.bfloat16, device=self.dev
        )[perm].contiguous()
        self.v_flat = torch.randn(
            self.TOTAL_SLOTS, self.H, self.D, dtype=torch.bfloat16, device=self.dev
        )[perm].contiguous()

    def _check(self, *, cache_seqlens, cu_seqlens_q, causal):
        cache_seqlens = cache_seqlens.to(self.dev)
        cu_seqlens_q = cu_seqlens_q.to(self.dev)
        num_rows = int(cu_seqlens_q[-1])
        m = int(cache_seqlens.max())
        # Rows packed valid-first; scatter distinct slots per request (build the
        # permutation on CPU, then move -- randperm(device="xpu") is unreliable).
        page_table = (
            torch.stack(
                [
                    torch.randperm(self.TOTAL_SLOTS)[:m]
                    for _ in range(cache_seqlens.numel())
                ]
            )
            .to(torch.int32)
            .to(self.dev)
        )
        q = torch.randn(num_rows, self.H, self.D, dtype=torch.bfloat16, device=self.dev)
        scale = 0.5
        max_seqlen_q = int((cu_seqlens_q[1:] - cu_seqlens_q[:-1]).max())

        got = self.backend._varlen_gather_attn(
            q=q,
            k_flat=self.k_flat,
            v_flat=self.v_flat,
            page_table=page_table,
            cache_seqlens=cache_seqlens,
            cu_seqlens_q=cu_seqlens_q,
            max_seqlen_q=max_seqlen_q,
            scale=scale,
            softcap=0.0,
            causal=causal,
        )
        want = _sdpa_ref(
            q=q,
            k_flat=self.k_flat,
            v_flat=self.v_flat,
            page_table=page_table,
            cache_seqlens=cache_seqlens,
            cu_seqlens_q=cu_seqlens_q,
            scale=scale,
            causal=causal,
        )
        self.assertEqual(tuple(got.shape), (num_rows, self.H, self.D))
        # bf16 kernel vs fp32 reference: loose tolerance.
        torch.testing.assert_close(got.float(), want, rtol=2e-2, atol=2e-2)

    def test_cross_attention_decode_on_xpu(self):
        # 1 query/request, attend all encoder KV, non-causal, unequal lengths.
        self._check(
            cache_seqlens=torch.tensor([5, 8], dtype=torch.int32),
            cu_seqlens_q=torch.tensor([0, 1, 2], dtype=torch.int32),
            causal=False,
        )

    def test_decoder_self_attention_decode_on_xpu(self):
        self._check(
            cache_seqlens=torch.tensor([4, 6], dtype=torch.int32),
            cu_seqlens_q=torch.tensor([0, 1, 2], dtype=torch.int32),
            causal=True,
        )

    def test_encoder_lens_zero_no_device_loss(self):
        # The exact Whisper text-only warmup path: a cross-attention layer with
        # encoder_lens == 0. Launching the empty-KV kernel here would fault the XPU
        # (UR_RESULT_ERROR_DEVICE_LOST); the guard must return zeros instead.
        from types import SimpleNamespace

        metadata = SimpleNamespace(
            encoder_page_table=torch.zeros(1, 0, dtype=torch.int32, device=self.dev),
            encoder_lens_int32=torch.zeros(1, dtype=torch.int32, device=self.dev),
            page_table=torch.zeros(1, 0, dtype=torch.int32, device=self.dev),
            cache_seqlens_int32=torch.zeros(1, dtype=torch.int32, device=self.dev),
            cu_seqlens_q=torch.tensor([0, 1], dtype=torch.int32, device=self.dev),
            max_seq_len_q=1,
        )
        layer = SimpleNamespace(
            is_cross_attention=True,
            tp_q_head_num=self.H,
            tp_k_head_num=self.H,
            tp_v_head_num=self.H,
            head_dim=self.D,
            scaling=0.5,
            logit_cap=0.0,
        )
        key_cache = self.k_flat.view(-1, 1, self.H, self.D)
        value_cache = self.v_flat.view(-1, 1, self.H, self.D)
        q = torch.randn(1, self.H * self.D, dtype=torch.bfloat16, device=self.dev)

        out = self.backend._forward_encoder_decoder_mha(
            q=q,
            key_cache=key_cache,
            value_cache=value_cache,
            layer=layer,
            metadata=metadata,
            decode=True,
        )
        torch.xpu.synchronize()
        self.assertEqual(tuple(out.shape), (1, self.H, self.D))
        self.assertTrue(bool((out == 0).all()))

        # Device must still be alive: a subsequent real op would raise
        # UR_RESULT_ERROR_DEVICE_LOST if the empty-KV call had faulted it.
        probe = (torch.ones(4, device=self.dev) * 2).sum()
        torch.xpu.synchronize()
        self.assertEqual(int(probe.item()), 8)


if __name__ == "__main__":
    unittest.main()
