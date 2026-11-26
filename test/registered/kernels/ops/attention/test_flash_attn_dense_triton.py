# SPDX-License-Identifier: Apache-2.0
"""Numerics of the dense Triton prefill attention kernel (ROCm)."""

import unittest

import torch

from sglang.kernels.ops.diffusion.attention.flash_attn_dense_triton import (
    can_use_flash_attn_dense,
    flash_attn_dense,
)
from sglang.test.ci.ci_register import register_amd_ci
from sglang.test.test_utils import CustomTestCase

register_amd_ci(est_time=30, suite="stage-b-test-1-gpu-small-amd")

_ATOL = 2e-2
_RTOL = 1e-2


def _reference(query, key, value):
    """Head-major [1, H, S, D], fp32 math."""
    q, k, v = query.float(), key.float(), value.float()
    scores = (q @ k.transpose(-1, -2)) * (q.shape[-1] ** -0.5)
    return scores.softmax(-1) @ v


@unittest.skipUnless(
    torch.cuda.is_available() and torch.version.hip is not None, "ROCm only"
)
class TestFlashAttnDenseTriton(CustomTestCase):
    def _operands(self, heads, seq, kv_len=None, dim=128):
        kv_len = kv_len or seq
        q = torch.randn(1, heads, seq, dim, dtype=torch.bfloat16, device="cuda")
        k = torch.randn(1, heads, kv_len, dim, dtype=torch.bfloat16, device="cuda")
        v = torch.randn(1, heads, kv_len, dim, dtype=torch.bfloat16, device="cuda")
        return q, k, v

    def test_matches_reference(self):
        for heads, seq in ((1, 128), (2, 256), (4, 512)):
            q, k, v = self._operands(heads, seq)
            out = flash_attn_dense(
                q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
            )
            self.assertEqual(out.shape, (1, seq, heads, 128))
            torch.testing.assert_close(
                out.transpose(1, 2).float(), _reference(q, k, v), atol=_ATOL, rtol=_RTOL
            )

    def test_padded_key_value_is_ignored(self):
        # 32 zero rows are appended so the key/value length stays a multiple of
        # the block size; the kernel drops their mass from the denominator.
        heads, seq, kv_real, kv_pad = 2, 256, 288, 32
        q, k, v = self._operands(heads, seq, kv_len=kv_real + kv_pad)
        k[:, :, kv_real:].zero_()
        v[:, :, kv_real:].zero_()
        self.assertTrue(
            can_use_flash_attn_dense(
                q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2), kv_pad=kv_pad
            )
        )
        out = flash_attn_dense(
            q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2), kv_pad=kv_pad
        )
        torch.testing.assert_close(
            out.transpose(1, 2).float(),
            _reference(q, k[:, :, :kv_real], v[:, :, :kv_real]),
            atol=_ATOL,
            rtol=_RTOL,
        )

    def test_predicate_rejects_unsupported_operands(self):
        q, k, v = self._operands(2, 128)
        operands = (q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2))
        self.assertTrue(can_use_flash_attn_dense(*operands))
        # head dim 64
        self.assertFalse(
            can_use_flash_attn_dense(*(t[..., :64].contiguous() for t in operands))
        )
        # query length not a multiple of the block size
        self.assertFalse(
            can_use_flash_attn_dense(*(t[:, :96].contiguous() for t in operands))
        )
        # fp16
        self.assertFalse(can_use_flash_attn_dense(*(t.half() for t in operands)))

    def test_cpu_operands_are_rejected(self):
        q, k, v = self._operands(1, 128)
        cpu = (
            q.cpu().transpose(1, 2),
            k.cpu().transpose(1, 2),
            v.cpu().transpose(1, 2),
        )
        self.assertFalse(can_use_flash_attn_dense(*cpu))


if __name__ == "__main__":
    unittest.main()
