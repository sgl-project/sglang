"""MI35x regression test for the direct-write a8w8 bmm output (PR #34498).

The per-channel fp8 kv_b bmm on gfx95 used to write the a8w8 GEMM output in
(heads, tokens, vdim) layout (``YQ=None, transpose_bm=False``) and then run
``attn_output.transpose(0, 1).flatten(1, 2)`` in the o_proj epilogue, where the
transpose forces a memory copy. The PR preallocates ``_bmm_buf`` in
(tokens, heads, vdim) layout and passes ``YQ=_bmm_buf, transpose_bm=True`` so the
GEMM writes the final layout directly, turning the downstream ``flatten(1, 2)``
into a free view.

This test drives the same aiter kernel BOTH ways with identical inputs and
asserts:
  * numerical parity      — same kernel + same quant => bit-identical results,
  * (tokens, heads, vdim) shape, C-contiguity of the direct-write buffer,
  * ``flatten(1, 2)`` shares storage with the buffer (no copy),
  * the ``YQ + transpose_bm=True`` path actually executes,
covering decode (M=1) and multiple tokens.

Requires ROCm/aiter on gfx95 (MI35x); skipped elsewhere.

NOTE (validate on MI35x): ``w_scale`` here is built as per-batched-tensor
(heads, 1, 1); if the pinned aiter kernel expects a different scale rank, adjust
``_make_inputs`` — the parity assertion is independent of the exact shape since
both paths share it.
"""

import unittest

import torch

from sglang.test.ci.ci_register import register_amd_ci
from sglang.test.test_utils import CustomTestCase

register_amd_ci(est_time=30, suite="stage-b-test-1-gpu-small-amd-mi35x")

try:
    from aiter.ops.triton.batched_gemm_a8w8_a_per_token_group_prequant_w_per_batched_tensor_quant import (
        batched_gemm_a8w8_a_per_token_group_prequant_w_per_batched_tensor_quant as _bmm_a8w8,
    )

    from sglang.srt.models.deepseek_common.utils import _use_aiter_gfx95

    _HAS_PATH = _use_aiter_gfx95 and torch.cuda.is_available()
except Exception:
    _HAS_PATH = False


@unittest.skipUnless(_HAS_PATH, "requires ROCm/aiter gfx95 (MI35x)")
class TestDirectWriteA8W8Bmm(CustomTestCase):
    H, K, N = 8, 512, 128  # heads, kv_lora_rank (in), v_head_dim (out); N=vdim
    GROUP = 128

    def _make_inputs(self, M):
        device = "cuda"
        # attn_output: (tokens, heads, kv_lora) bf16 — matches production X.
        attn_output = torch.randn(
            M, self.H, self.K, dtype=torch.bfloat16, device=device
        )
        # w_vc: (heads, kv_lora, vdim) fp8; WQ passed as (heads, vdim, kv_lora).
        w_vc = (torch.randn(self.H, self.K, self.N, device=device) * 0.1).to(
            torch.float8_e4m3fn
        )
        # per-batched-tensor weight scale (see module NOTE).
        w_scale = (
            torch.rand(self.H, 1, 1, dtype=torch.float32, device=device) * 0.05 + 0.01
        )
        return attn_output, w_vc, w_scale

    def _original_path(self, attn_output, w_vc, w_scale):
        # Pre-PR behavior: (heads, tokens, vdim) then transpose(0,1).flatten(1,2).
        out = _bmm_a8w8(
            X=attn_output,
            WQ=w_vc.transpose(-1, -2),
            w_scale=w_scale,
            group_size=self.GROUP,
            YQ=None,
            transpose_bm=False,
            transpose_bm_in=True,
            dtype=torch.bfloat16,
        )
        return out.transpose(0, 1).flatten(1, 2)

    def _direct_write_path(self, attn_output, w_vc, w_scale):
        # PR behavior: preallocated (tokens, heads, vdim) buffer, transpose_bm=True.
        M = attn_output.shape[0]
        buf = torch.empty(
            M, self.H, self.N, device=attn_output.device, dtype=torch.bfloat16
        )
        _bmm_a8w8(
            X=attn_output,
            WQ=w_vc.transpose(-1, -2),
            w_scale=w_scale,
            group_size=self.GROUP,
            YQ=buf,
            transpose_bm=True,
            transpose_bm_in=True,
            dtype=torch.bfloat16,
        )
        return buf

    def test_direct_write_matches_original(self):
        for M in (1, 4):  # decode (M=1) and multi-token
            with self.subTest(tokens=M):
                attn_output, w_vc, w_scale = self._make_inputs(M)

                ref = self._original_path(attn_output, w_vc, w_scale)
                buf = self._direct_write_path(attn_output, w_vc, w_scale)

                # (tokens, heads, vdim) contiguous buffer.
                self.assertEqual(list(buf.shape), [M, self.H, self.N])
                self.assertTrue(buf.is_contiguous())

                # flatten(1, 2) is a free view sharing the buffer's storage.
                flat = buf.flatten(1, 2)
                self.assertEqual(list(flat.shape), [M, self.H * self.N])
                self.assertEqual(
                    flat.untyped_storage().data_ptr(),
                    buf.untyped_storage().data_ptr(),
                )

                # Same kernel + same quant => bit-identical to the original path.
                torch.testing.assert_close(flat, ref, rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()
