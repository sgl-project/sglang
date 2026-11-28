"""_safe_merge_state Triton fallback agrees with FlashInfer's merge_state.

FlashInfer's merge_state takes and returns base-2 log-sum-exp, while the
in-tree Triton fallback uses natural log. _safe_merge_state converts between
the two conventions; forcing the fallback with a patched
_merge_state_max_safe_num_heads and comparing against FlashInfer's own
merge_state on the same inputs catches the two branches drifting apart.
"""

import unittest
from unittest import mock

import torch

from sglang.srt.layers.attention import flashinfer_backend as backend
from sglang.srt.utils import is_flashinfer_available
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=5, stage="base-b-kernel-unit", runner_config="1-gpu-large")


@unittest.skipUnless(
    torch.cuda.is_available() and is_flashinfer_available(),
    "CUDA + flashinfer are required",
)
class TestSafeMergeStateTritonFallback(CustomTestCase):
    def test_triton_fallback_matches_flashinfer_merge_state(self):
        from flashinfer.cascade import merge_state

        torch.manual_seed(0)
        num_tokens, num_heads, head_dim = 4, 8, 64
        v_a = torch.randn(
            num_tokens,
            num_heads,
            head_dim,
            device="cuda",
            dtype=torch.float16,
        )
        v_b = torch.randn(
            num_tokens,
            num_heads,
            head_dim,
            device="cuda",
            dtype=torch.float16,
        )
        # Base-2 log-sum-exp, the convention flashinfer's merge_state uses.
        # fp32: FlashInfer returns LSE as fp32, and merge_state_triton fails
        # to compile on fp16 LSE (ternary with mixed fp32/fp16 types).
        s_a = torch.randn(num_tokens, num_heads, device="cuda", dtype=torch.float32)
        s_b = torch.randn(num_tokens, num_heads, device="cuda", dtype=torch.float32)

        with mock.patch.object(
            backend, "_merge_state_max_safe_num_heads", return_value=0
        ):
            v_merged, s_merged = backend._safe_merge_state(v_a, s_a, v_b, s_b)
        v_ref, s_ref = merge_state(v_a, s_a, v_b, s_b)

        torch.testing.assert_close(v_merged, v_ref, rtol=1e-2, atol=1e-2)
        torch.testing.assert_close(s_merged, s_ref, rtol=1e-2, atol=1e-2)


if __name__ == "__main__":
    unittest.main()
