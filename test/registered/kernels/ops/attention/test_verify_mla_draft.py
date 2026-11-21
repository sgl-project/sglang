"""The gfx950 monokernel's DFLASH MLA draft verify against extend_attention_fwd."""

import unittest

import torch

from sglang.kernels.ops.attention.extend_attention import extend_attention_fwd
from sglang.kernels.ops.attention.verify_mla_draft import verify_mla_draft_fwd
from sglang.srt.utils import is_gfx95_supported, is_hip
from sglang.test.ci.ci_register import register_amd_ci
from sglang.test.test_utils import CustomTestCase

# backend-specific: the DFLASH MLA draft verify serves the gfx950 monokernel only.
register_amd_ci(est_time=30, suite="base-b-test-1-gpu-small-amd-mi35x")


@unittest.skipUnless(
    is_hip() and is_gfx95_supported(), "the DFLASH MLA draft verify is gfx950's"
)
class TestDraftVerifyMLA(CustomTestCase):
    def test_matches_extend_attention(self):
        """A causal 6-token block (three live 2-row combine tiles) over a 300-key FP8
        prefix that the 256-key window trims, with BF16 dots."""
        torch.manual_seed(0)
        prefix, l_ext, h_q, d, dv, window = 300, 6, 4, 576, 512, 256
        q = torch.randn(l_ext, h_q, d, dtype=torch.bfloat16, device="cuda")
        k = torch.randn(l_ext, 1, d, dtype=torch.bfloat16, device="cuda")
        v = torch.randn(l_ext, 1, dv, dtype=torch.bfloat16, device="cuda")
        # the prefix sits scattered in a cache with spare rows
        k_buffer = torch.randn(prefix + 64, 1, d, device="cuda").to(torch.float8_e4m3fn)
        v_buffer = torch.randn(prefix + 64, 1, dv, device="cuda").to(
            torch.float8_e4m3fn
        )
        kv_indices = torch.randperm(prefix + 64, device="cuda")[:prefix]
        qo_indptr = torch.tensor([0, l_ext], dtype=torch.int32, device="cuda")
        kv_indptr = torch.tensor([0, prefix], dtype=torch.int32, device="cuda")
        cache = (k_buffer, v_buffer, qo_indptr, kv_indptr, kv_indices)
        mask = (None, True, None, l_ext, 1.0, 1.0)
        opts = dict(sm_scale=d**-0.5, sliding_window_size=window, bf16_dot=True)

        actual = torch.empty(l_ext, h_q, dv, dtype=torch.bfloat16, device="cuda")
        ran = verify_mla_draft_fwd(q, k, v, actual, *cache, *mask, max_bs=1, **opts)
        self.assertTrue(ran)
        expected = torch.empty_like(actual)
        extend_attention_fwd(q, k, v, expected, *cache, *mask, **opts)
        torch.testing.assert_close(actual, expected, atol=8e-2, rtol=2e-2)


if __name__ == "__main__":
    unittest.main()
