# SPDX-License-Identifier: Apache-2.0
import unittest
from unittest import mock

import torch
import torch.nn.functional as F

from sglang.test.ci.ci_register import register_amd_ci

register_amd_ci(est_time=15, stage="base-b", runner_config="1-gpu-small-amd")

HEADS = 8
HEAD_DIM = 128
SEQ = 1024


@unittest.skipUnless(
    torch.cuda.is_available() and torch.version.hip, "needs a ROCm GPU"
)
class TestRocmMhaV4Dispatch(unittest.TestCase):
    def setUp(self):
        try:
            from sglang.multimodal_gen.runtime.layers.attention.backends import (
                aiter as backend,
            )
        except ImportError as exc:
            self.skipTest(f"AITer backend unavailable: {exc}")
        if backend._MHA_V4 is None:
            self.skipTest("MHA v4 is not available in this aiter build")
        self.backend = backend
        self.impl = backend.AITerImpl(
            num_heads=HEADS, head_size=HEAD_DIM, softmax_scale=HEAD_DIM**-0.5
        )
        torch.manual_seed(0)
        shape = (1, SEQ, HEADS, HEAD_DIM)
        self.q, self.k, self.v = (
            torch.randn(shape, device="cuda", dtype=torch.bfloat16) for _ in range(3)
        )

    def _sdpa_reference(self):
        q, k, v = (t.permute(0, 2, 1, 3).float() for t in (self.q, self.k, self.v))
        out = F.scaled_dot_product_attention(q, k, v, scale=HEAD_DIM**-0.5)
        return out.permute(0, 2, 1, 3)

    def test_flag_on_matches_fp32_sdpa_reference(self):
        with mock.patch.object(self.backend, "_ROCM_MHA_V4_ENABLED", True):
            out = self.impl._forward_bf16(self.q, self.k, self.v)
        torch.testing.assert_close(
            out.float(), self._sdpa_reference(), atol=3e-2, rtol=3e-2
        )

    def test_flag_off_keeps_the_fmha_v3_path(self):
        with mock.patch.object(self.backend, "_ROCM_MHA_V4_ENABLED", False):
            with mock.patch.object(
                self.backend, "_flash", side_effect=AssertionError("v4 used")
            ):
                out = self.impl._forward_bf16(self.q, self.k, self.v)
        torch.testing.assert_close(
            out.float(), self._sdpa_reference(), atol=3e-2, rtol=3e-2
        )


if __name__ == "__main__":
    unittest.main()
