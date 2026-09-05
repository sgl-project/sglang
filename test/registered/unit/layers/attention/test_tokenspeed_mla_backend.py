# Copyright 2023-2026 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

import unittest

import torch

from sglang.srt.layers.attention import tokenspeed_mla_backend as tokenspeed_backend
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


class TestTokenspeedMlaBackend(unittest.TestCase):
    def test_skip_rope_fp8_quantize_packs_qk_without_rope(self):
        original_quantize = tokenspeed_backend.mla_quantize_without_rope_for_fp8
        captured = {}

        def fake_quantize(q_nope_tensor, q_pe_tensor, k_nope_tensor, k_pe_tensor):
            captured["k_pe"] = k_pe_tensor
            return (
                torch.cat((q_nope_tensor, q_pe_tensor), dim=-1),
                k_nope_tensor,
                k_pe_tensor,
            )

        tokenspeed_backend.mla_quantize_without_rope_for_fp8 = fake_quantize

        try:
            q_nope = torch.arange(12, dtype=torch.float32).view(2, 2, 3)
            q_pe = torch.arange(8, dtype=torch.float32).view(2, 2, 2) + 100
            k_nope = torch.arange(12, dtype=torch.float32).view(2, 2, 3) + 200
            k_pe = torch.arange(4, dtype=torch.float32).view(2, 1, 2) + 300

            q_out, k_out = (
                tokenspeed_backend.TokenspeedMLABackend._fused_rope_fp8_quantize(
                    object(),
                    q_nope=q_nope,
                    q_pe=q_pe,
                    k_nope=k_nope,
                    k_pe=k_pe,
                    cos_sin_cache=None,
                    positions=torch.tensor([0, 1], dtype=torch.int32),
                    is_neox=True,
                    qk_nope_head_dim=3,
                    qk_rope_head_dim=2,
                )
            )
        finally:
            tokenspeed_backend.mla_quantize_without_rope_for_fp8 = original_quantize

        expected_q = torch.cat((q_nope, q_pe), dim=-1)
        expected_k_pe = k_pe.expand(-1, q_nope.shape[1], -1)
        expected_k = torch.cat((k_nope, expected_k_pe), dim=-1)
        torch.testing.assert_close(captured["k_pe"], expected_k_pe)
        torch.testing.assert_close(q_out, expected_q)
        torch.testing.assert_close(k_out, expected_k)


if __name__ == "__main__":
    unittest.main()
