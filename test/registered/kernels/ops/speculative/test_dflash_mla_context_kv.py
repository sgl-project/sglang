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
"""DFLASH's fused MLA context KV must match the per-layer path it replaces.

python -m pytest test/registered/kernels/ops/speculative/test_dflash_mla_context_kv.py -v
"""

import unittest
from types import SimpleNamespace

import torch
import torch.nn.functional as F

from sglang.test.ci.ci_register import register_amd_ci
from sglang.test.test_utils import CustomTestCase

register_amd_ci(est_time=30, stage="jit-kernel-unit", runner_config="amd")


class TestMlaContextKV(CustomTestCase):
    def test_matches_the_per_layer_projection(self):
        """Stacking the layers must not mix one layer's kv_a rows, norm or RoPE into another's."""
        from sglang.kernels.ops.speculative.dflash_mla_ctx import MlaContextKV
        from sglang.srt.layers.layernorm import RMSNorm
        from sglang.srt.layers.rotary_embedding import get_rope

        torch.manual_seed(0)
        hidden, q_lora, kv_lora, rope_dim, layers, n = 1024, 64, 512, 64, 5, 7
        rotary_emb = get_rope(
            rope_dim, rope_dim, 4096, 50000, is_neox_style=False, dtype=torch.bfloat16
        )
        attns = []
        for _ in range(layers):
            norm = RMSNorm(kv_lora, eps=1e-5).cuda().to(torch.bfloat16)
            norm.weight.data.uniform_(0.5, 1.5)
            weight = (
                torch.randn(
                    q_lora + kv_lora + rope_dim,
                    hidden,
                    device="cuda",
                    dtype=torch.bfloat16,
                )
                * 0.02
            )
            attns.append(
                SimpleNamespace(
                    kv_lora_rank=kv_lora,
                    qk_rope_head_dim=rope_dim,
                    q_lora_rank=q_lora,
                    fused_qkv_a_proj_with_mqa=SimpleNamespace(weight=weight),
                    kv_a_layernorm=norm,
                    rotary_emb=rotary_emb.cuda(),
                )
            )
        ctx = MlaContextKV(attns)
        rows = torch.randn(n, hidden, device="cuda", dtype=torch.bfloat16)
        positions = torch.randint(0, 4000, (n,), device="cuda")
        k_nope, k_pe = ctx.project_all(positions, rows)
        for i, attn in enumerate(attns):
            latent = F.linear(rows, attn.fused_qkv_a_proj_with_mqa.weight[q_lora:])
            want_nope = attn.kv_a_layernorm(latent[..., :kv_lora]).unsqueeze(1)
            want_pe = latent[..., kv_lora:].unsqueeze(1).contiguous()
            _, want_pe = attn.rotary_emb(positions, torch.empty_like(want_pe), want_pe)
            with self.subTest(layer=i):
                # one bf16 ulp: the GEMM's tiling and the norm's rounding order may differ
                torch.testing.assert_close(k_nope[i], want_nope, atol=1e-2, rtol=8e-3)
                torch.testing.assert_close(
                    k_pe[i], want_pe.view_as(k_pe[i]), atol=1e-2, rtol=8e-3
                )


if __name__ == "__main__":
    unittest.main()
