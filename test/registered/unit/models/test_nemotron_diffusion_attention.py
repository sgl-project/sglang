import math
import unittest
from unittest.mock import patch

import torch
from torch import nn

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase, maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()

from sglang.srt.configs.nemotron_labs_diffusion import (  # noqa: E402
    NemotronLabsDiffusionConfig,
)
from sglang.srt.models.llama import LlamaAttention  # noqa: E402
from sglang.srt.models.nemotron_labs_diffusion import (  # noqa: E402
    NemotronLabsDiffusionAttention,
)

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class Projection(nn.Linear):
    def forward(self, x):
        return super().forward(x), None


class IdentityRoPE(nn.Module):
    def forward(self, positions, q, k):
        return q, k


class TestNemotronQueryScaling(CustomTestCase):
    def test_q_scale_uses_original_context_boundary_and_leaves_kv_unchanged(self):
        config = NemotronLabsDiffusionConfig(
            max_position_embeddings=262144,
            rope_parameters={
                "rope_type": "yarn",
                "factor": 16,
                "llama_4_scaling_beta": 0.1,
                "original_max_position_embeddings": 16384,
            },
        )

        def initialize(attention, **kwargs):
            nn.Module.__init__(attention)
            attention.num_heads = 2
            attention.num_kv_heads = 1
            attention.head_dim = 4
            attention.scaling = 0.5
            attention.attn = nn.Identity()
            attention.attn.layer_id = 0

        with patch.object(LlamaAttention, "__init__", initialize):
            attention = NemotronLabsDiffusionAttention(config=config)
        attention.q_size, attention.kv_size = 8, 4
        attention.qkv_proj = Projection(4, 16, bias=False)
        with torch.no_grad():
            attention.qkv_proj.weight.copy_(torch.arange(64).reshape(16, 4) / 64)
        attention.rotary_emb = IdentityRoPE()
        positions = torch.tensor([0, 16383, 16384, 32768, 262144])
        hidden = torch.arange(20).reshape(5, 4).float() / 20
        baseline = nn.functional.linear(hidden, attention.qkv_proj.weight)
        q, k, v = attention.forward_prepare_native(positions, hidden)
        scale = torch.tensor(
            [1 + 0.1 * math.log1p(p // 16384) for p in positions.tolist()]
        )
        torch.testing.assert_close(q, baseline[:, :8] * scale[:, None])
        torch.testing.assert_close(k, baseline[:, 8:12])
        torch.testing.assert_close(v, baseline[:, 12:])


if __name__ == "__main__":
    unittest.main()
