import unittest

import torch

from sglang.kernels.ops.attention.fused_qk_rmsnorm_rope_gate import (
    fused_qk_gemma_rmsnorm_rope_gate,
)
from sglang.srt.layers.rotary_embedding.mrope import MRotaryEmbedding
from sglang.srt.server_args import ServerArgs, set_global_server_args_for_scheduler
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=6, stage="base-b-kernel-unit", runner_config="1-gpu-large")


def gemma_rmsnorm(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    dtype = x.dtype
    x = x.float()
    x = x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + eps)
    return (x * (1.0 + weight.float())).to(dtype)


def neox_rope(
    x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, rotary_dim: int
) -> torch.Tensor:
    half = rotary_dim // 2
    rotated, passthrough = x[..., :rotary_dim], x[..., rotary_dim:]
    first, second = rotated[..., :half], rotated[..., half:]
    cos = cos.unsqueeze(1).to(x.dtype)
    sin = sin.unsqueeze(1).to(x.dtype)
    return torch.cat(
        [first * cos - second * sin, second * cos + first * sin, passthrough], dim=-1
    )


class TestFusedQKRMSNormRoPEGate(CustomTestCase):
    def setUp(self):
        set_global_server_args_for_scheduler(ServerArgs(model_path="dummy"))
        torch.manual_seed(0)
        self.tokens = 17
        self.num_q_heads = 8
        self.num_kv_heads = 2
        self.head_dim = 128
        self.rotary_dim = 64
        self.eps = 1e-6
        device, dtype = "cuda", torch.bfloat16
        self.q_gate = torch.randn(
            self.tokens,
            self.num_q_heads * 2 * self.head_dim,
            device=device,
            dtype=dtype,
        )
        self.k = torch.randn(
            self.tokens, self.num_kv_heads * self.head_dim, device=device, dtype=dtype
        )
        self.q_weight = torch.randn(self.head_dim, device=device, dtype=dtype)
        self.k_weight = torch.randn(self.head_dim, device=device, dtype=dtype)
        inv_freq = 10000 ** (
            -torch.arange(self.rotary_dim // 2, device=device).float()
            * 2
            / self.rotary_dim
        )
        angles = torch.arange(512, device=device).float().unsqueeze(1) * inv_freq
        self.cos_sin_cache = torch.cat([angles.cos(), angles.sin()], dim=-1).to(dtype)

    def call(self, positions, cos_sin_cache=None, mrope_axis_map=None, rotary_dim=None):
        if cos_sin_cache is None:
            cos_sin_cache = self.cos_sin_cache
        return fused_qk_gemma_rmsnorm_rope_gate(
            self.q_gate,
            self.k,
            self.q_weight,
            self.k_weight,
            cos_sin_cache,
            positions,
            self.eps,
            self.num_q_heads,
            self.num_kv_heads,
            self.head_dim,
            rotary_dim or self.rotary_dim,
            has_gate=True,
            mrope_axis_map=mrope_axis_map,
        )

    def build_mrope(self, mrope_section, interleaved):
        return MRotaryEmbedding(
            head_size=self.head_dim,
            rotary_dim=2 * sum(mrope_section),
            max_position_embeddings=512,
            base=10000,
            is_neox_style=True,
            dtype=torch.bfloat16,
            mrope_section=mrope_section,
            mrope_interleaved=interleaved,
        ).to("cuda")

    def graph_buffer_positions(self):
        buffer = torch.zeros(3, 4 * self.tokens, dtype=torch.int64, device="cuda")
        buffer[:, : self.tokens] = torch.stack(
            [
                torch.arange(self.tokens, device="cuda") % 7,
                torch.arange(self.tokens, device="cuda") % 5 + 3,
                torch.arange(self.tokens, device="cuda") % 3 + 11,
            ]
        )
        return buffer[:, : self.tokens]

    def test_matches_reference_for_1d_positions(self):
        positions = torch.arange(self.tokens, device="cuda", dtype=torch.int64)
        q_out, k_out, gate_out = self.call(positions)

        packed = self.q_gate.view(self.tokens, self.num_q_heads, 2 * self.head_dim)
        cos, sin = self.cos_sin_cache[positions].chunk(2, dim=-1)
        want_q = neox_rope(
            gemma_rmsnorm(packed[..., : self.head_dim], self.q_weight, self.eps),
            cos,
            sin,
            self.rotary_dim,
        )
        want_k = neox_rope(
            gemma_rmsnorm(
                self.k.view(self.tokens, self.num_kv_heads, self.head_dim),
                self.k_weight,
                self.eps,
            ),
            cos,
            sin,
            self.rotary_dim,
        )
        torch.testing.assert_close(
            q_out.view_as(want_q).float(), want_q.float(), atol=2e-2, rtol=2e-2
        )
        torch.testing.assert_close(
            k_out.view_as(want_k).float(), want_k.float(), atol=2e-2, rtol=2e-2
        )
        torch.testing.assert_close(
            gate_out.float(), packed[..., self.head_dim :].float(), atol=0, rtol=0
        )

    def test_mrope_matches_the_rotary_module(self):
        """With t == h == w every layout agrees, so only distinct rows catch a wrong
        axis. Interleaved [11, 11, 10] is what Qwen3.6-35B-A3B ships, and [24, 20, 20]
        fills the head dimension, leaving no pass-through tail.
        """
        for section, interleaved in (
            ([11, 11, 10], False),
            ([11, 11, 10], True),
            ([24, 20, 20], True),
        ):
            with self.subTest(section=section, interleaved=interleaved):
                rope = self.build_mrope(section, interleaved)
                positions = self.graph_buffer_positions()

                q_in = self.q_gate.view(
                    self.tokens, self.num_q_heads, 2 * self.head_dim
                )[..., : self.head_dim].reshape(self.tokens, -1)
                want_q, want_k = rope.forward_native(
                    positions,
                    gemma_rmsnorm(
                        q_in.view(self.tokens, self.num_q_heads, self.head_dim),
                        self.q_weight,
                        self.eps,
                    ).reshape(self.tokens, -1),
                    gemma_rmsnorm(
                        self.k.view(self.tokens, self.num_kv_heads, self.head_dim),
                        self.k_weight,
                        self.eps,
                    ).reshape(self.tokens, -1),
                )

                q_out, k_out, _gate = self.call(
                    positions,
                    cos_sin_cache=rope.cos_sin_cache,
                    mrope_axis_map=rope.axis_map,
                    rotary_dim=rope.rotary_dim,
                )
                torch.testing.assert_close(
                    q_out.float(), want_q.float(), atol=2e-2, rtol=2e-2
                )
                torch.testing.assert_close(
                    k_out.float(), want_k.float(), atol=2e-2, rtol=2e-2
                )

    def test_rejects_positions_and_map_apart(self):
        flat = torch.arange(self.tokens, device="cuda", dtype=torch.int64)
        axis_map = self.build_mrope([11, 11, 10], interleaved=True).axis_map
        cases = ((flat.unsqueeze(0).repeat(3, 1), None), (flat, axis_map))
        for positions, axis_map_passed in cases:
            with self.subTest(mrope_positions=positions.dim() == 2):
                with self.assertRaises(AssertionError):
                    self.call(positions, mrope_axis_map=axis_map_passed)


if __name__ == "__main__":
    unittest.main()
