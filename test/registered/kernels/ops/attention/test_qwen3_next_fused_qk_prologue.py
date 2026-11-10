import unittest

import torch
from torch import nn

from sglang.srt.layers.layernorm import GemmaRMSNorm
from sglang.srt.layers.rotary_embedding import RotaryEmbedding, get_rope
from sglang.srt.models.qwen3_next import (
    Qwen3HybridAttentionDecoderLayer,
    _supports_triton_prologue,
)
from sglang.srt.server_args import ServerArgs, set_global_server_args_for_scheduler
from sglang.srt.utils import is_hip
from sglang.test.ci.ci_register import register_amd_ci
from sglang.test.test_utils import CustomTestCase

register_amd_ci(est_time=30, stage="jit-kernel-unit", runner_config="amd")

HEAD_DIM = 256
# Qwen3-Next-80B-A3B: 16 q heads, 2 kv heads; per-rank widths at TP1/TP2/TP4.
HEADS_PER_RANK = [(16, 2), (8, 1), (4, 1)]
NUM_TOKENS = [1, 7, 64, 384, 8192]
# The RMSNorm reduction sums in a different order than the native GemmaRMSNorm,
# so a rare normalized value rounds to the neighboring bf16 and RoPE carries the
# difference: measured on MI355X, at most 7e-5 of q/k elements differ, by at most
# 2 bf16 ulp of the rotated pair's magnitude. v and gate are plain copies.
MAX_MISMATCH = 1e-3
MAX_PAIR_ULPS = 3


def _bf16_ulp(x: torch.Tensor) -> torch.Tensor:
    _, exp = torch.frexp(x.float().abs().clamp_min(2.0**-126))
    return torch.ldexp(torch.ones_like(exp, dtype=torch.float32), exp - 8)


def _rope_scale(x: torch.Tensor, num_heads: int, rotary_dim: int) -> torch.Tensor:
    # Per element magnitude bound: the norm of the rotated pair, or the value
    # itself on the pass-through lanes.
    x = x.float().reshape(x.shape[0], num_heads, HEAD_DIM)
    half = rotary_dim // 2
    pair = torch.hypot(x[..., :half], x[..., half:rotary_dim])
    scale = torch.cat([pair, pair, x[..., rotary_dim:].abs()], dim=-1)
    return scale.reshape(x.shape[0], -1)


def _make_layer(num_heads: int, num_kv_heads: int) -> Qwen3HybridAttentionDecoderLayer:
    layer = Qwen3HybridAttentionDecoderLayer.__new__(Qwen3HybridAttentionDecoderLayer)
    nn.Module.__init__(layer)
    layer.num_heads = num_heads
    layer.num_kv_heads = num_kv_heads
    layer.head_dim = HEAD_DIM
    layer.q_size = num_heads * HEAD_DIM
    layer.kv_size = num_kv_heads * HEAD_DIM
    layer.attn_output_gate = True
    layer.alt_stream = None
    layer.rotary_emb = get_rope(
        head_size=HEAD_DIM,
        rotary_dim=HEAD_DIM,
        max_position=32768,
        base=10000000,
        partial_rotary_factor=0.25,
        is_neox_style=True,
        dtype=torch.bfloat16,
    ).cuda()
    layer.q_norm = GemmaRMSNorm(HEAD_DIM, eps=1e-6).to("cuda", torch.bfloat16)
    layer.k_norm = GemmaRMSNorm(HEAD_DIM, eps=1e-6).to("cuda", torch.bfloat16)
    with torch.no_grad():
        layer.q_norm.weight.normal_(0, 0.5)
        layer.k_norm.weight.normal_(0, 0.5)
    return layer


@unittest.skipUnless(torch.cuda.is_available() and is_hip(), "ROCm only")
class TestQwen3NextFusedQKPrologue(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        set_global_server_args_for_scheduler(ServerArgs(model_path="dummy"))

    def test_matches_native(self):
        for num_heads, num_kv_heads in HEADS_PER_RANK:
            layer = _make_layer(num_heads, num_kv_heads)
            if not layer.rotary_emb.use_fallback_kernel:
                self.skipTest("the Triton prologue mirrors the fallback rotary kernel")
            width = layer.q_size * 2 + layer.kv_size * 2
            for num_tokens in NUM_TOKENS:
                with self.subTest(heads=num_heads, kv_heads=num_kv_heads, T=num_tokens):
                    torch.manual_seed(num_tokens)
                    qkv = torch.randn(
                        num_tokens, width, device="cuda", dtype=torch.bfloat16
                    )
                    positions = torch.randint(0, 32768, (num_tokens,), device="cuda")
                    q, k, v, gate = layer.forward_prepare_triton(positions, qkv)
                    q_ref, k_ref, v_ref, gate_ref = layer.forward_prepare_native(
                        positions, qkv.clone()
                    )
                    torch.testing.assert_close(v, v_ref, atol=0, rtol=0)
                    torch.testing.assert_close(gate, gate_ref, atol=0, rtol=0)
                    rotary_dim = layer.rotary_emb.rotary_dim
                    for name, out, ref, heads in (
                        ("q", q, q_ref, num_heads),
                        ("k", k, k_ref, num_kv_heads),
                    ):
                        self.assertEqual(out.shape, ref.shape, name)
                        mismatch = (out != ref).float().mean().item()
                        self.assertLessEqual(mismatch, MAX_MISMATCH, name)
                        diff = (out.float() - ref.float()).abs()
                        ulps = diff / _bf16_ulp(_rope_scale(ref, heads, rotary_dim))
                        self.assertLessEqual(ulps.max().item(), MAX_PAIR_ULPS, name)

    def test_empty_batch(self):
        num_heads, num_kv_heads = HEADS_PER_RANK[1]
        layer = _make_layer(num_heads, num_kv_heads)
        qkv = torch.empty(
            0, layer.q_size * 2 + layer.kv_size * 2, device="cuda", dtype=torch.bfloat16
        )
        positions = torch.empty(0, dtype=torch.int64, device="cuda")
        q, k, v, gate = layer.forward_prepare_triton(positions, qkv)
        self.assertEqual(q.shape, (0, layer.q_size))
        self.assertEqual(k.shape, (0, layer.kv_size))
        self.assertEqual(v.shape, (0, layer.kv_size))
        self.assertEqual(gate.shape, (0, layer.q_size))

    def test_rl_on_policy_keeps_native_prologue(self):
        """With --rl-on-policy-target, RoPE runs its native path to match the
        trainer bitwise; the Triton prologue must not replace it."""

        def make_rope():
            # Built directly: get_rope caches modules across server args. The
            # cuda device context matches how the model loader builds it.
            with torch.device("cuda"):
                return RotaryEmbedding(
                    head_size=HEAD_DIM,
                    rotary_dim=HEAD_DIM // 4,
                    max_position_embeddings=32768,
                    base=10000000,
                    is_neox_style=True,
                    dtype=torch.bfloat16,
                )

        if not _supports_triton_prologue(
            rope=make_rope(), head_dim=HEAD_DIM, attn_output_gate=True
        ):
            self.skipTest("the Triton prologue is not used on this device")
        try:
            set_global_server_args_for_scheduler(
                ServerArgs(model_path="dummy", rl_on_policy_target="fsdp")
            )
            rope = make_rope()
        finally:
            set_global_server_args_for_scheduler(ServerArgs(model_path="dummy"))
        self.assertFalse(
            _supports_triton_prologue(
                rope=rope, head_dim=HEAD_DIM, attn_output_gate=True
            )
        )


if __name__ == "__main__":
    unittest.main()
