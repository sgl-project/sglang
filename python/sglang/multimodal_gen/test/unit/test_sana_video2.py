# SPDX-License-Identifier: Apache-2.0

import unittest

import torch


class TestSanaVideo2(unittest.TestCase):
    def test_linear_attention_matches_explicit_token_sum(self):
        from sglang.multimodal_gen.runtime.models.dits.sana_video2 import (
            GatedLinearAttention,
        )

        torch.manual_seed(12)
        module = GatedLinearAttention(24, 12, fp32_attention=True).eval()
        x = torch.randn(2, 5, 24)
        q, k, v = module.qkv(x).chunk(3, dim=-1)
        q = module.q_norm(q).reshape(2, 5, 2, 12)
        k = module.k_norm(k).reshape(2, 5, 2, 12)
        v = v.reshape(2, 5, 2, 12)
        beta = module.beta_proj(x).sigmoid()
        scores = torch.einsum("bihd,bjhd->bhij", q, k)
        out = torch.einsum("bhij,bjh,bjhd->bihd", scores, beta, v)
        out = out.float() * (out.float().square().mean(-1, keepdim=True) + 1e-5).rsqrt()
        out = out * module.o_norm.weight
        out = out.reshape_as(x) * module.output_gate(x).sigmoid()
        expected = module.proj(out)
        torch.testing.assert_close(module(x), expected, atol=2e-6, rtol=2e-5)

    def test_softmax_attention_matches_dense_equation(self):
        from sglang.multimodal_gen.runtime.models.dits.sana_video2 import (
            GatedSoftmaxAttention,
        )

        torch.manual_seed(13)
        module = GatedSoftmaxAttention(24, 12, fp32_attention=True).eval()
        x = torch.randn(2, 5, 24)
        q, k, v = module.qkv(x).chunk(3, -1)
        q = module.q_norm(q).reshape(2, 5, 2, 12)
        k = module.k_norm(k).reshape(2, 5, 2, 12)
        v = v.reshape(2, 5, 2, 12)
        scores = torch.einsum("bihd,bjhd->bhij", q, k) / 12**0.5
        out = torch.einsum("bhij,bjhd->bihd", scores.softmax(-1), v)
        expected = module.proj(out.reshape_as(x) * module.output_gate(x).sigmoid())
        torch.testing.assert_close(module(x), expected)

    def test_attention_residual_weights_depth_per_token(self):
        from sglang.multimodal_gen.runtime.models.dits.sana_video2 import (
            BlockAttentionResidual,
        )

        torch.manual_seed(14)
        module = BlockAttentionResidual(4)
        with torch.no_grad():
            module.attn_proj.weight.copy_(torch.tensor([[0.3, -0.4, 0.5, 0.1]]))
        values = torch.randn(4, 2, 3, 4)
        keys = values * (values.square().mean(-1, keepdim=True) + 1e-6).rsqrt()
        scores = (keys * module.attn_proj.weight).sum(-1).softmax(0)
        expected = (scores.unsqueeze(-1) * values).sum(0)
        actual = module.attend_buffer(
            module.attn_proj, values.clone(), keys.clone(), 3, values[3]
        )
        torch.testing.assert_close(actual, expected)

    @staticmethod
    def small_model():
        from sglang.multimodal_gen.configs.models.dits.sana_video2 import (
            SanaVideo2ArchConfig,
            SanaVideo2Config,
        )
        from sglang.multimodal_gen.runtime.models.dits.sana_video2 import (
            SanaVideo2Transformer3DModel,
        )

        return SanaVideo2Transformer3DModel(
            SanaVideo2Config(
                arch_config=SanaVideo2ArchConfig(
                    hidden_size=24,
                    depth=4,
                    num_heads=2,
                    linear_head_dim=12,
                    softmax_head_dim=24,
                    in_channels=4,
                    caption_channels=8,
                    model_max_length=5,
                    mlp_ratio=2,
                    attn_res_block_size=2,
                )
            )
        ).eval()

    def test_uniform_frame_timesteps_equal_batch_timesteps(self):
        torch.manual_seed(15)
        model = self.small_model()
        x, y = torch.randn(2, 4, 3, 2, 2), torch.randn(2, 5, 8)
        t = torch.tensor([42.9, 123.9])
        mask = torch.tensor([[1, 1, 1, 0, 0], [1, 1, 1, 1, 0]])
        with torch.no_grad():
            batch = model(x, t, y, mask)
            frames = model(
                x, t[:, None, None, None, None].expand(-1, 1, 3, 1, 1), y[:, None], mask
            )
            changed_t = t[:, None, None, None, None].expand(-1, 1, 3, 1, 1).clone()
            changed_t[:, :, 0] = 0
            changed = model(x, changed_t, y, mask)
        self.assertEqual(batch.shape, x.shape)
        torch.testing.assert_close(batch, frames)
        self.assertFalse(torch.allclose(batch, changed))
        self.assertEqual(
            model.block_attention_types, ["linear", "linear", "linear", "softmax"]
        )

    def test_padding_tokens_do_not_change_flow_and_calls_do_not_share_shape(self):
        torch.manual_seed(16)
        model = self.small_model()
        x, y, t = torch.randn(1, 4, 2, 2, 2), torch.randn(1, 5, 8), torch.tensor([20.0])
        mask = torch.tensor([[1, 1, 1, 0, 0]])
        with torch.no_grad():
            original = model(x, t, y, mask)
            y[:, 3:] = 1000
            model(torch.randn(1, 4, 1, 1, 3), t, y, mask)
            actual = model(x, t, y, mask)
        torch.testing.assert_close(actual, original)

    def test_dtype_cast_preserves_complex_rotary_frequencies(self):
        model = self.small_model()
        before = model.rope_linear((2, 2, 2), torch.device("cpu"))
        model.to(dtype=torch.bfloat16)
        after = model.rope_linear((2, 2, 2), torch.device("cpu"))
        self.assertTrue(after.is_complex())
        torch.testing.assert_close(after, before, rtol=0, atol=0)

    def test_official_checkpoint_names_and_output_sign(self):
        torch.manual_seed(17)
        model = self.small_model()
        state = model.state_dict()
        for key in (
            "pos_embed",
            "y_embedder.y_embedding",
            "blocks.0.attn.beta_proj.weight",
            "blocks.3.attn.output_gate.weight",
            "blocks.0.cross_attn.kv_linear.weight",
            "blocks.0.mlp.gate_proj.weight",
            "attn_res.final_proj.weight",
        ):
            self.assertIn(key, state)
        model.load_state_dict(state, strict=True)
        with torch.no_grad():
            model.final_layer.linear.weight.zero_()
            model.final_layer.linear.bias.fill_(2)
            output = model(
                torch.randn(1, 4, 2, 2, 2), torch.tensor([1.0]), torch.randn(1, 5, 8)
            )
        torch.testing.assert_close(output, torch.full_like(output, 2))


if __name__ == "__main__":
    unittest.main()
