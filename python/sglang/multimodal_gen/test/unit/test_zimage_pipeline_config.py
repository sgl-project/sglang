import unittest
from contextlib import contextmanager
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.multimodal_gen.configs.pipeline_configs.zimage import ZImagePipelineConfig
from sglang.multimodal_gen.runtime.models.dits.zimage import (
    FeedForward,
    RopeEmbedder,
    ZImageAttention,
    ZImageRMSNorm,
    ZImageTransformer2DModel,
)


@contextmanager
def _single_process_parallel():
    """Stub the TP / SP world so parallel-linear modules (ZImageAttention,
    FeedForward) construct on the 'meta' device."""
    fake_tp_group = SimpleNamespace(world_size=1, rank_in_group=0)
    with (
        patch(
            "sglang.multimodal_gen.runtime.layers.attention.layer.get_ring_parallel_world_size",
            return_value=1,
        ),
        patch(
            "sglang.multimodal_gen.runtime.layers.linear.get_tp_group",
            return_value=fake_tp_group,
        ),
        patch(
            "sglang.multimodal_gen.runtime.models.dits.zimage.get_tp_world_size",
            return_value=1,
        ),
    ):
        yield


class TestZImageSharedPrimitiveDefaults(unittest.TestCase):
    """Shared primitive defaults preserve Z-Image behavior."""

    def test_rope_default_reproduces_the_pre_hook_fp32_formula_bitwise(self) -> None:
        axes_dims, axes_lens, theta = (16, 56, 56), (64, 128, 128), 256.0

        cos_list, sin_list = RopeEmbedder.precompute_freqs(
            axes_dims, axes_lens, theta=theta
        )

        for i, (d, e) in enumerate(zip(axes_dims, axes_lens)):
            inv = 1.0 / (theta ** (torch.arange(0, d, 2, dtype=torch.float64) / d))
            phase = torch.outer(torch.arange(e, dtype=torch.float64), inv).float()
            self.assertTrue(torch.equal(cos_list[i], torch.cos(phase)))
            self.assertTrue(torch.equal(sin_list[i], torch.sin(phase)))

    def test_rope_fp64_phase_is_opt_in_and_actually_differs(self) -> None:
        args = ((16, 56, 56), (64, 128, 128))
        cos32, _ = RopeEmbedder.precompute_freqs(*args)
        cos64, _ = RopeEmbedder.precompute_freqs(*args, freqs_dtype=torch.float64)

        self.assertEqual(cos32[0].dtype, torch.float32)
        self.assertEqual(cos64[0].dtype, torch.float32)
        self.assertFalse(all(torch.equal(a, b) for a, b in zip(cos32, cos64)))

    def test_attention_defaults_keep_zimage_norm_and_fusion(self) -> None:
        with _single_process_parallel(), torch.device("meta"):
            attn = ZImageAttention(dim=64, num_heads=4, num_kv_heads=4)

        self.assertIsInstance(attn.norm_q, ZImageRMSNorm)
        self.assertIsInstance(attn.norm_k, ZImageRMSNorm)
        self.assertTrue(attn.enable_zimage_qk_fusion)

    def test_allow_fused_qk_norm_rope_only_ever_disables(self) -> None:
        with _single_process_parallel(), torch.device("meta"):
            off = ZImageAttention(
                dim=64, num_heads=4, num_kv_heads=4, allow_fused_qk_norm_rope=False
            )
        self.assertFalse(off.enable_zimage_qk_fusion)

    def test_feed_forward_defaults_to_the_fused_silu(self) -> None:
        from sglang.multimodal_gen.runtime.layers.activation import SiluAndMul

        with _single_process_parallel(), torch.device("meta"):
            ff = FeedForward(dim=64, hidden_dim=128)

        self.assertIsInstance(ff.act, SiluAndMul)


class TestZImagePipelineConfig(unittest.TestCase):
    def test_rmsnorm_native_formula(self) -> None:
        norm = ZImageRMSNorm(4, eps=1e-5)
        with torch.no_grad():
            norm.weight.copy_(torch.tensor([1.0, 0.5, 1.5, 2.0]))
        x = torch.tensor(
            [[1.25, 0.5, -0.75, 3.0], [0.1, 2.3, -4.1, 0.7]],
            dtype=torch.bfloat16,
        )

        output = norm(x)
        expected = x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + 1e-5)
        expected = expected * norm.weight.to(dtype=x.dtype)

        self.assertEqual(output.dtype, x.dtype)
        self.assertTrue(torch.equal(output, expected))

    def test_explicit_sigmas(self) -> None:
        """Z-Image uses the native explicit flow sigmas schedule."""
        config = ZImagePipelineConfig()

        self.assertEqual(
            config.prepare_sigmas(None, 4).tolist(),
            [1.0, 0.75, 0.5, 0.25],
        )

    def test_autocast_disabled(self) -> None:
        """Official Z-Image runs bf16 weights without an outer autocast context."""
        self.assertFalse(ZImagePipelineConfig().enable_autocast)

    @patch("sglang.multimodal_gen.configs.pipeline_configs.zimage.get_sp_world_size")
    def test_image_rope_patch_tokens(self, mock_get_sp_world_size) -> None:
        mock_get_sp_world_size.return_value = 1

        config = ZImagePipelineConfig()
        config.vae_config.post_init()
        batch = SimpleNamespace(
            prompt_embeds=[torch.ones(113, 2560)],
            prompt_seq_lens=[[113]],
            negative_prompt_embeds=None,
            height=480,
            width=640,
        )

        def rotary_emb(pos_ids):
            return pos_ids

        _, image_pos_ids = config.prepare_pos_cond_kwargs(
            batch=batch,
            device=torch.device("cpu"),
            rotary_emb=rotary_emb,
            dtype=torch.float32,
        )["freqs_cis"]

        self.assertEqual(image_pos_ids.shape, (1216, 3))
        self.assertEqual(image_pos_ids[0].tolist(), [129, 0, 0])
        self.assertEqual(image_pos_ids[1199].tolist(), [129, 29, 39])
        self.assertEqual(image_pos_ids[-1].tolist(), [0, 0, 0])

    @patch("sglang.multimodal_gen.configs.pipeline_configs.zimage.get_sp_world_size")
    def test_negative_rope_len(self, mock_get_sp_world_size) -> None:
        """Negative CFG branch should build RoPE positions from negative prompt embeds."""
        mock_get_sp_world_size.return_value = 1

        config = ZImagePipelineConfig()
        pos_seq_len = 19
        neg_seq_len = 45
        batch = SimpleNamespace(
            prompt_embeds=[torch.ones(pos_seq_len, 2560)],
            prompt_seq_lens=[[pos_seq_len]],
            negative_prompt_embeds=[torch.ones(neg_seq_len, 2560)],
            negative_prompt_seq_lens=[[neg_seq_len]],
            height=16,
            width=16,
        )

        def rotary_emb(pos_ids):
            return pos_ids

        neg_kwargs = config.prepare_neg_cond_kwargs(
            batch=batch,
            device=torch.device("cpu"),
            rotary_emb=rotary_emb,
            dtype=torch.float32,
        )

        cap_pos_ids, image_pos_ids = neg_kwargs["freqs_cis"]
        neg_cap_padded_len = 64
        self.assertEqual(cap_pos_ids.shape, (neg_cap_padded_len, 3))
        self.assertEqual(image_pos_ids[0].tolist(), [neg_cap_padded_len + 1, 0, 0])

    def test_batched_rope_offsets(self) -> None:
        model = ZImageTransformer2DModel.__new__(ZImageTransformer2DModel)

        def rotary_emb(pos_ids):
            return (
                pos_ids.to(torch.float32),
                (pos_ids + 1000).to(torch.float32),
            )

        model.rotary_emb = rotary_emb

        images = [torch.zeros(16, 1, 60, 80), torch.zeros(16, 1, 60, 80)]
        cap_feats = [torch.zeros(113, 2560), torch.zeros(177, 2560)]

        cap_freqs, image_freqs = model._build_batched_freqs_cis(
            images,
            cap_feats,
            patch_size=2,
            f_patch_size=1,
            image_target_len=1216,
            cap_target_len=192,
        )

        self.assertEqual(cap_freqs[0].shape, (2, 192, 3))
        self.assertEqual(image_freqs[0].shape, (2, 1216, 3))
        self.assertEqual(image_freqs[0][0, 0].tolist(), [129.0, 0.0, 0.0])
        self.assertEqual(image_freqs[0][1, 0].tolist(), [193.0, 0.0, 0.0])
        self.assertEqual(cap_freqs[0][0, 127].tolist(), [128.0, 0.0, 0.0])
        self.assertEqual(cap_freqs[0][0, 128].tolist(), [0.0, 0.0, 0.0])


if __name__ == "__main__":
    unittest.main()
