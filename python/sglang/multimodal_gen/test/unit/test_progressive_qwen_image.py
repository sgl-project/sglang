# SPDX-License-Identifier: Apache-2.0
"""
CPU-only unit tests for Qwen-Image progressive resolution growing.

Covers:
  - _qwen_image_pack / _qwen_image_unpack: shape, roundtrip, dtype, ordering
  - QwenImageProgressiveDenoisingStage: inheritance, spectrum constants,
    _unpack_latent / _repack_latent hooks, _on_resolution_change
  - QwenImageProgressivePipeline: class attributes and inheritance

No GPU or model checkpoint is required.  All tests pass in < 10 s on CPU.
"""

import unittest
from types import SimpleNamespace
from typing import Any

import torch

from sglang.multimodal_gen.runtime.pipelines.qwen_image import QwenImagePipeline
from sglang.multimodal_gen.runtime.pipelines.qwen_image_progressive import (
    QWEN_IMAGE_SPECTRUM_A,
    QWEN_IMAGE_SPECTRUM_BETA,
    QwenImageProgressiveDenoisingStage,
    _qwen_image_pack,
    _qwen_image_unpack,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.progressive_resolution.denoising import (
    ProgressiveDenoisingStage,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_packed(B: int, h_lat: int, w_lat: int) -> torch.Tensor:
    """Return a random packed latent [B, S, 64]."""
    S = (h_lat // 2) * (w_lat // 2)
    return torch.randn(B, S, 64)


def _make_spatial(B: int, h_lat: int, w_lat: int) -> torch.Tensor:
    """Return a random spatial latent [B, 16, H_lat, W_lat]."""
    return torch.randn(B, 16, h_lat, w_lat)


# ---------------------------------------------------------------------------
# Pack / Unpack correctness
# ---------------------------------------------------------------------------


class TestQwenImagePack(unittest.TestCase):
    """Shape, roundtrip, and ordering tests for _qwen_image_pack / _unpack."""

    # ---- shape tests -------------------------------------------------------

    def test_unpack_output_shape(self):
        for B, H, W in [(1, 8, 8), (2, 16, 16), (1, 8, 16)]:
            with self.subTest(B=B, H=H, W=W):
                packed = _make_packed(B, H, W)
                out = _qwen_image_unpack(packed, H, W)
                self.assertEqual(out.shape, (B, 16, H, W))

    def test_pack_output_shape(self):
        for B, H, W in [(1, 8, 8), (2, 16, 16), (1, 8, 16)]:
            with self.subTest(B=B, H=H, W=W):
                spatial = _make_spatial(B, H, W)
                out = _qwen_image_pack(spatial, H, W)
                S = (H // 2) * (W // 2)
                self.assertEqual(out.shape, (B, S, 64))

    def test_pack_sequence_length(self):
        """S = (H_lat/2) * (W_lat/2): one token per 2×2 patch."""
        for H, W in [(8, 8), (16, 32), (32, 16), (64, 64)]:
            with self.subTest(H=H, W=W):
                spatial = _make_spatial(1, H, W)
                out = _qwen_image_pack(spatial, H, W)
                expected_S = (H // 2) * (W // 2)
                self.assertEqual(out.shape[1], expected_S)

    # ---- roundtrip tests ---------------------------------------------------

    def test_pack_unpack_roundtrip(self):
        """pack ∘ unpack = identity on packed latents."""
        for B, H, W in [(1, 8, 8), (2, 16, 16)]:
            with self.subTest(B=B, H=H, W=W):
                packed = _make_packed(B, H, W)
                reconstructed = _qwen_image_pack(_qwen_image_unpack(packed, H, W), H, W)
                torch.testing.assert_close(reconstructed, packed)

    def test_unpack_pack_roundtrip(self):
        """unpack ∘ pack = identity on spatial latents."""
        for B, H, W in [(1, 8, 8), (2, 16, 16)]:
            with self.subTest(B=B, H=H, W=W):
                spatial = _make_spatial(B, H, W)
                reconstructed = _qwen_image_unpack(
                    _qwen_image_pack(spatial, H, W), H, W
                )
                torch.testing.assert_close(reconstructed, spatial)

    # ---- dtype tests -------------------------------------------------------

    def test_unpack_dtype_preserved(self):
        for dtype in [torch.float32, torch.float16, torch.bfloat16]:
            with self.subTest(dtype=dtype):
                packed = _make_packed(1, 8, 8).to(dtype)
                out = _qwen_image_unpack(packed, 8, 8)
                self.assertEqual(out.dtype, dtype)

    def test_pack_dtype_preserved(self):
        for dtype in [torch.float32, torch.float16, torch.bfloat16]:
            with self.subTest(dtype=dtype):
                spatial = _make_spatial(1, 8, 8).to(dtype)
                out = _qwen_image_pack(spatial, 8, 8)
                self.assertEqual(out.dtype, dtype)

    # ---- spatial ordering --------------------------------------------------

    def test_pack_unpack_channel_ordering(self):
        """Each channel of the spatial tensor maps to a consistent slice of the
        64-dim token vector (channels 0..3, 4..7, 8..11, 12..15 in column order
        for a 2×2 patch).  Verify via a hand-crafted single-patch example."""
        # 1 batch, 1 patch → 2×2 spatial (h_lat=2, w_lat=2)
        spatial = torch.zeros(1, 16, 2, 2)
        for c in range(16):
            spatial[0, c, :, :] = c + 1.0  # channel c → value c+1

        packed = _qwen_image_pack(spatial, 2, 2)
        self.assertEqual(packed.shape, (1, 1, 64))

        recovered = _qwen_image_unpack(packed, 2, 2)
        torch.testing.assert_close(recovered, spatial)

    def test_different_spatial_positions_produce_different_tokens(self):
        """Two spatially distinct patches must produce different token vectors."""
        spatial = torch.randn(1, 16, 4, 4)
        packed = _qwen_image_pack(spatial, 4, 4)
        # packed shape: [1, 4, 64] (2×2 patches)
        self.assertFalse(torch.allclose(packed[0, 0], packed[0, 1]))

    def test_matches_config_pack_latents(self):
        """_qwen_image_pack must match QwenImagePipelineConfig._pack_latents."""
        from sglang.multimodal_gen.configs.pipeline_configs.qwen_image import (
            _pack_latents,
        )

        for B, H, W in [(1, 8, 8), (2, 16, 16)]:
            with self.subTest(B=B, H=H, W=W):
                spatial = _make_spatial(B, H, W)
                expected = _pack_latents(spatial, B, 16, H, W)
                got = _qwen_image_pack(spatial, H, W)
                torch.testing.assert_close(got, expected)


# ---------------------------------------------------------------------------
# QwenImageProgressiveDenoisingStage
# ---------------------------------------------------------------------------


class TestQwenImageProgressiveStage(unittest.TestCase):
    """Tests for QwenImageProgressiveDenoisingStage hooks (no GPU / no model).

    _make_stage returns a lightweight stub that inherits from
    QwenImageProgressiveDenoisingStage but bypasses DenoisingStage.__init__
    (which requires a running server and attention backend).  Only the minimal
    instance attributes that the hooks actually read are set manually.
    """

    def _make_stage(self) -> QwenImageProgressiveDenoisingStage:
        class _StubStage(QwenImageProgressiveDenoisingStage):
            """Bypasses DenoisingStage.__init__ (requires server + attn backend)."""

            def __init__(inner_self):  # noqa: N805
                inner_self.transformer = None
                inner_self.scheduler = None
                inner_self.vae = None
                inner_self.pipeline = None
                inner_self._spectrum_A = QWEN_IMAGE_SPECTRUM_A
                inner_self._spectrum_beta = QWEN_IMAGE_SPECTRUM_BETA

            @property
            def device(inner_self) -> torch.device:  # noqa: N805
                return torch.device("cpu")

        return _StubStage()

    # ---- class hierarchy ---------------------------------------------------

    def test_is_subclass_of_progressive_denoising_stage(self):
        self.assertTrue(
            issubclass(QwenImageProgressiveDenoisingStage, ProgressiveDenoisingStage)
        )

    def test_instantiation(self):
        stage = self._make_stage()
        self.assertIsInstance(stage, QwenImageProgressiveDenoisingStage)

    # ---- spectrum constants ------------------------------------------------

    def test_spectrum_A_positive(self):
        self.assertGreater(QWEN_IMAGE_SPECTRUM_A, 0.0)

    def test_spectrum_beta_in_physical_range(self):
        """beta should be between 1 and 4 for natural image VAE latents."""
        self.assertGreater(QWEN_IMAGE_SPECTRUM_BETA, 1.0)
        self.assertLess(QWEN_IMAGE_SPECTRUM_BETA, 4.0)

    def test_stage_uses_correct_spectrum_constants(self):
        stage = self._make_stage()
        self.assertEqual(stage._spectrum_A, QWEN_IMAGE_SPECTRUM_A)
        self.assertEqual(stage._spectrum_beta, QWEN_IMAGE_SPECTRUM_BETA)

    # ---- _unpack_latent / _repack_latent -----------------------------------

    def test_unpack_latent_shape(self):
        stage = self._make_stage()
        for B, H, W in [(1, 8, 8), (2, 16, 16), (1, 8, 16)]:
            with self.subTest(B=B, H=H, W=W):
                packed = _make_packed(B, H, W)
                out = stage._unpack_latent(packed, H, W)
                self.assertEqual(out.shape, (B, 16, H, W))

    def test_repack_latent_shape(self):
        stage = self._make_stage()
        for B, H, W in [(1, 8, 8), (2, 16, 16)]:
            with self.subTest(B=B, H=H, W=W):
                spatial = _make_spatial(B, H, W)
                out = stage._repack_latent(spatial, H, W, batch=None, server_args=None)
                S = (H // 2) * (W // 2)
                self.assertEqual(out.shape, (B, S, 64))

    def test_unpack_repack_roundtrip(self):
        stage = self._make_stage()
        for B, H, W in [(1, 8, 8), (2, 16, 16)]:
            with self.subTest(B=B, H=H, W=W):
                packed = _make_packed(B, H, W)
                spatial = stage._unpack_latent(packed, H, W)
                repacked = stage._repack_latent(
                    spatial, H, W, batch=None, server_args=None
                )
                torch.testing.assert_close(repacked, packed)

    def test_repack_unpack_roundtrip(self):
        stage = self._make_stage()
        for B, H, W in [(1, 8, 8), (2, 16, 16)]:
            with self.subTest(B=B, H=H, W=W):
                spatial = _make_spatial(B, H, W)
                packed = stage._repack_latent(
                    spatial, H, W, batch=None, server_args=None
                )
                recovered = stage._unpack_latent(packed, H, W)
                torch.testing.assert_close(recovered, spatial)

    # ---- _on_resolution_change: early exits --------------------------------

    def test_on_resolution_change_no_crash_when_cfg_policy_is_none(self):
        """Should return silently when ctx.cfg_policy is None."""
        stage = self._make_stage()
        ctx = SimpleNamespace(cfg_policy=None)
        stage._on_resolution_change(
            ctx, batch=None, server_args=None, new_h_pixel=512, new_w_pixel=512
        )

    # ---- _on_resolution_change: branch update logic ------------------------

    def _make_mock_ctx(
        self,
        freqs_cis_value: Any = "sentinel_freqs",
        img_shapes_value: Any = "sentinel_shapes",
    ) -> SimpleNamespace:
        """Build a minimal DenoisingContext-like namespace with two CFG branches."""
        branch_cond = SimpleNamespace(
            kwargs={"freqs_cis": freqs_cis_value, "img_shapes": img_shapes_value}
        )
        branch_uncond = SimpleNamespace(
            kwargs={"freqs_cis": freqs_cis_value, "img_shapes": img_shapes_value}
        )
        cfg_policy = SimpleNamespace(branches=[branch_cond, branch_uncond])
        return SimpleNamespace(
            cfg_policy=cfg_policy,
            pos_cond_kwargs={
                "freqs_cis": freqs_cis_value,
                "img_shapes": img_shapes_value,
            },
            target_dtype=torch.bfloat16,
        )

    def _make_mock_server_args(
        self,
        new_freqs_cis: Any,
        new_img_shapes: Any,
        vae_scale_factor: int = 8,
    ) -> SimpleNamespace:
        """Build a minimal ServerArgs namespace that returns prescribed kwargs."""

        def prepare_pos_cond_kwargs(batch, device, rotary_emb, dtype):
            return {"freqs_cis": new_freqs_cis, "img_shapes": new_img_shapes}

        vae_arch = SimpleNamespace(vae_scale_factor=vae_scale_factor)
        vae_config = SimpleNamespace(arch_config=vae_arch)
        pipeline_config = SimpleNamespace(
            prepare_pos_cond_kwargs=prepare_pos_cond_kwargs,
            vae_config=vae_config,
        )
        return SimpleNamespace(pipeline_config=pipeline_config)

    def test_on_resolution_change_updates_freqs_cis_in_all_branches(self):
        stage = self._make_stage()
        ctx = self._make_mock_ctx(freqs_cis_value="old_freqs")
        new_freqs = object()
        server_args = self._make_mock_server_args(
            new_freqs_cis=new_freqs, new_img_shapes=None
        )
        batch = SimpleNamespace(height=512, width=512)

        stage._on_resolution_change(ctx, batch, server_args, 512, 512)

        for branch in ctx.cfg_policy.branches:
            self.assertIs(branch.kwargs["freqs_cis"], new_freqs)
        self.assertIs(ctx.pos_cond_kwargs["freqs_cis"], new_freqs)

    def test_on_resolution_change_updates_img_shapes_in_all_branches(self):
        stage = self._make_stage()
        ctx = self._make_mock_ctx(img_shapes_value="old_shapes")
        new_shapes = [[(1, 32, 32)]]
        server_args = self._make_mock_server_args(
            new_freqs_cis=("img_cache", "txt_cache"), new_img_shapes=new_shapes
        )
        batch = SimpleNamespace(height=512, width=512)

        stage._on_resolution_change(ctx, batch, server_args, 512, 512)

        for branch in ctx.cfg_policy.branches:
            self.assertIs(branch.kwargs["img_shapes"], new_shapes)
        self.assertIs(ctx.pos_cond_kwargs["img_shapes"], new_shapes)

    def test_on_resolution_change_skips_freqs_cis_when_none_returned(self):
        """If prepare_pos_cond_kwargs returns freqs_cis=None, skip silently."""
        stage = self._make_stage()
        ctx = self._make_mock_ctx(freqs_cis_value="old_freqs")
        server_args = self._make_mock_server_args(
            new_freqs_cis=None, new_img_shapes=None
        )
        batch = SimpleNamespace(height=512, width=512)

        stage._on_resolution_change(ctx, batch, server_args, 512, 512)

        # freqs_cis must remain unchanged when None is returned
        for branch in ctx.cfg_policy.branches:
            self.assertEqual(branch.kwargs["freqs_cis"], "old_freqs")

    def test_on_resolution_change_skips_branches_without_freqs_cis_key(self):
        """Branches that don't have freqs_cis in their kwargs are left alone."""
        stage = self._make_stage()
        branch_no_freqs = SimpleNamespace(kwargs={"img_shapes": "shapes"})
        branch_with_freqs = SimpleNamespace(
            kwargs={"freqs_cis": "old", "img_shapes": "shapes"}
        )
        cfg_policy = SimpleNamespace(branches=[branch_no_freqs, branch_with_freqs])
        ctx = SimpleNamespace(
            cfg_policy=cfg_policy,
            pos_cond_kwargs={"freqs_cis": "old"},
            target_dtype=torch.bfloat16,
        )
        new_freqs = object()
        server_args = self._make_mock_server_args(
            new_freqs_cis=new_freqs, new_img_shapes=None
        )
        batch = SimpleNamespace(height=512, width=512)

        stage._on_resolution_change(ctx, batch, server_args, 512, 512)

        self.assertNotIn("freqs_cis", branch_no_freqs.kwargs)
        self.assertIs(branch_with_freqs.kwargs["freqs_cis"], new_freqs)

    def test_on_resolution_change_null_img_shapes_leaves_branch_img_shapes_intact(self):
        """When new img_shapes is None, existing branch img_shapes is unchanged."""
        stage = self._make_stage()
        ctx = self._make_mock_ctx(img_shapes_value="old_shapes")
        server_args = self._make_mock_server_args(
            new_freqs_cis=("img", "txt"), new_img_shapes=None
        )
        batch = SimpleNamespace(height=512, width=512)

        stage._on_resolution_change(ctx, batch, server_args, 512, 512)

        for branch in ctx.cfg_policy.branches:
            self.assertEqual(branch.kwargs["img_shapes"], "old_shapes")

    def test_on_resolution_change_updates_both_freqs_and_shapes_together(self):
        """freqs_cis and img_shapes are updated atomically in the same call."""
        stage = self._make_stage()
        ctx = self._make_mock_ctx(
            freqs_cis_value="old_freqs", img_shapes_value="old_shapes"
        )
        new_freqs = ("new_img_cache", "new_txt_cache")
        new_shapes = [[(1, 16, 16)]]
        server_args = self._make_mock_server_args(
            new_freqs_cis=new_freqs, new_img_shapes=new_shapes
        )
        batch = SimpleNamespace(height=256, width=256)

        stage._on_resolution_change(ctx, batch, server_args, 256, 256)

        for branch in ctx.cfg_policy.branches:
            self.assertEqual(branch.kwargs["freqs_cis"], new_freqs)
            self.assertIs(branch.kwargs["img_shapes"], new_shapes)
        self.assertEqual(ctx.pos_cond_kwargs["freqs_cis"], new_freqs)
        self.assertIs(ctx.pos_cond_kwargs["img_shapes"], new_shapes)


# ---------------------------------------------------------------------------
# QwenImagePipeline now carries QwenImageProgressiveDenoisingStage
# ---------------------------------------------------------------------------


class TestQwenImagePipelineUsesProgressiveStage(unittest.TestCase):
    """QwenImagePipeline is modified to always use QwenImageProgressiveDenoisingStage
    (following the same pattern as FluxPipeline and ZImagePipeline).
    """

    def test_pipeline_name(self):
        self.assertEqual(QwenImagePipeline.pipeline_name, "QwenImagePipeline")

    def test_has_add_qwen_denoising_stage_method(self):
        """QwenImagePipeline must expose _add_qwen_denoising_stage."""
        self.assertTrue(hasattr(QwenImagePipeline, "_add_qwen_denoising_stage"))

    def test_required_config_modules(self):
        self.assertIn("transformer", QwenImagePipeline._required_config_modules)
        self.assertIn("scheduler", QwenImagePipeline._required_config_modules)
        self.assertIn("vae", QwenImagePipeline._required_config_modules)


# ---------------------------------------------------------------------------
# Integration: pack/unpack consistency with ProgressiveDenoisingStage helpers
# ---------------------------------------------------------------------------


class TestQwenPackIntegrationWithProgressiveBase(unittest.TestCase):
    """Verify that pack/unpack is consistent with what the base class expects."""

    def test_spatial_channels_match_base_class_assumption(self):
        """Base class generates initial noise with C = in_channels // 4 = 16 channels.
        The pack/unpack must handle exactly 16 spatial channels."""
        in_channels = 64  # from QwenImageDitConfig
        C = in_channels // 4
        self.assertEqual(C, 16)

        spatial = torch.randn(1, C, 8, 8)
        packed = _qwen_image_pack(spatial, 8, 8)
        self.assertEqual(packed.shape[-1], 64)  # C * 4 = 64

    def test_pack_token_dim_is_in_channels(self):
        """Token dim after packing must equal in_channels (64)."""
        spatial = _make_spatial(1, 8, 8)
        packed = _qwen_image_pack(spatial, 8, 8)
        self.assertEqual(packed.shape[-1], 64)

    def test_upsample_2x_then_repack_doubles_sequence_length(self):
        """After 2× upsampling spatial → repack, S_high = 4 * S_low."""
        from sglang.multimodal_gen.runtime.pipelines_core.stages.progressive_resolution.upsample import (
            dct_upsample_2d,
        )

        h_low, w_low = 8, 8
        x_low = _make_spatial(1, h_low, w_low)
        x_high = dct_upsample_2d(x_low, sigma_t=0.3, seed=42)  # [1, 16, 16, 16]
        h_high, w_high = x_high.shape[2], x_high.shape[3]

        packed_low = _qwen_image_pack(x_low, h_low, w_low)
        packed_high = _qwen_image_pack(x_high, h_high, w_high)

        S_low = packed_low.shape[1]
        S_high = packed_high.shape[1]
        self.assertEqual(S_high, 4 * S_low)


if __name__ == "__main__":
    unittest.main()
