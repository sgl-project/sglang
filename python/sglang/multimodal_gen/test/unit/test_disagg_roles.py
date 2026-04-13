# SPDX-License-Identifier: Apache-2.0
"""Unit tests for disaggregation role-based pipeline filtering."""

import unittest

from sglang.multimodal_gen.runtime.disaggregation.roles import (
    RoleType,
    filter_modules_for_role,
    get_module_role,
)


class TestRoleType(unittest.TestCase):
    def test_from_string(self):
        self.assertEqual(RoleType.from_string("monolithic"), RoleType.MONOLITHIC)
        self.assertEqual(RoleType.from_string("encoder"), RoleType.ENCODER)
        self.assertEqual(RoleType.from_string("denoiser"), RoleType.DENOISER)
        self.assertEqual(RoleType.from_string("decoder"), RoleType.DECODER)
        self.assertEqual(RoleType.from_string("ENCODER"), RoleType.ENCODER)

    def test_from_string_backward_compat(self):
        # "denoising" is accepted as backward-compat alias for "denoiser"
        self.assertEqual(RoleType.from_string("denoising"), RoleType.DENOISER)

    def test_from_string_invalid(self):
        with self.assertRaises(ValueError):
            RoleType.from_string("invalid")

    def test_choices(self):
        choices = RoleType.choices()
        self.assertIn("monolithic", choices)
        self.assertIn("encoder", choices)
        self.assertIn("denoiser", choices)
        self.assertIn("decoder", choices)


class TestGetModuleRole(unittest.TestCase):
    def test_encoder_modules(self):
        self.assertEqual(get_module_role("text_encoder"), RoleType.ENCODER)
        self.assertEqual(get_module_role("text_encoder_2"), RoleType.ENCODER)
        self.assertEqual(get_module_role("tokenizer"), RoleType.ENCODER)
        self.assertEqual(get_module_role("tokenizer_2"), RoleType.ENCODER)
        self.assertEqual(get_module_role("image_encoder"), RoleType.ENCODER)
        self.assertEqual(get_module_role("image_processor"), RoleType.ENCODER)
        self.assertEqual(get_module_role("connectors"), RoleType.ENCODER)

    def test_denoising_modules(self):
        self.assertEqual(get_module_role("transformer"), RoleType.DENOISER)
        self.assertEqual(get_module_role("transformer_2"), RoleType.DENOISER)

    def test_decoder_modules(self):
        self.assertEqual(get_module_role("vae"), RoleType.DECODER)
        self.assertEqual(get_module_role("audio_vae"), RoleType.DECODER)
        self.assertEqual(get_module_role("vocoder"), RoleType.DECODER)

    def test_shared_modules(self):
        self.assertIsNone(get_module_role("scheduler"))


class TestFilterModulesForRole(unittest.TestCase):
    """Test module filtering for WanPipeline-like config."""

    WAN_MODULES = ["text_encoder", "tokenizer", "vae", "transformer", "scheduler"]

    def test_monolithic_keeps_all(self):
        result = filter_modules_for_role(self.WAN_MODULES, RoleType.MONOLITHIC)
        self.assertEqual(result, self.WAN_MODULES)

    def test_encoder_skips_transformer(self):
        result = filter_modules_for_role(self.WAN_MODULES, RoleType.ENCODER)
        self.assertIn("text_encoder", result)
        self.assertIn("tokenizer", result)
        self.assertIn("vae", result)  # encoder needs VAE for ImageVAEEncoding
        self.assertIn("scheduler", result)
        self.assertNotIn("transformer", result)

    def test_denoising_skips_encoders_and_vae(self):
        result = filter_modules_for_role(self.WAN_MODULES, RoleType.DENOISER)
        self.assertIn("transformer", result)
        self.assertIn("scheduler", result)
        self.assertNotIn("text_encoder", result)
        self.assertNotIn("tokenizer", result)
        self.assertNotIn("vae", result)

    def test_decoder_keeps_vae_and_scheduler(self):
        result = filter_modules_for_role(self.WAN_MODULES, RoleType.DECODER)
        self.assertIn("vae", result)
        self.assertIn("scheduler", result)
        self.assertNotIn("text_encoder", result)
        self.assertNotIn("transformer", result)


class TestFilterModulesFlux(unittest.TestCase):
    """Test module filtering for FluxPipeline-like config with dual encoders."""

    FLUX_MODULES = [
        "text_encoder",
        "text_encoder_2",
        "tokenizer",
        "tokenizer_2",
        "vae",
        "transformer",
        "scheduler",
    ]

    def test_denoising_skips_all_encoders(self):
        result = filter_modules_for_role(self.FLUX_MODULES, RoleType.DENOISER)
        self.assertEqual(set(result), {"transformer", "scheduler"})

    def test_encoder_keeps_all_encoders(self):
        result = filter_modules_for_role(self.FLUX_MODULES, RoleType.ENCODER)
        self.assertIn("text_encoder", result)
        self.assertIn("text_encoder_2", result)
        self.assertIn("tokenizer", result)
        self.assertIn("tokenizer_2", result)
        self.assertNotIn("transformer", result)


class TestFilterModulesLTX2(unittest.TestCase):
    """Test module filtering for LTX2Pipeline-like config with audio."""

    LTX2_MODULES = [
        "transformer",
        "text_encoder",
        "tokenizer",
        "scheduler",
        "vae",
        "audio_vae",
        "vocoder",
        "connectors",
    ]

    def test_decoder_includes_audio(self):
        result = filter_modules_for_role(self.LTX2_MODULES, RoleType.DECODER)
        self.assertIn("vae", result)
        self.assertIn("audio_vae", result)
        self.assertIn("vocoder", result)
        self.assertIn("scheduler", result)
        self.assertNotIn("transformer", result)
        self.assertNotIn("text_encoder", result)

    def test_encoder_includes_connectors(self):
        result = filter_modules_for_role(self.LTX2_MODULES, RoleType.ENCODER)
        self.assertIn("connectors", result)
        self.assertIn("text_encoder", result)
        self.assertNotIn("transformer", result)


class TestFilterModulesI2V(unittest.TestCase):
    """Test module filtering for WanImageToVideoPipeline-like config."""

    I2V_MODULES = [
        "text_encoder",
        "tokenizer",
        "vae",
        "transformer",
        "scheduler",
        "image_encoder",
        "image_processor",
    ]

    def test_encoder_includes_image_encoder(self):
        result = filter_modules_for_role(self.I2V_MODULES, RoleType.ENCODER)
        self.assertIn("image_encoder", result)
        self.assertIn("image_processor", result)
        self.assertIn("text_encoder", result)
        self.assertNotIn("transformer", result)

    def test_denoising_skips_all_encoders(self):
        result = filter_modules_for_role(self.I2V_MODULES, RoleType.DENOISER)
        self.assertEqual(set(result), {"transformer", "scheduler"})


if __name__ == "__main__":
    unittest.main()
