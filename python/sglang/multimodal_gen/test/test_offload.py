import unittest
from unittest.mock import Mock, patch

from sglang.multimodal_gen.configs.models.encoders.t5 import T5Config
from sglang.multimodal_gen.configs.pipeline_configs.base import PipelineConfig
from sglang.multimodal_gen.runtime.loader.component_loader import TextEncoderLoader
from sglang.multimodal_gen.runtime.platforms import current_platform


class TestTextEncoderOffload(unittest.TestCase):
    """Test text encoder CPU offload functionality."""

    def setUp(self):
        """Set up test fixtures."""
        self.loader = TextEncoderLoader()

    def test_should_offload_with_text_encoder_cpu_offload_enabled(self):
        """Test that should_offload returns True when text_encoder_cpu_offload is enabled."""
        server_args = Mock()
        server_args.text_encoder_cpu_offload = True
        server_args.pipeline_config = PipelineConfig()

        model_config = T5Config()
        model_config.arch_config._fsdp_shard_conditions = [
            lambda n, m: "block" in n and str.isdigit(n.split(".")[-1]),
        ]

        result = self.loader.should_offload(server_args, model_config)

        self.assertTrue(
            result,
            "should_offload should be True when text_encoder_cpu_offload is enabled and _fsdp_shard_conditions is not empty",
        )

    def test_should_offload_with_text_encoder_cpu_offload_disabled(self):
        """Test that should_offload returns False when text_encoder_cpu_offload is disabled."""
        server_args = Mock()
        server_args.text_encoder_cpu_offload = False
        server_args.pipeline_config = PipelineConfig()

        model_config = T5Config()
        model_config.arch_config._fsdp_shard_conditions = [
            lambda n, m: "block" in n and str.isdigit(n.split(".")[-1]),
        ]

        result = self.loader.should_offload(server_args, model_config)

        self.assertFalse(
            result,
            "should_offload should be False when text_encoder_cpu_offload is disabled",
        )

    def test_should_offload_with_empty_fsdp_shard_conditions(self):
        """Test that should_offload returns False when _fsdp_shard_conditions is empty."""
        server_args = Mock()
        server_args.text_encoder_cpu_offload = True
        server_args.pipeline_config = PipelineConfig()

        model_config = T5Config()
        model_config.arch_config._fsdp_shard_conditions = []

        result = self.loader.should_offload(server_args, model_config)

        self.assertFalse(
            result,
            "should_offload should be False when _fsdp_shard_conditions is empty",
        )

    def test_target_device_with_offload_enabled(self):
        """Test that target_device returns CPU device when offload is enabled."""
        import torch

        result = self.loader.target_device(should_offload=True)

        self.assertEqual(
            result.type,
            torch.device("cpu").type,
            "target_device should be CPU when should_offload is True",
        )

    def test_target_device_with_offload_disabled(self):
        """Test that target_device returns GPU device when offload is disabled."""
        import torch

        result = self.loader.target_device(should_offload=False)

        self.assertIn(
            result.type,
            [torch.device("cuda").type, torch.device("mps").type],
            "target_device should be GPU/MPS when should_offload is False",
        )

    @patch.object(current_platform, "is_mps", return_value=True)
    def test_mps_platform_excludes_fsdp(self, mock_is_mps):
        """Test that MPS platform excludes FSDP offloading."""
        server_args = Mock()
        server_args.text_encoder_cpu_offload = True
        server_args.pipeline_config = PipelineConfig()

        model_config = T5Config()
        model_config.arch_config._fsdp_shard_conditions = [
            lambda n, m: "block" in n and str.isdigit(n.split(".")[-1]),
        ]

        should_offload = self.loader.should_offload(server_args, model_config)
        target_device = self.loader.target_device(should_offload)

        self.assertTrue(should_offload, "should_offload should be True on MPS platform")
        self.assertEqual(
            target_device.type,
            "mps",
            "target_device should be MPS on MPS platform (FSDP is disabled but device is MPS)",
        )

    @patch.object(current_platform, "is_mps", return_value=False)
    def test_non_mps_platform_includes_fsdp(self, mock_is_mps):
        """Test that non-MPS platform includes FSDP offloading."""
        server_args = Mock()
        server_args.text_encoder_cpu_offload = True
        server_args.pipeline_config = PipelineConfig()

        model_config = T5Config()
        model_config.arch_config._fsdp_shard_conditions = [
            lambda n, m: "block" in n and str.isdigit(n.split(".")[-1]),
        ]

        should_offload = self.loader.should_offload(server_args, model_config)
        target_device = self.loader.target_device(should_offload)

        self.assertTrue(
            should_offload, "should_offload should be True on non-MPS platform"
        )
        self.assertEqual(
            target_device.type, "cpu", "target_device should be CPU for FSDP offloading"
        )


if __name__ == "__main__":
    unittest.main()
