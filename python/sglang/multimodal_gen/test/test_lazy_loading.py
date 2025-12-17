"""
Unit tests for lazy-loading code paths in denoising and decoding stages.

These tests verify that:
1. ComponentLoader.load() is called with correct arguments (4 positional args)
2. torch_compile_module() is used instead of direct torch.compile()
"""

import unittest
from unittest.mock import MagicMock, Mock, patch

from sglang.multimodal_gen.runtime.pipelines_core.schedule_batch import ScheduleBatch
from sglang.multimodal_gen.runtime.pipelines_core.stages.decoding import DecodingStage
from sglang.multimodal_gen.runtime.pipelines_core.stages.denoising import DenoisingStage
from sglang.multimodal_gen.runtime.server_args import ServerArgs


class TestLazyLoadingDenoising(unittest.TestCase):
    """Test lazy-loading in DenoisingStage"""

    def setUp(self) -> None:
        """Set up test fixtures"""
        self.server_args: ServerArgs = ServerArgs(
            model_path="test/model",
            enable_torch_compile=True,
        )
        self.server_args.model_loaded = {
            "transformer": False,  # Force lazy-loading path
            "vae": True,
        }
        self.server_args.model_paths = {
            "transformer": "test/model/transformer",
            "vae": "test/model/vae",
        }

        # Create mock objects
        self.mock_transformer = MagicMock()
        self.mock_scheduler = MagicMock()
        self.mock_scheduler.num_train_timesteps = 1000

        # Create DenoisingStage instance
        self.stage = DenoisingStage(
            transformer=self.mock_transformer,
            scheduler=self.mock_scheduler,
        )

    @patch(
        "sglang.multimodal_gen.runtime.pipelines_core.stages.denoising.TransformerLoader"
    )
    def test_loader_called_with_correct_arguments(self, mock_loader_class):
        """Test that ComponentLoader.load() is called with 4 required arguments"""
        # Setup mock
        mock_loader_instance = Mock()
        mock_loaded_transformer = MagicMock()
        mock_loader_instance.load.return_value = mock_loaded_transformer
        mock_loader_class.return_value = mock_loader_instance

        # Create mock batch
        batch = MagicMock(spec=ScheduleBatch)
        batch.num_inference_steps = 50

        # Call the method that triggers lazy-loading
        self.stage._prepare_denoising_loop(batch, self.server_args)

        # Verify loader.load() was called with exactly 4 arguments
        mock_loader_instance.load.assert_called_once_with(
            "test/model/transformer",  # component_model_path
            self.server_args,  # server_args
            "transformer",  # module_name
            "diffusers",  # transformers_or_diffusers
        )

        # Verify model_loaded flag was updated
        self.assertTrue(self.server_args.model_loaded["transformer"])

    @patch(
        "sglang.multimodal_gen.runtime.pipelines_core.stages.denoising.TransformerLoader"
    )
    def test_torch_compile_module_used(self, mock_loader_class):
        """Test that torch_compile_module() is used instead of direct torch.compile()"""
        # Setup mock
        mock_loader_instance = Mock()
        mock_loaded_transformer = MagicMock()
        mock_loader_instance.load.return_value = mock_loaded_transformer
        mock_loader_class.return_value = mock_loader_instance

        # Spy on torch_compile_module
        with patch.object(
            self.stage, "torch_compile_module", wraps=self.stage.torch_compile_module
        ) as mock_compile:
            # Create mock batch
            batch = MagicMock(spec=ScheduleBatch)
            batch.num_inference_steps = 50

            # Call the method that triggers lazy-loading
            self.stage._prepare_denoising_loop(batch, self.server_args)

            # Verify torch_compile_module was called
            mock_compile.assert_called_once()

            # Verify the transformer passed to torch_compile_module
            # is the loaded transformer
            args = mock_compile.call_args[0]
            self.assertEqual(args[0], mock_loaded_transformer)

    @patch(
        "sglang.multimodal_gen.runtime.pipelines_core.stages.denoising.TransformerLoader"
    )
    def test_lazy_loading_skipped_when_loaded(self, mock_loader_class):
        """Test that lazy-loading is skipped when transformer is already loaded"""
        # Set transformer as already loaded
        self.server_args.model_loaded["transformer"] = True

        # Create mock batch
        batch = MagicMock(spec=ScheduleBatch)
        batch.num_inference_steps = 50

        # Call the method
        self.stage._prepare_denoising_loop(batch, self.server_args)

        # Verify loader was NOT instantiated
        mock_loader_class.assert_not_called()


class TestLazyLoadingDecoding(unittest.TestCase):
    """Test lazy-loading in DecodingStage"""

    def setUp(self):
        """Set up test fixtures"""
        self.server_args = ServerArgs(
            model_path="test/model",
        )
        self.server_args.model_loaded = {
            "transformer": True,
            "vae": False,  # Force lazy-loading path
        }
        self.server_args.model_paths = {
            "transformer": "test/model/transformer",
            "vae": "test/model/vae",
        }

        # Create mock VAE
        self.mock_vae = MagicMock()

        # Create DecodingStage instance
        self.stage = DecodingStage(vae=self.mock_vae)

    @patch("sglang.multimodal_gen.runtime.pipelines_core.stages.decoding.VAELoader")
    def test_vae_loader_called_with_correct_arguments(self, mock_loader_class):
        """Test that VAELoader.load() is called with 4 required arguments"""
        # Setup mock
        mock_loader_instance = Mock()
        mock_loaded_vae = MagicMock()
        mock_loader_instance.load.return_value = mock_loaded_vae
        mock_loader_class.return_value = mock_loader_instance

        # Create mock batch
        batch = MagicMock(spec=ScheduleBatch)

        # Call the method that triggers lazy-loading
        self.stage._prepare_decoding(batch, self.server_args)

        # Verify loader.load() was called with exactly 4 arguments
        mock_loader_instance.load.assert_called_once_with(
            "test/model/vae",  # component_model_path
            self.server_args,  # server_args
            "vae",  # module_name
            "diffusers",  # transformers_or_diffusers
        )

        # Verify model_loaded flag was updated
        self.assertTrue(self.server_args.model_loaded["vae"])


if __name__ == "__main__":
    unittest.main()
