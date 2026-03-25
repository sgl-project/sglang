import os
import sys
import unittest
from unittest.mock import patch

from sglang.multimodal_gen.configs.pipeline_configs.base import PipelineConfig
from sglang.multimodal_gen.configs.pipeline_configs.qwen_image import (
    QwenImagePipelineConfig,
)
from sglang.multimodal_gen.registry import _get_config_info
from sglang.multimodal_gen.runtime.layers.quantization.configs.nunchaku_config import (
    NunchakuConfig,
)
from sglang.multimodal_gen.runtime.server_args import ServerArgs
from sglang.multimodal_gen.utils import FlexibleArgumentParser


class TestServerArgsPathExpansion(unittest.TestCase):
    def _from_dict_without_model_resolution(self, kwargs):
        with patch.object(
            PipelineConfig, "from_kwargs", return_value=QwenImagePipelineConfig()
        ):
            return ServerArgs.from_dict(kwargs)

    def test_tilde_model_path_is_expanded(self):
        args = self._from_dict_without_model_resolution(
            {"model_path": "~/fake/local/model"}
        )
        expected = os.path.expanduser("~/fake/local/model")
        self.assertEqual(args.model_path, expected)
        self.assertFalse(args.model_path.startswith("~"))

    def test_absolute_path_is_unchanged(self):
        args = self._from_dict_without_model_resolution(
            {"model_path": "/data/my-model"}
        )
        self.assertEqual(args.model_path, "/data/my-model")

    def test_component_paths_are_expanded_before_pipeline_resolution(self):
        args = self._from_dict_without_model_resolution(
            {
                "model_path": "/data/my-model",
                "component_paths": {"vae": "~/fake/local/vae"},
            }
        )

        self.assertEqual(
            args.component_paths["vae"], os.path.expanduser("~/fake/local/vae")
        )

    @patch("sglang.multimodal_gen.configs.quantization.is_nunchaku_available")
    @patch(
        "sglang.multimodal_gen.configs.quantization.torch.cuda.get_device_capability"
    )
    @patch("sglang.multimodal_gen.configs.quantization.torch.cuda.device_count")
    @patch("sglang.multimodal_gen.configs.quantization.current_platform.is_cuda")
    def test_nunchaku_args_are_resolved_out_of_server_args(
        self,
        mock_is_cuda,
        mock_device_count,
        mock_device_capability,
        mock_is_nunchaku_available,
    ):
        mock_is_cuda.return_value = True
        mock_device_count.return_value = 1
        mock_device_capability.return_value = (8, 0)
        mock_is_nunchaku_available.return_value = True

        args = self._from_dict_without_model_resolution(
            {
                "model_path": "/data/my-model",
                "transformer_weights_path": "/tmp/svdq-int4_r32-qwen-image.safetensors",
            }
        )

        self.assertIsInstance(args.nunchaku_config, NunchakuConfig)
        self.assertEqual(
            args.transformer_weights_path,
            "/tmp/svdq-int4_r32-qwen-image.safetensors",
        )
        self.assertEqual(args.nunchaku_config.precision, "int4")
        self.assertEqual(args.nunchaku_config.rank, 32)

    def test_non_nunchaku_transformer_weights_path_stays_as_generic_override(self):
        args = self._from_dict_without_model_resolution(
            {
                "model_path": "/data/my-model",
                "transformer_weights_path": "/tmp/flux2-dev-nvfp4-mixed.safetensors",
            }
        )

        self.assertIsNone(args.nunchaku_config)
        self.assertEqual(
            args.transformer_weights_path,
            "/tmp/flux2-dev-nvfp4-mixed.safetensors",
        )


class TestModelIdResolution(unittest.TestCase):
    def setUp(self):
        _get_config_info.cache_clear()

    def test_model_id_overrides_arbitrary_local_path(self):
        # a local path whose directory name does not match any HF repo name;
        # --model-id tells the engine which config to use
        info = _get_config_info("/data/my-custom-qwen", model_id="Qwen-Image")
        self.assertIsNotNone(info)

        self.assertIs(info.pipeline_config_cls, QwenImagePipelineConfig)

    def test_model_id_works_after_tilde_expansion(self):
        # simulate the full flow: user passes ~/..., engine expands and resolves
        expanded = os.path.expanduser("~/.cache/huggingface/hub/bbb/snapshots/ccc")
        _get_config_info.cache_clear()
        info = _get_config_info(expanded, model_id="Qwen-Image")
        self.assertIsNotNone(info)

    def test_hf_cache_snapshot_path_resolves_registered_nvfp4_model(self):
        path = (
            "/root/.cache/huggingface/hub/"
            "models--black-forest-labs--FLUX.2-dev-NVFP4/"
            "snapshots/142b87e70bc3006937b7093d89ff287b5f59f071"
        )
        info = _get_config_info(path)
        self.assertIsNotNone(info)

    def test_model_id_unknown_falls_back_without_crash(self):
        # unrecognized model_id: should warn and fall back to path-based detection
        # with an unresolvable path, expect RuntimeError from the detector step
        with self.assertRaises((RuntimeError, Exception)):
            _get_config_info("/data/no-such-model", model_id="NonExistentModelXYZ")


class TestPipelineResolutionCliOverride(unittest.TestCase):
    def setUp(self):
        _get_config_info.cache_clear()

    def test_resolution_flag_overrides_qwen_image_layered_pipeline_config(self):
        parser = FlexibleArgumentParser()
        ServerArgs.add_cli_args(parser)
        argv = [
            "--model-path",
            "Qwen/Qwen-Image-Layered",
            "--resolution",
            "768",
        ]

        with patch.object(sys, "argv", ["sglang"] + argv):
            args, unknown_args = parser.parse_known_args(argv)
            server_args = ServerArgs.from_cli_args(args, unknown_args)

        self.assertEqual(server_args.pipeline_config.resolution, 768)


if __name__ == "__main__":
    unittest.main()
