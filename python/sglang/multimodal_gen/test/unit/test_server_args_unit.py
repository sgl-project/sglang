import os
import unittest

from sglang.multimodal_gen.registry import _get_config_info
from sglang.multimodal_gen.runtime.server_args import ServerArgs


class TestServerArgsPathExpansion(unittest.TestCase):
    def test_tilde_model_path_is_expanded(self):
        args = ServerArgs.from_dict({"model_path": "~/fake/local/model"})
        expected = os.path.expanduser("~/fake/local/model")
        self.assertEqual(args.model_path, expected)
        self.assertFalse(args.model_path.startswith("~"))

    def test_absolute_path_is_unchanged(self):
        args = ServerArgs.from_dict({"model_path": "/data/my-model"})
        self.assertEqual(args.model_path, "/data/my-model")


class TestModelIdResolution(unittest.TestCase):
    def setUp(self):
        _get_config_info.cache_clear()

    def test_model_id_overrides_arbitrary_local_path(self):
        # a local path whose directory name does not match any HF repo name;
        # --model-id tells the engine which config to use
        info = _get_config_info("/data/my-custom-qwen", model_id="Qwen-Image")
        self.assertIsNotNone(info)
        from sglang.multimodal_gen.configs.pipeline_configs.qwen_image import (
            QwenImagePipelineConfig,
        )

        self.assertIs(info.pipeline_config_cls, QwenImagePipelineConfig)

    def test_model_id_works_after_tilde_expansion(self):
        # simulate the full flow: user passes ~/..., engine expands and resolves
        expanded = os.path.expanduser("~/.cache/huggingface/hub/bbb/snapshots/ccc")
        _get_config_info.cache_clear()
        info = _get_config_info(expanded, model_id="Qwen-Image")
        self.assertIsNotNone(info)

    def test_model_id_unknown_falls_back_without_crash(self):
        # unrecognized model_id: should warn and fall back to path-based detection
        # with an unresolvable path, expect RuntimeError from the detector step
        with self.assertRaises((RuntimeError, Exception)):
            _get_config_info("/data/no-such-model", model_id="NonExistentModelXYZ")


if __name__ == "__main__":
    unittest.main()
