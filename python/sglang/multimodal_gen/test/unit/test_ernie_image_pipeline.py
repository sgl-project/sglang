import unittest
from types import SimpleNamespace
from unittest.mock import patch

from sglang.multimodal_gen.runtime.pipelines.ernie_image import ErnieImagePipeline


class TestErnieImagePipeline(unittest.TestCase):
    def test_disable_pe_skips_prompt_enhancement_load(self):
        pipeline = ErnieImagePipeline.__new__(ErnieImagePipeline)
        args = SimpleNamespace(model_path="baidu/ERNIE-Image", disable_pe=True)

        with patch.object(
            ErnieImagePipeline, "_has_pe_in_model_index", return_value=True
        ):
            self.assertFalse(pipeline._should_load_pe(args))


if __name__ == "__main__":
    unittest.main()
