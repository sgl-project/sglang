import os
import shutil
import subprocess
import unittest
from unittest import mock

from sglang.srt.utils import prepare_model_and_tokenizer
from sglang.test.test_utils import CustomTestCase


class TestDownloadFromModelScope(CustomTestCase):

    @classmethod
    def setUpClass(cls):
        # Use a small model to keep test time short.
        cls.model = "LLM-Research/tinyllama-15M-stories"
        stat, output = subprocess.getstatusoutput("pip install modelscope")

        cls.with_modelscope_environ = {k: v for k, v in os.environ.items()}
        cls.with_modelscope_environ["SGLANG_USE_MODELSCOPE"] = "True"

    @classmethod
    def tearDownClass(cls):
        pass

    def test_prepare_model_and_tokenizer(self):
        from modelscope.utils.file_utils import get_model_cache_root

        model_cache_root = get_model_cache_root()
        if os.path.exists(model_cache_root):
            shutil.rmtree(model_cache_root)

        with mock.patch.dict(os.environ, self.with_modelscope_environ, clear=True):
            from sglang.srt.server_args import ServerArgs

            args = ServerArgs(self.model)
            safetensors_path = os.path.join(args.model_path, "model.safetensors")
            self.assertTrue(
                os.path.exists(safetensors_path),
                f"prepare modelscope model failed, {safetensors_path} not found",
            )


if __name__ == "__main__":
    unittest.main()
