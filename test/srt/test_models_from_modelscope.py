import os
import shutil
import subprocess
import unittest
from unittest import mock

from sglang.srt.utils import prepare_model_and_tokenizer


class TestDownloadFromModelScope(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.model = "iic/nlp_lstmcrf_word-segmentation_chinese-news"
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
            model_path, tokenizer_path = prepare_model_and_tokenizer(
                self.model, self.model
            )
            assert os.path.exists(os.path.join(model_path, "pytorch_model.bin"))
            assert os.path.exists(os.path.join(tokenizer_path, "config.json"))


if __name__ == "__main__":
    unittest.main()
