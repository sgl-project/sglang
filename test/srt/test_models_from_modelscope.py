from asyncio import subprocess
import shutil
import unittest
from types import SimpleNamespace
import os
from python.sglang.srt.utils import prepare_model, prepare_tokenizer


class TestDownloadFromModelScope(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.model = "iic/nlp_lstmcrf_word-segmentation_chinese-news"
        stat, output = subprocess.getstatusoutput('pip install modelscope')
        print(output)
        print(stat)
        
    @classmethod
    def tearDownClass(cls):
        pass

    def test_prepare_model(self):
        from modelscope.utils.file_utils import get_model_cache_root
                
        model_cache_root = get_model_cache_root()
        if os.path.exists(model_cache_root):
            shutil.rmtree(model_cache_root)
        model_path = prepare_model(self.model)
        assert os.path.exists(model_path, 'pytorch_model.bin')

    def test_prepare_tokenizer(self):
        from modelscope.utils.file_utils import get_model_cache_root
                
        model_cache_root = get_model_cache_root()
        if os.path.exists(model_cache_root):
            shutil.rmtree(model_cache_root)
        tokenizer_path = prepare_tokenizer(self.model)
        assert not os.path.exists(tokenizer_path, 'pytorch_model.bin')
        assert os.path.exists('config.json')
        


if __name__ == "__main__":
    unittest.main(warnings="ignore")