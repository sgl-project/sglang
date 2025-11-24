import unittest

from test_ascend_llm_models_2 import TestLlama_2_7B


class TestBaichuan2(TestLlama_2_7B):
    model = "/root/.cache/modelscope/hub/models/baichuan-inc/Baichuan2-13B-Chat"
    accuracy = -1


if __name__ == "__main__":
    unittest.main()
