import unittest

from test_ascend_llm_models_2 import TestLlama_2_7B


class TestChatglm2_6b(TestLlama_2_7B):
    model = "/root/.cache/modelscope/hub/models/ZhipuAI/chatglm2-6b"
    accuracy = 0.05


if __name__ == "__main__":
    unittest.main()
