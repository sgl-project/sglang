import unittest

from test_ascend_llm_models import TestMistral


class TestGLM4v9BChat(TestMistral):
    model = "/root/.cache/modelscope/hub/models/ZhipuAI/glm-4-9b-chat"
    accuracy = -1


if __name__ == "__main__":
    unittest.main()
