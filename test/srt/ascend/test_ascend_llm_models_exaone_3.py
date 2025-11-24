import unittest

from test_ascend_llm_models import TestMistral


class TestExaONE3(TestMistral):
    model = "/root/.cache/modelscope/hub/models/LGAI-EXAONE/EXAONE-3.5-7.8B-Instruct"
    accuracy = 0.00


if __name__ == "__main__":
    unittest.main()
