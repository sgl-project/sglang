import unittest

from test_ascend_llm_models import TestMistral


class TestGranite_3_1(TestMistral):
    model = "/root/.cache/modelscope/hub/models/ibm-granite/granite-3.1-8b-instruct"
    accuracy = 0.00


if __name__ == "__main__":
    unittest.main()
