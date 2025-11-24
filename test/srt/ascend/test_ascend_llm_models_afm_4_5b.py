import unittest

from test_ascend_llm_models import TestMistral


class TestArceeAFM(TestMistral):
    model = "/root/.cache/modelscope/hub/models/arcee-ai/AFM-4.5B-Base"
    accuracy = 0.00


if __name__ == "__main__":
    unittest.main()
