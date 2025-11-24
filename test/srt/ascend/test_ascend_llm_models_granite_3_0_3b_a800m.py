import unittest

from test_ascend_llm_models import TestMistral


class TestGranite3B_a800m(TestMistral):
    model = (
        "/root/.cache/modelscope/hub/models/ibm-granite/granite-3.0-3b-a800m-instruct"
    )
    accuracy = -1


if __name__ == "__main__":
    unittest.main()
