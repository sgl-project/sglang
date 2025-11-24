import unittest

from test_ascend_llm_models_2 import TestLlama_2_7B


class TestPhi_4_multimodal_instruct(TestLlama_2_7B):
    model = "/root/.cache/modelscope/hub/models/LLM-Research/Phi-4-multimodal-instruct"
    accuracy = 0.05


if __name__ == "__main__":
    unittest.main()
