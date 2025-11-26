import unittest
from types import SimpleNamespace

from test_vlm_utils import TestVLMModels


class TestJanus1Models(TestVLMModels):
    models = [
        SimpleNamespace(
            model="/root/.cache/modelscope/hub/models/deepseek-ai/Janus-Pro-1B",
            mmmu_accuracy=0.2,
        ),
    ]
    tp_size = 4
    mem_fraction_static = 0.35

    def test_vlm_mmmu_benchmark(self):
        self.vlm_mmmu_benchmark()


if __name__ == "__main__":
    unittest.main()
