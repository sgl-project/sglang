import unittest
from types import SimpleNamespace

from test_vlm_utils import TestVLMModels


class TestGemmaModels(TestVLMModels):
    models = [
        SimpleNamespace(
            model="/root/.cache/modelscope/hub/models/google/gemma-3-4b-it",
            mmmu_accuracy=0.2,
        ),
    ]
    tp_size = 4
    mem_fraction_static = 0.35

    def test_vlm_mmmu_benchmark(self):
        self.vlm_mmmu_benchmark()


if __name__ == "__main__":
    unittest.main()
