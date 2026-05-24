import random
import tempfile
import unittest
from types import SimpleNamespace

from sglang.test.ci.ci_register import register_amd_ci, register_cuda_ci
from sglang.test.kits.mmmu_vlm_kit import MMMUMultiModelTestBase
from sglang.test.test_utils import is_in_ci

register_cuda_ci(est_time=500, suite="nightly-4-gpu", nightly=True)
register_amd_ci(est_time=500, suite="nightly-amd-4-gpu", nightly=True)

MODELS = [
    SimpleNamespace(model="Qwen/Qwen2.5-VL-72B-Instruct", mmmu_accuracy=0.55),
    SimpleNamespace(model="Qwen/Qwen3-VL-32B-Instruct", mmmu_accuracy=0.55),
    SimpleNamespace(model="OpenGVLab/InternVL2_5-8B", mmmu_accuracy=0.52),
    SimpleNamespace(model="zai-org/GLM-4.1V-9B-Thinking", mmmu_accuracy=0.68),
]


class TestVLMEncoderDP(MMMUMultiModelTestBase):
    # --cuda-graph-max-bs 32 last-wins over the kit's default 64.
    other_args = [
        "--mm-enable-dp-encoder",
        "--tp=4",
        "--cuda-graph-max-bs",
        "32",
    ]

    def test_vlm_mmmu_benchmark(self):
        models_to_test = MODELS

        if is_in_ci():
            models_to_test = [random.choice(MODELS)]

        for model in models_to_test:
            # Per-model temp dir avoids cross-test cached results.
            with tempfile.TemporaryDirectory(
                prefix=f"encoder_dp_{model.model.replace('/', '_')}_"
            ) as output_path:
                self._run_vlm_mmmu_test(model, output_path)


if __name__ == "__main__":
    unittest.main()
