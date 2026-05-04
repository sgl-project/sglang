import random
import tempfile
import unittest
from types import SimpleNamespace

from sglang.srt.utils import is_hip
from sglang.test.ci.ci_register import register_amd_ci, register_cuda_ci
from sglang.test.kits.mmmu_vlm_kit import (
    MMMUMultiModelTestBase,
)
from sglang.test.test_utils import is_in_ci

# VLM (Vision Language Model) tests


register_cuda_ci(est_time=223, suite="stage-b-test-1-gpu-large")
register_amd_ci(est_time=850, suite="stage-b-test-1-gpu-small-amd-nondeterministic")

_is_hip = is_hip()
# VLM models for testing
if _is_hip:
    MODELS = [SimpleNamespace(model="openbmb/MiniCPM-V-2_6", mmmu_accuracy=0.4)]
else:
    MODELS = [
        SimpleNamespace(model="google/gemma-3-4b-it", mmmu_accuracy=0.38),
        SimpleNamespace(model="Qwen/Qwen2.5-VL-3B-Instruct", mmmu_accuracy=0.4),
        SimpleNamespace(model="openbmb/MiniCPM-V-2_6", mmmu_accuracy=0.4),
    ]


class TestVLMModels(MMMUMultiModelTestBase):
    def test_vlm_mmmu_benchmark(self):
        """Test VLM models against MMMU benchmark."""
        models_to_test = MODELS

        if is_in_ci():
            models_to_test = [random.choice(MODELS)]

        for model in models_to_test:
            # Use a unique temporary directory for each model to avoid cached results
            with tempfile.TemporaryDirectory(
                prefix=f"test_vlm_mmmu_{model.model.replace('/', '_')}_"
            ) as temp_dir:
                self._run_vlm_mmmu_test(model, temp_dir)


if __name__ == "__main__":
    unittest.main()
