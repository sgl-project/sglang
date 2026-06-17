import glob
import unittest
from importlib.util import find_spec

import torch

import sglang as sgl
from sglang.srt.model_loader.weight_utils import (
    download_weights_from_hf,
    instanttensor_weights_iterator,
    safetensors_weights_iterator,
)
from sglang.test.ci.ci_register import register_amd_ci, register_cuda_ci
from sglang.test.test_utils import DEFAULT_SMALL_MODEL_NAME_FOR_TEST, CustomTestCase

INSTANTTENSOR_AVAILABLE = find_spec("instanttensor") is not None

register_cuda_ci(est_time=30, stage="stage-b", runner_config="1-gpu-small")
register_amd_ci(est_time=30, stage="stage-b", runner_config="1-gpu-small-amd")

TEST_MODEL = DEFAULT_SMALL_MODEL_NAME_FOR_TEST
PROMPTS = [
    "Hello, my name is",
    "The president of the United States is",
    "The capital of France is",
    "The future of AI is",
]


@unittest.skipIf(not torch.cuda.is_available(), "Test requires CUDA")
@unittest.skipIf(not INSTANTTENSOR_AVAILABLE, "instanttensor is not installed")
class TestInstantTensorModelLoading(CustomTestCase):
    """Tests for InstantTensor weight loading and engine integration."""

    def test_instanttensor_weights_iterator_matches_safetensors(self):
        """InstantTensor iterator should match HF safetensors values exactly."""
        downloaded_path = download_weights_from_hf(
            TEST_MODEL,
            cache_dir=None,
            allow_patterns=["*.safetensors"],
        )
        safetensors_files = glob.glob(
            f"{downloaded_path}/**/*.safetensors", recursive=True
        )
        self.assertGreater(len(safetensors_files), 0)

        instanttensor_tensors = {}
        hf_safetensors_tensors = {}

        for name, tensor in instanttensor_weights_iterator(safetensors_files):
            # Copy immediately since InstantTensor may expose internal buffers.
            instanttensor_tensors[name] = tensor.to("cpu")

        for name, tensor in safetensors_weights_iterator(safetensors_files):
            hf_safetensors_tensors[name] = tensor

        self.assertEqual(len(instanttensor_tensors), len(hf_safetensors_tensors))

        for name, instanttensor_tensor in instanttensor_tensors.items():
            ref_tensor = hf_safetensors_tensors[name]
            self.assertEqual(instanttensor_tensor.dtype, ref_tensor.dtype)
            self.assertEqual(instanttensor_tensor.shape, ref_tensor.shape)
            self.assertTrue(torch.equal(instanttensor_tensor, ref_tensor))

    def test_engine_can_generate_with_instanttensor_loader(self):
        """Engine should generate non-empty outputs with instanttensor load format."""
        engine = sgl.Engine(
            model_path=TEST_MODEL,
            load_format="instanttensor",
        )
        try:
            outputs = engine.generate(PROMPTS, sampling_params={"max_new_tokens": 8})
            self.assertEqual(len(outputs), len(PROMPTS))
            for output in outputs:
                self.assertGreater(len(output["text"]), 0)
        finally:
            engine.shutdown()


if __name__ == "__main__":
    unittest.main()
