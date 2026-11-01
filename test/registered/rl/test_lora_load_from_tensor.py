import json
import os
import unittest

from huggingface_hub import snapshot_download
from safetensors.torch import load_file

import sglang as sgl
from sglang.test.ci.ci_register import register_amd_ci, register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=60, stage="base-b", runner_config="1-gpu-small")
# backend-specific: the HIP build of the Triton LoRA kernels serving an adapter
# that arrived as serialized tensors, checked by the exact-text oracle.
register_amd_ci(est_time=90, suite="extra-a-test-1-gpu-large-amd")

MODEL_PATH = "Qwen/Qwen3-0.6B"
LORA_REPO = "charent/self_cognition_Alice"
TEST_PROMPT = "Hello, my name is"
EXPECTED_OUTPUT = (
    " Alice, and I am a software engineer. I am excited to share my journey"
)
MAX_NEW_TOKENS = 16


class TestLoRALoadFromTensor(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        cls.engine = sgl.Engine(
            model_path=MODEL_PATH,
            enable_lora=True,
            max_lora_rank=64,
            lora_target_modules=["all"],
            mem_fraction_static=0.6,
            log_level="error",
        )

        lora_adapter = snapshot_download(
            repo_id=LORA_REPO,
            allow_patterns=["adapter_model.safetensors", "adapter_config.json"],
        )
        # Load tensors and config from downloaded adapter
        cls.lora_tensors = load_file(
            os.path.join(lora_adapter, "adapter_model.safetensors")
        )
        with open(os.path.join(lora_adapter, "adapter_config.json"), "r") as f:
            cls.lora_config_dict = json.load(f)

    @classmethod
    def tearDownClass(cls):
        if hasattr(cls, "engine"):
            cls.engine.shutdown()

    def _load(self, lora_name, tensors=None, **kwargs):
        result = self.engine.load_lora_adapter_from_tensors(
            lora_name=lora_name,
            tensors=self.lora_tensors if tensors is None else tensors,
            config_dict=self.lora_config_dict,
            **kwargs,
        )
        self.assertTrue(
            result.success, f"Failed to load LoRA from tensors: {result.error_message}"
        )

    def _generate(self, lora_path):
        """The first len(EXPECTED_OUTPUT) characters generated for TEST_PROMPT."""
        output = self.engine.generate(
            prompt=[TEST_PROMPT],
            sampling_params={"max_new_tokens": MAX_NEW_TOKENS, "temperature": 0.0},
            lora_path=lora_path,
        )
        print(f"[lora_path={lora_path}] {output[0]}")
        return output[0]["text"][: len(EXPECTED_OUTPUT)]

    def test_lora_e2e_load_from_tensor_params(self):
        """Load, generate, unload, reload, generate: the adapted text must be
        exact after the first load and again after the reload."""
        lora_name = "self_cognition_Alice"
        self._load(lora_name)

        self.assertNotEqual(
            self._generate(lora_path=None),
            EXPECTED_OUTPUT,
            "Output before applying LoRA should not match expected result",
        )
        self.assertEqual(
            self._generate(lora_path=[lora_name]),
            EXPECTED_OUTPUT,
            "Output after applying LoRA does not match expected result",
        )

        result = self.engine.unload_lora_adapter(lora_name)
        self.assertTrue(
            result.success, f"Failed to unload LoRA: {result.error_message}"
        )

        self._load(lora_name)
        self.assertEqual(
            self._generate(lora_path=[lora_name]),
            EXPECTED_OUTPUT,
            "Output after reloading LoRA does not match expected result",
        )

    def test_lora_e2e_load_from_flattened_bucket(self):
        """Test loading LoRA via FlattenedTensorBucket format (RL weight sync path)."""
        from sglang.srt.utils import MultiprocessingSerializer
        from sglang.srt.weight_sync.tensor_bucket import FlattenedTensorBucket

        named_tensors = list(self.lora_tensors.items())
        bucket = FlattenedTensorBucket(named_tensors=[(n, t) for n, t in named_tensors])
        bucket_dict = {
            "flattened_tensor": bucket.get_flattened_tensor(),
            "metadata": bucket.get_metadata(),
        }
        serialized = MultiprocessingSerializer.serialize(bucket_dict, output_str=True)

        # flattened_bucket callers pass one serialized copy per TP rank, same
        # as Engine.update_weights_from_tensor.
        self._load(
            "self_cognition_Alice_flattened",
            tensors=[serialized],
            load_format="flattened_bucket",
        )
        self.assertEqual(
            self._generate(lora_path=["self_cognition_Alice_flattened"]),
            EXPECTED_OUTPUT,
            "Output after applying LoRA via flattened bucket does not match expected",
        )


if __name__ == "__main__":
    unittest.main()
