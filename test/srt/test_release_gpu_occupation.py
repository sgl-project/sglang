import time
import unittest

import sglang as sgl
from sglang.test.test_utils import DEFAULT_SMALL_MODEL_NAME_FOR_TEST
from transformers import AutoModelForCausalLM

# (temporarily) set to true to observe memory usage in nvidia-smi more clearly
_DEBUG_EXTRA = True


class TestReleaseGPUOccupation(unittest.TestCase):
    def test_release_and_resume_occupation(self):
        prompt = "Today is a sunny day and I like"
        expect_output = " to spend it outdoors. I decided to"
        sampling_params = {"temperature": 0, "max_new_tokens": 8}
        model_name = DEFAULT_SMALL_MODEL_NAME_FOR_TEST

        engine = sgl.Engine(model_name=model_name, random_seed=42)
        hf_model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype="bfloat16")

        outputs = engine.generate(prompt, sampling_params)["text"]
        self.assertEqual(outputs, expect_output)

        if _DEBUG_EXTRA:
            time.sleep(3)

        t = time.time()
        engine.release_gpu_occupation()
        if _DEBUG_EXTRA:
            print("release_gpu_occupation", time.time() - t)

        if _DEBUG_EXTRA:
            time.sleep(3)

        t = time.time()
        engine.resume_gpu_occupation()
        if _DEBUG_EXTRA:
            print("resume_gpu_occupation", time.time() - t)

        # As if: PPO has updated hf model's weights, and now we sync it to SGLang
        for name, tensor in hf_model.named_parameters():
            engine.update_weights_from_tensor(name, tensor)

        outputs = engine.generate(prompt, sampling_params)["text"]
        self.assertEqual(outputs, expect_output)

        if _DEBUG_EXTRA:
            time.sleep(5)

        engine.shutdown()


if __name__ == "__main__":
    unittest.main()
