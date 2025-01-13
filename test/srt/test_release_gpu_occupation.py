import time
import unittest

import sglang as sgl
import torch
from sglang.test.test_utils import DEFAULT_SMALL_MODEL_NAME_FOR_TEST
from transformers import AutoModelForCausalLM

# (temporarily) set to true to observe memory usage in nvidia-smi more clearly
_DEBUG_EXTRA = True


class TestReleaseGPUOccupation(unittest.TestCase):
    def test_release_and_resume_occupation(self):
        prompt = "Today is a sunny day and I like"
        sampling_params = {"temperature": 0, "max_new_tokens": 8}
        model_name = DEFAULT_SMALL_MODEL_NAME_FOR_TEST
        expect_output = " to spend it outdoors. I decided to"

        engine = sgl.Engine(
            model_path=model_name,
            random_seed=42,
            memory_saver=True,
            # disable_cuda_graph=True,  # for debugging only
        )
        hf_model_new = AutoModelForCausalLM.from_pretrained(
            model_name, torch_dtype="bfloat16"
        )

        print("generate (#1)")
        outputs = engine.generate(prompt, sampling_params)["text"]
        self.assertEqual(outputs, expect_output)

        if _DEBUG_EXTRA:
            time.sleep(3)

        self.assertEqual(
            _try_allocate_big_tensor(),
            False,
            "Should not be able to allocate big tensors before releasing",
        )

        print("release_gpu_occupation start")
        t = time.time()
        engine.release_gpu_occupation()
        if _DEBUG_EXTRA:
            print("release_gpu_occupation", time.time() - t)

        if _DEBUG_EXTRA:
            time.sleep(5)

        self.assertEqual(
            _try_allocate_big_tensor(),
            True,
            "Should be able to allocate big tensors aftre releasing",
        )

        if _DEBUG_EXTRA:
            time.sleep(5)

        print("resume_gpu_occupation start")
        t = time.time()
        engine.resume_gpu_occupation()
        if _DEBUG_EXTRA:
            print("resume_gpu_occupation", time.time() - t)

        self.assertEqual(
            _try_allocate_big_tensor(),
            False,
            "Should not be able to allocate big tensors after resuming",
        )

        print("update_weights_from_tensor")
        # As if: PPO has updated hf model's weights, and now we sync it to SGLang
        engine.update_weights_from_tensor(list(hf_model_new.named_parameters()))

        print("generate (#2)")
        outputs = engine.generate(prompt, sampling_params)["text"]
        self.assertEqual(outputs, expect_output)

        if _DEBUG_EXTRA:
            time.sleep(4)

        engine.shutdown()


def _try_allocate_big_tensor(size: int = 20_000_000_000):
    try:
        torch.empty((size,), dtype=torch.uint8, device="cuda")
        torch.cuda.empty_cache()
        return True
    except torch.cuda.OutOfMemoryError:
        return False


if __name__ == "__main__":
    unittest.main()
