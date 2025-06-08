import time
import unittest

import torch
from transformers import AutoModelForCausalLM

import sglang as sgl
from sglang.test.test_utils import DEFAULT_SMALL_MODEL_NAME_FOR_TEST, CustomTestCase

# Change to local path to bypass internet access
DEFAULT_SMALL_MODEL_NAME_FOR_TEST = "/shared/public/elr-models/meta-llama/Llama-3.2-1B-Instruct/e9f8effbab1cbdc515c11ee6e098e3d5a9f51e14"

# (temporarily) set to true to observe memory usage in nvidia-smi more clearly
_DEBUG_EXTRA = True


class TestReleaseMemoryOccupation(CustomTestCase):
    def _setup_engine_and_model(self):
        """Common setup for engine and HF model."""
        model_name = DEFAULT_SMALL_MODEL_NAME_FOR_TEST
        engine = sgl.Engine(
            model_path=model_name,
            random_seed=42,
            enable_memory_saver=True,
            mem_fraction_static=0.9,
            # disable_cuda_graph=True,  # for debugging only
        )
        hf_model_new = AutoModelForCausalLM.from_pretrained(
            model_name, torch_dtype="bfloat16"
        )
        return engine, hf_model_new, model_name

    def _common_test_params(self):
        """Common test parameters."""
        return {
            "prompt": "Today is a sunny day and I like",
            "sampling_params": {"temperature": 0, "max_new_tokens": 8},
            "expect_output": " to spend it outdoors. I decided to",
        }

    def _test_initial_generation(self, engine, prompt, sampling_params, expect_output):
        """Test initial generation and memory allocation."""
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

    def _test_final_generation_and_cleanup(
        self, engine, hf_model_new, prompt, sampling_params, expect_output
    ):
        """Test final generation, weight update, and cleanup."""
        self.assertEqual(
            _try_allocate_big_tensor(),
            False,
            "Should not be able to allocate big tensors after resuming",
        )

        print("update_weights_from_tensor")
        # As if: PPO/RL Engine has updated hf model's weights, and now we sync it to SGLang
        engine.update_weights_from_tensor(list(hf_model_new.named_parameters()))

        print("generate (#2)")
        outputs = engine.generate(prompt, sampling_params)["text"]
        self.assertEqual(outputs, expect_output)

        if _DEBUG_EXTRA:
            time.sleep(4)

        engine.shutdown()

    def test_release_and_resume_occupation(self):
        engine, hf_model_new, model_name = self._setup_engine_and_model()
        params = self._common_test_params()

        self._test_initial_generation(engine, **params)

        print("release_memory_occupation start")
        t = time.perf_counter()
        engine.release_memory_occupation()
        if _DEBUG_EXTRA:
            print("release_memory_occupation", time.perf_counter() - t)

        if _DEBUG_EXTRA:
            time.sleep(5)

        self.assertEqual(
            _try_allocate_big_tensor(),
            True,
            "Should be able to allocate big tensors aftre releasing",
        )

        if _DEBUG_EXTRA:
            time.sleep(5)

        print("resume_memory_occupation start")
        t = time.perf_counter()
        engine.resume_memory_occupation()
        if _DEBUG_EXTRA:
            print("resume_memory_occupation", time.perf_counter() - t)

        self._test_final_generation_and_cleanup(engine, hf_model_new, **params)

    def test_multi_stage_release_and_resume(self):
        engine, hf_model_new, model_name = self._setup_engine_and_model()
        params = self._common_test_params()

        self._test_initial_generation(engine, **params)

        print("release_memory_occupation start")
        t = time.perf_counter()
        gpu_memory_usage_before_release_kv_cache = get_gpu_memory_gb()
        engine.release_memory_occupation(tags=["kv_cache"])

        gpu_memory_usage_after_release_kv_cache = get_gpu_memory_gb()

        self.assertLess(
            gpu_memory_usage_after_release_kv_cache,
            gpu_memory_usage_before_release_kv_cache,
        )
        engine.release_memory_occupation(tags=["weights"])

        gpu_memory_usage_after_release_weights = get_gpu_memory_gb()

        self.assertLess(
            gpu_memory_usage_after_release_weights,
            gpu_memory_usage_after_release_kv_cache,
        )

        if _DEBUG_EXTRA:
            print("release_memory_occupation", time.perf_counter() - t)
            print(
                f"gpu_memory_usage_before_release_kv_cache: {gpu_memory_usage_before_release_kv_cache} GB"
            )
            print(
                f"gpu_memory_usage_after_release_kv_cache: {gpu_memory_usage_after_release_kv_cache} GB"
            )
            print(
                f"gpu_memory_usage_after_release_weights: {gpu_memory_usage_after_release_weights} GB"
            )

        if _DEBUG_EXTRA:
            time.sleep(5)

        self.assertEqual(
            _try_allocate_big_tensor(),
            True,
            "Should be able to allocate big tensors aftre releasing",
        )

        if _DEBUG_EXTRA:
            time.sleep(5)

        print("resume_memory_occupation start")
        t = time.perf_counter()
        gpu_memory_usage_before_resume_kv_cache = get_gpu_memory_gb()
        engine.resume_memory_occupation(tags=["weights"])

        gpu_memory_usage_after_resume_weights = get_gpu_memory_gb()
        self.assertGreater(
            gpu_memory_usage_after_resume_weights,
            gpu_memory_usage_before_resume_kv_cache,
        )
        engine.resume_memory_occupation(tags=["kv_cache"])

        gpu_memory_usage_after_resume_kv_cache = get_gpu_memory_gb()
        self.assertGreater(
            gpu_memory_usage_after_resume_kv_cache,
            gpu_memory_usage_after_resume_weights,
        )

        if _DEBUG_EXTRA:
            print("resume_memory_occupation", time.perf_counter() - t)
            print(
                f"gpu_memory_usage_before_resume_kv_cache: {gpu_memory_usage_before_resume_kv_cache} GB"
            )
            print(
                f"gpu_memory_usage_after_resume_weights: {gpu_memory_usage_after_resume_weights} GB"
            )
            print(
                f"gpu_memory_usage_after_resume_kv_cache: {gpu_memory_usage_after_resume_kv_cache} GB"
            )

        self._test_final_generation_and_cleanup(engine, hf_model_new, **params)


def _try_allocate_big_tensor(size: int = 20_000_000_000):
    try:
        torch.empty((size,), dtype=torch.uint8, device="cuda")
        torch.cuda.empty_cache()
        return True
    except torch.cuda.OutOfMemoryError:
        return False


import subprocess


def get_gpu_memory_gb(gpu_id=0):
    cmd = f"nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits --id={gpu_id}"
    result = subprocess.check_output(cmd, shell=True, text=True).strip()
    return int(result) / 1024


if __name__ == "__main__":
    unittest.main()
