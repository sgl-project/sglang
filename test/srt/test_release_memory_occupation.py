import time
import unittest

import torch
from transformers import AutoModelForCausalLM

import sglang as sgl
from sglang.test.test_utils import DEFAULT_SMALL_MODEL_NAME_FOR_TEST, CustomTestCase

# TODO: Remove this line when PR is ready, we need this to bypass internet issue during development
DEFAULT_SMALL_MODEL_NAME_FOR_TEST = "/shared/public/elr-models/meta-llama/Meta-Llama-3.1-8B-Instruct/07eb05b21d191a58c577b4a45982fe0c049d0693"
MOCK_UPDATED_MODEL_NAME_FOR_TEST = "/shared/public/elr-models/meta-llama/Meta-Llama-3.1-8B/48d6d0fc4e02fb1269b36940650a1b7233035cbb"

# (temporarily) set to true to observe memory usage in nvidia-smi more clearly
_DEBUG_EXTRA = True


class TestReleaseMemoryOccupation(CustomTestCase):
    def _setup_engine(self, model_name, mem_fraction_static=0.8):
        """Common setup for engine and HF model."""
        engine = sgl.Engine(
            model_path=model_name,
            random_seed=42,
            enable_memory_saver=True,
            mem_fraction_static=mem_fraction_static,
            # disable_cuda_graph=True,  # for debugging only
        )

        return engine

    def _common_test_params(self):
        """Common test parameters."""
        return {
            "prompt": "Today is a sunny day and I like",
            "sampling_params": {"temperature": 0, "max_new_tokens": 8},
            "expect_output_before_update_weights": " to take a walk in the park.",
            "expect_output_after_update_weights": " to go out for a walk. I",
        }

    def _test_initial_generation(
        self, engine, prompt, sampling_params, expect_output_before_update_weights
    ):
        """Test initial generation and memory allocation."""
        print("generate (#1)")
        outputs = engine.generate(prompt, sampling_params)["text"]
        self.assertEqual(outputs, expect_output_before_update_weights)

        if _DEBUG_EXTRA:
            time.sleep(3)
            print(
                f"gpu_memory_usage_after_initial_generation: {get_gpu_memory_gb()} GB"
            )

        self.assertEqual(
            _try_allocate_big_tensor(),
            False,
            "Should not be able to allocate big tensors before releasing",
        )

    def _test_update_weights(
        self,
        engine,
        model_name,
        is_multi_stage,
    ):
        """Test final generation, weight update, and cleanup."""
        hf_model_new = AutoModelForCausalLM.from_pretrained(
            model_name, torch_dtype="bfloat16", device_map="cuda"
        )

        if is_multi_stage:
            # Even though we have two copies of weights, we can still allocate big tensors since we haven't resumed kv cache
            self.assertEqual(
                _try_allocate_big_tensor(),
                True,
                "Should be able to allocate big tensors for multi-stage release and resume",
            )
        else:
            # With two copies of weights plus the kv cache, we cannot allocate big tensors
            self.assertEqual(
                _try_allocate_big_tensor(),
                False,
                "Should not be able to allocate big tensors for naive release and resume",
            )

        print("update_weights_from_tensor")
        # As if: PPO/RL Engine has updated hf model's weights, and now we sync it to SGLang
        engine.update_weights_from_tensor(list(hf_model_new.named_parameters()))

        print("gpu_memory_usage_during_update_weights", get_gpu_memory_gb())

        # destroy the hf model
        del hf_model_new
        torch.cuda.empty_cache()

        if _DEBUG_EXTRA:
            time.sleep(4)

    def test_release_and_resume_occupation(self):
        # Without multi-stage release and resume, we need to carefully control the memory fraction to avoid OOM
        model_name = DEFAULT_SMALL_MODEL_NAME_FOR_TEST
        engine = self._setup_engine(model_name=model_name, mem_fraction_static=0.8)
        params = self._common_test_params()

        self._test_initial_generation(
            engine,
            params["prompt"],
            params["sampling_params"],
            params["expect_output_before_update_weights"],
        )

        print("release_memory_occupation start")
        t = time.perf_counter()
        engine.release_memory_occupation()
        if _DEBUG_EXTRA:
            print("release_memory_occupation", time.perf_counter() - t)
            print(f"gpu_memory_usage_after_release: {get_gpu_memory_gb()} GB")

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
            print(f"gpu_memory_usage_after_resume: {get_gpu_memory_gb()} GB")
        self._test_update_weights(
            engine, MOCK_UPDATED_MODEL_NAME_FOR_TEST, is_multi_stage=False
        )

        print("generate (#2)")
        outputs = engine.generate(params["prompt"], params["sampling_params"])["text"]
        self.assertEqual(outputs, params["expect_output_after_update_weights"])
        engine.shutdown()

    def test_multi_stage_release_and_resume(self):
        # With multi-stage release and resume, we can set the memory fraction to 0.95 without concern of OOM
        # Use TP2 to test multi-process environment (similar to GRPO's Ray workers)
        model_name = DEFAULT_SMALL_MODEL_NAME_FOR_TEST
        engine = sgl.Engine(
            model_path=model_name,
            random_seed=42,
            enable_memory_saver=True,
            mem_fraction_static=0.9,  # Higher memory pressure
            tp_size=2,  # Enable TP2 for multi-process testing
        )
        params = self._common_test_params()

        self._test_initial_generation(
            engine,
            params["prompt"],
            params["sampling_params"],
            params["expect_output_before_update_weights"],
        )

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
        gpu_memory_usage_before_resume_weights = get_gpu_memory_gb()
        engine.resume_memory_occupation(tags=["weights"])

        # Update weights from a trained model to serving engine, and then destroy the trained model
        self._test_update_weights(
            engine, MOCK_UPDATED_MODEL_NAME_FOR_TEST, is_multi_stage=True
        )

        gpu_memory_usage_after_resume_weights = get_gpu_memory_gb()
        self.assertGreater(
            gpu_memory_usage_after_resume_weights,
            gpu_memory_usage_before_resume_weights,
        )
        engine.resume_memory_occupation(tags=["kv_cache"])

        gpu_memory_usage_after_resume_kv_cache = get_gpu_memory_gb()
        self.assertGreater(
            gpu_memory_usage_after_resume_kv_cache,
            gpu_memory_usage_after_resume_weights,
        )

        if _DEBUG_EXTRA:
            print(
                "resume_memory_occupation and update weights", time.perf_counter() - t
            )
            print(
                f"gpu_memory_usage_before_resume_weights: {gpu_memory_usage_before_resume_weights} GB"
            )
            print(
                f"gpu_memory_usage_after_resume_weights: {gpu_memory_usage_after_resume_weights} GB"
            )
            print(
                f"gpu_memory_usage_after_resume_kv_cache: {gpu_memory_usage_after_resume_kv_cache} GB"
            )

        print("generate (#2)")
        outputs = engine.generate(params["prompt"], params["sampling_params"])["text"]
        self.assertEqual(outputs, params["expect_output_after_update_weights"])

        engine.shutdown()


def _try_allocate_big_tensor(size: int = 20_000_000_000):
    if torch.cuda.get_device_properties(0).total_memory / (1024**3) > 100:
        size = 30_000_000_000

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
