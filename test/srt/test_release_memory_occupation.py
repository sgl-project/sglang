import subprocess
import time
import unittest

import torch
from transformers import AutoModelForCausalLM

import sglang as sgl
from sglang.test.test_utils import (
    DEFAULT_SMALL_MODEL_NAME_FOR_TEST,
    DEFAULT_SMALL_MODEL_NAME_FOR_TEST_BASE,
    CustomTestCase,
)

# (temporarily) set to true to observe memory usage in nvidia-smi more clearly
_DEBUG_EXTRA = False


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
            "expect_output_before_update_weights": " to spend it outdoors. I decided to",
            "expect_output_after_update_weights": " to go for a walk. I like",
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

    def _test_update_weights(
        self,
        engine,
        hf_model_new,
        is_multi_stage,
    ):
        """Test final generation, weight update, and cleanup."""
        print("update_weights_from_tensor")
        # RL Engine has updated hf model's weights by model training, and now we sync it with SGLang
        engine.update_weights_from_tensor(list(hf_model_new.named_parameters()))

        if _DEBUG_EXTRA:
            time.sleep(3)

    def test_release_and_resume_occupation(self):
        # Without multi-stage release and resume, we need to carefully control the memory fraction to avoid OOM
        model_name = DEFAULT_SMALL_MODEL_NAME_FOR_TEST
        engine = self._setup_engine(model_name=model_name, mem_fraction_static=0.6)
        params = self._common_test_params()

        self._test_initial_generation(
            engine,
            params["prompt"],
            params["sampling_params"],
            params["expect_output_before_update_weights"],
        )

        t = time.perf_counter()
        engine.release_memory_occupation()
        print(
            f"Release took {time.perf_counter() - t:.2f}s, memory: {get_gpu_memory_gb():.1f} GB"
        )

        if _DEBUG_EXTRA:
            time.sleep(3)

        t = time.perf_counter()
        engine.resume_memory_occupation()
        print(
            f"Resume took {time.perf_counter() - t:.2f}s, memory: {get_gpu_memory_gb():.1f} GB"
        )

        hf_model_new = AutoModelForCausalLM.from_pretrained(
            DEFAULT_SMALL_MODEL_NAME_FOR_TEST_BASE,
            torch_dtype="bfloat16",
            device_map="cuda",
        )
        self._test_update_weights(engine, hf_model_new, is_multi_stage=False)

        # destroy the hf model
        del hf_model_new
        torch.cuda.empty_cache()

        print("generate (#2)")
        outputs = engine.generate(params["prompt"], params["sampling_params"])["text"]
        self.assertEqual(outputs, params["expect_output_after_update_weights"])
        engine.shutdown()

    def test_multi_stage_release_and_resume(self):
        # With multi-stage release and resume, we can set the memory fraction to 0.9 without concern of OOM
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

        print(f"Release took {time.perf_counter() - t:.2f}s")
        print(
            f"Memory: {gpu_memory_usage_before_release_kv_cache:.1f} → {gpu_memory_usage_after_release_kv_cache:.1f} → {gpu_memory_usage_after_release_weights:.1f} GB"
        )

        if _DEBUG_EXTRA:
            time.sleep(3)

        t = time.perf_counter()
        gpu_memory_usage_before_resume_weights = get_gpu_memory_gb()
        engine.resume_memory_occupation(tags=["weights"])
        gpu_memory_usage_after_resume_weights = get_gpu_memory_gb()

        self.assertGreater(
            gpu_memory_usage_after_resume_weights,
            gpu_memory_usage_before_resume_weights,
        )

        # Update weights from a trained model to serving engine, and then destroy the trained model
        hf_model_new = AutoModelForCausalLM.from_pretrained(
            DEFAULT_SMALL_MODEL_NAME_FOR_TEST_BASE,
            torch_dtype="bfloat16",
            device_map="cuda",
        )
        gpu_memory_usage_after_loaded_hf_model = get_gpu_memory_gb()
        self._test_update_weights(engine, hf_model_new, is_multi_stage=True)

        # destroy the hf model
        del hf_model_new
        torch.cuda.empty_cache()
        engine.resume_memory_occupation(tags=["kv_cache"])

        gpu_memory_usage_after_resume_kv_cache = get_gpu_memory_gb()
        self.assertGreater(
            gpu_memory_usage_after_resume_kv_cache,
            gpu_memory_usage_after_resume_weights,
        )

        print(f"Resume + update took {time.perf_counter() - t:.2f}s")
        print(
            f"Memory: {gpu_memory_usage_before_resume_weights:.1f} → {gpu_memory_usage_after_resume_weights:.1f} → {gpu_memory_usage_after_loaded_hf_model:.1f} → {gpu_memory_usage_after_resume_kv_cache:.1f} GB"
        )

        print("generate (#2)")
        outputs = engine.generate(params["prompt"], params["sampling_params"])["text"]
        self.assertEqual(outputs, params["expect_output_after_update_weights"])
        engine.shutdown()


def get_gpu_memory_gb():
    return torch.cuda.device_memory_used() / 1024**3


if __name__ == "__main__":
    unittest.main()
