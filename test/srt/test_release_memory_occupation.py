"""Test memory release and resume operations for SGLang engine in hybrid RL training.

This test suite evaluates the SGLang engine's memory management capabilities, focusing
on releasing and resuming memory occupation for KV cache and model weights. It simulates
an RL workflow where the SGLang engine acts as a rollout engine for experience collection.
The process involves initializing the engine, sending a small number of requests to simulate
rollout, releasing memory to mimic offloading during RL training, resuming memory occupation,
updating weights with a trained HuggingFace model, and verifying the updated weights.

Detailed in our proposal (https://github.com/sgl-project/sglang/pull/7099), two test cases
are included:

1. Basic Release and Resume: Uses a lower mem_fraction_static (0.6) to control memory allocation
and avoid OOM errors carefully. This test simulates a scenario without multi-stage memory management,
ensuring the engine can release and resume memory occupation while maintaining functionality after
weight updates.

2. Multi-Stage Release and Resume: Employs a higher mem_fraction_static (0.85) to simulate higher
memory pressure, leveraging multi-stage memory management. It sequentially releases and resumes
KV cache and model weights, verifying memory deallocation and reallocation at each stage, and
ensuring correct weight updates and text generation.

3. Tensor Parallel Tests: Tests memory release and resume operations with different tensor parallel
configurations (tp=1, tp=2) to ensure proper memory management in distributed settings. For different
data parallel size, we test it in verl.
"""

import time
import unittest

import torch
from transformers import AutoModelForCausalLM

import sglang as sgl
from sglang.srt.constants import GPU_MEMORY_TYPE_KV_CACHE, GPU_MEMORY_TYPE_WEIGHTS
from sglang.test.test_utils import (
    DEFAULT_SMALL_MODEL_NAME_FOR_TEST,
    DEFAULT_SMALL_MODEL_NAME_FOR_TEST_BASE,
    DEFAULT_SMALL_MOE_MODEL_NAME_FOR_TEST_BASE,
    DEFAULT_SMALL_MOE_MODEL_NAME_FOR_TEST_CHAT,
    CustomTestCase,
)

# (temporarily) set to true to observe memory usage in nvidia-smi more clearly
_DEBUG_EXTRA = False


def get_gpu_memory_gb():
    return torch.cuda.device_memory_used() / 1024**3


class TestReleaseMemoryOccupation(CustomTestCase):
    def _setup_engine(
        self,
        model_name,
        mem_fraction_static=0.8,
        tp_size=1,
        ep_size=1,
        enable_weights_cpu_backup=False,
    ):
        """Common setup for engine and HF model."""
        engine = sgl.Engine(
            model_path=model_name,
            random_seed=42,
            enable_memory_saver=True,
            mem_fraction_static=mem_fraction_static,
            tp_size=tp_size,
            ep_size=ep_size,
            enable_weights_cpu_backup=enable_weights_cpu_backup,
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
            "prompt_moe": "The weather is nice today, and I want to",
            "sampling_params_moe": {"temperature": 0, "max_new_tokens": 16},
            "expect_output_before_update_weights_moe": " go to the park. I have a picnic basket, a book, and a",
            "expect_output_after_update_weights_moe": " go to the park. I have a lot of things to do, but I",
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

    def test_release_and_resume_occupation(self):
        # Without multi-stage release and resume, we need to carefully control the memory fraction to avoid OOM
        model_name = DEFAULT_SMALL_MODEL_NAME_FOR_TEST
        assert (
            torch.cuda.device_count() >= 2
        ), "Need at least 2 GPUs for tensor parallel tests"

        for tp_size in [1, 2]:

            print(f"Testing tp_size={tp_size} for test_release_and_resume_occupation")
            engine = self._setup_engine(
                model_name=model_name, mem_fraction_static=0.6, tp_size=tp_size
            )
            params = self._common_test_params()

            self._test_initial_generation(
                engine,
                params["prompt"],
                params["sampling_params"],
                params["expect_output_before_update_weights"],
            )

            t = time.perf_counter()
            gpu_memory_usage_before_release = get_gpu_memory_gb()
            engine.release_memory_occupation()
            gpu_memory_usage_after_release = get_gpu_memory_gb()

            self.assertLess(
                gpu_memory_usage_after_release,
                gpu_memory_usage_before_release,
            )

            print(
                f"Release took {time.perf_counter() - t:.2f}s, memory: {gpu_memory_usage_before_release:.1f} GB → {gpu_memory_usage_after_release:.1f} GB"
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
            engine.update_weights_from_tensor(list(hf_model_new.named_parameters()))

            # destroy the hf model
            del hf_model_new
            torch.cuda.empty_cache()

            print("generate (#2)")
            outputs = engine.generate(params["prompt"], params["sampling_params"])[
                "text"
            ]
            self.assertEqual(outputs, params["expect_output_after_update_weights"])
            engine.shutdown()

    def test_release_and_resume_occupation_with_weights_cpu_backup(self):
        # Test release and resume occupation with weights CPU backup
        model_name = DEFAULT_SMALL_MODEL_NAME_FOR_TEST

        print("Testing test_release_and_resume_occupation_with_weights_cpu_backup")
        engine = self._setup_engine(
            model_name=model_name,
            mem_fraction_static=0.6,
            enable_weights_cpu_backup=True,
        )
        params = self._common_test_params()

        self._test_initial_generation(
            engine,
            params["prompt"],
            params["sampling_params"],
            params["expect_output_before_update_weights"],
        )

        t = time.perf_counter()
        gpu_memory_usage_before_release = get_gpu_memory_gb()
        engine.release_memory_occupation()
        gpu_memory_usage_after_release = get_gpu_memory_gb()

        self.assertLess(
            gpu_memory_usage_after_release,
            gpu_memory_usage_before_release,
        )

        print(
            f"Release took {time.perf_counter() - t:.2f}s, memory: {gpu_memory_usage_before_release:.1f} GB → {gpu_memory_usage_after_release:.1f} GB"
        )

        if _DEBUG_EXTRA:
            time.sleep(3)

        t = time.perf_counter()
        engine.resume_memory_occupation()
        print(
            f"Resume took {time.perf_counter() - t:.2f}s, memory: {get_gpu_memory_gb():.1f} GB"
        )

        print("generate post resume")
        outputs = engine.generate(params["prompt"], params["sampling_params"])["text"]
        self.assertEqual(outputs, params["expect_output_before_update_weights"])
        engine.shutdown()

    def test_multi_stage_release_and_resume(self):
        # With multi-stage release and resume, we can set the memory fraction to 0.85 without concern of OOM
        model_name = DEFAULT_SMALL_MODEL_NAME_FOR_TEST

        for tp_size in [1, 2]:
            if tp_size == 2 and torch.cuda.device_count() < 2:
                continue

            print(f"Testing tp_size={tp_size} for test_multi_stage_release_and_resume")
            engine = sgl.Engine(
                model_path=model_name,
                random_seed=42,
                enable_memory_saver=True,
                mem_fraction_static=0.85,  # Higher memory pressure
                tp_size=tp_size,
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
            engine.release_memory_occupation(tags=[GPU_MEMORY_TYPE_KV_CACHE])

            gpu_memory_usage_after_release_kv_cache = get_gpu_memory_gb()

            self.assertLess(
                gpu_memory_usage_after_release_kv_cache,
                gpu_memory_usage_before_release_kv_cache,
            )
            engine.release_memory_occupation(tags=[GPU_MEMORY_TYPE_WEIGHTS])

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

            # gpu_memory_usage_after_release_weights and gpu_memory_usage_before_resume_weights should be close

            self.assertAlmostEqual(
                gpu_memory_usage_after_release_weights,
                gpu_memory_usage_before_resume_weights,
                delta=3.0,
            )
            print(f"Resume weights took {time.perf_counter() - t:.2f}s")

            engine.resume_memory_occupation(tags=[GPU_MEMORY_TYPE_WEIGHTS])
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
            engine.update_weights_from_tensor(list(hf_model_new.named_parameters()))

            # destroy the hf model
            del hf_model_new
            torch.cuda.empty_cache()
            engine.resume_memory_occupation(tags=[GPU_MEMORY_TYPE_KV_CACHE])

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
            outputs = engine.generate(params["prompt"], params["sampling_params"])[
                "text"
            ]
            self.assertEqual(outputs, params["expect_output_after_update_weights"])
            engine.shutdown()

    def test_moe_model_release_and_resume(self):
        # Test with MoE model
        model_name = DEFAULT_SMALL_MOE_MODEL_NAME_FOR_TEST_CHAT

        tp_size = ep_size = 2

        print(
            f"Testing tp_size={tp_size} and ep_size={ep_size} for test_moe_model_release_and_resume"
        )
        engine = sgl.Engine(
            model_path=model_name,
            random_seed=42,
            enable_memory_saver=True,
            mem_fraction_static=0.5,
            tp_size=tp_size,
            ep_size=ep_size,
        )
        params = self._common_test_params()

        self._test_initial_generation(
            engine,
            params["prompt_moe"],
            params["sampling_params_moe"],
            params["expect_output_before_update_weights_moe"],
        )

        t = time.perf_counter()
        gpu_memory_usage_before_release = get_gpu_memory_gb()
        engine.release_memory_occupation()
        gpu_memory_usage_after_release = get_gpu_memory_gb()
        self.assertLess(
            gpu_memory_usage_after_release,
            gpu_memory_usage_before_release,
        )

        print(
            f"Release took {time.perf_counter() - t:.2f}s, memory: {gpu_memory_usage_before_release:.1f} GB → {gpu_memory_usage_after_release:.1f} GB"
        )

        if _DEBUG_EXTRA:
            time.sleep(3)

        t = time.perf_counter()
        engine.resume_memory_occupation()
        print(
            f"Resume took {time.perf_counter() - t:.2f}s, memory: {get_gpu_memory_gb():.1f} GB"
        )

        hf_model_new = AutoModelForCausalLM.from_pretrained(
            DEFAULT_SMALL_MOE_MODEL_NAME_FOR_TEST_BASE,
            torch_dtype="bfloat16",
            device_map="cuda",
        )
        engine.update_weights_from_tensor(list(hf_model_new.named_parameters()))

        # destroy the hf model
        del hf_model_new
        torch.cuda.empty_cache()

        print("generate (#2)")
        outputs = engine.generate(params["prompt_moe"], params["sampling_params_moe"])[
            "text"
        ]
        self.assertEqual(outputs, params["expect_output_after_update_weights_moe"])
        engine.shutdown()


if __name__ == "__main__":
    unittest.main()
