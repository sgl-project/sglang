"""Regression coverage for MLX KV writes retaining prior prefill chunks (#39675)."""

import gc
import unittest
from types import SimpleNamespace

import mlx.core as mx
import requests
from transformers import AutoTokenizer

from sglang.srt.hardware_backend.mlx.kv_cache import MlxAttentionKVPool
from sglang.srt.hardware_backend.mlx.model_runner import MlxModelRunner
from sglang.test.ci.ci_register import register_mlx_ci
from sglang.test.test_utils import (
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
    terminate_and_kill_process_tree,
    try_cached_model,
)

register_mlx_ci(est_time=60, suite="stage-b-e2e-mlx")


class TestMlxPoolWriteLifetime(CustomTestCase):
    def test_completed_writes_release_source_cache(self):
        """Completed pool writes must not retain every prior request's KV arrays."""
        runner = object.__new__(MlxModelRunner)
        runner._cache_layout = SimpleNamespace(full_attention_layer_indices=(0,))
        runner._attention_kv_pool = MlxAttentionKVPool(
            pool_size=32, num_layers=1, n_kv_heads=4, head_dim=32
        )
        mx.eval(*runner._attention_kv_pool.all_buffers())
        mx.synchronize()
        baseline = mx.get_active_memory()
        try:
            for index in range(8):
                keys = (mx.arange(4 * 4096 * 32) % 32).astype(mx.float16).reshape(
                    1, 4, 4096, 32
                ) + index
                values = keys + 1
                mx.eval(keys, values)
                runner._sync_new_kv_to_pool(
                    cache=[SimpleNamespace(keys=keys, values=values)],
                    cache_start=0,
                    slot_ids=[index + 1],
                )
                del keys, values
                mx.synchronize()

            # Eight one-slot writes must retain less than one 2 MiB source cache.
            self.assertLess(mx.get_active_memory() - baseline, 2 * 1024**2)
            slots = mx.arange(1, 9, dtype=mx.int32)
            actual_k, actual_v = runner._attention_kv_pool.get_kv(0, slots)
            expected = mx.arange(8, dtype=mx.float16)
            self.assertTrue(mx.array_equal(actual_k[:, 0, 0], expected).item())
            self.assertTrue(mx.array_equal(actual_v[:, 0, 0], expected + 1).item())
        finally:
            del runner
            gc.collect()
            mx.synchronize()
            mx.clear_cache()


class TestMlxDefaultMemoryBudget(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = try_cached_model("mlx-community/Qwen3-0.6B-4bit")
        cls.base_url = DEFAULT_URL_FOR_TEST
        cls.process = popen_launch_server(
            cls.model,
            cls.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=[
                "--cuda-graph-backend-decode",
                "disabled",
                "--cuda-graph-backend-prefill",
                "disabled",
                "--nccl-port",
                "29500",
            ],
            env={"SGLANG_USE_MLX": "1"},
        )
        tokenizer = AutoTokenizer.from_pretrained(cls.model)
        cls.prompt = tokenizer.encode(
            "The quick brown fox jumps over the lazy dog. " * 1000,
            add_special_tokens=False,
        )[:8016]

    @classmethod
    def tearDownClass(cls):
        if hasattr(cls, "process") and cls.process:
            terminate_and_kill_process_tree(cls.process)

    def test_chunked_prefill_and_prefix_reuse(self):
        self.assertEqual(len(self.prompt), 8016)
        previous = None
        for _ in range(3):
            response = requests.post(
                self.base_url + "/generate",
                json={
                    "input_ids": self.prompt,
                    "sampling_params": {
                        "temperature": 0,
                        "max_new_tokens": 16,
                        "ignore_eos": True,
                    },
                },
                timeout=120,
            )
            response.raise_for_status()
            result = response.json()
            self.assertEqual(result["meta_info"]["prompt_tokens"], 8016)
            self.assertEqual(result["meta_info"]["completion_tokens"], 16)
            if previous is not None:
                self.assertGreater(result["meta_info"]["cached_tokens"], 0)
                self.assertEqual(result["output_ids"], previous)
            previous = result["output_ids"]
        self.assertEqual(
            requests.get(self.base_url + "/health", timeout=10).status_code, 200
        )


if __name__ == "__main__":
    unittest.main()
