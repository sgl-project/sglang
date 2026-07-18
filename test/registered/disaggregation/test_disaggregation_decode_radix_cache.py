import asyncio
import os
import shutil
import tempfile
import time
import unittest
from types import SimpleNamespace

import requests

from sglang.benchmark.datasets.random import sample_random_requests
from sglang.benchmark.utils import get_tokenizer
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.kits.cache_hit_kit import (
    async_request_sglang_generate,
    gen_payload,
    run_multiturn_cache_hit_test,
)
from sglang.test.run_eval import run_eval
from sglang.test.server_fixtures.disaggregation_fixture import (
    PDDisaggregationServerBase,
)
from sglang.test.test_utils import (
    DEFAULT_MODEL_NAME_FOR_TEST,
    is_in_ci,
    try_cached_model,
)

register_cuda_ci(est_time=300, stage="base-c", runner_config="8-gpu-h20")


def _has_nixl():
    try:
        import nixl._api  # noqa: F401
    except ImportError:
        return False
    return True


def _has_mooncake():
    try:
        import mooncake.engine  # noqa: F401
    except ImportError:
        return False
    return True


class DisaggregationDecodeRadixCacheTestMixin:
    extra_decode_args = ["--disaggregation-decode-enable-radix-cache"]
    transfer_backend_name = None

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.model = try_cached_model(DEFAULT_MODEL_NAME_FOR_TEST)
        cls.transfer_backend = [
            "--disaggregation-transfer-backend",
            cls.transfer_backend_name,
        ]
        cls.launch_all()

    def _assert_process_healthy(self, name, process, url):
        self.assertIsNotNone(process, f"{name} process was not started")
        self.assertIsNone(
            process.poll(),
            f"{name} exited unexpectedly with code {process.returncode}",
        )
        response = requests.get(f"{url}/health", timeout=10)
        response.raise_for_status()

    def test_decode_radix_cache_hits_and_workers_stay_alive(self):
        decode_info = requests.get(f"{self.decode_url}/server_info", timeout=10).json()
        self.assertFalse(
            decode_info.get("disable_radix_cache", True),
            "decode server did not enable radix cache",
        )

        result = run_multiturn_cache_hit_test(
            base_url=self.base_url,
            model_path=self.model,
            num_clients=4,
            num_rounds=3,
            request_length=384,
            output_length=64,
            max_parallel=4,
        )
        self.assertGreater(
            result["overall"]["total_cached_tokens"],
            0,
            "expected decode radix cache to reuse at least some tokens",
        )

        # Give the schedulers a short idle window so any post-request leak/crash
        # paths have a chance to surface before the liveness checks below.
        time.sleep(5)

        self._assert_process_healthy("load balancer", self.process_lb, self.lb_url)
        self._assert_process_healthy("prefill", self.process_prefill, self.prefill_url)
        self._assert_process_healthy("decode", self.process_decode, self.decode_url)

    def test_gsm8k_accuracy_two_passes(self):
        """Run GSM8K twice to verify decode radix cache does not degrade accuracy."""
        args = SimpleNamespace(
            base_url=self.base_url,
            model=self.model,
            eval_name="gsm8k",
            api="completion",
            max_tokens=512,
            num_examples=500,
            num_threads=100,
            num_shots=6,
        )

        metrics_first = run_eval(args)
        print(f"First run metrics: {metrics_first}")

        metrics_second = run_eval(args)
        print(f"Second run metrics: {metrics_second}")

        self.assertGreater(metrics_first["score"], 0.80)
        self.assertGreater(metrics_second["score"], 0.80)

        accuracy_drop = metrics_first["score"] - metrics_second["score"]
        self.assertLessEqual(
            accuracy_drop,
            0.03,
            f"Second run accuracy dropped by {accuracy_drop:.4f} "
            f"(first={metrics_first['score']:.4f}, second={metrics_second['score']:.4f}), "
            f"exceeds 3% threshold",
        )


@unittest.skipUnless(
    is_in_ci() or _has_nixl(),
    "NIXL is required for decode radix cache disaggregation coverage.",
)
class TestDisaggregationDecodeRadixCacheNixl(
    DisaggregationDecodeRadixCacheTestMixin, PDDisaggregationServerBase
):
    kv_size_thres = 67516.4  # auto; update_memory_thresholds.py
    transfer_backend_name = "nixl"


@unittest.skipUnless(
    is_in_ci() or _has_mooncake(),
    "Mooncake is required for decode radix cache disaggregation coverage.",
)
class TestDisaggregationDecodeRadixCacheMooncake(
    DisaggregationDecodeRadixCacheTestMixin, PDDisaggregationServerBase
):
    kv_size_thres = 67516.4  # auto; update_memory_thresholds.py
    transfer_backend_name = "mooncake"


@unittest.skipUnless(
    is_in_ci() or _has_mooncake(),
    "Mooncake is required for decode radix cache disaggregation coverage.",
)
class TestDisaggregationDecodeRadixHiCacheFileBackend(PDDisaggregationServerBase):
    kv_size_thres = 67388.0  # auto; update_memory_thresholds.py
    extra_prefill_args = [
        "--enable-hierarchical-cache",
        "--hicache-ratio",
        "1.2",
        "--hicache-write-policy",
        "write_through",
        "--hicache-storage-backend",
        "file",
        "--hicache-storage-prefetch-policy",
        "wait_complete",
        "--hicache-io-backend",
        "kernel",
        "--hicache-mem-layout",
        "page_first",
        "--page-size",
        "64",
    ]
    extra_decode_args = [
        "--disaggregation-decode-enable-radix-cache",
        *extra_prefill_args,
    ]
    transfer_backend_name = "mooncake"

    @classmethod
    def setUpClass(cls):
        cls.hicache_dir = tempfile.mkdtemp(prefix="sglang-hicache-")
        os.environ["SGLANG_HICACHE_FILE_BACKEND_STORAGE_DIR"] = cls.hicache_dir

        super().setUpClass()
        cls.model = try_cached_model(DEFAULT_MODEL_NAME_FOR_TEST)
        cls.transfer_backend = [
            "--disaggregation-transfer-backend",
            cls.transfer_backend_name,
        ]
        cls.launch_all()

    @classmethod
    def tearDownClass(cls):
        super().tearDownClass()
        os.environ.pop("SGLANG_HICACHE_FILE_BACKEND_STORAGE_DIR", None)
        shutil.rmtree(cls.hicache_dir, ignore_errors=True)

    def _post_ok(self, url):
        response = requests.post(url, timeout=60)
        response.raise_for_status()

    def _flush_memory_cache(self):
        self._post_ok(f"{self.prefill_url}/flush_cache?timeout=30")
        self._post_ok(f"{self.decode_url}/flush_cache?timeout=30")

    def _generate(self, input_ids, output_len):
        output = asyncio.run(
            async_request_sglang_generate(
                gen_payload(input_ids, output_len),
                f"{self.base_url}/generate",
            )
        )
        self.assertTrue(output.success, output.error)
        return output

    def _sample_token_ids(self, input_len, output_len, num_prompts=1):
        tokenizer = get_tokenizer(self.model)
        return [
            list(request.prompt)
            for request in sample_random_requests(
                input_len=input_len,
                output_len=output_len,
                num_prompts=num_prompts,
                range_ratio=1.0,
                tokenizer=tokenizer,
                dataset_path="",
                return_text=False,
            )
        ]

    def test_decode_hicache_file_backend_l3_reuses_decode_output_after_flush(self):
        self._post_ok(f"{self.decode_url}/hicache/storage-backend/clear")
        self._flush_memory_cache()

        num_rounds = 5
        output_len = 64
        history = self._sample_token_ids(
            input_len=256, output_len=output_len, num_prompts=1
        )[0]
        suffixes = self._sample_token_ids(
            input_len=64, output_len=output_len, num_prompts=num_rounds - 1
        )

        prev_prompt_len = 0
        prev_output_len = 0
        for round_idx in range(num_rounds):
            output = self._generate(history, output_len)
            if round_idx == 0:
                self.assertEqual(output.cached_tokens, 0)
            else:
                self.assertGreaterEqual(
                    output.cached_tokens,
                    prev_prompt_len + prev_output_len,
                )

            history.extend(output.output_ids)
            prev_prompt_len = output.prompt_len
            prev_output_len = len(output.output_ids)

            if round_idx < num_rounds - 1:
                history.extend(suffixes[round_idx])
                time.sleep(1)
                self._flush_memory_cache()


if __name__ == "__main__":
    unittest.main()
