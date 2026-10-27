"""Pure GQA prefixes must survive DCP L2 eviction without Mamba state."""

import functools
import json
import math
import os
import unittest
from concurrent.futures import ThreadPoolExecutor

import requests
from transformers import AutoTokenizer

from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.kl_multiturn_utils import (
    _extract_output_logprobs,
    default_decode_cache_assert,
)
from sglang.test.kl_multiturn_utils import (
    test_input_output_logprobs_match_helper as check_multiturn_logprobs,
)
from sglang.test.kl_test_utils import _flush_cache, compare_kl_divergence
from sglang.test.test_utils import (
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
    terminate_and_kill_process_tree,
)

register_cuda_ci(est_time=180, stage="extra-b", runner_config="4-gpu-h100")


class TestGqaDcpHiCache(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = os.environ.get(
            "GQA_DCP_HICACHE_MODEL", "Qwen/Qwen2.5-1.5B-Instruct"
        )
        cls.base_url = os.environ.get("GQA_DCP_HICACHE_BASE_URL", DEFAULT_URL_FOR_TEST)
        dtype = os.environ.get("GQA_DCP_HICACHE_DTYPE", "bfloat16")
        kv_dtype = {"bfloat16": "bf16", "float16": "auto"}[dtype]
        if "GQA_DCP_HICACHE_BASE_URL" not in os.environ:
            process = popen_launch_server(
                cls.model,
                cls.base_url,
                timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
                other_args=[
                    "--tp-size",
                    "4",
                    "--dcp-size",
                    "2",
                    "--attention-backend",
                    "triton",
                    "--dtype",
                    dtype,
                    "--kv-cache-dtype",
                    kv_dtype,
                    "--page-size",
                    "64",
                    "--max-total-tokens",
                    "4096",
                    "--max-running-requests",
                    "4",
                    "--chunked-prefill-size",
                    "1024",
                    "--enable-hierarchical-cache",
                    "--hicache-size",
                    "2",
                    "--hicache-mem-layout",
                    "page_first_direct",
                    "--hicache-io-backend",
                    "direct",
                    "--hicache-write-policy",
                    "write_through",
                    "--enable-cache-report",
                    "--enable-metrics",
                    "--cuda-graph-backend-prefill",
                    "disabled",
                    "--cuda-graph-max-bs-decode",
                    "4",
                ],
            )
            cls.addClassCleanup(terminate_and_kill_process_tree, process)
        cls.tokenizer = AutoTokenizer.from_pretrained(cls.model)
        info = requests.get(cls.base_url + "/get_server_info", timeout=30).json()
        assert info["dcp_size"] > 1 and info["enable_hierarchical_cache"]
        cls.dcp_size = info["dcp_size"]
        cls.page_size = info["page_size"] * cls.dcp_size
        cls.capacity = info["internal_states"][0]["memory_usage"]["token_capacity"]
        cls.body = cls.tokenizer.encode(
            "Trees absorb carbon dioxide. Water freezes at zero degrees Celsius.\n"
        )

    def prompt(self, number, length=1025):
        header = self.tokenizer.encode(f"Document identifier {number}.\n")
        suffix = self.tokenizer.encode("\nSummarize the document.\nAnswer:")
        return (header + self.body * (length // len(self.body) + 1))[
            : length - len(suffix)
        ] + suffix

    def generate(self, ids, output=64):
        response = requests.post(
            self.base_url + "/generate",
            json={
                "input_ids": ids,
                "sampling_params": {
                    "temperature": 0,
                    "max_new_tokens": output,
                    "ignore_eos": True,
                },
                "return_logprob": True,
                "logprob_start_len": len(ids) - 1,
            },
            timeout=300,
        )
        response.raise_for_status()
        return response.json()

    def load_count(self):
        metrics = requests.get(self.base_url + "/metrics", timeout=30).text
        return sum(
            float(line.rsplit(" ", 1)[1])
            for line in metrics.splitlines()
            if line.startswith("sglang:load_back_tokens_total{") and 'pool="kv"' in line
        )

    def test_host_restore(self):
        _flush_cache(self.base_url, timeout_s=60)
        before = self.load_count()
        ids = self.prompt(900000)
        cold = self.generate(ids)
        self.assertEqual(cold["meta_info"]["cached_tokens"], 0)
        warm = self.generate(ids)
        self.assertGreater(warm["meta_info"]["cached_tokens_details"]["device"], 0)
        for number in range(math.ceil(self.capacity / len(ids)) + 1):
            self.generate(self.prompt(number + 1000), output=1)
        restored = self.generate(ids)
        details = restored["meta_info"]["cached_tokens_details"]
        self.assertEqual(details["device"], 0)
        self.assertGreaterEqual(
            details["host"], (len(ids) - 1) // self.page_size * self.page_size
        )
        self.assertGreater(self.load_count(), before)
        self.assertEqual(warm["output_ids"], restored["output_ids"])
        warm_lps = _extract_output_logprobs(warm)
        restored_lps = _extract_output_logprobs(restored)
        compare_kl_divergence(
            [warm_lps],
            [restored_lps],
            {self.model: {"kl_div": 0.003}},
            self.model,
            "Pure GQA DCP L1 versus L2 restore",
        )
        print(
            "GQA_DCP_RESULT "
            + json.dumps(
                {
                    "model": self.model,
                    "dcp_size": self.dcp_size,
                    "host_tokens": details["host"],
                    "device_tokens": details["device"],
                    "output_ids_equal": True,
                    "max_logprob_difference": max(
                        abs(a - b) for a, b in zip(warm_lps, restored_lps)
                    ),
                }
            )
        )

    def test_multiturn_branching_and_release(self):
        initial = [self.prompt(200 + i // 2, 513) + [100 + i] for i in range(4)]
        suffixes = [
            [
                self.tokenizer.encode(
                    f"\nQuestion {turn}/{i}: explain another detail.\n"
                )
                for i in range(4)
            ]
            for turn in range(2)
        ]
        check_multiturn_logprobs(
            self.base_url,
            self.model,
            0.003,
            initial,
            turn_suffixes=suffixes,
            max_new_tokens=64,
            assert_decode_cached_tokens=functools.partial(
                default_decode_cache_assert, page_size=self.page_size
            ),
            label="Pure GQA DCP branching",
            sampling_temperature=0,
        )
        _flush_cache(self.base_url, timeout_s=60)
        with ThreadPoolExecutor(max_workers=4) as executor:
            results = list(executor.map(self.generate, initial))
        self.assertTrue(all(r["meta_info"]["completion_tokens"] == 64 for r in results))
        _flush_cache(self.base_url, timeout_s=60)
        self.assertEqual(
            self.generate(initial[0], output=1)["meta_info"]["cached_tokens"], 0
        )


if __name__ == "__main__":
    unittest.main()
