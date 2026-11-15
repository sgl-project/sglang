"""Blackwell GDN prefix-cache replay must match the same eager workload.

SM100 covers BF16 fallback and qualified ModelOpt padded replay. SM120 covers
bounded small-prefill replay, including a cached 7 -> 8 token normalization
boundary. Compare eager execution with mixed chunk enabled, reusing recurrent
checkpoints alongside uncached extends:

    python3 test/registered/cuda_graph/breakable/test_qwen35_prefix_cache_blackwell.py
"""

import json
import math
import random
import re
import unittest

import numpy as np
import requests
import torch

from sglang.srt.utils.device import is_sm100_supported, is_sm120_supported
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import (
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
    terminate_and_kill_process_tree,
)

# Match the existing 9B NVFP4-KV test: there is no single-GPU SM100 CI pool.
register_cuda_ci(est_time=1200, stage="nightly", runner_config="4-gpu-b200")
register_cuda_ci(est_time=1200, stage="nightly", runner_config="1-gpu-small")

TRACK_INTERVAL = 128
BUCKETS = [16, 20, TRACK_INTERVAL]


class TestQwen35PrefixCacheSM100(CustomTestCase):
    model = "Qwen/Qwen3.5-9B"
    revision = None
    tp_size = 4
    allow_padded = False
    sm120 = False
    buckets = BUCKETS
    extend_lengths = (16, 17, 20)
    padded_limit = None
    output_tokens = 1
    logprob_atol = 1e-5
    seed_extra_token = True
    mem_fraction_static = 0.6
    attention_args = [
        "--kv-cache-dtype",
        "nvfp4",
        "--prefill-kv-cache-dequant-dtype",
        "nvfp4",
        "--attention-backend",
        "trtllm_mha",
        "--page-size",
        "16",
    ]

    @classmethod
    def setUpClass(cls):
        supported = is_sm120_supported() if cls.sm120 else is_sm100_supported()
        if not supported or torch.cuda.device_count() < cls.tp_size:
            raise unittest.SkipTest(
                f"requires {cls.tp_size} SM{120 if cls.sm120 else 100} GPUs"
            )
        cls.base_url = DEFAULT_URL_FOR_TEST

    def _graph_counts(self):
        # A generate response is streamed before report_prefill_stats updates the
        # scheduler counters. Round-trip through the scheduler before scraping,
        # including before a baseline so seed requests cannot leak into its delta.
        response = requests.get(self.base_url + "/server_info", timeout=30)
        response.raise_for_status()
        response = requests.get(self.base_url + "/metrics", timeout=30)
        response.raise_for_status()
        counts = {}
        for mode in ("prefill_cuda_graph", "prefill_none"):
            matches = re.findall(
                rf'^sglang:cuda_graph_passes_total\{{[^}}]*mode="{mode}"[^}}]*\}}'
                r"\s+([0-9.eE+-]+)$",
                response.text,
                re.MULTILINE,
            )
            # Prometheus does not expose a labeled counter before its first use.
            counts[mode] = sum(map(float, matches), 0.0)
        return counts

    def _assert_forward_counts(self, before, backend, extend_len):
        after = self._graph_counts()
        bucket = next((n for n in self.buckets if n >= extend_len), None)
        replay = (
            backend == "breakable"
            and bucket is not None
            and (
                bucket == extend_len
                or (
                    self.allow_padded
                    and (self.padded_limit is None or bucket <= self.padded_limit)
                )
            )
        )
        active_mode = "prefill_cuda_graph" if replay else "prefill_none"
        inactive_mode = "prefill_none" if replay else "prefill_cuda_graph"
        self.assertGreater(after[active_mode], before[active_mode])
        self.assertEqual(after[inactive_mode], before[inactive_mode])

    def _generate(self, input_ids, *, logprob_start_len=None, max_new_tokens=1):
        payload = {
            "input_ids": input_ids,
            "sampling_params": {
                "temperature": 0,
                "max_new_tokens": max_new_tokens,
                "ignore_eos": True,
            },
        }
        if logprob_start_len is not None:
            payload.update(
                return_logprob=True,
                logprob_start_len=logprob_start_len,
            )
        response = requests.post(self.base_url + "/generate", json=payload, timeout=180)
        response.raise_for_status()
        return response.json()

    def _run_server(self, backend, cases):
        graph_config = {
            "decode": {"backend": "disabled"},
            "prefill": {"backend": backend, "bs": self.buckets},
        }
        process = popen_launch_server(
            self.model,
            self.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=[
                "--tp-size",
                str(self.tp_size),
                "--dtype",
                "bfloat16",
                "--mamba-radix-cache-strategy",
                "extra_buffer",
                "--mamba-track-interval",
                str(TRACK_INTERVAL),
                "--mamba-ssm-dtype",
                "bfloat16",
                "--max-running-requests",
                "8",
                "--max-total-tokens",
                "8192",
                "--mem-fraction-static",
                str(self.mem_fraction_static),
                "--chunked-prefill-size",
                "512",
                "--enable-mixed-chunk",
                "--enable-metrics",
                "--random-seed",
                "42",
                "--cuda-graph-config",
                json.dumps(graph_config),
            ]
            + self.attention_args
            + (["--revision", self.revision] if self.revision is not None else []),
        )
        try:
            response = requests.get(self.base_url + "/server_info", timeout=30)
            response.raise_for_status()
            info = response.json()
            self.assertEqual(info["cuda_graph_config"]["prefill"]["backend"], backend)
            self.assertEqual(info["cuda_graph_config"]["decode"]["backend"], "disabled")
            self.assertTrue(info["enable_mixed_chunk"])
            self.assertFalse(info["disable_radix_cache"])
            if backend == "breakable":
                self.assertEqual(
                    info["cuda_graph_config"]["prefill"]["bs"], self.buckets
                )
            results = {}
            for prefix_len, extend_len, prefix, suffix in cases:
                with self.subTest(
                    backend=backend, prefix=prefix_len, extend=extend_len
                ):
                    response = requests.post(
                        self.base_url + "/flush_cache",
                        params={"timeout": 30},
                        timeout=40,
                    )
                    response.raise_for_status()
                    if prefix_len:
                        # SM100 uses a 129-token eager seed, with the tracked
                        # checkpoint strictly inside it. SM120 also exercises
                        # checkpoints written at the exact 128-token boundary.
                        seed_ids = prefix[:TRACK_INTERVAL]
                        if self.seed_extra_token:
                            seed_ids.append(100)
                        seed = self._generate(seed_ids)
                        self.assertEqual(seed["meta_info"]["cached_tokens"], 0)
                    if prefix_len == 2 * TRACK_INTERVAL:
                        # Advance the cached 128-token prefix through another
                        # exact 128-token bucket. The graph must write the new
                        # checkpoint at 256; the next request must actually
                        # restore it, not silently reuse the eager checkpoint.
                        before = self._graph_counts()
                        checkpoint = self._generate(prefix)
                        checkpoint_meta = checkpoint["meta_info"]
                        self.assertEqual(
                            checkpoint_meta["cached_tokens"], TRACK_INTERVAL
                        )
                        self.assertEqual(
                            checkpoint_meta["prompt_tokens"]
                            - checkpoint_meta["cached_tokens"],
                            TRACK_INTERVAL,
                        )
                        self._assert_forward_counts(before, backend, TRACK_INTERVAL)
                    before = self._graph_counts()
                    result = self._generate(
                        prefix + suffix,
                        logprob_start_len=prefix_len,
                        max_new_tokens=self.output_tokens,
                    )
                    meta = result["meta_info"]
                    self.assertEqual(meta["cached_tokens"], prefix_len)
                    self.assertEqual(
                        meta["prompt_tokens"] - meta["cached_tokens"], extend_len
                    )
                    self._assert_forward_counts(before, backend, extend_len)

                    # Input IDs are fixed, not the output of a preceding decode.
                    # The first input logprob is unavailable after a cache hit;
                    # check the remaining suffix and the sampled continuation.
                    input_logprobs = meta["input_token_logprobs"]
                    self.assertEqual([entry[1] for entry in input_logprobs], suffix)
                    self.assertIsNone(input_logprobs[0][0])
                    output_logprobs = meta["output_token_logprobs"]
                    self.assertEqual(len(output_logprobs), self.output_tokens)
                    scored = input_logprobs[1:] + output_logprobs
                    self.assertTrue(all(math.isfinite(entry[0]) for entry in scored))
                    results[prefix_len, extend_len] = scored
            return results
        finally:
            # Release the first server's GPU allocations before loading the next.
            terminate_and_kill_process_tree(process)

    def _make_cases(self):
        # Fixed raw token IDs avoid tokenizer revisions, datasets, generated-prefix
        # drift, and concurrent scheduling changing the intended bucket sizes.
        rng = random.Random(2026)
        cases = []
        for prefix_len in (0, TRACK_INTERVAL, 2 * TRACK_INTERVAL):
            prefix = [rng.randint(1000, 30000) for _ in range(prefix_len)]
            for extend_len in self.extend_lengths:
                suffix = [rng.randint(30001, 60000) for _ in range(extend_len)]
                cases.append((prefix_len, extend_len, prefix, suffix))
        return cases

    def test_mixed_chunk_plain_extends_match_eager(self):
        cases = self._make_cases()
        # TRT-LLM attention does not support batch-invariant mode with radix
        # caching. Compare the same requests with and without graph replay.
        eager = self._run_server("disabled", cases)
        graph = self._run_server("breakable", cases)
        for key, expected in eager.items():
            with self.subTest(prefix=key[0], extend=key[1]):
                actual = graph[key]
                self.assertEqual([x[1] for x in actual], [x[1] for x in expected])
                # Padded replay and exact-bucket fallback must both preserve
                # real-token logprobs and subsequently restored recurrent state.
                np.testing.assert_allclose(
                    [x[0] for x in actual],
                    [x[0] for x in expected],
                    rtol=0,
                    atol=self.logprob_atol,
                    err_msg=f"prefix={key[0]}, extend={key[1]}",
                )


class TestQwen38NVFP4PrefixCacheSM100(TestQwen35PrefixCacheSM100):
    """Keep padded replay for the effective FP8/FP4 paths of the reproducer."""

    model = "nvidia/Qwen3.8-27B-NVFP4"
    revision = "482ca0f3832238542f8f5295dde86b5f22711d80"
    # Preserve the TP1 projection dimensions reported in #49133.
    tp_size = 1
    allow_padded = True
    buckets = [8, 16, 20, TRACK_INTERVAL]
    extend_lengths = (5, 7, 13, 16, 17, 20)


class TestQwen38NVFP4PrefixCacheSM120(TestQwen38NVFP4PrefixCacheSM100):
    sm120 = True
    buckets = [8, 16, 32, 64, 128, 256, 512]
    padded_limit = 256
    output_tokens = 16
    logprob_atol = 0
    seed_extra_token = False
    mem_fraction_static = 0.86
    attention_args = [
        "--kv-cache-dtype",
        "fp8_e4m3",
        "--attention-backend",
        "flashinfer",
        "--page-size",
        "64",
    ]

    def _make_cases(self):
        # This repeated prompt reproduced divergent cached128+7 logprobs when
        # padding changed the gated norm's reduction layout from 1 to 2 rows.
        tokens = [
            760,
            2614,
            1830,
            90273,
            264,
            3177,
            6522,
            13,
            1049,
            682,
            3712,
            11,
            7785,
            11,
            36352,
            11,
            321,
            30223,
            6265,
            13,
            62239,
            628,
            16686,
            6265,
            364,
            2250,
            5381,
            13,
            79091,
            1204,
            19573,
            1438,
            10428,
            13,
            220,
        ]
        cases = []
        for prefix_len in (0, TRACK_INTERVAL, 2 * TRACK_INTERVAL):
            for extend_len in (5, 7, 13, 17, 31, 193, 257, 512):
                length = prefix_len + extend_len
                prompt = (tokens * ((length + len(tokens) - 1) // len(tokens)))[:length]
                cases.append(
                    (prefix_len, extend_len, prompt[:prefix_len], prompt[prefix_len:])
                )
        return cases


if __name__ == "__main__":
    unittest.main()
