"""Blackwell GDN prefix-cache replay must match the same eager workload.

Uses Qwen3.5-9B BF16 weights with NVFP4 KV (not modelopt-quantized weights).
With mixed chunk enabled, isolated requests are still ordinary EXTEND batches:
16/20-token extends must replay, while 17 tokens must not pad to the 20 bucket.
Both eager-written and graph-written recurrent checkpoints are reused, alongside
uncached extends. Run on 4 B200s:

    python3 test/registered/cuda_graph/breakable/test_qwen35_prefix_cache_sm100.py
"""

import json
import math
import random
import re
import unittest

import numpy as np
import requests
import torch

from sglang.srt.utils.device import is_sm100_supported
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import (
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
    terminate_and_kill_process_tree,
)

# Match the existing 9B NVFP4-KV test: there is no single-GPU SM100 CI pool.
register_cuda_ci(est_time=600, stage="nightly", runner_config="4-gpu-b200")

MODEL = "Qwen/Qwen3.5-9B"
TRACK_INTERVAL = 128
BUCKETS = [16, 20, TRACK_INTERVAL]


class TestQwen35PrefixCacheSM100(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        if not is_sm100_supported() or torch.cuda.device_count() < 4:
            raise unittest.SkipTest("requires 4 SM100 GPUs with CUDA 12.8+")
        cls.base_url = DEFAULT_URL_FOR_TEST

    def _graph_counts(self):
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
        replay = backend == "breakable" and extend_len in BUCKETS
        active_mode = "prefill_cuda_graph" if replay else "prefill_none"
        inactive_mode = "prefill_none" if replay else "prefill_cuda_graph"
        self.assertGreater(after[active_mode], before[active_mode])
        self.assertEqual(after[inactive_mode], before[inactive_mode])

    def _generate(self, input_ids, *, logprob_start_len=None):
        payload = {
            "input_ids": input_ids,
            "sampling_params": {
                "temperature": 0,
                "max_new_tokens": 1,
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
            "prefill": {"backend": backend, "bs": BUCKETS},
        }
        process = popen_launch_server(
            MODEL,
            self.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=[
                "--tp-size",
                "4",
                "--dtype",
                "bfloat16",
                "--kv-cache-dtype",
                "nvfp4",
                "--prefill-kv-cache-dequant-dtype",
                "nvfp4",
                "--attention-backend",
                "trtllm_mha",
                "--page-size",
                "16",
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
                "0.6",
                "--chunked-prefill-size",
                "512",
                "--enable-mixed-chunk",
                "--enable-metrics",
                "--random-seed",
                "42",
                "--cuda-graph-config",
                json.dumps(graph_config),
            ],
            # Never inherit the explicit unsafe padded-GDN opt-in from CI.
            env={"SGLANG_ENABLE_PADDED_GDN_PREFILL_GRAPH": "0"},
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
                self.assertEqual(info["cuda_graph_config"]["prefill"]["bs"], BUCKETS)
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
                        # Leave a full tracked checkpoint strictly inside the seed
                        # prompt: prefix reuse excludes the prompt's final token.
                        # The seed is larger than every bucket and runs eagerly in
                        # both servers. Its last token differs from every suffix.
                        seed = self._generate(prefix[:TRACK_INTERVAL] + [100])
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
                        prefix + suffix, logprob_start_len=prefix_len
                    )
                    meta = result["meta_info"]
                    self.assertEqual(meta["cached_tokens"], prefix_len)
                    self.assertEqual(
                        meta["prompt_tokens"] - meta["cached_tokens"], extend_len
                    )
                    self._assert_forward_counts(before, backend, extend_len)

                    # Input IDs are fixed, not the output of a preceding decode.
                    # The first input logprob is unavailable after a cache hit;
                    # check the remaining suffix and the final sampled token.
                    input_logprobs = meta["input_token_logprobs"]
                    self.assertEqual([entry[1] for entry in input_logprobs], suffix)
                    self.assertIsNone(input_logprobs[0][0])
                    output_logprobs = meta["output_token_logprobs"]
                    self.assertEqual(len(output_logprobs), 1)
                    scored = input_logprobs[1:] + output_logprobs
                    self.assertTrue(all(math.isfinite(entry[0]) for entry in scored))
                    results[prefix_len, extend_len] = scored
            return results
        finally:
            # Release the first server's GPU allocations before loading the next.
            terminate_and_kill_process_tree(process)

    def test_mixed_chunk_plain_extends_match_eager(self):
        # Fixed raw token IDs avoid tokenizer revisions, datasets, generated-prefix
        # drift, and concurrent scheduling changing the intended bucket sizes.
        rng = random.Random(2026)
        cases = []
        for prefix_len in (0, TRACK_INTERVAL, 2 * TRACK_INTERVAL):
            prefix = [rng.randint(1000, 30000) for _ in range(prefix_len)]
            for extend_len in (16, 17, 20):
                suffix = [rng.randint(30001, 60000) for _ in range(extend_len)]
                cases.append((prefix_len, extend_len, prefix, suffix))

        # Do not enable batch-invariant inference: TRT-LLM attention does not
        # support that mode with radix caching. Compare identical shapes instead.
        eager = self._run_server("disabled", cases)
        graph = self._run_server("breakable", cases)
        for key, expected in eager.items():
            with self.subTest(prefix=key[0], extend=key[1]):
                actual = graph[key]
                self.assertEqual([x[1] for x in actual], [x[1] for x in expected])
                # Same tokens, cached state, and GEMM M must not acquire the
                # numerical drift of padding 17 live rows to a captured 20.
                np.testing.assert_allclose(
                    [x[0] for x in actual],
                    [x[0] for x in expected],
                    rtol=0,
                    atol=1e-5,
                    err_msg=f"prefix={key[0]}, extend={key[1]}",
                )


if __name__ == "__main__":
    unittest.main()
