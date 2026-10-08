"""GLM-5.3-Flash full-vocabulary KL across branching and L2 host loadback.

Guards lost DSA indexes or recurrent checkpoints after prefix splits and GPU
slot reuse. Eight fixed continuation prefixes are scored in every arm; this
is a short state-restoration probe, not a long-decode consistency test.
CI stays disabled pending calibration on the exact NVFP4/FP8-KV/TRT-LLM/TP4 recipe.
"""

import json
import os
import random
import time
import unittest
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor

import requests
import torch

from sglang.srt.utils.hf_transformers_utils import get_config, get_tokenizer
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.kl_test_utils import _flush_cache
from sglang.test.test_utils import (
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
    terminate_and_kill_process_tree,
    try_cached_model,
)

register_cuda_ci(
    est_time=1200,
    stage="extra-b",
    runner_config="4-gpu-b200",
    disabled="#38474: full-vocabulary KL needs B200 calibration; until 2026-11-08",
)

MODEL = "RadixArk/GLM-5.3-Flash-NVFP4"
GROUP_SIZE = 256
OUTPUT_TOKENS = 8
REPEATS = 3
# Provisional investigation threshold, NOT calibrated for this NVFP4/TP4 recipe.
# #39156 measured 0.1 on NVFP4/B300; that evidence does not transfer here.
# Keep CI disabled until cold controls and a negative control validate this gate.
KL_THRESHOLD = 0.1


class TestGLM53FlashHiCacheKL(CustomTestCase):
    # Preserve CustomTestCase's setup-failure cleanup, but never retry numerics.
    def _callTestMethod(self, method):
        return unittest.TestCase._callTestMethod(self, method)

    @classmethod
    def setUpClass(cls):
        cls.model = os.environ.get("SGLANG_TEST_GLM53_MODEL") or try_cached_model(MODEL)
        cls.base_url = DEFAULT_URL_FOR_TEST
        cls.process = popen_launch_server(
            cls.model,
            cls.base_url,
            timeout=3600,
            other_args=[
                "--quantization",
                "modelopt_fp4",
                "--tp-size",
                "4",
                "--ep-size",
                "4",
                "--attention-backend",
                "dsa",
                "--dsa-prefill-backend",
                "trtllm",
                "--dsa-decode-backend",
                "trtllm",
                "--kv-cache-dtype",
                "fp8_e4m3",
                "--moe-runner-backend",
                "flashinfer_trtllm",
                "--max-total-tokens",
                "32768",
                "--max-running-requests",
                "8",
                "--max-mamba-cache-size",
                "128",
                "--chunked-prefill-size",
                "8192",
                "--context-length",
                "16384",
                "--mem-fraction-static",
                "0.75",
                "--enable-hierarchical-cache",
                "--hicache-ratio",
                "4",
                "--hicache-write-policy",
                "write_through",
                "--hicache-mem-layout",
                "page_first_direct",
                "--hicache-io-backend",
                "direct",
                "--mamba-radix-cache-strategy",
                "extra_buffer",
                "--enable-cache-report",
                "--random-seed",
                "38212",
            ],
        )
        cls.tokenizer = get_tokenizer(cls.model, trust_remote_code=True)
        config = get_config(cls.model, trust_remote_code=True).get_text_config()
        cls.vocab_ids = list(range(config.vocab_size))

    @classmethod
    def tearDownClass(cls):
        if getattr(cls, "process", None) is not None:
            terminate_and_kill_process_tree(cls.process, wait_timeout=60)
            cls.process = None

    def _generate(self, ids, *, count=1, capture=False, min_cached=0):
        payload = {
            "input_ids": ids,
            "sampling_params": {
                "temperature": 0,
                "max_new_tokens": count,
                "ignore_eos": True,
            },
        }
        if capture:
            payload.update(
                return_logprob=True,
                return_text_in_logprobs=False,
                logprob_start_len=-1,
                token_ids_logprob=self.vocab_ids,
            )
        response = requests.post(self.base_url + "/generate", json=payload, timeout=900)
        response.raise_for_status()
        result = response.json()
        self.assertNotIn("error", result)
        meta = result["meta_info"]
        self.assertEqual(meta["finish_reason"]["type"], "length")
        self.assertEqual(len(result["output_ids"]), count)
        self.assertGreaterEqual(meta["cached_tokens"], min_cached)
        if capture:
            rows = meta.pop("output_token_ids_logprobs")
            self.assertEqual(len(rows), count)
            for row in rows:
                self.assertEqual([entry[1] for entry in row], self.vocab_ids)
            logprobs = torch.tensor(
                [[entry[0] for entry in row] for row in rows], dtype=torch.float64
            )
            self.assertTrue(torch.isfinite(logprobs).all().item())
            torch.testing.assert_close(
                logprobs.exp().sum(dim=-1),
                torch.ones(count, dtype=torch.float64),
                atol=1e-5,
                rtol=0,
            )
            result["logprobs"] = logprobs
        return result

    def _make_case(self, offset):
        rng = random.Random(382120 + offset)
        marker = "__GLM53_HICACHE_CONTENT__"
        rendered = self.tokenizer.apply_chat_template(
            [{"role": "user", "content": marker}],
            tokenize=False,
            add_generation_prompt=True,
        )
        self.assertEqual(rendered.count(marker), 1)
        chat_prefix, chat_tail = rendered.split(marker)
        text = "Read these records and answer the final question.\n" + "".join(
            f"Record {i}: value {rng.randrange(1000000)}; group {rng.randrange(100)}.\n"
            for i in range(8192)
        )

        def encode(text):
            return self.tokenizer.encode(text, add_special_tokens=False)

        common = encode(chat_prefix + text)[: 8192 + offset]
        self.assertEqual(len(common), 8192 + offset)
        filler = encode(
            "".join(
                f"Unrelated inventory {i}: quantity {rng.randrange(1000000)}.\n"
                for i in range(1024)
            )
        )

        def suffix(lead, question):
            start, end = encode(lead), encode(question + chat_tail)
            remaining = 1024 - len(start) - len(end)
            self.assertGreater(remaining, 0)
            result = start + filler[:remaining] + end
            self.assertEqual(len(result), 1024)
            return result

        prompt = common + suffix(
            "CRITICAL LEDGER ENTRY. Record Q7-DELTA. The exact access phrase is: "
            f"quartz velvet {offset} juniper cobalt {offset + 17}. Preserve every word.\n",
            "\nWhat is the exact access phrase for Q7-DELTA? Return both numbers.\n",
        )
        branch = common + suffix(
            "ASTRONOMICAL OBSERVATION. The northern nebula contains ionized gas. "
            "Its apparent motion follows a curved trajectory across the sky.\n",
            "\nDescribe the nebula's composition and apparent motion.\n",
        )
        # The first suffix token must differ: otherwise the aligned control
        # would actually split after the intended boundary.
        self.assertNotEqual(prompt[len(common)], branch[len(common)])
        bridge = encode("\n")
        self.assertTrue(bridge)
        _flush_cache(self.base_url)
        # All arms score these same conditioning prefixes, even if argmax differs.
        continuation = self._generate(prompt + bridge, count=OUTPUT_TOKENS)[
            "output_ids"
        ]
        self.assertEqual(len(continuation), OUTPUT_TOKENS)
        pressure = [[rng.randint(1000, 25000) for _ in range(8192)] for _ in range(6)]
        self.assertEqual(len({ids[0] for ids in pressure}), len(pressure))
        self.assertNotIn(common[0], [ids[0] for ids in pressure])
        return common, prompt, branch, bridge, continuation, pressure

    def _score(self, prompt, bridge, continuation, *, cached=False, host=False):
        steps = []
        cache = []
        for i in range(OUTPUT_TOKENS):
            prefix = prompt + bridge + continuation[:i]
            if not cached:
                # Each reference position must be independent of cache reuse.
                _flush_cache(self.base_url)
            result = self._generate(prefix, capture=True)
            meta = result["meta_info"]
            details = meta.get("cached_tokens_details") or {}
            if cached:
                # Do not heal corrupted prefix state by recomputing all of A.
                self.assertGreaterEqual(
                    meta["cached_tokens"], len(prefix) - GROUP_SIZE + 1
                )
            else:
                self.assertEqual(meta["cached_tokens"], 0)
            if host and i == 0:
                # Pressure must evict substantially all of A, not just one page.
                self.assertGreaterEqual(
                    details.get("host", 0), len(prompt) - GROUP_SIZE + 1
                )
            else:
                self.assertEqual(details.get("host", 0), 0)
            cache.append({"cached_tokens": meta["cached_tokens"], "details": details})
            steps.append(result["logprobs"])
        return torch.cat(steps), cache

    def test_branching_and_host_loadback(self):
        """Exercise index-group boundaries without allowing silent cache misses."""
        kl_values = defaultdict(list)
        for offset in (64, 128, 192, 256):
            common, prompt, branch, bridge, continuation, pressure = self._make_case(
                offset
            )
            shared_length = len(common) // GROUP_SIZE * GROUP_SIZE
            for repeat in range(REPEATS):
                with self.subTest(offset=offset, repeat=repeat):

                    def score(**kwargs):
                        return self._score(prompt, bridge, continuation, **kwargs)

                    def record(name, reference, candidate):
                        p, q = reference[0], candidate[0]
                        # Exact KL(P || Q) over the complete vocabulary at each
                        # identical conditioning prefix, not selected-token k3.
                        values = (p.exp() * (p - q)).sum(dim=-1)
                        self.assertTrue(torch.isfinite(values).all().item())
                        self.assertGreaterEqual(values.min().item(), -1e-6)
                        mean_kl = values.mean().item()
                        kl_values[name].append(mean_kl)
                        print(
                            json.dumps(
                                {
                                    "offset": offset,
                                    "repeat": repeat,
                                    "scenario": name,
                                    "kl_per_position": values.tolist(),
                                    "mean_kl": mean_kl,
                                    "conditioning_ids": continuation[:-1],
                                    "reference_cache": reference[1],
                                    "candidate_cache": candidate[1],
                                }
                            ),
                            flush=True,
                        )

                    _flush_cache(self.base_url)
                    cold = score()
                    _flush_cache(self.base_url)
                    record("cold_repeat", cold, score())

                    _flush_cache(self.base_url)
                    self._generate(prompt)
                    record("device_hit", cold, score(cached=True))

                    _flush_cache(self.base_url)
                    self._generate(common)
                    self._generate(prompt, min_cached=shared_length)
                    self._generate(branch, min_cached=shared_length)
                    record("device_fork", cold, score(cached=True))

                    for concurrent in (False, True):
                        _flush_cache(self.base_url)
                        self._generate(common)
                        self._generate(prompt, min_cached=shared_length)
                        # Give write-through copies time to complete before eviction.
                        time.sleep(2)
                        if concurrent:
                            with ThreadPoolExecutor(max_workers=7) as executor:
                                jobs = [
                                    executor.submit(
                                        self._generate, branch, min_cached=shared_length
                                    )
                                ]
                                jobs.extend(
                                    executor.submit(self._generate, ids)
                                    for ids in pressure
                                )
                                for job in jobs:
                                    job.result()
                        else:
                            self._generate(branch, min_cached=shared_length)
                            time.sleep(2)
                            for ids in pressure:
                                self._generate(ids)
                        time.sleep(2)
                        name = "concurrent_host" if concurrent else "fork_host"
                        record(name, cold, score(cached=True, host=True))

        self.assertEqual(
            set(kl_values),
            {
                "cold_repeat",
                "device_hit",
                "device_fork",
                "fork_host",
                "concurrent_host",
            },
        )
        for name, values in kl_values.items():
            with self.subTest(scenario=name):
                self.assertEqual(len(values), 4 * REPEATS)
                # Every sequence must pass; a median would hide boundary failures.
                for case_id, value in enumerate(values):
                    with self.subTest(case_id=case_id):
                        self.assertLess(value, KL_THRESHOLD)


if __name__ == "__main__":
    unittest.main()
