"""GLM-5.3-Flash logprob checks across branching and L2 HiCache loadback."""

import math
import os
import random
import statistics
import time
import unittest
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor

import requests

from sglang.srt.utils.hf_transformers_utils import get_tokenizer
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.kl_test_utils import _flush_cache
from sglang.test.test_utils import (
    DEFAULT_URL_FOR_TEST,
    _wait_for_gpu_idle_in_ci,
    popen_launch_server,
    terminate_and_kill_process_tree,
    try_cached_model,
)

register_cuda_ci(est_time=1200, stage="extra-b", runner_config="4-gpu-b200")

MODEL = "zai-org/GLM-5.3-Flash"
GROUP_SIZE = 256
OUTPUT_TOKENS = 512
REPEATS = 3
# DSA logprob differences produce heavy-tailed k3 estimates. This provisional
# median gate tolerates outliers without clipping them or retrying failed cases.
# TODO(alphabetc1): Tighten after #38212 and deterministic GLM DSA support.
MEDIAN_KL_THRESHOLD = 0.5


# CustomTestCase retries in CI; fixed statistical repetitions must not retry.
class TestGLM53FlashHiCacheKL(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        _wait_for_gpu_idle_in_ci()
        cls.model = os.environ.get("SGLANG_TEST_GLM53_MODEL") or try_cached_model(MODEL)
        cls.base_url = DEFAULT_URL_FOR_TEST
        cls.process = popen_launch_server(
            cls.model,
            cls.base_url,
            timeout=3600,
            other_args=[
                "--tp-size",
                "4",
                "--ep-size",
                "4",
                "--attention-backend",
                "dsa",
                "--dsa-prefill-backend",
                "tilelang",
                "--dsa-decode-backend",
                "tilelang",
                "--kv-cache-dtype",
                "bfloat16",
                "--moe-runner-backend",
                "deep_gemm",
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
                "--random-seed",
                "38212",
            ],
        )
        cls.addClassCleanup(
            terminate_and_kill_process_tree, cls.process, wait_timeout=60
        )
        cls.tokenizer = get_tokenizer(cls.model, trust_remote_code=True)

    def _generate(self, ids, *, count=1, score_start=None, min_cached=0):
        payload = {
            "input_ids": ids,
            "sampling_params": {
                "temperature": 0,
                "max_new_tokens": count,
                "ignore_eos": True,
            },
        }
        if score_start is not None:
            payload.update(
                return_logprob=True,
                return_text_in_logprobs=False,
                logprob_start_len=score_start,
            )
        response = requests.post(self.base_url + "/generate", json=payload, timeout=900)
        response.raise_for_status()
        result = response.json()
        self.assertNotIn("error", result)
        self.assertEqual(result["meta_info"]["finish_reason"]["type"], "length")
        self.assertGreaterEqual(result["meta_info"]["cached_tokens"], min_cached)
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
            "\nCRITICAL LEDGER ENTRY. Record Q7-DELTA. The exact access phrase is: "
            f"quartz velvet {offset} juniper cobalt {offset + 17}. Preserve every word.\n",
            "\nWhat is the exact access phrase for Q7-DELTA? Return both numbers.\n",
        )
        branch = common + suffix(
            "\nASTRONOMICAL OBSERVATION. The northern nebula contains ionized gas. "
            "Its apparent motion follows a curved trajectory across the sky.\n",
            "\nDescribe the nebula's composition and apparent motion.\n",
        )
        bridge = encode("\n")
        self.assertTrue(bridge)
        _flush_cache(self.base_url)
        # Remove stochastic token-selection variance from this fixed-token
        # precision probe; it is not full-vocabulary KL.
        continuation = self._generate(prompt + bridge, count=OUTPUT_TOKENS)[
            "output_ids"
        ]
        self.assertEqual(len(continuation), OUTPUT_TOKENS)
        pressure = [[rng.randint(1000, 25000) for _ in range(8192)] for _ in range(6)]
        self.assertEqual(len({ids[0] for ids in pressure}), len(pressure))
        self.assertNotIn(common[0], [ids[0] for ids in pressure])
        return common, prompt, branch, bridge, continuation, pressure

    def _score(self, prompt, bridge, continuation, *, cached=False, host=False):
        # The logprob start token is None. Put it after A so scoring can reuse A's
        # Mamba checkpoint instead of recomputing its suffix and healing the index.
        prefix = prompt + bridge
        result = self._generate(
            prefix + continuation, count=0, score_start=len(prefix) - 1
        )
        meta = result["meta_info"]
        if cached:
            self.assertGreaterEqual(meta["cached_tokens"], len(prompt) - GROUP_SIZE + 1)
        else:
            self.assertEqual(meta["cached_tokens"], 0)
        if host:
            self.assertGreater(
                (meta.get("cached_tokens_details") or {}).get("host", 0), 0
            )
        values = meta["input_token_logprobs"][-len(continuation) :]
        self.assertEqual([item[1] for item in values], continuation)
        self.assertTrue(
            all(item[0] is not None and math.isfinite(item[0]) for item in values)
        )
        print(
            f"cached_tokens={meta['cached_tokens']} details={meta.get('cached_tokens_details')}"
        )
        return [item[0] for item in values]

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
                        delta = [q - p for p, q in zip(reference, candidate)]
                        self.assertTrue(
                            all(math.isfinite(d) and abs(d) < 700 for d in delta)
                        )
                        kl = sum(math.expm1(d) - d for d in delta) / len(delta)
                        kl_values[name].append(kl)
                        print(
                            f"offset={offset} repeat={repeat} scenario={name} kl={kl} "
                            f"max_abs_logprob_delta={max(map(abs, delta))} "
                            f"mean_abs_logprob_delta={statistics.mean(map(abs, delta))}"
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
                median_kl = statistics.median(values)
                print(f"scenario={name} kl_values={values} median_kl={median_kl}")
                self.assertLess(median_kl, MEDIAN_KL_THRESHOLD)


if __name__ == "__main__":
    unittest.main()
