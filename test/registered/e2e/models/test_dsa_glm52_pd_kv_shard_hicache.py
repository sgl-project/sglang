"""GLM DSA/MTP P/D transfer, CPU HiCache reload and GSM8K accuracy.

On four GB300s, P uses CP2 page sharding and D receives complete caches.
Independent prompts exceed P's logical GPU capacity while fitting in L2.
The final repeat must report a host hit, advance the H2D counter and match
the device-hit answer.

The accuracy test also uses P TP2/CP2 + D TP2 on four GB300s. It retains
layer-split's GSM8K scoring settings and requires H2D reload during evaluation.
"""

import re
import time
import unittest

import requests
import torch
from prometheus_client.parser import text_string_to_metric_families

from sglang.srt.utils.hf_transformers_utils import get_tokenizer
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.kits.eval_accuracy_kit import GSM8KMixin, _run_accuracy_eval
from sglang.test.server_fixtures.disaggregation_fixture import (
    PDDisaggregationServerBase,
)
from sglang.test.simple_eval_mixed_prefix_gsm8k import (
    GSM8K_URL,
    get_few_shot_examples,
)
from sglang.utils import download_and_cache_file, read_jsonl

register_cuda_ci(est_time=600, stage="base-c", runner_config="4-gpu-gb300")

PAGE_SIZE = 64
PREFILL_POOL_TOKENS = 4096  # Per rank: CP2 has 8192 logical GPU slots.
PRESSURE_TOKENS = 6144
FILLER = "The grass is green. The sky is blue. Here we go, there and back again. "
# Match test_disaggregation_aarch64.py on this GB300 runner: fabric NVLink
# requires exportable KV buffers from Mooncake's NVLink allocator.
MOONCAKE_ENV = {
    "MC_FORCE_MNNVL": "true",
    "SGLANG_MOONCAKE_CUSTOM_MEM_POOL": "NVLINK",
    "SGLANG_DISAGG_STAGING_BUFFER": "0",
}
COMMON_ARGS = [
    "--page-size",
    "64",
    "--context-length",
    "16384",
    "--max-running-requests",
    "1",
    "--mem-fraction-static",
    "0.85",
    "--kv-cache-dtype",
    "fp8_e4m3",
    "--dsa-prefill-backend",
    "trtllm",
    "--dsa-decode-backend",
    "trtllm",
    "--speculative-algorithm",
    "EAGLE",
    "--speculative-num-steps",
    "3",
    "--speculative-eagle-topk",
    "1",
    "--speculative-num-draft-tokens",
    "4",
    "--random-seed",
    "42",
    "--enable-cache-report",
    "--model-loader-extra-config",
    '{"enable_multithread_load": true, "num_threads": 6}',
]


class ShardedHiCacheTestMixin:
    def _generate(self, input_ids=None, max_tokens=64, *, text=None):
        response = requests.post(
            self.lb_url + "/generate",
            json={
                **(
                    {"input_ids": input_ids}
                    if input_ids is not None
                    else {"text": text}
                ),
                "sampling_params": {"temperature": 0, "max_new_tokens": max_tokens},
            },
            timeout=600,
        )
        self.assertEqual(response.status_code, 200, response.text)
        result = response.json()
        if input_ids is not None:
            self.assertEqual(result["meta_info"]["prompt_tokens"], len(input_ids))
        return result

    def _counter(self, name):
        response = requests.get(self.prefill_url + "/metrics", timeout=30)
        response.raise_for_status()
        # Logical token counters repeat across ranks; read only TP rank 0.
        return sum(
            sample.value
            for family in text_string_to_metric_families(response.text)
            for sample in family.samples
            if sample.name == name
            and sample.labels.get("pool") == "kv"
            and sample.labels.get("tp_rank") == "0"
        )

    def _wait_for_counter(self, name, minimum):
        deadline = time.monotonic() + 30
        value = self._counter(name)
        while value < minimum and time.monotonic() < deadline:
            time.sleep(0.2)
            value = self._counter(name)
        self.assertGreaterEqual(value, minimum, f"{name} did not reach {minimum}")


@unittest.skipUnless(torch.cuda.device_count() >= 4, "requires 4 CUDA GPUs")
class TestGLM52PDShardedHiCache(PDDisaggregationServerBase, ShardedHiCacheTestMixin):
    model = "nvidia/GLM-5.2-NVFP4"
    prefill_tp_size = decode_tp_size = 2
    decode_base_gpu_id = 2
    extra_prefill_env = {
        **MOONCAKE_ENV,
        "SGLANG_UNIFIED_RADIX_TREE_CORE_BACKEND": "python",
    }
    extra_decode_env = dict(MOONCAKE_ENV)
    extra_prefill_args = COMMON_ARGS + [
        "--enable-kv-cache-sharding",
        "--enable-prefill-cp",
        "--attn-cp-size",
        "2",
        "--cp-strategy",
        "interleave",
        "--chunked-prefill-size",
        "2048",
        "--max-total-tokens",
        str(PREFILL_POOL_TOKENS),
        "--enable-hierarchical-cache",
        "--hicache-ratio",
        "4",
        "--hicache-write-policy",
        "write_through",
        "--hicache-io-backend",
        "kernel",
        "--hicache-mem-layout",
        "page_first",
        "--enable-metrics",
    ]
    extra_decode_args = COMMON_ARGS + [
        "--max-total-tokens",
        "16384",
        "--cuda-graph-max-bs-decode",
        "1",
        # Every repeat must exercise P/D transfer, not decode-side prefix reuse.
        "--disable-radix-cache",
    ]

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.transfer_backend = ["--disaggregation-transfer-backend", "mooncake"]
        cls.launch_all()
        cls.tokenizer = get_tokenizer(cls.model, trust_remote_code=True)

    def test_mtp_answer_survives_pd_and_host_reload(self):
        codes = ["739391", "482617", "956083", "164729"]
        filler = self.tokenizer.encode(FILLER, add_special_tokens=False)
        block = self.tokenizer.decode((filler * (1024 // len(filler) + 1))[:1024])
        document = "Read the following document and remember its four records.\n"
        for name, code in zip(("alpha", "beta", "gamma", "delta"), codes):
            document += block + f"\nThe verification code for {name} is {code}.\n"
        document += (
            block
            + "\nReturn the alpha, beta, gamma and delta codes in that order. Reply with digits and spaces only."
        )
        prompt = self.tokenizer.apply_chat_template(
            [{"role": "user", "content": document}],
            tokenize=True,
            return_dict=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        self.assertGreater(len(prompt), PREFILL_POOL_TOKENS)
        self.assertLess(len(prompt), 2 * PREFILL_POOL_TOKENS)
        cached_prefix = (len(prompt) - 1) // PAGE_SIZE * PAGE_SIZE
        backup_counter = "sglang:hicache_backup_tokens_total"
        backups_before = self._counter(backup_counter)
        cold = self._generate(prompt)
        self.assertEqual(cold["meta_info"]["cached_tokens"], 0)
        # Ensure write-through finished before creating eviction pressure.
        self._wait_for_counter(backup_counter, backups_before + cached_prefix)
        warm = self._generate(prompt)
        warm_cache = warm["meta_info"]["cached_tokens_details"]
        self.assertGreaterEqual(warm_cache["device"], cached_prefix)
        self.assertEqual(warm_cache["host"], 0)

        # 2 * 6144 > CP2's 8192 device slots. Including the target, these
        # unrelated prefixes still fit in its 4x larger CPU cache.
        for i in range(2):
            prefix = self.tokenizer.encode(
                f"Unrelated document {i}: ", add_special_tokens=False
            )
            pressure = (prefix + filler * (PRESSURE_TOKENS // len(filler) + 1))[
                :PRESSURE_TOKENS
            ]
            self._generate(pressure, max_tokens=1)

        load_counter = "sglang:load_back_tokens_total"
        loads_before = self._counter(load_counter)
        restored = self._generate(prompt)
        restored_cache = restored["meta_info"]["cached_tokens_details"]
        self.assertGreater(restored_cache["host"], 0, "the repeat never exercised L2")
        self.assertGreaterEqual(restored["meta_info"]["cached_tokens"], cached_prefix)
        self._wait_for_counter(load_counter, loads_before + restored_cache["host"])
        for result in (cold, warm, restored):
            self.assertEqual(re.findall(r"\d+", result["text"]), codes, result["text"])
            self.assertGreater(result["meta_info"]["spec_verify_ct"], 0)
        self.assertEqual(restored["text"], warm["text"])


ACCURACY_CP_SIZE = 2
ACCURACY_PREFILL_POOL_TOKENS = 8192  # Per rank: CP2 has 16384 logical GPU slots.
ACCURACY_PRESSURE_TOKENS = 12288
ACCURACY_HICACHE_RATIO = 4
ACCURACY_COMMON_ARGS = [
    "--page-size",
    str(PAGE_SIZE),
    "--context-length",
    "16384",
    "--max-running-requests",
    "48",
    "--mem-fraction-static",
    "0.85",
    "--kv-cache-dtype",
    "fp8_e4m3",
    "--dsa-prefill-backend",
    "trtllm",
    "--dsa-decode-backend",
    "trtllm",
    "--speculative-algorithm",
    "EAGLE",
    "--speculative-num-steps",
    "5",
    "--speculative-eagle-topk",
    "1",
    "--speculative-num-draft-tokens",
    "6",
    "--model-loader-extra-config",
    '{"enable_multithread_load": true, "num_threads": 6}',
]


@unittest.skipUnless(torch.cuda.device_count() >= 4, "requires 4 CUDA GPUs")
class TestGLM52PDShardedHiCacheAccuracy(
    PDDisaggregationServerBase, ShardedHiCacheTestMixin, GSM8KMixin
):
    model = "nvidia/GLM-5.2-NVFP4"
    prefill_tp_size = decode_tp_size = 2
    decode_base_gpu_id = 2

    # Match the layer-split accuracy gate. The evaluator excludes the 20
    # few-shot examples from scoring, leaving 1299 held-out questions.
    gsm8k_accuracy_thres = 0.935
    gsm8k_num_questions = 1319
    gsm8k_num_threads = 48
    gsm8k_num_shots = 20

    # Use the same exportable KV allocator and fabric transport as the reload
    # regression on this four-GB300 runner.
    extra_prefill_env = {
        **MOONCAKE_ENV,
        "SGLANG_UNIFIED_RADIX_TREE_CORE_BACKEND": "python",
    }
    extra_decode_env = dict(MOONCAKE_ENV)
    extra_prefill_args = ACCURACY_COMMON_ARGS + [
        "--enable-kv-cache-sharding",
        "--enable-prefill-cp",
        "--attn-cp-size",
        str(ACCURACY_CP_SIZE),
        "--cp-strategy",
        "interleave",
        "--chunked-prefill-size",
        "4096",
        "--max-prefill-tokens",
        "4096",
        "--max-total-tokens",
        str(ACCURACY_PREFILL_POOL_TOKENS),
        "--enable-hierarchical-cache",
        "--hicache-ratio",
        str(ACCURACY_HICACHE_RATIO),
        "--hicache-write-policy",
        "write_through",
        "--hicache-io-backend",
        "kernel",
        "--hicache-mem-layout",
        "page_first",
        "--enable-metrics",
    ]
    extra_decode_args = ACCURACY_COMMON_ARGS + [
        "--max-total-tokens",
        "65536",
        "--cuda-graph-max-bs-decode",
        "48",
        # Every scored question must use P/D transfer, including repeated shots.
        "--disable-radix-cache",
    ]

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.transfer_backend = ["--disaggregation-transfer-backend", "mooncake"]
        cls.launch_all()
        cls.tokenizer = get_tokenizer(cls.model, trust_remote_code=True)

    def test_gsm8k(self):
        # Flush both servers before seeding; flushing after pressure would also
        # drop the host copy and turn this into a cold-cache accuracy test.
        for url in (self.prefill_url, self.decode_url):
            response = requests.get(url + "/flush_cache", timeout=30)
            self.assertEqual(response.status_code, 200, response.text)

        data_path = download_and_cache_file(GSM8K_URL)
        few_shot_prefix = get_few_shot_examples(
            list(read_jsonl(data_path)), self.gsm8k_num_shots
        )
        backup_counter = "sglang:hicache_backup_tokens_total"
        backups_before = self._counter(backup_counter)
        seeded = self._generate(text=few_shot_prefix, max_tokens=1)
        prefix_tokens = seeded["meta_info"]["prompt_tokens"]
        cached_prefix = (prefix_tokens - 1) // PAGE_SIZE * PAGE_SIZE
        self.assertGreaterEqual(cached_prefix, ACCURACY_CP_SIZE * PAGE_SIZE)
        self.assertLess(prefix_tokens, ACCURACY_PREFILL_POOL_TOKENS * ACCURACY_CP_SIZE)
        self._wait_for_counter(backup_counter, backups_before + cached_prefix)

        # Independent prefixes exceed the *logical*, not per-rank, GPU pool.
        # They and the seeded prefix fit in host memory, retaining its L2 copy.
        self.assertGreater(
            2 * ACCURACY_PRESSURE_TOKENS,
            ACCURACY_PREFILL_POOL_TOKENS * ACCURACY_CP_SIZE,
        )
        self.assertLess(
            prefix_tokens + 2 * ACCURACY_PRESSURE_TOKENS,
            ACCURACY_PREFILL_POOL_TOKENS * ACCURACY_CP_SIZE * ACCURACY_HICACHE_RATIO,
        )
        filler = self.tokenizer.encode("The grass is green. ", add_special_tokens=False)
        for i in range(2):
            prefix = self.tokenizer.encode(
                f"Unrelated document {i}: ", add_special_tokens=False
            )
            pressure = (
                prefix + filler * (ACCURACY_PRESSURE_TOKENS // len(filler) + 1)
            )[:ACCURACY_PRESSURE_TOKENS]
            self._generate(input_ids=pressure, max_tokens=1)

        load_counter = "sglang:load_back_tokens_total"
        loads_before = self._counter(load_counter)
        # Call the same evaluator used by GSM8KMixin without its initial flush.
        # Only scored requests run after this snapshot, so H2D must occur during
        # the accuracy evaluation, not merely during the setup traffic.
        _run_accuracy_eval(
            self,
            eval_name="gsm8k",
            score_threshold=self.gsm8k_accuracy_thres,
            num_examples=self.gsm8k_num_questions,
            num_threads=self.gsm8k_num_threads,
            api="completion",
            max_tokens=512,
            num_shots=self.gsm8k_num_shots,
            gsm8k_data_path=data_path,
        )
        self._wait_for_counter(load_counter, loads_before + cached_prefix)

        # The router does not expose decode's MTP statistics.
        response = requests.get(self.decode_url + "/server_info", timeout=30)
        response.raise_for_status()
        states = response.json()["internal_states"]
        self.assertTrue(states)
        for rank, state in enumerate(states):
            with self.subTest(rank=rank):
                self.assertGreater(state["avg_spec_accept_length"], 0)


if __name__ == "__main__":
    unittest.main()
