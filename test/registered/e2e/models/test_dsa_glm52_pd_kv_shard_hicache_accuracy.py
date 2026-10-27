"""GSM8K accuracy after reloading CP-sharded GLM DSA/MTP KV from CPU HiCache.

Use the full-eval and mixed-MoE settings from the layer-split E2E on eight B300s
(P TP4/CP4 + D TP4). Seed the evaluator's few-shot prefix, evict it from GPU,
then score GSM8K while requiring actual H2D reloads. The four-GB300 companion
test separately checks cold/device-hit/host-hit answers and PD/MTP metadata.
"""

import time
import unittest

import requests
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

register_cuda_ci(est_time=1500, stage="extra-b", runner_config="8-gpu-b300")

PAGE_SIZE = 64
CP_SIZE = 4
PREFILL_POOL_TOKENS = 8192  # Per rank: CP4 has 32768 logical GPU slots.
PRESSURE_TOKENS = 24576
HICACHE_RATIO = 4
COMMON_ARGS = [
    "--page-size",
    str(PAGE_SIZE),
    "--context-length",
    "32768",
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


class TestGLM52PDShardedHiCacheAccuracy(PDDisaggregationServerBase, GSM8KMixin):
    model = "nvidia/GLM-5.2-NVFP4"
    prefill_tp_size = decode_tp_size = 4
    decode_base_gpu_id = 4

    # Match the layer-split accuracy gate. The evaluator excludes the 20
    # few-shot examples from scoring, leaving 1299 held-out questions.
    gsm8k_accuracy_thres = 0.935
    gsm8k_num_questions = 1319
    gsm8k_num_threads = 48
    gsm8k_num_shots = 20

    # Use the B300 fixture's Mooncake/RDMA transport, as layer-split does.
    # The four-GB300 companion test has its own fabric NVLink configuration.
    extra_prefill_env = {"SGLANG_UNIFIED_RADIX_TREE_CORE_BACKEND": "python"}
    extra_prefill_args = COMMON_ARGS + [
        "--enable-kv-cache-sharding",
        "--enable-prefill-cp",
        "--attn-cp-size",
        str(CP_SIZE),
        "--cp-strategy",
        "interleave",
        "--chunked-prefill-size",
        "4096",
        "--max-prefill-tokens",
        "4096",
        "--max-total-tokens",
        str(PREFILL_POOL_TOKENS),
        "--enable-hierarchical-cache",
        "--hicache-ratio",
        str(HICACHE_RATIO),
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
        "131072",
        # Every scored question must use P/D transfer, including repeated shots.
        "--disable-radix-cache",
    ]

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.transfer_backend = ["--disaggregation-transfer-backend", "mooncake"]
        cls.launch_all()
        cls.tokenizer = get_tokenizer(cls.model, trust_remote_code=True)

    def _generate(self, **prompt):
        response = requests.post(
            self.base_url + "/generate",
            json={
                **prompt,
                "sampling_params": {"temperature": 0, "max_new_tokens": 1},
            },
            timeout=600,
        )
        self.assertEqual(response.status_code, 200, response.text)
        return response.json()

    def _counter(self, name):
        response = requests.get(self.prefill_url + "/metrics", timeout=30)
        response.raise_for_status()
        # Logical token counts repeat across CP ranks; read only TP rank 0.
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
        seeded = self._generate(text=few_shot_prefix)
        prefix_tokens = seeded["meta_info"]["prompt_tokens"]
        cached_prefix = (prefix_tokens - 1) // PAGE_SIZE * PAGE_SIZE
        self.assertGreaterEqual(cached_prefix, CP_SIZE * PAGE_SIZE)
        self.assertLess(prefix_tokens, PREFILL_POOL_TOKENS * CP_SIZE)
        self._wait_for_counter(backup_counter, backups_before + cached_prefix)

        # Independent prefixes exceed the *logical*, not per-rank, GPU pool.
        # They and the seeded prefix fit in host memory, retaining its L2 copy.
        self.assertGreater(2 * PRESSURE_TOKENS, PREFILL_POOL_TOKENS * CP_SIZE)
        self.assertLess(
            prefix_tokens + 2 * PRESSURE_TOKENS,
            PREFILL_POOL_TOKENS * CP_SIZE * HICACHE_RATIO,
        )
        filler = self.tokenizer.encode("The grass is green. ", add_special_tokens=False)
        for i in range(2):
            prefix = self.tokenizer.encode(
                f"Unrelated document {i}: ", add_special_tokens=False
            )
            pressure = (prefix + filler * (PRESSURE_TOKENS // len(filler) + 1))[
                :PRESSURE_TOKENS
            ]
            self._generate(input_ids=pressure)

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


class TestGLM52PDShardedHiCacheAccuracyMixedMoE(TestGLM52PDShardedHiCacheAccuracy):
    """Match layer-split's target DeepEP + TP-only EAGLE draft smoke coverage."""

    gsm8k_accuracy_thres = 0.8
    gsm8k_num_questions = 32
    gsm8k_num_threads = 32
    extra_decode_args = TestGLM52PDShardedHiCacheAccuracy.extra_decode_args + [
        "--attn-dp-size",
        "4",
        "--enable-dp-lm-head",
        "--ep-size",
        "4",
        "--moe-dense-tp-size",
        "1",
        "--moe-a2a-backend",
        "deepep",
        "--deepep-mode",
        "low_latency",
        "--moe-runner-backend",
        "flashinfer_cutedsl",
        "--speculative-moe-a2a-backend",
        "none",
        "--speculative-moe-runner-backend",
        "triton",
        "--cuda-graph-max-bs-decode",
        "8",
        "--max-running-requests",
        "8",
        "--disable-prefill-cuda-graph",
        "--disable-flashinfer-autotune",
    ]


if __name__ == "__main__":
    unittest.main()
