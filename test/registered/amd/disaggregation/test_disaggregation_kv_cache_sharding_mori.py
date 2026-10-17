"""ROCm PD accuracy and per-rank KV footprint with logical-page KV cache sharding
over MoRI.

Each prefill TP rank stores only its stripe of every KV page and sends only the
pages it owns; decode pulls from both prefill ranks. A wrong owner, local page
id, or send position corrupts decode's KV and collapses GSM8K accuracy. A prompt
whose KV is larger than one prefill rank's whole pool can only be served if no
rank keeps a full copy.
"""

import os
import random
import re
import unittest

import requests
import torch

from sglang.srt.utils.hf_transformers_utils import get_tokenizer
from sglang.test.ci.ci_register import register_amd_ci
from sglang.test.kits.eval_accuracy_kit import GSM8KMixin
from sglang.test.server_fixtures.disaggregation_fixture import (
    PDDisaggregationServerBase,
)
from sglang.test.test_utils import DEFAULT_MLA_MODEL_NAME_FOR_TEST

register_amd_ci(est_time=600, suite="stage-b-test-large-8-gpu-mi35x-disaggregation-amd")

PAGE_SIZE = 64
COMMON_ARGS = ["--attention-backend", "aiter", "--page-size", str(PAGE_SIZE)]
# Physical KV rows per prefill rank. 512 logical pages, so GSM8K's prefill
# radix footprint (~243 pages) never forces eviction.
PREFILL_POOL_TOKENS = 16384
# 1.5x one rank's rows, well under the sharded admission bound ~2 * (16384 - 64).
LONG_PROMPT_TOKENS = 24576
# Page p of a fresh prompt is owned by prefill rank p % 2, so these needles sit
# on different ranks. Page 20 ends before the last 16448 tokens; page 341
# starts after token 16448: a 16448-row full copy could not hold both.
NEEDLE_PAGES = {"red": 20, "blue": 341}
FILLER = (
    "The grass is green. The sky is blue. The sun is yellow. "
    "Here we go. There and back again. "
)


def build_needle_prompt(tokenizer, keys: dict[str, int], color: str) -> list[int]:
    """Token ids of LONG_PROMPT_TOKENS; each pass key starts its own page."""
    filler = tokenizer.encode(FILLER, add_special_tokens=False)

    def pad_to(ids: list[int], length: int) -> list[int]:
        while len(ids) < length:
            ids = ids + filler[: length - len(ids)]
        return ids

    ids = tokenizer.encode(
        "There is important info hidden inside a lot of irrelevant text. "
        "Find it and memorize it.\n"
    )
    for name, page in sorted(NEEDLE_PAGES.items(), key=lambda item: item[1]):
        ids = pad_to(ids, page * PAGE_SIZE)
        ids += tokenizer.encode(
            f"The pass key for the {name} door is {keys[name]}. Remember it. ",
            add_special_tokens=False,
        )
    # Prefill samples only the space; decode generates every digit from the KV
    # it received.
    question = tokenizer.encode(
        f"\nWhat is the pass key for the {color} door? "
        f"The pass key for the {color} door is",
        add_special_tokens=False,
    )
    return pad_to(ids, LONG_PROMPT_TOKENS - len(question)) + question


class TestDisaggregationKVCacheShardingMori(PDDisaggregationServerBase, GSM8KMixin):
    model = DEFAULT_MLA_MODEL_NAME_FOR_TEST
    prefill_tp_size = 2
    decode_tp_size = 2
    decode_base_gpu_id = 2
    # KV sharding with the radix cache on requires the Python tree core.
    extra_prefill_env = {
        "SGLANG_UNIFIED_RADIX_TREE_CORE_BACKEND": "python",
        "SGLANG_USE_AITER": "1",
        # Batches over this many KV tokens read the cached prefix in chunks
        # (MHA_CHUNKED_KV); the 128K default would leave that reader untested.
        "SGLANG_MAX_KV_CHUNK_CAPACITY": "8192",
    }
    extra_decode_env = {"SGLANG_USE_AITER": "1"}
    # 256-token chunks split each 5-shot prompt, so later chunks read a
    # sharded prefix and every rank sends pages from several chunks.
    extra_prefill_args = COMMON_ARGS + [
        "--enable-kv-cache-sharding",
        "--chunked-prefill-size",
        "256",
        "--max-total-tokens",
        str(PREFILL_POOL_TOKENS),
    ]
    extra_decode_args = COMMON_ARGS
    gsm8k_score_threshold = 0.70

    @classmethod
    def setUpClass(cls):
        if torch.cuda.device_count() < 4:
            raise unittest.SkipTest("MoRI KV sharding PD test requires 4 GPUs.")
        super().setUpClass()
        # In CI the shared fixture picks Mooncake and per-GPU NICs; MoRI runs
        # on the NICs in SGLANG_TEST_RDMA_DEVICE.
        cls.transfer_backend = ["--disaggregation-transfer-backend", "mori"]
        rdma_env = os.environ.get("SGLANG_TEST_RDMA_DEVICE")
        cls.rdma_devices = ["--disaggregation-ib-device", rdma_env] if rdma_env else []
        cls.launch_all()

    @classmethod
    def rdma_devices_for(cls, gpu_indices) -> list:
        return cls.rdma_devices

    def _server_info(self, url: str) -> dict:
        response = requests.get(url + "/server_info", timeout=60)
        response.raise_for_status()
        return response.json()

    def test_prompt_kv_larger_than_one_rank_pool(self):
        prefill_info = self._server_info(self.prefill_url)
        # Fail fast: a prefill-side reject reaches the client only through
        # decode's KV-wait timeout.
        self.assertGreater(prefill_info["max_req_input_len"], LONG_PROMPT_TOKENS)
        prefill_mem = prefill_info["internal_states"][0]["memory_usage"]
        decode_mem = self._server_info(self.decode_url)["internal_states"][0][
            "memory_usage"
        ]
        self.assertEqual(prefill_mem["token_capacity"], PREFILL_POOL_TOKENS)
        # Decode's unsharded pool of the same model gives the full KV per token.
        full_kv_gb_per_token = decode_mem["kvcache"] / decode_mem["token_capacity"]
        self.assertGreater(
            LONG_PROMPT_TOKENS * full_kv_gb_per_token, 1.3 * prefill_mem["kvcache"]
        )

        tokenizer = get_tokenizer(self.model)
        rng = random.Random(0)
        for color in NEEDLE_PAGES:
            # New keys per request: a flush does not zero the pools, so a
            # reused key could be answered from stale rows.
            keys = {name: rng.randint(10000, 99999) for name in NEEDLE_PAGES}
            # Start from an empty prefill cache so every page is allocated now.
            requests.post(
                self.prefill_url + "/flush_cache", params={"timeout": 30}, timeout=60
            ).raise_for_status()
            response = requests.post(
                self.lb_url + "/generate",
                json={
                    "input_ids": build_needle_prompt(tokenizer, keys, color),
                    "sampling_params": {"temperature": 0, "max_new_tokens": 8},
                },
                timeout=300,
            )
            self.assertEqual(response.status_code, 200, response.text)
            output = response.json()
            self.assertEqual(output["meta_info"]["prompt_tokens"], LONG_PROMPT_TOKENS)
            self.assertEqual(output["meta_info"]["cached_tokens"], 0)
            answer = re.search(r"\d+", output["text"])
            self.assertIsNotNone(answer, output["text"])
            self.assertEqual(answer.group(), str(keys[color]), output["text"])


if __name__ == "__main__":
    unittest.main()
