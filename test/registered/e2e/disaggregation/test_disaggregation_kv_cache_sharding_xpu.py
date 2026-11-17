"""Logical-page KV cache sharding for PD prefill on Intel XPU.

The XPU counterpart of test_disaggregation_kv_cache_sharding.py. KV is striped
across two prefill ranks: MLA latent KV over attention TP, or GQA KV over prefill
CP (zigzag). The intel_xpu backend reads each layer's prefix from the scratch the
ranks all-gather over XCCL, and decode pulls each rank's stripe. A wrong owner,
scratch translation or gather corrupts decode's KV and collapses GSM8K accuracy;
a prompt whose KV exceeds one prefill rank's pool can only be served if no rank
keeps a full copy.

Requirements: 4 XPUs (prefill TP2 on 0-1, decode TP2 on 2-3) and Mooncake built
with XPU/TENT support (docker/xpu.Dockerfile).
"""

import concurrent.futures
import random
import re
import unittest

import requests
import torch

from sglang.srt.hf_transformers import get_tokenizer
from sglang.test.ci.ci_register import register_xpu_ci
from sglang.test.kits.eval_accuracy_kit import GSM8KMixin
from sglang.test.server_fixtures.disaggregation_fixture import (
    PDDisaggregationServerBase,
)
from sglang.test.test_utils import (
    DEFAULT_MLA_MODEL_NAME_FOR_TEST,
    is_in_ci,
)

register_xpu_ci(est_time=1500, suite="nightly-kernel-main-test-4-gpu-xpu", nightly=True)

_XPU_DEVICE_COUNT = torch.xpu.device_count() if torch.xpu.is_available() else 0
_SKIP_REASON = f"KV sharding PD needs 4 XPUs, found {_XPU_DEVICE_COUNT}"

PAGE_SIZE = 128
COMMON_ARGS = [
    "--device",
    "xpu",
    "--attention-backend",
    "intel_xpu",
    "--page-size",
    str(PAGE_SIZE),
]
# Physical KV rows per prefill rank.
PREFILL_POOL_TOKENS = 16384
# 1.5x one rank's rows, under the sharded bootstrap limit 2 * 16384 - 3 * 128.
LONG_PROMPT_TOKENS = 24576
# Page p of a fresh prompt is owned by prefill rank p % 2, so these needles sit on
# different ranks. Page 10 ends before the last 16384 tokens and page 171 starts
# after token 16384, so a single 16384-row copy could not hold both.
NEEDLE_PAGES = {"red": 10, "blue": 171}
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
    question = tokenizer.encode(
        f"\nWhat is the pass key for the {color} door? "
        f"The pass key for the {color} door is",
        add_special_tokens=False,
    )
    return pad_to(ids, LONG_PROMPT_TOKENS - len(question)) + question


class _KVCacheShardingXpuMixin:
    kv_transfer_backend = "mooncake"
    prefill_tp_size = 2
    decode_tp_size = 2
    decode_base_gpu_id = 2
    use_rdma = False
    extra_prefill_env = {
        # KV sharding with the radix cache on requires the Python tree core.
        "SGLANG_UNIFIED_RADIX_TREE_CORE_BACKEND": "python",
        "MC_USE_TENT": "1",
    }
    extra_decode_env = {"MC_USE_TENT": "1"}
    # Decode ranks that split one token's KV: 1 for MLA (latent replicated per
    # rank), decode TP for GQA (heads split across ranks).
    decode_kv_split = 1
    extra_decode_args = COMMON_ARGS + ["--mem-fraction-static", "0.75"]

    @classmethod
    def setUpClass(cls):
        if _XPU_DEVICE_COUNT < 4:
            raise RuntimeError(_SKIP_REASON)
        super().setUpClass()
        cls.transfer_backend = [
            "--disaggregation-transfer-backend",
            cls.kv_transfer_backend,
        ]
        cls.rdma_devices = []
        cls.launch_all()

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
        full_kv_gb_per_token = (
            decode_mem["kvcache"] / decode_mem["token_capacity"] * self.decode_kv_split
        )
        prompt_kv_gb = LONG_PROMPT_TOKENS * full_kv_gb_per_token
        self.assertGreater(prompt_kv_gb, 1.3 * prefill_mem["kvcache"])

        tokenizer = get_tokenizer(self.model)
        rng = random.Random(0)
        for color in NEEDLE_PAGES:
            # New keys per request: a flush does not zero the pools.
            keys = {name: rng.randint(10000, 99999) for name in NEEDLE_PAGES}
            input_ids = build_needle_prompt(tokenizer, keys, color)
            requests.post(
                self.prefill_url + "/flush_cache", params={"timeout": 30}, timeout=60
            ).raise_for_status()
            response = requests.post(
                self.lb_url + "/generate",
                json={
                    "input_ids": input_ids,
                    "sampling_params": {"temperature": 0, "max_new_tokens": 8},
                },
                timeout=600,
            )
            self.assertEqual(response.status_code, 200, response.text)
            meta_info = response.json()["meta_info"]
            self.assertEqual(meta_info["prompt_tokens"], LONG_PROMPT_TOKENS)
            self.assertEqual(meta_info["cached_tokens"], 0)
            answer = re.search(r"\d+", response.json()["text"])
            self.assertIsNotNone(answer, response.json()["text"])
            self.assertEqual(answer.group(), str(keys[color]))

    def test_concurrent_short_prompts_all_finish(self):
        """Prompts of about one page leave some shard rank without a page in a
        chunk; every request must still finish rather than wait for a writer
        that never reports."""
        prompts = [
            f"Question {i}: what is {i} plus {i + 1}? Answer with the number only."
            for i in range(32)
        ]

        def generate(prompt):
            return requests.post(
                self.lb_url + "/generate",
                json={
                    "text": prompt,
                    "sampling_params": {"temperature": 0, "max_new_tokens": 8},
                },
                timeout=120,
            )

        with concurrent.futures.ThreadPoolExecutor(len(prompts)) as pool:
            responses = list(pool.map(generate, prompts))
        for response in responses:
            self.assertEqual(response.status_code, 200, response.text)


# 256-token chunks split the needle prompt into 96 chunks, so later chunks read a
# sharded prefix and every rank sends pages from many chunks. GSM8K prompts are
# zero-shot and fit in one chunk; they check decode on many short requests.
_SHARDED_PREFILL_ARGS = COMMON_ARGS + [
    "--enable-kv-cache-sharding",
    "--chunked-prefill-size",
    "256",
    "--max-total-tokens",
    str(PREFILL_POOL_TOKENS),
]


@unittest.skipUnless(_XPU_DEVICE_COUNT >= 4 or is_in_ci(), _SKIP_REASON)
class TestDisaggregationKVCacheShardingXpu(
    _KVCacheShardingXpuMixin, PDDisaggregationServerBase, GSM8KMixin
):
    """MLA latent KV striped over the two prefill attention-TP ranks."""

    model = DEFAULT_MLA_MODEL_NAME_FOR_TEST
    extra_prefill_args = _SHARDED_PREFILL_ARGS
    gsm8k_backend = "sgl_eval"
    gsm8k_score_threshold = 0.70
    # Uncapped greedy decoding loops past 14K tokens on one question.
    gsm8k_max_tokens = 2048


@unittest.skipUnless(_XPU_DEVICE_COUNT >= 4 or is_in_ci(), _SKIP_REASON)
class TestDisaggregationKVCacheShardingGqaCpXpu(
    _KVCacheShardingXpuMixin, PDDisaggregationServerBase
):
    """GQA KV striped over two zigzag prefill CP ranks (attention TP 1)."""

    model = "Qwen/Qwen3-8B"
    decode_kv_split = 2
    extra_prefill_args = _SHARDED_PREFILL_ARGS + [
        "--attn-cp-size",
        "2",
        "--enable-prefill-cp",
        "--cp-strategy",
        "zigzag",
    ]


@unittest.skipUnless(_XPU_DEVICE_COUNT >= 4 or is_in_ci(), _SKIP_REASON)
class TestDisaggregationKVCacheShardingGqaCpStagedXpu(
    TestDisaggregationKVCacheShardingGqaCpXpu
):
    """The GQA-CP stripes packed through the Mooncake staging buffer, as the XPU
    docs recommend: each CP rank writes its own pages of every chunk's region."""

    extra_prefill_env = {
        **_KVCacheShardingXpuMixin.extra_prefill_env,
        "SGLANG_DISAGG_STAGING_BUFFER": "1",
    }
    extra_decode_env = {
        **_KVCacheShardingXpuMixin.extra_decode_env,
        "SGLANG_DISAGG_STAGING_BUFFER": "1",
    }


if __name__ == "__main__":
    unittest.main()
