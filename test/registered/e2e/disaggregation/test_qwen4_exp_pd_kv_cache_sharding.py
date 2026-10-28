"""Qwen3.8-Flash-Next (Qwen4-Exp) PD with KV cache sharding on the CP-TP prefill.

The prefill node runs TP4 with prefill CP4 (CP-TP group sharing): QSA attention
is CP, the GDN linear attention keeps its TP head partition. With
--enable-kv-cache-sharding each prefill rank also stores only the KV pages it
owns, so a plain-TP4 decode rank must pull its KV pages from all four prefill
ranks, but its GDN head shard only from the prefill rank with the same TP rank.
A wrong owner, local page id, send position or state peer leaves decode with
corrupt KV or GDN state while the transfer still reports success, so every
check here verifies generated content.
"""

import random
import re
import unittest

import requests
from transformers import AutoTokenizer

from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.kits.eval_accuracy_kit import GSM8KMixin
from sglang.test.server_fixtures.disaggregation_fixture import (
    PDDisaggregationServerBase,
)
from sglang.test.test_utils import try_cached_model

register_cuda_ci(est_time=600, stage="nightly", runner_config="8-gpu-b200")

MODEL = "nvidia/Qwen3.8-Flash-Next-NVFP4"
CP_SIZE = 4
PAGE_SIZE = 64
CHUNK_SIZE = 4096

# The recipe of test_qwen4_exp_pd_prefill_cp.py, on both sides so the
# transferred GDN state has the same dtype and layout.
COMMON_ARGS = [
    "--chunked-prefill-size",
    str(CHUNK_SIZE),
    "--linear-attn-prefill-backend",
    "flashinfer",
    "--linear-attn-decode-backend",
    "flashinfer",
    "--mamba-ssm-dtype",
    "bfloat16",
    "--reasoning-parser",
    "qwen3-thinking",
]
# Physical KV rows per prefill rank: 256 pages, a quarter of what four ranks
# hold together.
PREFILL_POOL_TOKENS = 16384
PREFILL_ARGS = COMMON_ARGS + [
    "--attn-cp-size",
    str(CP_SIZE),
    "--enable-prefill-cp",
    "--cp-strategy",
    "zigzag",
    "--enable-kv-cache-sharding",
    "--max-total-tokens",
    str(PREFILL_POOL_TOKENS),
    "--mem-fraction-static",
    "0.8",
]
DECODE_ARGS = COMMON_ARGS + ["--mem-fraction-static", "0.85"]

# 12 QSA layers x 2 KV heads x (K + V) x 256 dims x bf16; decode TP4 holds
# only one KV head, so its pool is no ruler for this.
KV_BYTES_PER_TOKEN = 12 * 2 * 2 * 256 * 2
# 448 pages, 1.75x one rank's pool. A fresh chain owns page p on rank
# (base + p) % 4, so pages with distinct residues mod 4 sit on four different
# ranks whatever the rotation base. Pages 20 and 423 lie more than one rank's
# pool apart: a full copy on one rank could not hold both.
LONG_PROMPT_TOKENS = 28672
LONG_NEEDLE_PAGES = {"red": 20, "green": 101, "blue": 222, "white": 423}
# Two of these documents fit the prefill pool together, so the first stays
# cached while the second is served. The 5-token last chunk is under
# 2 * CP_SIZE tokens, so it runs the non-CP QSA path over a sharded prefix.
CACHED_PROMPT_TOKENS = 3 * CHUNK_SIZE + 5
CACHED_NEEDLE_PAGES = {"red": 20, "green": 101, "blue": 150, "white": 175}
FILLER = (
    "The grass is green. The sky is blue. The sun is yellow. "
    "Here we go. There and back again. "
)


class TestQwen4ExpPDKVCacheSharding(GSM8KMixin, PDDisaggregationServerBase):
    model = try_cached_model(MODEL)
    prefill_tp_size = CP_SIZE
    decode_tp_size = CP_SIZE
    decode_base_gpu_id = CP_SIZE
    # KV sharding with the radix cache on requires the Python tree core.
    extra_prefill_env = {"SGLANG_UNIFIED_RADIX_TREE_CORE_BACKEND": "python"}
    extra_prefill_args = PREFILL_ARGS
    extra_decode_args = DECODE_ARGS

    # The gate of the unsharded sibling, test_qwen4_exp_pd_prefill_cp.py.
    gsm8k_backend = "sgl_eval"
    gsm8k_thinking = True
    gsm8k_num_examples = 200
    gsm8k_num_threads = 32
    gsm8k_max_tokens = 16384
    gsm8k_score_threshold = 0.94

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.configure_transfer_backend()
        cls.tokenizer = AutoTokenizer.from_pretrained(cls.model)
        cls.launch_all()

    @classmethod
    def configure_transfer_backend(cls):
        """Keep the fixture's transfer backend (Mooncake in CI)."""

    def _server_info(self, url: str) -> dict:
        response = requests.get(url + "/server_info", timeout=60)
        response.raise_for_status()
        return response.json()

    def _needle_prompt(
        self, length: int, needle_pages: dict, keys: dict, color: str
    ) -> list[int]:
        """Token ids of ``length``; each pass key starts its own page."""
        encode = self.tokenizer.encode
        filler = encode(FILLER, add_special_tokens=False)

        def pad_to(ids: list[int], target: int) -> list[int]:
            while len(ids) < target:
                ids = ids + filler[: target - len(ids)]
            return ids

        ids = encode(
            "<|im_start|>user\nThere is important info hidden inside a lot of "
            "irrelevant text. Find it and memorize it.\n",
            add_special_tokens=False,
        )
        for name, page in sorted(needle_pages.items(), key=lambda item: item[1]):
            ids = pad_to(ids, page * PAGE_SIZE)
            ids += encode(
                f"The pass key for the {name} door is {keys[name]}. Remember it. ",
                add_special_tokens=False,
            )
        # Prefill samples one token; decode generates the rest of the key
        # from the KV and GDN state it received.
        question = encode(
            f"\nWhat is the pass key for the {color} door?<|im_end|>\n"
            "<|im_start|>assistant\n<think>\n\n</think>\n\n"
            f"The pass key for the {color} door is",
            add_special_tokens=False,
        )
        ids = pad_to(ids, length - len(question)) + question
        self.assertEqual(len(ids), length)
        return ids

    def _ask(self, input_ids: list[int]) -> dict:
        response = requests.post(
            self.lb_url + "/generate",
            json={
                "input_ids": input_ids,
                "sampling_params": {"temperature": 0, "max_new_tokens": 8},
            },
            timeout=300,
        )
        self.assertEqual(response.status_code, 200, response.text)
        return response.json()

    def _assert_key(self, output: dict, key: int):
        answer = re.search(r"\d+", output["text"])
        self.assertIsNotNone(answer, output["text"])
        self.assertEqual(answer.group(), str(key), output["text"])

    def test_a_topology_and_footprint(self):
        info = self._server_info(self.prefill_url)
        self.assertTrue(info["enable_cp_tp_group_sharing"])
        self.assertTrue(info["enable_kv_cache_sharding"])
        self.assertEqual(info["attn_cp_size"], CP_SIZE)
        self.assertEqual(info["page_size"], PAGE_SIZE)
        # Fail fast: a prefill-side reject reaches the client only through
        # decode's KV-wait timeout.
        self.assertGreater(info["max_req_input_len"], LONG_PROMPT_TOKENS)
        prefill_mem = info["internal_states"][0]["memory_usage"]
        self.assertEqual(prefill_mem["token_capacity"], PREFILL_POOL_TOKENS)
        # The prompt's full-attention KV exceeds one prefill rank's whole
        # cache, compressed keys included.
        prompt_kv_gb = LONG_PROMPT_TOKENS * KV_BYTES_PER_TOKEN / (1 << 30)
        print(f"{prompt_kv_gb=:.3f} vs one prefill rank's cache {prefill_mem=}")
        self.assertGreater(prompt_kv_gb, 1.3 * prefill_mem["kvcache"])

    def test_b_needles_on_every_owner_rank(self):
        rng = random.Random(0)
        for color in LONG_NEEDLE_PAGES:
            # New keys per request: a flush does not zero the pools, so a
            # reused key could be answered from stale rows.
            keys = {name: rng.randint(10000, 99999) for name in LONG_NEEDLE_PAGES}
            input_ids = self._needle_prompt(
                LONG_PROMPT_TOKENS, LONG_NEEDLE_PAGES, keys, color
            )
            # Start from an empty prefill cache so every page is allocated now.
            requests.post(
                self.prefill_url + "/flush_cache", params={"timeout": 30}, timeout=60
            ).raise_for_status()
            output = self._ask(input_ids)
            meta_info = output["meta_info"]
            print(f"{color}: {output['text']!r} {meta_info['cached_tokens']=}")
            self.assertEqual(meta_info["prompt_tokens"], LONG_PROMPT_TOKENS)
            self.assertEqual(meta_info["cached_tokens"], 0)
            self._assert_key(output, keys[color])

    def test_c_cached_prefix_transfers_every_page(self):
        rng = random.Random(1)
        requests.post(
            self.prefill_url + "/flush_cache", params={"timeout": 30}, timeout=60
        ).raise_for_status()
        docs = [
            {name: rng.randint(10000, 99999) for name in CACHED_NEEDLE_PAGES}
            for _ in range(2)
        ]
        for keys in docs:
            output = self._ask(
                self._needle_prompt(
                    CACHED_PROMPT_TOKENS, CACHED_NEEDLE_PAGES, keys, "red"
                )
            )
            self._assert_key(output, keys["red"])
        # The first document again: the prefill serves most of it from its
        # sharded radix cache, while decode still needs every page, written
        # into pages that held the second document.
        output = self._ask(
            self._needle_prompt(
                CACHED_PROMPT_TOKENS, CACHED_NEEDLE_PAGES, docs[0], "white"
            )
        )
        print(f"cached: {output['text']!r} {output['meta_info']['cached_tokens']=}")
        self.assertGreaterEqual(output["meta_info"]["cached_tokens"], CHUNK_SIZE)
        self._assert_key(output, docs[0]["white"])

    def test_d_short_prompt(self):
        # One page on one rank: the other three ranks own nothing and send only
        # their GDN head shard. Under 2 * CP_SIZE tokens the prefill runs the
        # non-CP path. Prefill samples " four"; decode must continue from the
        # transferred state.
        input_ids = self.tokenizer.encode("one, two, three,", add_special_tokens=False)
        self.assertLess(len(input_ids), 2 * CP_SIZE)
        response = requests.post(
            self.base_url + "/generate",
            json={
                "input_ids": input_ids,
                "sampling_params": {"temperature": 0, "max_new_tokens": 8},
            },
            timeout=600,
        )
        self.assertEqual(response.status_code, 200, response.text)
        self.assertIn("five, six", response.json()["text"])


if __name__ == "__main__":
    unittest.main()
