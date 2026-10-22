"""
E2E tests for the unified L3 key scheme on the Mooncake HiCache backend.

Mooncake stores one object per unified chunk, so these servers cut every page
into several chunks along the head and layer axes. An L3 reload must reproduce
the device-cache result bit for bit.

Usage:
    python3 -m pytest test/registered/hicache/test_hicache_storage_mooncake_unified.py -v
"""

import json
import random
import unittest
from types import SimpleNamespace

import requests
from test_hicache_storage_mooncake_backend import (
    HiCacheStorageMooncakeBackendBaseMixin,
)

from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.sgl_eval_utils import run_sgl_eval
from sglang.test.test_utils import DEFAULT_MLA_MODEL_NAME_FOR_TEST, CustomTestCase

register_cuda_ci(est_time=400, stage="base-b", runner_config="2-gpu-large")

PAGE_SIZE = 64


def run_gsm8k(*, base_url, model, num_examples):
    return run_sgl_eval(
        SimpleNamespace(
            base_url=base_url,
            model=model,
            eval_name="gsm8k",
            num_examples=num_examples,
            num_threads=64,
        )
    )["score"]


def storage_hit_tokens(base_url):
    metrics = requests.get(f"{base_url}/metrics", timeout=10).text
    return sum(
        float(line.rsplit(" ", 1)[1])
        for line in metrics.splitlines()
        if line.startswith("sglang:storage_prefetch_hit_tokens_total")
    )


class HiCacheMooncakeUnifiedMixin(HiCacheStorageMooncakeBackendBaseMixin):
    """Mooncake L3 under --hicache-storage-key-scheme unified, at TP2."""

    unified_server_args = {}

    @classmethod
    def _get_additional_server_args_and_env(cls):
        server_args, env_vars = super()._get_additional_server_args_and_env()
        extra_config = {
            "hicache_storage_pass_prefix_keys": True,
            # A single-shot GSM8K prompt spans 1-3 pages; prefetch them from L3.
            "prefetch_threshold": PAGE_SIZE,
        }
        server_args.update(
            {
                "--page-size": PAGE_SIZE,
                "--enable-metrics": True,
                "--hicache-storage-backend-extra-config": json.dumps(extra_config),
                "--hicache-storage-key-scheme": "unified",
                "--hicache-mem-layout": "page_unified",
                "--hicache-io-backend": "kernel",
                **cls.unified_server_args,
            }
        )
        # Room for every GSM8K page, so the second pass is not cut by eviction.
        env_vars["MOONCAKE_GLOBAL_SEGMENT_SIZE"] = str(16 << 30)
        return server_args, env_vars

    def _generate_greedy(self, input_ids):
        response = requests.post(
            f"{self.base_url}/generate",
            json={
                "input_ids": input_ids,
                "sampling_params": {
                    "temperature": 0,
                    "max_new_tokens": 32,
                    "ignore_eos": True,
                },
                "return_logprob": True,
            },
            timeout=120,
        )
        self.assertEqual(response.status_code, 200, response.text)
        return response.json()

    def test_l3_reload_is_bit_exact(self):
        """An L3 reload reproduces a device-cache hit token for token.

        Both reuse the same page-aligned prefix and compute the same suffix, so
        a chunk that came back short, overlapped another or went missing shows
        up. Placement across topologies is the cross-TP test's job.
        """
        rng = random.Random(20261021)
        # One token past a page boundary: both hits reuse exactly 12 full pages.
        input_ids = [rng.randint(1000, 30000) for _ in range(12 * PAGE_SIZE + 1)]

        self._generate_greedy(input_ids)
        device_hit = self._generate_greedy(input_ids)
        self.trigger_offloading_and_flush()
        storage_hit = self._generate_greedy(input_ids)

        meta = storage_hit["meta_info"]
        details = meta.get("cached_tokens_details") or {}
        self.assertEqual(details.get("storage"), 12 * PAGE_SIZE, details)
        self.assertEqual(details.get("storage_backend"), "MooncakeStore", details)
        self.assertEqual(
            meta["cached_tokens"], device_hit["meta_info"]["cached_tokens"]
        )
        self.assertEqual(storage_hit["output_ids"], device_hit["output_ids"])
        self.assertEqual(
            meta["output_token_logprobs"],
            device_hit["meta_info"]["output_token_logprobs"],
        )


class TestMooncakeUnifiedMHA(HiCacheMooncakeUnifiedMixin, CustomTestCase):
    """GQA at TP2: each rank owns 2 head groups x 4 layer ranges per page."""

    unified_server_args = {
        "--hicache-storage-head-partition": 2,
        "--hicache-storage-layer-partition": 8,
    }

    def test_eval_accuracy(self):
        """GSM8K holds when its prompt pages come back from mooncake chunks."""
        first = run_gsm8k(base_url=self.base_url, model=self.model, num_examples=200)
        self.flush_cache()
        hits_before = storage_hit_tokens(self.base_url)
        reloaded = run_gsm8k(base_url=self.base_url, model=self.model, num_examples=200)
        hit_tokens = storage_hit_tokens(self.base_url) - hits_before
        print(f"{first=:.3f} {reloaded=:.3f} {hit_tokens=:.0f}")
        self.assertGreaterEqual(hit_tokens, 100 * PAGE_SIZE)
        self.assertGreater(first, 0.6)
        self.assertGreater(reloaded, 0.6)
        self.assertLess(abs(first - reloaded), 0.03)


class TestMooncakeUnifiedMLA(HiCacheMooncakeUnifiedMixin, CustomTestCase):
    """MLA at TP2: TP0 alone writes, and 27 layers end in a short L24-27 chunk."""

    unified_server_args = {"--hicache-storage-layer-partition": 8}

    @classmethod
    def _get_model_name(cls):
        return DEFAULT_MLA_MODEL_NAME_FOR_TEST


if __name__ == "__main__":
    unittest.main(verbosity=2)
