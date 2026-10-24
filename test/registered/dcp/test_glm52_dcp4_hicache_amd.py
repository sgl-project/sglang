import os
import random
import unittest

import requests

from sglang.test.ci.ci_register import register_amd_ci
from sglang.test.kits.unified_radix_cache_kit import UnifiedRadixTreeTestMixin
from sglang.test.kl_multiturn_utils import (
    get_input_ids,
    make_mamba_decode_assert,
    make_mamba_prefill_assert,
)
from sglang.test.server_fixtures.dsa_mtp_fixture import DsaMtpServerBase

register_amd_ci(est_time=1800, suite="nightly-amd-8-gpu-mi35x", nightly=True)

GLM52_MODEL_PATH = os.environ.get("GLM52_MODEL_PATH", "amd/GLM-5.2-MXFP4")
DCP_SIZE = 4
WIDENED_PAGE = 64 * DCP_SIZE
DCP_ARGS = [
    "--kv-cache-dtype",
    "fp8_e4m3",
    "--dsa-prefill-backend",
    "triton",
    "--dsa-decode-backend",
    "triton",
    "--dcp-size",
    str(DCP_SIZE),
    "--random-seed",
    "0",
]


class TestGLM52MXFP4DCP4MTPHiCache(DsaMtpServerBase, UnifiedRadixTreeTestMixin):
    """GLM-5.2 TP4 + DCP4 + MTP with HiCache L2 on ROCm: the sharded KV, the
    widened index keys and the packed draft KV must replay from cache."""

    model = GLM52_MODEL_PATH
    tp_size = 4
    mem_fraction_static = 0.8
    gsm8k_threshold = 0.9
    # TP4 + MTP replays at KL 0.09-0.20 with or without DCP and HiCache
    # (MI355X, fp8 KV, temperature 1).
    kl_threshold = 0.25
    prefill_cache_assert = staticmethod(
        make_mamba_prefill_assert(chunk_size=WIDENED_PAGE)
    )
    decode_cache_assert = staticmethod(
        make_mamba_decode_assert(track_interval=WIDENED_PAGE)
    )
    extra_server_args = DCP_ARGS + [
        "--enable-hierarchical-cache",
        "--hicache-ratio",
        "1.5",
        "--hicache-write-policy",
        "write_through",
    ]

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.input_ids = get_input_ids(cls.model, num_samples=12, trust_remote_code=True)


class TestGLM52MXFP4DCP4MTPHiCacheEvicted(TestGLM52MXFP4DCP4MTPHiCache):
    """Same, with a device pool of a few sequences, so cached prefixes are
    evicted and the replays load their KV, index keys and draft KV from host."""

    extra_server_args = DCP_ARGS + [
        "--max-total-tokens",
        "4096",
        "--enable-hierarchical-cache",
        "--hicache-size",
        "60",
        "--hicache-write-policy",
        "write_through",
    ]

    def _generate(self, input_ids):
        response = requests.post(
            self.base_url + "/generate",
            json={
                "input_ids": input_ids,
                "sampling_params": {"max_new_tokens": 1, "temperature": 0},
            },
        )
        response.raise_for_status()
        return response.json()

    def test_evicted_prefix_reloads_from_host(self):
        rng = random.Random(7)
        prompt_len = 3000
        # --max-total-tokens caps each rank's rows, so DCP4 widens 4096 rows
        # into 16384 logical slots. Eight 3000-token prompts (24000 tokens)
        # overflow them, so the first one leaves the device and survives only
        # in the write-through host copy.
        prompts = [[rng.randint(1, 30000) for _ in range(prompt_len)] for _ in range(8)]
        for input_ids in prompts:
            self._generate(input_ids)
        result = self._generate(prompts[0])
        details = result["meta_info"].get("cached_tokens_details") or {}
        self.assertGreater(details.get("host", 0), 0, result["meta_info"])
        self.prefill_cache_assert(result, prompt_len, "host reload")


if __name__ == "__main__":
    unittest.main()
