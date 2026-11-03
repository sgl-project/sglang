"""HiCache L2 under decode context parallelism (DCP) + UnifiedRadixCache.

Under DCP the radix layer allocates widened logical indices while each rank's
buffers hold only its 1/dcp_size shard, so a missing translation makes cache
hits return another rank's KV. The KL cases catch that as a large divergence.

The MLA case requires Blackwell for ``cutedsl_mla``. The GQA case uses Triton.
"""

import math
import os
import shlex
import unittest

import requests

from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.kits.unified_radix_cache_kit import UnifiedRadixTreeTestMixin
from sglang.test.kl_multiturn_utils import (
    get_input_ids,
    make_mamba_decode_assert,
    make_mamba_prefill_assert,
)
from sglang.test.kl_test_utils import (
    _extract_output_logprobs,
    _flush_cache,
    _generate,
)
from sglang.test.test_utils import (
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
    terminate_and_kill_process_tree,
)

register_cuda_ci(est_time=214, stage="extra-b", runner_config="4-gpu-b200")

KIMI_LINEAR_MODEL = "moonshotai/Kimi-Linear-48B-A3B-Instruct"
DCP_SIZE = 4
PAGE_SIZE = 64
WIDENED_PAGE = PAGE_SIZE * DCP_SIZE
MAX_MAMBA_CACHE_SIZE = 256
# Bound the host pools directly. Sizing them off the device pool would need
# --max-total-tokens, which triggers out-of-range KV writes under DCP.
HICACHE_SIZE_GB = 10


class TestUnifiedKimiLinearDcpHiCache(UnifiedRadixTreeTestMixin, CustomTestCase):
    """Kimi Linear + DCP4 + HiCache L2 + UnifiedRadixCache."""

    kl_threshold = 0.01
    gsm8k_threshold = 0.85
    prefill_cache_assert = staticmethod(
        make_mamba_prefill_assert(chunk_size=WIDENED_PAGE)
    )
    decode_cache_assert = staticmethod(
        make_mamba_decode_assert(track_interval=WIDENED_PAGE)
    )

    @classmethod
    def setUpClass(cls):
        cls.model = KIMI_LINEAR_MODEL
        cls.base_url = DEFAULT_URL_FOR_TEST
        cls.process = popen_launch_server(
            cls.model,
            cls.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH * 3,
            other_args=[
                "--trust-remote-code",
                "--tp-size",
                "4",
                "--dcp-size",
                str(DCP_SIZE),
                "--page-size",
                str(PAGE_SIZE),
                "--attention-backend",
                "cutedsl_mla",
                "--dcp-comm-backend",
                "a2a",
                "--dcp-replicate-q-proj",
                "--dtype",
                "bfloat16",
                "--random-seed",
                "0",
                "--cuda-graph-max-bs-decode",
                "64",
                "--cuda-graph-backend-prefill",
                "disabled",
                "--mem-fraction-static",
                "0.80",
                "--enable-hierarchical-cache",
                "--hicache-size",
                str(HICACHE_SIZE_GB),
                "--hicache-write-policy",
                "write_through",
                "--max-running-requests",
                "64",
                "--max-mamba-cache-size",
                str(MAX_MAMBA_CACHE_SIZE),
                "--enable-metrics",
            ],
            env={
                "SGLANG_ENABLE_RANK_CONSENSUS_CHECKER": "1",
            },
        )
        cls.input_ids = get_input_ids(cls.model, num_samples=18, trust_remote_code=True)

    @classmethod
    def tearDownClass(cls):
        terminate_and_kill_process_tree(cls.process, wait_timeout=60)


class TestGqaDcpHiCache(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        model = os.environ.get("GQA_DCP_HICACHE_MODEL", "Qwen/Qwen2.5-1.5B-Instruct")
        cls.base_url = os.environ.get("GQA_DCP_HICACHE_BASE_URL", DEFAULT_URL_FOR_TEST)
        dtype = os.environ.get("GQA_DCP_HICACHE_DTYPE", "bfloat16")
        kv_dtype = {"bfloat16": "bf16", "float16": "auto"}[dtype]
        if "GQA_DCP_HICACHE_BASE_URL" not in os.environ:
            process = popen_launch_server(
                model,
                cls.base_url,
                timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
                other_args=shlex.split(
                    "--tp-size 4 --dcp-size 2 --attention-backend triton "
                    f"--dtype {dtype} --kv-cache-dtype {kv_dtype} --page-size 64 "
                    "--max-total-tokens 4096 --max-running-requests 4 "
                    "--chunked-prefill-size 1024 --enable-hierarchical-cache "
                    "--hicache-size 2 --hicache-write-policy write_through "
                    "--hicache-mem-layout page_first_direct --hicache-io-backend direct "
                    "--enable-cache-report --cuda-graph-backend-prefill disabled "
                    "--cuda-graph-max-bs-decode 4"
                ),
            )
            cls.addClassCleanup(terminate_and_kill_process_tree, process)
        info = requests.get(cls.base_url + "/get_server_info", timeout=30).json()
        assert info["dcp_size"] > 1 and info["enable_hierarchical_cache"]
        cls.page_size = info["page_size"] * info["dcp_size"]
        cls.capacity = info["internal_states"][0]["memory_usage"]["token_capacity"]

    def generate(self, ids, output=32):
        return _generate(self.base_url, ids, output, return_logprob=True)

    def test_host_restore(self):
        def prompt(seed):
            return [seed + i % 97 for i in range(1025)]

        _flush_cache(self.base_url)
        ids = prompt(1000)
        self.assertEqual(self.generate(ids)["meta_info"]["cached_tokens"], 0)
        warm = self.generate(ids)
        self.assertGreater(warm["meta_info"]["cached_tokens_details"]["device"], 0)
        # Distinct prefixes overwrite L1 while the original survives in L2.
        for i in range(math.ceil(self.capacity / len(ids)) + 1):
            self.generate(prompt(2000 + i), output=1)
        restored = self.generate(ids)
        hits = restored["meta_info"]["cached_tokens_details"]
        self.assertEqual(hits["device"], 0)
        self.assertGreaterEqual(
            hits["host"], len(ids) // self.page_size * self.page_size
        )
        self.assertEqual(warm["output_ids"], restored["output_ids"])
        self.assertEqual(
            _extract_output_logprobs(warm), _extract_output_logprobs(restored)
        )
        _flush_cache(self.base_url)
        self.assertEqual(self.generate(ids, output=1)["meta_info"]["cached_tokens"], 0)


if __name__ == "__main__":
    unittest.main()
