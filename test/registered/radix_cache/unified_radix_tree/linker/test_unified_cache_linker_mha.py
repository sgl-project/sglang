"""Mooncake direct linker on a plain MHA/GQA pool.

The MLA sibling covers a replicated latent pool; this covers the tp-sharded MHA
layout instead: one page object holds the K rows of every layer followed by the
V rows, and storage keys keep the tp suffix because the shards differ per rank.

The checkpoint is a plain GQA model without a sliding window: a windowed config
(such as Qwen2.5) routes to the hybrid SWA stack, which the linker rejects, and
that is a separate feature rather than a bug in this layout.
"""

import os
import unittest

from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.kits.unified_radix_cache_kit import _random_suffixes
from sglang.test.kl_test_utils import (
    _extract_output_logprobs,
    _flush_cache,
    _generate,
    get_input_ids,
)
from sglang.test.mooncake_utils import MooncakeTestServices
from sglang.test.test_utils import (
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    CustomTestCase,
    find_available_port,
    popen_launch_server,
    terminate_and_kill_process_tree,
    unified_radix_tree_server_env,
)

register_cuda_ci(
    est_time=180,
    stage="extra-b",
    # Direct GPU transfers require an RDMA-capable runner, even for TP=1.
    runner_config="4-gpu-h100",
)

_MODEL_PATH = os.environ.get("SGLANG_LINKER_MHA_MODEL", "Qwen/Qwen3-0.6B")

PAGE_SIZE = 64
PREFIX_LEN = 128


class TestPlainMHAUnifiedCacheLinker(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = _MODEL_PATH
        cls.base_url = f"http://127.0.0.1:{find_available_port(30000)}"
        cls.mooncake = MooncakeTestServices()
        cls.addClassCleanup(cls.mooncake.stop)
        cls.mooncake.start()
        cls.process = popen_launch_server(
            cls.model,
            cls.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=[
                "--trust-remote-code",
                # Deterministic inference only keeps the radix cache for
                # ascend/fa3/fa4/triton; the default flashinfer backend would
                # disable it, and the linker requires the radix cache.
                "--attention-backend",
                "triton",
                "--page-size",
                str(PAGE_SIZE),
                "--enable-deterministic-inference",
                # The restore logic is what is under test, not graph replay.
                "--disable-cuda-graph",
                "--mem-fraction-static",
                "0.6",
                # Size the pool from the prompt, not the card: a half-billion
                # parameter checkpoint on a large GPU would otherwise reserve
                # hundreds of GB and OOM while capturing decode CUDA graphs.
                "--max-total-tokens",
                "8192",
                "--max-running-requests",
                "1",
                "--enable-cache-report",
                "--enable-unified-cache-external-linker",
            ],
            env=unified_radix_tree_server_env("python", **cls.mooncake.server_env()),
        )
        cls.addClassCleanup(
            terminate_and_kill_process_tree, cls.process, wait_timeout=60
        )
        cls.input_ids = get_input_ids(
            tokenizer_path=cls.model, num_samples=4, trust_remote_code=True
        )

    def test_host_hit_matches_device_hit(self):
        prompts = [list(ids[: PREFIX_LEN + 1]) for ids in self.input_ids]
        for ids in prompts:
            _generate(self.base_url, [ids], 1)
        requests = [
            prompt + suffix
            for prompt, suffix in zip(
                prompts, _random_suffixes(len(prompts), PAGE_SIZE - 1, seed=100)
            )
        ]

        def run():
            return [
                _generate(self.base_url, [ids], 32, return_logprob=True)[0]
                for ids in requests
            ]

        device_results = run()
        _flush_cache(self.base_url)
        host_results = run()

        for device, host in zip(device_results, host_results):
            for result, host_tokens in ((device, 0), (host, PREFIX_LEN)):
                meta = result["meta_info"]
                details = meta.get("cached_tokens_details") or {}
                self.assertEqual(meta["cached_tokens"], PREFIX_LEN)
                self.assertEqual(int(details.get("host", 0)), host_tokens)
            self.assertEqual(host["output_ids"], device["output_ids"])
            self.assertEqual(
                _extract_output_logprobs(host), _extract_output_logprobs(device)
            )


if __name__ == "__main__":
    unittest.main()
