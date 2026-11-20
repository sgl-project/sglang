"""Real Mooncake L3 reloads for KV-sharded MLA PD prefill (four GPUs).

Run with mooncake_master/mooncake_client installed and RDMA configured for PD
(or set MC_FORCE_TCP=1 for a TCP transfer test):
    python3 -m pytest test/manual/hicache/test_kv_cache_sharding_mooncake_storage.py -v -s

The storage traffic uses TCP. Its external segment survives the prefill
worker restart, independently of the Mooncake PD transfer backend.
"""

import random
import time
import unittest

import requests
import torch

from sglang.test.mooncake_utils import MooncakeTestServices
from sglang.test.server_fixtures.disaggregation_fixture import (
    PDDisaggregationServerBase,
)
from sglang.test.test_utils import (
    DEFAULT_MLA_MODEL_NAME_FOR_TEST,
    terminate_and_kill_process_tree,
)

PAGE_SIZE = 64


@unittest.skipUnless(torch.cuda.device_count() >= 4, "Requires four CUDA GPUs")
class TestKVShardMooncakeStorage(PDDisaggregationServerBase):
    model = DEFAULT_MLA_MODEL_NAME_FOR_TEST
    prefill_tp_size = decode_tp_size = 2
    decode_base_gpu_id = 2

    @classmethod
    def setUpClass(cls):
        cls.storage = MooncakeTestServices(protocol="tcp", store_segment_size=2 << 30)
        cls.addClassCleanup(cls.storage.stop)
        cls.storage.start()
        cls.extra_prefill_env = {
            **cls.storage.server_env(),
            "SGLANG_UNIFIED_RADIX_TREE_CORE_BACKEND": "python",
        }
        backend = "trtllm_mla" if torch.cuda.get_device_capability()[0] >= 10 else "fa3"
        common_args = [
            "--attention-backend",
            backend,
            "--page-size",
            str(PAGE_SIZE),
            "--context-length",
            "4096",
            "--max-total-tokens",
            "8192",
            "--mem-fraction-static",
            "0.45",
            "--cuda-graph-backend-decode",
            "disabled",
            "--cuda-graph-backend-prefill",
            "disabled",
        ]
        cls.extra_decode_args = list(common_args)
        cls.extra_prefill_args = common_args + [
            "--enable-kv-cache-sharding",
            "--chunked-prefill-size",
            "2048",
            "--max-prefill-tokens",
            "2048",
            "--enable-hierarchical-cache",
            "--hicache-ratio",
            "2",
            "--hicache-mem-layout",
            "page_first",
            "--hicache-write-policy",
            "write_through",
            "--hicache-storage-prefetch-policy",
            "wait_complete",
            "--hicache-storage-backend",
            "mooncake",
            "--enable-metrics",
        ]
        super().setUpClass()
        cls._servers_stopped = False
        cls.addClassCleanup(cls._stop_servers)
        cls.launch_all()

    @classmethod
    def _stop_servers(cls):
        if not cls._servers_stopped:
            cls._servers_stopped = True
            super().tearDownClass()

    @classmethod
    def tearDownClass(cls):
        cls._stop_servers()

    def _generate(self, text):
        response = requests.post(
            self.lb_url + "/generate",
            json={
                "text": text,
                "sampling_params": {"temperature": 0, "max_new_tokens": 16},
            },
            timeout=180,
        )
        response.raise_for_status()
        return response.json()

    def _backuped_tokens(self):
        response = requests.get(self.prefill_url + "/metrics", timeout=15)
        response.raise_for_status()
        return sum(
            float(line.rsplit(" ", 1)[1])
            for line in response.text.splitlines()
            if line.startswith("sglang:backuped_tokens_total{")
            and 'tp_rank="0"' in line
        )

    def test_storage_reload_after_prefill_restart(self):
        rng = random.Random(0)
        code = str(rng.randrange(100000, 1000000))
        words = (
            "the a quick brown fox jumps over lazy dog and then runs far away".split()
        )
        filler = " ".join(rng.choice(words) for _ in range(1800))
        target = (
            f"The secret code is {code}. Remember it.\n{filler}\n"
            "Question: what is the secret code? Answer: The secret code is"
        )
        backup_before = self._backuped_tokens()
        cold = self._generate(target)
        self.assertIn(code, cold["text"])
        self.assertEqual(cold["meta_info"]["cached_tokens"], 0)

        # TP ranks report the same logical prefix. Wait for one rank's
        # counter to advance by the whole cacheable prompt.
        prompt_tokens = cold["meta_info"]["prompt_tokens"]
        cacheable_tokens = (prompt_tokens - 1) // PAGE_SIZE * PAGE_SIZE
        self.assertGreaterEqual(cacheable_tokens, 1024)
        deadline = time.monotonic() + 120
        while self._backuped_tokens() - backup_before < cacheable_tokens:
            self.assertLess(
                time.monotonic(), deadline, "Mooncake backup did not finish"
            )
            time.sleep(1)

        # Disable the fixture's fail-fast watcher before deliberately stopping
        # its prefill worker. Decode, router, and the storage segment stay up.
        self._fail_fast_stop.set()
        terminate_and_kill_process_tree(self.process_prefill, wait_timeout=60)
        self.__class__.start_prefill()
        self.wait_server_ready(
            self.prefill_url + "/health", process=self.process_prefill
        )
        reloaded = self._generate(target)
        self.assertEqual(reloaded["output_ids"], cold["output_ids"])
        details = reloaded["meta_info"]["cached_tokens_details"]
        self.assertEqual(details["storage"], cacheable_tokens)
        self.assertEqual(details["device"], 0)
        self.assertEqual(details["host"], 0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
