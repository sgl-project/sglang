"""
E2E test: an L3 cache written under the unified key scheme at one attention-TP
size is read back at another.

An external mooncake_client holds the store, so objects outlive each server.
TP2 writes, TP1 reads what TP2 wrote and writes more, and a restarted TP2 reads
both. A TP2 rank owns one head-group chunk per layer range and a TP1 rank owns
two, so any hit across sizes goes through the topology-free keys.

Usage:
    python3 -m pytest test/registered/hicache/test_hicache_storage_mooncake_unified_cross_tp.py -v
"""

import json
import os
import random
import signal
import socket
import subprocess
import time
import unittest
from contextlib import contextmanager

import requests
from test_hicache_storage_mooncake_backend import (
    HiCacheStorageMooncakeBackendBaseMixin,
)
from test_hicache_storage_mooncake_unified import run_gsm8k, storage_hit_tokens

from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import (
    DEFAULT_MODEL_NAME_FOR_TEST,
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    find_available_port,
    get_gpu_count,
    popen_launch_server,
    terminate_and_kill_process_tree,
)

register_cuda_ci(est_time=480, stage="extra-a", runner_config="2-gpu-large")

PAGE_SIZE = 64
PROMPT_PAGES = 12
NUM_PROMPTS = 4
# 8 kv heads in 4-head chunks tile both a TP1 rank (2 chunks) and a TP2 rank (1).
HEAD_PARTITION = 4
LAYER_PARTITION = 8
# Measured on H200: TP1 vs TP2 first-token logprobs differ by at most 0.06;
# swapping head-group or layer-range chunks moves them by 2.0 to 4.5.
LOGPROB_TOLERANCE = 0.5


def _prompts(seed):
    rng = random.Random(seed)
    # One token past a page boundary: the cached prefix is exactly PROMPT_PAGES.
    return [
        [rng.randint(1000, 30000) for _ in range(PROMPT_PAGES * PAGE_SIZE + 1)]
        for _ in range(NUM_PROMPTS)
    ]


@unittest.skipUnless(get_gpu_count() >= 2, "Requires 2 GPUs for the TP2 servers")
class TestMooncakeUnifiedCrossTopology(
    HiCacheStorageMooncakeBackendBaseMixin, CustomTestCase
):
    model = DEFAULT_MODEL_NAME_FOR_TEST
    base_url = DEFAULT_URL_FOR_TEST
    mooncake_store_port_base = 50052
    store_service_process = None

    # The mixin's single-server test needs a server from setUpClass; this class
    # starts its own servers per phase, so drop it from collection.
    test_basic_backup_and_prefetch = None

    @classmethod
    def setUpClass(cls):
        # Reuse only the master and metadata-server lifecycle; servers start
        # and stop inside the test.
        cls.mooncake_master_port = find_available_port(cls.mooncake_master_port_base)
        cls.mooncake_metadata_port = find_available_port(
            cls.mooncake_metadata_port_base
        )
        cls.mooncake_store_port = find_available_port(cls.mooncake_store_port_base)
        cls._start_mooncake_services()
        try:
            cls._start_mooncake_store_service()
        except Exception:
            cls._stop_mooncake_store_service()
            cls._stop_mooncake_services()
            raise

    @classmethod
    def tearDownClass(cls):
        cls._stop_mooncake_store_service()
        cls._stop_mooncake_services()

    @classmethod
    def _start_mooncake_store_service(cls):
        # Store capacity lives here, so cached pages survive server restarts.
        cls.store_service_process = subprocess.Popen(
            [
                "mooncake_client",
                "--host=127.0.0.1",
                f"--port={cls.mooncake_store_port}",
                f"--master_server_address=127.0.0.1:{cls.mooncake_master_port}",
                f"--metadata_server=http://127.0.0.1:{cls.mooncake_metadata_port}/metadata",
                "--protocol=tcp",
                "--device_names=",
                "--global_segment_size=17179869184",
            ],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            preexec_fn=os.setsid,
        )
        deadline = time.monotonic() + 90
        while time.monotonic() < deadline:
            if cls.store_service_process.poll() is not None:
                raise RuntimeError("Mooncake store service exited during startup.")
            try:
                with socket.create_connection(
                    ("127.0.0.1", cls.mooncake_store_port), timeout=2
                ):
                    return
            except OSError:
                time.sleep(1)
        raise TimeoutError("Timed out waiting for the Mooncake store service.")

    @classmethod
    def _stop_mooncake_store_service(cls):
        process = cls.store_service_process
        if process is None:
            return
        try:
            os.killpg(os.getpgid(process.pid), signal.SIGKILL)
            process.wait(timeout=10)
        except (ProcessLookupError, subprocess.TimeoutExpired, OSError) as e:
            print(f"Warning: could not stop the Mooncake store service: {e}")
        cls.store_service_process = None

    @contextmanager
    def _server(self, tp_size):
        env = {
            **os.environ,
            "MOONCAKE_MASTER": f"127.0.0.1:{self.mooncake_master_port}",
            "MOONCAKE_PROTOCOL": "tcp",
            "MC_MS_AUTO_DISC": "0",
            "MOONCAKE_DEVICE": "",
            "MOONCAKE_TE_META_DATA_SERVER": (
                f"http://127.0.0.1:{self.mooncake_metadata_port}/metadata"
            ),
            "MOONCAKE_GLOBAL_SEGMENT_SIZE": "0",
        }
        process = popen_launch_server(
            self.model,
            self.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=[
                "--tp-size",
                str(tp_size),
                "--mem-fraction-static",
                "0.6",
                # Bounds the pinned host pool, which is 2x the device pool.
                "--max-total-tokens",
                "131072",
                "--page-size",
                str(PAGE_SIZE),
                "--enable-hierarchical-cache",
                "--hicache-ratio",
                "2",
                "--enable-cache-report",
                "--enable-metrics",
                "--hicache-storage-prefetch-policy",
                "wait_complete",
                # A single-shot GSM8K prompt spans 1-3 pages; prefetch them from L3.
                "--hicache-storage-backend-extra-config",
                json.dumps({"prefetch_threshold": PAGE_SIZE}),
                "--hicache-storage-backend",
                "mooncake",
                "--hicache-storage-key-scheme",
                "unified",
                "--hicache-mem-layout",
                "page_unified",
                "--hicache-io-backend",
                "kernel",
                "--hicache-storage-head-partition",
                str(HEAD_PARTITION),
                "--hicache-storage-layer-partition",
                str(LAYER_PARTITION),
            ],
            env=env,
        )
        try:
            yield
        finally:
            terminate_and_kill_process_tree(process)

    def _generate(self, input_ids):
        response = requests.post(
            f"{self.base_url}/generate",
            json={
                "input_ids": input_ids,
                "sampling_params": {
                    "temperature": 0,
                    "max_new_tokens": 16,
                    "ignore_eos": True,
                },
                "return_logprob": True,
            },
            timeout=120,
        )
        self.assertEqual(response.status_code, 200, response.text)
        return response.json()

    def _device_hits(self, prompts):
        """Greedy outputs whose prefix KV this server computed and still holds."""
        hits = []
        for input_ids in prompts:
            self._generate(input_ids)
            hits.append(self._generate(input_ids))
        return hits

    def _flush_to_storage(self):
        """Drive pending backups, then drop L1 and L2 so the next hit is L3."""
        self._generate([5] * 3)
        time.sleep(2)
        response = requests.post(
            f"{self.base_url}/flush_cache", params={"timeout": 30}, timeout=40
        )
        response.raise_for_status()

    def _storage_hit(self, input_ids):
        output = self._generate(input_ids)
        details = output["meta_info"].get("cached_tokens_details") or {}
        self.assertEqual(details.get("storage"), PROMPT_PAGES * PAGE_SIZE, details)
        return output

    def _assert_close(self, output, reference):
        """Same first-token distribution up to TP numerics.

        Greedy tokens can flip on near-ties across TP sizes, so compare the
        first position's top-1 logprob, which is continuous in the logits.
        """
        got = output["meta_info"]["output_token_logprobs"][0][0]
        want = reference["meta_info"]["output_token_logprobs"][0][0]
        self.assertLess(abs(got - want), LOGPROB_TOLERANCE, (got, want))

    def _gsm8k(self):
        # 100 questions keep both servers' GSM8K pages inside the 16 GiB store,
        # so the first writer's prompts are not evicted before the last phase.
        return run_gsm8k(base_url=self.base_url, model=self.model, num_examples=100)

    def test_cache_moves_between_tp2_and_tp1(self):
        prompts_a, prompts_b = _prompts(seed=1), _prompts(seed=2)

        with self._server(tp_size=2):
            tp2_a = self._device_hits(prompts_a)
            tp2_score = self._gsm8k()
            self._flush_to_storage()
            for input_ids in prompts_a:
                self._storage_hit(input_ids)

        with self._server(tp_size=1):
            for input_ids, reference in zip(prompts_a, tp2_a):
                self._assert_close(self._storage_hit(input_ids), reference)
            # Ahead of the GSM8K backlog: a flush drops backups still queued.
            tp1_b = self._device_hits(prompts_b)
            hits_before = storage_hit_tokens(self.base_url)
            tp1_score = self._gsm8k()
            gsm8k_hit_tokens = storage_hit_tokens(self.base_url) - hits_before
            self._flush_to_storage()
            for input_ids in prompts_b:
                self._storage_hit(input_ids)

        with self._server(tp_size=2):
            # Same topology as the first writer: the reload is bit-exact.
            for input_ids, reference in zip(prompts_a, tp2_a):
                output = self._storage_hit(input_ids)
                self.assertEqual(output["output_ids"], reference["output_ids"])
                self.assertEqual(
                    output["meta_info"]["output_token_logprobs"],
                    reference["meta_info"]["output_token_logprobs"],
                )
            for input_ids, reference in zip(prompts_b, tp1_b):
                self._assert_close(self._storage_hit(input_ids), reference)

        print(f"{tp2_score=:.3f} {tp1_score=:.3f} {gsm8k_hit_tokens=:.0f}")
        # TP1 answered GSM8K from question pages TP2 stored.
        self.assertGreaterEqual(gsm8k_hit_tokens, 50 * PAGE_SIZE)
        self.assertGreater(tp2_score, 0.6)
        self.assertGreater(tp1_score, 0.6)


if __name__ == "__main__":
    unittest.main(verbosity=2)
