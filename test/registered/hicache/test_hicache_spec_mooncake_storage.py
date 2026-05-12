"""
E2E test for HiCache mooncake storage with EAGLE3 speculative decoding.

Verifies that after a server restart, both target and draft KV cache for a
long deterministic prompt are reloaded from the mooncake_master process and
the spec accept length does not regress.

Mooncake_master + http metadata server lifecycle is reused from
``test_hicache_storage_mooncake_backend.HiCacheStorageMooncakeBackendBaseMixin``;
this file only overrides the SGLang-side setup for EAGLE3 + spec loadback.

Usage:
    python3 -m pytest test/registered/hicache/test_hicache_spec_mooncake_storage.py -v
"""

import json
import os
import signal
import socket
import subprocess
import time
import unittest
from typing import Dict

import psutil
import requests
from test_hicache_storage_mooncake_backend import (
    HiCacheStorageMooncakeBackendBaseMixin,
)

from sglang.benchmark.utils import get_tokenizer
from sglang.srt.utils import is_hip, kill_process_tree
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import (
    DEFAULT_DRAFT_MODEL_EAGLE3,
    DEFAULT_TARGET_MODEL_EAGLE3,
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    find_available_port,
    popen_launch_server,
)
from sglang.utils import wait_for_http_ready

register_cuda_ci(est_time=240, suite="stage-b-test-2-gpu-large", nightly=True)


@unittest.skipIf(
    is_hip(), "HiCache + EAGLE3 mooncake-storage loadback e2e is CUDA-only."
)
class TestHiCacheSpecMooncakeStorage(
    HiCacheStorageMooncakeBackendBaseMixin, CustomTestCase
):
    """EAGLE3 + mooncake L3 loadback. After a server restart the long
    deterministic prompt's target+draft KV must be reloaded from
    mooncake_master and the spec accept length must not regress."""

    model = DEFAULT_TARGET_MODEL_EAGLE3
    draft_model = DEFAULT_DRAFT_MODEL_EAGLE3

    input_token_len = 1024
    max_new_tokens = 200
    first_measure_new_tokens = 128
    page_size = 64
    min_expected_accept_length = 7.0
    min_second_to_first_accept_ratio = 0.9
    # Mooncake exposes no external page count; wait this long after the
    # first prompt for HiCacheController's backup queue to drain into the
    # mooncake_master before restart.
    mooncake_backup_drain_seconds = 15
    mooncake_store_port_base = 50052
    mooncake_store_http_port_base = 8081

    # The inherited test from HiCacheStorageBaseMixin assumes a basic-storage
    # setup; it is already covered by test_hicache_storage_mooncake_backend.py.
    @unittest.skip("Covered by test_hicache_storage_mooncake_backend.py")
    def test_basic_backup_and_prefetch(self):  # noqa: D401
        pass

    @classmethod
    def setUpClass(cls):
        # Bypass the parent chain's basic-storage server bootstrap; only
        # reuse the inherited mooncake_master / metadata-server lifecycle.
        cls.mooncake_master_port = find_available_port(cls.mooncake_master_port_base)
        cls.mooncake_metadata_port = find_available_port(
            cls.mooncake_metadata_port_base
        )
        cls.mooncake_store_port = find_available_port(cls.mooncake_store_port_base)
        cls.mooncake_store_http_port = find_available_port(
            cls.mooncake_store_http_port_base
        )
        cls._start_mooncake_services()
        try:
            cls._start_mooncake_store_service()
            cls._launch_spec_server()
        except Exception:
            cls._stop_mooncake_store_service()
            cls._stop_mooncake_services()
            raise

    @classmethod
    def tearDownClass(cls):
        try:
            cls._stop_spec_server()
        finally:
            cls._stop_mooncake_store_service()
            cls._stop_mooncake_services()

    # ---- server side ----

    @classmethod
    def _start_mooncake_store_service(cls):
        print(
            f"Starting Mooncake store service on rpc port {cls.mooncake_store_port}, "
            f"http port {cls.mooncake_store_http_port}..."
        )
        cls.store_service_process = subprocess.Popen(
            [
                "mooncake_client",
                "--host=127.0.0.1",
                f"--port={cls.mooncake_store_port}",
                f"--master_server_address=127.0.0.1:{cls.mooncake_master_port}",
                f"--metadata_server=http://127.0.0.1:{cls.mooncake_metadata_port}/metadata",
                "--protocol=tcp",
                "--device_names=",
                "--global_segment_size=4294967296",
                "--enable_http_server=true",
                f"--http_port={cls.mooncake_store_http_port}",
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            preexec_fn=os.setsid,
        )
        cls._wait_for_mooncake_store_service_ready()

    @classmethod
    def _wait_for_mooncake_store_service_ready(cls, timeout: int = 90):
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            process = getattr(cls, "store_service_process", None)
            if process is not None and process.poll() is not None:
                raise RuntimeError(
                    f"Mooncake store service exited with code {process.returncode}"
                )
            try:
                with socket.create_connection(
                    ("127.0.0.1", cls.mooncake_store_port), timeout=2
                ):
                    pass
                print("Mooncake store service is ready")
                return
            except OSError:
                pass
            time.sleep(1)
        raise TimeoutError("Timed out waiting for Mooncake store service readiness.")

    @classmethod
    def _stop_mooncake_store_service(cls):
        process = getattr(cls, "store_service_process", None)
        if process is None:
            return
        print("Stopping Mooncake store service...")
        try:
            os.killpg(os.getpgid(process.pid), signal.SIGTERM)
            process.wait(timeout=15)
            print("Mooncake store service stopped")
        except (ProcessLookupError, subprocess.TimeoutExpired, OSError):
            try:
                os.killpg(os.getpgid(process.pid), signal.SIGKILL)
                process.wait(timeout=5)
            except (ProcessLookupError, subprocess.TimeoutExpired, OSError) as e:
                print(f"Warning: Could not stop Mooncake store service: {e}")
        finally:
            cls.store_service_process = None

    @classmethod
    def _launch_spec_server(cls):
        default_port = int(DEFAULT_URL_FOR_TEST.rsplit(":", 1)[1])
        cls.base_url = f"http://127.0.0.1:{find_available_port(default_port)}"
        cls.tokenizer = get_tokenizer(cls.model)
        cls.prompt_input_ids = cls._build_long_repetitive_prompt_ids(
            cls.tokenizer, cls.input_token_len
        )
        extra_config = {
            "hicache_storage_pass_prefix_keys": True,
        }
        cls.other_args = [
            "--enable-hierarchical-cache",
            "--enable-cache-report",
            "--mem-fraction-static",
            "0.3",
            "--hicache-ratio",
            "1.5",
            "--disable-cuda-graph",
            "--page-size",
            str(cls.page_size),
            "--hicache-storage-backend",
            "mooncake",
            "--hicache-storage-prefetch-policy",
            "wait_complete",
            "--hicache-storage-backend-extra-config",
            json.dumps(extra_config),
            "--speculative-algorithm",
            "EAGLE3",
            "--speculative-draft-model-path",
            cls.draft_model,
            "--speculative-num-steps",
            "7",
            "--speculative-eagle-topk",
            "1",
            "--speculative-num-draft-tokens",
            "8",
            "--dtype",
            "float16",
        ]
        cls.env = {
            **os.environ,
            "SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN": "1",
            "MOONCAKE_MASTER": f"127.0.0.1:{cls.mooncake_master_port}",
            "MOONCAKE_PROTOCOL": "tcp",
            "MC_MS_AUTO_DISC": "0",
            "MOONCAKE_DEVICE": "",
            "MOONCAKE_TE_META_DATA_SERVER": (
                f"http://127.0.0.1:{cls.mooncake_metadata_port}/metadata"
            ),
            # Keep storage capacity in the external mooncake_client so cached
            # pages survive SGLang server restart.
            "MOONCAKE_GLOBAL_SEGMENT_SIZE": "0",
        }
        cls.process = popen_launch_server(
            cls.model,
            cls.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=cls.other_args,
            env=cls.env,
        )
        wait_for_http_ready(
            url=f"{cls.base_url}/health",
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            process=cls.process,
        )

    @classmethod
    def _restart_spec_server(cls):
        cls._stop_spec_server()
        cls._launch_spec_server()

    @classmethod
    def _stop_spec_server(cls):
        if getattr(cls, "process", None) is None:
            return

        process = cls.process
        try:
            root = psutil.Process(process.pid)
            watched_procs = [root] + root.children(recursive=True)
        except psutil.NoSuchProcess:
            watched_procs = []

        try:
            process.terminate()
            process.wait(timeout=120)
            return
        except subprocess.TimeoutExpired:
            pass

        try:
            kill_process_tree(process.pid, wait_timeout=60)
        except RuntimeError:
            non_zombie_procs = []
            for proc in watched_procs:
                try:
                    if proc.is_running() and proc.status() != psutil.STATUS_ZOMBIE:
                        non_zombie_procs.append(proc)
                except psutil.NoSuchProcess:
                    pass
            if non_zombie_procs:
                raise
        finally:
            cls.process = None

    # ---- prompt / IO helpers ----

    @classmethod
    def _build_long_repetitive_prompt_ids(cls, tokenizer, target_len: int):
        bos_ids = (
            [tokenizer.bos_token_id]
            if getattr(tokenizer, "bos_token_id", None) is not None
            else []
        )
        suffix_ids = tokenizer.encode(
            "\n\nContinue the sequence with only the word apple separated by spaces.\n"
            "Answer: apple apple apple apple",
            add_special_tokens=False,
        )
        repeat_ids = tokenizer.encode(" apple", add_special_tokens=False)
        if not repeat_ids:
            raise ValueError(
                "Tokenizer produced no ids for the repetitive prompt seed."
            )
        if len(bos_ids) + len(suffix_ids) >= target_len:
            raise ValueError(
                f"Prompt suffix too long: {len(bos_ids)=}, {len(suffix_ids)=}, {target_len=}"
            )
        prefix_len = target_len - len(bos_ids) - len(suffix_ids)
        repeats = (prefix_len + len(repeat_ids) - 1) // len(repeat_ids)
        prefix_ids = (repeat_ids * repeats)[:prefix_len]
        prompt_ids = bos_ids + prefix_ids + suffix_ids
        assert len(prompt_ids) == target_len
        return prompt_ids

    def _send_long_prompt(self, max_new_tokens: int = None) -> Dict:
        response = requests.post(
            f"{self.base_url}/generate",
            json={
                "input_ids": self.prompt_input_ids,
                "sampling_params": {
                    "temperature": 0,
                    "max_new_tokens": max_new_tokens or self.max_new_tokens,
                    "ignore_eos": True,
                },
            },
            timeout=900,
        )
        self.assertEqual(response.status_code, 200, f"Request failed: {response.text}")
        return response.json()

    def _get_spec_accept_length(self, response_json: Dict) -> float:
        meta = response_json.get("meta_info", {})
        self.assertIn("spec_accept_length", meta, f"Missing spec_accept_length: {meta}")
        return float(meta["spec_accept_length"])

    # ---- the test ----

    def test_mooncake_storage_loadback_keeps_spec_accept_length(self):
        first = self._send_long_prompt(max_new_tokens=self.first_measure_new_tokens)
        first_accept = self._get_spec_accept_length(first)
        self.assertGreaterEqual(
            first_accept,
            self.min_expected_accept_length,
            f"First accept length too low: {first_accept=}",
        )

        print(
            f"[mooncake] draining backup queue "
            f"({self.mooncake_backup_drain_seconds}s)..."
        )
        time.sleep(self.mooncake_backup_drain_seconds)

        self._restart_spec_server()

        second = self._send_long_prompt()
        second_accept = self._get_spec_accept_length(second)
        cached_details = second.get("meta_info", {}).get("cached_tokens_details") or {}
        storage_cached = int(cached_details.get("storage", 0))

        print(
            f"{first_accept=:.3f}, {second_accept=:.3f}, "
            f"{storage_cached=}, {cached_details=}"
        )

        self.assertGreaterEqual(
            storage_cached,
            self.input_token_len - 2 * self.page_size,
            f"Expected mooncake loadback, got {cached_details=}",
        )
        self.assertEqual(
            cached_details.get("storage_backend"),
            "MooncakeStore",
            f"Expected MooncakeStore in cache report, got {cached_details=}",
        )
        self.assertGreaterEqual(
            second_accept,
            self.min_expected_accept_length,
            f"Second accept length too low: {second_accept=}",
        )
        self.assertGreaterEqual(
            second_accept,
            first_accept * self.min_second_to_first_accept_ratio,
            f"Accept length dropped after mooncake loadback: "
            f"{first_accept=:.3f}, {second_accept=:.3f}",
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
