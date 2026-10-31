"""External KV reuse on PD prefill with an independent LMCache daemon.

Requires LMCache, a PD transfer backend and two GPUs (four with TP=2).
Run with an activated environment so the daemon and servers use the same Python.
"""

import os
import socket
import subprocess
import sys
import tempfile
import time
import unittest
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import requests
import torch

from sglang.test.test_utils import (
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    popen_launch_pd_server,
    terminate_and_kill_process_tree,
)
from sglang.utils import wait_for_http_ready


class TestDisaggregationLMCache(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tp_size = int(os.environ.get("SGLANG_TEST_LMCACHE_TP_SIZE", "1"))
        if torch.cuda.device_count() < 2 * cls.tp_size:
            raise unittest.SkipTest(f"Requires {2 * cls.tp_size} GPUs")
        cls.model = os.environ.get("SGLANG_TEST_LMCACHE_MODEL", "Qwen/Qwen3-0.6B")
        cls.transfer_mode = os.environ.get("LMCACHE_MP_TRANSFER_MODE", "engine_driven")
        cls.use_shm = os.environ.get("SGLANG_TEST_LMCACHE_SHM") == "1"
        if cls.use_shm and cls.transfer_mode != "engine_driven":
            raise ValueError("SGLANG_TEST_LMCACHE_SHM requires engine_driven transfers")
        base_port = int(os.environ.get("SGLANG_TEST_LMCACHE_BASE_PORT", "31500"))
        cls.ports = dict(
            prefill=base_port + 1,
            decode=base_port + 2,
            bootstrap=base_port + 3,
            cache=base_port + 4,
            cache_http=base_port + 7,
        )
        cls.urls = {
            role: f"http://127.0.0.1:{cls.ports[role]}"
            for role in ("prefill", "decode")
        }
        cls.nccl_ports = [base_port + 5, base_port + 6]
        cls.processes = {}
        directory = tempfile.TemporaryDirectory(prefix="sglang-lmcache-pd-")
        cls.addClassCleanup(directory.cleanup)
        cls.addClassCleanup(cls._stop_all)
        cls.config_path = Path(directory.name) / "mp.yaml"
        cls.config_path.write_text(
            f"mp_host: 127.0.0.1\nmp_port: {cls.ports['cache']}\n"
            f"extra_config:\n  lmcache.mp.mp_transfer_mode: {cls.transfer_mode}\n"
        )
        for port in [*cls.ports.values(), *cls.nccl_ports]:
            with socket.socket() as probe:
                probe.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                probe.bind(("127.0.0.1", port))
        # Each invocation owns a fresh daemon, so cold misses and persistence
        # across engine restarts cannot be satisfied by a previous test run.
        cache_command = [
            sys.executable,
            "-m",
            "lmcache.cli.main",
            "server",
            "--host",
            "127.0.0.1",
            "--port",
            str(cls.ports["cache"]),
            "--l1-size-gb",
            "1",
            "--http-port",
            str(cls.ports["cache_http"]),
            "--chunk-size",
            "256",
            "--eviction-policy",
            "LRU",
            "--supported-transfer-mode",
            cls.transfer_mode,
        ]
        if cls.use_shm:
            cache_command += [
                "--no-l1-use-lazy",
                "--shm-name",
                f"sglang_kv_test_{uuid.uuid4().hex}",
            ]
        cls.processes["cache"] = subprocess.Popen(cache_command)
        deadline = time.monotonic() + 60
        while time.monotonic() < deadline:
            if cls.processes["cache"].poll() is not None:
                raise RuntimeError("LMCache daemon exited during startup")
            try:
                with socket.create_connection(("127.0.0.1", cls.ports["cache"]), 1):
                    break
            except OSError:
                time.sleep(0.1)
        else:
            raise TimeoutError("LMCache daemon did not start")

    @classmethod
    def _stop(cls, name):
        process = cls.processes.pop(name, None)
        if process is not None and process.poll() is None:
            terminate_and_kill_process_tree(process)

    @classmethod
    def _stop_all(cls):
        cls._stop("decode")
        cls._stop("prefill")
        cls._stop("cache")

    @classmethod
    def _start_pair(cls, enable_lmcache):
        common = [
            "--tp",
            str(cls.tp_size),
            "--attention-backend",
            "triton",
            "--enable-deterministic-inference",
            "--mem-fraction-static",
            "0.1",
            "--max-total-tokens",
            "8192",
            "--context-length",
            "4096",
            "--disable-cuda-graph",
            "--disaggregation-transfer-backend",
            os.environ.get("SGLANG_TEST_PD_DISAGG_BACKEND", "mooncake"),
            "--disaggregation-bootstrap-port",
            str(cls.ports["bootstrap"]),
        ]
        if os.environ.get("SGLANG_TEST_LMCACHE_DISABLE_OVERLAP") == "1":
            common += ["--disable-overlap-schedule"]
        for index, role in enumerate(("prefill", "decode")):
            args = common + [
                "--disaggregation-mode",
                role,
                "--base-gpu-id",
                str(index * cls.tp_size),
                "--nccl-port",
                str(cls.nccl_ports[index]),
            ]
            if enable_lmcache and role == "prefill":
                args += [
                    "--enable-lmcache",
                    "--lmcache-config-file",
                    str(cls.config_path),
                ]
            cls.processes[role] = popen_launch_pd_server(
                cls.model,
                cls.urls[role],
                timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
                other_args=args,
                env={"SGLANG_DISAGG_PREFILL_EARLY_SEND_CACHED_PREFIX": "1"},
            )
            wait_for_http_ready(
                cls.urls[role] + "/health",
                timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
                process=cls.processes[role],
            )

    def _generate(self, cache_salt):
        payload = {
            "text": (
                "Paris is the capital of France. Berlin is the capital of Germany. "
                * 80
            )
            + "\nQuestion: What is the capital of Germany?\nAnswer:",
            "sampling_params": {
                "temperature": 0,
                "max_new_tokens": 16,
                "ignore_eos": True,
            },
            "cache_salt": cache_salt,
            "return_logprob": True,
            "bootstrap_host": "127.0.0.1",
            "bootstrap_port": self.ports["bootstrap"],
            "bootstrap_room": uuid.uuid4().int % (2**63 - 1),
        }

        def post(role):
            body = dict(payload)
            if role == "prefill":
                body["sampling_params"] = dict(
                    payload["sampling_params"], max_new_tokens=1
                )
            response = requests.post(
                self.urls[role] + "/generate", json=body, timeout=180
            )
            response.raise_for_status()
            result = response.json()
            self.assertNotEqual(result["meta_info"]["finish_reason"]["type"], "abort")
            return result

        with ThreadPoolExecutor(max_workers=2) as pool:
            prefill = pool.submit(post, "prefill")
            decode = pool.submit(post, "decode")
            return prefill.result(), decode.result()

    def _flush_local(self):
        for url in self.urls.values():
            requests.post(
                url + "/flush_cache", params={"timeout": 30}, timeout=60
            ).raise_for_status()

    def test_external_hit_and_engine_restart(self):
        self._start_pair(enable_lmcache=False)
        _, baseline = self._generate("shared")
        reference = baseline["output_ids"]
        self.assertEqual(len(reference), 16)
        self._stop("decode")
        self._stop("prefill")
        self._start_pair(enable_lmcache=True)
        cold_prefill, cold_decode = self._generate("shared")
        self.assertEqual(cold_decode["output_ids"], reference)
        self.assertEqual(
            (cold_prefill["meta_info"]["cached_tokens_details"] or {}).get(
                "storage", 0
            ),
            0,
        )
        self._flush_local()
        warm_prefill, warm_decode = self._generate("shared")
        self._assert_external_hit(warm_prefill, warm_decode, reference)
        self._flush_local()
        isolated_prefill, isolated_decode = self._generate("isolated")
        self.assertEqual(isolated_decode["output_ids"], reference)
        self.assertEqual(
            (isolated_prefill["meta_info"]["cached_tokens_details"] or {}).get(
                "storage", 0
            ),
            0,
        )
        self._stop("decode")
        self._stop("prefill")
        self._start_pair(enable_lmcache=True)
        restarted_prefill, restarted_decode = self._generate("shared")
        self._assert_external_hit(restarted_prefill, restarted_decode, reference)

    def _assert_external_hit(self, prefill, decode, reference):
        details = prefill["meta_info"]["cached_tokens_details"]
        self.assertGreater(details["storage"], 0)
        self.assertEqual(details["device"], 0)
        self.assertEqual(decode["output_ids"], reference)
        self.assertEqual(
            decode["meta_info"]["cached_tokens_details"]["storage"], details["storage"]
        )


if __name__ == "__main__":
    unittest.main()
