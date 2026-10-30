"""Manual GPU check for LMCache lookup bypass and partial-hit admission.

Install LMCache in the active SGLang environment. Set
SGLANG_TEST_LMCACHE_TP_SIZE=2 to exercise two ranks.
"""

import os
import socket
import subprocess
import sys
import tempfile
import time
import unittest
import uuid
from pathlib import Path

import requests
import torch
from transformers import AutoTokenizer

from sglang.test.test_utils import (
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    terminate_and_kill_process_tree,
)
from sglang.utils import wait_for_http_ready


class TestLMCacheLoadAdmission(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tp_size = int(os.environ.get("SGLANG_TEST_LMCACHE_TP_SIZE", "1"))
        if torch.cuda.device_count() < cls.tp_size:
            raise unittest.SkipTest(f"Requires {cls.tp_size} GPUs")
        cls.use_shm = os.environ.get("SGLANG_TEST_LMCACHE_SHM") == "1"
        cls.model = os.environ.get("SGLANG_TEST_LMCACHE_MODEL", "Qwen/Qwen3-0.6B")
        port = int(os.environ.get("SGLANG_TEST_LMCACHE_BASE_PORT", "31700"))
        cls.engine_port = port + 1
        cls.url = f"http://127.0.0.1:{cls.engine_port}"
        cls.cache_port, cls.http_port, cls.nccl_port = port + 2, port + 3, port + 4
        cls.processes = {}
        directory = tempfile.TemporaryDirectory(prefix="sglang-lmcache-admission-")
        cls.addClassCleanup(directory.cleanup)
        cls.addClassCleanup(cls._stop_all)
        cls.config = Path(directory.name) / "mp.yaml"
        cls.config.write_text(
            f"mp_host: 127.0.0.1\nmp_port: {cls.cache_port}\n"
            "extra_config:\n  lmcache.mp.mp_transfer_mode: engine_driven\n"
        )
        for candidate in range(port + 1, port + 5):
            with socket.socket() as probe:
                probe.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                probe.bind(("127.0.0.1", candidate))

    @classmethod
    def _stop(cls, name):
        process = cls.processes.pop(name, None)
        if process is not None and process.poll() is None:
            terminate_and_kill_process_tree(process)

    @classmethod
    def _stop_all(cls):
        cls._stop("engine")
        cls._stop("cache")

    @classmethod
    def _start_cache(cls):
        command = [
            sys.executable,
            "-m",
            "lmcache.cli.main",
            "server",
            "--host",
            "127.0.0.1",
            "--port",
            str(cls.cache_port),
            "--http-port",
            str(cls.http_port),
            "--l1-size-gb",
            "1",
            "--chunk-size",
            "256",
            "--eviction-policy",
            "LRU",
            "--supported-transfer-mode",
            "engine_driven",
        ]
        if cls.use_shm:
            command += [
                "--no-l1-use-lazy",
                "--shm-name",
                f"sglang_kv_admission_{uuid.uuid4().hex}",
            ]
        cls.processes["cache"] = subprocess.Popen(command)
        deadline = time.monotonic() + 60
        while time.monotonic() < deadline:
            if cls.processes["cache"].poll() is not None:
                raise RuntimeError("LMCache daemon exited during startup")
            try:
                with socket.create_connection(("127.0.0.1", cls.cache_port), 1):
                    return
            except OSError:
                time.sleep(0.1)
        raise TimeoutError("LMCache daemon did not start")

    @classmethod
    def _start_engine(cls, enable_lmcache):
        args = [
            "--tp",
            str(cls.tp_size),
            "--nccl-port",
            str(cls.nccl_port),
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
        ]
        if enable_lmcache:
            args += [
                "--enable-lmcache",
                "--lmcache-config-file",
                str(cls.config),
                "--lmcache-min-retrieve-tokens",
                "512",
            ]
        cls.processes["engine"] = subprocess.Popen(
            [
                sys.executable,
                "-m",
                "sglang.launch_server",
                "--model-path",
                cls.model,
                "--host",
                "127.0.0.1",
                "--port",
                str(cls.engine_port),
                *args,
            ]
        )
        wait_for_http_ready(
            url=cls.url + "/health",
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            process=cls.processes["engine"],
        )

    def _generate(self, tokens, salt):
        response = requests.post(
            self.url + "/generate",
            json={
                "input_ids": tokens,
                "cache_salt": salt,
                "sampling_params": {
                    "temperature": 0,
                    "max_new_tokens": 8,
                    "ignore_eos": True,
                },
            },
            timeout=180,
        )
        response.raise_for_status()
        result = response.json()
        self.assertEqual(len(result["output_ids"]), 8)
        return result

    def _flush(self):
        requests.post(
            self.url + "/flush_cache", params={"timeout": 30}, timeout=60
        ).raise_for_status()

    def _assert_result(self, result, reference, storage_hits):
        self.assertEqual(result["output_ids"], reference["output_ids"])
        details = result["meta_info"]["cached_tokens_details"] or {}
        self.assertEqual(details.get("storage", 0), storage_hits)
        self.assertEqual(details.get("device", 0), 0)

    def test_bypass_partial_hit_and_reuse_after_recompute(self):
        tokenizer = AutoTokenizer.from_pretrained(self.model)
        filler = tokenizer.encode("Berlin is the capital of Germany. " * 300)
        question = tokenizer.encode(
            "\nWhat is the capital of Germany?\nAnswer:", add_special_tokens=False
        )
        long = filler[: 1025 - len(question)] + question
        short = long[:257]
        self._start_engine(enable_lmcache=False)
        short_reference = self._generate(short, "baseline")
        self._flush()
        long_reference = self._generate(long, "baseline")
        self._stop("engine")
        self._start_cache()
        self._start_engine(enable_lmcache=True)

        self._assert_result(self._generate(short, "partial"), short_reference, 0)
        self._flush()
        # The full prompt is above the gate, but only its first chunk exists.
        self._assert_result(self._generate(long, "partial"), long_reference, 0)
        self._flush()
        # Recomputed KV is still stored, and a large enough hit remains usable.
        self._assert_result(self._generate(long, "partial"), long_reference, 1024)
        self._flush()
        self._assert_result(self._generate(short, "partial"), short_reference, 0)
        self._flush()
        self._assert_result(self._generate(long, "isolated"), long_reference, 0)


if __name__ == "__main__":
    unittest.main()
