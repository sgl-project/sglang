"""
E2E tests for HiCache buffer_only host memory mode.

Host RAM is a transient staging buffer between GPU and the L3 storage
backend, not an L2 cache tier. These tests drive the two serving patterns
buffer mode exists for — multi-turn conversations and long-context prefix
sharing — across the cache hierarchy:

- tier-1 (device) reuse: repeat queries without flushing must hit the radix
  tree directly, with no storage traffic;
- through-storage reuse: after a device flush, prefixes must come back from
  L3 via host staging, with greedy outputs identical to the warm run;
- minimal redundant writes: re-inserting stored prefixes must not re-write
  the storage backend (the local existence cache absorbs the re-hit);
- bounded host usage: staging returns to ~zero when idle.

buffer_only is FULL/SWA-only (Mamba has no state-handoff channel on the
admission-time load-back path); init_hicache rejects Mamba trees.

Model selection: CI uses the SWA default below; local runs on machines
without HF access can override via
    SGLANG_TEST_HICACHE_SWA_MODEL=<path>      (SWA hybrid class)
The override must be a hybrid-SWA model that selects the unified radix
tree (see the note at DEFAULT_SWA_MODEL).

Usage:
    python3 -m pytest test/registered/hicache/test_hicache_buffer_only.py -v
"""

import json
import os
import random
import tempfile
import time
import unittest
from typing import Dict, List, Optional
from urllib.parse import urlparse

import requests

from sglang.benchmark.utils import get_tokenizer
from sglang.srt.utils import kill_process_tree
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import (
    CustomTestCase,
    DEFAULT_SMALL_MODEL_NAME_FOR_TEST,
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    popen_launch_server,
)
from sglang.utils import wait_for_http_ready

register_cuda_ci(est_time=240, stage="base-b", runner_config="1-gpu-small")

DEFAULT_SWA_MODEL = "google/gemma-4-E2B-it"


class BufferOnlyBaseMixin:
    """Server fixture + helpers for buffer_only e2e tests."""

    model_env_var = "SGLANG_TEST_HICACHE_MODEL"
    default_model = DEFAULT_SMALL_MODEL_NAME_FOR_TEST
    tp_size = 1

    @classmethod
    def setUpClass(cls):
        cls.temp_dir = tempfile.mkdtemp()
        cls.model = os.environ.get(cls.model_env_var, cls.default_model)
        cls.base_url = DEFAULT_URL_FOR_TEST
        parsed_url = urlparse(cls.base_url)
        cls.base_host = parsed_url.hostname
        cls.base_port = str(parsed_url.port)
        cls.tokenizer = get_tokenizer(cls.model)
        cls.process = cls._launch_server()
        wait_for_http_ready(
            url=f"{cls.base_url}/health",
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            process=cls.process,
        )

    @classmethod
    def tearDownClass(cls):
        if hasattr(cls, "process") and cls.process:
            kill_process_tree(cls.process.pid)
        import shutil

        if hasattr(cls, "temp_dir"):
            shutil.rmtree(cls.temp_dir, ignore_errors=True)

    @classmethod
    def _extra_server_args(cls) -> Dict:
        return {}

    @classmethod
    def _launch_server(cls):
        server_args = {
            "--enable-hierarchical-cache": True,
            "--hicache-host-memory-mode": "buffer_only",
            "--hicache-storage-backend": "file",
            "--hicache-write-policy": "write_through",
            "--hicache-storage-prefetch-policy": "wait_complete",
            "--hicache-storage-backend-extra-config": json.dumps(
                {"prefetch_threshold": 64}
            ),
            "--mem-fraction-static": 0.5,
            "--page-size": 64,
            "--enable-cache-report": True,
            "--enable-metrics": True,
            "--tp-size": cls.tp_size,
        }
        server_args.update(cls._extra_server_args())
        attention_backend = os.environ.get("SGLANG_TEST_ATTENTION_BACKEND")
        if attention_backend:
            server_args["--attention-backend"] = attention_backend

        final_server_args = []
        for k, v in server_args.items():
            if isinstance(v, bool):
                final_server_args.append(str(k))
            else:
                final_server_args.extend([str(k), str(v)])
        # Environment-specific escape hatch for local runs (e.g. kernel
        # backends unavailable behind a proxy).
        final_server_args.extend(
            os.environ.get("SGLANG_TEST_EXTRA_SERVER_ARGS", "").split()
        )

        env_vars = {
            **os.environ,
            "SGLANG_HICACHE_FILE_BACKEND_STORAGE_DIR": cls.temp_dir,
            "SGLANG_ENABLE_DETERMINISTIC_INFERENCE": "1",
        }
        return popen_launch_server(
            cls.model,
            cls.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=final_server_args,
            env=env_vars,
        )

    # ---------------- request helpers ----------------

    def send_request(
        self, prompt: str, max_tokens: int = 64, timeout: int = 120
    ) -> Dict:
        response = requests.post(
            f"{self.base_url}/generate",
            json={
                "text": prompt,
                "sampling_params": {
                    "temperature": 0.0,
                    "max_new_tokens": max_tokens,
                    "ignore_eos": True,
                },
            },
            timeout=timeout,
        )
        self.assertEqual(
            response.status_code,
            200,
            f"Request failed: {response.status_code} - {response.text}",
        )
        return response.json()

    def cached_tokens(self, response_json: Dict) -> int:
        return int(response_json.get("meta_info", {}).get("cached_tokens", 0))

    def flush_cache(self):
        res = requests.post(
            f"{self.base_url}/flush_cache", params={"timeout": 60}, timeout=90
        )
        res.raise_for_status()

    def gen_prompt(self, token_num: int, seed: Optional[int] = None) -> str:
        rng = random.Random(seed)
        vocab = list(self.tokenizer.get_vocab().values())
        # Avoid ids near the top of the vocab (special / unused tokens can
        # retokenize unstably); re-encode to pin the exact token count.
        candidates = [t for t in vocab if t < min(len(vocab) - 1000, 32000)]
        text = self.tokenizer.decode(rng.choices(candidates, k=token_num * 2))
        ids = self.tokenizer.encode(text)[:token_num]
        return self.tokenizer.decode(ids)

    def scrape_metric(self, name: str) -> float:
        """Sum a prometheus counter/gauge across label sets; 0 if absent."""
        res = requests.get(f"{self.base_url}/metrics", timeout=30)
        res.raise_for_status()
        total = 0.0
        found = False
        for line in res.text.splitlines():
            if line.startswith(name + "{") or line.startswith(name + " "):
                total += float(line.rsplit(" ", 1)[1])
                found = True
        return total if found else 0.0

    def storage_file_count(self) -> int:
        return len(os.listdir(self.temp_dir))

    def wait_until(self, cond, timeout: float = 60.0, msg: str = "condition"):
        deadline = time.time() + timeout
        while time.time() < deadline:
            if cond():
                return
            time.sleep(0.5)
        self.fail(f"timeout waiting for {msg}")

    def kick_scheduler_stats(self):
        """Hicache gauges refresh only inside batch stats reports: when the
        final storage ack lands after the last decode's report, an idle
        scheduler serves a frozen non-zero gauge forever. A sub-page prompt
        forces a fresh report without touching storage (it never fills a
        page, so it produces no backup and no prefetch)."""
        self.send_request("ping", max_tokens=1)

    def wait_backups_settled(self, timeout: float = 90.0):
        """Wait until write-path activity quiesces (storage dir stable and
        host staging drained)."""

        def settled():
            before = self.storage_file_count()
            time.sleep(1.0)
            self.kick_scheduler_stats()
            return (
                self.storage_file_count() == before
                and self.scrape_metric("sglang:hicache_host_used_tokens") == 0
            )

        self.wait_until(settled, timeout=timeout, msg="backups to settle")

    # ---------------- shared test bodies ----------------

    def test_multiturn_conversation_through_storage(self):
        """Multi-turn pattern: each round's history must be reusable, both
        from the device tier and — after a flush — from L3 via host staging.

        Reuse is asserted via cached_tokens, not output equality: after a
        flush the replay prefills from a different cache split point, and
        split-point numeric invariance is not a radix-cache property (in any
        mode). Byte-exactness of the KV round trip itself is pinned by the
        unit suites (torch.equal on loaded pages). Sizes are interface-
        validation scale (a few pages per turn), not a benchmark."""
        conversations: List[str] = [
            self.gen_prompt(256, seed=1000 + i) for i in range(2)
        ]

        for round_idx in range(2):
            for i in range(len(conversations)):
                question = self.gen_prompt(64, seed=7000 + 10 * round_idx + i)
                conversations[i] += question
                result = self.send_request(conversations[i], max_tokens=16)
                if round_idx > 0:
                    # tier-1: the running history is device-resident.
                    self.assertGreater(
                        self.cached_tokens(result),
                        0,
                        f"round {round_idx} conv {i}: no device reuse",
                    )
                conversations[i] += result["text"]

        self.wait_backups_settled()
        self.flush_cache()

        # Replay each full history from a cold device: nearly all of it must
        # come back from storage (page-aligned tail excepted).
        for i, conversation in enumerate(conversations):
            history_tokens = len(self.tokenizer.encode(conversation))
            result = self.send_request(conversation, max_tokens=8)
            self.assertGreaterEqual(
                self.cached_tokens(result),
                int(history_tokens * 0.75),
                f"conv {i}: history ({history_tokens} tokens) not served "
                f"from storage after flush "
                f"(cached={self.cached_tokens(result)})",
            )

    def test_long_context_prefix_sharing_through_storage(self):
        """One shared prefix, divergent continuations: the prefix must be
        stored once and reused from L3 by every continuation after a device
        flush."""
        shared_prefix = self.gen_prompt(1024, seed=42)
        continuations = [self.gen_prompt(64, seed=100 + i) for i in range(2)]

        for continuation in continuations:
            self.send_request(shared_prefix + continuation, max_tokens=16)

        self.wait_backups_settled()
        files_after_warm = self.storage_file_count()
        self.flush_cache()

        for idx, continuation in enumerate(continuations):
            result = self.send_request(shared_prefix + continuation, max_tokens=16)
            self.assertGreaterEqual(
                self.cached_tokens(result),
                768,
                f"continuation {idx}: shared prefix not served from storage "
                f"(cached={self.cached_tokens(result)})",
            )

        # Reuse must not have duplicated the shared prefix in storage.
        self.wait_backups_settled()
        self.assertLessEqual(
            self.storage_file_count(),
            files_after_warm + len(continuations) * 4,
            "storage grew far beyond the divergent tails on reuse",
        )

    def storage_snapshot(self) -> Dict[str, float]:
        """Backend ground truth: file name -> mtime. A redundant re-write of
        already-stored content keeps the file COUNT constant (same page
        hashes -> same names), so mtimes are the observable that catches it.
        """
        return {
            name: os.stat(os.path.join(self.temp_dir, name)).st_mtime
            for name in os.listdir(self.temp_dir)
        }

    def test_minimal_redundant_writes(self):
        """Re-inserting an already-stored prefix must not re-write storage:
        the existence cache absorbs the re-hit at backup admission, so the
        backend's files stay byte-for-byte untouched (names AND mtimes)."""
        prompt = self.gen_prompt(512, seed=77)
        self.send_request(prompt, max_tokens=8)
        self.wait_backups_settled()

        snapshot_before = self.storage_snapshot()
        self.assertGreater(len(snapshot_before), 0, "warm-up stored nothing")

        for _ in range(2):
            self.send_request(prompt, max_tokens=8)
        self.wait_backups_settled()

        self.assertEqual(
            self.storage_snapshot(),
            snapshot_before,
            "re-inserting a stored prefix re-wrote the storage backend",
        )

    def test_host_staging_returns_to_zero(self):
        """Host memory is staging, not a cache: when traffic stops, usage
        must drain back to zero."""
        for i in range(2):
            self.send_request(self.gen_prompt(512, seed=300 + i), max_tokens=8)

        def drained():
            self.kick_scheduler_stats()
            return self.scrape_metric("sglang:hicache_host_used_tokens") == 0

        self.wait_until(drained, timeout=90, msg="host staging to drain")


# Dense (plain HiRadix) buffer mode was removed: --hicache-host-memory-mode
# buffer_only requires the unified radix tree (hybrid SWA / Mamba models).


class TestBufferOnlySWA(BufferOnlyBaseMixin, CustomTestCase):
    """SWA hybrid (unified FULL+SWA tree) buffer mode.

    Covers the whole hierarchy for hybrid models: device-tier reuse inside
    the tests' warm phases, and the through-storage path (KV pages plus the
    SWA trailing-window pool under its own key namespace) after flushes.
    """

    model_env_var = "SGLANG_TEST_HICACHE_SWA_MODEL"
    default_model = DEFAULT_SWA_MODEL


if __name__ == "__main__":
    unittest.main(verbosity=2)
