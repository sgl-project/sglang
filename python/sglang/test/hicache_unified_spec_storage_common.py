"""Shared helpers for unified-key HiCache storage + speculative decoding tests.

One server at a time serves a class; a test may stop it and start another one
(another speculative setup, another topology) on the same L3 store.
"""

import hashlib
import json
import math
import os
import random
import tempfile
import time
import unittest
import uuid
from typing import Dict, List, Optional

import requests

from sglang.srt.utils import kill_process_tree
from sglang.test.test_utils import (
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    find_available_port,
    popen_launch_server,
)

PAGE_SIZE = 64


def require_model_snapshot(repo_id: str, filename: str = "config.json") -> str:
    """Path of one file of a model repository, or skip the test class."""
    from huggingface_hub import hf_hub_download

    try:
        return hf_hub_download(repo_id, filename)
    # Hub errors derive from these: offline cache miss, gated or missing repo.
    except (OSError, ValueError) as error:
        raise unittest.SkipTest(f"model_unavailable: {repo_id} ({error})") from error


class HiCacheUnifiedSpecStorageMixin:
    """Launch, request and L3 audit helpers under --hicache-storage-key-scheme unified."""

    model: str = None
    storage_backend = "file"
    expected_storage_backend = "HiCacheFile"
    # Pages of the shared prompt. A few tokens more are sent, so that both a
    # device hit and a storage hit reuse exactly this many full pages: one
    # more, or two when the tree keys on token pairs (EAGLE family).
    prompt_pages = 12
    prompt_extra_tokens = 1
    max_new_tokens = 32
    # Server arguments shared by every launch of the class.
    base_server_args: List[str] = [
        "--mem-fraction-static",
        "0.6",
        "--max-total-tokens",
        "32768",
        "--context-length",
        "8192",
        "--max-running-requests",
        "8",
    ]
    unified_server_args: List[str] = []

    process = None
    base_url = None

    @classmethod
    def setUpClass(cls):
        cls._storage_dir = tempfile.TemporaryDirectory(prefix="hicache_unified_spec_")
        cls.storage_path = cls._storage_dir.name
        cls.mooncake = None
        if cls.storage_backend == "mooncake":
            from sglang.test.mooncake_utils import MooncakeTestServices

            # The store lives in the external client, so pages survive the
            # servers that a test starts and stops.
            cls.mooncake = MooncakeTestServices(
                protocol="tcp", store_segment_size=16 << 30
            )
            cls.mooncake.start()

    @classmethod
    def tearDownClass(cls):
        cls.stop_server()
        if cls.mooncake is not None:
            cls.mooncake.stop()
        cls._storage_dir.cleanup()

    @classmethod
    def launch_server(
        cls, spec_args: List[str], extra_args: Optional[List[str]] = None
    ):
        """Start a server on the class's L3 store, replacing a running one."""
        cls.stop_server()
        default_port = int(DEFAULT_URL_FOR_TEST.rsplit(":", 1)[1])
        cls.base_url = f"http://127.0.0.1:{find_available_port(default_port)}"
        extra_config = {
            "hicache_storage_pass_prefix_keys": True,
            "prefetch_threshold": PAGE_SIZE,
        }
        args = [
            "--enable-hierarchical-cache",
            "--enable-cache-report",
            "--page-size",
            str(PAGE_SIZE),
            "--hicache-ratio",
            "2",
            "--hicache-write-policy",
            "write_through",
            "--hicache-storage-backend",
            cls.storage_backend,
            "--hicache-storage-key-scheme",
            "unified",
            "--hicache-storage-prefetch-policy",
            "wait_complete",
            "--hicache-storage-backend-extra-config",
            json.dumps(extra_config),
            *cls.base_server_args,
            *cls.unified_server_args,
            *spec_args,
            *(extra_args or []),
        ]
        env = {
            **os.environ,
            "SGLANG_HICACHE_FILE_BACKEND_STORAGE_DIR": cls.storage_path,
            **(cls.mooncake.server_env() if cls.mooncake is not None else {}),
        }
        cls.process = popen_launch_server(
            cls.model,
            cls.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=args,
            env=env,
        )

    @classmethod
    def stop_server(cls):
        if cls.process is not None:
            kill_process_tree(cls.process.pid)
            cls.process = None

    def random_prompt(self, seed: int) -> List[int]:
        rng = random.Random(seed)
        length = self.prompt_pages * PAGE_SIZE + self.prompt_extra_tokens
        return [rng.randint(1000, 30000) for _ in range(length)]

    def scrub_prompt(self, seed: int) -> List[int]:
        """A prompt of the same length that overwrites device slots."""
        return self.random_prompt(seed)

    def text_prompt(self, paragraph: str) -> List[int]:
        """Natural text, cut to the prompt length: a draft has to be right
        about some tokens for its KV to show in the accept length."""
        from sglang.benchmark.utils import get_tokenizer

        length = self.prompt_pages * PAGE_SIZE + self.prompt_extra_tokens
        tokenizer = get_tokenizer(self.model)
        input_ids = tokenizer.encode(paragraph * 60)[:length]
        self.assertEqual(len(input_ids), length)
        return input_ids

    def ledger_prompt(self, seed: int, pages: Optional[int] = None) -> List[int]:
        """Natural sentences of one pattern whose particulars never repeat.

        A draft can follow the pattern from the prefix but has to guess the
        particulars, so its accept length stays below that of a perfect
        draft, and falls when its KV of the prefix is lost. A longer prompt
        of the same seed starts with the shorter one.
        """
        from sglang.benchmark.utils import get_tokenizer

        rng = random.Random(seed)
        days = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday"]
        sellers = ["the baker", "the miller", "the weaver", "the potter", "the smith"]
        buyers = ["a teacher", "a sailor", "a farmer", "a doctor", "a carpenter"]
        goods = ["loaves", "sacks of flour", "blankets", "clay jars", "iron nails"]
        towns = ["Aldery", "Brinmoor", "Castow", "Dunmere", "Elstead", "Farrow"]
        text = "The ledger of the market, one sale to a line.\n"
        for _ in range(400):
            text += (
                f"On {rng.choice(days)} {rng.choice(sellers)} sold "
                f"{rng.randint(2, 97)} {rng.choice(goods)} to {rng.choice(buyers)} "
                f"from {rng.choice(towns)} for {rng.randint(3, 480)} crowns.\n"
            )
        length = (pages or self.prompt_pages) * PAGE_SIZE + self.prompt_extra_tokens
        input_ids = get_tokenizer(self.model).encode(text)[:length]
        self.assertEqual(len(input_ids), length)
        return input_ids

    @staticmethod
    def spec_stats(result: Dict) -> tuple:
        meta = result["meta_info"]
        return meta["spec_verify_ct"], meta["spec_accept_length"]

    def generate(self, input_ids: List[int], cache_salt: Optional[str] = None) -> Dict:
        body = {
            "input_ids": input_ids,
            "sampling_params": {
                "temperature": 0,
                "max_new_tokens": self.max_new_tokens,
                "ignore_eos": True,
            },
            "return_logprob": True,
        }
        if cache_salt is not None:
            body["cache_salt"] = cache_salt
        response = requests.post(f"{self.base_url}/generate", json=body, timeout=600)
        self.assertEqual(response.status_code, 200, response.text)
        return response.json()

    def flush_cache(self):
        """Drop the device and host tiers, so the next hit has to come from L3."""
        response = requests.post(
            f"{self.base_url}/flush_cache", params={"timeout": 30}, timeout=40
        )
        response.raise_for_status()

    def server_info(self) -> Dict:
        response = requests.get(f"{self.base_url}/server_info", timeout=30)
        response.raise_for_status()
        return response.json()

    def assert_storage_hit(self, result: Dict, pages: Optional[int] = None):
        details = result["meta_info"].get("cached_tokens_details") or {}
        expected = (pages or self.prompt_pages) * PAGE_SIZE
        self.assertEqual(details.get("storage"), expected, details)
        self.assertEqual(details.get("device"), 0, details)
        self.assertEqual(details.get("host"), 0, details)
        self.assertEqual(
            details.get("storage_backend"), self.expected_storage_backend, details
        )

    def assert_close_output(self, result: Dict, reference: Dict, tolerance=0.5):
        """Same first-token distribution as another process or topology.

        Greedy tokens can flip on near-ties across processes, so compare the
        first position's top-1 logprob, which is continuous in the logits.
        """
        got = result["meta_info"]["output_token_logprobs"][0][0]
        want = reference["meta_info"]["output_token_logprobs"][0][0]
        self.assertLess(abs(got - want), tolerance, (got, want))

    def assert_kl_against_recompute(
        self, result: Dict, input_ids: List[int], threshold: float = 0.005
    ):
        """The cached run's output distribution matches a cache-free prefill.

        The prompt plus the generated tokens is prefilled under a fresh salt,
        so nothing is reused, and the logprobs it assigns to the generated
        tokens are compared with those of the run under test (the estimator
        of kl_test_utils.compare_kl_divergence).
        """
        output_ids = result["output_ids"]
        cached = [x[0] for x in result["meta_info"]["output_token_logprobs"]]
        response = requests.post(
            f"{self.base_url}/generate",
            json={
                "input_ids": input_ids + output_ids,
                "sampling_params": {"temperature": 0, "max_new_tokens": 0},
                "return_logprob": True,
                # The first position of the window carries no logprob.
                "logprob_start_len": len(input_ids) - 1,
                "cache_salt": uuid.uuid4().hex,
            },
            timeout=600,
        )
        self.assertEqual(response.status_code, 200, response.text)
        meta = response.json()["meta_info"]
        self.assertEqual(meta["cached_tokens"], 0)
        recomputed = [x[0] for x in meta["input_token_logprobs"]][-len(output_ids) :]
        self.assertEqual(len(recomputed), len(cached))
        kl = sum(
            (math.exp(r - c) - 1) - (r - c) for r, c in zip(recomputed, cached)
        ) / len(cached)
        self.assertLess(kl, threshold, (kl, recomputed[:4], cached[:4]))

    def assert_same_output(self, result: Dict, reference: Dict):
        """Token ids and logprobs equal to a device-cache hit of the same request."""
        self.assertEqual(result["output_ids"], reference["output_ids"])
        self.assertEqual(
            result["meta_info"]["output_token_logprobs"],
            reference["meta_info"]["output_token_logprobs"],
        )

    def reload_in_process(self, input_ids: List[int], salt: str, seed: int) -> Dict:
        """Compute, hit on device, drop the upper tiers, hit from L3.

        Returns the device-cache hit and the L3 hit, which has to reproduce
        it bit for bit. Between the two, another prompt of the same length
        is computed, so device slots no longer hold what the first run left.
        """
        computed = self.generate(input_ids, salt)
        self.assertEqual(computed["meta_info"]["cached_tokens"], 0)
        device_hit = self.generate(input_ids, salt)
        details = device_hit["meta_info"]["cached_tokens_details"]
        self.assertEqual(details["device"], self.prompt_pages * PAGE_SIZE, details)
        self.flush_cache()
        self.generate(self.scrub_prompt(seed + 1), uuid.uuid4().hex)
        self.flush_cache()
        storage_hit = self.generate(input_ids, salt)
        self.assert_storage_hit(storage_hit)
        self.assert_same_output(storage_hit, device_hit)
        return device_hit, storage_hit

    def wait_for_backups(self, timeout: float = 120.0) -> Dict[str, tuple]:
        """The file backend's objects once the server has stopped writing.

        Backups run behind the requests. Stopping a server in the middle of
        one leaves a page without some of its chunks, so wait until the
        directory has not changed for a few seconds.
        """
        deadline = time.monotonic() + timeout
        listing, stable_since = None, time.monotonic()
        while time.monotonic() < deadline:
            # Objects only: a temporary file can vanish while it is listed.
            current = {
                name: os.path.getsize(os.path.join(self.storage_path, name))
                for name in os.listdir(self.storage_path)
                if name.endswith(".bin")
            }
            if current != listing:
                listing, stable_since = current, time.monotonic()
            elif time.monotonic() - stable_since >= 5:
                return self.l3_objects()
            time.sleep(1)
        self.fail("the L3 store kept changing")

    def l3_objects(self) -> Dict[str, tuple]:
        """name -> (size, sha256) of every object of the file backend."""
        objects = {}
        for name in sorted(os.listdir(self.storage_path)):
            if not name.endswith(".bin"):
                continue
            with open(os.path.join(self.storage_path, name), "rb") as f:
                data = f.read()
            objects[name] = (len(data), hashlib.sha256(data).hexdigest())
        return objects
