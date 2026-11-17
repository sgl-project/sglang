"""Shared helpers for unified-key HiCache storage + speculative decoding tests.

One server at a time serves a class; a test may stop it and start another one
(another speculative setup, another topology) on the same L3 store.
"""

import hashlib
import json
import math
import os
import random
import re
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


def mha_chunk_rows(name: str) -> int:
    """K and V rows, one per token and layer, of an MHA chunk object."""
    coordinate = name[: -len(".bin")].split("_", 2)[2]
    first, last = re.match(r"[LD](\d+)-(\d+)", coordinate).groups()
    return (int(last) - int(first)) * 2 * PAGE_SIZE


def row_cosines(a: bytes, b: bytes, rows: int):
    """Cosine similarity of each of `rows` equal slices of two bfloat16 objects."""
    import numpy as np

    def decode(data):
        bits = np.frombuffer(data, dtype="<u2").astype(np.uint32) << 16
        return bits.view(np.float32).astype(np.float64).reshape(rows, -1)

    x, y = decode(a), decode(b)
    norms = np.linalg.norm(x, axis=1) * np.linalg.norm(y, axis=1)
    return (x * y).sum(axis=1) / np.where(norms > 0, norms, 1.0)


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
    def hicache_server_args(cls) -> List[str]:
        """HiCache under unified keys on the class's L3 store."""
        extra_config = {
            "hicache_storage_pass_prefix_keys": True,
            "prefetch_threshold": PAGE_SIZE,
        }
        return [
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
        ]

    @classmethod
    def hicache_server_env(cls) -> Dict[str, str]:
        """Where a server finds the class's L3 store."""
        return {
            "SGLANG_HICACHE_FILE_BACKEND_STORAGE_DIR": cls.storage_path,
            **(cls.mooncake.server_env() if cls.mooncake is not None else {}),
        }

    @classmethod
    def launch_server(
        cls, spec_args: List[str], extra_args: Optional[List[str]] = None, log=None
    ):
        """Start a server on the class's L3 store, replacing a running one.

        `log`, a text sink, also receives what the server prints.
        """
        cls.stop_server()
        default_port = int(DEFAULT_URL_FOR_TEST.rsplit(":", 1)[1])
        cls.base_url = f"http://127.0.0.1:{find_available_port(default_port)}"
        args = [*cls.hicache_server_args(), *spec_args, *(extra_args or [])]
        env = {**os.environ, **cls.hicache_server_env()}
        cls.process = popen_launch_server(
            cls.model,
            cls.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=args,
            env=env,
            return_stdout_stderr=(log, log) if log is not None else None,
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

    def generate(
        self,
        input_ids: List[int],
        cache_salt: Optional[str] = None,
        max_new_tokens: Optional[int] = None,
    ) -> Dict:
        """One request, or one batch when `input_ids` is a list of prompts."""
        if max_new_tokens is None:
            max_new_tokens = self.max_new_tokens
        body = {
            "input_ids": input_ids,
            "sampling_params": {
                "temperature": 0,
                "max_new_tokens": max_new_tokens,
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

    def assert_device_hit(self, result: Dict, pages: Optional[int] = None) -> Dict:
        details = result["meta_info"].get("cached_tokens_details") or {}
        expected = (pages or self.prompt_pages) * PAGE_SIZE
        self.assertEqual(details.get("device"), expected, details)
        return result

    def compute_and_hit(self, input_ids: List[int], salt: str) -> Dict:
        """Compute a prompt, then return its device-cache hit."""
        computed = self.generate(input_ids, salt)
        self.assertEqual(computed["meta_info"]["cached_tokens"], 0)
        return self.assert_device_hit(self.generate(input_ids, salt))

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
        device_hit = self.compute_and_hit(input_ids, salt)
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

    def use_new_store(self) -> str:
        """Point the class at another, empty file store until the test ends.

        Returns the directory of the store used so far.
        """
        directory = tempfile.TemporaryDirectory(prefix="hicache_unified_spec_")
        previous = self.storage_path
        self.addCleanup(directory.cleanup)
        self.addCleanup(setattr, type(self), "storage_path", previous)
        # Runs first: no server may write into a store that is being removed.
        self.addCleanup(self.stop_server)
        type(self).storage_path = directory.name
        return previous

    def zero_objects(self, names):
        """Overwrite objects of the file backend with zeros of the same size."""
        for name in names:
            path = os.path.join(self.storage_path, name)
            with open(path, "r+b") as f:
                f.write(bytes(os.path.getsize(path)))

    def matching_rows(self, names, store_a: str, store_b: str) -> tuple:
        """Compare same-named MHA chunk objects of two stores row by row.

        Returns (share of rows whose cosine similarity is at least 0.999,
        lowest cosine). Rows computed twice from the same tokens differ by
        rounding only; a missing or misplaced row has nothing in common.
        """
        matching, total, lowest = 0, 0, 1.0
        for name in sorted(names):
            with open(os.path.join(store_a, name), "rb") as f:
                a = f.read()
            with open(os.path.join(store_b, name), "rb") as f:
                b = f.read()
            cosines = row_cosines(a, b, mha_chunk_rows(name))
            matching += int((cosines >= 0.999).sum())
            total += len(cosines)
            lowest = min(lowest, float(cosines.min()))
        self.assertGreater(total, 0)
        return matching / total, lowest

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
