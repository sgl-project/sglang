"""
E2E test for HiCache file storage with EAGLE3 speculative decoding.

Usage:
    python3 -m pytest test/registered/hicache/test_hicache_spec_file_storage.py -v
"""

import json
import os
import shutil
import tempfile
import time
import unittest
from typing import Dict, List

import psutil
import requests

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

register_cuda_ci(est_time=168, suite="stage-b-test-1-gpu-large")


@unittest.skipIf(is_hip(), "HiCache + EAGLE3 file-storage loadback e2e is CUDA-only.")
class TestHiCacheSpecFileStorage(CustomTestCase):
    model = DEFAULT_TARGET_MODEL_EAGLE3
    draft_model = DEFAULT_DRAFT_MODEL_EAGLE3

    input_token_len = 1024
    max_new_tokens = 200
    page_size = 64
    min_expected_accept_length = 7.0
    min_second_to_first_accept_ratio = 0.9
    storage_wait_timeout = 30
    first_measure_new_tokens = 128

    @classmethod
    def setUpClass(cls):
        cls.temp_dir = tempfile.mkdtemp()
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
            "file",
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
            "SGLANG_HICACHE_FILE_BACKEND_STORAGE_DIR": cls.temp_dir,
        }
        cls.process = None
        cls._launch_server()

    @classmethod
    def _launch_server(cls):
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
    def _stop_server(cls):
        if getattr(cls, "process", None) is None:
            return

        process = cls.process
        try:
            root = psutil.Process(process.pid)
            watched_procs = [root] + root.children(recursive=True)
        except psutil.NoSuchProcess:
            watched_procs = []

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

    @classmethod
    def _restart_server(cls):
        cls._stop_server()
        cls._launch_server()

    @classmethod
    def _count_file_storage_pages(cls):
        try:
            filenames = os.listdir(cls.temp_dir)
        except FileNotFoundError:
            return 0, 0

        target_pages = 0
        draft_pages = 0
        for filename in filenames:
            if not filename.endswith(".bin"):
                continue
            if filename.startswith("d:"):
                draft_pages += 1
            else:
                target_pages += 1
        return target_pages, draft_pages

    @classmethod
    def _wait_for_file_storage_pages(cls):
        min_pages = (cls.input_token_len - 2 * cls.page_size) // cls.page_size
        deadline = time.monotonic() + cls.storage_wait_timeout
        target_pages = draft_pages = 0

        while time.monotonic() < deadline:
            target_pages, draft_pages = cls._count_file_storage_pages()
            if target_pages >= min_pages and draft_pages >= min_pages:
                return target_pages, draft_pages
            time.sleep(0.2)

        raise AssertionError(
            "Timed out waiting for HiCache file storage pages before restart: "
            f"{target_pages=}, {draft_pages=}, {min_pages=}"
        )

    @classmethod
    def tearDownClass(cls):
        cls._stop_server()
        if hasattr(cls, "temp_dir"):
            shutil.rmtree(cls.temp_dir, ignore_errors=True)

    @classmethod
    def _encode_without_special_tokens(cls, tokenizer, text: str) -> List[int]:
        return tokenizer.encode(text, add_special_tokens=False)

    @classmethod
    def _build_long_repetitive_prompt_ids(cls, tokenizer, target_len: int) -> List[int]:
        bos_ids = (
            [tokenizer.bos_token_id]
            if getattr(tokenizer, "bos_token_id", None) is not None
            else []
        )
        suffix_ids = cls._encode_without_special_tokens(
            tokenizer,
            "\n\nContinue the sequence with only the word apple separated by spaces.\n"
            "Answer: apple apple apple apple",
        )
        repeat_ids = cls._encode_without_special_tokens(tokenizer, " apple")
        if not repeat_ids:
            raise ValueError(
                "Tokenizer produced no ids for the repetitive prompt seed."
            )
        if len(bos_ids) + len(suffix_ids) >= target_len:
            raise ValueError(
                "Prompt suffix is too long: "
                f"{len(bos_ids)=}, {len(suffix_ids)=}, {target_len=}."
            )

        prefix_len = target_len - len(bos_ids) - len(suffix_ids)
        repeats = (prefix_len + len(repeat_ids) - 1) // len(repeat_ids)
        prefix_ids = (repeat_ids * repeats)[:prefix_len]
        prompt_ids = bos_ids + prefix_ids + suffix_ids
        assert len(prompt_ids) == target_len
        return prompt_ids

    def _send_long_prompt(self, max_new_tokens: int = None) -> Dict:
        if max_new_tokens is None:
            max_new_tokens = self.max_new_tokens
        response = requests.post(
            f"{self.base_url}/generate",
            json={
                "input_ids": self.prompt_input_ids,
                "sampling_params": {
                    "temperature": 0,
                    "max_new_tokens": max_new_tokens,
                    "ignore_eos": True,
                },
            },
            timeout=900,
        )
        self.assertEqual(
            response.status_code,
            200,
            f"Request failed: {response.status_code} - {response.text}",
        )
        return response.json()

    def _get_spec_accept_length(self, response_json: Dict) -> float:
        meta_info = response_json.get("meta_info", {})
        self.assertIn(
            "spec_accept_length",
            meta_info,
            f"Missing spec_accept_length in meta_info: {meta_info}",
        )
        return float(meta_info["spec_accept_length"])

    def test_file_storage_loadback_keeps_spec_accept_length(self):
        first = self._send_long_prompt(max_new_tokens=self.first_measure_new_tokens)
        first_accept_length = self._get_spec_accept_length(first)
        self.assertGreaterEqual(
            first_accept_length,
            self.min_expected_accept_length,
            f"First prompt accept length is too low: {first_accept_length=}",
        )

        target_pages, draft_pages = self._wait_for_file_storage_pages()
        print(f"file_storage_before_restart: {target_pages=}, {draft_pages=}")

        self._restart_server()

        second = self._send_long_prompt()
        second_accept_length = self._get_spec_accept_length(second)
        second_meta = second.get("meta_info", {})
        cached_details = second_meta.get("cached_tokens_details") or {}
        storage_cached_tokens = int(cached_details.get("storage", 0))

        print(
            f"{first_accept_length=:.3f}, {second_accept_length=:.3f}, "
            f"{storage_cached_tokens=}, {cached_details=}"
        )

        self.assertGreaterEqual(
            storage_cached_tokens,
            self.input_token_len - 2 * self.page_size,
            "Expected the second request to load the long prompt KV cache from "
            f"file storage, got {cached_details=}",
        )
        self.assertEqual(
            cached_details.get("storage_backend"),
            "HiCacheFile",
            f"Expected file storage backend in cache report, got {cached_details=}",
        )
        self.assertGreaterEqual(
            second_accept_length,
            self.min_expected_accept_length,
            f"Second prompt accept length is too low: {second_accept_length=}",
        )
        self.assertGreaterEqual(
            second_accept_length,
            first_accept_length * self.min_second_to_first_accept_ratio,
            "Spec accept length dropped after file-storage loadback: "
            f"{first_accept_length=:.3f}, {second_accept_length=:.3f}",
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
