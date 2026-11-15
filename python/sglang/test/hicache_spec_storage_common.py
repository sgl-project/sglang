"""Shared helpers for HiCache storage + EAGLE3 speculative decoding tests."""

import json
import os
import subprocess
from typing import Dict, List

import psutil
import requests

from sglang.benchmark.utils import get_tokenizer
from sglang.srt.utils import kill_process_tree
from sglang.test.test_utils import (
    DEFAULT_DRAFT_MODEL_EAGLE3,
    DEFAULT_TARGET_MODEL_EAGLE3,
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    find_available_port,
    popen_launch_server,
)
from sglang.utils import wait_for_http_ready


class HiCacheSpecStorageMixin:
    """Common EAGLE3 + HiCache storage loadback flow.

    Subclasses provide the storage backend, environment, and the backend-specific
    wait condition before server restart.
    """

    model = DEFAULT_TARGET_MODEL_EAGLE3
    draft_model = DEFAULT_DRAFT_MODEL_EAGLE3

    input_token_len = 1024
    max_new_tokens = 200
    first_measure_new_tokens = 128
    page_size = 64
    min_expected_accept_length = 7.0
    min_second_to_first_accept_ratio = 0.9

    storage_backend = None
    expected_storage_backend = None

    @classmethod
    def _get_storage_backend_extra_config(cls):
        return {
            "hicache_storage_pass_prefix_keys": True,
        }

    @classmethod
    def _get_spec_server_env(cls) -> Dict[str, str]:
        return {}

    @classmethod
    def _get_spec_server_args(cls) -> List[str]:
        if cls.storage_backend is None:
            raise ValueError("storage_backend must be set by subclasses.")

        return [
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
            cls.storage_backend,
            "--hicache-storage-prefetch-policy",
            "wait_complete",
            "--hicache-storage-backend-extra-config",
            json.dumps(cls._get_storage_backend_extra_config()),
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

    @classmethod
    def _build_spec_prompt(cls):
        cls.tokenizer = get_tokenizer(cls.model)
        cls.prompt_input_ids = cls._build_long_repetitive_prompt_ids(
            cls.tokenizer, cls.input_token_len
        )

    @classmethod
    def _launch_spec_server(cls):
        default_port = int(DEFAULT_URL_FOR_TEST.rsplit(":", 1)[1])
        cls.base_url = f"http://127.0.0.1:{find_available_port(default_port)}"
        cls._build_spec_prompt()
        cls.other_args = cls._get_spec_server_args()
        cls.env = {
            **os.environ,
            "SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN": "1",
            **cls._get_spec_server_env(),
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

    def _wait_for_storage_before_restart(self):
        raise NotImplementedError

    def _run_storage_loadback_keeps_spec_accept_length(self):
        first = self._send_long_prompt(max_new_tokens=self.first_measure_new_tokens)
        first_accept_length = self._get_spec_accept_length(first)
        self.assertGreaterEqual(
            first_accept_length,
            self.min_expected_accept_length,
            f"First prompt accept length is too low: {first_accept_length=}",
        )

        self._wait_for_storage_before_restart()
        self._restart_spec_server()

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
            f"{self.storage_backend} storage, got {cached_details=}",
        )
        self.assertEqual(
            cached_details.get("storage_backend"),
            self.expected_storage_backend,
            f"Expected {self.expected_storage_backend} in cache report, "
            f"got {cached_details=}",
        )
        self.assertGreaterEqual(
            second_accept_length,
            self.min_expected_accept_length,
            f"Second prompt accept length is too low: {second_accept_length=}",
        )
        self.assertGreaterEqual(
            second_accept_length,
            first_accept_length * self.min_second_to_first_accept_ratio,
            f"Spec accept length dropped after {self.storage_backend}-storage "
            f"loadback: {first_accept_length=:.3f}, {second_accept_length=:.3f}",
        )
