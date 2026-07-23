"""Pinned real-model correctness gate for MLX Gemma 4 Frozen-KV MTP.

This Stage B test intentionally launches target-only and MTP in separate fresh
processes.  It is not part of the offline per-PR Stage A lane.
"""

from __future__ import annotations

import importlib.util
import os
import platform
import threading
import time
import unittest
from contextlib import contextmanager

import requests

from sglang.srt.utils import kill_process_tree
from sglang.test.ci.ci_register import register_cpu_ci, register_mlx_ci
from sglang.test.test_utils import (
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
    try_cached_model,
)

register_cpu_ci(est_time=1, suite="base-a-test-cpu")
register_mlx_ci(est_time=8, suite="stage-b-e2e-mlx")

TARGET = os.environ.get("SGLANG_MLX_GEMMA4_TARGET", "mlx-community/gemma-4-e2b-it-4bit")
TARGET_REVISION = os.environ.get(
    "SGLANG_MLX_GEMMA4_TARGET_REVISION",
    "238767527555cb75a05732a84dff5d6ba0dd6809",
)
ASSISTANT = os.environ.get(
    "SGLANG_MLX_GEMMA4_ASSISTANT",
    "mlx-community/gemma-4-E2B-it-assistant-bf16",
)
ASSISTANT_REVISION = os.environ.get(
    "SGLANG_MLX_GEMMA4_ASSISTANT_REVISION",
    "a7770799b560135ebdbfae8b7f468947415003bc",
)

SHORT_PROMPT = "Explain why the sky appears blue in two sentences."
CROSS_WINDOW_PROMPT = (
    "Read this repeated context carefully. "
    + ("blue green amber violet " * 140)
    + "Now state the first color in one sentence."
)

_HAS_RUNTIME = (
    platform.system() == "Darwin"
    and platform.machine() == "arm64"
    and importlib.util.find_spec("mlx") is not None
    and importlib.util.find_spec("mlx_vlm.speculative.drafters.gemma4_assistant")
    is not None
)


@unittest.skipUnless(_HAS_RUNTIME, "requires Apple Silicon, MLX, and mlx-vlm")
class TestGemma4MTPMlxCorrectness(CustomTestCase):
    base_url = DEFAULT_URL_FOR_TEST
    maxDiff = None

    @classmethod
    @contextmanager
    def _server(cls, *, mtp: bool):
        model = try_cached_model(TARGET)
        args = [
            "--revision",
            TARGET_REVISION,
            "--served-model-name",
            TARGET,
            "--disable-radix-cache",
            "--disable-overlap-schedule",
            "--chunked-prefill-size",
            "-1",
            "--max-running-requests",
            "1",
            "--context-length",
            "2048",
            "--max-total-tokens",
            "2048",
            "--log-level",
            "warning",
        ]
        if mtp:
            args.extend(
                [
                    "--speculative-algorithm",
                    "FROZEN_KV_MTP",
                    "--speculative-draft-model-path",
                    try_cached_model(ASSISTANT),
                    "--speculative-draft-model-revision",
                    ASSISTANT_REVISION,
                    "--speculative-num-steps",
                    "1",
                    "--speculative-eagle-topk",
                    "1",
                    "--speculative-num-draft-tokens",
                    "2",
                ]
            )
        env = os.environ.copy()
        env["SGLANG_USE_MLX"] = "1"
        process = popen_launch_server(
            model,
            cls.base_url,
            timeout=max(DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH, 600),
            other_args=args,
            env=env,
        )
        try:
            yield
        finally:
            kill_process_tree(process.pid)

    @classmethod
    def _raw(cls, prompt: str, horizon: int, *, ignore_eos: bool) -> dict:
        payload = {
            "text": prompt,
            "sampling_params": {
                "temperature": 0,
                "max_new_tokens": horizon,
                "ignore_eos": ignore_eos,
            },
        }
        response = requests.post(f"{cls.base_url}/generate", json=payload, timeout=180)
        response.raise_for_status()
        result = response.json()
        assert len(result["output_ids"]) <= horizon
        return result

    @classmethod
    def _chat(cls, prompt: str, horizon: int, *, ignore_eos: bool) -> dict:
        payload = {
            "model": TARGET,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0,
            "max_tokens": horizon,
            "ignore_eos": ignore_eos,
            # SGLang's non-streaming extension exposes exact output token IDs
            # when metadata is requested; logprobs remain disabled.
            "return_meta_info": True,
        }
        response = requests.post(
            f"{cls.base_url}/v1/chat/completions", json=payload, timeout=180
        )
        response.raise_for_status()
        result = response.json()
        choice = result["choices"][0]
        assert len(choice["output_token_ids"]) <= horizon
        return result

    @classmethod
    def _spec_state(cls) -> dict:
        response = requests.get(f"{cls.base_url}/server_info", timeout=30)
        response.raise_for_status()
        states = response.json()["internal_states"]
        assert len(states) == 1
        return states[0]["speculative_worker"]

    @classmethod
    def _wait_for_clean_state(cls, timeout: float = 15) -> dict:
        deadline = time.monotonic() + timeout
        last = None
        while time.monotonic() < deadline:
            last = cls._spec_state()
            if (
                last["active_request_count"] == 0
                and last["native_request_count"] == 0
                and last["assistant_request_binding_count"] == 0
            ):
                return last
            time.sleep(0.05)
        raise AssertionError(f"MLX MTP request state did not drain: {last}")

    @classmethod
    def _abort_streaming_request(cls) -> None:
        rid = "gemma4-mtp-e2e-abort"
        opened = threading.Event()
        result: dict[str, object] = {}
        errors: list[Exception] = []

        def generate() -> None:
            try:
                with requests.post(
                    f"{cls.base_url}/generate",
                    json={
                        "rid": rid,
                        "text": "Count upward forever, one integer per line.",
                        "sampling_params": {
                            "temperature": 0,
                            "max_new_tokens": 1024,
                            "ignore_eos": True,
                        },
                        "stream": True,
                    },
                    stream=True,
                    timeout=180,
                ) as response:
                    result["status"] = response.status_code
                    opened.set()
                    result["body"] = b"\n".join(response.iter_lines())
            except Exception as exc:  # surfaced on the test thread below
                errors.append(exc)
                opened.set()

        thread = threading.Thread(target=generate, daemon=True)
        thread.start()
        if not opened.wait(30):
            raise AssertionError("stream did not produce a response")
        abort = requests.post(
            f"{cls.base_url}/abort_request", json={"rid": rid}, timeout=30
        )
        abort.raise_for_status()
        thread.join(30)
        if thread.is_alive():
            raise AssertionError("aborted stream did not terminate")
        if errors:
            raise errors[0]
        if result.get("status") != 200:
            raise AssertionError(f"aborted stream returned {result.get('status')}")
        if b'"type":"abort"' not in result.get("body", b""):
            raise AssertionError("stream did not report an abort finish reason")

    @staticmethod
    def _raw_signature(result: dict) -> tuple[list[int], str, str]:
        return (
            result["output_ids"],
            result["text"],
            result["meta_info"]["finish_reason"]["type"],
        )

    @staticmethod
    def _chat_signature(result: dict) -> tuple[list[int], str, str]:
        choice = result["choices"][0]
        return (
            choice["output_token_ids"],
            choice["message"]["content"],
            choice["finish_reason"],
        )

    def test_target_only_and_mtp_exact_server_parity(self):
        prompts = {"short": SHORT_PROMPT, "cross_window": CROSS_WINDOW_PROMPT}
        fixed_cases = [
            (endpoint, prompt_name, horizon)
            for endpoint in ("raw", "chat")
            for prompt_name in prompts
            for horizon in (1, 16, 64)
        ]
        target: dict[tuple[str, str, int], tuple[list[int], str, str]] = {}

        with self._server(mtp=False):
            for endpoint, prompt_name, horizon in fixed_cases:
                if endpoint == "raw":
                    result = self._raw(prompts[prompt_name], horizon, ignore_eos=True)
                    signature = self._raw_signature(result)
                    prompt_tokens = result["meta_info"]["prompt_tokens"]
                else:
                    result = self._chat(prompts[prompt_name], horizon, ignore_eos=True)
                    signature = self._chat_signature(result)
                    prompt_tokens = result["usage"]["prompt_tokens"]
                self.assertEqual(len(signature[0]), horizon)
                self.assertEqual(signature[2], "length")
                if prompt_name == "cross_window":
                    self.assertGreater(prompt_tokens, 512)
                target[(endpoint, prompt_name, horizon)] = signature

            natural = self._chat(
                "Reply with exactly one word: OK", 64, ignore_eos=False
            )
            target[("chat", "natural_stop", 64)] = self._chat_signature(natural)
            self.assertEqual(target[("chat", "natural_stop", 64)][2], "stop")
            repeated = self._raw(SHORT_PROMPT, 16, ignore_eos=True)
            self.assertEqual(
                self._raw_signature(repeated), target[("raw", "short", 16)]
            )

        with self._server(mtp=True):
            observed_accept = 0
            observed_proposed = 0
            accept_and_reject = False
            for endpoint, prompt_name, horizon in fixed_cases:
                if endpoint == "raw":
                    result = self._raw(prompts[prompt_name], horizon, ignore_eos=True)
                    signature = self._raw_signature(result)
                    meta = result["meta_info"]
                else:
                    result = self._chat(prompts[prompt_name], horizon, ignore_eos=True)
                    signature = self._chat_signature(result)
                    meta = result["choices"][0]["meta_info"]
                self.assertEqual(
                    signature,
                    target[(endpoint, prompt_name, horizon)],
                    (endpoint, prompt_name, horizon),
                )
                observed_accept += int(meta.get("spec_num_correct_drafts", 0))
                observed_proposed += int(meta.get("spec_num_proposed_drafts", 0))
                histogram = meta.get("spec_correct_drafts_histogram") or []
                if len(histogram) >= 2 and histogram[0] > 0 and histogram[1] > 0:
                    accept_and_reject = True

            natural = self._chat(
                "Reply with exactly one word: OK", 64, ignore_eos=False
            )
            self.assertEqual(
                self._chat_signature(natural),
                target[("chat", "natural_stop", 64)],
            )
            repeated = self._raw(SHORT_PROMPT, 16, ignore_eos=True)
            self.assertEqual(
                self._raw_signature(repeated), target[("raw", "short", 16)]
            )
            self.assertGreater(observed_proposed, 0)
            self.assertGreater(observed_accept, 0)
            self.assertTrue(accept_and_reject)

            self._abort_streaming_request()
            before_flush = self._wait_for_clean_state()
            self.assertGreater(before_flush["proposed_tokens"], 0)
            self.assertGreater(before_flush["verified_tokens"], 0)
            self.assertGreaterEqual(
                before_flush["proposed_tokens"], before_flush["verified_tokens"]
            )
            self.assertLessEqual(
                before_flush["accepted_draft_tokens"],
                before_flush["verified_tokens"],
            )

            fingerprint = before_flush["assistant_fingerprint"]
            load_count = before_flush["assistant_load_count"]
            flush = requests.post(f"{self.base_url}/flush_cache", timeout=30)
            flush.raise_for_status()
            after_flush = self._wait_for_clean_state()
            self.assertEqual(after_flush["assistant_fingerprint"], fingerprint)
            self.assertEqual(after_flush["assistant_load_count"], load_count)

            post_flush = self._raw(SHORT_PROMPT, 16, ignore_eos=True)
            self.assertEqual(
                self._raw_signature(post_flush), target[("raw", "short", 16)]
            )

            # Active MLX memory must settle after warmup/request churn.  The
            # allocator cache may remain populated and is intentionally not
            # required to return to its initial value.
            memory_samples = []
            for _ in range(5):
                self._raw(SHORT_PROMPT, 16, ignore_eos=True)
                memory_samples.append(
                    self._wait_for_clean_state()["mlx_active_memory_bytes"]
                )
            allowance = max(256 * 1024 * 1024, memory_samples[0] // 10)
            self.assertLessEqual(memory_samples[-1], memory_samples[0] + allowance)


if __name__ == "__main__":
    unittest.main()
