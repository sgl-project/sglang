"""Validation E2E Tests.

Tests for validation features like ignore_eos and large token handling.

Source: Migrated from e2e_grpc/validation/test_openai_server_ignore_eos.py
        and e2e_grpc/validation/test_large_max_new_tokens.py
"""

from __future__ import annotations

import logging
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import pytest

logger = logging.getLogger(__name__)

# Lazy load tokenizer to avoid import errors if transformers not installed
_tokenizer_cache: dict = {}
_tokenizer_lock = threading.Lock()


def get_tokenizer(model_path: str):
    """Get tokenizer for a model, with caching."""
    if model_path not in _tokenizer_cache:
        with _tokenizer_lock:
            # Re-check after acquiring the lock to handle race conditions
            if model_path not in _tokenizer_cache:
                from transformers import AutoTokenizer

                _tokenizer_cache[model_path] = AutoTokenizer.from_pretrained(model_path)
    return _tokenizer_cache[model_path]


# =============================================================================
# Ignore EOS Tests (Llama 8B)
# =============================================================================


@pytest.mark.model("llama-8b")
@pytest.mark.gateway(extra_args=["--history-backend", "memory"])
@pytest.mark.parametrize("setup_backend", ["grpc"], indirect=True)
class TestIgnoreEOS:
    """Tests for ignore_eos feature."""

    def test_ignore_eos(self, setup_backend):
        """Test that ignore_eos=True allows generation to continue beyond EOS token.

        When ignore_eos=True, the model should generate until max_tokens is reached,
        even if it encounters an EOS token.
        """
        _, model, client, _ = setup_backend

        tokenizer = get_tokenizer(model)
        max_tokens = 200

        # Request without ignore_eos (default behavior - stops at EOS)
        response_default = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "Count from 1 to 20."},
            ],
            temperature=0,
            max_tokens=max_tokens,
            extra_body={"ignore_eos": False},
        )

        # Request with ignore_eos=True (continues past EOS until max_tokens)
        response_ignore_eos = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "Count from 1 to 20."},
            ],
            temperature=0,
            max_tokens=max_tokens,
            extra_body={"ignore_eos": True},
        )

        default_tokens = len(
            tokenizer.encode(response_default.choices[0].message.content)
        )
        ignore_eos_tokens = len(
            tokenizer.encode(response_ignore_eos.choices[0].message.content)
        )

        # Check if ignore_eos resulted in more tokens or exactly max_tokens
        # The ignore_eos response should either:
        # 1. Have more tokens than the default response (if default stopped at EOS before max_tokens)
        # 2. Have exactly max_tokens (if it reached the max_tokens limit)
        assert (
            ignore_eos_tokens > default_tokens or ignore_eos_tokens >= max_tokens
        ), f"ignore_eos did not generate more tokens: {ignore_eos_tokens} vs {default_tokens}"

        assert response_ignore_eos.choices[0].finish_reason == "length", (
            f"Expected finish_reason='length' for ignore_eos=True, "
            f"got {response_ignore_eos.choices[0].finish_reason}"
        )


# =============================================================================
# EOS Token Stripping Tests (Llama 1B)
#
# Verify that EOS tokens are stripped from the gateway's decoded output even
# when `skip_special_tokens=False`. gRPC backends return raw token IDs
# including the EOS that ended generation; without the StopDecoder stripping
# them at the token-id level, EOS strings like `<|eom_id|>` leak into
# `message.content` (and tool-call arguments).
#
# This is the user-visible regression guard for the gateway-half wiring of
# smg PR #1122 (`create_stop_decoder` consumes `tokenizer.eos_token_ids()`).
# Ported from upstream smg `e2e_test/chat_completions/test_validation.py`.
# =============================================================================


@pytest.mark.model("llama-1b")
@pytest.mark.gateway(
    extra_args=["--tool-call-parser", "llama", "--history-backend", "memory"]
)
@pytest.mark.parametrize("setup_backend", ["grpc"], indirect=True)
class TestEosTokenStripping:
    """Verify EOS tokens are stripped from output by the gateway StopDecoder.

    Llama-3.2-1B-Instruct lists `<|eom_id|>`, `<|eot_id|>`, `<|end_of_text|>`
    as EOS tokens via `generation_config.json`; none should appear in
    `message.content` or tool-call arguments under normal settings.
    """

    EOS_TOKENS = ["<|eom_id|>", "<|eot_id|>", "<|end_of_text|>"]

    def _assert_no_eos(self, text: str, label: str) -> None:
        for eos in self.EOS_TOKENS:
            assert eos not in text, f"{label}: EOS token {eos} leaked into {text!r}"

    def _assert_stopped_on_eos(self, choice, label: str) -> str:
        """Precondition for any "no EOS leaked" assertion: the model must
        have actually emitted EOS (finish_reason="stop") and returned
        non-empty content. Without this, `_assert_no_eos("")` passes
        vacuously and the test silently no-ops."""
        assert choice.finish_reason == "stop", (
            f"{label}: model hit {choice.finish_reason} before EOS — test "
            f"inconclusive, raise max_tokens or shorten the prompt"
        )
        content = choice.message.content
        assert content, f"{label}: expected non-empty content, got {content!r}"
        return content

    def test_no_eos_in_content_default(self, setup_backend):
        """Default settings: EOS must not appear in content."""
        _, model, client, _ = setup_backend
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": "Say hello in one sentence."}],
            temperature=0,
            max_tokens=50,
        )
        content = self._assert_stopped_on_eos(response.choices[0], "default")
        self._assert_no_eos(content, "default")

    def test_no_eos_in_content_skip_special_false(self, setup_backend):
        """`skip_special_tokens=False`: EOS must still be stripped at the
        token-id level by the StopDecoder. The published llm-tokenizer 1.3.2
        leaks EOS here, which is the regression this PR's patch fixes."""
        _, model, client, _ = setup_backend
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": "Say hello in one sentence."}],
            temperature=0,
            max_tokens=50,
            extra_body={"skip_special_tokens": False},
        )
        content = self._assert_stopped_on_eos(
            response.choices[0], "skip_special_tokens=False"
        )
        self._assert_no_eos(content, "skip_special_tokens=False")

    def test_no_eos_in_streaming_content_skip_special_false(self, setup_backend):
        """Streaming path: EOS must not leak into any delta or the aggregated
        content. Exercises `process_streaming_chunks` (non-streaming tests
        don't reach the per-index stop-decoder cache in streaming.rs)."""
        _, model, client, _ = setup_backend
        stream = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": "Say hello in one sentence."}],
            temperature=0,
            max_tokens=50,
            stream=True,
            extra_body={"skip_special_tokens": False},
        )
        aggregated = ""
        finish_reason: str | None = None
        for chunk in stream:
            if not chunk.choices:
                continue
            choice = chunk.choices[0]
            if choice.delta and choice.delta.content:
                self._assert_no_eos(choice.delta.content, "streaming delta")
                aggregated += choice.delta.content
            if choice.finish_reason:
                finish_reason = choice.finish_reason
        assert finish_reason == "stop", (
            f"streaming: model hit {finish_reason} before EOS — test " f"inconclusive"
        )
        assert aggregated, "streaming: aggregated content is empty"
        self._assert_no_eos(aggregated, "streaming aggregated")

    def test_no_eos_in_tool_call_with_tools(self, setup_backend):
        """Tool-call arguments must not contain EOS — even with
        `skip_special_tokens=False` and a tool-call-parser configured."""
        _, model, client, _ = setup_backend
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "add",
                    "description": "Compute the sum of two numbers",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "a": {"type": "integer", "description": "A number"},
                            "b": {"type": "integer", "description": "A number"},
                        },
                        "required": ["a", "b"],
                    },
                },
            }
        ]
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": "Compute (3+5)"}],
            tools=tools,
            tool_choice="required",
            temperature=0,
            max_tokens=200,
            extra_body={"skip_special_tokens": False},
        )
        choice = response.choices[0]
        # `tool_choice="required"` must produce at least one tool_call;
        # without this guard, an empty tool_calls + empty content silently
        # passes the test.
        assert (
            choice.message.tool_calls
        ), f"tool_choice=required must produce tool_calls, got {choice.message!r}"
        for tc in choice.message.tool_calls:
            self._assert_no_eos(tc.function.arguments, "tool_call.arguments")
        if choice.message.content:
            self._assert_no_eos(choice.message.content, "fallback content")

    def test_no_stop_trim_with_skip_special_true(self, setup_backend):
        """`no_stop_trim=True` + default `skip_special_tokens=True`: EOS
        is kept in the token-id list but decoded as an empty string, so it
        must remain invisible in `content`."""
        _, model, client, _ = setup_backend
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": "Say hello in one sentence."}],
            temperature=0,
            max_tokens=50,
            extra_body={"no_stop_trim": True},
        )
        content = self._assert_stopped_on_eos(
            response.choices[0], "no_stop_trim + skip_special=True"
        )
        self._assert_no_eos(content, "no_stop_trim + skip_special=True")

    def test_no_stop_trim_with_skip_special_false(self, setup_backend):
        """`no_stop_trim=True` + `skip_special_tokens=False`: EOS becomes a
        VISIBLE stop, decoded into `content` (matches sglang behavior)."""
        _, model, client, _ = setup_backend
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": "Say hello in one sentence."}],
            temperature=0,
            max_tokens=50,
            extra_body={"no_stop_trim": True, "skip_special_tokens": False},
        )
        choice = response.choices[0]
        content = choice.message.content
        assert content is not None
        assert choice.finish_reason == "stop", (
            f"Model hit max_tokens before EOS — test inconclusive "
            f"(finish_reason={choice.finish_reason})"
        )
        assert any(eos in content for eos in self.EOS_TOKENS), (
            f"EOS must be visible with no_stop_trim=True + "
            f"skip_special_tokens=False, got {content!r}"
        )


# =============================================================================
# Large Max New Tokens Tests (Llama 8B)
#
# NOTE: This test verifies concurrent request handling with large token limits.
# The original test monitored server logs to verify concurrency, which is not
# possible with the pool-based infrastructure. This simplified version verifies
# that concurrent requests complete successfully.
# =============================================================================


@pytest.mark.model("llama-8b")
@pytest.mark.gateway(extra_args=["--history-backend", "memory"])
@pytest.mark.parametrize("setup_backend", ["grpc"], indirect=True)
class TestLargeMaxNewTokens:
    """Tests for handling large max_new_tokens with concurrent requests."""

    def test_concurrent_chat_completions(self, setup_backend):
        """Test that multiple concurrent requests with large token generation complete.

        This test sends multiple requests that ask for long outputs concurrently
        to verify the server can handle concurrent long-running requests.
        """
        _, model, client, _ = setup_backend

        num_requests = 4

        def run_chat_completion():
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": "You are a helpful AI assistant"},
                    {
                        "role": "user",
                        "content": "Please repeat the word 'hello' for 100 times.",
                    },
                ],
                temperature=0,
                max_tokens=256,  # Reasonable limit for concurrent test
            )
            return response

        # Send concurrent requests
        start_time = time.time()
        futures = []
        with ThreadPoolExecutor(max_workers=num_requests) as executor:
            for _ in range(num_requests):
                futures.append(executor.submit(run_chat_completion))

            # Wait for all to complete and collect results
            responses = [f.result() for f in futures]

        elapsed = time.time() - start_time
        logger.info("Completed %d concurrent requests in %.2fs", num_requests, elapsed)

        # Verify all requests completed successfully
        assert len(responses) == num_requests
        for i, response in enumerate(responses):
            assert response.choices[
                0
            ].message.content, f"Request {i} returned empty content"
            assert response.choices[0].finish_reason in ("stop", "length"), (
                f"Request {i} had unexpected finish_reason: "
                f"{response.choices[0].finish_reason}"
            )
