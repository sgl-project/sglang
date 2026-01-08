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
