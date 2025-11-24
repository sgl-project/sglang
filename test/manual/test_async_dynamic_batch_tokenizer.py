"""
Unit tests for AsyncDynamicbatchTokenizer.

Tests the async dynamic batching functionality for tokenization,
including batch efficiency, timeout handling, and error cases.
"""

import asyncio
import logging
import time
from unittest.mock import Mock

import pytest
from transformers import AutoTokenizer

from sglang.srt.managers.async_dynamic_batch_tokenizer import AsyncDynamicbatchTokenizer


class TestAsyncDynamicbatchTokenizer:
    """Test suite for AsyncDynamicbatchTokenizer."""

    @pytest.fixture
    def mock_tokenizer(self):
        """Create a mock tokenizer that behaves like HuggingFace tokenizer."""

        def mock_encode(texts, **kwargs):
            is_single = isinstance(texts, str)
            if is_single:
                texts = [texts]

            # Simulate tokenization - convert text to mock token ids
            input_ids = []
            token_type_ids = []

            for text in texts:
                # Simple mock: text length determines number of tokens
                tokens = [i for i in range(len(text.split()))]
                input_ids.append(tokens)

                if kwargs.get("return_token_type_ids", False):
                    token_type_ids.append([0] * len(tokens))

            result = {"input_ids": input_ids}
            if kwargs.get("return_token_type_ids", False):
                result["token_type_ids"] = token_type_ids

            # For single inputs, return individual result (not wrapped in a list)
            if is_single:
                result = {"input_ids": input_ids[0]}
                if kwargs.get("return_token_type_ids", False):
                    result["token_type_ids"] = token_type_ids[0]

            # Create a proper BatchEncoding-like object that supports dict operations
            class MockBatchEncoding(dict):
                def __init__(self, data):
                    super().__init__(data)
                    for key, value in data.items():
                        setattr(self, key, value)

            return MockBatchEncoding(result)

        # Return the function directly - the AsyncDynamicbatchTokenizer will call it
        return mock_encode

    @pytest.fixture
    def async_tokenizer(self, mock_tokenizer):
        """Create AsyncDynamicbatchTokenizer instance."""
        return AsyncDynamicbatchTokenizer(
            tokenizer=mock_tokenizer, max_batch_size=4, batch_wait_timeout_s=0.01
        )

    @pytest.mark.asyncio
    async def test_single_request(self, async_tokenizer):
        """Test tokenizing a single request."""
        text = "hello world"
        result = await async_tokenizer.encode(text)

        assert "input_ids" in result
        assert result["input_ids"] == [0, 1]  # 2 words -> 2 tokens

    @pytest.mark.asyncio
    async def test_single_request_with_token_type_ids(self, async_tokenizer):
        """Test tokenizing with token type IDs."""
        text = "hello world"
        result = await async_tokenizer.encode(text, return_token_type_ids=True)

        assert "input_ids" in result
        assert "token_type_ids" in result
        assert result["input_ids"] == [0, 1]
        assert result["token_type_ids"] == [0, 0]

    @pytest.mark.asyncio
    async def test_concurrent_requests_same_kwargs(self, async_tokenizer):
        """Test that concurrent requests with same kwargs get batched."""
        texts = ["hello world", "how are you", "fine thanks", "good morning"]

        # Start all requests concurrently
        tasks = [async_tokenizer.encode(text) for text in texts]
        results = await asyncio.gather(*tasks)

        # Verify all results
        assert len(results) == 4
        for i, result in enumerate(results):
            assert "input_ids" in result
            expected_tokens = list(range(len(texts[i].split())))
            assert result["input_ids"] == expected_tokens

    @pytest.mark.asyncio
    async def test_concurrent_requests_different_kwargs(self, async_tokenizer):
        """Test that requests with different kwargs are processed individually."""
        text1 = "hello world"
        text2 = "how are you"

        # One with token_type_ids, one without
        task1 = async_tokenizer.encode(text1, return_token_type_ids=True)
        task2 = async_tokenizer.encode(text2)

        result1, result2 = await asyncio.gather(task1, task2)

        # First result should have token_type_ids
        assert "input_ids" in result1
        assert "token_type_ids" in result1
        assert result1["input_ids"] == [0, 1]
        assert result1["token_type_ids"] == [0, 0]

        # Second result should not have token_type_ids
        assert "input_ids" in result2
        assert "token_type_ids" not in result2
        assert result2["input_ids"] == [0, 1, 2]

    @pytest.mark.asyncio
    async def test_batch_timeout(self, async_tokenizer):
        """Test that batching respects timeout."""
        # Send first request
        task1 = asyncio.create_task(async_tokenizer.encode("hello world"))

        # Wait longer than batch timeout
        await asyncio.sleep(0.02)  # Longer than 0.01s timeout

        # Send second request
        task2 = asyncio.create_task(async_tokenizer.encode("how are you"))

        results = await asyncio.gather(task1, task2)

        # Both should complete successfully
        assert len(results) == 2
        assert results[0]["input_ids"] == [0, 1]
        assert results[1]["input_ids"] == [0, 1, 2]

    @pytest.mark.asyncio
    async def test_max_batch_size_limit(self, async_tokenizer):
        """Test that batching respects max_batch_size."""
        # Send more requests than max_batch_size (4)
        texts = [f"text {i}" for i in range(6)]
        tasks = [async_tokenizer.encode(text) for text in texts]

        results = await asyncio.gather(*tasks)

        # All should complete successfully
        assert len(results) == 6
        for i, result in enumerate(results):
            assert "input_ids" in result
            assert result["input_ids"] == [0, 1]  # "text i" -> 2 tokens

    @pytest.mark.asyncio
    async def test_callable_interface(self, async_tokenizer):
        """Test that the tokenizer is callable."""
        text = "hello world"
        result = await async_tokenizer(text)

        assert "input_ids" in result
        assert result["input_ids"] == [0, 1]

    @pytest.mark.asyncio
    async def test_lazy_initialization(self, mock_tokenizer):
        """Test that initialization happens lazily."""
        tokenizer = AsyncDynamicbatchTokenizer(mock_tokenizer)

        # Should not be initialized yet
        assert not tokenizer._initialized

        # First encode should initialize
        await tokenizer.encode("hello")

        # Should now be initialized
        assert tokenizer._initialized

    @pytest.mark.asyncio
    async def test_error_handling_in_tokenizer(self, mock_tokenizer):
        """Test error handling when tokenizer fails."""

        # Create a new async tokenizer with a failing tokenizer
        def failing_tokenizer(*args, **kwargs):
            raise ValueError("Tokenizer error")

        async_tokenizer = AsyncDynamicbatchTokenizer(
            tokenizer=failing_tokenizer, max_batch_size=4, batch_wait_timeout_s=0.01
        )

        with pytest.raises(ValueError, match="Tokenizer error"):
            await async_tokenizer.encode("hello world")

    @pytest.mark.asyncio
    async def test_batch_processing_logs(self, async_tokenizer, caplog):
        """Test that batch processing logs are generated."""
        caplog.set_level(logging.DEBUG)

        # Send multiple requests to trigger batching
        tasks = [
            async_tokenizer.encode("hello world"),
            async_tokenizer.encode("how are you"),
        ]

        await asyncio.gather(*tasks)

        # Should have batch processing log
        assert any(
            "Processing dynamic batch of size" in record.message
            for record in caplog.records
        )

    @pytest.mark.asyncio
    async def test_empty_queue_immediate_processing(self, async_tokenizer):
        """Test that single requests are processed immediately when queue is empty."""
        start_time = time.time()
        result = await async_tokenizer.encode("hello world")
        end_time = time.time()

        # Should complete quickly (much less than batch timeout)
        assert end_time - start_time < 0.005  # 5ms should be plenty
        assert result["input_ids"] == [0, 1]

    @pytest.mark.asyncio
    async def test_real_tokenizer_integration(self):
        """Test with a real HuggingFace tokenizer."""
        try:
            # Use a small, fast tokenizer for testing
            real_tokenizer = AutoTokenizer.from_pretrained("gpt2")
            async_tokenizer = AsyncDynamicbatchTokenizer(
                tokenizer=real_tokenizer, max_batch_size=2, batch_wait_timeout_s=0.01
            )

            text = "Hello, world!"
            result = await async_tokenizer.encode(text)

            # Should get actual token IDs
            assert "input_ids" in result
            assert isinstance(result["input_ids"], list)
            assert len(result["input_ids"]) > 0
            assert all(isinstance(token_id, int) for token_id in result["input_ids"])

        except Exception as e:
            pytest.skip(f"Real tokenizer test skipped: {e}")

    @pytest.mark.asyncio
    async def test_concurrent_mixed_requests(self, async_tokenizer):
        """Test mixing single and batched requests."""
        # Start some requests
        task1 = asyncio.create_task(async_tokenizer.encode("hello"))
        task2 = asyncio.create_task(async_tokenizer.encode("world"))

        # Wait a bit
        await asyncio.sleep(0.005)

        # Start more requests
        task3 = asyncio.create_task(async_tokenizer.encode("how are"))
        task4 = asyncio.create_task(async_tokenizer.encode("you doing"))

        results = await asyncio.gather(task1, task2, task3, task4)

        # All should complete successfully
        assert len(results) == 4
        for result in results:
            assert "input_ids" in result
            assert isinstance(result["input_ids"], list)

    def test_cleanup_on_destruction(self, mock_tokenizer):
        """Test that resources are cleaned up properly."""
        tokenizer = AsyncDynamicbatchTokenizer(mock_tokenizer)

        # Mock the executor and task
        tokenizer._executor = Mock()
        tokenizer._batcher_task = Mock()
        tokenizer._batcher_task.done.return_value = False

        # Call destructor
        tokenizer.__del__()

        # Should cancel task and shutdown executor
        tokenizer._batcher_task.cancel.assert_called_once()
        tokenizer._executor.shutdown.assert_called_once_with(wait=False)


if __name__ == "__main__":
    pytest.main([__file__])
