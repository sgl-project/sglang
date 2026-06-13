"""Chat Completions API E2E Tests - OpenAI Server Compatibility.

Tests for OpenAI-compatible chat completions API through the gateway.

Source: Migrated from e2e_grpc/basic/test_openai_server.py
"""

from __future__ import annotations

import json
import logging

import pytest

logger = logging.getLogger(__name__)


# =============================================================================
# Chat Completion Tests (Llama 8B)
# =============================================================================


@pytest.mark.model("llama-8b")
@pytest.mark.gateway(extra_args=["--history-backend", "memory"])
@pytest.mark.parametrize("setup_backend", ["grpc"], indirect=True)
class TestChatCompletion:
    """Tests for OpenAI-compatible chat completions API."""

    @pytest.mark.parametrize("logprobs", [None, 5])
    @pytest.mark.parametrize("parallel_sample_num", [1, 2])
    def test_chat_completion(self, setup_backend, logprobs, parallel_sample_num):
        """Test non-streaming chat completion with logprobs and parallel sampling."""
        _, model, client, gateway = setup_backend
        self._run_chat_completion(client, model, logprobs, parallel_sample_num)

    @pytest.mark.parametrize("logprobs", [None, 5])
    @pytest.mark.parametrize("parallel_sample_num", [1, 2])
    def test_chat_completion_stream(self, setup_backend, logprobs, parallel_sample_num):
        """Test streaming chat completion with logprobs and parallel sampling."""
        _, model, client, gateway = setup_backend
        self._run_chat_completion_stream(client, model, logprobs, parallel_sample_num)

    def test_regex(self, setup_backend):
        """Test structured output with regex constraint."""
        _, model, client, gateway = setup_backend

        regex = (
            r"""\{\n"""
            + r"""   "name": "[\w]+",\n"""
            + r"""   "population": [\d]+\n"""
            + r"""\}"""
        )

        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "You are a helpful AI assistant"},
                {"role": "user", "content": "Introduce the capital of France."},
            ],
            temperature=0,
            max_tokens=128,
            extra_body={"regex": regex},
        )
        text = response.choices[0].message.content

        try:
            js_obj = json.loads(text)
        except (TypeError, json.decoder.JSONDecodeError):
            raise
        assert isinstance(js_obj["name"], str)
        assert isinstance(js_obj["population"], int)

    def test_penalty(self, setup_backend):
        """Test frequency penalty parameter."""
        _, model, client, gateway = setup_backend

        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "You are a helpful AI assistant"},
                {"role": "user", "content": "Introduce the capital of France."},
            ],
            temperature=0,
            max_tokens=32,
            frequency_penalty=1.0,
        )
        text = response.choices[0].message.content
        assert isinstance(text, str)

    def test_response_prefill(self, setup_backend):
        """Test assistant message prefill with continue_final_message."""
        _, model, client, gateway = setup_backend

        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "You are a helpful AI assistant"},
                {
                    "role": "user",
                    "content": """
Extract the name, size, price, and color from this product description as a JSON object:

<description>
The SmartHome Mini is a compact smart home assistant available in black or white for only $49.99. At just 5 inches wide, it lets you control lights, thermostats, and other connected devices via voice or app—no matter where you place it in your home. This affordable little hub brings convenient hands-free control to your smart devices.
</description>
""",
                },
                {
                    "role": "assistant",
                    "content": "{\n",
                },
            ],
            temperature=0,
            extra_body={"continue_final_message": True},
        )

        assert (
            response.choices[0]
            .message.content.strip()
            .startswith('"name": "SmartHome Mini",')
        )

    def test_model_list(self, setup_backend):
        """Test listing available models."""
        _, model, client, gateway = setup_backend

        models = list(client.models.list().data)
        assert len(models) == 1

    @pytest.mark.skip(
        reason="Skipping retrieve model test as it is not supported by the router"
    )
    def test_retrieve_model(self, setup_backend):
        """Test retrieving a specific model."""
        import openai

        _, model, client, gateway = setup_backend

        retrieved_model = client.models.retrieve(model)
        assert retrieved_model.id == model
        assert retrieved_model.root == model

        with pytest.raises(openai.NotFoundError):
            client.models.retrieve("non-existent-model")

    # -------------------------------------------------------------------------
    # Helper methods
    # -------------------------------------------------------------------------

    def _run_chat_completion(self, client, model, logprobs, parallel_sample_num):
        """Run a non-streaming chat completion and verify response."""
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "You are a helpful AI assistant"},
                {
                    "role": "user",
                    "content": "What is the capital of France? Answer in a few words.",
                },
            ],
            temperature=0,
            logprobs=logprobs is not None and logprobs > 0,
            top_logprobs=logprobs,
            n=parallel_sample_num,
        )

        if logprobs:
            assert isinstance(
                response.choices[0].logprobs.content[0].top_logprobs[0].token, str
            )

            ret_num_top_logprobs = len(
                response.choices[0].logprobs.content[0].top_logprobs
            )
            assert (
                ret_num_top_logprobs == logprobs
            ), f"{ret_num_top_logprobs} vs {logprobs}"

        assert len(response.choices) == parallel_sample_num
        assert response.choices[0].message.role == "assistant"
        assert isinstance(response.choices[0].message.content, str)
        assert response.id
        assert response.created
        assert response.usage.prompt_tokens > 0
        assert response.usage.completion_tokens > 0
        assert response.usage.total_tokens > 0

    def _run_chat_completion_stream(
        self, client, model, logprobs, parallel_sample_num=1
    ):
        """Run a streaming chat completion and verify response chunks."""
        generator = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "You are a helpful AI assistant"},
                {"role": "user", "content": "What is the capital of France?"},
            ],
            temperature=0,
            logprobs=logprobs is not None and logprobs > 0,
            top_logprobs=logprobs,
            stream=True,
            stream_options={"include_usage": True},
            n=parallel_sample_num,
        )

        is_firsts = {}
        is_finished = {}
        finish_reason_counts = {}
        for response in generator:
            usage = response.usage
            if usage is not None:
                assert usage.prompt_tokens > 0, "usage.prompt_tokens was zero"
                assert usage.completion_tokens > 0, "usage.completion_tokens was zero"
                assert usage.total_tokens > 0, "usage.total_tokens was zero"
                continue

            index = response.choices[0].index
            finish_reason = response.choices[0].finish_reason
            if finish_reason is not None:
                is_finished[index] = True
                finish_reason_counts[index] = finish_reason_counts.get(index, 0) + 1

            data = response.choices[0].delta

            if is_firsts.get(index, True):
                assert (
                    data.role == "assistant"
                ), "data.role was not 'assistant' for first chunk"
                is_firsts[index] = False
                continue

            if logprobs and not is_finished.get(index, False):
                assert response.choices[0].logprobs, "logprobs was not returned"
                assert isinstance(
                    response.choices[0].logprobs.content[0].top_logprobs[0].token, str
                ), "top_logprobs token was not a string"
                assert isinstance(
                    response.choices[0].logprobs.content[0].top_logprobs, list
                ), "top_logprobs was not a list"
                ret_num_top_logprobs = len(
                    response.choices[0].logprobs.content[0].top_logprobs
                )
                assert (
                    ret_num_top_logprobs == logprobs
                ), f"{ret_num_top_logprobs} vs {logprobs}"

            assert (
                isinstance(data.content, str)
                or isinstance(data.reasoning_content, str)
                or (isinstance(data.tool_calls, list) and len(data.tool_calls) > 0)
                or response.choices[0].finish_reason
            )
            assert response.id
            assert response.created

        for index in range(parallel_sample_num):
            assert not is_firsts.get(
                index, True
            ), f"index {index} is not found in the response"

        for index in range(parallel_sample_num):
            assert (
                index in finish_reason_counts
            ), f"No finish_reason found for index {index}"
            assert finish_reason_counts[index] == 1, (
                f"Expected 1 finish_reason chunk for index {index}, "
                f"got {finish_reason_counts[index]}"
            )


# =============================================================================
# Chat Completion Tests (GPT-OSS)
#
# NOTE: Some tests are skipped because they don't work with OSS models:
# - test_regex: OSS models don't support regex constraints
# - test_penalty: OSS models don't support frequency_penalty
# - test_response_prefill: OSS models don't support continue_final_message
# =============================================================================


@pytest.mark.model("gpt-oss")
@pytest.mark.gateway(
    extra_args=["--reasoning-parser=gpt-oss", "--history-backend", "memory"]
)
class TestChatCompletionGptOss(TestChatCompletion):
    """Tests for chat completions API with GPT-OSS model.

    Inherits from TestChatCompletion and overrides tests that don't work
    with OSS models.
    """

    @pytest.mark.parametrize("logprobs", [None])  # No logprobs for OSS
    @pytest.mark.parametrize("parallel_sample_num", [1, 2])
    def test_chat_completion(self, setup_backend, logprobs, parallel_sample_num):
        """Test non-streaming chat completion with parallel sampling (no logprobs)."""
        super().test_chat_completion(setup_backend, logprobs, parallel_sample_num)

    @pytest.mark.parametrize("logprobs", [None])  # No logprobs for OSS
    @pytest.mark.parametrize("parallel_sample_num", [1, 2])
    def test_chat_completion_stream(self, setup_backend, logprobs, parallel_sample_num):
        """Test streaming chat completion with parallel sampling (no logprobs)."""
        super().test_chat_completion_stream(
            setup_backend, logprobs, parallel_sample_num
        )

    @pytest.mark.skip(reason="OSS models don't support regex constraints")
    def test_regex(self, setup_backend):
        pass

    @pytest.mark.skip(reason="OSS models don't support frequency_penalty")
    def test_penalty(self, setup_backend):
        pass

    @pytest.mark.skip(reason="OSS models don't support continue_final_message")
    def test_response_prefill(self, setup_backend):
        pass
