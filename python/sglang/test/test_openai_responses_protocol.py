"""Regression tests for OpenAI Responses request cache identity."""

from sglang.srt.entrypoints.openai.protocol import ResponsesRequest


def test_prompt_cache_key_is_preserved_when_cache_salt_is_unset():
    request = ResponsesRequest(
        model="test-model",
        input="hello",
        prompt_cache_key="tenant-a",
    )

    assert request.prompt_cache_key == "tenant-a"
    assert request.cache_salt is None


def test_explicit_cache_salt_takes_precedence_over_prompt_cache_key():
    request = ResponsesRequest(
        model="test-model",
        input="hello",
        cache_salt="internal-namespace",
        prompt_cache_key="tenant-a",
    )

    effective_cache_salt = request.cache_salt or request.prompt_cache_key
    assert effective_cache_salt == "internal-namespace"
