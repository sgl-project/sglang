"""Real-GPU e2e tests for `return_routed_experts` on sgl-model-gateway.

Verifies that the gateway correctly forwards `return_routed_experts: true`
to a live SGLang server (MoE model) and propagates the resulting
`sglext.routed_experts` (or `meta_info.routed_experts` for `/generate`)
back to the client. Mirrors the mock-based suites under
``tests/routing/pd_routing_test.rs`` and
``tests/api/api_endpoints_test.rs::sglang_extension_tests`` but against
real workers — catches regressions where the mock and the real SGLang
protocol have drifted (e.g. the typed `SglExt` envelope shape upstream
in ``python/sglang/srt/entrypoints/openai/protocol.py``).

Two scenarios:

* Unified router (1 worker, no merge step). Streaming + flag is a
  passthrough — the gateway does not gate it.

* PD router (1 prefill + 1 decode worker, gateway merges
  `prefill_bytes ++ decode_bytes[prefill_len..]` on the response).
  Streaming + flag is rejected with 400 because the SSE pipeline does
  not merge.

Model: ``deepseek-ai/DeepSeek-V2-Lite`` (resolved via the ``dsv2-lite``
spec in ``MODEL_SPECS``). Small MoE (64 routed experts, top-6) at
``tp=1``. Picking an MoE matters — `return_routed_experts` is a no-op
on dense models, which would weaken the assertion to "no 5xx".

GPU footprint:

* Unified: 1 GPU (``tp=1``).
* PD: 2 GPUs (1 prefill ``tp=1`` + 1 decode ``tp=1``).

Run locally::

    pytest e2e_test/router/test_routed_experts.py -v

These tests are decorated with ``@pytest.mark.e2e`` so they're filtered
out of unit-style runs.
"""

from __future__ import annotations

import base64
import json
import logging

import pytest
import requests

logger = logging.getLogger(__name__)

# API key is not validated by the gateway, but required for OpenAI-compatible
# headers (matches the convention from other e2e_test/chat_completions tests).
API_KEY = "not-used"

# gpt-oss requires its own reasoning parser; pulling it from the responses-API
# tests that already exercise this model.
_GPT_OSS_GATEWAY_ARGS = ["--reasoning-parser=gpt-oss"]


def _assert_valid_routed_experts_b64(b64: str, where: str) -> None:
    """Decode `b64` as base64 and assert it carries at least one byte of
    expert-id data. The exact byte-pattern depends on the model and the
    request — the gateway only forwards bytes — so we validate shape
    rather than value."""
    assert isinstance(b64, str), f"{where}: expected str, got {type(b64)}"
    decoded = base64.b64decode(b64, validate=True)
    assert len(decoded) > 0, f"{where}: routed_experts decoded to empty bytes"


# =============================================================================
# Unified router (regular HTTP, no PD merge)
# =============================================================================


@pytest.mark.e2e
@pytest.mark.model("dsv2-lite")
@pytest.mark.parametrize("setup_backend", ["http"], indirect=True)
class TestUnifiedRoutedExperts:
    """Unified-router e2e against a single real SGLang MoE worker."""

    def test_chat_completion_returns_routed_experts(self, setup_backend):
        """`return_routed_experts: true` on /v1/chat/completions surfaces
        the field in `sglext.routed_experts` (the typed `SglExt`
        envelope SGLang emits on `ChatCompletionResponse`)."""
        _, model, _, gateway = setup_backend

        resp = requests.post(
            f"{gateway.base_url}/v1/chat/completions",
            headers={"Authorization": f"Bearer {API_KEY}"},
            json={
                "model": model,
                "messages": [{"role": "user", "content": "Hello"}],
                "max_tokens": 4,
                "temperature": 0,
                "return_routed_experts": True,
                "stream": False,
            },
            timeout=120,
        )
        assert resp.status_code == 200, f"{resp.status_code}: {resp.text}"
        data = resp.json()
        sglext = data.get("sglext")
        assert sglext is not None, f"expected sglext envelope: {data}"
        _assert_valid_routed_experts_b64(
            sglext.get("routed_experts"), "sglext.routed_experts"
        )

    def test_generate_returns_routed_experts(self, setup_backend):
        """Same contract on /generate via `meta_info.routed_experts`
        (the SGLang-native envelope, not the OpenAI-shaped one)."""
        _, model, _, gateway = setup_backend

        resp = requests.post(
            f"{gateway.base_url}/generate",
            headers={"Authorization": f"Bearer {API_KEY}"},
            json={
                "model": model,
                "text": "Hello",
                "sampling_params": {"max_new_tokens": 4, "temperature": 0},
                "return_routed_experts": True,
                "stream": False,
            },
            timeout=120,
        )
        assert resp.status_code == 200, f"{resp.status_code}: {resp.text}"
        data = resp.json()
        meta_info = data.get("meta_info")
        assert meta_info is not None, f"expected meta_info envelope: {data}"
        _assert_valid_routed_experts_b64(
            meta_info.get("routed_experts"), "meta_info.routed_experts"
        )

    def test_streaming_chat_with_routed_experts_is_not_rejected(self, setup_backend):
        """Unified streaming + flag must NOT be rejected — the unified
        router has no merge step, so SGLang's per-chunk SglExt envelope
        flows through SSE proxying unchanged. (PD rejects this combo
        with 400; see TestPDRoutedExperts.)"""
        _, model, _, gateway = setup_backend

        with requests.post(
            f"{gateway.base_url}/v1/chat/completions",
            headers={"Authorization": f"Bearer {API_KEY}"},
            json={
                "model": model,
                "messages": [{"role": "user", "content": "Hello"}],
                "max_tokens": 4,
                "temperature": 0,
                "return_routed_experts": True,
                "stream": True,
            },
            stream=True,
            timeout=120,
        ) as resp:
            assert (
                resp.status_code == 200
            ), f"streaming + flag should be 200 on unified, got {resp.status_code}: {resp.text}"

            # Drain at least one SSE event to confirm the stream is alive.
            saw_event = False
            for raw_line in resp.iter_lines():
                if not raw_line:
                    continue
                line = raw_line.decode("utf-8", errors="replace")
                if line.startswith("data: "):
                    saw_event = True
                    payload = line[len("data: ") :].strip()
                    if payload == "[DONE]":
                        break
                    try:
                        json.loads(payload)
                    except json.JSONDecodeError:
                        pytest.fail(f"non-JSON SSE payload: {payload!r}")
                    break
            assert saw_event, "expected at least one SSE event from the unified stream"

    def test_chat_completion_rejects_invalid_extension_type(self, setup_backend):
        """Type-mismatched extension fields surface as 400 on the
        unified router (mirrors the PD contract)."""
        _, model, _, gateway = setup_backend

        resp = requests.post(
            f"{gateway.base_url}/v1/chat/completions",
            headers={"Authorization": f"Bearer {API_KEY}"},
            json={
                "model": model,
                "messages": [{"role": "user", "content": "Hello"}],
                "max_tokens": 4,
                "temperature": 0,
                "return_routed_experts": "yes",  # wrong type
                "stream": False,
            },
            timeout=30,
        )
        assert (
            resp.status_code == 400
        ), f"non-bool return_routed_experts should yield 400, got {resp.status_code}: {resp.text}"


# =============================================================================
# PD router (prefill + decode, merge on the response)
# =============================================================================


@pytest.mark.e2e
@pytest.mark.model("dsv2-lite")
@pytest.mark.workers(prefill=1, decode=1)
@pytest.mark.parametrize("setup_backend", ["pd"], indirect=True)
class TestPDRoutedExperts:
    """PD-router e2e: real prefill + decode SGLang MoE workers."""

    def test_chat_completion_merges_routed_experts(self, setup_backend):
        """Round-trip through `route_chat → execute_dual_dispatch →
        merge_prefill_json`. The merged `sglext.routed_experts` is
        `prefill_bytes ++ decode_bytes[prefill_len..]`; we can't compare
        against an oracle (would need to intercept both legs) but
        non-empty valid base64 + non-zero length proves the merge ran
        and produced output, which is the gateway-side contract."""
        _, model, _, gateway = setup_backend

        resp = requests.post(
            f"{gateway.base_url}/v1/chat/completions",
            headers={"Authorization": f"Bearer {API_KEY}"},
            json={
                "model": model,
                "messages": [{"role": "user", "content": "Hello"}],
                "max_tokens": 4,
                "temperature": 0,
                "return_routed_experts": True,
                "stream": False,
            },
            timeout=180,
        )
        assert resp.status_code == 200, f"{resp.status_code}: {resp.text}"
        data = resp.json()
        sglext = data.get("sglext")
        assert sglext is not None, f"expected sglext envelope: {data}"
        _assert_valid_routed_experts_b64(
            sglext.get("routed_experts"), "sglext.routed_experts (PD merged)"
        )

    def test_generate_merges_routed_experts(self, setup_backend):
        """Same shape on `/generate`, exercising the `meta_info` branch
        of `merge_prefill_json` (the chat test exercises the `sglext`
        branch)."""
        _, model, _, gateway = setup_backend

        resp = requests.post(
            f"{gateway.base_url}/generate",
            headers={"Authorization": f"Bearer {API_KEY}"},
            json={
                "model": model,
                "text": "Hello",
                "sampling_params": {"max_new_tokens": 4, "temperature": 0},
                "return_routed_experts": True,
                "stream": False,
            },
            timeout=180,
        )
        assert resp.status_code == 200, f"{resp.status_code}: {resp.text}"
        data = resp.json()
        meta_info = data.get("meta_info")
        assert meta_info is not None, f"expected meta_info envelope: {data}"
        _assert_valid_routed_experts_b64(
            meta_info.get("routed_experts"), "meta_info.routed_experts (PD merged)"
        )

    def test_streaming_chat_with_routed_experts_returns_400(self, setup_backend):
        """The SSE pipeline doesn't merge prefill/decode bytes, so PD
        rejects this combo up front rather than silently returning
        decode-only experts."""
        _, model, _, gateway = setup_backend

        resp = requests.post(
            f"{gateway.base_url}/v1/chat/completions",
            headers={"Authorization": f"Bearer {API_KEY}"},
            json={
                "model": model,
                "messages": [{"role": "user", "content": "Hello"}],
                "max_tokens": 4,
                "temperature": 0,
                "return_routed_experts": True,
                "stream": True,
            },
            timeout=30,
        )
        assert (
            resp.status_code == 400
        ), f"PD streaming + flag should yield 400, got {resp.status_code}: {resp.text}"

    def test_chat_completion_rejects_invalid_extension_type(self, setup_backend):
        """Type-mismatched extension fields are surfaced as 400 by the
        PD router. This complements the unified-side reject test —
        each router has its own `parse_sglang_extensions` call
        (`pd_router.rs::route_chat` vs `router.rs::route_chat`)."""
        _, model, _, gateway = setup_backend

        resp = requests.post(
            f"{gateway.base_url}/v1/chat/completions",
            headers={"Authorization": f"Bearer {API_KEY}"},
            json={
                "model": model,
                "messages": [{"role": "user", "content": "Hello"}],
                "max_tokens": 4,
                "temperature": 0,
                "return_routed_experts": "yes",  # wrong type
                "stream": False,
            },
            timeout=30,
        )
        assert (
            resp.status_code == 400
        ), f"PD non-bool return_routed_experts should yield 400, got {resp.status_code}: {resp.text}"
