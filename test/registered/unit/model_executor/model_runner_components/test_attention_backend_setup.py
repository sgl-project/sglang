import sys
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from sglang.srt.layers.attention.hybrid_attn_backend import HybridAttnBackend
from sglang.srt.model_executor.forward_batch_info import ForwardMode
from sglang.srt.model_executor.model_runner_components import (
    attention_backend_setup,
)
from sglang.srt.model_executor.model_runner_components.attention_backend_setup import (
    ResolvedAttentionBackendStr,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class _FakeBackend:
    def __init__(self, name):
        self.name = name
        # Real backends always carry this (AttentionBackend class attribute).
        self.needs_cpu_seq_lens = True


def test_split_full_attention_applies_model_wrapper_once():
    runner = SimpleNamespace(
        server_args=SimpleNamespace(speculative_attention_mode="prefill"),
        model_config=SimpleNamespace(context_len=2048),
        kv_cache_dtype=None,
        token_to_kv_pool=object(),
        req_to_token_pool=object(),
        init_new_workspace=None,
    )
    wrapper_inputs = []
    wrapped_backend = object()

    def wrap_once(model_runner, backend):
        assert model_runner is runner
        wrapper_inputs.append(backend)
        return wrapped_backend

    constructors = {
        "decode-test": lambda model_runner: _FakeBackend("decode"),
        "prefill-test": lambda model_runner: _FakeBackend("prefill"),
    }
    resolved = ResolvedAttentionBackendStr(decode="decode-test", prefill="prefill-test")

    with (
        patch.dict(attention_backend_setup.ATTENTION_BACKENDS, constructors),
        patch.object(
            attention_backend_setup,
            "attn_backend_wrapper",
            side_effect=wrap_once,
        ),
    ):
        result = attention_backend_setup._build_resolved_backend(
            model_runner=runner,
            resolved=resolved,
            init_new_workspace=True,
        )

    assert result is wrapped_backend
    assert len(wrapper_inputs) == 1
    split_backend = wrapper_inputs[0]
    assert isinstance(split_backend, HybridAttnBackend)
    assert split_backend.decode_backend.name == "decode"
    assert split_backend.prefill_backend.name == "prefill"
    assert runner.init_new_workspace is True


def _hybrid_backend(prefill, decode):
    runner = SimpleNamespace(
        server_args=SimpleNamespace(speculative_attention_mode="prefill"),
        model_config=SimpleNamespace(context_len=2048),
        kv_cache_dtype=None,
        token_to_kv_pool=object(),
        req_to_token_pool=object(),
    )
    return HybridAttnBackend(
        model_runner=runner, prefill_backend=prefill, decode_backend=decode
    )


class _RecordingBackend(_FakeBackend):
    def __init__(self, name):
        super().__init__(name)
        self.calls = []
        self.extend_dummy_seqs_capped_by_req_pool = False

    def forward(self, *args, **kwargs):
        self.calls.append((args, kwargs))
        return self.name


def test_forward_routes_linear_attention_args():
    """HybridAttnBackend.forward must keep the mixed_qkv/a/b linear-attention path.

    The class used to declare forward() twice; the second definition shadowed the
    first and silently dropped mixed_qkv/a/b.
    """
    prefill = _RecordingBackend("prefill")
    backend = _hybrid_backend(prefill, _RecordingBackend("decode"))

    result = backend.forward(
        layer="layer",
        forward_batch=SimpleNamespace(forward_mode=ForwardMode.EXTEND),
        mixed_qkv="mixed",
        a="a",
        b="b",
    )

    assert result == "prefill"
    _, kwargs = prefill.calls[0]
    assert kwargs["mixed_qkv"] == "mixed"
    assert kwargs["a"] == "a"
    assert kwargs["b"] == "b"


def test_forward_still_routes_regular_attention_args():
    prefill = _RecordingBackend("prefill")
    backend = _hybrid_backend(prefill, _RecordingBackend("decode"))

    backend.forward(
        "q",
        "k",
        "v",
        "layer",
        SimpleNamespace(forward_mode=ForwardMode.EXTEND),
    )

    args, kwargs = prefill.calls[0]
    assert args[:3] == ("q", "k", "v")
    assert "mixed_qkv" not in kwargs


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
