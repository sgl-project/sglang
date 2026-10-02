import sys
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest

from sglang.srt.layers.attention.base_attn_backend import AttentionBackend
from sglang.srt.layers.attention.flashattention_dense_backend import (
    FlashAttentionDenseBackend,
)
from sglang.srt.layers.attention.hybrid_attn_backend import HybridAttnBackend
from sglang.srt.layers.attention.tbo_backend import TboAttnBackend
from sglang.srt.model_executor.forward_batch_info import ForwardMode
from sglang.srt.model_executor.model_runner_components import (
    attention_backend_setup,
)
from sglang.srt.model_executor.model_runner_components.attention_backend_setup import (
    ResolvedAttentionBackendStr,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=12, suite="base-a-test-cpu")


@pytest.mark.parametrize("wrapper_type", [HybridAttnBackend, TboAttnBackend])
def test_prefill_wrappers_forward_graph_capabilities(wrapper_type):
    wrapper = wrapper_type.__new__(wrapper_type)
    child = Mock(spec=AttentionBackend)
    child.full_cuda_graph_uses_chunked_prefix = False
    child.get_prefill_cuda_graph_max_query_len.return_value = 256
    if wrapper_type is HybridAttnBackend:
        wrapper.prefill_backend = child
        wrapper.decode_backend = Mock(spec=AttentionBackend)
    else:
        wrapper.primary = child
    assert wrapper.full_cuda_graph_uses_chunked_prefix is False
    assert wrapper.get_prefill_cuda_graph_max_query_len(1024, 8) == 256
    assert (
        wrapper.get_cuda_graph_variants(None, ForwardMode.DLLM_EXTEND, 256)
        is child.get_cuda_graph_variants.return_value
    )


def test_dense_adapter_keeps_quantized_cache_on_native_path():
    import torch

    backend = FlashAttentionDenseBackend.__new__(FlashAttentionDenseBackend)
    backend.extend_attention_fwd = Mock(return_value="native")
    q = torch.empty(2, 16, 256, dtype=torch.bfloat16)
    k = torch.empty(2, 2, 256, dtype=q.dtype)
    result = backend._forward_extend_kernel(
        SimpleNamespace(sliding_window_size=None),
        q,
        k,
        k,
        torch.empty_like(q),
        k.to(torch.float8_e4m3fn),
        k,
        None,
        None,
        None,
        None,
        False,
        None,
        2,
        1.0,
        1.0,
    )
    assert result == "native"


class _FakeBackend:
    def __init__(self, name):
        self.name = name
        # Real backends always carry this (AttentionBackend class attribute).
        self.needs_cpu_seq_lens = True


def test_split_full_attention_applies_model_wrapper_once():
    # The hybrid backend takes the speculative attention mode from the
    # published configuration.
    from sglang.srt.runtime_context import get_context

    override = get_context().override_server_args(speculative_attention_mode="prefill")
    override.install()
    try:
        runner = SimpleNamespace(
            server_args=SimpleNamespace(speculative_attention_mode="prefill"),
            model_config=SimpleNamespace(context_len=2048),
            kv_cache_dtype=None,
            token_to_kv_pool=object(),
            req_to_token_pool=object(),
            kv_index_translator=None,
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
        resolved = ResolvedAttentionBackendStr(
            decode="decode-test", prefill="prefill-test"
        )

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
    finally:
        override.restore()


def test_equal_resolved_backends_ignore_stale_global_backend():
    runner = SimpleNamespace(
        server_args=SimpleNamespace(
            attention_backend="global-test",
            speculative_attention_mode="prefill",
        ),
        kv_cache_dtype=None,
        token_to_kv_pool=object(),
        req_to_token_pool=object(),
        init_new_workspace=None,
    )
    constructors = {
        "global-test": lambda _runner: _FakeBackend("global"),
        "resolved-test": lambda _runner: _FakeBackend("resolved"),
    }
    resolved = ResolvedAttentionBackendStr(
        decode="resolved-test",
        prefill="resolved-test",
    )

    with (
        patch.dict(attention_backend_setup.ATTENTION_BACKENDS, constructors),
        patch.object(
            attention_backend_setup,
            "attn_backend_wrapper",
            side_effect=lambda _runner, backend: backend,
        ),
    ):
        result = attention_backend_setup._build_resolved_backend(
            model_runner=runner,
            resolved=resolved,
            init_new_workspace=False,
        )

    assert result.name == "resolved"


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
