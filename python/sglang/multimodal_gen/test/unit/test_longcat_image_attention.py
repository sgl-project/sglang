"""Unit tests for LongCat-Image attention backend selection."""

from types import SimpleNamespace
from unittest.mock import patch

from torch.nn.attention import SDPBackend

from sglang.multimodal_gen.runtime.layers.attention.backends.sdpa import SDPAImpl
from sglang.multimodal_gen.runtime.models.dits.longcat_image import (
    _longcat_attention_backend_kwargs,
)


@patch(
    "sglang.multimodal_gen.runtime.models.dits.longcat_image.current_platform.is_sm120",
    return_value=True,
)
def test_longcat_enables_cudnn_sdpa_fallback_chain_on_sm120(_mock_is_sm120):
    assert _longcat_attention_backend_kwargs() == {"prefer_cudnn_sdp": True}


@patch(
    "sglang.multimodal_gen.runtime.models.dits.longcat_image.current_platform.is_sm120",
    return_value=False,
)
def test_longcat_keeps_existing_attention_selection_elsewhere(_mock_is_sm120):
    assert _longcat_attention_backend_kwargs() == {}


@patch("sglang.multimodal_gen.runtime.layers.attention.backends.sdpa.sdpa_kernel")
def test_prefer_cudnn_marks_the_sdpa_backend_order_as_priority(mock_sdpa_kernel):
    query = SimpleNamespace(device=SimpleNamespace(type="cuda"))
    impl = SDPAImpl(
        num_heads=24,
        head_size=128,
        causal=False,
        softmax_scale=128**-0.5,
        prefer_cudnn_sdp=True,
    )

    impl._sdpa_context(query)

    backends = mock_sdpa_kernel.call_args.args[0]
    assert backends[0] == SDPBackend.CUDNN_ATTENTION
    assert mock_sdpa_kernel.call_args.kwargs == {"set_priority": True}


@patch("sglang.multimodal_gen.runtime.layers.attention.backends.sdpa.sdpa_kernel")
def test_allow_cudnn_keeps_the_existing_non_priority_behavior(mock_sdpa_kernel):
    query = SimpleNamespace(device=SimpleNamespace(type="cuda"))
    impl = SDPAImpl(
        num_heads=24,
        head_size=128,
        causal=False,
        softmax_scale=128**-0.5,
        allow_cudnn_sdp=True,
    )

    impl._sdpa_context(query)

    assert mock_sdpa_kernel.call_args.kwargs == {"set_priority": False}
