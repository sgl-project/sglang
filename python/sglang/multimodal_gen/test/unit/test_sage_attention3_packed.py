# SPDX-License-Identifier: Apache-2.0
"""Packed-varlen contracts for the optional SageAttention3 backend."""

import importlib
import sys
import types
from unittest.mock import Mock, patch

import pytest
import torch


@pytest.fixture
def sage_attention3_module():
    module_name = "sglang.multimodal_gen.runtime.layers.attention.backends.sage_attn3"
    fake_sageattn3 = types.ModuleType("sageattn3")
    fake_sageattn3.sageattn3_blackwell = Mock()
    original_module = sys.modules.pop(module_name, None)
    try:
        with patch.dict(sys.modules, {"sageattn3": fake_sageattn3}):
            yield importlib.import_module(module_name)
    finally:
        sys.modules.pop(module_name, None)
        if original_module is not None:
            sys.modules[module_name] = original_module


def _attention_impl(module):
    return module.SageAttention3Impl(
        num_heads=2,
        head_size=8,
        causal=False,
        softmax_scale=8**-0.5,
    )


def test_h3_packed_attention_skips_and_zeroes_trailing_padding(
    sage_attention3_module,
):
    query = torch.randn(2, 8, 8).transpose(0, 1)
    key = torch.randn_like(query)
    value = torch.randn_like(query)
    attention = _attention_impl(sage_attention3_module)
    attention.forward = Mock(return_value=torch.full((1, 5, 2, 8), 3.0))

    output = attention._sage_packed(
        query,
        key,
        value,
        bounds=(0, 5, 8),
        max_seqlen=5,
    )

    assert sage_attention3_module.SageAttention3Backend.supports_packed_varlen()
    torch.testing.assert_close(output[:5], torch.full_like(output[:5], 3.0))
    torch.testing.assert_close(output[5:], torch.zeros_like(output[5:]))
    attention.forward.assert_called_once()
    assert attention.forward.call_args.args[0].shape == (1, 5, 2, 8)


def test_packed_attention_dispatches_each_nonempty_interval(
    sage_attention3_module,
):
    query = torch.randn(6, 2, 8)
    key = torch.randn_like(query)
    value = torch.randn_like(query)
    attention = _attention_impl(sage_attention3_module)
    attention.forward = Mock(side_effect=lambda q, *_args, **_kwargs: q + 1)

    output = attention._sage_packed(
        query,
        key,
        value,
        bounds=(0, 2, 2, 6),
        max_seqlen=4,
    )

    torch.testing.assert_close(output, query + 1)
    assert attention.forward.call_count == 2
    assert attention.forward.call_args_list[0].args[0].shape == (1, 2, 2, 8)
    assert attention.forward.call_args_list[1].args[0].shape == (1, 4, 2, 8)


def test_varlen_uses_host_boundaries_and_preserves_input_layout(
    sage_attention3_module,
):
    class DeviceBoundaries:
        def tolist(self):
            raise AssertionError("host boundaries must avoid a device synchronization")

    query = torch.randn(2, 6, 8).transpose(0, 1)
    key = torch.randn_like(query)
    value = torch.randn_like(query)
    attention = _attention_impl(sage_attention3_module)
    expected = object()
    attention._sage_packed = Mock(return_value=expected)

    output = attention.forward_varlen(
        query,
        key,
        value,
        cu_seqlens=DeviceBoundaries(),
        cu_seqlens_host=(0, 6),
        max_seqlen=6,
    )

    assert output is expected
    attention._sage_packed.assert_called_once()
    call = attention._sage_packed.call_args
    assert call.kwargs == {"bounds": (0, 6), "max_seqlen": 6}
    assert call.args[0] is query
    assert call.args[1] is key
    assert call.args[2] is value
    assert not query.is_contiguous()


def test_attention_does_not_expose_sage_attention3_key_mutation(
    sage_attention3_module,
):
    query = torch.randn(1, 6, 2, 8)
    key = torch.randn_like(query)
    value = torch.randn_like(query)
    original_key = key.clone()
    attention = _attention_impl(sage_attention3_module)

    def mutate_key(q, k, _v, **_kwargs):
        k.zero_()
        return torch.ones_like(q)

    sage_attention3_module.sageattn3_blackwell.side_effect = mutate_key
    output = attention.forward(query, key, value, None)

    torch.testing.assert_close(key, original_key)
    torch.testing.assert_close(output, torch.ones_like(output))
