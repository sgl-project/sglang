# SPDX-License-Identifier: Apache-2.0
"""CPU contract tests for the MindIE-SD FIA attention adapter."""

import importlib
import sys
from types import ModuleType
from unittest.mock import Mock

import pytest
import torch
import torch.nn.functional as F

from sglang.multimodal_gen.runtime.layers.attention.backends.attention_backend import (
    AttentionRequirements,
)
from sglang.multimodal_gen.runtime.platforms import AttentionBackendEnum

_MODULE = "sglang.multimodal_gen.runtime.layers.attention.backends.fia_attn"


@pytest.fixture
def fia(monkeypatch):
    # The adapter is testable on CPU without loading MindIE-SD's NPU extension.
    mindiesd = ModuleType("mindiesd")
    mindiesd.quant_attention = Mock()
    monkeypatch.setitem(sys.modules, "mindiesd", mindiesd)
    module = importlib.import_module(_MODULE)
    monkeypatch.setattr(module, "quant_attention", mindiesd.quant_attention)
    return module, mindiesd.quant_attention


def _impl(module, **kwargs):
    return module.FIAAttentionImpl(
        num_heads=2, head_size=8, causal=False, softmax_scale=0.37, **kwargs
    )


def _reference(query, key, value, **kwargs):
    return F.scaled_dot_product_attention(
        query.transpose(1, 2),
        key.transpose(1, 2),
        value.transpose(1, 2),
        scale=kwargs["scale"],
        is_causal=kwargs["next_tokens"] == 0,
    ).transpose(1, 2)


def test_dense_calls_quant_attention_with_fia_options(fia):
    module, quant_attention = fia
    query = torch.randn(1, 7, 2, 8, dtype=torch.bfloat16)
    key = torch.randn(1, 9, 2, 8, dtype=torch.bfloat16)
    value = torch.randn_like(key)
    expected = torch.randn_like(query)
    quant_attention.return_value = expected

    actual = _impl(module).forward(query, key, value, None)

    assert actual is expected
    args, options = quant_attention.call_args
    assert args[0] is query and args[1] is key and args[2] is value
    assert options == {
        "precision": "fp8",
        "layout": "BSND",
        "scale": 0.37,
        "fp8_fa_mode": "HIGH_PRECISION",
        "pre_tokens": 2147483647,
        "next_tokens": 2147483647,
    }


def test_dense_batches_are_independent_batch_one_calls(fia):
    module, quant_attention = fia
    quant_attention.side_effect = _reference
    torch.manual_seed(1)
    query, key, value = (torch.randn(3, 7, 2, 8) for _ in range(3))

    actual = _impl(module).forward(query, key, value, None)

    expected = _reference(query, key, value, scale=0.37, next_tokens=2147483647)
    torch.testing.assert_close(actual, expected)
    assert quant_attention.call_count == 3
    assert all(call.args[0].shape[0] == 1 for call in quant_attention.call_args_list)


def test_two_real_packed_segments_are_isolated(fia):
    module, quant_attention = fia
    quant_attention.side_effect = _reference
    query = torch.zeros(7, 2, 8)
    key = torch.zeros_like(query)
    value = torch.cat((torch.ones(3, 2, 8), torch.full((4, 2, 8), 23.0)))

    actual = _impl(module, packed_trailing_padding=True).forward_varlen(
        query,
        key,
        value,
        cu_seqlens=torch.tensor([0, 3, 7], dtype=torch.int32),
        cu_seqlens_host=(0, 3, 7),
        max_seqlen=4,
    )

    # [0, 3, 7] is two real sequences here because max_seqlen != bounds[1].
    torch.testing.assert_close(actual[:3], torch.ones(3, 2, 8))
    torch.testing.assert_close(actual[3:], torch.full((4, 2, 8), 23.0))
    assert [call.args[0].shape[1] for call in quant_attention.call_args_list] == [3, 4]


def test_minimax_trailing_padding_is_not_executed(fia):
    module, quant_attention = fia
    quant_attention.side_effect = _reference
    query = torch.zeros(7, 2, 8)
    key = torch.zeros_like(query)
    value = torch.cat((torch.ones(3, 2, 8), torch.full((4, 2, 8), 23.0)))

    actual = _impl(module, packed_trailing_padding=True).forward_varlen(
        query,
        key,
        value,
        cu_seqlens=torch.tensor([0, 3, 7], dtype=torch.int32),
        cu_seqlens_host=(0, 3, 7),
        max_seqlen=3,
    )

    torch.testing.assert_close(actual[:3], torch.ones(3, 2, 8))
    torch.testing.assert_close(actual[3:], torch.zeros(4, 2, 8))
    quant_attention.assert_called_once()
    assert quant_attention.call_args.args[0].shape == (1, 3, 2, 8)


def test_packed_host_bounds_avoid_device_read_and_skip_empty_segments(fia):
    module, quant_attention = fia
    query = torch.randn(5, 2, 8)
    quant_attention.return_value = query.unsqueeze(0)
    device_bounds = Mock()
    device_bounds.tolist.side_effect = AssertionError("must use supplied host bounds")

    actual = _impl(module).forward_varlen(
        query,
        query,
        query,
        cu_seqlens=device_bounds,
        cu_seqlens_host=(0, 0, 5, 5),
        max_seqlen=5,
    )

    torch.testing.assert_close(actual, query)
    quant_attention.assert_called_once()


@pytest.mark.parametrize("bounds", [(1, 5), (0, 4), (0, 6, 5), (0,)])
def test_invalid_packed_bounds_fail_before_kernel(fia, bounds):
    module, quant_attention = fia
    query = torch.randn(5, 2, 8)
    with pytest.raises(ValueError, match="monotonically cover"):
        _impl(module).forward_varlen(
            query,
            query,
            query,
            cu_seqlens=torch.tensor(bounds),
            max_seqlen=5,
        )
    quant_attention.assert_not_called()


def test_native_failure_propagates_without_fallback(fia):
    module, quant_attention = fia
    quant_attention.side_effect = RuntimeError("FIA native error")
    query = torch.randn(1, 7, 2, 8)
    with pytest.raises(RuntimeError, match="FIA native error"):
        _impl(module).forward(query, query, query, None)
    quant_attention.assert_called_once()


def test_causal_attention_is_rejected_before_kernel(fia):
    module, quant_attention = fia
    with pytest.raises(ValueError, match="does not support causal attention"):
        module.FIAAttentionImpl(2, 128, True, 0.37)
    quant_attention.assert_not_called()


def test_npu_registry_and_minimax_packed_capability(fia):
    module, _ = fia
    from sglang.multimodal_gen.runtime.platforms.npu import NPUPlatformBase

    backend = module.FIAAttentionBackend
    assert backend.get_enum() is AttentionBackendEnum.FIA_ATTN
    assert backend.get_supported_head_sizes() == [64, 128]
    assert not AttentionBackendEnum.FIA_ATTN.is_sparse
    assert (
        backend.unsupported_requirements(AttentionRequirements(packed_varlen=True))
        == ()
    )
    assert (
        NPUPlatformBase.get_attn_backend_cls_str(
            AttentionBackendEnum.FIA_ATTN, 128, torch.bfloat16
        )
        == f"{_MODULE}.FIAAttentionBackend"
    )
