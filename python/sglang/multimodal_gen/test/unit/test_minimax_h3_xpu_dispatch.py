# SPDX-License-Identifier: Apache-2.0
"""Exercise XPU buffer contracts using CPU storage and mocked native kernels."""

from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch
import torch.nn.functional as F

from sglang.multimodal_gen.configs.task_type import DataType
from sglang.multimodal_gen.runtime.models.dits import minimax_h3 as h3
from sglang.multimodal_gen.runtime.platforms import AttentionBackendEnum


class _XpuTensor(torch.Tensor):
    @property
    def device(self):
        return torch.device("xpu", 0)


def _tensor(*shape, dtype=torch.bfloat16):
    return torch.ones(*shape, dtype=dtype).as_subclass(_XpuTensor)


@pytest.mark.parametrize("allow_inplace", [False, True])
def test_gate_preserves_requested_buffer_ownership(monkeypatch, allow_inplace):
    x, other = _tensor(4, 16), _tensor(4, 16)
    gate = _tensor(3, 48)[:, 16:32]
    indices = _tensor(4, dtype=torch.int64)
    original = x.clone()

    def kernel(out, gate, other, indices):
        out.add_(gate.index_select(0, indices) * other)
        return out

    native = Mock(side_effect=kernel)
    monkeypatch.setattr(h3, "_xpu_indexed_gate", native)
    out = h3._modulate_gate(
        x, gate, other, indices, dtype=torch.bfloat16, allow_inplace=allow_inplace
    )
    native.assert_called_once()
    assert (out is x) == allow_inplace
    if not allow_inplace:
        assert torch.equal(x, original)
    assert torch.equal(out, original + gate.index_select(0, indices) * other)


def test_qk_unequal_epsilon_keeps_unfused_path(monkeypatch):
    q, k = _tensor(4, 2, 128), _tensor(4, 2, 128)
    q_norm = SimpleNamespace(weight=_tensor(128), eps=1e-6)
    k_norm = SimpleNamespace(weight=_tensor(128), eps=1e-4)
    native = Mock(side_effect=AssertionError("unequal eps must not fuse"))
    monkeypatch.setattr(h3, "_xpu_qknorm_rope", native, raising=False)
    monkeypatch.setattr(h3, "_apply_qk_norm", lambda *args: (q, k))
    monkeypatch.setattr(h3, "_apply_rope_qk", lambda *args, **kwargs: (q, k))
    out = h3._apply_qk_norm_rope(
        q, k, q_norm, k_norm, 128, _tensor(4, 96), _tensor(4, dtype=torch.int64), None
    )
    assert out[0] is q and out[1] is k
    native.assert_not_called()


def test_silu_output_does_not_alias_packed_input(monkeypatch):
    hidden = _tensor(4, 32)
    result = _tensor(4, 16)
    native = Mock(return_value=result)
    monkeypatch.setattr(h3, "_xpu_silu_and_mul", native, raising=False)
    assert h3._silu_mul(hidden, reuse_input=True) is result
    native.assert_called_once_with(hidden)
    assert result.data_ptr() != hidden.data_ptr()


def test_qk_fusion_updates_packed_views_in_place(monkeypatch):
    q, k = _tensor(4, 3, 2, 128).unbind(1)[:2]
    norm = SimpleNamespace(weight=_tensor(128), eps=1e-6)
    cache = _tensor(4, 96)
    positions = _tensor(4, dtype=torch.int64)
    native = Mock()
    monkeypatch.setattr(h3, "_xpu_qknorm_rope", native, raising=False)
    out = h3._apply_qk_norm_rope(q, k, norm, norm, 128, cache, positions, None)
    assert out[0] is q and out[1] is k
    native.assert_called_once_with(
        q,
        k,
        norm.weight,
        norm.weight,
        cache,
        positions,
        is_neox=True,
        eps=1e-6,
        head_dim=128,
        rope_dim=96,
    )


def test_norm_fusion_uses_dtype_default_epsilon(monkeypatch):
    x = _tensor(4, 16)
    norm = SimpleNamespace(weight=_tensor(16), eps=None)
    shift, scale = _tensor(3, 16), _tensor(3, 16)
    indices = _tensor(4, dtype=torch.int64)
    native = Mock(return_value=x.clone())
    monkeypatch.setattr(h3, "_xpu_rmsnorm_indexed_adaln", native)
    monkeypatch.setattr(
        h3, "can_use_h3_rmsnorm_indexed_adaln", lambda *args: True, raising=False
    )
    out = h3._modulate_rmsnorm_scale_shift(
        x, norm, shift, scale, indices, dtype=torch.bfloat16
    )
    assert out is native.return_value
    native.assert_called_once_with(
        x, norm.weight, shift, scale, indices, torch.finfo(torch.bfloat16).eps
    )


@pytest.fixture
def dense_attention(monkeypatch):
    attention = SimpleNamespace(
        _attention_backend_enum=AttentionBackendEnum.FA,
        _attention_impl=SimpleNamespace(causal=False, forward_varlen=Mock()),
        prefix="blocks.0.attn",
        softmax_scale=128**-0.5,
    )
    sampling = SimpleNamespace(data_type=DataType.IMAGE)
    context = SimpleNamespace(forward_batch=SimpleNamespace(sampling_params=sampling))
    monkeypatch.setattr(h3, "get_forward_context_or_none", lambda: context)
    native = Mock(
        side_effect=lambda q, k, v: F.scaled_dot_product_attention(
            q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
        ).transpose(1, 2)
    )
    loader = Mock(return_value=native)
    monkeypatch.setattr(h3, "_minimax_h3_xpu_sdp", loader)
    return attention, sampling, native, loader


@pytest.mark.parametrize("bounds", [(0, 3, 8), (0, 3, 3, 8), (0, 8, 8)])
def test_dense_attention_keeps_packed_documents_isolated(dense_attention, bounds):
    attention, _, native, _ = dense_attention
    q, k, v = _tensor(8, 3, 2, 128).unbind(1)
    v[:3].fill_(1)
    v[3:].fill_(100)
    out = h3._minimax_h3_xpu_dense_attention(attention, q, k, v, bounds)
    expected = torch.empty_like(q)
    for start, stop in zip(bounds, bounds[1:]):
        if stop > start:
            expected[start:stop] = v[start:stop].mean(0)
    torch.testing.assert_close(out, expected)
    assert native.call_count == sum(b > a for a, b in zip(bounds, bounds[1:]))


@pytest.mark.parametrize("bounds", [None, (), (0,), (1, 4), (0, 3), (0, 5, 4)])
def test_dense_attention_invalid_bounds_keeps_fa(monkeypatch, dense_attention, bounds):
    attention, _, native, loader = dense_attention
    q = _tensor(4, 2, 128)
    monkeypatch.setattr(
        torch.Tensor, "tolist", Mock(side_effect=AssertionError("host sync"))
    )
    assert h3._minimax_h3_xpu_dense_attention(attention, q, q, q, bounds) is None
    loader.assert_not_called()
    native.assert_not_called()


@pytest.mark.parametrize(
    "unsupported",
    [
        "cpu",
        "dtype",
        "shape",
        "device",
        "scale",
        "causal",
        "backend",
        "refiner",
        "video",
        "no_context",
        "compile",
    ],
)
def test_dense_attention_preserves_other_paths(
    monkeypatch, dense_attention, unsupported
):
    attention, sampling, native, loader = dense_attention
    q = _tensor(4, 2, 128)
    k = v = q
    if unsupported == "cpu":
        q = k = v = torch.ones(4, 2, 128, dtype=torch.bfloat16)
    elif unsupported == "dtype":
        q = k = v = _tensor(4, 2, 128, dtype=torch.float32)
    elif unsupported == "shape":
        k = _tensor(3, 2, 128)
    elif unsupported == "device":
        k = torch.ones(4, 2, 128, dtype=torch.bfloat16)
    elif unsupported == "scale":
        attention.softmax_scale = 1.0
    elif unsupported == "causal":
        attention._attention_impl.causal = True
    elif unsupported == "backend":
        attention._attention_backend_enum = AttentionBackendEnum.TORCH_SDPA
    elif unsupported == "refiner":
        attention.prefix = "token_refiner.blocks.0.attn"
    elif unsupported == "video":
        sampling.data_type = DataType.VIDEO
    elif unsupported == "no_context":
        monkeypatch.setattr(h3, "get_forward_context_or_none", lambda: None)
    elif unsupported == "compile":
        monkeypatch.setattr(torch.compiler, "is_compiling", lambda: True)
    assert h3._minimax_h3_xpu_dense_attention(attention, q, k, v, (0, 4)) is None
    loader.assert_not_called()
    native.assert_not_called()


@pytest.mark.parametrize("available", [False, True])
def test_dense_core_selects_native_kernel_or_original_fa(
    monkeypatch, dense_attention, available
):
    attention, _, native, loader = dense_attention
    loader.return_value = native if available else None
    monkeypatch.setattr(h3, "_minimax_h3_bind_attention_backend", lambda *args: None)
    q = _tensor(4, 2, 128)
    result = h3._minimax_h3_attention_core_impl(
        attention,
        q,
        q,
        q,
        cu_seqlens=torch.tensor([0, 4]),
        cu_seqlens_host=(0, 4),
        max_seqlen=4,
        ulysses_active=False,
    )
    if available:
        torch.testing.assert_close(result, q)
        native.assert_called_once()
        attention._attention_impl.forward_varlen.assert_not_called()
    else:
        assert result is attention._attention_impl.forward_varlen.return_value
        attention._attention_impl.forward_varlen.assert_called_once()
        native.assert_not_called()


@pytest.mark.parametrize("ring,ulysses", [(True, False), (False, True)])
def test_parallel_attention_keeps_existing_path(
    monkeypatch, dense_attention, ring, ulysses
):
    attention, _, native, loader = dense_attention
    q = _tensor(4, 2, 128)
    monkeypatch.setattr(h3, "_minimax_h3_bind_attention_backend", lambda *args: None)
    monkeypatch.setattr(
        h3, "_minimax_h3_pipelined_dense_attention", lambda *args, **kw: None
    )
    monkeypatch.setattr(
        h3, "_minimax_h3_collective_pipelined_attention", lambda *args, **kw: None
    )
    monkeypatch.setattr(h3, "_usp_input_all_to_all_packed_qkv", lambda *args: args)
    monkeypatch.setattr(h3, "_usp_output_all_to_all", lambda out, **kw: out)
    monkeypatch.setattr(h3, "get_ring_ctx", lambda: (2, 0))
    ring_call = Mock(return_value=q)
    monkeypatch.setattr(h3, "_ring_attention_varlen", ring_call)
    attention._attention_impl.forward_varlen.return_value = q
    result = h3._minimax_h3_attention_core_impl(
        attention,
        q,
        q,
        q,
        cu_seqlens=torch.tensor([0, 4]),
        cu_seqlens_host=(0, 4),
        max_seqlen=4,
        ulysses_active=ulysses,
        ring_active=ring,
    )
    torch.testing.assert_close(result, q)
    assert ring_call.call_count == int(ring)
    assert attention._attention_impl.forward_varlen.call_count == int(not ring)
    loader.assert_not_called()
    native.assert_not_called()


def test_dense_kernel_failure_is_not_silently_hidden(dense_attention):
    attention, _, native, _ = dense_attention
    native.side_effect = RuntimeError("kernel execution failed")
    q = _tensor(4, 2, 128)
    with pytest.raises(RuntimeError, match="kernel execution failed"):
        h3._minimax_h3_xpu_dense_attention(attention, q, q, q, (0, 4))


@pytest.mark.parametrize(
    "capability,available", [((2, 0), True), ((2, 0), False), ((3, 5), True)]
)
def test_dense_kernel_resolution_is_cached_and_bmg_only(
    monkeypatch, capability, available
):
    from sgl_kernel import diffusion

    h3._minimax_h3_xpu_sdp.cache_clear()
    probe = Mock(return_value=capability)
    native = Mock()
    availability = Mock(return_value=available)
    monkeypatch.setattr(torch.ops.sgl_kernel, "query_device", probe)
    monkeypatch.setattr(diffusion, "is_available", availability, raising=False)
    monkeypatch.setattr(diffusion, "sdp", native, raising=False)
    try:
        expected = native if capability == (2, 0) and available else None
        assert h3._minimax_h3_xpu_sdp(0) is expected
        assert h3._minimax_h3_xpu_sdp(0) is expected
        probe.assert_called_once_with(0)
        assert availability.call_count == int(capability == (2, 0))
    finally:
        h3._minimax_h3_xpu_sdp.cache_clear()
