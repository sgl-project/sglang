# SPDX-License-Identifier: Apache-2.0
"""Contracts for dense Sage consuming prepared QKV without requantization."""

import sys
from contextlib import contextmanager
from types import SimpleNamespace

import pytest
import torch

from sglang.multimodal_gen.runtime.distributed import sage_ulysses_route
from sglang.multimodal_gen.runtime.layers.attention.backends import (
    prepared_sage_attn as dense,
)
from sglang.multimodal_gen.runtime.layers.attention.backends import (
    sage_attn,
)
from sglang.multimodal_gen.runtime.layers.attention.backends.subblock_sparse import (
    prepared_sage_attention as subblock,
)
from sglang.multimodal_gen.runtime.layers.attention.backends.subblock_sparse_attn import (
    SubBlockSparseAttentionImpl,
)


@pytest.mark.parametrize(
    "mode,layer", [("sage_fp8", 0), ("bf16", 0), ("sage_fp8", None)]
)
def test_sage_ssa_uses_dense_sage_for_dit_warmup(monkeypatch, mode, layer):
    from sglang.multimodal_gen.runtime.layers.attention import selector

    selected = []
    fallback = object()
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda: (12, 0))
    monkeypatch.setattr(
        sage_attn.envs.SGLANG_USE_SAGE_ATTENTION_IMPL, "get", lambda: "auto"
    )
    monkeypatch.setattr(
        selector,
        "get_attn_backend",
        lambda *args, **kwargs: (
            selected.append(kwargs)
            or SimpleNamespace(get_impl_cls=lambda: lambda **kwargs: fallback)
        ),
    )
    impl = SimpleNamespace(
        schedule=SimpleNamespace(compute_mode=mode, quantize_before_a2a=True),
        layer_idx=layer,
        num_heads=8,
        head_size=128,
        num_kv_heads=8,
        prefix="blocks.0.attn" if layer is not None else "token_refiner.blocks.0.attn",
        softmax_scale=0.125,
    )
    result = SubBlockSparseAttentionImpl._build_dense_impl(impl, causal=False)
    if mode == "sage_fp8" and layer is not None:
        assert isinstance(result, sage_attn.SageAttentionImpl)
        assert result.quantize_before_a2a and result.packed_trailing_padding
        assert not selected
    else:
        assert result is fallback and len(selected) == 1


@pytest.mark.parametrize("gate", ["step", "layer", "length"])
def test_prepared_sage_ssa_selects_dense_before_sparse_admission(monkeypatch, gate):
    q = torch.zeros((64, 8, 128), dtype=torch.bfloat16)
    output = torch.ones_like(q)
    dense_impl = SimpleNamespace(quantize_before_a2a=True)
    impl = SimpleNamespace(
        schedule=SimpleNamespace(
            quantize_before_a2a=True, min_seq_len=131 if gate == "length" else 0
        ),
        layer_enabled=gate != "layer",
        _step_enabled=lambda: gate != "step",
        dense_impl=dense_impl,
    )
    calls = []
    transport = object()

    def forward(*args, **kwargs):
        assert args == (q, q, q)
        assert kwargs["impl"] is dense_impl and kwargs["transport"] is transport
        assert kwargs["cu_seqlens_host"] == (0, 130, 256)
        calls.append("dense")
        return output

    monkeypatch.setattr(dense, "prepared_dense_sage_attention_forward", forward)
    result = subblock.prepared_subblock_sage_attention_forward(
        q,
        q,
        q,
        impl=impl,
        cu_seqlens_host=(0, 130, 256),
        max_seqlen=130,
        ulysses_active=True,
        ring_active=False,
        transport=transport,
    )
    assert result is output and calls == ["dense"]


@pytest.mark.parametrize("same_stream", [False, True])
@pytest.mark.parametrize("fail", [False, True])
def test_dense_kernel_stream_orders_exchange_and_gather(monkeypatch, same_stream, fail):
    calls = []

    class Stream:
        def __init__(self, name):
            self.name = name

        def wait_stream(self, other):
            calls.append((self.name, "wait", other.name))

    default = Stream("default")
    current = default if same_stream else Stream("compute")

    @contextmanager
    def select(stream):
        assert stream is default
        calls.append("enter")
        try:
            yield
        finally:
            calls.append("exit")

    monkeypatch.setattr(torch.cuda, "current_stream", lambda device: current)
    monkeypatch.setattr(torch.cuda, "default_stream", lambda device: default)
    monkeypatch.setattr(torch.cuda, "stream", select)

    def run():
        with dense._sage_kernel_stream(torch.device("cuda:0")):
            calls.append("kernel")
            if fail:
                raise RuntimeError("kernel failed")

    if fail:
        with pytest.raises(RuntimeError, match="kernel failed"):
            run()
    else:
        run()
    middle = ["enter", "kernel", "exit"]
    expected = (
        middle
        if same_stream
        else [("default", "wait", "compute"), *middle, ("compute", "wait", "default")]
    )
    assert calls == expected


def operands(used, dtype=torch.bfloat16):
    return SimpleNamespace(
        q=torch.zeros((2, 256, 2, 128), dtype=torch.int8),
        k=torch.zeros((2, 256, 2, 128), dtype=torch.int8),
        v=torch.zeros((2, 128, 2, 256), dtype=torch.float8_e4m3fn),
        q_scale=torch.ones((2, 2, (used + 127) // 128 * 4)),
        k_scale=torch.ones((2, 2, (used + 63) // 64)),
        v_scale=torch.ones((2, 2, 128)),
        layout="sage2_sm89_sm120",
        logical_sequence=256,
        used_sequence=used,
        input_dtype=dtype,
    )


@pytest.mark.parametrize("used", [1, 64, 130, 256])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_dense_consumer_reuses_operands_and_masks_padding(monkeypatch, used, dtype):
    encoded = operands(used, dtype)
    calls = []

    def kernel(q, k, v, output, qs, ks, vs, *flags):
        calls.append(output)
        assert q.shape == k.shape == output.shape == (2, used, 2, 128)
        assert q.data_ptr() == encoded.q.data_ptr()
        assert k.data_ptr() == encoded.k.data_ptr()
        assert q.stride() == encoded.q.stride()
        assert k.stride() == encoded.k.stride()
        assert v is encoded.v and qs is encoded.q_scale
        assert ks is encoded.k_scale and vs is encoded.v_scale
        assert output.dtype == dtype and output.stride(-1) == 1
        assert flags == (0, 0, 2, 0.125, 0)
        output.fill_(3)
        # The real kernel returns LSE, not its caller-owned attention output.
        return torch.tensor(-99)

    monkeypatch.setattr(dense, "_load_prepared_dense_sage_kernel", lambda: kernel)
    actual = dense.PreparedDenseSageAttention()(encoded, softmax_scale=0.125)
    assert len(calls) == 1 and actual.shape == encoded.q.shape
    assert actual.dtype == dtype and actual.data_ptr() == calls[0].data_ptr()
    assert torch.all(actual[:, :used] == 3)
    assert not actual[:, used:].count_nonzero().item()


@pytest.mark.parametrize(
    "layout,used",
    [("sage2_sm90", 130), ("sage2_sm89_sm120", 0), ("sage2_sm89_sm120", 257)],
)
def test_dense_consumer_rejects_incompatible_metadata(monkeypatch, layout, used):
    def unexpected(*args, **kwargs):
        pytest.fail("Invalid metadata must not enter the native kernel")

    monkeypatch.setattr(dense, "_load_prepared_dense_sage_kernel", lambda: unexpected)
    consumer = dense.PreparedDenseSageAttention()
    with pytest.raises(ValueError, match="operands|prefix"):
        consumer(
            SimpleNamespace(layout=layout, used_sequence=used, logical_sequence=256),
            softmax_scale=0.125,
        )


def test_missing_prequantized_kernel_has_actionable_error(monkeypatch):
    dense._load_prepared_dense_sage_kernel.cache_clear()
    monkeypatch.setitem(sys.modules, "sageattention.sm89_compile", None)
    try:
        with pytest.raises(ImportError, match="SageAttention build with SM120"):
            dense._load_prepared_dense_sage_kernel()
    finally:
        dense._load_prepared_dense_sage_kernel.cache_clear()


@pytest.mark.parametrize("eligible", [False, True])
def test_dense_forward_uses_shared_transport_without_subblock_metadata(
    monkeypatch, eligible
):
    calls = []
    encoded = operands(130)
    # Use a one-batch record with the same number of elements as rank-local Q.
    encoded = SimpleNamespace(
        **{
            **vars(encoded),
            "q": encoded.q[:1],
            "k": encoded.k[:1],
            "v": encoded.v[:1],
            "q_scale": encoded.q_scale[:1],
            "k_scale": encoded.k_scale[:1],
            "v_scale": encoded.v_scale[:1],
        }
    )
    q = torch.zeros((64, 8, 128), dtype=torch.bfloat16)
    monkeypatch.setattr(
        sage_ulysses_route,
        "check_prepared_sage_route",
        lambda *args, **kwargs: SimpleNamespace(enabled=eligible, reason="test"),
    )

    def load_kernel():
        calls.append("load")

        def kernel(*args):
            assert args[0].shape[1] == args[1].shape[1] == 130
            calls.append("attention")
            args[3].fill_(5)

        return kernel

    def exchange(query, key, value):
        assert query is key is value is q
        calls.append("exchange")
        return encoded

    def gather(output):
        calls.append("gather")
        assert torch.all(output[:, :130] == 5)
        assert not output[:, 130:].count_nonzero().item()
        return output.reshape(1, *q.shape)

    monkeypatch.setattr(dense, "_load_prepared_dense_sage_kernel", load_kernel)
    result = dense.prepared_dense_sage_attention_forward(
        q,
        q,
        q,
        impl=SimpleNamespace(
            quantize_before_a2a=True, causal=False, dropout=0, softmax_scale=0.125
        ),
        cu_seqlens_host=(0, 130, 256),
        max_seqlen=130,
        ulysses_active=True,
        ring_active=False,
        transport=SimpleNamespace(exchange=exchange, gather=gather),
    )
    assert calls == (["load", "exchange", "attention", "gather"] if eligible else [])
    if eligible:
        assert result.shape == q.shape
    else:
        assert result is None


@pytest.mark.parametrize("gate", ["disabled", "causal", "dropout"])
def test_dense_semantic_guards_precede_collectives(monkeypatch, gate):
    def unexpected(*args, **kwargs):
        pytest.fail("Rejected calls must not enter collective route resolution")

    monkeypatch.setattr(dense, "resolve_prepared_sage_route", unexpected)
    q = torch.zeros((64, 8, 128), dtype=torch.bfloat16)
    assert (
        dense.prepared_dense_sage_attention_forward(
            q,
            q,
            q,
            impl=SimpleNamespace(
                quantize_before_a2a=gate != "disabled",
                causal=gate == "causal",
                dropout=0.1 if gate == "dropout" else 0,
            ),
            cu_seqlens_host=(0, 130, 256),
            max_seqlen=130,
            ulysses_active=True,
            ring_active=False,
        )
        is None
    )


@pytest.mark.parametrize("value", [False, True, "false", 1])
def test_dense_opt_in_is_strictly_boolean(monkeypatch, value):
    monkeypatch.setattr(
        sage_attn.envs.SGLANG_USE_SAGE_ATTENTION_IMPL, "get", lambda: "auto"
    )
    kwargs = dict(
        num_heads=8,
        head_size=128,
        causal=False,
        softmax_scale=0.125,
        quantize_before_a2a=value,
    )
    if type(value) is not bool:
        with pytest.raises(ValueError, match="must be a boolean"):
            sage_attn.SageAttentionImpl(**kwargs)
    else:
        assert sage_attn.SageAttentionImpl(**kwargs).quantize_before_a2a is value


@pytest.mark.parametrize("implementation", ["trtllm_gen", "trtllm-gen"])
def test_prepared_opt_in_rejects_a_conflicting_sage_implementation(
    monkeypatch, implementation
):
    monkeypatch.setattr(
        sage_attn.envs.SGLANG_USE_SAGE_ATTENTION_IMPL, "get", lambda: implementation
    )
    with pytest.raises(ValueError, match="cannot be combined"):
        sage_attn.SageAttentionImpl(
            num_heads=8,
            head_size=128,
            causal=False,
            softmax_scale=0.125,
            quantize_before_a2a=True,
        )


@pytest.mark.parametrize("enabled", [False, True])
def test_h3_backend_binding_passes_the_dense_opt_in(monkeypatch, enabled):
    from sglang.multimodal_gen.runtime import server_args
    from sglang.multimodal_gen.runtime.models.dits import minimax_h3 as model

    monkeypatch.setattr(
        server_args,
        "get_global_server_args",
        lambda: SimpleNamespace(
            attention_backend_config={"quantize_before_a2a": enabled}
        ),
    )
    monkeypatch.setattr(
        sage_attn.envs.SGLANG_USE_SAGE_ATTENTION_IMPL, "get", lambda: "auto"
    )
    attention = SimpleNamespace(
        num_heads=8,
        head_dim=128,
        softmax_scale=0.125,
        prefix="blocks.0.attn",
        _cube_sparse_capable=True,
    )
    model.MiniMaxH3Attention._set_attention_backend(
        attention, sage_attn.SageAttentionBackend
    )
    assert attention._attention_impl.quantize_before_a2a is enabled
    assert attention._attention_impl.packed_trailing_padding


@pytest.mark.parametrize("enabled", [False, True])
def test_h3_dispatches_prepared_dense_sage_before_the_native_exchange(
    monkeypatch, enabled
):
    from sglang.multimodal_gen.runtime.models.dits import minimax_h3 as model

    q = torch.zeros((64, 8, 128), dtype=torch.bfloat16)
    output = torch.ones_like(q)
    calls = []
    monkeypatch.setattr(model, "_minimax_h3_bind_attention_backend", lambda *args: None)

    def prepared(*args, **kwargs):
        assert enabled and kwargs["cu_seqlens_host"] == (0, 130, 256)
        calls.append("prepared")
        return output

    def native(*args, **kwargs):
        calls.append("native")
        return output

    monkeypatch.setattr(dense, "prepared_dense_sage_attention_forward", prepared)
    monkeypatch.setattr(model, "_minimax_h3_pipelined_dense_attention", native)
    attention = SimpleNamespace(
        _attention_backend_enum=sage_attn.AttentionBackendEnum.SAGE_ATTN,
        _attention_impl=SimpleNamespace(quantize_before_a2a=enabled),
    )
    result = model._minimax_h3_attention_core_impl(
        attention,
        q,
        q,
        q,
        cu_seqlens=torch.tensor([0, 130, 256], dtype=torch.int32),
        cu_seqlens_host=(0, 130, 256),
        max_seqlen=130,
        ulysses_active=True,
    )
    assert result is output and calls == (["prepared"] if enabled else ["native"])
