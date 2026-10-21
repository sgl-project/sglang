"""MiniMax-H3 TP2 x U2 grouped-QKV ownership gates."""

from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest
import torch

from sglang.multimodal_gen.configs.models.dits.minimax_h3 import MiniMaxH3DiTArchConfig
from sglang.multimodal_gen.runtime.layers.linear import MergedColumnParallelLinear
from sglang.multimodal_gen.runtime.models.dits.minimax_h3 import (
    MiniMaxH3Attention,
    MiniMaxH3DiTModel,
    _can_use_ulysses_gather_qkv,
    _minimax_h3_attention_core_impl,
)
from sglang.multimodal_gen.runtime.platforms import AttentionBackendEnum

_MODEL = "sglang.multimodal_gen.runtime.models.dits.minimax_h3"


def _supported_server_args(**overrides):
    values = dict(
        performance_mode="memory",
        enable_torch_compile=False,
        enable_breakable_cuda_graph=False,
        use_fsdp_inference=False,
        lora_path=None,
    )
    return SimpleNamespace(**(values | overrides))


@pytest.mark.parametrize(
    "override",
    [
        {"performance_mode": "speed"},
        {"enable_torch_compile": True},
        {"enable_breakable_cuda_graph": True},
        {"use_fsdp_inference": True},
        {"lora_path": "adapter"},
    ],
)
def test_unsupported_execution_keeps_full_qkv_ownership(override):
    arch = MiniMaxH3DiTArchConfig()
    partition = dict(tp_size=2, ulysses_size=2, ring_size=1)
    with patch(f"{_MODEL}.current_platform.is_cuda", return_value=True):
        assert _can_use_ulysses_gather_qkv(
            arch, None, **partition, server_args=_supported_server_args()
        )
        assert not _can_use_ulysses_gather_qkv(
            arch, None, **partition, server_args=_supported_server_args(**override)
        )


@pytest.mark.parametrize(
    "partition",
    [
        dict(tp_size=1, ulysses_size=2, ring_size=1),
        dict(tp_size=2, ulysses_size=1, ring_size=1),
        dict(tp_size=2, ulysses_size=2, ring_size=2),
    ],
)
def test_unvalidated_partitions_keep_full_qkv_ownership(partition):
    with patch(f"{_MODEL}.current_platform.is_cuda", return_value=True):
        assert not _can_use_ulysses_gather_qkv(
            MiniMaxH3DiTArchConfig(),
            None,
            **partition,
            server_args=_supported_server_args(),
        )


@pytest.mark.parametrize(
    "field,value",
    [
        ("hidden_size", 4096),
        ("num_attention_heads", 32),
        ("attention_head_dim", 64),
        ("checkpoint_uses_diffusers_layout", True),
        ("has_gate_compress", True),
        ("hybrid_attention", object()),
    ],
)
def test_unvalidated_architectures_keep_full_qkv_ownership(field, value):
    arch = MiniMaxH3DiTArchConfig()
    setattr(arch, field, value)
    with patch(f"{_MODEL}.current_platform.is_cuda", return_value=True):
        assert not _can_use_ulysses_gather_qkv(
            arch,
            None,
            tp_size=2,
            ulysses_size=2,
            ring_size=1,
            server_args=_supported_server_args(),
        )


def test_quantized_or_unconfigured_models_keep_full_qkv_ownership():
    arch = MiniMaxH3DiTArchConfig()
    partition = dict(tp_size=2, ulysses_size=2, ring_size=1)
    with patch(f"{_MODEL}.current_platform.is_cuda", return_value=True):
        assert not _can_use_ulysses_gather_qkv(
            arch, object(), **partition, server_args=_supported_server_args()
        )
        assert not _can_use_ulysses_gather_qkv(
            arch, None, **partition, server_args=None
        )
    with patch(f"{_MODEL}.current_platform.is_cuda", return_value=False):
        assert not _can_use_ulysses_gather_qkv(
            arch, None, **partition, server_args=_supported_server_args()
        )


def test_live_lora_cannot_bypass_sharded_qkv_projection():
    model = SimpleNamespace(_use_ulysses_gather_qkv=True)
    with pytest.raises(ValueError, match="restarting without"):
        MiniMaxH3DiTModel.prepare_lora_adapter(model, {})
    with pytest.raises(ValueError, match="restarting without"):
        MiniMaxH3DiTModel.validate_lora_layers(model, ["blocks.0.attn.qkv_proj"])


@pytest.mark.parametrize("tp_rank,ulysses_rank", [(0, 0), (0, 1), (1, 0), (1, 1)])
@pytest.mark.parametrize("sharded", [False, True])
@pytest.mark.parametrize("contiguous", [False, True])
def test_installed_qkv_loader_preserves_head_ownership(
    tp_rank, ulysses_rank, sharded, contiguous
):
    heads, head_dim, hidden = 8, 2, 3
    partition = 2 if sharded else 1
    projection = MergedColumnParallelLinear(
        hidden,
        [heads * head_dim // partition] * 3,
        bias=False,
        params_dtype=torch.bfloat16,
        tp_group=SimpleNamespace(world_size=2, rank_in_group=tp_rank),
    )
    attention = SimpleNamespace(
        qkv_proj=projection,
        tp_size=2,
        _use_ulysses_gather_qkv=sharded,
        _ulysses_size=2,
        _ulysses_rank=ulysses_rank,
    )
    MiniMaxH3Attention._install_qkv_weight_loader(
        attention,
        SimpleNamespace(num_attention_heads=heads, attention_head_dim=head_dim),
    )
    dense = torch.arange(heads * 3 * head_dim * hidden, dtype=torch.bfloat16).reshape(
        -1, hidden
    )
    if not contiguous:
        dense = dense.t().contiguous().t()
    assert dense.is_contiguous() == contiguous
    projection.weight.weight_loader(projection.weight, dense)
    local_heads = heads // (2 * partition)
    start = (tp_rank * partition + (ulysses_rank if sharded else 0)) * local_heads
    expected = dense.reshape(heads, 3, head_dim, hidden)[start : start + local_heads]
    expected = expected.permute(1, 0, 2, 3).reshape_as(projection.weight)
    torch.testing.assert_close(projection.weight, expected, rtol=0, atol=0)


@pytest.mark.parametrize("qkv_ready", [False, True])
@pytest.mark.parametrize("pipeline_available", [False, True])
def test_attention_exchanges_pre_gathered_qkv_only_once(qkv_ready, pipeline_available):
    q = torch.arange(8, dtype=torch.float32).reshape(4, 1, 2)
    expected = q + 1
    attention = SimpleNamespace(
        _attention_backend_enum=AttentionBackendEnum.FA,
        _attention_impl=SimpleNamespace(forward_varlen=Mock(return_value=expected)),
    )
    usp = "sglang.multimodal_gen.runtime.layers.usp"
    with (
        patch(f"{_MODEL}._minimax_h3_bind_attention_backend"),
        patch(
            f"{_MODEL}._minimax_h3_pipelined_dense_attention",
            return_value=expected if pipeline_available else None,
        ) as pipeline,
        patch(
            f"{usp}._usp_input_all_to_all_packed_qkv", return_value=(q, q, q)
        ) as input_a2a,
        patch(
            f"{usp}._usp_output_all_to_all", side_effect=lambda x, **kw: x
        ) as output_a2a,
    ):
        actual = _minimax_h3_attention_core_impl(
            attention,
            q,
            q,
            q,
            cu_seqlens=torch.tensor([0, 4], dtype=torch.int32),
            cu_seqlens_host=(0, 4),
            max_seqlen=4,
            ulysses_active=True,
            ulysses_qkv_ready=qkv_ready,
        )
    torch.testing.assert_close(actual, expected)
    assert pipeline.call_count == int(not qkv_ready)
    pipeline_used = pipeline_available and not qkv_ready
    assert input_a2a.call_count == int(not qkv_ready and not pipeline_used)
    assert output_a2a.call_count == int(not pipeline_used)
    assert attention._attention_impl.forward_varlen.call_count == int(not pipeline_used)


@pytest.mark.parametrize("qkv_ready", [False, True])
def test_fused_rope_pipeline_respects_pre_gathered_qkv(qkv_ready):
    x = torch.zeros(2, 4)
    q = torch.zeros((4, 1, 2) if qkv_ready else (2, 2, 2))
    expected = torch.arange(8, dtype=torch.float32).reshape(2, 2, 2)
    norm = SimpleNamespace(weight=torch.ones(2), eps=1e-6)
    attention = SimpleNamespace(
        _use_ulysses_gather_qkv=qkv_ready,
        _use_fused_qknorm_rope=True,
        q_norm=norm,
        k_norm=norm,
        num_heads=2,
        head_dim=2,
        _attention_backend_enum=AttentionBackendEnum.FA,
        bcg_breakpoint=False,
    )
    with (
        patch(
            f"{_MODEL}._minimax_h3_qknorm_rope_pipelined_attention",
            return_value=expected,
        ) as pipeline,
        patch(f"{_MODEL}.fused_inplace_qknorm_rope") as fused_rope,
        patch(
            f"{_MODEL}._minimax_h3_attention_core_impl", return_value=expected
        ) as core,
    ):
        actual = MiniMaxH3Attention._attend(
            attention,
            x,
            q,
            q,
            q,
            rope_cache=(torch.zeros(4, 2), torch.arange(q.shape[0])),
            cu_seqlens=torch.tensor([0, 4], dtype=torch.int32),
            cu_seqlens_host=(0, 4),
            max_seqlen=4,
            subblock_sparse_query_block_mask=None,
            ulysses_active=True,
            ring_active=False,
        )
    torch.testing.assert_close(actual, expected.reshape(2, 4))
    assert pipeline.call_count == int(not qkv_ready)
    assert fused_rope.call_count == int(qkv_ready)
    assert core.call_count == int(qkv_ready)
    if qkv_ready:
        assert core.call_args.kwargs["ulysses_qkv_ready"] is True
