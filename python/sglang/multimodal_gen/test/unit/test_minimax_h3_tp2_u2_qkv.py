"""MiniMax-H3 TP2 x U2 grouped-QKV ownership gates."""

from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch

from sglang.multimodal_gen.configs.models.dits.minimax_h3 import MiniMaxH3DiTArchConfig
from sglang.multimodal_gen.runtime.layers.linear import MergedColumnParallelLinear
from sglang.multimodal_gen.runtime.models.dits.minimax_h3 import (
    MiniMaxH3Attention,
    MiniMaxH3DiTModel,
    _can_use_ulysses_gather_qkv,
)

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
