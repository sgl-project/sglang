"""MiniMax-H3 TP2 x U2 grouped-QKV ownership gates."""

from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch

from sglang.multimodal_gen.configs.models.dits.minimax_h3 import MiniMaxH3DiTArchConfig
from sglang.multimodal_gen.runtime.models.dits.minimax_h3 import (
    MiniMaxH3DiTModel,
    _can_use_ulysses_gather_qkv,
    _copy_grouped_qkv_tp_ulysses_shard,
    _reorder_grouped_qkv_to_qkv,
    _reorder_grouped_qkv_to_ulysses_qkv,
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


NUM_HEADS = 56
TP_SIZE = 2
ULYSSES_SIZE = 2
HEAD_DIM = 2
HIDDEN = 3


def _dense_weight() -> torch.Tensor:
    return torch.arange(
        NUM_HEADS * 3 * HEAD_DIM * HIDDEN,
        dtype=torch.bfloat16,
    ).reshape(NUM_HEADS * 3 * HEAD_DIM, HIDDEN)


def _rank_shard(
    dense: torch.Tensor, *, tp_rank: int, ulysses_rank: int
) -> torch.Tensor:
    local_heads = NUM_HEADS // (TP_SIZE * ULYSSES_SIZE)
    shard = torch.empty(
        3 * local_heads * HEAD_DIM,
        HIDDEN,
        dtype=torch.bfloat16,
    )
    shard.output_dim = 0
    assert _copy_grouped_qkv_tp_ulysses_shard(
        shard,
        dense,
        num_query_groups=NUM_HEADS,
        head_dim=HEAD_DIM,
        tp_rank=tp_rank,
        tp_size=TP_SIZE,
        ulysses_rank=ulysses_rank,
        ulysses_size=ULYSSES_SIZE,
    )
    return shard.reshape(3, local_heads, HEAD_DIM, HIDDEN)


def test_tp2_u2_shards_reconstruct_dense_qkv() -> None:
    dense = _dense_weight()
    baseline = _reorder_grouped_qkv_to_qkv(
        dense,
        num_query_groups=NUM_HEADS,
        heads_per_group=1,
        head_dim=HEAD_DIM,
    ).reshape(3, NUM_HEADS, HEAD_DIM, HIDDEN)
    reconstructed = torch.cat(
        [
            torch.cat(
                [
                    _rank_shard(dense, tp_rank=tp_rank, ulysses_rank=ulysses_rank)
                    for ulysses_rank in range(ULYSSES_SIZE)
                ],
                dim=1,
            )
            for tp_rank in range(TP_SIZE)
        ],
        dim=1,
    )
    torch.testing.assert_close(reconstructed, baseline, rtol=0, atol=0)

    for ulysses_rank in range(ULYSSES_SIZE):
        reordered = _reorder_grouped_qkv_to_ulysses_qkv(
            dense,
            num_query_groups=NUM_HEADS,
            head_dim=HEAD_DIM,
            tp_size=TP_SIZE,
            ulysses_size=ULYSSES_SIZE,
            ulysses_rank=ulysses_rank,
        ).reshape(3, NUM_HEADS // ULYSSES_SIZE, HEAD_DIM, HIDDEN)
        expected = torch.cat(
            [
                _rank_shard(dense, tp_rank=tp_rank, ulysses_rank=ulysses_rank)
                for tp_rank in range(TP_SIZE)
            ],
            dim=1,
        )
        torch.testing.assert_close(reordered, expected, rtol=0, atol=0)


def test_tp2_u2_gather_project_matches_tp_local_projection() -> None:
    dense = _dense_weight()
    gathered_x = torch.tensor(
        [
            [0.25, -0.50, 0.75],
            [1.00, 0.50, -0.25],
            [-1.00, 0.25, 0.50],
            [0.75, -0.75, 0.25],
        ],
        dtype=torch.float32,
    )
    for tp_rank in range(TP_SIZE):
        u_weights = [
            _rank_shard(dense, tp_rank=tp_rank, ulysses_rank=ulysses_rank)
            for ulysses_rank in range(ULYSSES_SIZE)
        ]
        tp_weight = torch.cat(u_weights, dim=1)
        baseline = torch.einsum("si,qhdi->sqhd", gathered_x, tp_weight.float())
        candidate = torch.cat(
            [
                torch.einsum("si,qhdi->sqhd", gathered_x, weight.float())
                for weight in u_weights
            ],
            dim=2,
        )
        torch.testing.assert_close(candidate, baseline, rtol=0, atol=0)
