"""Unit tests for the unified SP shard helpers (pure logic, no distributed)."""

import pytest
import torch

from sglang.multimodal_gen.runtime.distributed import sp_shard as sps
from sglang.multimodal_gen.runtime.distributed.sp_shard import (
    SpShard,
    shard_like,
    tail_attn_meta,
)


def _fake_sp(monkeypatch, sp_size, sp_rank=0, ring=1):
    monkeypatch.setattr(sps, "get_sp_world_size", lambda: sp_size)
    monkeypatch.setattr(sps, "get_sp_parallel_rank", lambda: sp_rank)
    monkeypatch.setattr(sps, "get_ring_parallel_world_size", lambda: ring)


# --- plan_shard math --------------------------------------------------------


def test_plan_shard_divisible(monkeypatch):
    _fake_sp(monkeypatch, 2, 1)
    s = sps.build_shard_plan(16)
    assert (s.local_len, s.num_pad, s.local_pad) == (8, 0, 0)


def test_plan_shard_padded_last_rank(monkeypatch):
    _fake_sp(monkeypatch, 4, 3)
    s = sps.build_shard_plan(14)
    assert (s.local_len, s.num_pad) == (4, 2)
    assert s.local_pad == 2 and s.local_real_len == 2


def test_plan_shard_pad_only_on_last_rank(monkeypatch):
    _fake_sp(monkeypatch, 4, 0)
    s = sps.build_shard_plan(14)
    assert s.local_pad == 0 and s.local_real_len == 4


def test_plan_shard_sp1_noop(monkeypatch):
    _fake_sp(monkeypatch, 1)
    s = sps.build_shard_plan(15)
    assert (s.local_len, s.num_pad, s.sp_size) == (15, 0, 1)


# --- shard_like -------------------------------------------------------------


def test_shard_like_zero_pads_tail():
    shard = SpShard(orig_len=15, local_len=8, num_pad=1, sp_size=2, sp_rank=1)
    x = torch.arange(15, dtype=torch.float32).unsqueeze(0).unsqueeze(-1)
    local = shard_like(x, shard, dim=1)
    assert local.shape[1] == 8
    assert local[0, -1, 0].item() == 0.0  # tail pad
    assert local[0, 0, 0].item() == 8.0  # rank1 starts at token 8


def test_shard_like_repeat_last():
    shard = SpShard(orig_len=15, local_len=8, num_pad=1, sp_size=2, sp_rank=1)
    x = torch.arange(15, dtype=torch.float32).unsqueeze(-1)
    local = shard_like(x, shard, dim=0, pad_mode="repeat_last")
    assert local[-1, 0].item() == 14.0  # repeated last row, not zero


def test_shard_like_chunks_align_across_tensors():
    # RoPE cache sharded with the same plan stays aligned with hidden states.
    shard = SpShard(orig_len=15, local_len=8, num_pad=1, sp_size=2, sp_rank=0)
    x = torch.arange(15).unsqueeze(0).unsqueeze(-1).float()
    rope = torch.arange(15).unsqueeze(-1).float()
    assert torch.equal(
        shard_like(x, shard, dim=1)[0, :, 0], shard_like(rope, shard, dim=0)[:, 0]
    )


# --- tail_attn_meta ---------------------------------------------------------


def test_tail_meta_none_when_divisible():
    shard = SpShard(orig_len=16, local_len=8, num_pad=0, sp_size=2, sp_rank=0)
    assert tail_attn_meta(shard, 1, torch.device("cpu")) is None


def test_tail_meta_single_stream():
    shard = SpShard(orig_len=15, local_len=8, num_pad=1, sp_size=2, sp_rank=1)
    meta = tail_attn_meta(shard, 1, torch.device("cpu"))
    assert meta["pad_start"] == 15 and meta["pad_end"] == 16
    assert meta["local_pad"] == 1
    assert meta["cu_seqlens_tail"].tolist() == [0, 15, 16]
    assert meta["max_seqlen_tail"] == 15


def test_tail_meta_joint_layout_and_batch():
    # sp=2, local_txt=8 (1 pad), img=100 per rank -> S = 2*(8+100) = 216.
    shard = SpShard(orig_len=15, local_len=8, num_pad=1, sp_size=2, sp_rank=1)
    meta = tail_attn_meta(shard, 2, torch.device("cpu"), image_seq_len=100)
    assert meta["pad_start"] == 215 and meta["pad_end"] == 216
    assert meta["cu_seqlens_tail"].tolist() == [0, 215, 216, 431, 432]


def test_tail_meta_matches_legacy_gap_formula():
    # The tail layout puts the pad exactly where the legacy per-model gap
    # formula pointed, minus the relocation: end == S (global tail).
    sp, local_txt, img, num_pad = 3, 5, 40, 2
    shard = SpShard(
        orig_len=sp * local_txt - num_pad,
        local_len=local_txt,
        num_pad=num_pad,
        sp_size=sp,
        sp_rank=sp - 1,
    )
    meta = tail_attn_meta(shard, 1, torch.device("cpu"), image_seq_len=img)
    seq = sp * (local_txt + img)
    assert meta["pad_end"] == seq
    assert meta["pad_start"] == seq - num_pad


# --- plan_text_strategy -----------------------------------------------------


def test_strategy_sp1_replicates(monkeypatch):
    _fake_sp(monkeypatch, 1)
    assert sps.plan_text_strategy(100) == "replicate"


def test_strategy_shard_when_legal(monkeypatch):
    _fake_sp(monkeypatch, 2)
    assert sps.plan_text_strategy(15) == "shard"
    assert sps.plan_text_strategy(16) == "shard"


def test_strategy_ring_blocks_padded_shard(monkeypatch):
    _fake_sp(monkeypatch, 2, ring=2)
    assert sps.plan_text_strategy(15) == "replicate"  # padded shard needs mask
    assert sps.plan_text_strategy(16) == "shard"  # divisible: no mask needed


def test_strategy_min_len_threshold(monkeypatch):
    _fake_sp(monkeypatch, 2)
    monkeypatch.setattr(sps, "_TEXT_SHARD_MIN", 64)
    assert sps.plan_text_strategy(32) == "replicate"
    assert sps.plan_text_strategy(64) == "shard"


# --- gather_seq -------------------------------------------------------------


def test_gather_seq_sp1_noop(monkeypatch):
    _fake_sp(monkeypatch, 1)
    x = torch.randn(1, 5, 2)
    assert sps.gather_seq(x, 5, dim=1) is x


def test_gather_seq_trims(monkeypatch):
    _fake_sp(monkeypatch, 2)
    monkeypatch.setattr(
        sps, "sequence_model_parallel_all_gather", lambda t, dim: torch.cat([t, t], dim)
    )
    local = torch.randn(1, 8, 2)
    out = sps.gather_seq(local, 15, dim=1)
    assert out.shape[1] == 15


if __name__ == "__main__":
    pytest.main([__file__, "-q"])
