"""Unit tests for the unified SP shard helpers (pure logic, no distributed)."""

import sys

import pytest
import torch

from sglang.multimodal_gen.runtime.distributed import sp_shard_utils as sps
from sglang.multimodal_gen.runtime.distributed.sp_shard_utils import (
    SpShard,
    shard_like,
    tail_attn_meta,
)
from sglang.multimodal_gen.runtime.models.dits.zimage_joint_sp import shard_block_inputs
from sglang.multimodal_gen.runtime.platforms import current_platform


def _fake_sp(monkeypatch, sp_size, sp_rank=0):
    monkeypatch.setattr(sps, "get_sp_world_size", lambda: sp_size)
    monkeypatch.setattr(sps, "get_sp_parallel_rank", lambda: sp_rank)


# --- build_shard_plan math --------------------------------------------------------


@pytest.mark.parametrize("rank", [0, 1])
def test_joint_sp_preserves_batched_condition_and_rope(monkeypatch, rank):
    _fake_sp(monkeypatch, 2, rank)
    tokens = torch.tensor(
        [[0, 1, 2, 100, 101, 102, 103, 104], [3, 4, 5, 105, 106, 107, 108, 109]]
    ).unsqueeze(-1)
    positions = torch.arange(16)
    cache = positions.square().unsqueeze(-1)
    expected = slice(rank * 4, (rank + 1) * 4)
    local, (indexed_cache, indices, _) = shard_block_inputs(
        tokens, (cache, positions, None), None, None
    )
    torch.testing.assert_close(local, tokens[:, expected])
    torch.testing.assert_close(
        indexed_cache[indices].view(2, 4, 1), cache.view(2, 8, 1)[:, expected]
    )

    single = torch.arange(16, dtype=torch.float32).view(1, 8, 2)
    cache_2d = torch.arange(8, dtype=torch.float32).unsqueeze(-1)
    complex_freqs = torch.complex(cache_2d, -cache_2d)
    local_single, (sharded_cache, pos_out, sharded_complex) = shard_block_inputs(
        single, (cache_2d, None, complex_freqs), None, None
    )
    assert pos_out is None
    assert local_single.stride() == (8, 2, 1)
    torch.testing.assert_close(local_single, single[:, expected])
    torch.testing.assert_close(sharded_cache, cache_2d[expected])
    torch.testing.assert_close(sharded_complex, complex_freqs[expected])


@pytest.mark.parametrize(
    "seq_len,mask,mask_meta",
    [
        (7, None, None),
        (8, torch.ones(1, 8, dtype=torch.bool), None),
        (8, None, {"max_seqlen": 8}),
    ],
)
def test_joint_sp_rejects_padded_or_masked_sequences(
    monkeypatch, seq_len, mask, mask_meta
):
    _fake_sp(monkeypatch, 2, 0)
    tokens = torch.zeros(1, seq_len, 2)
    with pytest.raises(ValueError, match="evenly divisible"):
        shard_block_inputs(tokens, (None, None, None), mask, mask_meta)


@pytest.mark.skipif(
    not torch.cuda.is_available() and not current_platform.is_cpu(),
    reason="zimage import requires CUDA or CPU diffusion platform",
)
def test_zimage_forward_joint_sp_shards_blocks_and_runs_final_layer_before_gather(
    monkeypatch,
):
    import threading

    from sglang.multimodal_gen.runtime.models.dits import zimage as zimage_mod

    local_rank = threading.local()
    local_rank.rank = 0
    observed = {
        rank: {"events": [], "noise_shapes": [], "layer_calls": []} for rank in (0, 1)
    }

    class _LinearStub(torch.nn.Module):
        def forward(self, x):
            return torch.cat([x, x + 0.5], dim=-1), None

    class _BlockStub(torch.nn.Module):
        def __init__(self, tag):
            super().__init__()
            self.tag = tag

        def forward(
            self,
            x,
            freqs_cis,
            adaln_input=None,
            rope_cos_sin_cache=None,
            rope_positions=None,
            attn_mask=None,
            attn_mask_meta=None,
            num_replicated_suffix=0,
            rope_freqs_complex=None,
        ):
            rec = observed[local_rank.rank]
            if self.tag == "noise":
                rec["noise_shapes"].append(tuple(x.shape))
                return x + 1.0
            if self.tag == "layer":
                rec["layer_calls"].append((tuple(x.shape), num_replicated_suffix))
                return x + 2.0
            return x

    class _FinalLayerStub(torch.nn.Module):
        def forward(self, x, c):
            observed[local_rank.rank]["events"].append(("final_layer", tuple(x.shape)))
            return (x[..., :4] + x[..., 4:]) * 1.5

    class _Host(zimage_mod.ZImageTransformer2DModel):
        def __init__(self):
            torch.nn.Module.__init__(self)
            self.shard_joint_sequence = True
            self.all_patch_size = (2,)
            self.all_f_patch_size = (1,)
            self.t_scale = 1.0
            self.out_channels = 1
            self._rope_caches = {}
            self._unified_freqs_cis = None
            self.t_embedder = lambda t: t.unsqueeze(-1)
            self.all_x_embedder = torch.nn.ModuleDict({"2-1": _LinearStub()})
            self.cap_embedder = _LinearStub()
            self.x_pad_token = torch.nn.Parameter(torch.zeros(1, 8))
            self.cap_pad_token = torch.nn.Parameter(torch.zeros(1, 8))
            self.noise_refiner = torch.nn.ModuleList([_BlockStub("noise")])
            self.context_refiner = torch.nn.ModuleList([_BlockStub("ctx")])
            self.layers = torch.nn.ModuleList([_BlockStub("layer")])
            self.all_final_layer = torch.nn.ModuleDict({"2-1": _FinalLayerStub()})

    hidden = torch.arange(256, dtype=torch.float32).view(1, 1, 1, 8, 32)
    cap = [torch.arange(128, dtype=torch.float32).view(32, 4)]
    freqs = (
        (torch.ones(32, 4), torch.zeros(32, 4)),
        (torch.ones(64, 4), torch.zeros(64, 4)),
    )
    t = torch.tensor([0.5])

    _fake_sp(monkeypatch, 1, 0)
    monkeypatch.setattr(zimage_mod, "get_sp_world_size", lambda: 1)
    ref = _Host()(hidden, cap, t, freqs_cis=freqs)

    for rec in observed.values():
        rec["events"].clear()
        rec["noise_shapes"].clear()
        rec["layer_calls"].clear()
    monkeypatch.setattr(sps, "get_sp_world_size", lambda: 2)
    monkeypatch.setattr(sps, "get_sp_parallel_rank", lambda: local_rank.rank)
    monkeypatch.setattr(zimage_mod, "get_sp_world_size", lambda: 2)

    barrier = threading.Barrier(2)
    gather_slots: list[torch.Tensor | None] = [None, None]

    def _gather(tensor, orig_len, dim=1):
        rank = local_rank.rank
        observed[rank]["events"].append(("gather", tuple(tensor.shape), orig_len))
        gather_slots[rank] = tensor
        barrier.wait(timeout=5.0)
        gathered = torch.cat([gather_slots[0], gather_slots[1]], dim=dim)[:, :orig_len]
        barrier.wait(timeout=5.0)
        return gathered

    monkeypatch.setattr(sps, "gather_seq", _gather)
    outs: dict[int, torch.Tensor] = {}
    errors: list[BaseException] = []

    def _run_rank(rank: int) -> None:
        local_rank.rank = rank
        try:
            outs[rank] = _Host()(hidden, cap, t, freqs_cis=freqs)
        except BaseException as exc:
            errors.append(exc)
            barrier.abort()

    threads = [threading.Thread(target=_run_rank, args=(r,)) for r in (0, 1)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=5.0)
        assert not thread.is_alive()
    assert not errors

    for rank in (0, 1):
        assert observed[rank]["noise_shapes"] == [(1, 32, 8)]
        assert observed[rank]["layer_calls"] == [((1, 48, 8), 0)]
        assert observed[rank]["events"][0] == ("gather", (1, 32, 8), 64)
        assert observed[rank]["events"][1] == ("final_layer", (1, 48, 8))
        assert observed[rank]["events"][2] == ("gather", (1, 48, 4), 96)
        torch.testing.assert_close(outs[rank], ref)


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


def test_tail_meta_max_seqlen_covers_pad_segment():
    # Degenerate short sequence: num_pad (3) > valid (1). FA requires
    # max_seqlen >= the longest segment, i.e. the pad block here.
    shard = SpShard(orig_len=1, local_len=1, num_pad=3, sp_size=4, sp_rank=3)
    meta = tail_attn_meta(shard, 1, torch.device("cpu"))
    assert meta["max_seqlen_tail"] == 3


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA graph capture")
def test_tail_meta_is_cuda_graph_capturable():
    """The meta is built inside the DiT forward, which breakable-cuda-graph
    captures; host-staged tensor construction aborts capture with 'Cannot copy
    between CPU and CUDA tensors'."""
    device = torch.device("cuda")
    shard = SpShard(orig_len=15, local_len=8, num_pad=1, sp_size=2, sp_rank=1)
    eager = tail_attn_meta(shard, 2, device, image_seq_len=100)

    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):
        tail_attn_meta(shard, 2, device, image_seq_len=100)
    torch.cuda.current_stream().wait_stream(side)

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured = tail_attn_meta(shard, 2, device, image_seq_len=100)
    graph.replay()
    torch.cuda.synchronize()
    assert torch.equal(captured["cu_seqlens_tail"], eager["cu_seqlens_tail"])


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


def test_strategy_replicates_when_padding_spans_multiple_shards(monkeypatch):
    _fake_sp(monkeypatch, 8)
    assert sps.plan_text_strategy(1) == "replicate"
    assert sps.plan_text_strategy(6) == "replicate"
    assert sps.plan_text_strategy(7) == "shard"
    assert sps.plan_text_strategy(9) == "replicate"
    assert sps.plan_text_strategy(13) == "replicate"
    assert sps.plan_text_strategy(14) == "shard"


def test_strategy_shards_padded_text_under_ring(monkeypatch):
    # Tail-padded shards ride the ring kernel now; the strategy no longer
    # consults the ring degree at all.
    _fake_sp(monkeypatch, 2)
    assert sps.plan_text_strategy(15) == "shard"
    assert sps.plan_text_strategy(16) == "shard"


def test_strategy_min_len_threshold(monkeypatch):
    _fake_sp(monkeypatch, 2)
    monkeypatch.setattr(sps, "_TEXT_SHARD_MIN", 64)
    assert sps.plan_text_strategy(32) == "replicate"
    assert sps.plan_text_strategy(64) == "shard"


# --- join_seqs / split_seqs / shard_seq_prefix ------------------------------


def test_join_split_roundtrip_with_pad():
    # Joint [text, image] with 2 tail-pad rows relocated behind the image.
    txt = torch.arange(6, dtype=torch.float32).view(1, 6, 1)  # rows 4,5 are pad
    img = (torch.arange(3, dtype=torch.float32) + 100).view(1, 3, 1)
    joint = sps.join_seqs(txt, img, local_pad=2)
    assert joint[0, :, 0].tolist() == [0, 1, 2, 3, 100, 101, 102, 4, 5]
    txt_back, img_back = sps.split_seqs(joint, prefix_len=6, local_pad=2)
    assert torch.equal(txt_back, txt) and torch.equal(img_back, img)


def test_join_split_roundtrip_no_pad():
    txt = torch.randn(1, 4, 2)
    img = torch.randn(1, 3, 2)
    joint = sps.join_seqs(txt, img, local_pad=0)
    assert torch.equal(joint, torch.cat([txt, img], dim=1))
    txt_back, img_back = sps.split_seqs(joint, prefix_len=4, local_pad=0)
    assert torch.equal(txt_back, txt) and torch.equal(img_back, img)


def test_shard_seq_prefix_only_touches_prefix():
    # Joint RoPE cache [txt(15); img(4)]: text segment shards, image stays.
    shard = SpShard(orig_len=15, local_len=8, num_pad=1, sp_size=2, sp_rank=1)
    cache = torch.arange(19, dtype=torch.float32).unsqueeze(-1)
    out = sps.shard_seq_prefix(cache, 15, shard, dim=0)
    assert out.shape[0] == 8 + 4
    assert out[0, 0].item() == 8.0  # rank1 text chunk starts at token 8
    assert out[-4:, 0].flatten().tolist() == [15, 16, 17, 18]  # image untouched


def test_should_shard_text_gate(monkeypatch):
    _fake_sp(monkeypatch, 2)
    assert sps.should_shard_text(15) is True
    _fake_sp(monkeypatch, 1)
    assert sps.should_shard_text(15) is False


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
    sys.exit(pytest.main([__file__, "-q"]))
