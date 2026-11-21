import sys

import numpy as np
import pytest
import torch
from torch import nn

from sglang.srt.layers.attention.vision import (
    VisionFlash3Attention,
    prepare_vision_attention_metadata,
)
from sglang.srt.multimodal.vit_bucketed_cuda_graph_runner import (
    ViTBucketedCudaGraphRunner,
    parse_token_buckets,
)
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=30, stage="base-b", runner_config="1-gpu-large")

MERGE_UNIT = 4


def test_parse_token_buckets():
    assert parse_token_buckets("1024, 256,1024", MERGE_UNIT) == [256, 1024]
    assert parse_token_buckets("", MERGE_UNIT)[-1] == 4096
    with pytest.raises(ValueError):
        parse_token_buckets("258", MERGE_UNIT)


@pytest.mark.parametrize(
    "cu_seqlens",
    [[0, 4], [0, 64, 68, 200], [0, 1024], [0] + [4 * i for i in range(1, 257)]],
)
def test_padded_layout_isolates_padding_and_fits(cu_seqlens):
    bucket = 1024
    runner = ViTBucketedCudaGraphRunner(None, None, MERGE_UNIT, [bucket])
    layout = runner._seqlens(np.array(cu_seqlens, dtype=np.int32), bucket)
    max_seqs = bucket // MERGE_UNIT
    cu, seq_lens = layout[: max_seqs + 2], layout[max_seqs + 2 :]

    assert cu[: len(cu_seqlens)].tolist() == cu_seqlens
    assert cu[-1] == bucket and np.all(np.diff(cu) >= 0)
    assert np.array_equal(seq_lens, np.diff(cu))
    # Padding never joins a real sequence and is cut into short sequences.
    pad = seq_lens[len(cu_seqlens) - 1 :]
    assert pad.sum() == bucket - cu_seqlens[-1]
    assert pad.max(initial=0) <= 64 and np.all(pad % MERGE_UNIT == 0)


class _Tower(nn.Module):
    """Two varlen-attention blocks and a spatial merger, like Qwen3-VL."""

    def __init__(self, hidden=64, heads=2, out=32):
        super().__init__()
        self.heads = heads
        self.qkv = nn.ModuleList(nn.Linear(hidden, 3 * hidden) for _ in range(2))
        self.proj = nn.ModuleList(nn.Linear(hidden, hidden) for _ in range(2))
        self.merger = nn.Linear(hidden * MERGE_UNIT, out)
        self.attn = VisionFlash3Attention()
        self.use_data_parallel = True

    def blocks(self, x, cu_seqlens, cos, sin, metadata):
        seq_len, _, hidden = x.shape
        for qkv, proj in zip(self.qkv, self.proj):
            q, k, v = qkv(x[:, 0]).chunk(3, dim=-1)
            rope = cos.repeat(1, hidden // cos.shape[1]) + sin.sum(-1, keepdim=True)
            q = (q * rope).to(x.dtype)
            q, k, v = (t.reshape(seq_len, self.heads, -1) for t in (q, k, v))
            attn = self.attn(
                q, k, v, None, 1, seq_len, forward_metadata=metadata
            ).reshape(seq_len, hidden)
            x = x + proj(attn).unsqueeze(1)
        return self.merger(x.reshape(-1, hidden * MERGE_UNIT))


def _inputs(grids, device):
    lens = [h * w for h, w in grids]
    total = sum(lens)
    x = torch.randn(total, 64, device=device, dtype=torch.bfloat16)
    cos = torch.randn(total, 16, device=device, dtype=torch.float32)
    sin = torch.randn(total, 16, device=device, dtype=torch.float32)
    cu = np.concatenate([[0], np.cumsum(lens)]).astype(np.int32)
    return x, cu, cos, sin


def _eager(tower, x, cu, cos, sin):
    metadata = prepare_vision_attention_metadata(torch.from_numpy(cu), x.device)
    return tower.blocks(
        x.unsqueeze(1).clone(), metadata.cu_seqlens, cos, sin, metadata
    ).float()


@pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] != 9,
    reason="FA3 varlen attention needs SM90",
)
def test_bucketed_graph_matches_eager_for_mixed_layouts():
    torch.manual_seed(0)
    device = torch.device("cuda")
    tower = _Tower().to(device=device, dtype=torch.bfloat16)
    runner = ViTBucketedCudaGraphRunner(tower, tower.blocks, MERGE_UNIT, [256, 1024])

    layouts = [
        [(8, 8)],
        [(4, 6), (10, 12), (2, 2)],
        [(16, 16), (16, 32)],
        [(2, 2)] * 64,
        [(32, 32)],  # exactly fills the largest bucket
    ]
    kept = []
    with torch.no_grad():
        for grids in layouts:
            x, cu, cos, sin = _inputs(grids, device)
            out = runner.run(x, cu, cos, sin)
            ref = _eager(tower, x, cu, cos, sin)
            torch.testing.assert_close(out.float(), ref, rtol=1e-2, atol=1e-2)
            kept.append((out, out.clone()))
        assert len(runner.graphs) == 2

        # Later replays must not overwrite returned outputs.
        for out, snapshot in kept:
            assert torch.equal(out, snapshot)

        # Over budget: the caller falls back to eager.
        x, cu, cos, sin = _inputs([(32, 34)], device)
        assert runner.run(x, cu, cos, sin) is None


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
