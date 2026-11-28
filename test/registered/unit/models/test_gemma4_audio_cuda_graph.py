"""Frame-bucket CUDA graphs of the Gemma 4 audio encoder."""

import sys

import pytest
import torch

from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=30, stage="base-b", runner_config="1-gpu-small")

from transformers import Gemma4AudioConfig

from sglang.srt.models import gemma4_audio
from sglang.srt.models.gemma4_audio import Gemma4AudioEncoder
from sglang.srt.runtime_context import get_parallel

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")

MEL_BINS = 128


def _single_rank():
    return get_parallel().override(
        tp_size=1,
        tp_rank=0,
        tp_group=None,
        attn_tp_size=1,
        attn_tp_rank=0,
        attn_tp_group=None,
    )


@pytest.fixture(scope="module")
def encoder():
    config = Gemma4AudioConfig(
        hidden_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        subsampling_conv_channels=[8, 4],
        output_proj_dims=32,
    )
    with _single_rank():
        enc = Gemma4AudioEncoder(config)
        torch.manual_seed(0)
        with torch.no_grad():
            for p in enc.parameters():
                p.normal_(0, 0.05)
        enc = enc.to("cuda", torch.bfloat16).eval()
        yield enc


def _clip(frames, valid=None):
    mel = torch.randn(1, frames, MEL_BINS, device="cuda", dtype=torch.bfloat16)
    mask = torch.zeros(1, frames, dtype=torch.bool, device="cuda")
    if valid is not None:
        mask[:, valid:] = True  # True = padding
    return mel, mask


def _padded_eager(encoder, mel, mask):
    frames = mel.shape[1]
    bucket = next(b for b in gemma4_audio._AUDIO_GRAPH_FRAME_BUCKETS if b >= frames)
    pad_mel = mel.new_zeros(1, bucket, MEL_BINS)
    pad_mel[:, :frames] = mel
    pad_mask = mask.new_ones(1, bucket)
    pad_mask[:, :frames] = mask
    return encoder._forward_eager(pad_mel, pad_mask)


@pytest.mark.parametrize("frames,valid", [(37, None), (128, None), (300, 251)])
def test_graph_replay_matches_padded_eager(encoder, frames, valid):
    mel, mask = _clip(frames, valid)
    with _single_rank(), torch.no_grad():
        graph_enc, graph_mask = encoder(mel, mask)
        ref_enc, ref_mask = _padded_eager(encoder, mel, mask)
        eager_enc, eager_mask = encoder._forward_eager(mel, mask)
    assert encoder._graphs
    assert torch.equal(graph_enc, ref_enc)
    assert torch.equal(graph_mask, ref_mask)
    # Same valid frames as the unpadded run; the bucket tail is padding.
    n = eager_mask.shape[1]
    assert torch.equal(graph_mask[:, :n], eager_mask)
    assert bool(graph_mask[:, n:].all())
    valid_rows = ~eager_mask
    torch.testing.assert_close(
        graph_enc[:, :n][valid_rows].float(),
        eager_enc[valid_rows].float(),
        atol=5e-2,
        rtol=5e-2,
    )


def test_no_stale_frames_after_longer_clip(encoder):
    long_mel, long_mask = _clip(1000)
    mel, mask = _clip(900, valid=700)  # same bucket, shorter clip
    with _single_rank(), torch.no_grad():
        encoder(long_mel, long_mask)
        graph_enc, _ = encoder(mel, mask)
        ref_enc, _ = _padded_eager(encoder, mel, mask)
    assert torch.equal(graph_enc, ref_enc)


def test_long_and_batched_inputs_run_eagerly(encoder):
    long_mel, long_mask = _clip(gemma4_audio._AUDIO_GRAPH_FRAME_BUCKETS[-1] + 5)
    batch_mel = torch.randn(2, 200, MEL_BINS, device="cuda", dtype=torch.bfloat16)
    batch_mask = torch.zeros(2, 200, dtype=torch.bool, device="cuda")
    with _single_rank(), torch.no_grad():
        for mel, mask in ((long_mel, long_mask), (batch_mel, batch_mask)):
            assert not encoder._cuda_graphs_enabled(mel)
            got, _ = encoder(mel, mask)
            ref, _ = encoder._forward_eager(mel, mask)
            assert torch.equal(got, ref)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__]))
