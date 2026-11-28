"""Host-side chunk/window planning of the Qwen3-Omni / Qwen3-ASR audio tower."""

import random
import sys
from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F

from sglang.srt.models.qwen3_omni_moe import (
    Qwen3OmniMoeAudioEncoder,
    _pack_audio_features,
)
from sglang.srt.multimodal.mrope_rope_index import _get_feat_extract_output_lengths
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=10, suite="base-a-test-cpu")


def _reference_plan(feature_lens, n_window, n_window_infer):
    """The tensor math the encoder ran on the device before host planning."""
    feature_lens = torch.tensor(feature_lens, dtype=torch.long)
    aftercnn_lens = _get_feat_extract_output_lengths(feature_lens)
    chunk_num = torch.ceil(feature_lens / (n_window * 2)).long()
    chunk_lengths = torch.tensor([n_window * 2] * chunk_num.sum(), dtype=torch.long)
    tail_chunk_index = F.pad(chunk_num, (1, 0), value=-1).cumsum(0)[1:]
    chunk_lengths[tail_chunk_index] = feature_lens % (n_window * 2)
    chunk_lengths[chunk_lengths == 0] = n_window * 2
    chunk_out_lens = _get_feat_extract_output_lengths(chunk_lengths)
    max_len_after_cnn = int(chunk_out_lens.max())
    window = max_len_after_cnn * (n_window_infer // (n_window * 2))
    cu_chunk_lens = [0]
    for cnn_len in aftercnn_lens.tolist():
        cu_chunk_lens.extend([window] * (cnn_len // window))
        if cnn_len % window:
            cu_chunk_lens.append(cnn_len % window)
    cu_seqlens = torch.tensor(cu_chunk_lens).cumsum(-1).tolist()
    seq_lens = [b - a for a, b in zip(cu_seqlens, cu_seqlens[1:])]
    return (
        chunk_lengths.tolist(),
        chunk_out_lens.tolist(),
        max_len_after_cnn,
        cu_seqlens,
        max(seq_lens),
    )


@pytest.mark.parametrize("n_window,n_window_infer", [(50, 800), (100, 400)])
def test_plan_windows_matches_device_math(n_window, n_window_infer):
    encoder = SimpleNamespace(n_window=n_window, n_window_infer=n_window_infer)
    rng = random.Random(0)
    chunk = n_window * 2
    cases = [[1], [chunk], [chunk + 1], [chunk * 8], [chunk * 8 + 3, 7, chunk - 1]]
    cases += [
        [rng.randint(1, 6000) for _ in range(rng.randint(1, 12))] for _ in range(200)
    ]
    for lens in cases:
        plan = Qwen3OmniMoeAudioEncoder.plan_windows(encoder, lens)
        expected = _reference_plan(lens, n_window, n_window_infer)
        got = (
            plan.chunk_lengths,
            plan.chunk_out_lens,
            plan.max_len_after_cnn,
            plan.cu_seqlens,
            plan.max_seqlen,
        )
        assert got == expected, lens


def _reference_pack(features, mask):
    lengths = mask.sum(-1)
    packed = features.permute(0, 2, 1)[mask.bool()].permute(1, 0)
    return packed, lengths


def test_pack_audio_features_matches_boolean_mask():
    torch.manual_seed(0)
    lengths = [3000, 17, 2999, 1, 1234]
    features = torch.randn(len(lengths), 128, 3000)
    mask = torch.zeros(len(lengths), 3000, dtype=torch.long)
    for i, n in enumerate(lengths):
        mask[i, :n] = 1
    packed, lens, lens_cpu = _pack_audio_features(
        [SimpleNamespace(feature=features, feature_attention_mask=mask)],
        device=torch.device("cpu"),
        dtype=torch.bfloat16,
    )
    expected, expected_lens = _reference_pack(features.to(torch.bfloat16), mask)
    assert torch.equal(packed, expected)
    assert torch.equal(lens, expected_lens)
    assert lens_cpu == lengths


def test_pack_audio_features_left_padded_mask_falls_back():
    torch.manual_seed(0)
    features = torch.randn(2, 8, 10)
    mask = torch.ones(2, 10, dtype=torch.long)
    mask[0, :3] = 0
    packed, _, lens_cpu = _pack_audio_features(
        [SimpleNamespace(feature=features, feature_attention_mask=mask)],
        device=torch.device("cpu"),
        dtype=torch.float32,
    )
    expected, _ = _reference_pack(features, mask)
    assert torch.equal(packed, expected)
    assert lens_cpu == [7, 10]


if __name__ == "__main__":
    sys.exit(pytest.main([__file__]))
