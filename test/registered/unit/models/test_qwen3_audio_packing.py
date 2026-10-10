"""Regression tests for heterogeneous Qwen3 Omni/ASR audio batches."""

from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn

from sglang.srt.managers import mm_schedule
from sglang.srt.managers.schedule_batch import Modality, MultimodalDataItem
from sglang.srt.models.qwen3_asr import Qwen3ASRForConditionalGeneration
from sglang.srt.models.qwen3_omni_moe import (
    Qwen3OmniMoeThinkerForConditionalGeneration,
    _pack_audio_features,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


def _audio_item(features: torch.Tensor, mask: torch.Tensor) -> MultimodalDataItem:
    return MultimodalDataItem(
        modality=Modality.AUDIO,
        feature=features,
        model_specific_data={"feature_attention_mask": mask},
    )


class _RecordingAudioTower(nn.Module):
    def __init__(self):
        super().__init__()
        self.anchor = nn.Parameter(torch.empty(0), requires_grad=False)
        self.calls = []

    @property
    def dtype(self):
        return self.anchor.dtype

    def forward(self, input_features, feature_lens):
        self.calls.append((input_features.clone(), feature_lens.clone()))
        return SimpleNamespace(last_hidden_state=input_features.transpose(0, 1))


class _FakeQwen3OmniThinker:
    get_audio_feature = Qwen3OmniMoeThinkerForConditionalGeneration.get_audio_feature

    def __init__(self):
        self.audio_tower = _RecordingAudioTower()


class _FakeQwen3ASR:
    get_audio_feature = Qwen3ASRForConditionalGeneration.get_audio_feature

    def __init__(self):
        self.audio_tower = _RecordingAudioTower()


def test_pack_audio_features_with_heterogeneous_request_lengths():
    features_900 = torch.arange(2 * 900, dtype=torch.float32).reshape(1, 2, 900)
    features_700 = (
        torch.arange(2 * 700, dtype=torch.float32).reshape(1, 2, 700) + 10_000
    )
    items = [
        _audio_item(features_900, torch.ones(1, 900, dtype=torch.long)),
        _audio_item(features_700, torch.ones(1, 700, dtype=torch.long)),
    ]

    packed, feature_lens = _pack_audio_features(
        items, device=torch.device("cpu"), dtype=torch.float32
    )

    assert feature_lens.tolist() == [900, 700]
    assert packed.shape == (2, 1600)
    torch.testing.assert_close(
        packed,
        torch.cat([features_900[0], features_700[0]], dim=1),
    )


def test_cross_request_mm_batch_encodes_heterogeneous_audio_once():
    """Exercise the scheduler path that exposed request-local padding shapes."""
    features_900 = torch.arange(2 * 900, dtype=torch.float32).reshape(1, 2, 900)
    features_700 = (
        torch.arange(2 * 700, dtype=torch.float32).reshape(1, 2, 700) + 10_000
    )
    item_900 = _audio_item(features_900, torch.ones(1, 900, dtype=torch.long))
    item_700 = _audio_item(features_700, torch.ones(1, 700, dtype=torch.long))
    item_900.hash = 900
    item_700.hash = 700

    requests = [
        mm_schedule.PerImageRequestInfo(
            req_idx=0,
            items=[item_900],
            items_offset=[(0, 899)],
            extend_prefix_len=0,
            extend_seq_len=900,
        ),
        mm_schedule.PerImageRequestInfo(
            req_idx=1,
            items=[item_700],
            items_offset=[(0, 699)],
            extend_prefix_len=0,
            extend_seq_len=700,
        ),
    ]
    model = _FakeQwen3OmniThinker()
    mm_schedule.init_mm_embedding_cache(1 << 20)

    embeddings = mm_schedule._batch_encode_per_image_misses(
        model.get_audio_feature,
        requests,
        torch.device("cpu"),
    )

    assert len(model.audio_tower.calls) == 1
    packed, feature_lens = model.audio_tower.calls[0]
    assert packed.shape == (2, 1600)
    assert feature_lens.tolist() == [900, 700]
    torch.testing.assert_close(
        packed,
        torch.cat([features_900[0], features_700[0]], dim=1),
    )
    torch.testing.assert_close(embeddings[900], features_900[0].transpose(0, 1))
    torch.testing.assert_close(embeddings[700], features_700[0].transpose(0, 1))


def test_qwen3_asr_packs_multi_sample_and_heterogeneous_items_once():
    features_900 = torch.arange(4 * 900, dtype=torch.float32).reshape(2, 2, 900)
    features_700 = (
        torch.arange(2 * 700, dtype=torch.float32).reshape(1, 2, 700) + 10_000
    )
    item_900 = _audio_item(features_900, torch.ones(2, 900, dtype=torch.long))
    item_700 = _audio_item(features_700, torch.ones(1, 700, dtype=torch.long))
    item_900.hash = 1_800
    item_700.hash = 700

    requests = [
        mm_schedule.PerImageRequestInfo(
            req_idx=0,
            items=[item_900],
            items_offset=[(0, 1799)],
            extend_prefix_len=0,
            extend_seq_len=1800,
        ),
        mm_schedule.PerImageRequestInfo(
            req_idx=1,
            items=[item_700],
            items_offset=[(0, 699)],
            extend_prefix_len=0,
            extend_seq_len=700,
        ),
    ]
    model = _FakeQwen3ASR()
    mm_schedule.init_mm_embedding_cache(1 << 20)

    embeddings = mm_schedule._batch_encode_per_image_misses(
        model.get_audio_feature,
        requests,
        torch.device("cpu"),
    )

    assert len(model.audio_tower.calls) == 1
    packed, feature_lens = model.audio_tower.calls[0]
    expected_900 = torch.cat([features_900[0], features_900[1]], dim=1)
    assert packed.shape == (2, 2500)
    assert feature_lens.tolist() == [900, 900, 700]
    torch.testing.assert_close(
        packed,
        torch.cat([expected_900, features_700[0]], dim=1),
    )
    torch.testing.assert_close(embeddings[1_800], expected_900.transpose(0, 1))
    torch.testing.assert_close(embeddings[700], features_700[0].transpose(0, 1))


def test_pack_audio_features_preserves_equal_length_behavior():
    features = [
        torch.arange(12, dtype=torch.float32).reshape(1, 3, 4),
        torch.arange(12, 24, dtype=torch.float32).reshape(1, 3, 4),
    ]
    masks = [
        torch.tensor([[1, 1, 0, 0]], dtype=torch.long),
        torch.tensor([[1, 0, 1, 0]], dtype=torch.long),
    ]
    items = [_audio_item(feature, mask) for feature, mask in zip(features, masks)]

    packed, feature_lens = _pack_audio_features(
        items, device=torch.device("cpu"), dtype=torch.float32
    )

    old_features = torch.cat(features, dim=0)
    old_mask = torch.cat(masks, dim=0).bool()
    expected = old_features.permute(0, 2, 1)[old_mask].transpose(0, 1)
    torch.testing.assert_close(packed, expected)
    assert feature_lens.tolist() == [2, 2]


@pytest.mark.parametrize(
    ("features", "mask", "message"),
    [
        (torch.zeros(2, 4), torch.ones(2, 4), "features must have shape"),
        (torch.zeros(1, 2, 4), torch.ones(1, 1, 4), "attention mask must have shape"),
        (torch.zeros(1, 2, 4), torch.ones(2, 4), "batch size mismatch"),
        (torch.zeros(1, 2, 4), torch.ones(1, 3), "time dimension mismatch"),
    ],
)
def test_pack_audio_features_validates_item_shapes(features, mask, message):
    with pytest.raises(ValueError, match=message):
        _pack_audio_features(
            [_audio_item(features, mask)],
            device=torch.device("cpu"),
            dtype=torch.float32,
        )


def test_pack_audio_features_rejects_mismatched_mel_dimensions():
    items = [
        _audio_item(torch.zeros(1, 2, 4), torch.ones(1, 4)),
        _audio_item(torch.zeros(1, 3, 4), torch.ones(1, 4)),
    ]

    with pytest.raises(ValueError, match="mel dimension mismatch"):
        _pack_audio_features(
            items,
            device=torch.device("cpu"),
            dtype=torch.float32,
        )
