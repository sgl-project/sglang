"""Mixed-input ordering and configuration round-trip regressions for LFM3 Omni."""

import pytest
import torch

from sglang.srt.configs.lfm3_omni import Lfm3OmniConfig
from sglang.srt.managers.schedule_batch import Modality
from sglang.srt.multimodal.processors.lfm3_omni import Lfm3OmniMultimodalProcessor
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=10, suite="base-a-test-cpu")


@pytest.mark.parametrize("edge_audio", [False, True])
def test_audio_clips_keep_disjoint_offsets_and_drop_padding(
    edge_audio, lfm3_audio_reference
):
    from sglang.srt.managers.mm_utils import get_new_expanded_mm_items

    processor = object.__new__(Lfm3OmniMultimodalProcessor)
    processor.feature_extractor = lfm3_audio_reference[1]
    processor.audio_token_id = 1
    processor.ATTR_NAME_TO_MODALITY = {
        "pixel_values": Modality.IMAGE,
        "image_grid_thw": Modality.IMAGE,
    }
    processor.FEATURE_NAMES = ["pixel_values", "input_features"]
    features = torch.arange(2 * 128 * 25).reshape(2, 128, 25).float()
    mask = torch.zeros(2, 25, dtype=torch.long)
    mask[0, :25] = 1
    mask[1, :17] = 1
    data = {
        "input_ids": torch.tensor([[0, 1, 1, 2, 1, 1, 0, 1, 1, 1, 0]]),
        "input_features": features,
        "feature_attention_mask": mask,
        "pixel_values": torch.ones(4, 8),
        "image_grid_thw": torch.tensor([[1, 2, 2]]),
    }
    if edge_audio:
        data["input_ids"] = data["input_ids"][:, 1:-1]
    items = processor.collect_mm_items_from_processor_output(data)
    assert items[0].modality == Modality.IMAGE
    shift = int(edge_audio)
    assert [item.offsets for item in items[1:]] == [
        [(1 - shift, 2 - shift), (4 - shift, 5 - shift)],
        [(7 - shift, 9 - shift)],
    ]
    torch.testing.assert_close(items[1].feature, features[:1])
    torch.testing.assert_close(items[2].feature, features[1:2, :, :17])
    assert get_new_expanded_mm_items(items[1:]) == items[1:]
    assert data["input_features"].shape == (2, 128, 25)


def test_config_round_trip_preserves_modalities(lfm3_omni_reference):
    reference = type(lfm3_omni_reference)(
        architectures=["Lfm3OmniForConditionalGeneration"], projector_hidden_size=1024
    )
    config = Lfm3OmniConfig.from_hf_config(reference)
    restored = Lfm3OmniConfig.from_dict(config.to_dict())
    assert restored.architectures == ["Lfm3OmniForConditionalGeneration"]
    assert restored.text_config.to_dict() == config.text_config.to_dict()
    assert restored.vision_config.to_dict() == config.vision_config.to_dict()
    assert restored.projector_hidden_size == 1024
    assert restored.eos_token_id == [
        restored.text_config.eos_token_id,
        restored.text_end_token_id,
    ]
    assert (
        restored.audio_encoder_config.to_dict() == config.audio_encoder_config.to_dict()
    )


def test_interleaved_soundtrack_is_not_split_into_mel_bins():
    from sglang.srt.managers.mm_utils import get_new_expanded_mm_items
    from sglang.srt.managers.schedule_batch import MultimodalDataItem

    # As many disjoint spans as mel bins must still represent one audio clip.
    item = MultimodalDataItem(
        modality=Modality.AUDIO,
        feature=torch.zeros(1, 128, 2048),
        offsets=[(i * 3, i * 3 + 1) for i in range(128)],
    )
    expanded = get_new_expanded_mm_items([item])
    assert len(expanded) == 1
    assert expanded[0].feature.shape == (1, 128, 2048)
    assert expanded[0].offsets == item.offsets
