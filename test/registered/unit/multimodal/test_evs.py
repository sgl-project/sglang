from dataclasses import asdict, dataclass
from types import SimpleNamespace

import pytest
import torch

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import run_doctests

register_cpu_ci(est_time=12, suite="base-a-test-cpu")


def test_resolve_evs_config():
    from sglang.srt.multimodal.evs import EVS, EVSConfig, EVSProcessor

    @dataclass(frozen=True, kw_only=True)
    class EVSModelConfig:
        video_pruning_rate: float = 0.1
        spatial_merge_size: int = 2

    class EVSModel(EVS):
        @staticmethod
        def create_evs_config(hf_config: EVSModelConfig) -> EVSConfig:
            return EVSConfig(
                video_pruning_rate=hf_config.video_pruning_rate,
                spatial_merge_size=hf_config.spatial_merge_size,
            )

    processor = EVSProcessor(
        hf_config=EVSModelConfig(spatial_merge_size=3),
        config_to_evs_model={EVSModelConfig: EVSModel},
    )
    expected = EVSConfig(video_pruning_rate=0.1, spatial_merge_size=3)
    assert asdict(processor.evs_config) == asdict(expected)

    # No EVS for pruning rate 0.0
    processor = EVSProcessor(
        hf_config=EVSModelConfig(video_pruning_rate=0.0),
        config_to_evs_model={EVSModelConfig: EVSModel},
    )
    assert processor.evs_config is None

    # No EVS for non-EVS config
    processor = EVSProcessor(
        hf_config=SimpleNamespace(),
        config_to_evs_model={EVSModelConfig: EVSModel},
    )
    assert processor.evs_config is None


def test_replace_offsets_with_tokens_per_frame():
    from sglang.srt.multimodal.evs.evs_core import replace_offsets_with_tokens_per_frame

    run_doctests(replace_offsets_with_tokens_per_frame)


def test_evs_items_store_wire_data_in_model_specific_data():
    from sglang.srt.managers.schedule_batch import MultimodalDataItem
    from sglang.srt.multimodal.evs import EVSConfig, EVSProcessor

    processor = EVSProcessor.__new__(EVSProcessor)
    processor.evs_config = EVSConfig(video_pruning_rate=0.1)
    make_items, _ = processor.static_size_data_items(
        frames_per_video=[2], num_images=1, rows=2, cols=3
    )
    items = make_items(
        input_ids_list=[1, 2, 3],
        image=torch.zeros(1),
        image_offsets=[(0, 0)],
        video=torch.zeros(1),
        video_offsets=[(1, 2)],
    )

    assert all(type(item) is MultimodalDataItem for item in items)
    assert items[0].thw_grids == [(1, 2, 3)]
    assert items[1].thw_grids == [(2, 2, 3)]
    assert items[1].pre_chunked_input_ids == [1, 2, 3]


def test_single_frame_videos_are_returned_unpruned_as_a_tensor():
    """Videos of one frame (or one tubelet) get a single placeholder span, so the
    scheduler batches them like images and expects a plain tensor back; EVS must
    not wrap or reject them, including when several such items are batched."""
    from sglang.srt.managers.schedule_batch import Modality, MultimodalDataItem
    from sglang.srt.multimodal.evs import EVS, EVSConfig

    def video_item():
        return MultimodalDataItem(
            modality=Modality.VIDEO,
            feature=torch.zeros(1),
            model_specific_data={"thw_grids": [(1, 2, 2)], "pre_chunked_input_ids": []},
        )

    first, second = video_item(), video_item()
    features = {id(first): torch.randn(1, 4, 8), id(second): torch.randn(1, 4, 8)}

    class EVSModel(EVS):
        @staticmethod
        def create_evs_config(hf_config) -> EVSConfig:
            return EVSConfig(video_pruning_rate=hf_config.video_pruning_rate)

        def get_video_feature(self, items):
            return torch.cat([features[id(item)] for item in items])

    model = EVSModel(SimpleNamespace(video_pruning_rate=0.7))

    result = model.get_video_feature([first, second])

    assert isinstance(result, torch.Tensor)
    torch.testing.assert_close(
        result, torch.cat([features[id(first)], features[id(second)]])
    )


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__]))
