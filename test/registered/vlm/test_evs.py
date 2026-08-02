from dataclasses import asdict, dataclass
from types import SimpleNamespace

import pytest
import torch

from sglang.test.ci.ci_register import (
    register_amd_ci,
    register_cpu_ci,
    register_cuda_ci,
)
from sglang.test.test_utils import run_doctests

register_cuda_ci(est_time=11, stage="base-b", runner_config="1-gpu-small")
register_amd_ci(est_time=20, suite="stage-b-test-1-gpu-small-amd")
register_cpu_ci(est_time=8, suite="base-c-test-cpu")


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


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__]))
