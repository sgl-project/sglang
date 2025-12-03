from dataclasses import asdict, dataclass
from types import SimpleNamespace

import pytest


def test_resolve_evs_config():
    from sglang.srt.multimodal.evs import EVS, EVSConfig, EVSProcessor

    class EVSModel(EVS):
        @classmethod
        def create_evs_config(cls, hf_config):
            return EVSConfig(
                video_pruning_rate=hf_config.video_pruning_rate,
                temporal_patch_size=getattr(hf_config, "temporal_patch_size", 1),
            )

    class NonEVSModel:
        pass

    models = [EVSModel, NonEVSModel]

    processor = EVSProcessor(
        hf_config=SimpleNamespace(
            model_type="EVSModel",
            video_pruning_rate=0.1,
            temporal_patch_size=2,
        ),
        models=models,
    )
    expected = EVSConfig(
        video_pruning_rate=0.1, spatial_merge_size=1, temporal_patch_size=2
    )
    assert asdict(processor.evs_config) == asdict(expected)

    # No EVS for pruning rate 0.0
    processor = EVSProcessor(
        hf_config=SimpleNamespace(model_type="EVSModel", video_pruning_rate=0.0),
        models=models,
    )
    assert processor.evs_config is None

    # No EVS for non-EVS model
    processor = EVSProcessor(
        hf_config=SimpleNamespace(model_type="NonEVSModel"), models=models
    )
    assert processor.evs_config is None


@dataclass(kw_only=True)
class Case:
    input_ids: list[int]
    frame_offsets_inclusive: list[tuple[int, int]]
    num_tokens_per_frame: list[int]
    expected_output_ids: list[int]


FILL = 0

# fmt: off
@pytest.mark.parametrize("case", [
    Case(
        input_ids=[1, FILL, FILL, 4, 5, FILL, FILL, FILL, 9, 10, FILL, FILL, 12, 13],
        expected_output_ids=[1, FILL, 4, 5, FILL, FILL, FILL, FILL, 9, 10, FILL, FILL, 12, 13],
        frame_offsets_inclusive=[(1, 2), (5, 7), (10, 11)],
        num_tokens_per_frame=[1, 4, 2],
    ),
    Case(
        input_ids=[1, FILL, FILL, 4, 5, 9, 10, FILL, FILL, FILL],
        expected_output_ids=[1, FILL, 4, 5, 9, 10, FILL, FILL, FILL, FILL],
        frame_offsets_inclusive=[(1, 2), (7, 9)],
        num_tokens_per_frame=[1, 4],
    ),
    Case(
        input_ids=[FILL, FILL, 1, 4, FILL, FILL, FILL, 5, 9, 10],
        expected_output_ids=[FILL, 1, 4, FILL, FILL, FILL, FILL, 5, 9, 10],
        frame_offsets_inclusive=[(0, 1), (4, 6)],
        num_tokens_per_frame=[1, 4],
    ),
]) # fmt: off
def test_redistribute_placeholder_tokens_by_tokens_per_frame(case: Case):
    import torch

    from sglang.srt.multimodal.evs import (
        redistribute_placeholder_tokens_by_tokens_per_frame,
    )

    output_ids = redistribute_placeholder_tokens_by_tokens_per_frame(input_ids=torch.tensor(case.input_ids), frame_offsets_inclusive=case.frame_offsets_inclusive, num_tokens_per_frame=case.num_tokens_per_frame)
    assert output_ids.tolist() == case.expected_output_ids

if __name__ == "__main__":
    pytest.main([__file__])
