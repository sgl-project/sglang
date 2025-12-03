from dataclasses import asdict, dataclass
from types import SimpleNamespace

import pytest


def test_resolve_evs_config():
    from sglang.srt.multimodal.evs import (
        EVS,
        EVSConfig,
        EVSProcessorMixin,
        NonEVSConfig,
    )

    class EVSModel(EVS):
        @classmethod
        def create_evs_config(cls, hf_config):
            return EVSConfig(
                video_pruning_rate=hf_config.video_pruning_rate,
                full_frame_num_tokens=256,
            )

    class NonEVSModel:
        pass

    class Processor(EVSProcessorMixin):
        @staticmethod
        def create_non_evs_config(*args):
            return NonEVSConfig(frame_num_tokens=256)

        models = [EVSModel, NonEVSModel]

    processor = Processor(
        SimpleNamespace(model_type=EVSModel.__name__, video_pruning_rate=0.1)
    )
    assert asdict(processor.evs_config) == {
        "video_pruning_rate": 0.1,
        "full_frame_num_tokens": 256,
    }
    assert asdict(processor.non_evs_config) == {"frame_num_tokens": 256}

    # No EVS for pruning rate 0.0
    processor = Processor(
        SimpleNamespace(model_type=EVSModel.__name__, video_pruning_rate=0.0)
    )
    assert processor.evs_config is None
    assert asdict(processor.non_evs_config) == {"frame_num_tokens": 256}

    # No EVS for non-EVS model
    processor = Processor(SimpleNamespace(model_type=NonEVSModel.__name__))
    assert processor.evs_config is None
    assert asdict(processor.non_evs_config) == {"frame_num_tokens": 256}


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
