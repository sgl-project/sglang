"""Qwen3-VL family videos on the token-space route are sampled and resized as qwen-vl-utils 0.0.14 does.

synthetic mp4 -> preprocess_qwen3_vl_video_sync -> frame indices, size == qwen-vl-utils 0.0.14 (pinned)
                                                 -> pixels == BICUBIC + antialias resize of the decoded frames
synthetic mp4 -> QwenTokenSpaceProcessStrategy.process_video
              == those frames through Qwen3VLVideoProcessor(do_resize=False)

The pinned values come from qwen-vl-utils 0.0.14 fetch_video(image_patch_size=16)
with its torchcodec reader on this video. Configs cover fps and nframes sampling,
the total_pixels and min_pixels bounds, and an explicit resized size.
"""

import sys

import av
import numpy as np
import pytest
import torch
from tokenizers import Tokenizer, models
from torchvision.transforms import InterpolationMode
from torchvision.transforms.functional import resize
from transformers import (
    PreTrainedTokenizerFast,
    Qwen2VLImageProcessor,
    Qwen3VLConfig,
    Qwen3VLProcessor,
    Qwen3VLVideoProcessor,
)

from sglang.srt.multimodal.qwen_vl_media_processing import (
    preprocess_qwen3_vl_video_sync,
)
from sglang.srt.multimodal.token_space_mm_process.models.qwen_vl import (
    QwenTokenSpaceProcessStrategy,
)
from sglang.srt.utils.video_decoder import VideoDecoderWrapper
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=20, suite="base-a-test-cpu")

# Qwen3-VL vision: 16-pixel patches, 2x2 spatial merge.
_PATCH_SIZE = 16
_MERGE_SIZE = 2
# (config, qwen-vl-utils 0.0.14 frames_indices, resized (height, width)); 72 frames at 24 fps.
_CASES = [
    ({"fps": 2.0}, [0, 14, 28, 43, 57, 71], (288, 480)),
    ({"nframes": 6}, [0, 14, 28, 43, 57, 71], (288, 480)),
    (
        {
            "fps": 4.0,
            "max_frames": 10,
            "min_pixels": 4 * 32 * 32,
            "total_pixels": 20 * 32 * 32,
        },
        [0, 8, 16, 24, 32, 39, 47, 55, 63, 71],
        (32, 64),
    ),
    ({"fps": 1.0, "min_pixels": 300 * 32 * 32}, [0, 24, 47, 71], (448, 736)),
    (
        {"nframes": 4, "resized_height": 100, "resized_width": 150},
        [0, 24, 47, 71],
        (96, 160),
    ),
]


@pytest.fixture(scope="module")
def video_path(tmp_path_factory):
    # Moving gradients at a size that is no multiple of 32 and not square.
    path = tmp_path_factory.mktemp("video") / "gradient.mp4"
    height, width, num_frames = 130, 210, 72
    rows, cols = np.mgrid[0:height, 0:width]
    with av.open(str(path), "w") as container:
        stream = container.add_stream("mpeg4", rate=24)
        stream.height, stream.width, stream.pix_fmt = height, width, "yuv420p"
        for index in range(num_frames):
            frame = np.stack(
                [
                    (rows * 2 + index * 3) % 256,
                    (cols + index * 5) % 256,
                    (rows + cols + index * 7) % 256,
                ],
                axis=-1,
            ).astype(np.uint8)
            container.mux(stream.encode(av.VideoFrame.from_ndarray(frame, "rgb24")))
        container.mux(stream.encode())
    return str(path)


def _reference_frames(video_path, indices, size):
    decoder = VideoDecoderWrapper(video_path)
    frames = torch.as_tensor(decoder.get_frames_at(indices)).permute(0, 3, 1, 2)
    return resize(
        frames, list(size), interpolation=InterpolationMode.BICUBIC, antialias=True
    ).float()


@pytest.mark.parametrize("config, indices, size", _CASES, ids=str)
def test_frames_match_qwen_vl_utils(video_path, config, indices, size):
    frames, metadata = preprocess_qwen3_vl_video_sync(
        VideoDecoderWrapper(video_path),
        image_factor=_PATCH_SIZE * _MERGE_SIZE,
        video_config=config,
    )

    assert metadata["frames_indices"] == indices
    assert (metadata["fps"], metadata["total_num_frames"]) == (24, 72)
    assert torch.equal(frames, _reference_frames(video_path, indices, size))


def _hf_processor():
    tokens = ["[UNK]", "[PAD]", "<|image_pad|>", "<|video_pad|>"]
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=Tokenizer(
            models.WordLevel(
                {token: index for index, token in enumerate(tokens)},
                unk_token="[UNK]",
            )
        ),
        unk_token="[UNK]",
        pad_token="[PAD]",
        additional_special_tokens=tokens[2:],
    )
    return Qwen3VLProcessor(
        tokenizer=tokenizer,
        image_processor=Qwen2VLImageProcessor(
            patch_size=_PATCH_SIZE, merge_size=_MERGE_SIZE
        ),
        video_processor=Qwen3VLVideoProcessor(
            patch_size=_PATCH_SIZE, merge_size=_MERGE_SIZE
        ),
    )


@pytest.mark.parametrize("config, indices, size", _CASES, ids=str)
def test_processed_video_takes_the_frames_unresized(video_path, config, indices, size):
    hf_processor = _hf_processor()
    strategy = QwenTokenSpaceProcessStrategy(Qwen3VLConfig(), hf_processor)
    output = strategy.process_video(
        VideoDecoderWrapper(video_path), hf_processor, {}, **config
    )

    reference = hf_processor.video_processor(
        [_reference_frames(video_path, indices, size)],
        video_metadata=[{"fps": 24, "frames_indices": indices, "total_num_frames": 72}],
        do_sample_frames=False,
        do_resize=False,
        return_metadata=True,
    )
    for key in ("pixel_values_videos", "video_grid_thw"):
        assert torch.equal(output[key], reference[key]), key


def test_item_size_within_the_qwen3_vl_budget_is_accepted(video_path):
    # 640 x 960 is above the Qwen2-VL default cap (768 * 28 * 28) and within Qwen3-VL's (768 * 32 * 32).
    hf_processor = _hf_processor()
    strategy = QwenTokenSpaceProcessStrategy(Qwen3VLConfig(), hf_processor)
    output = strategy.process_video(
        VideoDecoderWrapper(video_path),
        hf_processor,
        {"nframes": 4, "resized_height": 640, "resized_width": 960},
    )

    assert output["video_grid_thw"].tolist() == [[2, 40, 60]]


if __name__ == "__main__":
    sys.exit(pytest.main([__file__]))
