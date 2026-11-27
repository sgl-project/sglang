# Copyright 2023-2026 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

import array
import asyncio
import json
import os
import tempfile
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest
import torch
from PIL import Image
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import WhitespaceSplit
from transformers import PreTrainedTokenizerFast
from transformers.video_utils import VideoMetadata

from sglang.srt.configs.muse_glimmer import MuseGlimmerConfig, MuseGlimmerVisionConfig
from sglang.srt.configs.muse_glimmer_processing import (
    MuseGlimmerImageProcessor,
    MuseGlimmerProcessor,
    MuseGlimmerVideoProcessor,
)
from sglang.srt.managers.io_struct import VideoData
from sglang.srt.managers.mm_schedule import init_mm_embedding_cache
from sglang.srt.managers.mm_utils import embed_mm_inputs, pad_mm_tokens
from sglang.srt.managers.req_types import MultimodalInputs
from sglang.srt.models.muse_glimmer import (
    MuseGlimmerForConditionalGeneration,
    MuseGlimmerVisionModel,
)
from sglang.srt.multimodal.modality import Modality
from sglang.srt.multimodal.processors import base_processor as base_processor_mod
from sglang.srt.multimodal.processors.muse_glimmer import (
    MuseGlimmerMultimodalProcessor,
)
from sglang.srt.runtime_context import publish, reset_context
from sglang.srt.server_args import ServerArgs
from sglang.srt.utils.video_decoder import VideoDecoderWrapper
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=15, suite="base-a-test-cpu")


@pytest.fixture(autouse=True)
def _reset_runtime_context():
    reset_context()
    yield
    reset_context()


def _build_tokenizer() -> PreTrainedTokenizerFast:
    vocab = {
        "<unk>": 0,
        "<pad>": 1,
        "describe": 2,
        "this": 3,
        "image": 4,
        "video": 5,
        "and": 6,
        "Time:": 7,
        "0.0s": 8,
        "0.5s": 9,
        "1.0s": 10,
        "1.5s": 11,
        "2.0s": 12,
        "3.0s": 13,
    }
    specials = [
        "<|patch|>",
        "<|image_start|>",
        "<|image_end|>",
        "<|video|>",
        "<|vid_start|>",
        "<|vid_end|>",
        "<|vid_frame_separator|>",
    ]
    for idx, token in enumerate(specials, start=len(vocab)):
        vocab[token] = idx

    backend = Tokenizer(WordLevel(vocab=vocab, unk_token="<unk>"))
    backend.pre_tokenizer = WhitespaceSplit()
    return PreTrainedTokenizerFast(
        tokenizer_object=backend,
        unk_token="<unk>",
        pad_token="<pad>",
        additional_special_tokens=specials,
    )


def _build_processor(
    video_kwargs: dict | None = None,
    image_kwargs: dict | None = None,
) -> MuseGlimmerProcessor:
    tokenizer = _build_tokenizer()
    image_processor = MuseGlimmerImageProcessor(**(image_kwargs or {}))
    video_processor = MuseGlimmerVideoProcessor(**(video_kwargs or {}))
    return MuseGlimmerProcessor(
        image_processor=image_processor,
        video_processor=video_processor,
        tokenizer=tokenizer,
    )


def _build_mm_processor(
    processor: MuseGlimmerProcessor | None = None,
    mm_process_config: dict | None = None,
) -> MuseGlimmerMultimodalProcessor:
    processor = processor or _build_processor()
    hf_config = MuseGlimmerConfig(
        image_token_id=processor.image_token_id,
        video_token_id=processor.video_token_id,
        vision_config=MuseGlimmerVisionConfig(),
    )
    server_args = ServerArgs(
        model_path="dummy",
        mm_process_config=mm_process_config or {},
    )
    publish(server_args, role="test")
    return MuseGlimmerMultimodalProcessor(
        hf_config=hf_config,
        server_args=server_args,
        _processor=processor,
        transport_mode="default",
        skip_mm_pool=True,
    )


class _InMemoryVideo(VideoDecoderWrapper):
    """In-memory VideoDecoderWrapper that tracks accessed indices and close()."""

    def __init__(
        self, frames: np.ndarray, fps: float | None, fail_on_decode: bool = False
    ):
        self._frames = frames
        self._fps = fps
        self._fail_on_decode = fail_on_decode
        self.decoded_indices: list[list[int]] = []
        self.closed = False

    def __len__(self):
        return len(self._frames)

    def __getitem__(self, idx):
        return self._frames[idx]

    @property
    def avg_fps(self):
        return self._fps

    def get_frames_at(self, indices):
        self.decoded_indices.append(list(indices))
        if self._fail_on_decode:
            raise RuntimeError("simulated decode failure")
        return self._frames[indices]

    def get_frames_as_tensor(self, indices):
        self.decoded_indices.append(list(indices))
        if self._fail_on_decode:
            raise RuntimeError("simulated decode failure")
        return torch.from_numpy(self._frames[indices])

    def close(self):
        self.closed = True


def _synthetic_frames(
    num_frames: int, height: int, width: int, seed: int = 0
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.integers(0, 256, (num_frames, height, width, 3), dtype=np.uint8)


# ---------------------------------------------------------------------------
# MuseGlimmerVideoProcessor unit tests
# ---------------------------------------------------------------------------


def test_video_processor_sample_frames_fps_cap_and_alignment():
    vp = MuseGlimmerVideoProcessor(fps=2.0, num_frames=8, temporal_patch_size=2)

    # 48 frames at 24 fps -> 2.0s -> 4 sampled frames
    meta = VideoMetadata(total_num_frames=48, fps=24.0)
    indices = vp.sample_frames(meta)
    assert indices.tolist() == [0, 15, 31, 47]

    # Cap at num_frames=8 for a long clip
    long_meta = VideoMetadata(total_num_frames=480, fps=24.0)
    long_indices = vp.sample_frames(long_meta)
    assert len(long_indices) == 8
    assert long_indices[0].item() == 0
    assert long_indices[-1].item() == 479

    # Floor down to multiple of temporal_patch_size (5 -> 4)
    odd_meta = VideoMetadata(total_num_frames=60, fps=24.0)
    odd_indices = vp.sample_frames(odd_meta)
    assert len(odd_indices) == 4

    # Single-frame clip: clamped to total_num_frames=1
    single_meta = VideoMetadata(total_num_frames=1, fps=24.0)
    assert vp.sample_frames(single_meta).tolist() == [0]

    # Missing or non-positive fps falls back to 24
    no_fps_meta = VideoMetadata(total_num_frames=48, fps=None)
    assert len(vp.sample_frames(no_fps_meta)) == 4
    assert no_fps_meta.fps == 24

    bad_fps_meta = VideoMetadata(total_num_frames=48, fps=0.0)
    assert len(vp.sample_frames(bad_fps_meta)) == 4
    assert bad_fps_meta.fps == 24

    with pytest.raises(ValueError, match="at least one frame"):
        vp.sample_frames(VideoMetadata(total_num_frames=0, fps=24.0))


def test_video_processor_patchify_and_batch_ordering():
    vp = MuseGlimmerVideoProcessor(
        patch_size=14,
        temporal_patch_size=2,
        merge_size=2,
        max_video_frame_tokens=16,
    )
    # Video 0: 4 frames, 56x84 (grid_t=2, grid_h=4, grid_w=6 -> 2*4*6=48 patches)
    vid0 = torch.randint(0, 256, (4, 3, 56, 84), dtype=torch.uint8)
    # Video 1: 3 frames (odd -> padded to 4 -> grid_t=2), 84x56 (grid_h=6, grid_w=4 -> 48 patches)
    vid1 = torch.randint(0, 256, (3, 3, 84, 56), dtype=torch.uint8)
    # Video 2: 2 frames, 140x140 (downscaled to fit max_video_frame_tokens=16 -> 4x4 merged -> 112x112 -> grid_h=8, grid_w=8)
    vid2 = torch.randint(0, 256, (2, 3, 140, 140), dtype=torch.uint8)

    out = vp(
        [vid0, vid1, vid2],
        do_sample_frames=False,
        return_tensors="pt",
    )
    assert out["video_grid_thw"].tolist() == [
        [2, 4, 6],
        [2, 6, 4],
        [1, 8, 8],
    ]
    expected_rows = 2 * 4 * 6 + 2 * 6 * 4 + 1 * 8 * 8
    assert out["pixel_values_videos"].shape == (expected_rows, 2 * 3 * 14 * 14)
    assert torch.isfinite(out["pixel_values_videos"]).all()

    # Verify odd-frame temporal padding repeats the last frame verbatim in the last temporal slot
    single_odd = torch.zeros((3, 3, 28, 28), dtype=torch.uint8)
    single_odd[2] = 255
    odd_out = vp(
        [single_odd],
        do_resize=False,
        do_rescale=False,
        do_normalize=False,
        do_sample_frames=False,
        return_tensors="pt",
    )
    # grid_t=2, grid_h=2, grid_w=2 -> 8 patches; last 4 patches are temporal group 1 (frame 2 + repeated frame 2)
    assert odd_out["video_grid_thw"].tolist() == [[2, 2, 2]]
    group0_patches = odd_out["pixel_values_videos"][:4]
    group1_patches = odd_out["pixel_values_videos"][4:]
    assert (group0_patches == 0).all()
    assert (group1_patches == 255).all()


# ---------------------------------------------------------------------------
# MuseGlimmerProcessor unit tests
# ---------------------------------------------------------------------------


def test_processor_image_delimiters_and_video_timestamps():
    processor = _build_processor()
    img = Image.new("RGB", (84, 56), color=(64, 128, 192))
    vid = torch.randint(0, 256, (4, 3, 56, 84), dtype=torch.uint8)
    meta = VideoMetadata(total_num_frames=48, fps=24.0, frames_indices=[0, 12, 24, 36])

    out = processor(
        text=["describe <|patch|> and <|video|>"],
        images=[img],
        videos=[vid],
        video_metadata=[meta],
        do_sample_frames=False,
        return_tensors="pt",
    )
    assert "video_metadata" not in out
    assert out["image_grid_thw"].tolist() == [[1, 4, 6]]
    assert out["video_grid_thw"].tolist() == [[2, 4, 6]]

    tokens = processor.tokenizer.convert_ids_to_tokens(out["input_ids"][0].tolist())
    # Image: <|image_start|> + 6 * <|patch|> + <|image_end|>
    img_start = tokens.index("<|image_start|>")
    img_end = tokens.index("<|image_end|>")
    assert tokens[img_start + 1 : img_end] == ["<|patch|>"] * 6

    # Video: <|vid_start|> Time: 0.0s + 6 * <|video|> + <|vid_frame_separator|> Time: 1.0s + 6 * <|video|> + <|vid_end|>
    vid_start = tokens.index("<|vid_start|>")
    vid_sep = tokens.index("<|vid_frame_separator|>")
    vid_end = tokens.index("<|vid_end|>")
    assert tokens[vid_start + 1 : vid_start + 3] == ["Time:", "0.0s"]
    assert tokens[vid_start + 3 : vid_sep] == ["<|video|>"] * 6
    assert tokens[vid_sep + 1 : vid_sep + 3] == ["Time:", "1.0s"]
    assert tokens[vid_sep + 3 : vid_end] == ["<|video|>"] * 6


def test_processor_return_metadata_false_and_missing_fps_fallback():
    processor = _build_processor()
    vid = torch.randint(0, 256, (3, 3, 28, 28), dtype=torch.uint8)
    meta_no_fps = VideoMetadata(total_num_frames=3, fps=None)

    out = processor(
        text=["describe <|video|>"],
        videos=[vid],
        video_metadata=[meta_no_fps],
        do_sample_frames=False,
        return_metadata=False,
        return_tensors="pt",
    )
    assert "video_metadata" not in out
    assert out["video_grid_thw"].tolist() == [[2, 2, 2]]
    tokens = processor.tokenizer.convert_ids_to_tokens(out["input_ids"][0].tolist())
    assert tokens.count("<|video|>") == 2
    assert "<|vid_start|>" in tokens
    assert "<|vid_frame_separator|>" in tokens
    assert "<|vid_end|>" in tokens


def test_processor_from_pretrained_loads_image_and_video_configs():
    tokenizer = _build_tokenizer()
    with tempfile.TemporaryDirectory() as tmpdir:
        tokenizer.save_pretrained(tmpdir)
        config = {
            "processor_class": "MuseGlimmerProcessor",
            "image_processor": {
                "image_processor_type": "MuseGlimmerImageProcessor",
                "max_image_tokens": 512,
                "patch_size": 14,
                "merge_size": 2,
            },
            "video_processor": {
                "video_processor_type": "MuseGlimmerVideoProcessor",
                "max_video_frame_tokens": 64,
                "num_frames": 32,
                "fps": 1.5,
                "frame_timestamp_format": "T={timestamp:.1f}s",
            },
        }
        with open(os.path.join(tmpdir, "processor_config.json"), "w") as f:
            json.dump(config, f)

        loaded = MuseGlimmerProcessor.from_pretrained(tmpdir)
        assert loaded.image_processor.max_image_tokens == 512
        assert loaded.video_processor.max_video_frame_tokens == 64
        assert loaded.video_processor.num_frames == 32
        assert loaded.video_processor.fps == 1.5
        assert loaded.video_processor.frame_timestamp_format == "T={timestamp:.1f}s"

    # Also verify default video_processor fallback when omitted from __init__
    default_proc = MuseGlimmerProcessor(
        image_processor=MuseGlimmerImageProcessor(),
        tokenizer=tokenizer,
    )
    assert isinstance(default_proc.video_processor, MuseGlimmerVideoProcessor)


def test_processor_parity_with_upstream_hf_transformers():
    hf_mod = pytest.importorskip("transformers.models.muse_glimmer")
    tokenizer = _build_tokenizer()
    sgl_proc = MuseGlimmerProcessor(
        image_processor=MuseGlimmerImageProcessor(),
        video_processor=MuseGlimmerVideoProcessor(),
        tokenizer=tokenizer,
    )
    hf_proc = hf_mod.MuseGlimmerProcessor(
        image_processor=hf_mod.MuseGlimmerImageProcessor(),
        video_processor=hf_mod.MuseGlimmerVideoProcessor(return_metadata=True),
        tokenizer=tokenizer,
    )

    img = Image.new("RGB", (84, 56), color=(100, 150, 200))
    vid1 = torch.randint(0, 256, (4, 3, 56, 84), dtype=torch.uint8)
    vid2 = torch.randint(0, 256, (3, 3, 84, 56), dtype=torch.uint8)
    meta1 = VideoMetadata(total_num_frames=48, fps=24.0, frames_indices=[0, 16, 32, 47])
    meta2 = VideoMetadata(total_num_frames=36, fps=12.0, frames_indices=[0, 18, 35])
    prompt = "describe <|patch|> and <|video|> and <|video|>"

    sgl_out = sgl_proc(
        text=[prompt],
        images=[img],
        videos=[vid1, vid2],
        video_metadata=[meta1, meta2],
        do_sample_frames=False,
        return_tensors="pt",
    )
    hf_out = hf_proc(
        text=[prompt],
        images=[img],
        videos=[vid1, vid2],
        video_metadata=[meta1, meta2],
        do_sample_frames=False,
        return_tensors="pt",
    )

    assert torch.equal(sgl_out["input_ids"], hf_out["input_ids"])
    assert torch.equal(sgl_out["image_grid_thw"], hf_out["image_grid_thw"])
    assert torch.equal(sgl_out["video_grid_thw"], hf_out["video_grid_thw"])
    assert torch.allclose(sgl_out["pixel_values"], hf_out["pixel_values"])
    assert torch.allclose(sgl_out["pixel_values_videos"], hf_out["pixel_values_videos"])


# ---------------------------------------------------------------------------
# MuseGlimmerMultimodalProcessor end-to-end async tests
# ---------------------------------------------------------------------------


def test_mm_processor_image_delimiters_and_avoid_retokenize():
    mm_proc = _build_mm_processor()
    img = Image.new("RGB", (84, 56), color=(10, 20, 30))
    tokenizer = mm_proc._processor.tokenizer

    output = asyncio.run(
        mm_proc.process_mm_data_async(
            image_data=[img],
            input_text="describe <|patch|> image",
            request_obj=SimpleNamespace(video_data=None),
        )
    )
    assert output.im_token_id == mm_proc.image_token_id
    assert output.video_token_id == mm_proc.video_token_id
    assert len(output.mm_items) == 1
    item = output.mm_items[0]
    assert item.modality == Modality.IMAGE
    assert item.offsets == [(start, end) for start, end in item.offsets]
    (start, end) = item.offsets[0]
    assert end - start + 1 == 6
    assert output.input_ids[start - 1] == mm_proc.image_start_token_id
    assert output.input_ids[end + 1] == mm_proc.image_end_token_id
    assert all(
        tok == mm_proc.image_token_id for tok in output.input_ids[start : end + 1]
    )

    # Verify SGLANG_MM_AVOID_RETOKENIZE preserves surrounding ids AND image delimiters
    pretokenized = tokenizer.encode(
        "describe <|patch|> image", add_special_tokens=False
    )
    with patch.object(
        base_processor_mod.envs.SGLANG_MM_AVOID_RETOKENIZE, "get", return_value=True
    ):
        output_no_retok = asyncio.run(
            mm_proc.process_mm_data_async(
                image_data=[img],
                input_text=pretokenized,
                request_obj=SimpleNamespace(video_data=None),
            )
        )
    assert output_no_retok.input_ids == [
        tokenizer.convert_tokens_to_ids("describe"),
        mm_proc.image_start_token_id,
        *([mm_proc.image_token_id] * 6),
        mm_proc.image_end_token_id,
        tokenizer.convert_tokens_to_ids("image"),
    ]
    assert output_no_retok.mm_items[0].offsets == [(2, 7)]


def test_mm_processor_video_decoder_wrapper_lazy_sampling_and_close():
    mm_proc = _build_mm_processor()
    video = _InMemoryVideo(_synthetic_frames(48, 56, 84, seed=1), fps=24.0)

    with patch(
        "sglang.srt.multimodal.multimodal_processor_mixin.load_video",
        return_value=video,
    ):
        output = asyncio.run(
            mm_proc.process_mm_data_async(
                image_data=None,
                input_text="describe <|video|>",
                request_obj=SimpleNamespace(video_data=["clip.mp4"]),
            )
        )

    # Lazy sampling: only the 4 sampled indices out of 48 were decoded, and close() was called
    assert video.decoded_indices == [[0, 15, 31, 47]]
    assert video.closed is True

    assert output.video_token_id == mm_proc.video_token_id
    assert len(output.mm_items) == 1
    item = output.mm_items[0]
    assert item.modality == Modality.VIDEO
    assert item.video_grid_thw.tolist() == [[2, 4, 6]]
    assert item.feature.shape == (2 * 4 * 6, 2 * 3 * 14 * 14)
    # Two temporal groups -> two separate offset spans of 6 tokens each
    assert len(item.offsets) == 2
    assert [end - start + 1 for start, end in item.offsets] == [6, 6]
    for start, end in item.offsets:
        assert all(
            tok == mm_proc.video_token_id for tok in output.input_ids[start : end + 1]
        )


def test_mm_processor_closes_video_decoder_on_decode_error():
    mm_proc = _build_mm_processor()
    failing_video = _InMemoryVideo(
        _synthetic_frames(24, 28, 28, seed=2), fps=24.0, fail_on_decode=True
    )

    with patch(
        "sglang.srt.multimodal.multimodal_processor_mixin.load_video",
        return_value=failing_video,
    ):
        with pytest.raises(RuntimeError, match="simulated decode failure"):
            asyncio.run(
                mm_proc.process_mm_data_async(
                    image_data=None,
                    input_text="describe <|video|>",
                    request_obj=SimpleNamespace(video_data=["broken.mp4"]),
                )
            )
    assert failing_video.closed is True


def test_mm_processor_multi_video_mixed_image_and_pad_mm_tokens():
    mm_proc = _build_mm_processor()
    img = Image.new("RGB", (56, 56), color=(20, 80, 140))
    # Video 0: 72 frames @ 24fps with per-video fps=2.0 -> 6 frames -> grid_t=3, 56x84 (6 tokens/group)
    vid0 = _InMemoryVideo(_synthetic_frames(72, 56, 84, seed=3), fps=24.0)
    # Video 1: 48 frames @ 24fps with OpenAI-style max_frames=2 -> 2 frames -> grid_t=1, 84x56 (6 tokens/group)
    vid1 = _InMemoryVideo(_synthetic_frames(48, 84, 56, seed=4), fps=24.0)
    videos_by_url = {"v0.mp4": vid0, "v1.mp4": vid1}

    def _fake_load_video(item, use_gpu=True):
        url = item.url if isinstance(item, VideoData) else item
        return videos_by_url[url]

    video_data = [
        VideoData(url="v0.mp4", preprocess_kwargs={"fps": 2.0}),
        VideoData(url="v1.mp4", preprocess_kwargs={"max_frames": 2}),
    ]
    with patch(
        "sglang.srt.multimodal.multimodal_processor_mixin.load_video",
        side_effect=_fake_load_video,
    ):
        output = asyncio.run(
            mm_proc.process_mm_data_async(
                image_data=[img],
                input_text="describe <|patch|> and <|video|> and <|video|>",
                request_obj=SimpleNamespace(video_data=video_data),
            )
        )

    assert vid0.closed and vid1.closed
    assert len(vid0.decoded_indices[0]) == 6
    assert len(vid1.decoded_indices[0]) == 2

    img_items = [item for item in output.mm_items if item.modality == Modality.IMAGE]
    vid_items = [item for item in output.mm_items if item.modality == Modality.VIDEO]
    assert len(img_items) == 1
    assert len(vid_items) == 2

    assert img_items[0].image_grid_thw.tolist() == [[1, 4, 4]]
    assert img_items[0].feature.shape == (16, 2 * 3 * 14 * 14)
    assert len(img_items[0].offsets) == 1

    # Multi-video bundle is split into per-video MultimodalDataItems with grouped spans
    assert vid_items[0].video_grid_thw.tolist() == [[3, 4, 6]]
    assert vid_items[0].feature.shape == (3 * 4 * 6, 2 * 3 * 14 * 14)
    assert len(vid_items[0].offsets) == 3
    assert [e - s + 1 for s, e in vid_items[0].offsets] == [6, 6, 6]

    assert vid_items[1].video_grid_thw.tolist() == [[1, 6, 4]]
    assert vid_items[1].feature.shape == (1 * 6 * 4, 2 * 3 * 14 * 14)
    assert len(vid_items[1].offsets) == 1
    assert [e - s + 1 for s, e in vid_items[1].offsets] == [6]

    # pad_mm_tokens replaces only <|patch|> and <|video|> spans, never delimiters or timestamps
    mm_inputs = MultimodalInputs.from_processor_output(output)
    orig_ids = array.array("q", output.input_ids)
    padded_ids = pad_mm_tokens(orig_ids, mm_inputs)
    assert len(padded_ids) == len(orig_ids)
    mm_positions = {
        pos
        for item in mm_inputs.mm_items
        for start, end in item.offsets
        for pos in range(start, end + 1)
    }
    for idx, (orig, padded) in enumerate(zip(orig_ids, padded_ids)):
        if idx not in mm_positions:
            assert orig == padded


def test_mm_processor_pil_frame_list_and_raw_tensor_video_inputs():
    mm_proc = _build_mm_processor()

    # Pre-extracted frame list of 4 PIL images: all 4 frames must be preserved (not dropped)
    pil_frames = [Image.new("RGB", (56, 56), color=(i * 50, 20, 40)) for i in range(4)]
    out_pil = asyncio.run(
        mm_proc.process_mm_data_async(
            image_data=None,
            input_text="describe <|video|>",
            request_obj=SimpleNamespace(video_data=[pil_frames]),
        )
    )
    assert len(out_pil.mm_items) == 1
    assert out_pil.mm_items[0].video_grid_thw.tolist() == [[2, 4, 4]]
    assert len(out_pil.mm_items[0].offsets) == 2

    # Raw 4D NHWC tensor (48 frames @ assumed 24fps, target 2fps -> 4 frames -> grid_t=2)
    raw_nhwc = torch.randint(0, 256, (48, 56, 84, 3), dtype=torch.uint8)
    out_tensor = asyncio.run(
        mm_proc.process_mm_data_async(
            image_data=None,
            input_text="describe <|video|>",
            request_obj=SimpleNamespace(video_data=[raw_nhwc]),
        )
    )
    assert len(out_tensor.mm_items) == 1
    assert out_tensor.mm_items[0].video_grid_thw.tolist() == [[2, 4, 6]]


# ---------------------------------------------------------------------------
# Model get_video_feature + embed_mm_inputs scatter tests
# ---------------------------------------------------------------------------


def test_model_get_video_feature_and_mixed_embed_scatter():
    mm_proc = _build_mm_processor()
    img = Image.new("RGB", (56, 56), color=(10, 100, 200))
    vid = _InMemoryVideo(_synthetic_frames(48, 56, 84, seed=7), fps=24.0)

    with patch(
        "sglang.srt.multimodal.multimodal_processor_mixin.load_video",
        return_value=vid,
    ):
        output = asyncio.run(
            mm_proc.process_mm_data_async(
                image_data=[img],
                input_text="describe <|patch|> and <|video|>",
                request_obj=SimpleNamespace(video_data=["clip.mp4"]),
            )
        )

    hidden_size = 16
    merge_size = 2

    # Build a lightweight MuseGlimmerForConditionalGeneration using real get_image_feature /
    # get_video_feature / _encode_vision_features and real pixel_shuffle on t > 1 video grids.
    model = MuseGlimmerForConditionalGeneration.__new__(
        MuseGlimmerForConditionalGeneration
    )
    torch.nn.Module.__init__(model)
    vision_stub = MuseGlimmerVisionModel.__new__(MuseGlimmerVisionModel)
    torch.nn.Module.__init__(vision_stub)
    vision_stub.merge_size = merge_size

    def _stub_vision_tower(
        pixel_values: torch.Tensor, grid_thw: torch.Tensor
    ) -> torch.Tensor:
        # Project each patch to a small dim, then run real MuseGlimmerVisionModel.pixel_shuffle
        patch_proj = pixel_values[:, :4].to(torch.float32)
        return vision_stub.pixel_shuffle(patch_proj, grid_thw)

    model.vision_tower = _stub_vision_tower
    model.vision_adapter = lambda x: x
    model.vision_projection = lambda x: (x, None)
    model.perception_emb_norm = lambda x: x

    mm_inputs = MultimodalInputs.from_processor_output(output)
    padded_ids = model.pad_input_ids(array.array("q", output.input_ids), mm_inputs)
    input_ids = torch.tensor(padded_ids, dtype=torch.long)
    embed_tokens = torch.nn.Embedding(len(mm_proc._processor.tokenizer), hidden_size)
    with torch.no_grad():
        embed_tokens.weight.fill_(1.0)

    # Directly verify get_image_feature and get_video_feature shapes
    img_items = [it for it in mm_inputs.mm_items if it.modality == Modality.IMAGE]
    vid_items = [it for it in mm_inputs.mm_items if it.modality == Modality.VIDEO]
    img_feats = model.get_image_feature(img_items)
    vid_feats = model.get_video_feature(vid_items)
    assert img_feats.shape == (4, hidden_size)
    assert vid_feats.shape == (12, hidden_size)

    init_mm_embedding_cache(max_size=0)

    # Verify embed_mm_inputs scatters image + multi-span video features into exact positions
    input_embeds, _ = embed_mm_inputs(
        mm_inputs_list=[mm_inputs],
        extend_prefix_lens=[0],
        extend_seq_lens=[len(input_ids)],
        input_ids=input_ids.clone(),
        input_embedding=embed_tokens,
        multimodal_model=model,
    )
    assert input_embeds.shape == (len(input_ids), hidden_size)

    img_positions = [
        p for start, end in img_items[0].offsets for p in range(start, end + 1)
    ]
    vid_positions = [
        p for start, end in vid_items[0].offsets for p in range(start, end + 1)
    ]
    assert torch.allclose(input_embeds[img_positions], img_feats)
    assert torch.allclose(input_embeds[vid_positions], vid_feats)

    # All non-mm tokens (text, <|image_start|>, <|image_end|>, <|vid_start|>, Time: X.Xs, <|vid_frame_separator|>, <|vid_end|>) keep text embeddings
    non_mm_positions = [
        i
        for i in range(len(input_ids))
        if i not in set(img_positions) and i not in set(vid_positions)
    ]
    assert torch.allclose(
        input_embeds[non_mm_positions],
        torch.ones((len(non_mm_positions), hidden_size)),
    )

    # Also verify chunked prefill split right across the middle of the video's temporal groups
    split_at = vid_items[0].offsets[0][1] + 1
    chunk0, _ = embed_mm_inputs(
        mm_inputs_list=[mm_inputs],
        extend_prefix_lens=[0],
        extend_seq_lens=[split_at],
        input_ids=input_ids[:split_at].clone(),
        input_embedding=embed_tokens,
        multimodal_model=model,
    )
    chunk1, _ = embed_mm_inputs(
        mm_inputs_list=[mm_inputs],
        extend_prefix_lens=[split_at],
        extend_seq_lens=[len(input_ids) - split_at],
        input_ids=input_ids[split_at:].clone(),
        input_embedding=embed_tokens,
        multimodal_model=model,
    )
    assert torch.allclose(torch.cat([chunk0, chunk1], dim=0), input_embeds)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
