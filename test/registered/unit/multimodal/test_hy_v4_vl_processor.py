import array
import asyncio
import concurrent.futures
import json
import shutil
import sys
import threading
from contextlib import nullcontext
from types import SimpleNamespace

import pytest
import torch
from PIL import Image
from tokenizers import Tokenizer, pre_tokenizers
from tokenizers.models import WordLevel
from transformers import PreTrainedTokenizerFast
from transformers.video_utils import VideoMetadata

from sglang.srt.configs.hy_v4 import HYV4VLConfig
from sglang.srt.configs.hy_v4_flash import HYV4FlashConfig
from sglang.srt.configs.hy_v4_vl_processing import (
    HYV4VLImageProcessor,
    HYV4VLProcessor,
    HYV4VLVideoProcessor,
    vision_token_count,
)
from sglang.srt.entrypoints.openai.protocol import ChatCompletionRequest
from sglang.srt.entrypoints.openai.serving_chat import OpenAIServingChat
from sglang.srt.hf_transformers.processor import get_processor
from sglang.srt.managers.mm_utils import MultiModalityDataPaddingPatternMultimodalTokens
from sglang.srt.managers.req_types import MultimodalInputs
from sglang.srt.model_executor.forward_batch_info import ForwardBatch
from sglang.srt.multimodal import multimodal_processor_mixin
from sglang.srt.multimodal.modality import Modality
from sglang.srt.multimodal.processors.hy_v4_vl import (
    HYV4VLMultimodalProcessor,
    get_hy_v4_vl_mrope_positions,
)
from sglang.srt.parser.jinja_template_utils import (
    detect_jinja_template_content_format,
)
from sglang.srt.runtime_context import publish, reset_context
from sglang.srt.server_args import ServerArgs
from sglang.srt.utils.video_decoder import VideoDecoderWrapper
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=30, suite="base-a-test-cpu")

SPECIAL_TOKENS = {
    "image_start_token": "<hy_image_start>",
    "image_end_token": "<hy_image_end>",
    "image_token": "<hy_image_pad>",
    "image_newline_token": "<hy_image_newline>",
    "video_start_token": "<hy_video_start>",
    "video_end_token": "<hy_video_end>",
    "video_token": "<hy_video_pad>",
}
VOCAB = [
    "[UNK]",
    "[PAD]",
    *SPECIAL_TOKENS.values(),
    "<image>",
    "<video>",
    "hi",
    "<0.0",
    "<0.5",
    "seconds>",
    "<0.2",
    "<1.0",
    "hi there",
]
HI, TS_0_0, TS_0_5, TS_0_2, TS_1_0, SECONDS = (
    VOCAB.index(t) for t in ("hi", "<0.0", "<0.5", "<0.2", "<1.0", "seconds>")
)
IMAGE_MARKER, VIDEO_MARKER = VOCAB.index("<image>"), VOCAB.index("<video>")
HI_THERE = VOCAB.index("hi there")
PAD = VOCAB.index("[PAD]")
START, END, IMAGE_PAD, VIDEO_START, VIDEO_END, VIDEO_PAD = 2, 3, 4, 6, 7, 8


@pytest.fixture(scope="module")
def checkpoint_dir(tmp_path_factory):
    path = tmp_path_factory.mktemp("hy_v4_vl")
    tokenizer = Tokenizer(
        WordLevel(dict(zip(VOCAB, range(len(VOCAB)))), unk_token="[UNK]")
    )
    tokenizer.pre_tokenizer = pre_tokenizers.WhitespaceSplit()
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=tokenizer,
        unk_token="[UNK]",
        pad_token="[PAD]",
        extra_special_tokens=SPECIAL_TOKENS,
    )
    tokenizer.add_special_tokens(
        {"extra_special_tokens": ["<image>", "<video>"]},
        replace_extra_special_tokens=False,
    )
    tokenizer.save_pretrained(path)
    HYV4VLConfig(
        image_start_token_id=START,
        image_end_token_id=END,
        image_token_id=IMAGE_PAD,
        image_newline_token_id=5,
        video_start_token_id=VIDEO_START,
        video_end_token_id=VIDEO_END,
        video_token_id=VIDEO_PAD,
        vision_config={"max_image_size": 64},
    ).save_pretrained(path)
    (path / "preprocessor_config.json").write_text(
        json.dumps(
            {
                "min_pixels": 4096,
                "max_pixels": 16384,
                "min_frames": 2,
                "max_frames": 8,
                "timestamp_token_reserve": 0,
                "video_max_image_size": 64,
            }
        )
    )
    return path


@pytest.fixture
def mm_processor(checkpoint_dir):
    reset_context()
    server_args = ServerArgs(model_path="dummy", mm_process_config={})
    publish(server_args, role="tokenizer")
    hf_processor = get_processor(str(checkpoint_dir), trust_remote_code=False)
    processor = HYV4VLMultimodalProcessor(
        hf_processor.config, server_args, hf_processor, "auto"
    )
    yield processor
    processor.shutdown()
    reset_context()


@pytest.mark.parametrize(
    "height,width,expected_grid,expected_tokens",
    [
        # Below the 512^2 budget: scaled up, then ceil-aligned to 32 px.
        (300, 200, (40, 28), 20 * 15),
        # Exactly 2048^2 stays put.
        (2048, 2048, (128, 128), 64 * 65),
        # Above 2048^2: scaled down, then floor-aligned to 32 px.
        (3500, 4000, (118, 136), 59 * 69),
        # Aspect ratio at the 200 cap; the newline column makes it orientation-dependent.
        (32, 6400, (4, 454), 2 * 228),
        (6400, 32, (454, 4), 227 * 3),
    ],
)
def test_image_token_count_at_pixel_budget_boundaries(
    height, width, expected_grid, expected_tokens
):
    grid = HYV4VLImageProcessor().get_number_of_image_patches(height, width)
    assert grid == expected_grid
    assert (
        vision_token_count((1, *grid), HYV4VLConfig().vision_config, "image")
        == expected_tokens
    )


def test_image_aspect_ratio_above_cap_is_rejected():
    with pytest.raises(ValueError, match="aspect ratio"):
        HYV4VLImageProcessor().get_number_of_image_patches(32, 6432)


@pytest.mark.parametrize(
    "total_frames,video_fps,expected_frames",
    [
        (300, 30, 20),  # 10 s at 2 fps
        (30, 30, 4),  # 1 s raised to the 4-frame floor
        (6000, 30, 100),  # 200 s capped at 100 frames
        (3, 30, 3),  # never more frames than the video has
    ],
)
def test_video_sampling_is_two_fps_within_frame_bounds(
    total_frames, video_fps, expected_frames
):
    indices = HYV4VLVideoProcessor().sample_frames(
        VideoMetadata(total_num_frames=total_frames, fps=video_fps)
    )
    assert len(indices) == expected_frames
    assert indices.min() >= 0 and indices.max() < total_frames


# Verbatim from the released checkpoint.
RELEASE_IMAGE_PREPROCESSOR = {
    "min_pixels": 4096,
    "max_pixels": 4194304,
    "patch_size": 16,
    "resample": 1,
    "temporal_patch_size": 1,
    "spatial_patch_size": 1,
    "merge_size": 2,
    "image_mean": [0.5, 0.5, 0.5],
    "image_std": [0.5, 0.5, 0.5],
    "image_processor_type": "HYV4VLImageProcessor",
    "processor_class": "HYV4VLProcessor",
}
RELEASE_VIDEO_PREPROCESSOR = {
    "max_pixels": 122880000,
    "patch_size": 16,
    "resample": 1,
    "temporal_patch_size": 1,
    "merge_size": 2,
    "image_mean": [0.5, 0.5, 0.5],
    "image_std": [0.5, 0.5, 0.5],
    "timestamp_token_per_frame": 6,
    "timestamp_token_reserve": 1000,
    "min_frames": 2,
    "max_frames": 1024,
    "video_max_image_size": 1024,
    "video_processor_type": "HYV4VLVideoProcessor",
    "processor_class": "HYV4VLProcessor",
}


def _release_vl_config():
    return HYV4VLConfig(
        image_start_token_id=START,
        image_end_token_id=END,
        image_token_id=IMAGE_PAD,
        image_newline_token_id=5,
        video_start_token_id=VIDEO_START,
        video_end_token_id=VIDEO_END,
        video_token_id=VIDEO_PAD,
        vision_config={"img_max_token_num": 4096, "max_vit_seq_len": 16384},
    )


@pytest.fixture(scope="module")
def release_checkpoint_dir(checkpoint_dir, tmp_path_factory):
    path = tmp_path_factory.mktemp("hy_v4_vl_release")
    shutil.copytree(checkpoint_dir, path, dirs_exist_ok=True)
    _release_vl_config().save_pretrained(path)
    (path / "preprocessor_config.json").write_text(
        json.dumps(RELEASE_IMAGE_PREPROCESSOR)
    )
    (path / "video_preprocessor_config.json").write_text(
        json.dumps(RELEASE_VIDEO_PREPROCESSOR)
    )
    return path


@pytest.mark.parametrize(
    "checkpoint,image_max_pixels,video_settings",
    [
        ("release_checkpoint_dir", 4194304, (122880000, 2, 1024, 6, 1000)),
        # Without a video config the image config also sets the video options.
        ("checkpoint_dir", 16384, (16384, 2, 8, 4, 0)),
    ],
)
def test_video_processor_loads_the_video_preprocessor_config(
    request, checkpoint, image_max_pixels, video_settings
):
    processor = get_processor(
        str(request.getfixturevalue(checkpoint)), trust_remote_code=False
    )
    video = processor.video_processor
    assert processor.image_processor.size["longest_edge"] == image_max_pixels
    assert (
        video.size["longest_edge"],
        video.min_frames,
        video.max_frames,
        video.timestamp_token_per_frame,
        video.timestamp_token_reserve,
    ) == video_settings


def test_flash_checkpoint_loads_the_flash_config_through_the_shared_processor(
    checkpoint_dir, tmp_path
):
    """The shared HYV4 processor is registered for Hy4-VL and Hy4-Flash; each
    checkpoint must load its own family's config, not a fixed default."""
    flash_dir = tmp_path / "flash"
    shutil.copytree(checkpoint_dir, flash_dir)
    # Written as a checkpoint ships it: Hy4-Flash keeps the checkpoint's
    # layer-type spelling, which Transformers 5.19 rejects in save_pretrained.
    HYV4FlashConfig(
        image_start_token_id=START,
        image_end_token_id=END,
        image_token_id=IMAGE_PAD,
        image_newline_token_id=5,
        video_start_token_id=VIDEO_START,
        video_end_token_id=VIDEO_END,
        video_token_id=VIDEO_PAD,
        vision_config={"max_image_size": 64},
    ).to_json_file(flash_dir / "config.json")
    processor = get_processor(str(flash_dir), trust_remote_code=False)
    assert type(processor.config) is HYV4FlashConfig


def test_release_video_frames_keep_their_resolution(release_checkpoint_dir):
    # Under the image budget (4096 tokens - 2000 reserve - 4 per frame), 8
    # frames of 512^2 would be shrunk to 480^2.
    processor = get_processor(str(release_checkpoint_dir), trust_remote_code=False)
    frames = torch.zeros((8, 512, 512, 3), dtype=torch.uint8)
    output = processor.video_processor(videos=[frames], return_tensors="pt")
    assert output["video_grid_thw"].tolist() == [[8, 32, 32]]


def test_release_video_frames_stay_within_the_frame_size_cap(release_checkpoint_dir):
    # 20 frames of 4K overrun the token budget; scaling to the budget alone
    # would leave 1824x3264 frames, past video_max_image_size=1024 squared.
    processor = get_processor(str(release_checkpoint_dir), trust_remote_code=False)
    frames = torch.zeros((1, 2160, 3840, 3), dtype=torch.uint8).expand(20, -1, -1, -1)
    output = processor.video_processor(videos=[frames], return_tensors="pt")
    assert output["video_grid_thw"].tolist() == [[20, 48, 84]]


@pytest.mark.parametrize(
    "limit,max_pixels",
    [("img_max_token_num", 4097 * 32**2), ("max_vit_seq_len", 16385 * 16**2)],
)
def test_image_budget_above_vision_token_limits_is_rejected(
    mm_processor, limit, max_pixels
):
    with pytest.raises(ValueError, match=limit):
        HYV4VLProcessor(
            HYV4VLImageProcessor(max_pixels=max_pixels),
            mm_processor._processor.tokenizer,
            HYV4VLVideoProcessor(),
            config=_release_vl_config(),
        )


# Expected positions are the reference implementation's own test fixtures.
REF_IMAGE_TOKENS = [10, 120118, *([120120] * 6), 120119, 11]
REF_IMAGE_POSITIONS = [
    [0, 1, 2, 2, 2, 2, 2, 2, 5, 6],
    [0, 1, 2, 2, 2, 3, 3, 3, 5, 6],
    [0, 1, 2, 3, 4, 2, 3, 4, 5, 6],
]
REF_VIDEO_FRAME = [120118, *([120124] * 6), 120119]
REF_VIDEO_TOKENS = [120122, 10, *REF_VIDEO_FRAME, 11, *REF_VIDEO_FRAME, 120123]
REF_VIDEO_POSITIONS = [
    [0, 1, 2, 3, 3, 3, 3, 3, 3, 6, 7, 8, 9, 9, 9, 9, 9, 9, 12, 13],
    [0, 1, 2, 3, 3, 3, 4, 4, 4, 6, 7, 8, 9, 9, 9, 10, 10, 10, 12, 13],
    [0, 1, 2, 3, 4, 5, 3, 4, 5, 6, 7, 8, 9, 10, 11, 9, 10, 11, 12, 13],
]


@pytest.mark.parametrize(
    "input_ids,image_grids,video_grids,expected",
    [
        ([10, 11, 12], [], [], [[0, 1, 2]] * 3),
        (REF_IMAGE_TOKENS, [[1, 4, 4]], [], REF_IMAGE_POSITIONS),
        (REF_VIDEO_TOKENS, [], [[2, 4, 4]], REF_VIDEO_POSITIONS),
        (
            REF_IMAGE_TOKENS[:-1] + REF_IMAGE_TOKENS[1:-1],
            [[1, 4, 4], [1, 4, 4]],
            [],
            [
                [0, 1, 2, 2, 2, 2, 2, 2, 5, 6, 7, 7, 7, 7, 7, 7, 10],
                [0, 1, 2, 2, 2, 3, 3, 3, 5, 6, 7, 7, 7, 8, 8, 8, 10],
                [0, 1, 2, 3, 4, 2, 3, 4, 5, 6, 7, 8, 9, 7, 8, 9, 10],
            ],
        ),
    ],
    ids=["text", "image", "video", "two_images"],
)
def test_mrope_positions_and_delta(input_ids, image_grids, video_grids, expected):
    positions, delta = get_hy_v4_vl_mrope_positions(
        input_ids=input_ids,
        image_grid_thw=image_grids,
        video_grid_thw=video_grids,
        hf_config=HYV4VLConfig(),
    )
    torch.testing.assert_close(positions, torch.tensor(expected))
    expected_delta = max(map(max, expected)) + 1 - len(input_ids)
    assert delta.tolist() == [[expected_delta]]


@pytest.mark.parametrize(
    "malformed",
    [
        REF_IMAGE_TOKENS[:2] + [10] + REF_IMAGE_TOKENS[2:],
        # Same length and pad count as a valid span; only pad identity catches it.
        REF_IMAGE_TOKENS[:4] + [120124] + REF_IMAGE_TOKENS[5:],
    ],
    ids=["inserted_text", "video_pad_in_image_span"],
)
def test_mrope_rejects_placeholders_that_disagree_with_grid(malformed):
    with pytest.raises(ValueError, match="placeholders"):
        get_hy_v4_vl_mrope_positions(
            input_ids=malformed,
            image_grid_thw=[[1, 4, 4]],
            video_grid_thw=[],
            hf_config=HYV4VLConfig(),
        )


IMAGE_TRIPLE = "".join(
    SPECIAL_TOKENS[name]
    for name in ("image_start_token", "image_token", "image_end_token")
)
# 64x64 -> 2x(2+1) tokens, 64x96 -> 3x(2+1); each 2-frame video frame is
# squeezed to 32x32 by the 16384-pixel video budget -> 1x(1+1) tokens.
VIDEO_FRAME = [START, VIDEO_PAD, VIDEO_PAD, END]
MIXED_INPUT_IDS = [
    HI,
    *[START, *[IMAGE_PAD] * 6, END],
    HI,
    *[START, *[IMAGE_PAD] * 9, END],
    HI,
    VIDEO_START,
    *[TS_0_0, SECONDS, *VIDEO_FRAME],
    *[TS_0_5, SECONDS, *VIDEO_FRAME],
    VIDEO_END,
    HI,
]


def _two_frame_video():
    frames = torch.zeros((2, 64, 64, 3), dtype=torch.uint8)
    return frames, {"fps": 2.0, "frames_indices": [0, 1], "do_sample_frames": False}


def _process(mm_processor, prompt, images, videos=None):
    request = SimpleNamespace(video_data=videos, audio_data=None, rid="rid-0")
    return asyncio.run(
        mm_processor.process_mm_data_async(
            image_data=images, input_text=prompt, request_obj=request
        )
    )


@pytest.mark.parametrize(
    "prompt",
    [
        f"hi <image> hi {IMAGE_TRIPLE} hi <video> hi",
        [HI, IMAGE_MARKER, HI, START, IMAGE_PAD, END, HI, VIDEO_MARKER, HI],
    ],
    ids=["text", "token_ids"],
)
def test_processor_expands_both_placeholder_forms_and_video(mm_processor, prompt):
    output = _process(
        mm_processor,
        prompt,
        [Image.new("RGB", (64, 64)), Image.new("RGB", (64, 96))],
        [_two_frame_video()],
    )

    assert output.input_ids == MIXED_INPUT_IDS
    assert [(item.modality, item.offsets) for item in output.mm_items] == [
        (Modality.IMAGE, [(2, 7)]),
        (Modality.IMAGE, [(11, 19)]),
        (Modality.VIDEO, [(26, 27), (32, 33)]),
    ]
    image_1, image_2 = [2, 2, 2, 2, 2, 2], [8] * 9
    frame_1, frame_2 = [17, 17], [23, 23]
    assert output.mrope_positions[0].tolist() == [
        *[0, 1, *image_1, 5],
        *[6, 7, *image_2, 11],
        *[12, 13, 14, 15, 16, *frame_1],
        *[19, 20, 21, 22, *frame_2, 25],
        *[26, 27],
    ]
    assert output.mrope_position_delta.tolist() == [[28 - len(output.input_ids)]]


def _scheduled_mrope_positions(mm_inputs, prefix_lens, extend_lens, decode=False):
    seq_lens = torch.tensor(prefix_lens) + torch.tensor(extend_lens)
    forward_batch = SimpleNamespace(seq_lens_cpu=seq_lens, seq_lens=seq_lens)
    batch = SimpleNamespace(
        multimodal_inputs=mm_inputs,
        prefix_lens=prefix_lens,
        extend_lens=extend_lens,
        reqs=[SimpleNamespace(session=None)] * len(mm_inputs),
    )
    compute = (
        ForwardBatch._compute_mrope_positions_decode
        if decode
        else ForwardBatch._compute_mrope_positions_extend
    )
    compute(forward_batch, SimpleNamespace(device="cpu"), batch)
    return forward_batch.mrope_positions


def test_scheduled_mrope_positions_follow_chunks_batches_and_decode(mm_processor):
    """Scheduler-built positions for a HYV4 image request match the processor's
    across chunk boundaries inside the image span, keep request order in a batch
    with a text request, and decode continues at prompt length + delta."""
    output = _process(
        mm_processor, f"hi {IMAGE_TRIPLE} hi hi", [Image.new("RGB", (64, 96))]
    )
    mm_inputs = MultimodalInputs.from_processor_output(output)
    prompt_len = len(output.input_ids)
    ((span_start, span_end),) = output.mm_items[0].offsets
    cuts = [0, span_start + 2, span_end - 1, prompt_len]
    assert span_start < cuts[1] < cuts[2] < span_end

    chunks = [
        _scheduled_mrope_positions([mm_inputs], [start], [end - start])
        for start, end in zip(cuts, cuts[1:])
    ]
    torch.testing.assert_close(torch.cat(chunks, dim=1), output.mrope_positions)

    text_len = 4
    mixed = _scheduled_mrope_positions(
        [None, mm_inputs], [0, cuts[1]], [text_len, prompt_len - cuts[1]]
    )
    torch.testing.assert_close(
        mixed,
        torch.cat(
            [
                torch.arange(text_len).repeat(3, 1),
                output.mrope_positions[:, cuts[1] :],
            ],
            dim=1,
        ),
    )

    decode = _scheduled_mrope_positions(
        [mm_inputs, None], [prompt_len, text_len], [1, 1], decode=True
    )
    first_image_position = prompt_len + int(output.mrope_position_delta)
    assert first_image_position == int(output.mrope_positions.max()) + 1
    assert decode.tolist() == [[first_image_position, text_len]] * 3


def test_token_id_image_triple_keeps_text_ids_and_expands(mm_processor):
    # Decoding the whole list spells the triple with separators between tokens,
    # and "hi there" does not survive decode + retokenize.
    output = _process(
        mm_processor,
        [HI_THERE, START, IMAGE_PAD, END, HI],
        [Image.new("RGB", (64, 64))],
    )
    assert output.input_ids == [HI_THERE, START, *[IMAGE_PAD] * 6, END, HI]
    assert [item.offsets for item in output.mm_items] == [[(2, 7)]]


@pytest.mark.parametrize(
    "prompt,num_images",
    [
        (f"hi {SPECIAL_TOKENS['image_start_token']}{SPECIAL_TOKENS['image_token']}", 1),
        ([HI, START, IMAGE_PAD, IMAGE_PAD, END], 1),
        ("hi <image>", 2),
        (f"hi <image> {SPECIAL_TOKENS['video_end_token']}", 1),
        ([HI, START, IMAGE_PAD, END, VIDEO_END], 1),
    ],
    ids=[
        "unclosed_triple",
        "already_expanded_ids",
        "more_images_than_placeholders",
        "stray_video_end_text",
        "stray_video_end_ids",
    ],
)
def test_media_placeholder_mismatch_is_rejected(mm_processor, prompt, num_images):
    with pytest.raises(ValueError, match="placeholder"):
        _process(mm_processor, prompt, [Image.new("RGB", (64, 64))] * num_images)


def test_processor_output_pad_outside_visual_span_is_rejected(mm_processor):
    input_ids = [HI, START, *[IMAGE_PAD] * 6, END, IMAGE_PAD]
    processor_output = {
        "format": "processor_output",
        "pixel_values": torch.zeros(16, 768),
        "image_grid_thw": torch.tensor([[1, 4, 4]]),
    }
    with pytest.raises(ValueError, match="placeholders"):
        _process(mm_processor, input_ids, [processor_output])


@pytest.mark.parametrize(
    "options,expected_ids",
    [
        (dict(padding=True), [[HI, HI], [HI, PAD]]),
        (dict(truncation=True, max_length=1), [[HI], [HI]]),
    ],
    ids=["padding", "truncation"],
)
def test_padded_or_truncated_call_succeeds_while_another_thread_is_encoding(
    checkpoint_dir, options, expected_ids
):
    """A padded or truncated processor call must not fail with "Already borrowed"
    while another thread (off-loop chat-template encoding) holds the shared tokenizer."""
    processor = HYV4VLProcessor.from_pretrained(str(checkpoint_dir))
    # Warm first: the custom pre-tokenizer installed below cannot be deep-copied.
    processor(text=["hi"])
    entered, release = threading.Event(), threading.Event()

    class HoldFirstEncode:
        def pre_tokenize(self, pretok):
            if not entered.is_set():
                entered.set()
                assert release.wait(10)
            pretok.split(lambda _, text: text.split(" ", "removed"))

    processor.tokenizer.backend_tokenizer.pre_tokenizer = (
        pre_tokenizers.PreTokenizer.custom(HoldFirstEncode())
    )
    with concurrent.futures.ThreadPoolExecutor(1) as pool:
        holder = pool.submit(processor.tokenizer.encode, "hi")
        try:
            assert entered.wait(10)
            output = processor(text=["hi hi", "hi"], return_tensors="pt", **options)
        finally:
            release.set()
        assert holder.result(timeout=10) == [HI]
    assert output["input_ids"].tolist() == expected_ids


def _text_outcome(call):
    try:
        output = call()
    except ValueError as error:
        return str(error)
    return {
        key: value.tolist() if isinstance(value, torch.Tensor) else value
        for key, value in output.items()
    }


@pytest.mark.parametrize(
    "options",
    [
        dict(max_length=2),
        dict(padding=True, return_length=True, return_offsets_mapping=True),
        dict(
            padding=True,
            return_length=True,
            return_offsets_mapping=True,
            return_tensors="pt",
        ),
        dict(padding=True, truncation=True, max_length=3, pad_to_multiple_of=2),
    ],
    ids=["bare_max_length", "length_and_offsets", "offsets_tensor", "bad_multiple"],
)
def test_text_options_behave_as_on_the_tokenizer(checkpoint_dir, options):
    """Text kwargs (implicit truncation, padded offsets/lengths, joint validation)
    must give exactly what the tokenizer gives for the expanded prompt."""
    processor = HYV4VLProcessor.from_pretrained(str(checkpoint_dir))
    texts = ["hi hi hi hi", "hi"]
    expected = _text_outcome(
        lambda: processor.tokenizer(texts, add_special_tokens=False, **options)
    )
    assert _text_outcome(lambda: processor(text=texts, **options)) == expected


def test_scheduler_fallback_accepts_hash_padded_spans_only(mm_processor):
    output = _process(
        mm_processor,
        f"hi <image> hi {IMAGE_TRIPLE} hi <video> hi",
        [Image.new("RGB", (64, 64)), Image.new("RGB", (64, 96))],
        [_two_frame_video()],
    )
    for item in output.mm_items:
        item.set_pad_value()
    mm_inputs = SimpleNamespace(
        mm_items=output.mm_items,
        im_token_id=IMAGE_PAD,
        video_token_id=VIDEO_PAD,
        audio_token_id=None,
    )
    padded = MultiModalityDataPaddingPatternMultimodalTokens().pad_input_tokens(
        array.array("q", output.input_ids), mm_inputs
    )

    positions, delta = mm_processor.compute_mrope_positions(padded, output.mm_items)
    torch.testing.assert_close(positions, output.mrope_positions)
    torch.testing.assert_close(delta, output.mrope_position_delta)

    padded[4] = HI
    with pytest.raises(ValueError, match="placeholders"):
        mm_processor.compute_mrope_positions(padded, output.mm_items)


def test_patterned_image_patch_order_and_normalization():
    pixels = torch.arange(64 * 64 * 3, dtype=torch.uint8).reshape(64, 64, 3)
    processor = HYV4VLImageProcessor(
        min_pixels=4096, max_pixels=16384, image_mean=[0.5] * 3, image_std=[0.5] * 3
    )
    result = processor(images=[Image.fromarray(pixels.numpy())], return_tensors="pt")
    # Row-major 16x16 patches, each flattened channel-first, scaled to [-1, 1].
    expected = pixels.permute(2, 0, 1).float() / 127.5 - 1
    expected = expected.reshape(3, 4, 16, 4, 16).permute(1, 3, 0, 2, 4).reshape(16, -1)
    torch.testing.assert_close(result["pixel_values"], expected)
    assert result["image_grid_thw"].tolist() == [[1, 4, 4]]


def test_odd_frame_video_with_temporal_merge_and_extra_tokens(mm_processor):
    config = HYV4VLConfig(
        image_start_token_id=START,
        image_end_token_id=END,
        image_token_id=IMAGE_PAD,
        image_newline_token_id=5,
        video_start_token_id=VIDEO_START,
        video_end_token_id=VIDEO_END,
        video_token_id=VIDEO_PAD,
        vision_config={"temporal_patch_size": 2, "cat_extra_token": 1},
    )
    processor = HYV4VLProcessor(
        HYV4VLImageProcessor(temporal_patch_size=2),
        mm_processor._processor.tokenizer,
        HYV4VLVideoProcessor(temporal_patch_size=2),
        config=config,
    )
    frames = torch.zeros((3, 64, 64, 3), dtype=torch.uint8)
    metadata = {"fps": 2.0, "frames_indices": [0, 1, 2], "do_sample_frames": False}
    output = processor(
        text=processor.get_placeholder("video"),
        videos=[(frames, metadata)],
        return_tensors="pt",
    )

    # 3 frames merge in pairs to 2 (the last frame is repeated), each with
    # 2x(2+1) tokens plus begin/end extras; timestamps average each pair.
    assert output["video_grid_thw"].tolist() == [[3, 4, 4]]
    assert output["video_timestamps"].tolist() == [0.25, 1.0]
    frame = [START, *[VIDEO_PAD] * 8, END]
    assert output["input_ids"].tolist() == [
        [VIDEO_START, TS_0_2, SECONDS, *frame, TS_1_0, SECONDS, *frame, VIDEO_END]
    ]


class _FakeVideoDecoder(VideoDecoderWrapper):
    def __init__(self, num_frames: int, fps: float):
        self.frames = (
            torch.arange(num_frames, dtype=torch.uint8)
            .reshape(num_frames, 1, 1, 1)
            .expand(num_frames, 64, 64, 3)
        )
        self.fps = fps
        self.decode_threads = []

    def __len__(self):
        return len(self.frames)

    @property
    def avg_fps(self):
        return self.fps

    def get_frames_as_tensor(self, indices):
        self.decode_threads.append(threading.get_ident())
        return self.frames[indices]


def test_video_decoder_is_sampled_at_two_fps(mm_processor):
    # 60 frames at 30 fps is 2 s -> 4 frames, one per segment midpoint.
    frames, metadata = mm_processor._sample_video(_FakeVideoDecoder(60, 30.0))
    assert metadata == {
        "fps": 30.0,
        "total_num_frames": 60,
        "frames_indices": [7, 22, 37, 51],
        "do_sample_frames": False,
    }
    assert frames[:, 0, 0, 0].tolist() == [7, 22, 37, 51]


def test_video_frames_are_decoded_off_the_event_loop(mm_processor, monkeypatch):
    decoder = _FakeVideoDecoder(4, 2.0)
    monkeypatch.setattr(
        multimodal_processor_mixin, "load_video", lambda *args, **kwargs: decoder
    )

    async def run():
        request = SimpleNamespace(video_data=["video.mp4"], audio_data=None, rid="r")
        output = await mm_processor.process_mm_data_async(
            image_data=None, input_text="hi <video>", request_obj=request
        )
        return output, threading.get_ident()

    output, loop_thread = asyncio.run(run())
    assert [item.modality for item in output.mm_items] == [Modality.VIDEO]
    assert decoder.decode_threads and loop_thread not in decoder.decode_threads


def test_video_preprocessing_stays_off_the_gpu(mm_processor, monkeypatch):
    """Video pixels must not be preprocessed on the scheduler's GPU; long videos OOM it."""
    devices = []
    call = HYV4VLProcessor.__call__

    def spy(self, *args, device=None, **kwargs):
        devices.append(device)
        return call(self, *args, **kwargs)

    monkeypatch.setattr(HYV4VLProcessor, "__call__", spy)
    monkeypatch.setattr(
        mm_processor, "_fast_image_processor_device", lambda processor: "cuda:0"
    )
    monkeypatch.setattr(
        mm_processor,
        "_temporary_fast_processor_cuda_pool",
        lambda device: nullcontext(),
    )
    _process(mm_processor, "hi <video> hi", None, [_two_frame_video()])
    assert devices == ["cpu"]


def test_precomputed_embeddings_are_rejected(mm_processor):
    request = SimpleNamespace(video_data=None, audio_data=None, rid="rid-1")
    with pytest.raises(ValueError, match="precomputed"):
        asyncio.run(
            mm_processor.process_mm_data_async(
                image_data=[
                    {"format": "precomputed_embedding", "feature": torch.zeros(6, 8)}
                ],
                input_text="hi <image>",
                request_obj=request,
            )
        )


# Mirrors the checkpoint template: list content is iterated inside a macro and
# only text parts are rendered, so content-format detection yields "string".
TEXT_ONLY_TEMPLATE = (
    "{%- macro render_content(content) -%}"
    "{%- if content is string -%}{{- content -}}"
    "{%- else -%}{%- for item in content -%}"
    "{%- if item is mapping and item.type == 'text' -%}{{- item.text -}}{%- endif -%}"
    "{%- endfor -%}{%- endif -%}"
    "{%- endmacro -%}"
    "{%- for message in messages -%}{{- render_content(message['content']) -}}"
    "{%- endfor -%}"
)
MARKER_TEMPLATE = TEXT_ONLY_TEMPLATE.replace(
    "{%- endif -%}{%- endfor -%}",
    "{%- elif item is mapping and item.type == 'image' -%}{{- '<image>' -}}"
    "{%- endif -%}{%- endfor -%}",
)
IMAGES = {"image-a": (64, 64), "image-b": (64, 96)}
IMAGE_A_SPAN = [START, *[IMAGE_PAD] * 6, END]
IMAGE_B_SPAN = [START, *[IMAGE_PAD] * 9, END]


class _ChatTokenizerManager:
    def __init__(self, mm_processor):
        self.tokenizer = mm_processor._processor.tokenizer
        self.mm_processor = mm_processor
        self.model_path = "dummy"
        self.server_args = None
        self.model_config = SimpleNamespace(
            is_multimodal=True,
            hf_config=mm_processor.hf_config,
            get_default_sampling_params=lambda: None,
        )

    def config_value(self, name):
        return None


def _render_chat(mm_processor, template, content):
    tokenizer_manager = _ChatTokenizerManager(mm_processor)
    tokenizer_manager.tokenizer.chat_template = template
    template_manager = SimpleNamespace(
        chat_template_name=None,
        jinja_template_content_format=detect_jinja_template_content_format(template),
        reasoning_config=None,
        force_reasoning=False,
        jinja_template_may_reorder_tool_results=False,
        python_chat_template=None,
    )
    chat = OpenAIServingChat(tokenizer_manager, template_manager)
    request = ChatCompletionRequest(
        model="x", messages=[{"role": "user", "content": content}]
    )
    return chat._apply_jinja_template(request, None, is_multimodal=True)


def _image_part(url):
    return {"type": "image_url", "image_url": {"url": url}}


@pytest.mark.parametrize(
    "template,content,expected_ids",
    [
        (
            TEXT_ONLY_TEMPLATE,
            [{"type": "text", "text": "hi"}, _image_part("image-a")],
            [HI, *IMAGE_A_SPAN],
        ),
        (
            TEXT_ONLY_TEMPLATE,
            [
                _image_part("image-a"),
                {"type": "text", "text": "hi"},
                _image_part("image-b"),
            ],
            [*IMAGE_A_SPAN, HI, *IMAGE_B_SPAN],
        ),
        (
            MARKER_TEMPLATE,
            [_image_part("image-a"), {"type": "text", "text": "hi"}],
            [*IMAGE_A_SPAN, HI],
        ),
    ],
    ids=[
        "text_then_image",
        "two_images",
        "template_emits_marker",
    ],
)
def test_chat_image_parts_reach_the_processor_in_content_order(
    mm_processor, template, content, expected_ids
):
    rendered = _render_chat(mm_processor, template, content)
    urls = [image.url for image in rendered.image_data or []]
    output = _process(
        mm_processor,
        rendered.prompt,
        [Image.new("RGB", IMAGES[url]) for url in urls],
    )
    assert output.input_ids == expected_ids


def test_chat_video_part_becomes_a_video_placeholder(mm_processor):
    rendered = _render_chat(
        mm_processor,
        TEXT_ONLY_TEMPLATE,
        [
            {"type": "text", "text": "hi"},
            {"type": "video_url", "video_url": {"url": "video-a"}},
        ],
    )
    assert rendered.video_data == ["video-a"]
    assert rendered.prompt.split() == ["hi", "<video>"]


@pytest.mark.parametrize(
    "content",
    ["hi hi", [{"type": "text", "text": "hi"}, {"type": "text", "text": "hi"}]],
    ids=["string_content", "text_parts"],
)
def test_chat_text_only_flattening_is_unchanged(mm_processor, content):
    rendered = _render_chat(mm_processor, TEXT_ONLY_TEMPLATE, content)
    assert rendered.image_data is None and rendered.video_data is None
    assert rendered.prompt_ids == [HI, HI]


if __name__ == "__main__":
    sys.exit(pytest.main([__file__]))
