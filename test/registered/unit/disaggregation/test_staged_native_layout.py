"""CPU coverage of metadata-only P/D reconstruction for native staged models."""

import re
from types import SimpleNamespace

import msgspec
import pytest
import torch

from sglang.srt.disaggregation.encoder.handoff import EncoderHandoff
from sglang.srt.disaggregation.encoder.receiver import WaitingStagedImage
from sglang.srt.managers.req_types import MultimodalDataItem
from sglang.srt.multimodal.inkling2.processor import Inkling2MultimodalProcessor
from sglang.srt.multimodal.modality import Modality
from sglang.srt.multimodal.processors.dots_note_omni import DotsNoteOmniProcessor
from sglang.srt.multimodal.processors.glm4v import Glm4vImageProcessor
from sglang.srt.multimodal.processors.kimi_k3 import KimiK3ImageProcessor
from sglang.srt.multimodal.processors.kimi_k25 import KimiK2_5VLImageProcessor
from sglang.srt.multimodal.processors.kimi_vl import KimiVLImageProcessor
from sglang.srt.multimodal.processors.mimo_v2 import MiMoV2Processor
from sglang.srt.multimodal.processors.minimax_m3_vl import MiniMaxM3VLProcessor
from sglang.srt.multimodal.processors.muse_spark_omni import (
    MuseSpark1xOmniMultimodalProcessor,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=4, suite="base-a-test-cpu")


def identity(modality, tensor):
    return MultimodalDataItem(
        modality=modality, precomputed_embeddings=tensor
    ).cache_key.hex()


def descriptor(model, images, videos=()):
    transfer_id = "ab" * 16
    return dict(
        version=3,
        peer="grpc://127.0.0.1:1",
        native_model_type=model,
        embedding_width=2,
        request_id="req",
        transfer_id=transfer_id,
        part_id=transfer_id + "_local_part_0",
        part_idx=0,
        num_parts=1,
        image_count=len(images),
        images=images,
        videos=list(videos),
    )


@pytest.mark.parametrize(
    "model,cls",
    [
        ("kimi_vl", KimiVLImageProcessor),
        ("kimi_k25", KimiK2_5VLImageProcessor),
        ("kimi_k3", KimiK3ImageProcessor),
        ("kimi_k31", KimiK3ImageProcessor),
        ("inkling2", Inkling2MultimodalProcessor),
        ("dots3_note", DotsNoteOmniProcessor),
    ],
)
def test_native_image_layout_and_identity(model, cls):
    p = cls.__new__(cls)
    p.mm_tokens = SimpleNamespace(image_token_id=8)
    p.hf_config = SimpleNamespace(
        model_type=model, vision_config=SimpleNamespace(merge_kernel_size=(2, 2))
    )
    p.IMAGE_TOKEN_ID = 8
    p.mm_token_ids = {"im_token_id": 8}
    p.image_preprocessor = SimpleNamespace(merge_size=2)
    p._tokenizer = SimpleNamespace(
        encode=lambda text, **kwargs: [51] if "640x480" in text else [52]
    )
    h, w = (1, 2) if model == "inkling2" else (2, 4)
    rows = torch.arange(4, dtype=torch.float32).reshape(2, 2)
    image = dict(
        t=1, h=h, w=w, embedding_rows=2, identity=identity(Modality.IMAGE, rows)
    )
    if model in {"kimi_k3", "kimi_k31"}:
        image["original_size"] = [640, 480]
    payload = descriptor(model, [image])
    handoff = EncoderHandoff.parse(msgspec.json.encode(payload), 1)
    assert handoff.native_transfer().images[0].embedding_rows == 2
    prompt = [8, 9]
    p_layout = handoff.build_layout(p, prompt, 24)
    d_layout = handoff.build_layout(p, prompt, 24)
    expected = [51, 8, 8, 52, 9] if model in {"kimi_k3", "kimi_k31"} else [8, 8, 9]
    assert p_layout.input_ids == d_layout.input_ids == expected
    assert d_layout.mm_items[0].precomputed_embeddings is None
    handoff.bind_identities(d_layout.mm_items, None)
    p_layout.mm_items[0].precomputed_embeddings = rows
    handoff.bind_identities(p_layout.mm_items, None)
    assert d_layout.mm_items[0].cache_key == p_layout.mm_items[0].cache_key
    assert d_layout.mm_items[0].cache_key.hex() != image["identity"]
    p_layout.mm_items[0].precomputed_embeddings = rows + 1
    with pytest.raises(ValueError, match="identity mismatch"):
        handoff.bind_identities(p_layout.mm_items, None)
    image["embedding_rows"] = 1
    with pytest.raises(ValueError, match="rows"):
        EncoderHandoff.parse(msgspec.json.encode(payload), 1).build_layout(
            p, prompt, 24
        )


@pytest.mark.parametrize("model", ["glm4v", "glm4v_moe", "glm5_next"])
def test_glm_interleaved_frame_offsets_positions_and_packed_rows(model):
    p = Glm4vImageProcessor.__new__(Glm4vImageProcessor)
    p.IM_TOKEN_ID, p.VIDEO_TOKEN_ID = 8, 10
    p.hf_config = SimpleNamespace(
        model_type=model,
        image_token_id=8,
        video_start_token_id=11,
        video_end_token_id=12,
        vision_config=SimpleNamespace(spatial_merge_size=2),
    )
    component = SimpleNamespace(merge_size=2)
    p._processor = SimpleNamespace(
        image_processor=component,
        video_processor=component,
        replace_frame_token_id=lambda ts, num_image_tokens: (
            "A" + "x" * num_image_tokens + "B" + str(int(ts))
        ),
    )
    p._tokenizer = SimpleNamespace(
        encode=lambda text, **kwargs: [
            {"A": 17, "B": 18, "x": 8}.get(
                token, int(token) if token.isdigit() else None
            )
            for token in text
        ]
    )
    img = torch.tensor([[1.0, 2.0]])
    vid = torch.arange(8, dtype=torch.float32).reshape(4, 2)
    payload = descriptor(
        model,
        [dict(t=1, h=2, w=2, embedding_rows=1, identity=identity(Modality.IMAGE, img))],
        [
            dict(
                t=2,
                h=2,
                w=4,
                embedding_rows=4,
                identities=[identity(Modality.VIDEO, vid)],
                timestamps=[0.0, 2.0],
            )
        ],
    )
    handoff = EncoderHandoff.parse(msgspec.json.encode(payload), 1, 1)
    prompt = [11, 10, 12, 17, 8, 18]
    layout = handoff.build_layout(p, prompt, 24)
    assert [i.offsets for i in layout.mm_items] == [[(2, 3), (7, 8)], [(13, 13)]]
    assert layout.input_ids == [11, 17, 8, 8, 18, 0, 17, 8, 8, 18, 2, 12, 17, 8, 18]
    assert tuple(layout.mrope_positions.shape) == (3, len(layout.input_ids))
    decode = handoff.build_layout(p, prompt, 24)
    handoff.bind_identities(decode.mm_items, None)
    waiting = WaitingStagedImage.__new__(WaitingStagedImage)
    waiting.recv_req = SimpleNamespace(
        encoder_handoff=handoff, encoder_routing_hashes=None, mm_routing_labels=None
    )
    waiting.staged_layout = layout
    waiting._bind_pool_slot_to_mm_inputs = lambda mm: None
    waiting._finish_assemble({Modality.IMAGE: torch.cat((img, vid))})
    torch.testing.assert_close(layout.mm_items[0].precomputed_embeddings, vid)
    torch.testing.assert_close(layout.mm_items[1].precomputed_embeddings, img)
    assert [i.cache_key for i in layout.mm_items] == [
        i.cache_key for i in decode.mm_items
    ]
    p.hf_config.model_type = "kimi_vl"
    with pytest.raises(ValueError, match="model types"):
        handoff.build_layout(p, prompt, 24)


@pytest.mark.parametrize("model", ["muse_spark", "minimax_m3_vl", "mimo_v2"])
def test_native_clip_and_segment_layout(model):
    cls = {
        "muse_spark": MuseSpark1xOmniMultimodalProcessor,
        "minimax_m3_vl": MiniMaxM3VLProcessor,
        "mimo_v2": MiMoV2Processor,
    }[model]
    p = cls.__new__(cls)
    p.hf_config = SimpleNamespace(model_type=model)
    p.IM_TOKEN_ID, p.VIDEO_TOKEN_ID = 8, 10
    p.IM_START_TOKEN_ID, p.IM_END_TOKEN_ID = 11, 12
    tokenizer = SimpleNamespace(
        decode=lambda ids: "".join(
            {70: "<image>", 71: "<video>", 8: "<image>", 10: "<video>"}.get(i, "x")
            for i in ids
        ),
        encode=lambda text, **kwargs: [26],
    )
    if model == "muse_spark":
        p._tokenizer = tokenizer
        p.preprocess_config = SimpleNamespace(format_timestamp=str)
        p.patch_id, p.video_id, p.audio_id = 8, 10, 14
        p.image_placeholder_ids, p.video_placeholder_ids, p.audio_placeholder_ids = (
            [70],
            [71],
            [72],
        )
        p.image_start_id, p.image_end_id, p.video_start_id, p.video_end_id = (
            11,
            12,
            15,
            16,
        )
        p.audio_start_id, p.audio_end_id, p.video_separator_id = 17, 18, 19
    elif model == "minimax_m3_vl":
        tokens = {
            p.IMAGE_START_TOKEN: 11,
            p.IMAGE_END_TOKEN: 12,
            p.IMAGE_TOKEN: 8,
            p.VIDEO_TOKEN: 10,
        }
        tokenizer.encode = lambda text, **kwargs: [
            tokens.get(token, 26) for token in re.findall(r"\]<\].*?\[>\[", text)
        ]
        p._processor = SimpleNamespace(
            tokenizer=tokenizer,
            image_processor=SimpleNamespace(merge_size=2),
            video_processor=SimpleNamespace(merge_size=2),
        )
    else:
        p._tokenizer = tokenizer
        p.vision_config = SimpleNamespace(spatial_merge_size=2)
        p.mimo_processor = SimpleNamespace(
            temporal_compression_ratio=2,
            rope_type="normal",
            tokenizer=tokenizer,
            format_timestamp=str,
            vision_start_token_id=11,
            vision_end_token_id=12,
            video_start_token_id=15,
            video_end_token_id=16,
            image_token_id=8,
            video_token_id=10,
            audio_token_id=14,
        )
        p.mm_tokens = SimpleNamespace(
            get_combined_regex=lambda: r"(<image>|<video>)",
            get_modality_of_token=lambda s: {
                "<image>": Modality.IMAGE,
                "<video>": Modality.VIDEO,
            }.get(s),
        )
        p.AUDIO_START_TOKEN_ID, p.AUDIO_END_TOKEN_ID = 17, 18

    img = torch.tensor([[0.0, 1.0]])
    vid = torch.arange(8, dtype=torch.float32).reshape(4, 2)
    merge = 1 if model == "muse_spark" else 2
    video_ids = [
        identity(Modality.VIDEO, chunk)
        for chunk in (vid.chunk(2) if model == "mimo_v2" else (vid,))
    ]
    data = descriptor(
        model,
        [
            dict(
                t=1,
                h=merge,
                w=merge,
                embedding_rows=1,
                identity=identity(Modality.IMAGE, img),
            )
        ],
        [
            dict(
                t=4 if model == "mimo_v2" else 2,
                h=merge,
                w=merge * 2,
                embedding_rows=4,
                identities=video_ids,
                timestamps=[0.0, 1.0],
            )
        ],
    )
    handoff = EncoderHandoff.parse(msgspec.json.encode(data), None, None)
    layout = handoff.build_layout(p, [10, 8], 48)
    assert [item.modality for item in layout.mm_items] == (
        [Modality.VIDEO] * len(video_ids) + [Modality.IMAGE]
    )
    assert sum(len(item.offsets) for item in layout.mm_items) == 3
    handoff.bind_identities(layout.mm_items, None)
    if model == "minimax_m3_vl":
        data["videos"][0]["timestamps"] = []
        plain = EncoderHandoff.parse(
            msgspec.json.encode(data), None, None
        ).build_layout(p, [10, 8], 48)
        assert len(plain.input_ids) == len(layout.input_ids) - 2
