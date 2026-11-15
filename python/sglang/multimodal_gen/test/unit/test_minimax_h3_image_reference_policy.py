# SPDX-License-Identifier: Apache-2.0
"""Image-release hooks in the shared H3 reference path."""

from types import SimpleNamespace

import pytest
import torch
from PIL import Image

from sglang.multimodal_gen.configs.models.vaes.minimax_h3_video import (
    MiniMaxH3VideoVAEArchConfig,
)
from sglang.multimodal_gen.runtime.cache.conditioning import (
    ConditioningCache,
    invalidate_conditioning_caches,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.minimax_h3 import (
    material_io,
    presentation,
    reference_encoding,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.minimax_h3.constants import (
    MINIMAX_H3_REFERENCE_IMAGE_ROWS_EXTRA_KEY as REFS,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.minimax_h3.constants import (
    MINIMAX_H3_TEXT_EMBEDDINGS_EXTRA_KEY as TEXT,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.minimax_h3.keyframe_encoding import (
    minimax_h3_encode_keyframe_cond_rows,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.minimax_h3.packed_tokens import (
    minimax_h3_patchify_video_latent,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.minimax_h3.reference_encoding import (
    minimax_h3_image_reference_size,
    minimax_h3_prepared_reference_image,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.minimax_h3.stages.text_encoding import (
    MiniMaxH3TextEncodingStage,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.minimax_h3.stages.visual_encoding import (
    MiniMaxH3VisualEncodingStage,
)
from sglang.multimodal_gen.test.unit.test_minimax_h3_image_pipeline import (
    make_req,
)
from sglang.multimodal_gen.test.unit.test_minimax_h3_image_pipeline import (
    server as server,
)


class LatentVAE(torch.nn.Module):
    def __init__(self, latent):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.zeros(1))
        self.latent = latent

    def encode_images(self, image, use_fp16_latent):
        assert use_fp16_latent
        return [self.latent]


def test_even_crop_keeps_leading_latent_rows_and_columns():
    latent = torch.randn(24, 1, 3, 5).half()
    arch = MiniMaxH3VideoVAEArchConfig(latents_mean=[0.5] * 24, latents_std=[2.0] * 24)
    rows = minimax_h3_encode_keyframe_cond_rows(
        LatentVAE(latent), object(), arch, crop_to_even=True
    )
    z = (latent[None, :, :, :2, :4].float() - 0.5) / 2.0
    expected = minimax_h3_patchify_video_latent(z, patch_size=[1, 2, 2])
    assert torch.equal(rows, expected)


def _prepare(path, monkeypatch):
    monkeypatch.setattr(
        material_io, "minimax_h3_localize_material_uri", lambda *a, **k: str(path)
    )
    material = SimpleNamespace(
        uri=str(path),
        condition_type="image",
        condition_index=0,
        material_chain="image.reference_preserve",
    )
    plan = SimpleNamespace(output_modality="image", materials=[material])
    return minimax_h3_prepared_reference_image(SimpleNamespace(extra={}), plan)


@pytest.mark.parametrize(
    "orientation,transpose",
    [
        (3, Image.Transpose.ROTATE_180),
        (6, Image.Transpose.ROTATE_270),
        (8, Image.Transpose.ROTATE_90),
        (2, Image.Transpose.FLIP_LEFT_RIGHT),
    ],
)
def test_image_release_reads_references_upright(
    tmp_path, monkeypatch, orientation, transpose
):
    # A PNG keeps the pixels exactly, so the comparison is byte for byte.
    pixels = Image.effect_noise((83, 61), 64).convert("RGB")
    path = tmp_path / "oriented.png"
    exif = Image.Exif()
    exif[0x0112] = orientation
    pixels.save(path, exif=exif)
    image = _prepare(path, monkeypatch)["images"][0]["image"]
    upright = pixels.transpose(transpose)
    assert image.mode == "RGB"
    assert image.size == upright.size
    assert image.tobytes() == upright.tobytes()


def test_image_release_keeps_untagged_references_unchanged(tmp_path, monkeypatch):
    pixels = Image.effect_noise((83, 61), 64).convert("RGB")
    path = tmp_path / "plain.png"
    pixels.save(path)
    image = _prepare(path, monkeypatch)["images"][0]["image"]
    assert image.size == (83, 61)
    assert image.tobytes() == pixels.tobytes()


def test_image_release_rejects_animated_references(tmp_path, monkeypatch):
    path = tmp_path / "animated.png"
    frames = [Image.new("RGB", (64, 64), color) for color in ("red", "blue")]
    frames[0].save(path, save_all=True, append_images=frames[1:])
    with pytest.raises(ValueError, match="animated"):
        _prepare(path, monkeypatch)


@pytest.mark.parametrize(
    "bounds,size,expected",
    [
        # Short edge below the minimum: scaled up to it, long edge on the 32px grid.
        ((1024, 1536), (720, 1280), (1024, 1824)),
        ((1024, 1536), (1280, 720), (1824, 1024)),
        # Above the maximum: scaled down to it.
        ((1024, 1536), (3000, 4000), (1536, 2048)),
        ((1024, 1536), (4096, 4096), (1536, 1536)),
        # Inside the bounds, including on either bound: the original pixels.
        ((1024, 1536), (1080, 1920), None),
        ((1024, 1536), (1024, 4096), None),
        ((1024, 1536), (1536, 3000), None),
        # Equal bounds fix the short edge, both up and down.
        ((512, 512), (720, 1280), (512, 896)),
        ((512, 512), (1920, 1080), (896, 512)),
        ((512, 512), (300, 400), (512, 672)),
        ((512, 512), (512, 777), None),
    ],
)
def test_size_puts_the_short_edge_within_the_bounds(bounds, size, expected):
    low, high = bounds
    assert (
        minimax_h3_image_reference_size(*size, min_short_edge=low, max_short_edge=high)
        == expected
    )


class SizeRecordingVAE(torch.nn.Module):
    parallel_tiling = False

    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.zeros(1))
        self.sizes = []

    def encode_images(self, image, use_fp16_latent, generator=None):
        self.sizes.append(image.size)
        width, height = image.size
        return [torch.zeros(24, 1, height // 16, width // 16).half()]


def _bound(server, monkeypatch, consumer, low, high):
    for name, value in (("min", low), ("max", high)):
        monkeypatch.setattr(
            server.pipeline_config,
            f"edit_reference_{consumer}_{name}_short_edge",
            value,
        )


def test_the_vae_input_follows_the_vae_bounds(server, monkeypatch):
    _bound(server, monkeypatch, "vae", 1024, 1536)
    prepared = {
        "images": [
            {"image": Image.new("RGB", size), "condition_index": index}
            for index, size in enumerate([(720, 1280), (1080, 1920)])
        ]
    }
    monkeypatch.setattr(
        reference_encoding,
        "minimax_h3_prepared_reference_image",
        lambda batch, plan: prepared,
    )
    req = make_req(server, "edit")
    req.extra.pop(REFS)
    vae = SizeRecordingVAE()
    MiniMaxH3VisualEncodingStage(
        vae, server.pipeline_config.vae_config.arch_config
    ).forward(req, server)

    assert vae.sizes == [(1024, 1824), (1080, 1920)]
    # Rows, grids (and the noise drawn on them) follow the VAE input.
    refs = req.extra[REFS]["images"]
    assert [(ref["latent_h"], ref["latent_w"]) for ref in refs] == [
        (114, 64),
        (120, 66),
    ]
    assert [ref["rows"].shape[0] for ref in refs] == [114 * 64 // 4, 120 * 66 // 4]
    # The prepared images stay at their original size for the Qwen side.
    assert [item["image"].size for item in prepared["images"]] == [
        (720, 1280),
        (1080, 1920),
    ]


class Encoder(torch.nn.Module):
    model = SimpleNamespace(get_rope_index=None)

    def encode_ids(self, ids, **kwargs):
        # H3 text_dim: the Qwen3-VL hidden size.
        return torch.zeros(ids.shape[0], 5120, dtype=torch.bfloat16)


def test_the_qwen_input_follows_the_qwen_bounds(server, monkeypatch):
    _bound(server, monkeypatch, "qwen", 512, 512)
    sizes = []

    def image_processor(images, return_tensors):
        sizes.extend(image.size for image in images)
        return {
            "pixel_values": torch.zeros(len(images) * 4, 1536),
            "image_grid_thw": torch.tensor([[1, 2, 2]] * len(images)),
        }

    processor = SimpleNamespace(image_processor=image_processor)
    image_processor.merge_size = 2
    # (padded_ids, token_tags, mRoPE positions) for a 4-token prompt.
    monkeypatch.setattr(
        presentation,
        "minimax_h3_image_presentation",
        lambda tokenizer, **kwargs: (
            torch.arange(4),
            torch.ones(4, dtype=torch.long),
            torch.zeros(3, 1, 4, dtype=torch.long),
        ),
    )
    prepared = {
        "images": [
            {"image": Image.new("RGB", size), "condition_index": index}
            for index, size in enumerate([(720, 1280), (1920, 1080), (512, 777)])
        ]
    }
    monkeypatch.setattr(
        reference_encoding,
        "minimax_h3_prepared_reference_image",
        lambda batch, plan: prepared,
    )
    req = make_req(server, "edit")
    req.extra.pop(TEXT)
    MiniMaxH3TextEncodingStage(Encoder(), object(), processor).forward(req, server)

    # Short edge pinned to 512: enlarged, shrunk, left as is.
    assert sizes == [(512, 896), (896, 512), (512, 777)]
    # The VAE side reads the same prepared images, at their original size.
    assert [item["image"].size for item in prepared["images"]] == [
        (720, 1280),
        (1920, 1080),
        (512, 777),
    ]


@torch.no_grad()
def test_image_conditioning_reuses_preprocessing_with_content_invalidation(
    server, monkeypatch
):
    class CountingEncoder(Encoder):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.ones(()))
            self.calls = 0

        def encode_ids(self, ids, **kwargs):
            self.calls += 1
            return torch.ones(ids.shape[0], 5120, dtype=torch.bfloat16) * self.weight

    processed = []

    def image_processor(images, return_tensors):
        processed.append([image.size for image in images])
        return {
            "pixel_values": torch.zeros(len(images) * 4, 1536),
            "image_grid_thw": torch.tensor([[1, 2, 2]] * len(images)),
        }

    image_processor.merge_size = 2
    monkeypatch.setattr(
        presentation,
        "minimax_h3_image_presentation",
        lambda tokenizer, **kwargs: (
            torch.arange(4),
            torch.ones(4, dtype=torch.long),
            torch.zeros(3, 1, 4, dtype=torch.long),
        ),
    )
    prepared = {
        "images": [
            {"image": Image.new("RGB", (32, 32), color)} for color in ("red", "blue")
        ]
    }
    monkeypatch.setattr(
        reference_encoding,
        "minimax_h3_prepared_reference_image",
        lambda batch, plan: prepared,
    )
    encoder = CountingEncoder().eval()
    stage = MiniMaxH3TextEncodingStage(
        encoder, object(), SimpleNamespace(image_processor=image_processor)
    )
    cache = ConditioningCache(256 * 1024)
    plan = SimpleNamespace(prompt="first", materials=[object()])
    with cache.scope():
        stage._encode_image(None, plan, server.pipeline_config)["positive"][
            "hidden_states"
        ].zero_()
        restored = stage._encode_image(None, plan, server.pipeline_config)
        assert encoder.calls == len(processed) == 1
        assert restored["positive"]["hidden_states"].eq(1).all()
        plan.prompt = "changed"
        stage._encode_image(None, plan, server.pipeline_config)
        prepared["images"].reverse()
        stage._encode_image(None, plan, server.pipeline_config)
        prepared["images"][0]["image"].putpixel((0, 0), (1, 2, 3))
        stage._encode_image(None, plan, server.pipeline_config)
        _bound(server, monkeypatch, "qwen", 64, 64)
        stage._encode_image(None, plan, server.pipeline_config)
        assert encoder.calls == len(processed) == 5
        invalidate_conditioning_caches([encoder])
        stage._encode_image(None, plan, server.pipeline_config)
        assert encoder.calls == len(processed) == 6
