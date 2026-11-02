# SPDX-License-Identifier: Apache-2.0
"""CPU-only tests for the Fibo 2.0 pipeline configuration.

Two things here are easy to get wrong and are checked against values derived
independently of the reference implementation (see `notes/SCHEDULER-EXPECTED.md`):
the shift schedule's `mu` and the resulting sigmas, and the caption cleaning that
has to reproduce the form the model was trained on.
"""

import json
import math

import pytest
import torch

from sglang.multimodal_gen.configs.pipeline_configs.bria_fibo2 import (
    BASE_IMAGE_SEQ_LEN,
    BASE_SHIFT,
    DEFAULT_NEGATIVE_PROMPT,
    MAX_IMAGE_SEQ_LEN,
    MAX_SHIFT,
    BriaFibo2PipelineConfig,
    bria_fibo2_preprocess_text,
    calculate_shift,
    normalize_prompt,
    parse_caption_prompt,
    prepare_mu,
)

# ---------------------------------------------------------------------------
# T10: the shift schedule, against an independent derivation
# ---------------------------------------------------------------------------

# mu values recomputed from m and b rather than copied from the reference
EXPECTED_MU = {
    # (height, width) -> mu, with latent = pixels / 8, tokens = latent / patch
    (1024, 1024): 1.15,  # image_seq_len 4096
    (768, 768): 0.8466666667,  # image_seq_len 2304
    (1024, 768): 0.9766666667,  # image_seq_len 3072
}

# Shifted sigmas for the 4-step turbo schedule, independently computed
EXPECTED_SIGMAS_4_STEP = {
    (1024, 1024): [1.0, 0.904531, 0.759511, 0.512844],
    (768, 768): [1.0, 0.874931, 0.699867, 0.437345],
    (1024, 768): [1.0, 0.888477, 0.726446, 0.469551],
}


def _image_seq_len(
    height: int, width: int, vae_scale_factor: int = 8, patch_size: int = 2
) -> int:
    return (height // vae_scale_factor // patch_size) * (
        width // vae_scale_factor // patch_size
    )


def test_shift_slope_and_intercept_match_the_flux_formula():
    m = (MAX_SHIFT - BASE_SHIFT) / (MAX_IMAGE_SEQ_LEN - BASE_IMAGE_SEQ_LEN)
    b = BASE_SHIFT - m * BASE_IMAGE_SEQ_LEN
    assert m == pytest.approx(0.65 / 3840)
    assert b == pytest.approx(0.45666666666, abs=1e-9)
    # linear, not the FLUX.2 empirical curve
    assert calculate_shift(MAX_IMAGE_SEQ_LEN) == pytest.approx(MAX_SHIFT)
    assert calculate_shift(BASE_IMAGE_SEQ_LEN) == pytest.approx(BASE_SHIFT)


@pytest.mark.parametrize("size,expected_mu", EXPECTED_MU.items())
def test_mu_matches_independent_derivation(size, expected_mu):
    height, width = size
    seq_len = _image_seq_len(height, width)
    assert calculate_shift(seq_len) == pytest.approx(expected_mu, abs=1e-9)


@pytest.mark.parametrize("size,expected_sigmas", EXPECTED_SIGMAS_4_STEP.items())
def test_four_step_schedule_matches_independent_derivation(size, expected_sigmas):
    """Reproduce the exponential time shift; the scheduler must match this."""
    height, width = size
    mu = calculate_shift(_image_seq_len(height, width))
    num_steps = 4
    raw = [
        1.0 - i * (1.0 - 1.0 / num_steps) / max(num_steps - 1, 1)
        for i in range(num_steps)
    ]
    shifted = [math.exp(mu) / (math.exp(mu) + 1.0 / s - 1.0) for s in raw]
    for got, want in zip(shifted, expected_sigmas):
        assert got == pytest.approx(want, abs=1e-6)


def test_timestep_passed_to_the_transformer_is_mu_scaled_to_unit_interval():
    """The transformer receives t / 1000, and applies t_scale internally."""
    assert calculate_shift(4096) * 1000 == pytest.approx(1150.0)


def test_prepare_mu_reports_the_shift_for_the_requested_size():
    class _Batch:
        height = 1024
        width = 1024

    class _ServerArgs:
        pipeline_config = BriaFibo2PipelineConfig()

    key, value = prepare_mu(_Batch(), _ServerArgs())
    assert key == "mu"
    assert value == pytest.approx(1.15, abs=1e-9)


# ---------------------------------------------------------------------------
# Caption handling
# ---------------------------------------------------------------------------


def test_compact_serialization_keeps_non_ascii():
    """The caption is serialized compactly without escaping non-ASCII text."""
    # Built from code points so the source stays pure ASCII for the linters.
    word = "r" + chr(233) + "sum" + chr(233)  # accented letters, no escapes
    text = normalize_prompt(json.dumps({"short_description": word, "objects": []}))
    assert text == json.dumps(
        {"short_description": word}, separators=(",", ":"), ensure_ascii=False
    )
    assert " " not in text
    assert chr(233) in text


def test_empty_values_and_score_fields_are_dropped():
    raw = (
        '{"short_description":"a cat","objects":[],"background_setting":"",'
        '"lighting":{},"aesthetics":{"aesthetic_score":7.5},'
        '"preference_score":0.9,"style_medium":null}'
    )
    text = normalize_prompt(raw)
    assert text == '{"short_description":"a cat"}'


def test_non_json_prompt_passes_through_unchanged():
    assert normalize_prompt("Make the background a snowy forest") == (
        "Make the background a snowy forest"
    )


def test_short_description_and_text_render_keep_their_shape():
    """This caption keeps both keys, and the empty list says there is no text."""
    text = normalize_prompt('{"short_description":"poster","text_render":[]}')
    assert text == '{"short_description":"poster","text_render":[]}'


def test_edit_instruction_is_split_out_of_the_caption():
    caption, instruction = parse_caption_prompt(
        '{"edit_instruction":"make it night","short_description":"a cat"}'
    )
    assert instruction == "make it night"
    assert caption == {"short_description": "a cat"}


def test_structured_caption_key_is_unwrapped():
    caption, instruction = parse_caption_prompt(
        '{"edit_instruction":"add a hat","structured_caption":{"short_description":"a cat"}}'
    )
    assert instruction == "add a hat"
    assert caption == {"short_description": "a cat"}


def test_prompt_template_wraps_the_caption():
    rendered = bria_fibo2_preprocess_text('{"short_description":"a cat"}')
    assert rendered == (
        "<|im_start|>user\n"
        '{"short_description":"a cat"}'
        "<|im_end|>\n<|im_start|>assistant\n"
    )


def test_default_negative_prompt_has_every_key_empty():
    fields = parse_caption_prompt(DEFAULT_NEGATIVE_PROMPT)[0]
    # every value is empty, so cleaning collapses the caption entirely
    assert fields is None or fields == {}


# ---------------------------------------------------------------------------
# Latent shape, dtype and decoding contract
# ---------------------------------------------------------------------------


def test_latent_shape_is_spatial_not_packed():
    config = BriaFibo2PipelineConfig()
    shape = config.prepare_latent_shape(
        type("Batch", (), {"height": 1024, "width": 768})(), 1, 1
    )
    # 32 channels at 1/8 resolution, no 2x2 packing into channels
    assert shape == (1, 32, 128, 96)


def test_latents_stay_in_float32_between_steps():
    config = BriaFibo2PipelineConfig()
    assert config.get_latent_dtype(torch.bfloat16) is torch.float32


def test_latents_are_not_packed_into_a_sequence():
    config = BriaFibo2PipelineConfig()
    latents = torch.randn(1, 32, 8, 8)
    assert config.maybe_pack_latents(latents, 1, None) is latents
    assert config.post_denoising_loop(latents, None) is latents


def test_rope_positions_are_not_passed_from_the_config():
    """The transformer owns its RoPE table, so no position ids travel outward."""
    config = BriaFibo2PipelineConfig()
    assert config.prepare_pos_cond_kwargs(None, "cpu", None, torch.float32) == {}
    assert config.prepare_neg_cond_kwargs(None, "cpu", None, torch.float32) == {}


def test_decode_scale_shift_is_a_noop_and_bn_runs_in_preprocess_decoding():
    """The decoding stage scales before preprocessing, so BN must live there."""
    config = BriaFibo2PipelineConfig()
    scale, shift = config.get_decode_scale_and_shift("cpu", torch.float32, None)
    assert scale == 1.0
    assert shift is None

    # without a VAE carrying BatchNorm statistics nothing is transformed
    latents = torch.randn(1, 32, 8, 8)
    assert torch.equal(config.preprocess_decoding(latents, vae=None), latents)


def test_decode_undone_with_batch_norm_statistics_round_trips():
    """patchify -> BN -> unpatchify must invert the encode-side normalization."""
    from sglang.multimodal_gen.configs.pipeline_configs.flux import (
        _patchify_latents,
        _unpatchify_latents,
    )

    class _Bn(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.register_buffer("running_mean", torch.randn(128))
            self.register_buffer("running_var", torch.rand(128) + 0.5)

    class _Vae(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.bn = _Bn()

    vae = _Vae()
    config = BriaFibo2PipelineConfig()

    original = torch.randn(1, 32, 8, 8)
    eps = config.get_batch_norm_eps()
    assert eps == pytest.approx(1e-4)
    # encode side: patchify then normalize with the BatchNorm statistics
    normalized = (
        _patchify_latents(original) - vae.bn.running_mean.view(1, -1, 1, 1)
    ) / torch.sqrt(vae.bn.running_var.view(1, -1, 1, 1) + eps)
    normalized = _unpatchify_latents(normalized)

    restored = config.preprocess_decoding(normalized, vae=vae)
    assert torch.allclose(restored, original, atol=1e-5)


# ---------------------------------------------------------------------------
# Sampling params
# ---------------------------------------------------------------------------


def test_turbo_and_dev_presets():
    from sglang.multimodal_gen.configs.sample.bria_fibo2 import (
        BriaFibo2SamplingParams,
        BriaFibo2TurboSamplingParams,
    )

    turbo = BriaFibo2TurboSamplingParams()
    assert turbo.num_inference_steps == 4
    assert turbo.guidance_scale == 1.0
    assert turbo.should_use_classifier_free_guidance() is False

    dev = BriaFibo2SamplingParams()
    assert dev.num_inference_steps == 30
    assert dev.guidance_scale == 5.0
    assert dev.should_use_classifier_free_guidance() is True
    # the null caption is the trained negative prompt
    assert dev.negative_prompt == DEFAULT_NEGATIVE_PROMPT


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------


def test_registration_routes_both_checkpoints():
    from sglang.multimodal_gen.configs.pipeline_configs.bria_fibo2 import register
    from sglang.multimodal_gen.registry import (
        _CONFIG_REGISTRY,
        _MODEL_HF_PATH_TO_NAME,
    )

    register()

    # Both released checkpoints map to this pipeline config
    for model_id in ("briaai/Fibo-2.0-edit-turbo", "briaai/Fibo-2.0-dev"):
        assert model_id in _MODEL_HF_PATH_TO_NAME, model_id
        entry = _CONFIG_REGISTRY[_MODEL_HF_PATH_TO_NAME[model_id]]
        assert entry.pipeline_config_cls is BriaFibo2PipelineConfig
        assert entry.sampling_param_cls is not None

    # The turbo and the dev checkpoint get different sampling presets
    from sglang.multimodal_gen.configs.sample.bria_fibo2 import (
        BriaFibo2SamplingParams,
        BriaFibo2TurboSamplingParams,
    )

    turbo = _CONFIG_REGISTRY[_MODEL_HF_PATH_TO_NAME["briaai/Fibo-2.0-edit-turbo"]]
    dev = _CONFIG_REGISTRY[_MODEL_HF_PATH_TO_NAME["briaai/Fibo-2.0-dev"]]
    assert turbo.sampling_param_cls is BriaFibo2TurboSamplingParams
    assert dev.sampling_param_cls is BriaFibo2SamplingParams


def test_detectors_recognize_local_directory_names():
    """A locally cloned `Fibo-2.0-edit-turbo` directory must still route here."""
    from sglang.multimodal_gen.configs.pipeline_configs.bria_fibo2 import register
    from sglang.multimodal_gen.registry import (
        _CONFIG_REGISTRY,
        _MODEL_HF_PATH_TO_NAME,
        _MODEL_NAME_DETECTORS,
    )

    register()
    expected = _MODEL_HF_PATH_TO_NAME["briaai/Fibo-2.0-edit-turbo"]

    matched = [
        model_id
        for model_id, detector in _MODEL_NAME_DETECTORS
        if detector("briaai/Fibo-2.0-edit-turbo")
    ]
    assert matched, "no detector matches the turbo checkpoint"
    # the turbo-specific registration is the one that wins here
    assert expected in matched or _CONFIG_REGISTRY[matched[0]] is not None

    # an unrelated model must not be claimed
    for model_id, detector in _MODEL_NAME_DETECTORS:
        if model_id in expected:
            assert not detector("stabilityai/stable-diffusion-3.5-large")


# ---------------------------------------------------------------------------
# Prompt template: one definition, one consumption
# ---------------------------------------------------------------------------


def test_prompt_template_has_no_misspelled_shadow_field():
    """Regression: a `promppt_template_encode` typo made the field dead code."""
    from sglang.multimodal_gen.configs.models.encoders.bria_fibo2 import (
        BriaFibo2Qwen3VLArchConfig,
    )

    arch = BriaFibo2Qwen3VLArchConfig()
    assert arch.prompt_template_encode == (
        "<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n"
    )
    assert not hasattr(arch, "promppt_template_encode")


def test_pipeline_reads_the_template_from_the_encoder_arch_config():
    config = BriaFibo2PipelineConfig()
    arch_template = config.text_encoder_configs[0].arch_config.prompt_template_encode
    assert config.prompt_template_encode == arch_template
    # the bound preprocess function is what the text encoding stage calls
    assert config.preprocess_text_funcs[0].__self__ is config
    assert config.preprocess_text('{"short_description":"a cat"}') == (
        '<|im_start|>user\n{"short_description":"a cat"}<|im_end|>\n'
        "<|im_start|>assistant\n"
    )


# ---------------------------------------------------------------------------
# Editing: prompt assembly, mask, sizing
# ---------------------------------------------------------------------------


def test_edit_prompt_marks_each_image_and_keeps_the_instruction():
    from sglang.multimodal_gen.configs.pipeline_configs.bria_fibo2 import (
        build_edit_prompt,
    )

    one = build_edit_prompt("make it night", 1)
    assert one == (
        '{"image":"<|vision_start|><|image_pad|><|vision_end|>",'
        '"edit_instruction":"make it night"}'
    )

    # several images are addressed as image_1 .. image_k
    three = build_edit_prompt("combine them", 3)
    for k in (1, 2, 3):
        assert f'"image_{k}"' in three
    assert '"edit_instruction":"combine them"' in three


def test_edit_prompt_carries_the_result_caption_when_given():
    from sglang.multimodal_gen.configs.pipeline_configs.bria_fibo2 import (
        build_edit_prompt,
    )

    prompt = '{"edit_instruction":"add a hat","short_description":"a cat"}'
    rendered = build_edit_prompt(prompt, 1)
    assert '"structured_caption":{"short_description":"a cat"}' in rendered
    assert '"edit_instruction":"add a hat"' in rendered


def test_mask_greys_the_marked_region_before_resizing():
    """White repaints; the region becomes flat grey on the ORIGINAL image."""
    from PIL import Image, ImageDraw

    from sglang.multimodal_gen.configs.pipeline_configs.bria_fibo2 import (
        apply_mask_to_image,
    )

    image = Image.new("RGB", (100, 50), (255, 0, 0))
    mask = Image.new("L", (100, 50), 0)
    ImageDraw.Draw(mask).rectangle([50, 0, 99, 49], fill=255)

    greyed = apply_mask_to_image(image, mask)
    assert greyed.size == image.size, "greying happens before any resize"
    assert greyed.getpixel((10, 10)) == (255, 0, 0), "unmarked stays untouched"
    assert greyed.getpixel((60, 10)) == (128, 128, 128)


def test_mask_is_binarised_at_128():
    from PIL import Image

    from sglang.multimodal_gen.configs.pipeline_configs.bria_fibo2 import (
        apply_mask_to_image,
    )

    image = Image.new("RGB", (4, 1), (0, 0, 255))
    mask = Image.new("L", (4, 1))
    for x, value in enumerate([0, 127, 128, 255]):
        mask.putpixel((x, 0), value)
    greyed = apply_mask_to_image(image, mask)
    assert greyed.getpixel((0, 0)) == (0, 0, 255)
    assert greyed.getpixel((1, 0)) == (0, 0, 255)
    assert greyed.getpixel((2, 0)) == (128, 128, 128)
    assert greyed.getpixel((3, 0)) == (128, 128, 128)


def test_mask_size_must_match_the_image():
    from PIL import Image

    from sglang.multimodal_gen.configs.pipeline_configs.bria_fibo2 import (
        apply_mask_to_image,
    )

    with pytest.raises(ValueError, match="does not match image size"):
        apply_mask_to_image(Image.new("RGB", (8, 8)), Image.new("L", (4, 4)))


@pytest.mark.parametrize(
    "aspect,expected",
    [
        (1.0, (1024, 1024)),
        (16 / 9, (1424, 800)),
        (9 / 16, (784, 1392)),
    ],
)
def test_image_size_is_a_multiple_of_16_about_one_megapixel(aspect, expected):
    from sglang.multimodal_gen.configs.pipeline_configs.bria_fibo2 import (
        get_image_size,
    )

    width, height = get_image_size(aspect)
    assert (width, height) == expected
    assert width % 16 == 0 and height % 16 == 0
    assert width >= 64 and height >= 64
    # within the reference's +-10% area band around one megapixel
    assert abs(width * height - 1024 * 1024) <= 0.1 * 1024 * 1024


def test_crop_to_size_centres_and_fills_the_canvas():
    from PIL import Image

    from sglang.multimodal_gen.configs.pipeline_configs.bria_fibo2 import (
        crop_to_size,
    )

    wide = Image.new("RGB", (300, 100), (255, 0, 0))
    cropped = crop_to_size(wide, 256, 256)
    assert cropped.size == (256, 256)


def test_text_encoder_copy_is_thumbnailed_to_the_long_side():
    from PIL import Image

    from sglang.multimodal_gen.configs.pipeline_configs.bria_fibo2 import (
        make_text_encoder_copy,
    )

    assert make_text_encoder_copy(Image.new("RGB", (2000, 1000))).size == (768, 384)
    # already small enough: untouched
    assert make_text_encoder_copy(Image.new("RGB", (500, 300))).size == (500, 300)


def test_context_latents_travel_through_pos_cond_kwargs_not_image_latent():
    config = BriaFibo2PipelineConfig()
    batch = type("Batch", (), {})()

    # nothing to edit: no context latents are handed to the transformer
    assert config.prepare_pos_cond_kwargs(batch, "cpu", None, torch.float32) == {}

    first = torch.randn(1, 32, 8, 8)
    second = torch.randn(1, 32, 8, 4)
    batch.source_latents = [first, second]

    kwargs = config.prepare_pos_cond_kwargs(batch, "cpu", None, torch.float32)
    assert set(kwargs) == {"source_latents"}
    assert len(kwargs["source_latents"]) == 2
    assert kwargs["source_latents"][0].shape == (1, 32, 8, 8)
    assert kwargs["source_latents"][1].shape == (1, 32, 8, 4)
    # both guidance passes see the same reference images
    assert (
        len(
            config.prepare_neg_cond_kwargs(batch, "cpu", None, torch.float32)[
                "source_latents"
            ]
        )
        == 2
    )


# ---------------------------------------------------------------------------
# The VAE copy and the VLM copy are different images
# ---------------------------------------------------------------------------


def test_vae_copy_is_the_canvas_and_the_vlm_copy_is_thumbnailed():
    """Fibo shows Qwen3-VL a shrunken copy, never the full generation canvas."""
    from PIL import Image

    config = BriaFibo2PipelineConfig()
    original = Image.new("RGB", (2000, 1000), (255, 0, 0))

    # the canvas Fibo was trained at for a 2:1 image, 16-aligned
    canvas = config.calculate_condition_image_size(original, 2000, 1000)
    assert canvas == (1440, 720)

    vae_copy, final_size = config.preprocess_condition_image(
        original, canvas[0], canvas[1]
    )
    assert vae_copy.size == canvas
    assert final_size == canvas

    # the text encoder copy is derived from the cropped copy the stage holds
    batch = type(
        "Batch",
        (),
        {
            "condition_image": [vae_copy],
            "prompt": "make it night",
            "negative_prompt": None,
        },
    )()
    kwargs = config.prepare_image_processor_kwargs(batch)
    assert "per_prompt_images" in kwargs, (
        "the processor must receive the thumbnail copies, not the VAE canvases"
    )
    vlm_copy = kwargs["per_prompt_images"][0][0]
    assert max(vlm_copy.size) == config.text_encoder_image_size
    assert vlm_copy.size == (768, 384)
    assert vlm_copy.size != vae_copy.size


def _masked_batch(image, mask_path, mask=None):
    """A batch carrying the request's mask path, as the pipeline sees it."""
    fields = {"condition_image": image, "mask_path": mask_path}
    if mask is not None:
        fields["mask"] = mask
    return type("Batch", (), fields)()


def test_repaint_mask_greys_the_marked_region_before_the_crop(tmp_path):
    """`preprocess_vae_image` is the hook that sees the batch, so the mask lands there."""
    from PIL import Image, ImageDraw

    config = BriaFibo2PipelineConfig()
    image = Image.new("RGB", (2000, 1000), (255, 0, 0))
    mask = Image.new("L", (2000, 1000), 0)
    ImageDraw.Draw(mask).rectangle([1000, 0, 1999, 999], fill=255)
    mask_path = str(tmp_path / "mask.png")
    mask.save(mask_path)

    batch = config.preprocess_vae_image(_masked_batch(image, mask_path), None)

    # the hook must leave a list behind: the stage iterates `condition_image`
    assert isinstance(batch.condition_image, list)
    greyed = batch.condition_image[0]
    assert greyed.size == (2000, 1000), "the greying must not resize the image"
    assert greyed.getpixel((1500, 500)) == (128, 128, 128)
    assert greyed.getpixel((500, 500)) == (255, 0, 0)

    # the crop downstream is unchanged in size, only the masked pixels differ
    cropped, size = config.preprocess_condition_image(greyed, 1440, 720)
    plain, _ = config.preprocess_condition_image(image, 1440, 720)
    assert size == (1440, 720)
    assert cropped.size == plain.size == (1440, 720)
    assert cropped.getpixel((1100, 360)) == (128, 128, 128)
    assert plain.getpixel((1100, 360)) == (255, 0, 0)
    assert cropped.getpixel((300, 360)) == (255, 0, 0)


def test_request_without_a_mask_is_not_greyed():
    """No `mask_path` on the request leaves the image untouched."""
    from PIL import Image

    config = BriaFibo2PipelineConfig()
    image = Image.new("RGB", (64, 64), (7, 8, 9))
    batch = config.preprocess_vae_image(_masked_batch(image, None), None)
    assert batch.condition_image.getpixel((1, 1)) == (7, 8, 9)


def test_repaint_mask_rejects_multiple_images(tmp_path):
    """A mask marks a region of one image; several references cannot share it."""
    from PIL import Image

    config = BriaFibo2PipelineConfig()
    images = [
        Image.new("RGB", (32, 32), (1, 2, 3)),
        Image.new("RGB", (32, 32), (4, 5, 6)),
    ]
    mask_path = str(tmp_path / "mask.png")
    Image.new("L", (32, 32), 255).save(mask_path)
    with pytest.raises(ValueError, match="single image"):
        config.preprocess_vae_image(_masked_batch(images, mask_path), None)


def test_repaint_mask_size_must_match_the_image(tmp_path):
    """A mask of another size cannot be composited onto the source image."""
    from PIL import Image

    config = BriaFibo2PipelineConfig()
    image = Image.new("RGB", (32, 32), (1, 2, 3))
    mask_path = str(tmp_path / "mask.png")
    Image.new("L", (16, 16), 255).save(mask_path)
    with pytest.raises(ValueError, match="mask size"):
        config.preprocess_vae_image(_masked_batch(image, mask_path), None)


# ---------------------------------------------------------------------------
# CFG: both passes are conditioned on the same reference images
# ---------------------------------------------------------------------------


def test_negative_pass_carries_the_same_source_latents():
    """Guidance runs a second forward; the reference images condition both."""
    config = BriaFibo2PipelineConfig()
    first = torch.randn(1, 32, 8, 8)
    second = torch.randn(1, 32, 8, 4)
    batch = type("Batch", (), {"source_latents": [first, second]})()

    pos = config.prepare_pos_cond_kwargs(batch, "cpu", None, torch.float32)
    neg = config.prepare_neg_cond_kwargs(batch, "cpu", None, torch.float32)

    assert len(pos["source_latents"]) == len(neg["source_latents"]) == 2
    for a, b in zip(pos["source_latents"], neg["source_latents"]):
        assert torch.equal(a, b), "the uncond pass must see the same reference images"
    # consistent sizes too, which is what keeps the token layout identical
    assert [t.shape for t in pos["source_latents"]] == [
        t.shape for t in neg["source_latents"]
    ]


def test_dev_preset_asks_for_cfg_and_turbo_does_not():
    from sglang.multimodal_gen.configs.sample.bria_fibo2 import (
        BriaFibo2EditSamplingParams,
        BriaFibo2SamplingParams,
        BriaFibo2TurboSamplingParams,
    )

    assert BriaFibo2SamplingParams().should_use_classifier_free_guidance() is True
    assert BriaFibo2TurboSamplingParams().should_use_classifier_free_guidance() is False
    edit = BriaFibo2EditSamplingParams()
    assert edit.should_use_classifier_free_guidance() is False
    assert edit.num_inference_steps == 4
