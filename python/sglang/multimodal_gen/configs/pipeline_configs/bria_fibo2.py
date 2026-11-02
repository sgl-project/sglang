# SPDX-License-Identifier: Apache-2.0
"""Pipeline configuration for BRIA Fibo 2.0.

Fibo 2.0 is text-to-image and image-editing in one model family: the same
transformer generates a structured-caption image and edits up to five reference
images, which it conditions on as clean "context" latents. This configuration
covers the text-to-image path; the editing path (vision side of Qwen3-VL plus
context latents) plugs into the same classes.

The FLUX.2 VAE the model ships normalizes latents with its own BatchNorm
statistics rather than a scalar scaling factor. `Flux2PipelineConfig` already
implements exactly that, so this class inherits it and overrides only what the
two models do differently:

* latents stay spatial (32 channels) instead of being packed into a sequence,
* the RoPE table is built inside the transformer, so no position ids are passed,
* the BatchNorm de-normalization runs inside `preprocess_decoding`, because the
  decoding stage applies `get_decode_scale_and_shift` *before* it and our latents
  are only 32 channels wide until the patchify step.
"""

import math
from dataclasses import dataclass, field
from typing import Any, Callable, ClassVar, Optional

import torch

from sglang.multimodal_gen.configs.models import DiTConfig, EncoderConfig, VAEConfig
from sglang.multimodal_gen.configs.models.dits.bria_fibo2 import BriaFibo2DitConfig
from sglang.multimodal_gen.configs.models.encoders.base import BaseEncoderOutput
from sglang.multimodal_gen.configs.models.encoders.bria_fibo2 import (
    PROMPT_TEMPLATE_ENCODE,
    VISION_MARKER,
    BriaFibo2Qwen3VLConfig,
    build_caption_bundles,
)
from sglang.multimodal_gen.configs.models.vaes.flux import Flux2VAEConfig
from sglang.multimodal_gen.configs.pipeline_configs.base import (
    TextConditioningOutput,
)
from sglang.multimodal_gen.configs.pipeline_configs.flux import Flux2PipelineConfig
from sglang.multimodal_gen.configs.task_type import ModelTaskType

# FLUX's linear shift schedule. Not FLUX.2's empirical formula.
BASE_IMAGE_SEQ_LEN = 256
MAX_IMAGE_SEQ_LEN = 4096
BASE_SHIFT = 0.5
MAX_SHIFT = 1.15

# The null caption Fibo 2.0 was trained with: every structured-caption key
# present, every value empty. Only used when guidance is on.
DEFAULT_NEGATIVE_PROMPT = (
    '{"short_description":"","objects":[],"background_setting":"","lighting":{},'
    '"aesthetics":{},"photographic_characteristics":{},"style_medium":"",'
    '"context":"","artistic_style":""}'
)

# Structured-caption keys training drops entirely
DROPPED_CAPTION_KEYS = ("aesthetic_score", "preference_score")


def calculate_shift(
    image_seq_len: int,
    base_seq_len: int = BASE_IMAGE_SEQ_LEN,
    max_seq_len: int = MAX_IMAGE_SEQ_LEN,
    base_shift: float = BASE_SHIFT,
    max_shift: float = MAX_SHIFT,
) -> float:
    """FLUX's linear shift, matching the reference `calculate_shift`."""
    m = (max_shift - base_shift) / (max_seq_len - base_seq_len)
    b = base_shift - m * base_seq_len
    return image_seq_len * m + b


def _is_empty(value: Any) -> bool:
    if isinstance(value, float):
        return value != value  # NaN
    return value is None or (isinstance(value, (str, dict, list)) and not value)


def _clean_caption(value: Any) -> Any:
    if isinstance(value, dict):
        cleaned = {k: _clean_caption(v) for k, v in value.items()}
        return {k: v for k, v in cleaned.items() if not _is_empty(v)}
    if isinstance(value, list):
        return [v for v in map(_clean_caption, value) if not _is_empty(v)]
    return value


def parse_caption_prompt(prompt: str) -> tuple[Any, Optional[str]]:
    """Split a prompt into its structured caption and its edit instruction.

    Mirrors the reference `_parse_prompt`: a prompt that is not a JSON object is
    returned as-is with no instruction. A caption is cleaned the way training
    cleaned captions - empty values and the two score fields are dropped - and a
    caption of exactly ``short_description`` plus ``text_render`` keeps both keys
    in that order, with the list even when empty (it says the image has no text).
    """
    try:
        import json

        caption = json.loads(prompt)
    except Exception:
        return prompt, None
    if not isinstance(caption, dict):
        return prompt, None

    instruction = caption.pop("edit_instruction", None)
    if "structured_caption" in caption:
        caption = caption.pop("structured_caption")
        if not isinstance(caption, dict):
            return caption or None, instruction

    keys = set(caption)
    cleaned = _clean_caption(caption)
    caption = {}
    for key, value in cleaned.items():
        if key in DROPPED_CAPTION_KEYS:
            continue
        if isinstance(value, dict):
            value = {k: v for k, v in value.items() if k not in DROPPED_CAPTION_KEYS}
        if value == {}:
            continue
        caption[key] = value

    if keys == {"short_description", "text_render"}:
        caption = {
            "short_description": caption.get("short_description") or "",
            "text_render": caption.get("text_render") or [],
        }
    return caption or None, instruction


def serialize_caption(caption: Any) -> str:
    """Compact JSON, keeping non-ASCII characters as they are."""
    import json

    return json.dumps(caption, separators=(",", ":"), ensure_ascii=False)


def normalize_prompt(prompt: str) -> str:
    """Put a generation caption in the trained form, compactly serialized.

    A prompt that is not a JSON caption, such as a short description, is left
    untouched.
    """
    caption, _ = parse_caption_prompt(prompt)
    if isinstance(caption, str):
        return prompt
    return serialize_caption(caption or {})


def render_prompt(
    prompt: str, prompt_template_encode: str = PROMPT_TEMPLATE_ENCODE
) -> str:
    """Normalize a caption, then wrap it in the Qwen3-VL chat turn."""
    return prompt_template_encode.format(normalize_prompt(prompt))


def bria_fibo2_preprocess_text(prompt: str) -> str:
    """Module-level default; the pipeline config binds the arch config's template."""
    return render_prompt(prompt)


# ---------------------------------------------------------------------------
# Editing: prompt assembly, mask handling and image sizing
# ---------------------------------------------------------------------------


def build_edit_prompt(prompt: str, num_images: int) -> str:
    """Assemble the JSON edit prompt Qwen3-VL sees, one marker per image.

    The reference builds one JSON object: a vision marker per image to edit, the
    caption of the result when the prompt carries one, then the edit
    instruction. A prompt that is not JSON is the instruction alone.
    """
    if num_images < 1:
        raise ValueError(f"an edit needs at least one image, got {num_images}")

    caption, instruction = parse_caption_prompt(prompt)
    if instruction is None:
        # the whole prompt is the instruction
        caption, instruction = None, prompt

    edit: dict[str, Any] = {}
    if num_images == 1:
        edit["image"] = VISION_MARKER
    else:
        for k in range(1, num_images + 1):
            edit[f"image_{k}"] = VISION_MARKER
    if caption:
        edit["structured_caption"] = caption
    edit["edit_instruction"] = instruction
    return serialize_caption(edit)


def apply_mask_to_image(image: Any, mask: Any) -> Any:
    """Grey out the region the mask marks before the image is resized.

    The mask is white where the model should repaint; that region is replaced
    with flat grey ``(128, 128, 128)`` on the *original* image, matching how
    training prepared edit pairs. The mask therefore never reaches the
    transformer: the generator learns the region from the greying itself.
    """
    from PIL import Image

    if mask.size != image.size:
        raise ValueError(
            f"mask size {mask.size} does not match image size {image.size}"
        )

    # white (>= 128) marks the region to regenerate
    binary = mask.convert("L").point(lambda value: 255 if value >= 128 else 0)

    image = image.convert("RGB")
    grey = Image.new("RGB", image.size, (128, 128, 128))
    return Image.composite(grey, image, binary)


def get_image_size(
    aspect_ratio: float,
    area: int = 1024 * 1024,
    multiple_of: int = 16,
    min_side: int = 64,
    tolerance: float = 0.1,
) -> tuple[int, int]:
    """Pick a trained-resolution canvas for an aspect ratio, about ``area``.

    Candidates are multiples of ``multiple_of`` with both sides at least
    ``min_side`` and an area within ``tolerance`` of ``area``. The reference
    prefers the canvas that crops the input the least, then the one whose area is
    closest, then the wider one. Returns ``(width, height)``.
    """
    if aspect_ratio <= 0:
        raise ValueError(f"aspect ratio must be positive, got {aspect_ratio}")

    target_area = area
    low, high = target_area * (1.0 - tolerance), target_area * (1.0 + tolerance)

    best = None
    # Walk the width axis in multiples; derive the height from the ratio.
    width = multiple_of
    while width * width / max(aspect_ratio, 1e-9) <= high * 4:
        height = int(round(width / aspect_ratio))
        height = (height // multiple_of) * multiple_of or multiple_of
        if height < min_side:
            width += multiple_of
            continue
        candidate_area = width * height
        if candidate_area < low:
            width += multiple_of
            continue
        if candidate_area > high:
            break

        # how much of the input this canvas has to crop
        crop_ratio = 1.0 - min(
            (width / height) / aspect_ratio, aspect_ratio / (width / height)
        )
        key = (round(crop_ratio, 6), abs(candidate_area - target_area), -width)
        if best is None or key < best[0]:
            best = (key, (width, height))
        width += multiple_of

    if best is None:
        # No candidate inside the band (extreme aspect ratio): fall back to the
        # nearest multiple pair for the target area.
        width = int(round(math.sqrt(target_area * aspect_ratio)))
        width = max(min_side, (width // multiple_of) * multiple_of or multiple_of)
        height = int(round(width / aspect_ratio))
        height = max(min_side, (height // multiple_of) * multiple_of or multiple_of)
        return width, height

    return best[1]


def crop_to_size(image: Any, width: int, height: int) -> Any:
    """Scale then centre-crop to the target canvas, the reference resize mode."""
    from PIL import Image

    scale = max(width / image.width, height / image.height)
    resized_size = (
        max(1, int(round(image.width * scale))),
        max(1, int(round(image.height * scale))),
    )
    resized = image.resize(resized_size, Image.Resampling.LANCZOS)
    left = (resized.width - width) // 2
    top = (resized.height - height) // 2
    return resized.crop((left, top, left + width, top + height))


def make_text_encoder_copy(image: Any, max_side: int = 768) -> Any:
    """The copy Qwen3-VL sees: the cropped image, shrunk to ``max_side``."""
    copy = image.copy()
    copy.thumbnail((max_side, max_side))
    return copy


def bria_fibo2_postprocess_text(
    outputs: BaseEncoderOutput,
    text_inputs: Any,
    pipeline_config: Any = None,
) -> TextConditioningOutput:
    """Turn the text tower's hidden states into the transformer's six bundles."""
    hidden_states = outputs.hidden_states
    if hidden_states is None:
        raise ValueError(
            "BriaFibo2 needs the text encoder's hidden states; the encoder was "
            "called without `output_hidden_states=True`."
        )

    if pipeline_config is not None:
        arch_config = pipeline_config.text_encoder_configs[0].arch_config
        num_hidden_layers = int(arch_config.num_hidden_layers)
    else:
        # Without the config, fall back to the tower's own length: the bundle
        # builder still validates it against the offset.
        num_hidden_layers = len(hidden_states)

    bundles = build_caption_bundles(hidden_states, num_hidden_layers)

    attention_mask = text_inputs["attention_mask"].bool()
    seq_lens = [int(m.sum().item()) for m in attention_mask]
    return TextConditioningOutput(
        prompt_embeds=bundles,
        prompt_embeds_mask=attention_mask,
        prompt_seq_lens=seq_lens,
    )


def prepare_mu(batch: Any, server_args: Any) -> tuple[str, float]:
    """Shift for the denoising schedule, from the image token count."""
    pipeline_config = server_args.pipeline_config
    vae_scale_factor = pipeline_config.get_vae_scale_factor()
    patch_size = pipeline_config.dit_config.arch_config.patch_size
    height_tokens = (int(batch.height) // vae_scale_factor) // patch_size
    width_tokens = (int(batch.width) // vae_scale_factor) // patch_size
    image_seq_len = height_tokens * width_tokens
    return "mu", calculate_shift(image_seq_len)


# The FLUX.2 VAE config carries its BatchNorm epsilon only when loaded from the
# checkpoint; the reference default is 1e-4.
DEFAULT_BATCH_NORM_EPS = 1e-4

# The released checkpoints all ship a 8x FLUX.2 VAE (block_out_channels
# [128, 256, 512, 512]); only used before the checkpoint config has been read.
FLUX2_DEFAULT_VAE_SCALE_FACTOR = 8


@dataclass
class BriaFibo2PipelineConfig(Flux2PipelineConfig):
    # One model family serves text-to-image and editing. The text-to-image path
    # runs on its own; editing adds the vision side of Qwen3-VL and the reference
    # latents, and is driven by `BriaFibo2EditPipeline`.
    task_type: ModelTaskType = ModelTaskType.T2I
    supported_task_types: ClassVar[tuple[ModelTaskType, ...]] = (
        ModelTaskType.T2I,
        ModelTaskType.I2I,
        ModelTaskType.TI2I,
    )

    dit_config: DiTConfig = field(default_factory=BriaFibo2DitConfig)
    vae_config: VAEConfig = field(default_factory=Flux2VAEConfig)

    # The transformer has no guidance embedding: classifier-free guidance is a
    # second forward pass, which the reference only runs for the dev checkpoint.
    should_use_guidance: bool = False

    text_encoder_configs: tuple[EncoderConfig, ...] = field(
        default_factory=lambda: (BriaFibo2Qwen3VLConfig(),)
    )
    text_encoder_precisions: tuple[str, ...] = field(default_factory=lambda: ("bf16",))
    vae_precision: str = "bf16"

    preprocess_text_funcs: tuple[Callable, ...] = field(
        default_factory=lambda: (bria_fibo2_preprocess_text,)
    )
    postprocess_text_funcs: tuple[Callable, ...] = field(
        default_factory=lambda: (bria_fibo2_postprocess_text,)
    )

    default_negative_prompt: str = DEFAULT_NEGATIVE_PROMPT
    # Output size defaults, mirroring the reference: 128 latent pixels, 8x VAE
    default_sample_size: int = 128
    tokenizer_max_length: int = 4096
    # Qwen3-VL sees each image at no more than this long side
    text_encoder_image_size: int = 768

    def __post_init__(self) -> None:
        # Flux2's Mistral-specific tokenizer extras do not apply here. The base
        # dataclasses define no `__post_init__`, so nothing is chained.
        self.text_encoder_extra_args = [
            dict(
                max_length=self.tokenizer_max_length,
                padding="longest",
                truncation=True,
                return_tensors="pt",
            )
        ]
        # One definition, one consumption: the template lives on the encoder arch
        # config, and both the prompt rendering and the pipeline read it from
        # there rather than repeating the literal.
        self.prompt_template_encode = self.text_encoder_configs[
            0
        ].arch_config.prompt_template_encode
        self.preprocess_text_funcs = (self.preprocess_text,)

    # -- text encoding ------------------------------------------------------

    def preprocess_text(self, prompt: str) -> str:
        return render_prompt(prompt, self.prompt_template_encode)

    def tokenize_prompt(self, prompts: list[str], tokenizer, tok_kwargs) -> dict:
        """Tokenize the already chat-templated prompts."""
        effective_max_length = tok_kwargs.pop("max_length", self.tokenizer_max_length)
        return tokenizer(
            prompts,
            padding="longest",
            truncation=True,
            max_length=effective_max_length,
            return_tensors="pt",
        )

    def get_text_encoder_attention_mask(self, text_inputs, encoder_index):
        return text_inputs.get("attention_mask")

    def get_text_encoder_pooler_output(self, outputs, encoder_index):
        # Fibo has no pooled text embedding
        return None

    # -- latents ------------------------------------------------------------

    def get_vae_scale_factor(self) -> int:
        """Latent downscale: 8 for the FLUX.2 VAE Fibo ships.

        `vae_loader` calls `vae_config.post_init()`, which copies the value onto
        the arch config from `block_out_channels`; neither exists until the
        checkpoint config has been read. `spatial_compression_ratio` is not a
        usable fallback because `FluxVAEArchConfig` defaults it to 1 for the
        FLUX.1 VAE, so fall back to the FLUX.2 value directly.
        """
        arch = self.vae_config.arch_config
        scale = getattr(arch, "vae_scale_factor", None)
        if scale:
            return int(scale)
        block_out_channels = getattr(arch, "block_out_channels", None)
        if block_out_channels:
            return 2 ** (len(block_out_channels) - 1)
        return FLUX2_DEFAULT_VAE_SCALE_FACTOR

    def get_batch_norm_eps(self) -> float:
        arch = self.vae_config.arch_config
        return float(getattr(arch, "batch_norm_eps", DEFAULT_BATCH_NORM_EPS))

    def prepare_latent_shape(self, batch, batch_size, num_frames):
        """Spatial latents: the transformer patchifies them itself."""
        vae_scale_factor = self.get_vae_scale_factor()
        height = int(batch.height) // vae_scale_factor
        width = int(batch.width) // vae_scale_factor
        channels = self.dit_config.arch_config.in_channels
        return (batch_size, channels, height, width)

    def get_latent_dtype(self, prompt_dtype: torch.dtype) -> torch.dtype:
        # The reference samples in fp32 and keeps the scheduler state there,
        # casting only the model input to the transformer's dtype
        return torch.float32

    def maybe_pack_latents(self, latents, batch_size, batch):
        # Fibo keeps latents spatial; nothing is packed into a sequence
        return latents

    def maybe_prepare_latent_ids(self, latents):
        return None

    def post_denoising_loop(self, latents, batch):
        return latents

    def slice_noise_pred(self, noise, latents):
        return noise

    # -- conditioning -------------------------------------------------------

    def prepare_pos_cond_kwargs(self, batch, device, rotary_emb, dtype):
        cond_kwargs: dict[str, Any] = {}
        # The transformer owns its three-axis RoPE table, so no position ids
        # travel from here. The images to edit arrive as `batch.source_latents`,
        # the same field the in-tree LLaDA edit model uses: a list of
        # (batch, in_channels, h_i, w_i) tensors, each keeping its own size,
        # which is why they cannot be sequence-concatenated into
        # `batch.image_latent`.
        source_latents = getattr(batch, "source_latents", None)
        if source_latents:
            cond_kwargs["source_latents"] = [
                latent.to(dtype=dtype) for latent in source_latents
            ]
        return cond_kwargs

    def prepare_neg_cond_kwargs(self, batch, device, rotary_emb, dtype):
        # The images to edit condition both guidance passes alike
        return self.prepare_pos_cond_kwargs(batch, device, rotary_emb, dtype)

    def get_pos_prompt_embeds(self, batch):
        return batch.prompt_embeds[0]

    def get_neg_prompt_embeds(self, batch):
        if batch.negative_prompt_embeds is None:
            return None
        return batch.negative_prompt_embeds[0]

    # -- editing: image preprocessing --------------------------------------

    def calculate_condition_image_size(self, image, width, height):
        """Canvas for a reference image: about one megapixel, 16-aligned.

        The reference generates at the trained resolution closest to the first
        image's aspect ratio rather than at the image's own size.
        """
        if image is None:
            return None
        aspect_ratio = image.width / image.height
        multiple_of = self.get_vae_scale_factor() * 2
        return get_image_size(aspect_ratio, multiple_of=multiple_of)

    def prepare_calculated_size(self, image):
        return self.calculate_condition_image_size(image, None, None)

    def _load_mask(self, batch):
        """The repaint mask the request asked for, as a PIL image, if any."""
        from PIL import Image

        mask_path = getattr(batch, "mask_path", None)
        if mask_path is None:
            return None
        return Image.open(mask_path).convert("L")

    def preprocess_vae_image(self, batch, vae_image_processor):
        """Grey out the requested repaint region on the *original* images.

        The request mask arrives as `mask_path` and this hook is the first place
        the batch is available, so the greying happens here: on the whole image,
        before `preprocess_condition_image` crops it, exactly as training prepared
        edit pairs. The transformer never sees the mask, it infers the region from
        the flat grey.

        `InputValidationStage` calls `preprocess_condition_image` per image
        without the batch, so applying the mask there would never reach a request.
        """
        mask = self._load_mask(batch)
        if mask is None:
            return batch

        condition_image = batch.condition_image
        images = (
            list(condition_image)
            if isinstance(condition_image, (list, tuple))
            else [condition_image]
        )
        if len(images) != 1:
            raise ValueError(
                "A repaint mask applies to a single image, but the request "
                f"edits {len(images)} images."
            )
        # `InputValidationStage` iterates `condition_image` right after this hook,
        # so the greyed image has to stay a one-element list.
        batch.condition_image = [apply_mask_to_image(images[0], mask)]
        return batch

    def preprocess_condition_image(
        self, image, target_width, target_height, vae_image_processor=None
    ):
        """Crop the image to the generation canvas.

        The VAE copy is this crop; a repaint mask has already been applied to
        `batch.condition_image` by `preprocess_vae_image`.
        """
        return crop_to_size(image, target_width, target_height), (
            target_width,
            target_height,
        )

    # -- editing: text side -------------------------------------------------

    def prepare_image_processor_kwargs(self, batch, neg=False) -> dict:
        """Build the edit prompts Qwen3-VL sees, one JSON object per prompt.

        Every prompt edits all the request's images, so each gets one vision
        marker per image, in the order the images are handed to the processor.
        """
        prompt = batch.prompt if not neg else batch.negative_prompt
        if not prompt:
            return {}

        prompts = [prompt] if isinstance(prompt, str) else list(prompt)
        images = getattr(batch, "condition_image", None)
        if images is None:
            return {}

        images = images if isinstance(images, (list, tuple)) else [images]
        num_images = len(images)

        rendered = [
            self.prompt_template_encode.format(build_edit_prompt(p, num_images))
            for p in prompts
        ]

        # Qwen3-VL must not see the full-resolution canvas: it gets the same
        # cropped copy shrunk to a 768 pixel long side. `ImageEncodingStage`
        # prefers `per_prompt_images` over `batch.condition_image`, which is why
        # the two copies can differ.
        text_encoder_images = self._text_encoder_copies(images)
        return dict(
            text=rendered, per_prompt_images=[text_encoder_images] * len(rendered)
        )

    def _text_encoder_copies(self, images):
        """The copies Qwen3-VL sees: the VAE crop, shrunk to 768 on the long side.

        `InputValidationStage` runs before text encoding, so `images` are already
        the cropped canvases (and greyed, when a mask was supplied); only the
        downscale is left to do here.
        """
        return [
            make_text_encoder_copy(image, max_side=self.text_encoder_image_size)
            for image in images
        ]

    # -- VAE ----------------------------------------------------------------

    def get_decode_scale_and_shift(self, device, dtype, vae):
        # The BatchNorm statistics are per patchified channel (128), while the
        # latents are still 32 channels wide here. The decoding stage applies
        # this *before* `preprocess_decoding`, so the whole de-normalization
        # happens there instead; keep this a no-op.
        return 1.0, None

    def preprocess_decoding(self, latents, server_args=None, vae=None):
        """Undo the BatchNorm normalization, then return 32-channel latents.

        Mirrors the reference decode order: patchify (32 -> 128 channels),
        multiply by the BatchNorm std and add its mean, unpatchify, then decode.
        """
        if vae is None or not getattr(vae, "bn", None) is not None:
            return latents

        from sglang.multimodal_gen.configs.pipeline_configs.flux import (
            _patchify_latents,
            _unpatchify_latents,
        )

        patchified = _patchify_latents(latents.float())
        bn_mean = vae.bn.running_mean.view(1, -1, 1, 1).to(
            patchified.device, patchified.dtype
        )
        bn_std = torch.sqrt(
            vae.bn.running_var.view(1, -1, 1, 1) + self.get_batch_norm_eps()
        ).to(patchified.device, patchified.dtype)
        denormalized = patchified * bn_std + bn_mean
        return _unpatchify_latents(denormalized)

    def standard_text_len(self) -> int:
        return self.tokenizer_max_length


def register() -> None:
    from sglang.multimodal_gen.configs.sample.bria_fibo2 import (
        BriaFibo2SamplingParams,
        BriaFibo2TurboSamplingParams,
    )
    from sglang.multimodal_gen.registry import register_configs

    register_configs(
        sampling_param_cls=BriaFibo2TurboSamplingParams,
        pipeline_config_cls=BriaFibo2PipelineConfig,
        hf_model_paths=["briaai/Fibo-2.0-edit-turbo"],
        model_detectors=[
            lambda hf_id: "fibo-2" in hf_id.lower() and "turbo" in hf_id.lower()
        ],
    )
    register_configs(
        sampling_param_cls=BriaFibo2SamplingParams,
        pipeline_config_cls=BriaFibo2PipelineConfig,
        hf_model_paths=["briaai/Fibo-2.0-dev"],
        model_detectors=[
            lambda hf_id: "fibo-2" in hf_id.lower() or "fibo2" in hf_id.lower()
        ],
    )
    # `BriaFibo2Pipeline` and `BriaFibo2EditPipeline` are registered by the
    # pipeline discovery pass, which scans `EntryClass` (a list in that module);
    # the checkpoint's `_class_name` picks which one runs.
