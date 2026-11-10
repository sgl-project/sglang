# SPDX-License-Identifier: Apache-2.0
"""Model-specific stages for HiDream-O1-Image.

HiDream-O1-Image is a Qwen3-VL backbone that predicts clean 32x32 raw pixel
patches, so it has neither a VAE nor a separate text encoder. The
before-denoising stage therefore produces token ids, interleaved-mrope positions
and a mixed causal/bidirectional attention mask instead of prompt embeddings,
the denoising stage converts the predicted clean image into a flow velocity, and
decoding is a pure unpatchify.
"""

from __future__ import annotations

import math
from copy import copy
from typing import Any

import numpy as np
import torch
import torchvision.transforms.v2 as transforms
from PIL import Image

from sglang.multimodal_gen import envs
from sglang.multimodal_gen.configs.pipeline_configs.hidream_o1_image import (
    HIDREAM_O1_COND_KEY,
    HIDREAM_O1_NEG_COND_KEY,
)
from sglang.multimodal_gen.configs.sample.hidream_o1_image import (
    _VARIANT_DEFAULTS,
    HIDREAM_O1_SUPPORTED_RESOLUTIONS,
    _resolve_variant,
)
from sglang.multimodal_gen.runtime.cache.cache_dit_integration import (
    CacheDitConfig,
    enable_cache_on_transformer,
    refresh_context_on_transformer,
)
from sglang.multimodal_gen.runtime.disaggregation.roles import RoleType
from sglang.multimodal_gen.runtime.distributed import (
    get_tp_group,
    model_parallel_is_initialized,
)
from sglang.multimodal_gen.runtime.models.dits.hidream_o1_image import (
    HiO1CacheBlockAdapter,
)
from sglang.multimodal_gen.runtime.models.schedulers.scheduling_flow_match_euler_discrete import (
    FlowMatchEulerDiscreteScheduler,
)
from sglang.multimodal_gen.runtime.models.schedulers.scheduling_flow_unipc_multistep import (
    FlowUniPCMultistepScheduler,
)
from sglang.multimodal_gen.runtime.pipelines_core.schedule_batch import (
    OutputBatch,
    Req,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.base import PipelineStage
from sglang.multimodal_gen.runtime.pipelines_core.stages.denoising import (
    DenoisingContext,
    DenoisingStage,
    DenoisingStepState,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.hidream_o1_image.prompt_enhance import (
    prepare_prompt,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.validators import (
    V,
    VerificationResult,
)
from sglang.multimodal_gen.runtime.server_args import ServerArgs
from sglang.multimodal_gen.runtime.utils.logging_utils import init_logger
from sglang.multimodal_gen.runtime.utils.nvtx_pytorch_hooks import maybe_nvtx_range
from sglang.multimodal_gen.runtime.utils.precision import resolve_precision

logger = init_logger(__name__)


TIMESTEP_TOKEN_NUM = 1
PATCH_SIZE = 32
HIDREAM_O1_T_EPS = 0.001
CONDITION_IMAGE_SIZE = 384
REROLL_OUT_OF_RANGE_THRESHOLD = 0.001
REROLL_MAX_ATTEMPTS = 3

# Historical HIDREAM_O1_* aliases: the tested package surface imports these
# names, while the recipe code reads the per-variant noise table.
HIDREAM_O1_BOI_TOKEN = "<|boi_token|>"
HIDREAM_O1_TMS_TOKEN = "<|tms_token|>"
# Image mrope positions start at a fixed offset instead of continuing the text
# positions, so the grid does not shift when the prompt length changes.
HIDREAM_O1_MROPE_FIX_POINT = 4096

# Single source of truth is the sampling-params config; aliased here under
# the historical name (the manual parity harness extracts this symbol).
PREDEFINED_RESOLUTIONS = HIDREAM_O1_SUPPORTED_RESOLUTIONS

DEFAULT_TIMESTEPS = [
    999,
    987,
    974,
    960,
    945,
    929,
    913,
    895,
    877,
    857,
    836,
    814,
    790,
    764,
    737,
    707,
    675,
    640,
    602,
    560,
    515,
    464,
    409,
    347,
    278,
    199,
    110,
    8,
]


def _set_flow_match_fixed_timesteps(
    sched: FlowMatchEulerDiscreteScheduler,
    timesteps_list,
    device: torch.device,
) -> None:
    """Pin the fixed ``DEFAULT_TIMESTEPS`` schedule onto a flow-match Euler
    scheduler, bypassing the repo scheduler's default sigma shifting (the
    timesteps are already shifted)."""
    timesteps = torch.tensor(timesteps_list, device=device, dtype=torch.float32)
    sigmas = torch.tensor(
        [t / 1000.0 for t in timesteps_list] + [0.0],
        device=device,
        dtype=torch.float32,
    )
    sched.timesteps = timesteps
    sched.sigmas = sigmas
    sched.num_inference_steps = len(timesteps_list)
    sched._step_index = None
    sched._begin_index = None


class FlashFlowMatchScheduler:
    """Official DEV-model flash scheduler: Euler with per-step noise injection."""

    def __init__(self):
        self.timesteps = None
        self.sigmas = None
        self._step_index = None
        self.num_inference_steps = None

    def set_timesteps(
        self, num_inference_steps: int, shift: float, device: torch.device
    ):
        base_min = 1.0 / 1000
        sigma_min = shift * base_min / (1 + (shift - 1) * base_min)
        sigma_max = 1.0
        base = np.linspace(sigma_max, sigma_min, num_inference_steps)
        sigmas = shift * base / (1 + (shift - 1) * base)
        timesteps = (sigmas * 1000.0).astype(np.float32)
        sigmas_full = np.concatenate([sigmas, [0.0]]).astype(np.float32)

        self.sigmas = torch.from_numpy(sigmas_full).to(
            device=device, dtype=torch.float32
        )
        self.timesteps = torch.from_numpy(timesteps).to(
            device=device, dtype=torch.float32
        )
        self.num_inference_steps = num_inference_steps
        self._step_index = None

    def set_timesteps_list(self, timesteps_list, device: torch.device):
        timesteps = torch.tensor(timesteps_list, device=device, dtype=torch.long)
        sigmas = [t.item() / 1000.0 for t in timesteps]
        sigmas.append(0.0)
        self.sigmas = torch.tensor(sigmas, device=device, dtype=torch.float32)
        self.timesteps = timesteps
        self.num_inference_steps = len(timesteps)
        self._step_index = None

    def step(
        self,
        model_output: torch.Tensor,
        timestep: torch.Tensor,
        sample: torch.Tensor,
        s_noise: float = 1.0,
        noise_clip_std: float = 0.0,
        generator: torch.Generator | None = None,
        noise: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Euler step; pre-drawn ``noise`` replaces the internal per-step draw."""
        if self._step_index is None:
            self._step_index = 0
        sigma = self.sigmas[self._step_index]
        sample = sample.to(torch.float32)
        denoised = sample - model_output * sigma
        if self._step_index < self.num_inference_steps:
            sigma_next = self.sigmas[self._step_index + 1]
            if noise is None:
                if generator is not None and generator.device.type == "cpu":
                    noise = torch.randn(
                        model_output.shape, generator=generator, dtype=denoised.dtype
                    ).to(model_output.device)
                else:
                    noise = torch.randn(
                        model_output.shape,
                        device=model_output.device,
                        dtype=denoised.dtype,
                        generator=generator,
                    )
                if noise_clip_std > 0:
                    clip_val = noise_clip_std * noise.std().item()
                    noise = noise.clamp(min=-clip_val, max=clip_val)
            sample = sigma_next * noise * s_noise + (1.0 - sigma_next) * denoised
        self._step_index += 1
        return sample


def build_scheduler(
    num_inference_steps: int,
    timesteps_list,
    shift: float,
    device: torch.device,
    scheduler_name: str = "default",
):
    if scheduler_name == "flash":
        sched = FlashFlowMatchScheduler()
    elif scheduler_name == "flow_match":
        sched = FlowMatchEulerDiscreteScheduler(shift=shift)
    elif scheduler_name == "default":
        sched = FlowUniPCMultistepScheduler(use_dynamic_shifting=False, shift=shift)
    else:
        raise ValueError(f"Unknown scheduler_name={scheduler_name!r}")
    if timesteps_list is not None:
        if scheduler_name == "flow_match":
            _set_flow_match_fixed_timesteps(sched, timesteps_list, device)
        else:
            sched.set_timesteps_list(timesteps_list, device)
    elif scheduler_name == "flash":
        sched.set_timesteps(num_inference_steps, shift, device)
    else:
        sched.set_timesteps(num_inference_steps, device)
    return sched


def resolve_scheduler_name(
    editing: bool,
    model_type: str,
    scheduler_name: str | None = None,
) -> str:
    if scheduler_name is None:
        scheduler_name = _VARIANT_DEFAULTS[model_type]["scheduler_name"]
    if editing and model_type != "full" and scheduler_name != "flow_match":
        logger.info("Editing: scheduler %s -> flow_match", scheduler_name)
        return "flow_match"
    return scheduler_name


def find_closest_resolution(width: int, height: int) -> tuple[int, int]:
    """Find the closest predefined resolution by aspect ratio (matches official)."""
    img_ratio = width / height
    best_res = None
    min_diff = float("inf")
    for w, h in PREDEFINED_RESOLUTIONS:
        ratio = w / h
        diff = abs(ratio - img_ratio)
        if diff < min_diff:
            min_diff = diff
            best_res = (w, h)
    return best_res


def get_rope_index_fix_point(
    spatial_merge_size: int,
    image_token_id: int,
    video_token_id: int,
    vision_start_token_id: int,
    input_ids: torch.Tensor,
    image_grid_thw: torch.Tensor | None = None,
    video_grid_thw: torch.Tensor | None = None,
    attention_mask: torch.Tensor | None = None,
    skip_vision_start_token: list[int] | None = None,
    fix_point: int = 4096,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute 3D MRoPE position IDs (ported from official utils.py)."""
    if skip_vision_start_token is None:
        skip_vision_start_token = [1]

    if video_grid_thw is not None:
        video_grid_thw = torch.repeat_interleave(
            video_grid_thw, video_grid_thw[:, 0], dim=0
        )
        video_grid_thw[:, 0] = 1

    mrope_position_deltas = []
    if input_ids is not None and (
        image_grid_thw is not None or video_grid_thw is not None
    ):
        total_input_ids = input_ids
        if attention_mask is None:
            attention_mask = torch.ones_like(total_input_ids)
        position_ids = torch.ones(
            3,
            input_ids.shape[0],
            input_ids.shape[1],
            dtype=input_ids.dtype,
            device=input_ids.device,
        )
        image_index, video_index = 0, 0
        attention_mask = attention_mask.to(total_input_ids.device)

        for i, input_ids_i in enumerate(total_input_ids):
            input_ids_i = input_ids_i[attention_mask[i] == 1]
            vision_start_indices = torch.argwhere(
                input_ids_i == vision_start_token_id
            ).squeeze(1)
            vision_tokens = input_ids_i[vision_start_indices + 1]
            image_nums = (vision_tokens == image_token_id).sum()
            video_nums = (vision_tokens == video_token_id).sum()
            input_tokens = input_ids_i.tolist()
            llm_pos_ids_list: list = []
            st = 0
            remain_images, remain_videos = image_nums, video_nums

            for _ in range(image_nums + video_nums):
                ed_image = (
                    input_tokens.index(image_token_id, st)
                    if image_token_id in input_tokens and remain_images > 0
                    else len(input_tokens) + 1
                )
                ed_video = (
                    input_tokens.index(video_token_id, st)
                    if video_token_id in input_tokens and remain_videos > 0
                    else len(input_tokens) + 1
                )
                if ed_image < ed_video:
                    t, h, w = (
                        image_grid_thw[image_index][0],
                        image_grid_thw[image_index][1],
                        image_grid_thw[image_index][2],
                    )
                    image_index += 1
                    remain_images -= 1
                    ed = ed_image
                else:
                    t, h, w = (
                        video_grid_thw[video_index][0],
                        video_grid_thw[video_index][1],
                        video_grid_thw[video_index][2],
                    )
                    video_index += 1
                    remain_videos -= 1
                    ed = ed_video

                llm_grid_t = t.item()
                llm_grid_h = h.item() // spatial_merge_size
                llm_grid_w = w.item() // spatial_merge_size
                text_len = ed - st
                text_len -= skip_vision_start_token[image_index - 1]
                text_len = max(0, text_len)

                st_idx = (
                    llm_pos_ids_list[-1].max() + 1 if len(llm_pos_ids_list) > 0 else 0
                )
                llm_pos_ids_list.append(
                    torch.arange(text_len).view(1, -1).expand(3, -1) + st_idx
                )

                t_index = (
                    torch.arange(llm_grid_t)
                    .view(-1, 1)
                    .expand(-1, llm_grid_h * llm_grid_w)
                    .flatten()
                )
                h_index = (
                    torch.arange(llm_grid_h)
                    .view(1, -1, 1)
                    .expand(llm_grid_t, -1, llm_grid_w)
                    .flatten()
                )
                w_index = (
                    torch.arange(llm_grid_w)
                    .view(1, 1, -1)
                    .expand(llm_grid_t, llm_grid_h, -1)
                    .flatten()
                )

                if skip_vision_start_token[image_index - 1]:
                    if fix_point > 0:
                        fix_point = fix_point - st_idx
                    llm_pos_ids_list.append(
                        torch.stack([t_index, h_index, w_index]) + fix_point + st_idx
                    )
                    fix_point = 0
                else:
                    llm_pos_ids_list.append(
                        torch.stack([t_index, h_index, w_index]) + text_len + st_idx
                    )
                st = ed + llm_grid_t * llm_grid_h * llm_grid_w

            if st < len(input_tokens):
                st_idx = (
                    llm_pos_ids_list[-1].max() + 1 if len(llm_pos_ids_list) > 0 else 0
                )
                text_len = len(input_tokens) - st
                llm_pos_ids_list.append(
                    torch.arange(text_len).view(1, -1).expand(3, -1) + st_idx
                )

            llm_positions = torch.cat(llm_pos_ids_list, dim=1).reshape(3, -1)
            position_ids[..., i, attention_mask[i] == 1] = llm_positions.to(
                position_ids.device
            )
            mrope_position_deltas.append(
                llm_positions.max() + 1 - len(total_input_ids[i])
            )

        mrope_position_deltas = torch.tensor(
            mrope_position_deltas, device=input_ids.device
        ).unsqueeze(1)
        return position_ids, mrope_position_deltas
    else:
        if attention_mask is not None:
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)
            position_ids = (
                position_ids.unsqueeze(0).expand(3, -1, -1).to(attention_mask.device)
            )
            max_position_ids = position_ids.max(0, keepdim=False)[0].max(
                -1, keepdim=True
            )[0]
            mrope_position_deltas = max_position_ids + 1 - attention_mask.shape[-1]
        else:
            position_ids = (
                torch.arange(input_ids.shape[1], device=input_ids.device)
                .view(1, 1, -1)
                .expand(3, input_ids.shape[0], -1)
            )
            mrope_position_deltas = torch.zeros(
                [input_ids.shape[0], 1], device=input_ids.device, dtype=input_ids.dtype
            )
        return position_ids, mrope_position_deltas


def build_t2i_text_sample(
    prompt: str,
    height: int,
    width: int,
    tokenizer: Any,
    processor: Any,
    model_config: Any,
) -> dict[str, torch.Tensor]:
    """Build text sample for text-to-image generation."""
    image_token_id = model_config.image_token_id
    video_token_id = model_config.video_token_id
    vision_start_token_id = model_config.vision_start_token_id
    image_len = (height // PATCH_SIZE) * (width // PATCH_SIZE)

    boi_token = getattr(tokenizer, "boi_token", "<|boi_token|>")
    tms_token = getattr(tokenizer, "tms_token", "<|tms_token|>")

    messages = [{"role": "user", "content": prompt}]
    template_caption = (
        processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        + boi_token
        + tms_token * TIMESTEP_TOKEN_NUM
    )
    input_ids = tokenizer.encode(
        template_caption, return_tensors="pt", add_special_tokens=False
    )

    image_grid_thw = torch.tensor(
        [1, height // PATCH_SIZE, width // PATCH_SIZE], dtype=torch.int64
    ).unsqueeze(0)

    vision_tokens = torch.zeros((1, image_len), dtype=input_ids.dtype) + image_token_id
    vision_tokens[0, 0] = vision_start_token_id
    input_ids_pad = torch.cat([input_ids, vision_tokens], dim=-1)

    position_ids, _ = get_rope_index_fix_point(
        1,
        image_token_id,
        video_token_id,
        vision_start_token_id,
        input_ids=input_ids_pad,
        image_grid_thw=image_grid_thw,
        video_grid_thw=None,
        attention_mask=None,
        skip_vision_start_token=[1],
    )

    txt_seq_len = input_ids.shape[-1]
    all_seq_len = position_ids.shape[-1]

    token_types = torch.zeros((1, all_seq_len), dtype=input_ids.dtype)
    bgn = txt_seq_len - TIMESTEP_TOKEN_NUM
    token_types[0, bgn : bgn + image_len + TIMESTEP_TOKEN_NUM] = 1
    token_types[0, txt_seq_len - TIMESTEP_TOKEN_NUM : txt_seq_len] = 3

    vinput_mask = token_types == 1
    token_types_bin = (token_types > 0).to(token_types.dtype)

    return {
        "input_ids": input_ids,
        "position_ids": position_ids,
        "token_types": token_types_bin,
        "vinput_mask": vinput_mask,
    }


def patchify_pixels(pixels: torch.Tensor, patch_size: int) -> torch.Tensor:
    batch, channels, height, width = pixels.shape
    h_patches, w_patches = height // patch_size, width // patch_size
    patches = pixels.reshape(
        batch, channels, h_patches, patch_size, w_patches, patch_size
    )
    patches = patches.permute(0, 2, 4, 1, 3, 5)
    return patches.reshape(batch, h_patches * w_patches, channels * patch_size**2)


def unpatchify_pixels(
    patches: torch.Tensor,
    *,
    channels: int,
    patch_size: int,
    h_patches: int,
    w_patches: int,
) -> torch.Tensor:
    batch = patches.shape[0]
    pixels = patches.reshape(
        batch, h_patches, w_patches, channels, patch_size, patch_size
    )
    pixels = pixels.permute(0, 3, 1, 4, 2, 5)
    return pixels.reshape(
        batch, channels, h_patches * patch_size, w_patches * patch_size
    )


def build_hidream_o1_position_ids(
    *,
    text_len: int,
    h_patches: int,
    w_patches: int,
    device: torch.device,
) -> torch.Tensor:
    """Interleaved-mrope positions for [prompt tokens][image patch tokens].

    Closed form of the reference ``get_rope_index_fix_point`` for the single
    image / ``spatial_merge_size=1`` / ``skip_vision_start_token=[1]`` case:
    text tokens count up from 0 on all three axes and the patch grid restarts at
    the fix point as (t, row, col).
    """
    image_len = h_patches * w_patches
    position_ids = torch.empty(
        (3, 1, text_len + image_len), dtype=torch.long, device=device
    )
    position_ids[:, 0, :text_len] = torch.arange(text_len, device=device)
    rows = torch.arange(h_patches, device=device).repeat_interleave(w_patches)
    cols = torch.arange(w_patches, device=device).repeat(h_patches)
    position_ids[0, 0, text_len:] = HIDREAM_O1_MROPE_FIX_POINT
    position_ids[1, 0, text_len:] = HIDREAM_O1_MROPE_FIX_POINT + rows
    position_ids[2, 0, text_len:] = HIDREAM_O1_MROPE_FIX_POINT + cols
    return position_ids


def build_hidream_o1_attention_mask(
    *,
    text_len: int,
    image_len: int,
    dtype: torch.dtype,
    device: torch.device,
) -> torch.Tensor:
    """Causal prompt rows plus bidirectional timestep/patch rows, as [1, 1, S, S].

    The timestep token is the last prompt token, and it shares the patch span's
    full attention so it can read the whole image.
    """
    seq_len = text_len + image_len
    mask = torch.triu(
        torch.full(
            (1, seq_len, seq_len), torch.finfo(dtype).min, dtype=dtype, device=device
        ),
        diagonal=1,
    )
    mask[:, text_len - 1 :, :] = 0
    return mask.unsqueeze(1)


def resize_pilimage(pil_image, image_size, patch_size=16, resampler=None):
    """Aspect-preserving resize + center crop (official utils.py)."""
    if resampler is None:
        resampler = Image.BICUBIC
    while min(*pil_image.size) >= 2 * image_size:
        pil_image = pil_image.resize(
            tuple(x // 2 for x in pil_image.size), resample=Image.BOX
        )

    m = patch_size
    width, height = pil_image.width, pil_image.height
    s_max = image_size * image_size
    scale = math.sqrt(s_max / (width * height))

    new_sizes = [
        (round(width * scale) // m * m, round(height * scale) // m * m),
        (round(width * scale) // m * m, math.floor(height * scale) // m * m),
        (math.floor(width * scale) // m * m, round(height * scale) // m * m),
        (math.floor(width * scale) // m * m, math.floor(height * scale) // m * m),
    ]
    new_sizes = sorted(new_sizes, key=lambda x: x[0] * x[1], reverse=True)

    new_size = new_sizes[-1]
    for cand in new_sizes:
        if cand[0] * cand[1] <= s_max:
            new_size = cand
            break

    s1 = width / new_size[0]
    s2 = height / new_size[1]
    if s1 < s2:
        pil_image = pil_image.resize(
            [new_size[0], round(height / s1)], resample=resampler
        )
        top = (round(height / s1) - new_size[1]) // 2
        pil_image = pil_image.crop((0, top, new_size[0], top + new_size[1]))
    else:
        pil_image = pil_image.resize(
            [round(width / s2), new_size[1]], resample=resampler
        )
        left = (round(width / s2) - new_size[0]) // 2
        pil_image = pil_image.crop((left, 0, left + new_size[0], new_size[1]))
    return pil_image


def calculate_dimensions(max_size, ratio):
    """Largest patch-aligned (w, h) with aspect `ratio` fitting max_size^2."""
    width = math.sqrt(max_size * max_size * ratio)
    height = width / ratio
    width = int(width / 32) * 32
    height = int(height / 32) * 32
    return width, height


def _patchify_image_tensor(x, patch_size):
    """(C, H, W) -> (H*W, C*patch_size^2) in the model's patch layout."""
    c, h, w = x.shape
    h_patches, w_patches = h // patch_size, w // patch_size
    x = x.reshape(c, h_patches, patch_size, w_patches, patch_size)
    x = (
        x.permute(1, 3, 0, 2, 4)
        .contiguous()
        .reshape(h_patches * w_patches, c * patch_size * patch_size)
    )
    return x


def _ref_tensor_transform():
    """PIL -> float32 tensor in [-1, 1] (official TENSOR_TRANSFORM)."""
    return transforms.Compose(
        [
            transforms.ToImage(),
            transforms.ToDtype(torch.float32, scale=True),
            transforms.Normalize([0.5], [0.5]),
        ]
    )


def build_ti2i_text_sample(
    prompt,
    ref_pils_vlm,
    ref_pils_resized,
    height,
    width,
    ref_image_lens,
    tokenizer,
    processor,
    model_config,
):
    """Build the TI2I sample (reference-image conditioning, official I2I branch)."""
    image_token_id = model_config.image_token_id
    video_token_id = model_config.video_token_id
    vision_start_token_id = model_config.vision_start_token_id
    spatial_merge_size = model_config.vision_spatial_merge_size
    K = len(ref_pils_vlm)
    tgt_image_len = (height // PATCH_SIZE) * (width // PATCH_SIZE)

    content = [{"type": "image"} for _ in range(K)]
    content.append({"type": "text", "text": prompt})
    messages = [{"role": "user", "content": content}]
    template_caption = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    proc = processor(
        text=[template_caption],
        images=ref_pils_vlm,
        padding="longest",
        return_tensors="pt",
    )
    boi_token = getattr(tokenizer, "boi_token", "<|boi_token|>")
    tms_token = getattr(tokenizer, "tms_token", "<|tms_token|>")
    input_ids_2 = tokenizer.encode(
        boi_token + tms_token * TIMESTEP_TOKEN_NUM,
        return_tensors="pt",
        add_special_tokens=False,
    )
    input_ids = torch.cat([proc.input_ids, input_ids_2], dim=-1)

    igthw_cond = proc.image_grid_thw.clone()
    for i in range(K):
        igthw_cond[i, 1] //= spatial_merge_size
        igthw_cond[i, 2] //= spatial_merge_size
    image_grid_thw_tgt = torch.tensor(
        [[1, height // PATCH_SIZE, width // PATCH_SIZE]], dtype=torch.int64
    )
    image_grid_thw_ref = torch.tensor(
        [[1, p.height // PATCH_SIZE, p.width // PATCH_SIZE] for p in ref_pils_resized],
        dtype=torch.int64,
    )
    igthw_all = torch.cat([igthw_cond, image_grid_thw_tgt, image_grid_thw_ref], dim=0)

    vision_tokens_list = []
    vt_tgt = torch.full((1, tgt_image_len), image_token_id, dtype=input_ids.dtype)
    vt_tgt[0, 0] = vision_start_token_id
    vision_tokens_list.append(vt_tgt)
    for rl in ref_image_lens:
        vt_ref = torch.full((1, rl), image_token_id, dtype=input_ids.dtype)
        vt_ref[0, 0] = vision_start_token_id
        vision_tokens_list.append(vt_ref)
    vision_tokens = torch.cat(vision_tokens_list, dim=1)
    input_ids_pad = torch.cat([input_ids, vision_tokens], dim=-1)

    position_ids, _ = get_rope_index_fix_point(
        1,
        image_token_id,
        video_token_id,
        vision_start_token_id,
        input_ids=input_ids_pad,
        image_grid_thw=igthw_all,
        video_grid_thw=None,
        attention_mask=None,
        skip_vision_start_token=[0] * K + [1] + [1] * K,
    )
    txt_seq_len = input_ids.shape[-1]
    all_seq_len = position_ids.shape[-1]
    total_ref_len = sum(ref_image_lens)

    token_types_raw = torch.zeros((1, all_seq_len), dtype=input_ids.dtype)
    bgn = txt_seq_len - TIMESTEP_TOKEN_NUM
    end = bgn + tgt_image_len + TIMESTEP_TOKEN_NUM
    token_types_raw[0, bgn:end] = 1
    token_types_raw[0, end : end + total_ref_len] = 2
    token_types_raw[0, txt_seq_len - TIMESTEP_TOKEN_NUM : txt_seq_len] = 3

    vinput_mask = torch.logical_or(token_types_raw == 1, token_types_raw == 2)
    token_types_bin = (token_types_raw > 0).to(token_types_raw.dtype)

    return {
        "input_ids": input_ids,
        "position_ids": position_ids,
        "token_types": token_types_bin,
        "vinput_mask": vinput_mask,
        "pixel_values": proc.pixel_values,
        "image_grid_thw": proc.image_grid_thw,
    }


def _generation_work(batch: Req, prompts: list) -> list[dict]:
    """Prompt-major work items (x num_outputs); row order maps back to requests."""
    n = max(1, int(batch.num_outputs_per_prompt or 1))
    seeds = batch.seeds
    if seeds is None:
        seeds = batch.extra.get("dynamic_batch_seeds") or batch.seed
    if not isinstance(seeds, list):
        seeds = [seeds]
    seeds = [s[0] if isinstance(s, list) else int(s) for s in seeds]

    p_count = len(prompts)
    if len(seeds) == p_count:
        # one base seed per prompt: expand consecutive outputs
        seeds = [base + j for base in seeds for j in range(n)]
    elif len(seeds) == 1:
        seeds = [seeds[0] + i for i in range(p_count * n)]
    elif len(seeds) != p_count * n:
        raise ValueError(
            f"HiDream requires one seed per output; got {len(seeds)} "
            f"for {p_count} prompts x {n} outputs"
        )

    work = []
    for p_idx, prompt in enumerate(prompts):
        for o_idx in range(n):
            work.append(
                {
                    "prompt": prompt,
                    "seed": seeds[p_idx * n + o_idx],
                    "prompt_index": p_idx,
                    "output_index": o_idx,
                }
            )
    return work


def _stack_cond_samples(samples: list[dict], pad_token_id: int) -> dict:
    """Stack text samples into one batch; unequal lengths are padded to
    [text_wo_tms | pads | tms | vision] so the gen region aligns across rows."""
    txt_lens = [s["input_ids"].shape[-1] for s in samples]
    if all(t == txt_lens[0] for t in txt_lens):
        return _carry_pixel_fields(
            {
                "input_ids": torch.cat([s["input_ids"] for s in samples], dim=0),
                "position_ids": torch.cat([s["position_ids"] for s in samples], dim=1),
                "token_types": torch.cat([s["token_types"] for s in samples], dim=0),
                "pad_mask": None,
                "txt_len": txt_lens[0],
            },
            samples,
        )

    txt_max = max(txt_lens)
    rows_ids, rows_pos, rows_types, rows_pad = [], [], [], []
    for s in samples:
        ids = s["input_ids"]  # [1, txt_i]
        pos = s["position_ids"]  # [3, 1, txt_i + vision_i]
        types = s["token_types"]  # [1, txt_i + vision_i]
        txt_i = ids.shape[-1]
        vision = pos.shape[-1] - txt_i
        pad_n = txt_max - txt_i

        pad_row = torch.zeros(1, txt_max + vision, dtype=torch.bool)
        pad_row[:, txt_i - 1 : txt_i - 1 + pad_n] = True

        # Text region laid out [text_wo_tms | pads | tms] so the gen region
        # starts at txt_max - 1 for every row.
        ids_row = torch.cat(
            [ids[:, :-1], ids.new_full((1, pad_n), pad_token_id), ids[:, -1:]],
            dim=1,
        )
        types_row = torch.cat(
            [types[:, : txt_i - 1], types.new_zeros(1, pad_n), types[:, txt_i - 1 :]],
            dim=1,
        )
        pos_row = torch.cat(
            [
                pos[:, :, : txt_i - 1],
                pos.new_zeros(3, 1, pad_n),
                pos[:, :, txt_i - 1 :],
            ],
            dim=2,
        )
        rows_ids.append(ids_row)
        rows_pos.append(pos_row)
        rows_types.append(types_row)
        rows_pad.append(pad_row)

    return _carry_pixel_fields(
        {
            "input_ids": torch.cat(rows_ids, dim=0),
            "position_ids": torch.cat(rows_pos, dim=1),
            "token_types": torch.cat(rows_types, dim=0),
            "pad_mask": torch.cat(rows_pad, dim=0),
            "txt_len": txt_max,
        },
        samples,
    )


def _carry_pixel_fields(stacked: dict, samples: list[dict]) -> dict:
    """Attach batched vision inputs (TI2I) to a stacked sample."""
    if samples and samples[0].get("pixel_values") is not None:
        stacked["pixel_values"] = torch.cat([s["pixel_values"] for s in samples], dim=0)
        stacked["image_grid_thw"] = torch.cat(
            [s["image_grid_thw"] for s in samples], dim=0
        )
    return stacked


def _chunk_items(items: list[dict], cap: int) -> list[list[dict]]:
    """Group items into <=cap-row chunks by resolution + reference signature."""
    groups: dict[tuple, list[dict]] = {}
    for item in items:
        groups.setdefault(item["group_key"], []).append(item)
    chunks = []
    for group in groups.values():
        for start in range(0, len(group), max(1, cap)):
            chunks.append(group[start : start + max(1, cap)])
    return chunks


def _max_forward_rows(server_args: ServerArgs) -> int:
    return int(getattr(server_args, "batching_max_size", 1) or 1)


def _resolve_single_prompt(prompt, *, field_name: str) -> str:
    if isinstance(prompt, str):
        return prompt
    if isinstance(prompt, list) and len(prompt) == 1 and isinstance(prompt[0], str):
        return prompt[0]
    raise ValueError(
        f"HiDream-O1-Image serves one prompt per request; {field_name} must be a "
        f"single string, got {prompt!r}"
    )


def _per_row_cpu_generators(seeds: list[int]) -> list[torch.Generator]:
    """One CPU generator per row, seeded with seed + 1.

    CPU generators keep the noise bit-identical to the reference implementation
    regardless of the accelerator in use; one generator per row keeps every
    output on its own RNG stream (bit-identical to serial mode).
    """
    gens = []
    for seed in seeds:
        gen = torch.Generator("cpu")
        gen.manual_seed(int(seed) + 1)
        gens.append(gen)
    return gens


class HiDreamO1ImageBeforeDenoisingStage(PipelineStage):
    def __init__(
        self,
        tokenizer,
        processor,
        *,
        vision_start_token_id: int,
        tms_token_id: int,
    ) -> None:
        super().__init__()
        self.tokenizer = tokenizer
        self.processor = processor
        self.vision_start_token_id = vision_start_token_id
        self.tms_token_id = tms_token_id

    @property
    def role_affinity(self) -> RoleType:
        return RoleType.DENOISER

    def _enhance_prompts(self, batch: Req, server_args: ServerArgs) -> list[str]:
        prompts = batch.prompt if isinstance(batch.prompt, list) else [batch.prompt]

        # The enhancer samples at temperature 1.0, so running it independently on
        # every TP rank yields different prompts and, in turn, different token
        # lengths. That desyncs the vocab-parallel embedding's all_reduce
        # (HcclAllReduce EI0005 "parameter count" mismatch). Enhance once on the
        # TP rank-0 worker and broadcast the result so all ranks tokenize the
        # identical prompt.
        tp_group = get_tp_group() if model_parallel_is_initialized() else None
        if tp_group is None or tp_group.world_size <= 1:
            return [
                prepare_prompt(p, pe_server_url=server_args.pe_server_url)
                for p in prompts
            ]

        if tp_group.rank_in_group == 0:
            enhanced = [
                prepare_prompt(p, pe_server_url=server_args.pe_server_url)
                for p in prompts
            ]
        else:
            enhanced = [None] * len(prompts)
        tp_group.broadcast_object_list(enhanced, src=0)
        return enhanced

    def _build_ctx(self, batch: Req, server_args: ServerArgs) -> dict:
        pad_token_id = (
            getattr(self.tokenizer, "pad_token_id", None)
            or getattr(self.tokenizer, "eos_token_id", None)
            or 0
        )
        return {
            "num_steps": batch.sampling_params.num_inference_steps,
            "model_type": _resolve_variant(server_args.model_path),
            "guidance_scale": float(batch.guidance_scale),
            "noise_scale_start": float(batch.noise_scale_start),
            "noise_scale_end": float(batch.noise_scale_end),
            "noise_clip_std": float(batch.noise_clip_std),
            "shift": float(batch.shift),
            "explicit_fields": set(
                getattr(batch.sampling_params, "_explicit_fields", ()) or ()
            ),
            "model_config": server_args.pipeline_config.dit_config.arch_config,
            "pad_token_id": int(pad_token_id),
            "max_forward_rows": _max_forward_rows(server_args),
        }

    def _prepare_item(
        self, item: dict, batch: Req, ctx: dict, ref_images_batch, device, dtype
    ) -> dict:
        """Per-image preparation: resolution snap, references, cond sample."""
        prompt = item["prompt"]
        height = int(batch.height)
        width = int(batch.width)

        ref_paths = None
        if ref_images_batch:
            ref_paths = (
                [ref_images_batch]
                if isinstance(ref_images_batch, str)
                else [p for p in ref_images_batch if p]
            )
            if not ref_paths:
                ref_paths = None

        # TI2I (single reference) keeps the reference aspect by default:
        # the target size derives from the reference resized to 2048
        # (patch-aligned). An explicit width AND height in the request
        # wins (snapped); T2I always snaps the requested size.
        preresized_ref_pil = None
        explicit_fields = ctx["explicit_fields"]
        explicit_size = "width" in explicit_fields and "height" in explicit_fields
        if ref_paths and len(ref_paths) == 1 and not explicit_size:
            preresized_ref_pil = resize_pilimage(
                Image.open(ref_paths[0]).convert("RGB"), 2048, PATCH_SIZE
            )
            width, height = preresized_ref_pil.size
            logger.info("TI2I: target size %dx%d from reference aspect", width, height)
        else:
            best_w, best_h = find_closest_resolution(width, height)
            if best_w != width or best_h != height:
                logger.info(
                    "Resolution snapped from %dx%d to %dx%d",
                    width,
                    height,
                    best_w,
                    best_h,
                )
                width, height = best_w, best_h

        h_patches = height // PATCH_SIZE
        w_patches = width // PATCH_SIZE

        ref_patches = None
        ref_image_lens = None
        ref_pils_vlm = None
        ref_pils_resized = None
        if ref_paths:
            if preresized_ref_pil is not None:
                ref_pils = [preresized_ref_pil]
            else:
                ref_pils = [Image.open(p).convert("RGB") for p in ref_paths]
            K = len(ref_pils)

            if K == 1:
                max_size = max(height, width)
            elif K == 2:
                max_size = max(height, width) * 48 // 64
            elif K <= 4:
                max_size = max(height, width) // 2
            elif K <= 8:
                max_size = max(height, width) * 24 // 64
            else:
                max_size = max(height, width) // 4

            ref_pils_resized = []
            ref_images = []
            transform = _ref_tensor_transform()
            for pil in ref_pils:
                # already patch-aligned via the TI2I reference resize
                if preresized_ref_pil is not None and pil is preresized_ref_pil:
                    pil_r = pil
                else:
                    pil_r = resize_pilimage(pil, max_size, PATCH_SIZE)
                ref_pils_resized.append(pil_r)
                ref_images.append(_patchify_image_tensor(transform(pil_r), PATCH_SIZE))
            ref_image_lens = [x.shape[0] for x in ref_images]
            ref_patches = torch.cat(ref_images, dim=0).unsqueeze(0).to(device, dtype)

            if K <= 4:
                cond_img_size = CONDITION_IMAGE_SIZE
            elif K <= 8:
                cond_img_size = CONDITION_IMAGE_SIZE * 48 // 64
            else:
                cond_img_size = CONDITION_IMAGE_SIZE // 2
            ref_pils_vlm = []
            for p in ref_pils_resized:
                cond_w, cond_h = calculate_dimensions(cond_img_size, p.width / p.height)
                ref_pils_vlm.append(p.resize((cond_w, cond_h), Image.LANCZOS))
            logger.info(
                "TI2I: %d reference image(s), %d conditioning patches",
                K,
                sum(ref_image_lens),
            )

        if ref_paths:
            cond_sample = build_ti2i_text_sample(
                prompt,
                ref_pils_vlm,
                ref_pils_resized,
                height,
                width,
                ref_image_lens,
                self.tokenizer,
                self.processor,
                ctx["model_config"],
            )
            uncond_sample = build_ti2i_text_sample(
                " ",
                ref_pils_vlm,
                ref_pils_resized,
                height,
                width,
                ref_image_lens,
                self.tokenizer,
                self.processor,
                ctx["model_config"],
            )

        else:
            cond_sample = build_t2i_text_sample(
                prompt,
                height,
                width,
                self.tokenizer,
                self.processor,
                ctx["model_config"],
            )
            uncond_sample = build_t2i_text_sample(
                " ",
                height,
                width,
                self.tokenizer,
                self.processor,
                ctx["model_config"],
            )

        item.update(
            {
                "height": height,
                "width": width,
                "h_patches": h_patches,
                "w_patches": w_patches,
                "ref_paths": ref_paths,
                "ref_patches": ref_patches,
                "ref_image_lens": ref_image_lens,
                "ref_pils_vlm": ref_pils_vlm,
                "ref_pils_resized": ref_pils_resized,
                "cond_sample": cond_sample,
                "uncond_sample": uncond_sample,
                "group_key": (
                    height,
                    width,
                    tuple(ref_image_lens) if ref_image_lens else (),
                    tuple(ref_paths or ()),
                ),
            }
        )
        return item

    def _encode_prompt(self, prompt: str, *, device: torch.device) -> torch.Tensor:
        messages = [{"role": "user", "content": prompt}]
        template_caption = (
            self.processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            + HIDREAM_O1_BOI_TOKEN
            + HIDREAM_O1_TMS_TOKEN
        )
        input_ids = self.tokenizer.encode(
            template_caption, return_tensors="pt", add_special_tokens=False
        )
        if bool((input_ids == self.vision_start_token_id).any()):
            raise ValueError(
                "The prompt encodes the vision-start token, which would shift the "
                "image mrope grid; remove it from the prompt."
            )
        # Fast tokenizers split on added tokens even with add_special_tokens=False,
        # so a spelled-out token in the prompt lands here as a real id and would
        # silently take a copy of the timestep embedding.
        tms_count = int((input_ids == self.tms_token_id).sum())
        if tms_count != 1:
            raise ValueError(
                f"The prompt must yield exactly one {HIDREAM_O1_TMS_TOKEN} slot for "
                f"the timestep embedding, got {tms_count}; remove the token from "
                "the prompt."
            )
        return input_ids.to(device)

    def _build_conditioning(
        self,
        prompt: str,
        *,
        batch_size: int,
        h_patches: int,
        w_patches: int,
        mask_dtype: torch.dtype,
        device: torch.device,
    ) -> dict[str, torch.Tensor]:
        input_ids = self._encode_prompt(prompt, device=device)
        text_len = input_ids.shape[-1]
        image_len = h_patches * w_patches
        return {
            # A stride-0 expand is safe here: the DiT only ever reads input_ids
            # (embedding lookup and the timestep-slot comparison), never writes.
            "input_ids": input_ids.expand(batch_size, -1),
            "position_ids": build_hidream_o1_position_ids(
                text_len=text_len,
                h_patches=h_patches,
                w_patches=w_patches,
                device=device,
            ),
            "attention_mask": build_hidream_o1_attention_mask(
                text_len=text_len,
                image_len=image_len,
                dtype=mask_dtype,
                device=device,
            ),
        }

    def _prepare_latents(
        self,
        *,
        seeds: list[int],
        height: int,
        width: int,
        patch_size: int,
        channels: int,
        dtype: torch.dtype,
        device: torch.device,
        noise_scale: float | None,
    ) -> torch.Tensor:
        # CPU generators seeded with seed + 1 keep the noise bit-identical to the
        # reference implementation regardless of the accelerator in use.
        noise = torch.cat(
            [
                noise_scale
                * torch.randn(
                    (1, channels, height, width),
                    generator=torch.Generator("cpu").manual_seed(int(seed) + 1),
                )
                for seed in seeds
            ]
        )
        noise = noise.to(device=device, dtype=dtype)
        return patchify_pixels(noise, patch_size)

    @torch.no_grad()
    def forward(self, batch: Req, server_args: ServerArgs) -> Req:
        device = self.device
        dit_dtype = resolve_precision(
            server_args, "dit", precision_attr="dit_precision"
        )
        ctx = self._build_ctx(batch, server_args)

        prompts = self._enhance_prompts(batch, server_args)
        ref_images_batch = getattr(batch, "image_path", None)

        work = _generation_work(batch, prompts)
        items = [
            self._prepare_item(item, batch, ctx, ref_images_batch, device, dit_dtype)
            for item in work
        ]
        for i, item in enumerate(items):
            item["index"] = i

        # Editing detection drives the scheduler override; references are
        # identical across a request's rows.
        ctx["is_editing"] = (
            items[0]["ref_paths"] is not None and len(items[0]["ref_paths"]) == 1
        )
        editing = ctx["is_editing"]

        # Determine the scheduler *name* only. The denoising stage builds the
        # scheduler itself per request, so no scheduler object (or step counter)
        # is ever shared between requests.
        ctx["scheduler_name"] = resolve_scheduler_name(
            editing=editing,
            model_type=ctx["model_type"],
        )

        # The framework denoising loop (NVTX, CFG gating, cfg-parallel, BCG,
        # trajectory) serves the full model's plain T2I path: single prompt,
        # UniPC recipe, no cache-dit/torch-compile. Everything else — Dev
        # recipes, TI2I, cache-dit, merged multi-prompt batches — runs on the
        # hand-rolled trajectory below.
        single_prompt = not isinstance(batch.prompt, list) or len(batch.prompt) == 1
        uni_pc = ctx["model_type"] == "full" and ctx["scheduler_name"] in (
            "default",
            "unipc",
            "flow_unipc",
        )
        # cache-dit runs on the hand-rolled flash trajectory, never the framework
        # dense path, so a cache-dit request must stay off ``framework_path``.
        cache_dit_requested = (
            envs.SGLANG_CACHE_DIT_ENABLED
            if batch.sampling_params.enable_cache_dit is None
            else batch.sampling_params.enable_cache_dit
        )
        framework_path = (
            single_prompt and not editing and uni_pc and not cache_dit_requested
        )
        ctx["framework_path"] = framework_path
        ctx["cache_dit_requested"] = cache_dit_requested
        ctx["device"] = str(device)

        first = items[0]
        height, width = first["height"], first["width"]
        h_patches, w_patches = first["h_patches"], first["w_patches"]

        latents = self._prepare_latents(
            seeds=[it["seed"] for it in items],
            height=height,
            width=width,
            patch_size=PATCH_SIZE,
            channels=3,
            dtype=dit_dtype,
            device=device,
            noise_scale=ctx["noise_scale_start"],
        )

        if framework_path:
            # Main's contract: conditioning stashed for the framework CFG
            # branches; the backbone embeds the prompt itself, so
            # prompt_embeds only satisfy the denoising stage's contract.
            cond = self._build_conditioning(
                _resolve_single_prompt(prompts, field_name="prompt"),
                batch_size=len(items),
                h_patches=h_patches,
                w_patches=w_patches,
                mask_dtype=dit_dtype,
                device=device,
            )
            neg_prompt = _resolve_single_prompt(
                batch.negative_prompt, field_name="negative_prompt"
            )
            neg_cond = self._build_conditioning(
                neg_prompt,
                batch_size=len(items),
                h_patches=h_patches,
                w_patches=w_patches,
                mask_dtype=dit_dtype,
                device=device,
            )
            batch.extra[HIDREAM_O1_COND_KEY] = cond
            batch.extra[HIDREAM_O1_NEG_COND_KEY] = neg_cond
            batch.prompt_embeds = [cond["input_ids"]]
            batch.negative_prompt_embeds = [neg_cond["input_ids"]]

        batch.extra["hidream_o1_ctx"] = ctx
        batch.extra["hidream_o1_items"] = items
        batch.latents = latents
        batch.raw_latent_shape = latents.shape
        batch.height = height
        batch.width = width
        return batch

    def verify_output(self, batch: Req, server_args: ServerArgs):
        result = VerificationResult()
        result.add_check("latents", batch.latents, [V.is_tensor, V.with_dims(3)])
        return result


def _register_hidream_o1_cache_dit_spec() -> None:
    from cache_dit import ForwardPattern

    from sglang.multimodal_gen.runtime.cache.cache_dit_integration import (
        _CUSTOM_BLOCK_ADAPTER_SPECS,
        CustomBlockAdapterSpec,
    )

    _CUSTOM_BLOCK_ADAPTER_SPECS.setdefault(
        HiO1CacheBlockAdapter.__name__,
        CustomBlockAdapterSpec(
            blocks_attr="blocks", forward_pattern=ForwardPattern.Pattern_3
        ),
    )


class HiDreamO1ImageDenoisingStage(DenoisingStage):
    """Denoising for a model that predicts the clean image, not the noise.

    Full model (UniPC, plain T2I): the framework loop runs unchanged and this
    stage only overrides ``_run_denoising_step`` — classifier-free guidance is
    applied to x_pred and converted to the velocity the scheduler expects
    (exact because the conversion is affine in x_pred; the before-denoising
    stage refuses the non-linear CFG knobs that would break the reordering).

    Dev checkpoints and TI2I: the hand-rolled trajectory below — per-row
    private CPU generators, per-step noise re-injection, deterministic TI2I
    reroll and cache-dit — none of which the shared loop can express.
    """

    def __init__(self, model: torch.nn.Module) -> None:
        # Route the model through the base init so the framework's compile/
        # offload and attention-backend wiring see the transformer.
        #
        # No scheduler is held here: the before-denoising stage only resolves
        # the scheduler *name* into ``ctx["scheduler_name"]``, and this stage
        # builds a fresh scheduler per request in ``forward`` (the Dev editing
        # recipe swaps flash for flow_match, and the step counter must not be
        # shared between requests). Text samples (cond + uncond) are built in
        # the before-denoising stage, which owns the tokenizer/processor.
        super().__init__(transformer=model, scheduler=None)
        self.model = model
        self._cache_dit_adapter: HiO1CacheBlockAdapter | None = None
        self._cache_dit_num_steps: int | None = None

    @property
    def role_affinity(self) -> RoleType:
        return RoleType.DENOISER

    def verify_input(self, batch: Req, server_args: ServerArgs) -> VerificationResult:
        # ``batch.timesteps`` is created per request inside ``forward`` (the
        # scheduler is built there), so it cannot be validated before the stage
        # runs; the latents shape is all this stage needs at its boundary.
        result = VerificationResult()
        result.add_check("latents", batch.latents, [V.is_tensor, V.with_dims(3)])
        return result

    def _build_cache_dit_config(self, num_inference_steps: int) -> CacheDitConfig:
        return CacheDitConfig(
            enabled=True,
            Fn_compute_blocks=envs.SGLANG_CACHE_DIT_FN,
            Bn_compute_blocks=envs.SGLANG_CACHE_DIT_BN,
            max_warmup_steps=envs.SGLANG_CACHE_DIT_WARMUP,
            residual_diff_threshold=envs.SGLANG_CACHE_DIT_RDT,
            max_continuous_cached_steps=envs.SGLANG_CACHE_DIT_MC,
            enable_taylorseer=envs.SGLANG_CACHE_DIT_TAYLORSEER,
            taylorseer_order=envs.SGLANG_CACHE_DIT_TS_ORDER,
            num_inference_steps=num_inference_steps,
        )

    def _maybe_enable_cache_dit(
        self, num_inference_steps: int, batch: Req | None = None
    ) -> None:
        """Mount cache-dit on the flash decoder loop (env/per-request gated, idempotent).

        The O1 backbone is not a diffusers DiT, so it is exposed via
        HiO1CacheBlockAdapter (see that adapter's docstring for the naming
        constraint); SGLANG_CACHE_DIT_* then apply as for GLM-Image/Wan.
        """
        if not envs.SGLANG_CACHE_DIT_ENABLED:
            return
        if self._cache_dit_adapter is None:
            _register_hidream_o1_cache_dit_spec()
            self._cache_dit_adapter = HiO1CacheBlockAdapter(self.model.model)
            tp_group = None
            if model_parallel_is_initialized():
                group = get_tp_group()
                tp_group = group.device_group if group.world_size > 1 else None
            enable_cache_on_transformer(
                self._cache_dit_adapter,
                self._build_cache_dit_config(num_inference_steps),
                model_name="hidream_o1_image",
                tp_group=tp_group,
                has_separate_cfg=True,
            )
            self.log_info(
                "cache-dit enabled on HiDream-O1-Image (Fn=%d Bn=%d W=%d R=%.2f "
                "MC=%d TaylorSeer=%s)",
                envs.SGLANG_CACHE_DIT_FN,
                envs.SGLANG_CACHE_DIT_BN,
                envs.SGLANG_CACHE_DIT_WARMUP,
                envs.SGLANG_CACHE_DIT_RDT,
                envs.SGLANG_CACHE_DIT_MC,
                envs.SGLANG_CACHE_DIT_TAYLORSEER,
            )
        # Refresh every request: a new generation resets cache-dit's step counter.
        refresh_context_on_transformer(self._cache_dit_adapter, num_inference_steps)
        self._cache_dit_num_steps = num_inference_steps

    @torch.no_grad()
    def forward(self, batch: Req, server_args: ServerArgs) -> Req:
        ctx = batch.extra["hidream_o1_ctx"]
        # Build the request's scheduler from the recipe the before-denoising
        # stage resolved, and publish it for the framework loop
        # (``batch.scheduler`` / ``batch.timesteps``) and the hand-rolled
        # trajectory (``ctx["scheduler"]``).
        scheduler = build_scheduler(
            num_inference_steps=ctx["num_steps"],
            timesteps_list=(
                DEFAULT_TIMESTEPS
                if ctx["scheduler_name"] in ("flash", "flow_match")
                else None
            ),
            shift=ctx["shift"],
            device=self.device,
            scheduler_name=ctx["scheduler_name"],
        )
        ctx["scheduler"] = scheduler
        batch.scheduler = scheduler
        batch.timesteps = scheduler.timesteps
        batch.sigmas = scheduler.sigmas.tolist()
        if ctx.get("framework_path"):
            return super().forward(batch, server_args)
        if ctx.get("cache_dit_requested"):
            self._maybe_enable_cache_dit(ctx["num_steps"])
        items = batch.extra["hidream_o1_items"]
        chunks = _chunk_items(items, ctx["max_forward_rows"])
        z_rows: list = [None] * len(items)
        for chunk in chunks:
            for item, z in zip(chunk, self._run_trajectory(chunk, ctx, batch)):
                z_rows[item["index"]] = z
        batch.latents = torch.cat(z_rows, dim=0)
        return batch

    @torch.no_grad()
    def run_grouped_requests(
        self, batches: list[Req], server_args: ServerArgs
    ) -> list[Req]:
        """Fuse expanded outputs into one hand-rolled denoising forward."""
        if len(batches) == 1 or any(
            batch.extra["hidream_o1_ctx"].get("framework_path") for batch in batches
        ):
            return super().run_grouped_requests(batches, server_args)

        merged = copy(batches[0])
        merged.extra = dict(batches[0].extra)
        merged.extra["hidream_o1_ctx"] = dict(batches[0].extra["hidream_o1_ctx"])
        items = [
            dict(item) for batch in batches for item in batch.extra["hidream_o1_items"]
        ]
        for index, item in enumerate(items):
            item["index"] = index
        merged.extra["hidream_o1_items"] = items
        request_ids = [
            request_id
            for batch in batches
            if batch.report_progress
            for request_id in batch.client_request_ids()
        ]
        merged.merged_request_ids = request_ids or None
        merged.report_progress = bool(request_ids)

        merged = self.forward(merged, server_args)
        offset = 0
        for batch in batches:
            count = len(batch.extra["hidream_o1_items"])
            batch.latents = merged.latents[offset : offset + count]
            batch.raw_latent_shape = batch.latents.shape
            offset += count
        return batches

    def _run_denoising_step(
        self,
        ctx: DenoisingContext,
        step: DenoisingStepState,
        batch: Req,
        server_args: ServerArgs,
    ) -> None:
        use_nvtx = self.current_use_nvtx
        latents_dtype = ctx.latents.dtype
        # ``ctx.timesteps`` is int64, so re-deriving sigma as ``t / 1000`` would
        # truncate it; ``convert_model_output`` inverts this conversion with
        # ``scheduler.sigmas[step_index]``, and the two sigmas must be the same
        # value or the x_pred -> velocity -> x_pred round trip loses accuracy.
        sigma = ctx.scheduler.sigmas[step.step_index].to(
            device=ctx.latents.device, dtype=torch.float32
        )
        # The pixel head is conditioned on the flow position (0 -> 1), which runs
        # opposite to the scheduler's timestep (1000 -> 0).
        flow_position = (1.0 - sigma).reshape(-1).expand(ctx.latents.shape[0])

        with maybe_nvtx_range("predict_noise", use_nvtx):
            x_pred = self._predict_noise_with_cfg(
                current_model=step.current_model,
                latent_model_input=ctx.latents.to(ctx.target_dtype),
                timestep=flow_position,
                batch=batch,
                timestep_index=step.step_index,
                attn_metadata=step.attn_metadata,
                target_dtype=ctx.target_dtype,
                current_guidance_scale=step.current_guidance_scale,
                cfg_policy=ctx.cfg_policy,
                cfg_gate_state=ctx.extra.get("cfg_gate_state"),
                server_args=server_args,
                guidance=ctx.guidance,
                latents=ctx.latents,
            )
        with maybe_nvtx_range("scheduler_step", use_nvtx):
            sample = ctx.latents.float()
            # Exact inverse of the scheduler's own convert_model_output, which
            # recovers x_pred as ``sample - sigma * model_output``.
            model_output = (sample - x_pred.float()) / sigma.clamp_min(HIDREAM_O1_T_EPS)
            if server_args.comfyui_mode:
                batch.noise_pred = model_output
            ctx.latents = ctx.scheduler.step(
                model_output=model_output,
                timestep=step.t_device,
                sample=sample,
                **ctx.extra_step_kwargs,
                return_dict=False,
            )[0].to(latents_dtype)

    def _run_trajectory(
        self, chunk: list[dict], ctx: dict, batch: Req
    ) -> list[torch.Tensor]:
        """Run one batched denoising trajectory for same-shape work items."""
        device = self.device
        dit_dtype = (
            chunk[0]["ref_patches"].dtype
            if chunk[0]["ref_patches"] is not None
            else torch.bfloat16
        )
        first = chunk[0]
        height, width = first["height"], first["width"]
        h_patches, w_patches = first["h_patches"], first["w_patches"]
        tgt_image_len = h_patches * w_patches
        ref_paths = first["ref_paths"]
        ref_patches = None
        if first["ref_patches"] is not None:
            ref_patches = torch.cat([it["ref_patches"] for it in chunk], dim=0)

        cond = _stack_cond_samples(
            [it["cond_sample"] for it in chunk], ctx["pad_token_id"]
        )
        txt_len = cond["txt_len"]

        guidance_scale = ctx["guidance_scale"]
        uncond = None
        if guidance_scale > 1.0:
            uncond = self._build_uncond_sample(first, ctx, len(chunk))

        sched = ctx["scheduler"]
        # The flash scheduler is the one that injects per-step noise; key off
        # the scheduler in use so the name cannot desync the trajectory.
        inject_noise = isinstance(sched, FlashFlowMatchScheduler)
        num_steps = len(sched.timesteps)
        if inject_noise and num_steps > 1:
            noise_scale_schedule = [
                ctx["noise_scale_start"]
                + (ctx["noise_scale_end"] - ctx["noise_scale_start"])
                * i
                / (num_steps - 1)
                for i in range(num_steps)
            ]
        else:
            noise_scale_schedule = [ctx["noise_scale_start"]]

        init_noise_scale = ctx["noise_scale_start"]

        noise_note = (
            f" (per-step noise={ctx['noise_scale_start']}->"
            f"{ctx['noise_scale_end']} clip={ctx['noise_clip_std']})"
            if inject_noise
            else ""
        )
        logger.info(
            "HiDream %s/%s: cfg=%s shift=%s steps=%d z_init=%s "
            "inject=%s%s rows=%d seeds=%s",
            ctx["model_type"],
            ctx["scheduler_name"],
            guidance_scale,
            ctx["shift"],
            num_steps,
            init_noise_scale,
            "per-step" if inject_noise else "off",
            noise_note,
            len(chunk),
            [it["seed"] for it in chunk],
        )

        # one private CPU generator per row (seed + 1, main's construction):
        # draws stay per-sample so every output keeps its own RNG stream
        gens = _per_row_cpu_generators([it["seed"] for it in chunk])
        z_rows = []
        for gen in gens:
            noise = init_noise_scale * torch.randn(
                (1, 3, height, width), generator=gen, dtype=torch.float32
            )
            noise = noise.to(device=device, dtype=dit_dtype)
            patches = noise.reshape(1, 3, h_patches, PATCH_SIZE, w_patches, PATCH_SIZE)
            z_rows.append(
                patches.permute(0, 2, 4, 1, 3, 5)
                .contiguous()
                .reshape(1, h_patches * w_patches, -1)
            )
        z = torch.cat(z_rows, dim=0)

        # Deterministic TI2I (flow_match) re-injects no per-step noise, so
        # the trajectory is fixed by the initial draw; a bad draw must be
        # re-rolled.
        reroll_enabled = not inject_noise and ref_paths is not None

        precomputed = (None, None)
        for step_idx, step_t in enumerate(
            self.progress_bar(sched.timesteps, total=num_steps, batch=batch)
        ):
            t_pixeldit = 1.0 - step_t.float() / 1000.0
            sigma = (step_t.float() / 1000.0).clamp_min(HIDREAM_O1_T_EPS)

            vinputs = (
                torch.cat([z, ref_patches], dim=1)
                if ref_patches is not None
                else z.clone()
            )

            with torch.autocast(
                device_type=device.type,
                dtype=torch.bfloat16,
                cache_enabled=False,
            ):
                x_pred_cond, embeds = self._forward_batch(
                    cond,
                    vinputs,
                    t_pixeldit.to(device),
                    precomputed,
                    use_cache_dit=step_idx >= 1,
                )
            if embeds is not None and precomputed[0] is None:
                precomputed = embeds
            x_tgt = x_pred_cond[:, txt_len : txt_len + tgt_image_len]

            if reroll_enabled and step_idx == 0:
                z, x_tgt = self._reroll_rows(
                    chunk,
                    cond,
                    z,
                    x_tgt,
                    t_pixeldit.to(device),
                    precomputed,
                    ref_patches,
                    tgt_image_len,
                    init_noise_scale,
                    height,
                    width,
                    h_patches,
                    w_patches,
                )

            v_cond = (x_tgt.float() - z.float()) / sigma

            if uncond is not None:
                with torch.autocast(
                    device_type=device.type,
                    dtype=torch.bfloat16,
                    cache_enabled=False,
                ):
                    x_pred_uncond, _ = self._forward_batch(
                        uncond,
                        vinputs,
                        t_pixeldit.to(device),
                        precomputed,
                        use_cache_dit=step_idx >= 1,
                    )
                x_tgt_uncond = x_pred_uncond[
                    :, uncond["txt_len"] : uncond["txt_len"] + tgt_image_len
                ]
                v_uncond = (x_tgt_uncond.float() - z.float()) / sigma
                v_guided = v_uncond + guidance_scale * (v_cond - v_uncond)
            else:
                v_guided = v_cond

            model_output = -v_guided
            if inject_noise:
                noise = self._draw_step_noise(z, gens, ctx["noise_clip_std"])
                z = sched.step(
                    model_output.float(),
                    step_t.to(dtype=torch.float32),
                    z.float(),
                    s_noise=noise_scale_schedule[step_idx],
                    noise=noise,
                ).to(dit_dtype)
            elif isinstance(sched, FlowMatchEulerDiscreteScheduler):
                # The flow-match Euler scheduler is diffusers-style: with
                # ``return_dict=False`` it returns a ``(prev_sample,)`` tuple.
                step_out = sched.step(
                    model_output.float(),
                    step_t.to(dtype=torch.float32),
                    z.float(),
                    return_dict=False,
                )
                z = step_out[0].to(dit_dtype)
            else:
                # UniPC (full-model TI2I) is diffusers-style: it accepts
                # ``return_dict`` and returns a ``(prev_sample,)`` tuple.
                step_out = sched.step(
                    model_output.float(),
                    step_t.to(dtype=torch.float32),
                    z.float(),
                    return_dict=False,
                )
                z = (step_out[0] if isinstance(step_out, tuple) else step_out).to(
                    dit_dtype
                )

        return [z[i : i + 1] for i in range(len(chunk))]

    def _reroll_rows(
        self,
        chunk: list[dict],
        cond: dict,
        z: torch.Tensor,
        x_tgt: torch.Tensor,
        t_pixeldit: torch.Tensor,
        precomputed: tuple,
        ref_patches: torch.Tensor | None,
        tgt_image_len: int,
        init_noise_scale: float,
        height: int,
        width: int,
        h_patches: int,
        w_patches: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Per-row step-0 TI2I reroll; failing rows redraw independently."""
        threshold = REROLL_OUT_OF_RANGE_THRESHOLD
        frac = (x_tgt.float().abs() > 1.0).float().mean(dim=(1, 2))
        best_frac = frac.clone()
        best_z = z.clone()
        best_x = x_tgt.clone()
        best_seed = [int(it["seed"]) for it in chunk]
        active = [i for i in range(z.shape[0]) if frac[i] > threshold]
        attempt = 0
        while active and attempt < REROLL_MAX_ATTEMPTS:
            attempt += 1
            reroll_seeds = {}
            for i in active:
                reroll_seeds[i] = int(chunk[i]["seed"]) + attempt
                logger.warning(
                    "TI2I seed reroll: step-0 x_pred out-of-range fraction "
                    "%.3f > %.3f (row %d, seed %s); re-rolling to generator "
                    "seed %d (attempt %d/%d)",
                    best_frac[i],
                    threshold,
                    i,
                    chunk[i]["seed"],
                    reroll_seeds[i],
                    attempt,
                    REROLL_MAX_ATTEMPTS,
                )
                reroll_gen = torch.Generator("cpu")
                reroll_gen.manual_seed(reroll_seeds[i] + 1)
                noise = init_noise_scale * torch.randn(
                    (1, 3, height, width), generator=reroll_gen, dtype=torch.float32
                )
                noise = noise.to(device=z.device, dtype=z.dtype)
                patches = noise.reshape(
                    1, 3, h_patches, PATCH_SIZE, w_patches, PATCH_SIZE
                )
                z[i] = (
                    patches.permute(0, 2, 4, 1, 3, 5)
                    .contiguous()
                    .reshape(1, h_patches * w_patches, -1)[0]
                )
            vinputs = (
                torch.cat([z, ref_patches], dim=1)
                if ref_patches is not None
                else z.clone()
            )
            with torch.autocast(
                device_type=z.device.type,
                dtype=torch.bfloat16,
                cache_enabled=False,
            ):
                x_pred_sub, _ = self._forward_batch(
                    cond, vinputs, t_pixeldit, precomputed, use_cache_dit=False
                )
            x_tgt_all = x_pred_sub[:, cond["txt_len"] : cond["txt_len"] + tgt_image_len]
            still_active = []
            for i in active:
                f = (x_tgt_all[i : i + 1].float().abs() > 1.0).float().mean().item()
                if f < best_frac[i].item():
                    best_frac[i] = f
                    best_z[i] = z[i]
                    best_x[i] = x_tgt_all[i]
                    best_seed[i] = reroll_seeds[i]
                if best_frac[i].item() > threshold:
                    still_active.append(i)
            active = still_active
        rerolled = sum(
            1 for i in range(z.shape[0]) if best_frac[i].item() < frac[i].item()
        )
        if rerolled:
            logger.warning(
                "TI2I seed reroll: kept best draws for %d/%d row(s) "
                "(generator seeds %s)",
                rerolled,
                z.shape[0],
                best_seed,
            )
        return best_z, best_x

    def _draw_step_noise(
        self, z: torch.Tensor, gens: list, noise_clip_std: float
    ) -> torch.Tensor:
        """Per-row flash-step noise; preserves each row's CPU generator stream."""
        rows = []
        for gen in gens:
            n_i = torch.randn(
                (1, z.shape[1], z.shape[2]),
                dtype=torch.float32,
                generator=gen,
            ).to(z.device)
            if noise_clip_std > 0:
                clip_val = noise_clip_std * n_i.std().item()
                n_i = n_i.clamp(min=-clip_val, max=clip_val)
            rows.append(n_i)
        return torch.cat(rows, dim=0)

    def _build_uncond_sample(self, first: dict, ctx: dict, batch_rows: int) -> dict:
        """Expand the before-denoising stage's uncond sample to all chunk rows.

        The unconditional base sample (empty prompt) is built in the
        before-denoising stage, which owns the tokenizer; this stage only
        broadcasts it to the chunk's rows.
        """
        del ctx
        sample = first["uncond_sample"]
        b = batch_rows
        expanded = {
            "input_ids": sample["input_ids"].expand(b, -1),
            "position_ids": torch.cat([sample["position_ids"]] * b, dim=1),
            "token_types": sample["token_types"].expand(b, -1),
            "pad_mask": None,
            "txt_len": sample["input_ids"].shape[-1],
        }
        if sample.get("pixel_values") is not None:
            expanded["pixel_values"] = sample["pixel_values"]
            expanded["image_grid_thw"] = sample["image_grid_thw"]
        return expanded

    def _forward_batch(
        self,
        sample: dict,
        vinputs: torch.Tensor,
        t_pixeldit: torch.Tensor,
        precomputed: tuple,
        use_cache_dit: bool = True,
    ) -> tuple[torch.Tensor, tuple | None]:
        """Run one (possibly batched) two-pass forward; returns (x_pred, embeds)."""
        device = vinputs.device
        precomputed_image_embeds, precomputed_deepstack_image_embeds = precomputed
        pixel_values = sample.get("pixel_values")
        vision_kwargs = {}
        if pixel_values is not None:
            vision_kwargs["pixel_values"] = pixel_values.to(device)
            vision_kwargs["image_grid_thw"] = sample["image_grid_thw"].to(device)
            if precomputed_image_embeds is not None:
                vision_kwargs["precomputed_image_embeds"] = precomputed_image_embeds
                vision_kwargs["precomputed_deepstack_image_embeds"] = (
                    precomputed_deepstack_image_embeds
                )
        common_kwargs = {
            "input_ids": sample["input_ids"].to(device),
            "position_ids": sample["position_ids"].to(device),
            "vinputs": vinputs,
            "timestep": t_pixeldit.reshape(-1).to(device),
            "token_types": sample["token_types"].to(device),
        }

        pad_mask = sample.get("pad_mask")
        if pad_mask is not None:
            pad_mask = pad_mask.to(device)

        if (
            use_cache_dit
            and self._cache_dit_adapter is not None
            and not getattr(self.model, "_fsdp_wrapped", False)
        ):
            # cache-dit drives the flash decoder loop itself: embed once, run the
            # wrapped block chain, then project with the final RMSNorm applied.
            # TI2I reference features (vision embeds + per-layer deepstack) flow
            # through ``forward_pre`` and are re-injected per block, so the
            # reference image still conditions the output. ``pad_mask`` is
            # step-invariant, so it threads as a constant cache-dit condition.
            pad_mask = sample.get("pad_mask")
            if pad_mask is not None:
                pad_mask = pad_mask.to(device)
            inputs_embeds, conds, embeds_pair = self.model.forward_pre(
                input_ids=common_kwargs["input_ids"],
                position_ids=common_kwargs["position_ids"],
                vinputs=common_kwargs["vinputs"],
                timestep=common_kwargs["timestep"],
                token_types=common_kwargs["token_types"],
                pixel_values=vision_kwargs.get("pixel_values"),
                image_grid_thw=vision_kwargs.get("image_grid_thw"),
                precomputed_image_embeds=precomputed_image_embeds,
                precomputed_deepstack_image_embeds=precomputed_deepstack_image_embeds,
            )
            hidden_states = self._cache_dit_adapter(
                inputs_embeds,
                conds["cos"],
                conds["sin"],
                conds["idx_ar"],
                conds["visual_pos_masks"],
                conds["deepstack_visual_embeds"],
                pad_mask,
            )
            embeds = embeds_pair if pixel_values is not None else None
            return (
                self.model.forward_post(hidden_states, apply_norm=True)["x_pred"],
                embeds,
            )

        outputs = self.model(
            **common_kwargs,
            use_flash_attn=True,  # official flash two-pass attention path
            pad_mask=pad_mask,
            **vision_kwargs,
        )

        embeds = None
        if "cond_image_embeds" in outputs:
            embeds = (
                outputs["cond_image_embeds"],
                outputs["cond_deepstack_image_embeds"],
            )
        return outputs["x_pred"], embeds


class HiDreamO1ImageDecodingStage(PipelineStage):
    """Unpatchify the denoised pixel patches; the model ships no VAE."""

    @property
    def role_affinity(self) -> RoleType:
        return RoleType.DECODER

    def _decode(
        self, patches: torch.Tensor, batch: Req, server_args: ServerArgs
    ) -> torch.Tensor:
        arch_config = server_args.pipeline_config.dit_config.arch_config
        patch_size = arch_config.patch_size
        pixels = unpatchify_pixels(
            patches.float(),
            channels=arch_config.in_channels,
            patch_size=patch_size,
            h_patches=batch.height // patch_size,
            w_patches=batch.width // patch_size,
        )
        # Patches live in [-1, 1]; the framework expects [0, 1] and a frame axis.
        frames = ((pixels + 1.0) / 2.0).clamp(0.0, 1.0).unsqueeze(2)
        return server_args.pipeline_config.post_decoding(frames, server_args)

    @torch.no_grad()
    def forward(self, batch: Req, server_args: ServerArgs) -> OutputBatch:
        frames = self._decode(batch.latents, batch, server_args)

        trajectory_decoded = None
        if batch.return_trajectory_decoded:
            assert batch.trajectory_latents is not None, (
                "batch should have trajectory latents"
            )
            # Stacked as [batch, timesteps, patches, channels * patch^2]; decode
            # per timestep, since unpatchify is cheap enough not to need batching.
            trajectory_decoded = [
                self._decode(batch.trajectory_latents[:, i], batch, server_args)
                for i in range(batch.trajectory_latents.shape[1])
            ]

        return OutputBatch(
            output=frames,
            trajectory_timesteps=batch.trajectory_timesteps,
            trajectory_latents=batch.trajectory_latents,
            rollout_trajectory_data=batch.rollout_trajectory_data,
            trajectory_decoded=trajectory_decoded,
            metrics=batch.metrics,
            noise_pred=None,
            usage=batch.usage,
        )
