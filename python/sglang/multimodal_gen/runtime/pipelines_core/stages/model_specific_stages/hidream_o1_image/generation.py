# Adapted from the official HiDream-O1-Image repository (MIT License):
# https://github.com/HiDream-ai/HiDream-O1-Image
"""HiDream-O1-Image generation stage: preprocessing, denoising, and decoding."""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import torch
import torchvision.transforms.v2 as transforms
from PIL import Image
from sglang.multimodal_gen.runtime.models.schedulers.scheduling_flow_unipc_multistep import (
    FlowUniPCMultistepScheduler,
)
from sglang.multimodal_gen.runtime.pipelines_core.schedule_batch import Req
from sglang.multimodal_gen.runtime.server_args import ServerArgs
from sglang.multimodal_gen.runtime.utils.logging_utils import init_logger

logger = init_logger(__name__)

TIMESTEP_TOKEN_NUM = 1
PATCH_SIZE = 32
T_EPS = 0.001
CONDITION_IMAGE_SIZE = 384
NOISE_SCALE = 8.0
REROLL_OUT_OF_RANGE_THRESHOLD = 0.001
REROLL_MAX_ATTEMPTS = 3

# Historical HIDREAM_O1_* aliases: the tested package surface imports these
# names, while the recipe code reads NOISE_SCALE / T_EPS.
HIDREAM_O1_BOI_TOKEN = "<|boi_token|>"
HIDREAM_O1_TMS_TOKEN = "<|tms_token|>"
HIDREAM_O1_NOISE_SCALE = NOISE_SCALE
HIDREAM_O1_T_EPS = T_EPS
# Image mrope positions start at a fixed offset instead of continuing the text
# positions, so the grid does not shift when the prompt length changes.
HIDREAM_O1_MROPE_FIX_POINT = 4096

PREDEFINED_RESOLUTIONS = [
    (2048, 2048),
    (2304, 1728),
    (1728, 2304),
    (2560, 1440),
    (1440, 2560),
    (2496, 1664),
    (1664, 2496),
    (3104, 1312),
    (1312, 3104),
    (2304, 1792),
    (1792, 2304),
]

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

_HIDREAM_CACHE_DIT_RDT = 0.05


class FlowMatchScheduler:
    """Simple Euler flow matching scheduler with sigma shift."""

    def __init__(self, shift: float = 3.0):
        self.shift = shift
        self.timesteps = None
        self.sigmas = None

    def set_timesteps(self, num_inference_steps: int, device: torch.device):
        sigmas = np.linspace(1.0, 1.0 / num_inference_steps, num_inference_steps)
        sigmas = self.shift * sigmas / (1 + (self.shift - 1) * sigmas)
        timesteps = sigmas * 1000
        sigmas_full = np.concatenate([sigmas, [0.0]]).astype(np.float32)
        self.sigmas = torch.from_numpy(sigmas_full).to(
            device=device, dtype=torch.float32
        )
        self.timesteps = torch.from_numpy(timesteps.astype(np.int64)).to(
            device=device, dtype=torch.long
        )

    def set_timesteps_list(self, timesteps_list, device: torch.device):
        timesteps = torch.tensor(timesteps_list, device=device, dtype=torch.long)
        sigmas = [t.item() / 1000.0 for t in timesteps]
        sigmas.append(0.0)
        self.sigmas = torch.tensor(sigmas, device=device, dtype=torch.float32)
        self.timesteps = timesteps

    def step(
        self, model_output: torch.Tensor, timestep: torch.Tensor, sample: torch.Tensor
    ) -> torch.Tensor:
        """Euler step: z_next = z + dt * model_output."""
        step_idx = (
            (self.timesteps == timestep.long()).nonzero(as_tuple=True)[0][0].item()
        )
        sigma = self.sigmas[step_idx]
        sigma_next = self.sigmas[step_idx + 1]
        dt = sigma_next - sigma
        return sample + dt * model_output


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
    """Official models/pipeline.py build_scheduler.

    The UniPC branch passes only what the official call passes: the ported
    FlowUniPCMultistepScheduler already ships the fork's diffusion defaults
    (flow_prediction, predict_x0, 1000 train timesteps). A ``timesteps_list``
    replaces the computed schedule for every scheduler type: the timesteps
    stay as given and the sigmas become t/1000 with a final 0.
    """
    if scheduler_name == "flash":
        # The official call passes shift here, but the Dev recipe always
        # supplies the fixed timesteps list, which overrides the schedule.
        sched = FlashFlowMatchScheduler()
    elif scheduler_name == "flow_match":
        sched = FlowMatchScheduler(shift=shift)
    elif scheduler_name == "default":
        sched = FlowUniPCMultistepScheduler(use_dynamic_shifting=False, shift=shift)
    else:
        raise ValueError(f"Unknown scheduler_name={scheduler_name!r}")
    if timesteps_list is not None:
        sched.set_timesteps_list(timesteps_list, device)
    elif scheduler_name == "flash":
        sched.set_timesteps(num_inference_steps, shift, device)
    else:
        sched.set_timesteps(num_inference_steps, device)
    return sched


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


def decode_patches_to_image(
    z: torch.Tensor,
    h_patches: int,
    w_patches: int,
    patch_size: int = PATCH_SIZE,
) -> torch.Tensor:
    """Convert patch tensor [B, h*w, 3*ps^2] to images [B, 3, H, W] in [0, 1]."""
    img = (z.float() + 1) / 2
    b = z.shape[0]
    img = img.reshape(b, h_patches, w_patches, 3, patch_size, patch_size)
    # .contiguous() required on NPU before reshape after permute
    img = (
        img.permute(0, 3, 1, 4, 2, 5)
        .contiguous()
        .reshape(b, 3, h_patches * patch_size, w_patches * patch_size)
    )
    return img


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
    if len(seeds) == p_count * n:
        pass
    elif len(seeds) == p_count:
        # one base seed per prompt: expand consecutive outputs
        seeds = [base + j for base in seeds for j in range(n)]
    elif len(seeds) == 1:
        seeds = [seeds[0] + i for i in range(p_count * n)]
    else:
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
