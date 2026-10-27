"""HiDream-O1-Image generation stage: preprocessing, denoising, and decoding."""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import torch
from PIL import Image
from tqdm.auto import tqdm

from sglang.multimodal_gen.runtime.disaggregation.roles import RoleType
from sglang.multimodal_gen.runtime.pipelines_core.schedule_batch import (
    OutputBatch,
    Req,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.base import PipelineStage
from sglang.multimodal_gen.runtime.server_args import ServerArgs
from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.hidream.layout import (
    create_layout_reference_images,
    load_layout_bboxes,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.hidream.prompt_enhance import (
    enhance_prompt,
)
from sglang.multimodal_gen.runtime.utils.logging_utils import init_logger

logger = init_logger(__name__)

TIMESTEP_TOKEN_NUM = 1
PATCH_SIZE = 32
T_EPS = 0.001
CONDITION_IMAGE_SIZE = 384
NOISE_SCALE = 8.0

PREDEFINED_RESOLUTIONS = [
    (2048, 2048), (2304, 1728), (1728, 2304), (2560, 1440), (1440, 2560),
    (2496, 1664), (1664, 2496), (3104, 1312), (1312, 3104), (2304, 1792),
    (1792, 2304),
]

DEFAULT_TIMESTEPS = [
    999, 987, 974, 960, 945, 929, 913, 895, 877, 857, 836, 814, 790, 764,
    737, 707, 675, 640, 602, 560, 515, 464, 409, 347, 278, 199, 110, 8,
]


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
        self.sigmas = torch.from_numpy(sigmas_full).to(device=device, dtype=torch.float32)
        self.timesteps = torch.from_numpy(timesteps.astype(np.int64)).to(
            device=device, dtype=torch.long)

    def set_timesteps_list(self, timesteps_list, device: torch.device):
        timesteps = torch.tensor(timesteps_list, device=device, dtype=torch.long)
        sigmas = [t.item() / 1000.0 for t in timesteps]
        sigmas.append(0.0)
        self.sigmas = torch.tensor(sigmas, device=device, dtype=torch.float32)
        self.timesteps = timesteps

    def step(self, model_output: torch.Tensor, timestep: torch.Tensor,
             sample: torch.Tensor) -> torch.Tensor:
        """Euler step: z_next = z + dt * model_output."""
        step_idx = (self.timesteps == timestep.long()).nonzero(as_tuple=True)[0][0].item()
        sigma = self.sigmas[step_idx]
        sigma_next = self.sigmas[step_idx + 1]
        dt = sigma_next - sigma
        return sample + dt * model_output


class FlashFlowMatchScheduler:
    """Official DEV-model flash scheduler: Euler with per-step noise injection."""

    def __init__(self):
        self.timesteps = None
        self.sigmas = None
        self.num_inference_steps = None

    def set_timesteps(self, num_inference_steps: int, shift: float,
                      device: torch.device):
        base_min = 1.0 / 1000
        sigma_min = shift * base_min / (1 + (shift - 1) * base_min)
        sigma_max = 1.0
        base = np.linspace(sigma_max, sigma_min, num_inference_steps)
        sigmas = shift * base / (1 + (shift - 1) * base)
        timesteps = (sigmas * 1000.0).astype(np.float32)
        sigmas_full = np.concatenate([sigmas, [0.0]]).astype(np.float32)

        self.sigmas = torch.from_numpy(sigmas_full).to(
            device=device, dtype=torch.float32)
        self.timesteps = torch.from_numpy(timesteps).to(
            device=device, dtype=torch.float32)
        self.num_inference_steps = num_inference_steps

    def set_timesteps_list(self, timesteps_list, device: torch.device):
        timesteps = torch.tensor(timesteps_list, device=device, dtype=torch.long)
        sigmas = [t.item() / 1000.0 for t in timesteps]
        sigmas.append(0.0)
        self.sigmas = torch.tensor(sigmas, device=device, dtype=torch.float32)
        self.timesteps = timesteps
        self.num_inference_steps = len(timesteps)

    def step(self, model_output: torch.Tensor, timestep: torch.Tensor,
             sample: torch.Tensor, s_noise: float = 1.0,
             noise_clip_std: float = 0.0,
             generator: torch.Generator | None = None) -> torch.Tensor:
        step_idx = (self.timesteps == timestep.long()).nonzero(
            as_tuple=True)[0][0].item()
        sigma = self.sigmas[step_idx]
        sample = sample.to(torch.float32)
        denoised = sample - model_output * sigma
        if step_idx + 1 < len(self.sigmas):
            sigma_next = self.sigmas[step_idx + 1]
            noise = torch.randn(model_output.shape,
                                device=generator.device,
                                dtype=denoised.dtype, generator=generator)
            noise = noise.to(model_output.device)
            if noise_clip_std > 0:
                clip_val = noise_clip_std * noise.std().item()
                noise = noise.clamp(min=-clip_val, max=clip_val)
            sample = (sigma_next * noise * s_noise
                      + (1.0 - sigma_next) * denoised)
        return sample


class FlowUniPCScheduler:
    """UniPC multistep scheduler (predict_x0, bh2, order 2), ported from official."""

    def __init__(self, shift: float = 3.0, solver_order: int = 2):
        self.shift = shift
        self.solver_order = solver_order
        self.predict_x0 = True
        self.lower_order_final = True
        self.timesteps = None
        self.sigmas = None
        self._step_index = None
        self.model_outputs = [None] * solver_order
        self.timestep_list = [None] * solver_order
        self.lower_order_nums = 0
        self.last_sample = None
        self.this_order = 1

    @property
    def step_index(self):
        return self._step_index

    def set_timesteps(self, num_inference_steps: int, device: torch.device):
        base_max = 1.0 - 1.0 / 1000
        sigma_max = self.shift * base_max / (1 + (self.shift - 1) * base_max)
        sigma_min = 0.0
        sigmas = np.linspace(sigma_max, sigma_min, num_inference_steps + 1)[:-1]
        sigmas = self.shift * sigmas / (1 + (self.shift - 1) * sigmas)
        timesteps = sigmas * 1000
        sigmas_full = np.concatenate([sigmas, [0.0]]).astype(np.float32)

        self.sigmas = torch.from_numpy(sigmas_full)
        self.timesteps = torch.from_numpy(timesteps.astype(np.int64)).to(
            device=device, dtype=torch.long)
        self.num_inference_steps = len(timesteps)

        self.model_outputs = [None] * self.solver_order
        self.timestep_list = [None] * self.solver_order
        self.lower_order_nums = 0
        self.last_sample = None
        self._step_index = None
        self.this_order = 1

    def _init_step_index(self, timestep):
        if isinstance(timestep, torch.Tensor):
            timestep = timestep.to(self.timesteps.device)
        indices = (self.timesteps == timestep).nonzero()
        pos = 1 if len(indices) > 1 else 0
        self._step_index = indices[pos].item()

    def _sigma_to_alpha_sigma_t(self, sigma):
        return 1 - sigma, sigma

    def convert_model_output(self, model_output, sample):
        sigma = self.sigmas[self.step_index]
        x0_pred = sample - sigma * model_output
        return x0_pred

    def multistep_uni_p_bh_update(self, sample, order):
        x = sample
        sigma_t = self.sigmas[self.step_index + 1]
        sigma_s0 = self.sigmas[self.step_index]
        alpha_t, sigma_t_a = self._sigma_to_alpha_sigma_t(sigma_t)
        alpha_s0, sigma_s0_a = self._sigma_to_alpha_sigma_t(sigma_s0)

        m0 = self.model_outputs[-1]

        if sigma_t_a == 0:  # final step: result is the x0 prediction
            return m0.to(x.dtype)

        lambda_t = torch.log(alpha_t) - torch.log(sigma_t_a)
        lambda_s0 = torch.log(alpha_s0) - torch.log(sigma_s0_a)
        h = lambda_t - lambda_s0
        device = x.device

        rks = []
        D1s = []
        for i in range(1, order):
            si = self.step_index - i
            mi = self.model_outputs[-(i + 1)]
            alpha_si, sigma_si = self._sigma_to_alpha_sigma_t(self.sigmas[si])
            lambda_si = torch.log(alpha_si) - torch.log(sigma_si)
            rk = (lambda_si - lambda_s0) / h
            rks.append(rk)
            D1s.append((mi - m0) / rk)

        rks.append(1.0)
        rks = torch.tensor(rks, device=device)

        R = []
        b = []
        hh = -h
        h_phi_1 = torch.expm1(hh)
        h_phi_k = h_phi_1 / hh - 1
        B_h = torch.expm1(hh)

        factorial_i = 1
        for i in range(1, order + 1):
            R.append(torch.pow(rks, i - 1))
            b.append(h_phi_k * factorial_i / B_h)
            factorial_i *= i + 1
            h_phi_k = h_phi_k / hh - 1 / factorial_i

        R = torch.stack(R)
        b = torch.tensor(b, device=device)

        if len(D1s) > 0:
            D1s = torch.stack(D1s, dim=1)
            rhos_p = torch.tensor([0.5], dtype=x.dtype, device=device)
        else:
            D1s = None
            rhos_p = None

        x_t_ = sigma_t_a / sigma_s0_a * x - alpha_t * h_phi_1 * m0
        if D1s is not None:
            pred_res = torch.einsum("k,bkc...->bc...", rhos_p, D1s)
        else:
            pred_res = 0
        x_t = x_t_ - alpha_t * B_h * pred_res

        return x_t.to(x.dtype)

    def multistep_uni_c_bh_update(self, this_model_output, last_sample, this_sample, order):
        model_output_list = self.model_outputs
        m0 = model_output_list[-1]
        x = last_sample
        model_t = this_model_output

        sigma_t = self.sigmas[self.step_index]
        sigma_s0 = self.sigmas[self.step_index - 1]
        alpha_t, sigma_t_a = self._sigma_to_alpha_sigma_t(sigma_t)
        alpha_s0, sigma_s0_a = self._sigma_to_alpha_sigma_t(sigma_s0)

        lambda_t = torch.log(alpha_t) - torch.log(sigma_t_a)
        lambda_s0 = torch.log(alpha_s0) - torch.log(sigma_s0_a)
        h = lambda_t - lambda_s0
        device = this_sample.device

        rks = []
        D1s = []
        for i in range(1, order):
            si = self.step_index - (i + 1)
            mi = model_output_list[-(i + 1)]
            alpha_si, sigma_si = self._sigma_to_alpha_sigma_t(self.sigmas[si])
            lambda_si = torch.log(alpha_si) - torch.log(sigma_si)
            rk = (lambda_si - lambda_s0) / h
            rks.append(rk)
            D1s.append((mi - m0) / rk)

        rks.append(1.0)
        rks = torch.tensor(rks, device=device)

        R = []
        b = []
        hh = -h
        h_phi_1 = torch.expm1(hh)
        h_phi_k = h_phi_1 / hh - 1
        B_h = torch.expm1(hh)

        factorial_i = 1
        for i in range(1, order + 1):
            R.append(torch.pow(rks, i - 1))
            b.append(h_phi_k * factorial_i / B_h)
            factorial_i *= i + 1
            h_phi_k = h_phi_k / hh - 1 / factorial_i

        R = torch.stack(R)
        b = torch.tensor(b, device=device)

        if len(D1s) > 0:
            D1s = torch.stack(D1s, dim=1)
        else:
            D1s = None

        if order == 1:
            rhos_c = torch.tensor([0.5], dtype=x.dtype, device=device)
        else:
            # 2x2 solve of R = [[1, 1], [r1, 1]] (NPU: no torch.linalg.solve)
            r1 = rks[0]
            det = 1 - r1
            rhos_c = torch.stack([
                (b[0] - b[1]) / det,
                (b[1] - r1 * b[0]) / det,
            ]).to(dtype=x.dtype, device=device)

        x_t_ = sigma_t_a / sigma_s0_a * x - alpha_t * h_phi_1 * m0
        if D1s is not None:
            corr_res = torch.einsum("k,bkc...->bc...", rhos_c[:-1], D1s)
        else:
            corr_res = 0
        D1_t = model_t - m0
        x_t = x_t_ - alpha_t * B_h * (corr_res + rhos_c[-1] * D1_t)

        return x_t.to(x.dtype)

    def step(self, model_output, timestep, sample):
        if self.step_index is None:
            self._init_step_index(timestep)

        use_corrector = (
            self.step_index > 0 and
            self.last_sample is not None
        )

        model_output_convert = self.convert_model_output(model_output, sample)
        if use_corrector:
            sample = self.multistep_uni_c_bh_update(
                this_model_output=model_output_convert,
                last_sample=self.last_sample,
                this_sample=sample,
                order=self.this_order,
            )

        for i in range(self.solver_order - 1):
            self.model_outputs[i] = self.model_outputs[i + 1]
            self.timestep_list[i] = self.timestep_list[i + 1]
        self.model_outputs[-1] = model_output_convert
        self.timestep_list[-1] = timestep

        if self.lower_order_final:
            this_order = min(self.solver_order,
                             len(self.timesteps) - self.step_index)
        else:
            this_order = self.solver_order
        self.this_order = min(this_order, self.lower_order_nums + 1)

        self.last_sample = sample
        prev_sample = self.multistep_uni_p_bh_update(sample, self.this_order)

        if self.lower_order_nums < self.solver_order:
            self.lower_order_nums += 1
        self._step_index += 1

        return prev_sample


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
        video_grid_thw = torch.repeat_interleave(video_grid_thw, video_grid_thw[:, 0], dim=0)
        video_grid_thw[:, 0] = 1

    mrope_position_deltas = []
    if input_ids is not None and (image_grid_thw is not None or video_grid_thw is not None):
        total_input_ids = input_ids
        if attention_mask is None:
            attention_mask = torch.ones_like(total_input_ids)
        position_ids = torch.ones(
            3, input_ids.shape[0], input_ids.shape[1],
            dtype=input_ids.dtype, device=input_ids.device,
        )
        image_index, video_index = 0, 0
        attention_mask = attention_mask.to(total_input_ids.device)

        for i, input_ids_i in enumerate(total_input_ids):
            input_ids_i = input_ids_i[attention_mask[i] == 1]
            vision_start_indices = torch.argwhere(input_ids_i == vision_start_token_id).squeeze(1)
            vision_tokens = input_ids_i[vision_start_indices + 1]
            image_nums = (vision_tokens == image_token_id).sum()
            video_nums = (vision_tokens == video_token_id).sum()
            input_tokens = input_ids_i.tolist()
            llm_pos_ids_list: list = []
            st = 0
            remain_images, remain_videos = image_nums, video_nums

            for _ in range(image_nums + video_nums):
                ed_image = input_tokens.index(image_token_id, st) if image_token_id in input_tokens and remain_images > 0 else len(input_tokens) + 1
                ed_video = input_tokens.index(video_token_id, st) if video_token_id in input_tokens and remain_videos > 0 else len(input_tokens) + 1
                if ed_image < ed_video:
                    t, h, w = image_grid_thw[image_index][0], image_grid_thw[image_index][1], image_grid_thw[image_index][2]
                    image_index += 1
                    remain_images -= 1
                    ed = ed_image
                else:
                    t, h, w = video_grid_thw[video_index][0], video_grid_thw[video_index][1], video_grid_thw[video_index][2]
                    video_index += 1
                    remain_videos -= 1
                    ed = ed_video

                llm_grid_t = t.item()
                llm_grid_h = h.item() // spatial_merge_size
                llm_grid_w = w.item() // spatial_merge_size
                text_len = ed - st
                text_len -= skip_vision_start_token[image_index - 1]
                text_len = max(0, text_len)

                st_idx = llm_pos_ids_list[-1].max() + 1 if len(llm_pos_ids_list) > 0 else 0
                llm_pos_ids_list.append(torch.arange(text_len).view(1, -1).expand(3, -1) + st_idx)

                t_index = torch.arange(llm_grid_t).view(-1, 1).expand(-1, llm_grid_h * llm_grid_w).flatten()
                h_index = torch.arange(llm_grid_h).view(1, -1, 1).expand(llm_grid_t, -1, llm_grid_w).flatten()
                w_index = torch.arange(llm_grid_w).view(1, 1, -1).expand(llm_grid_t, llm_grid_h, -1).flatten()

                if skip_vision_start_token[image_index - 1]:
                    if fix_point > 0:
                        fix_point = fix_point - st_idx
                    llm_pos_ids_list.append(torch.stack([t_index, h_index, w_index]) + fix_point + st_idx)
                    fix_point = 0
                else:
                    llm_pos_ids_list.append(torch.stack([t_index, h_index, w_index]) + text_len + st_idx)
                st = ed + llm_grid_t * llm_grid_h * llm_grid_w

            if st < len(input_tokens):
                st_idx = llm_pos_ids_list[-1].max() + 1 if len(llm_pos_ids_list) > 0 else 0
                text_len = len(input_tokens) - st
                llm_pos_ids_list.append(torch.arange(text_len).view(1, -1).expand(3, -1) + st_idx)

            llm_positions = torch.cat(llm_pos_ids_list, dim=1).reshape(3, -1)
            position_ids[..., i, attention_mask[i] == 1] = llm_positions.to(position_ids.device)
            mrope_position_deltas.append(llm_positions.max() + 1 - len(total_input_ids[i]))

        mrope_position_deltas = torch.tensor(mrope_position_deltas, device=input_ids.device).unsqueeze(1)
        return position_ids, mrope_position_deltas
    else:
        if attention_mask is not None:
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)
            position_ids = position_ids.unsqueeze(0).expand(3, -1, -1).to(attention_mask.device)
            max_position_ids = position_ids.max(0, keepdim=False)[0].max(-1, keepdim=True)[0]
            mrope_position_deltas = max_position_ids + 1 - attention_mask.shape[-1]
        else:
            position_ids = torch.arange(input_ids.shape[1], device=input_ids.device).view(1, 1, -1).expand(3, input_ids.shape[0], -1)
            mrope_position_deltas = torch.zeros([input_ids.shape[0], 1], device=input_ids.device, dtype=input_ids.dtype)
        return position_ids, mrope_position_deltas


def build_t2i_text_sample(
    prompt: str, height: int, width: int,
    tokenizer: Any, processor: Any, model_config: Any,
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
        processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        + boi_token
        + tms_token * TIMESTEP_TOKEN_NUM
    )
    input_ids = tokenizer.encode(template_caption, return_tensors="pt", add_special_tokens=False)

    image_grid_thw = torch.tensor(
        [1, height // PATCH_SIZE, width // PATCH_SIZE], dtype=torch.int64
    ).unsqueeze(0)

    vision_tokens = torch.zeros((1, image_len), dtype=input_ids.dtype) + image_token_id
    vision_tokens[0, 0] = vision_start_token_id
    input_ids_pad = torch.cat([input_ids, vision_tokens], dim=-1)

    position_ids, _ = get_rope_index_fix_point(
        1, image_token_id, video_token_id, vision_start_token_id,
        input_ids=input_ids_pad, image_grid_thw=image_grid_thw,
        video_grid_thw=None, attention_mask=None,
        skip_vision_start_token=[1],
    )

    txt_seq_len = input_ids.shape[-1]
    all_seq_len = position_ids.shape[-1]

    token_types = torch.zeros((1, all_seq_len), dtype=input_ids.dtype)
    bgn = txt_seq_len - TIMESTEP_TOKEN_NUM
    token_types[0, bgn: bgn + image_len + TIMESTEP_TOKEN_NUM] = 1
    token_types[0, txt_seq_len - TIMESTEP_TOKEN_NUM: txt_seq_len] = 3

    vinput_mask = (token_types == 1)
    token_types_bin = (token_types > 0).to(token_types.dtype)

    return {
        "input_ids": input_ids,
        "position_ids": position_ids,
        "token_types": token_types_bin,
        "vinput_mask": vinput_mask,
    }


def decode_patches_to_image(
    z: torch.Tensor, h_patches: int, w_patches: int, patch_size: int = PATCH_SIZE,
) -> torch.Tensor:
    """Convert patch tensor to image tensor in [0, 1] range."""
    img = (z.float() + 1) / 2
    img = img.reshape(1, h_patches, w_patches, 3, patch_size, patch_size)
    # .contiguous() required on NPU before reshape after permute
    img = img.permute(0, 3, 1, 4, 2, 5).contiguous().reshape(
        1, 3, h_patches * patch_size, w_patches * patch_size)
    return img


def resize_pilimage(pil_image, image_size, patch_size=16,
                    resampler=None):
    """Aspect-preserving resize + center crop (official utils.py)."""
    from PIL import Image as _Image
    if resampler is None:
        resampler = _Image.BICUBIC
    while min(*pil_image.size) >= 2 * image_size:
        pil_image = pil_image.resize(
            tuple(x // 2 for x in pil_image.size), resample=_Image.BOX)

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
            [new_size[0], round(height / s1)], resample=resampler)
        top = (round(height / s1) - new_size[1]) // 2
        pil_image = pil_image.crop((0, top, new_size[0], top + new_size[1]))
    else:
        pil_image = pil_image.resize(
            [round(width / s2), new_size[1]], resample=resampler)
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
    x = x.permute(1, 3, 0, 2, 4).contiguous().reshape(
        h_patches * w_patches, c * patch_size * patch_size)
    return x


def _ref_tensor_transform():
    """PIL -> float32 tensor in [-1, 1] (official TENSOR_TRANSFORM)."""
    import torchvision.transforms.v2 as transforms
    return transforms.Compose([
        transforms.ToImage(),
        transforms.ToDtype(torch.float32, scale=True),
        transforms.Normalize([0.5], [0.5]),
    ])


def build_ti2i_text_sample(prompt, ref_pils_vlm, ref_pils_resized, height, width,
                           ref_image_lens, tokenizer, processor, model_config):
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
        messages, tokenize=False, add_generation_prompt=True)
    proc = processor(text=[template_caption], images=ref_pils_vlm,
                     padding="longest", return_tensors="pt")
    boi_token = getattr(tokenizer, "boi_token", "<|boi_token|>")
    tms_token = getattr(tokenizer, "tms_token", "<|tms_token|>")
    input_ids_2 = tokenizer.encode(
        boi_token + tms_token * TIMESTEP_TOKEN_NUM, return_tensors="pt",
        add_special_tokens=False)
    input_ids = torch.cat([proc.input_ids, input_ids_2], dim=-1)

    igthw_cond = proc.image_grid_thw.clone()
    for i in range(K):
        igthw_cond[i, 1] //= spatial_merge_size
        igthw_cond[i, 2] //= spatial_merge_size
    image_grid_thw_tgt = torch.tensor(
        [[1, height // PATCH_SIZE, width // PATCH_SIZE]], dtype=torch.int64)
    image_grid_thw_ref = torch.tensor(
        [[1, p.height // PATCH_SIZE, p.width // PATCH_SIZE]
         for p in ref_pils_resized], dtype=torch.int64)
    igthw_all = torch.cat([igthw_cond, image_grid_thw_tgt, image_grid_thw_ref],
                          dim=0)

    vision_tokens_list = []
    vt_tgt = torch.full((1, tgt_image_len), image_token_id,
                        dtype=input_ids.dtype)
    vt_tgt[0, 0] = vision_start_token_id
    vision_tokens_list.append(vt_tgt)
    for rl in ref_image_lens:
        vt_ref = torch.full((1, rl), image_token_id, dtype=input_ids.dtype)
        vt_ref[0, 0] = vision_start_token_id
        vision_tokens_list.append(vt_ref)
    vision_tokens = torch.cat(vision_tokens_list, dim=1)
    input_ids_pad = torch.cat([input_ids, vision_tokens], dim=-1)

    position_ids, _ = get_rope_index_fix_point(
        1, image_token_id, video_token_id, vision_start_token_id,
        input_ids=input_ids_pad, image_grid_thw=igthw_all,
        video_grid_thw=None, attention_mask=None,
        skip_vision_start_token=[0] * K + [1] + [1] * K,
    )
    txt_seq_len = input_ids.shape[-1]
    all_seq_len = position_ids.shape[-1]
    total_ref_len = sum(ref_image_lens)

    token_types_raw = torch.zeros((1, all_seq_len), dtype=input_ids.dtype)
    bgn = txt_seq_len - TIMESTEP_TOKEN_NUM
    end = bgn + tgt_image_len + TIMESTEP_TOKEN_NUM
    token_types_raw[0, bgn:end] = 1
    token_types_raw[0, end:end + total_ref_len] = 2
    token_types_raw[0, txt_seq_len - TIMESTEP_TOKEN_NUM: txt_seq_len] = 3

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


def _per_prompt_work(batch: Req, prompts: list) -> list[tuple]:
    """(prompt, height, width, seed) per image of a possibly merged request;
    validation prevents zip from silently truncating it."""
    n = len(prompts)
    if int(batch.num_outputs_per_prompt) != 1:
        raise ValueError(
            "HiDream expects per-output expansion before generation; "
            f"got num_outputs_per_prompt={batch.num_outputs_per_prompt}."
        )

    def broadcast(value, name):
        values = value if isinstance(value, list) else [value]
        if len(values) == 1:
            values *= n
        if len(values) != n:
            raise ValueError(
                f"HiDream requires one {name} per prompt; got "
                f"{len(values)} for {n} prompts"
            )
        return values

    seeds = batch.seeds
    if seeds is None:
        seeds = batch.extra.get("dynamic_batch_seeds") or batch.seed
    if not isinstance(seeds, list):
        seeds = [seeds]
    seeds = [s[0] if isinstance(s, list) else int(s) for s in seeds]

    return list(zip(
        prompts,
        broadcast(batch.height, "height"),
        broadcast(batch.width, "width"),
        broadcast(seeds, "seed"),
    ))


class HiDreamGenerationStage(PipelineStage):
    """Full generation stage: preprocessing -> denoising -> decoding."""

    def __init__(self, model: torch.nn.Module, processor: Any):
        super().__init__()
        self.model = model
        self.processor = processor

    @property
    def role_affinity(self) -> RoleType:
        return RoleType.DENOISER

    def forward(self, batch: Req, server_args: ServerArgs) -> OutputBatch:
        first_param = next(self.model.parameters())
        device = first_param.device

        prompts = batch.prompt if isinstance(batch.prompt, list) else [batch.prompt]
        sampling_params = getattr(batch, "sampling_params", None)
        dtype = torch.bfloat16
        w_dtype = first_param.dtype

        requested_steps = int(
            getattr(sampling_params, "num_inference_steps", None)
            or getattr(batch, "num_inference_steps", None)
            or getattr(server_args.pipeline_config, "num_inference_steps", None)
            or len(DEFAULT_TIMESTEPS)
        )
        model_type = batch.model_type if batch.model_type is not None else "dev"
        if model_type == "full":
            num_steps = requested_steps
        else:
            # DEV checkpoints always run the fixed 28-step schedule.
            num_steps = len(DEFAULT_TIMESTEPS)
            if requested_steps != num_steps:
                logger.warning(
                    "HiDream DEV generation always uses num_inference_steps=%d "
                    "(the official fixed DEFAULT_TIMESTEPS schedule); ignoring "
                    "the requested %d.",
                    num_steps, requested_steps,
                )
        guidance_scale = batch.guidance_scale
        if guidance_scale is None:
            guidance_scale = 0.0
        guidance_scale = float(guidance_scale)
        scheduler_name = (
            batch.scheduler_name if batch.scheduler_name is not None else "flash")
        noise_scale_start = float(
            batch.noise_scale_start
            if batch.noise_scale_start is not None else NOISE_SCALE)
        noise_scale_end = float(
            batch.noise_scale_end
            if batch.noise_scale_end is not None else NOISE_SCALE)
        noise_clip_std = float(
            batch.noise_clip_std if batch.noise_clip_std is not None else 0.0)
        shift = float(batch.shift) if batch.shift is not None else 1.0
        layout_bboxes = batch.layout_bboxes

        tokenizer = self.processor.tokenizer if hasattr(self.processor, "tokenizer") else self.processor
        model_config = self.model.config
        ref_images_batch = getattr(batch, "image_path", None)

        # Prompt enhancement: skip editing prompts and warmup requests.
        if (
            server_args.pe_server_url
            and not ref_images_batch
            and not getattr(batch, "is_warmup", False)
        ):
            prompts = [
                enhance_prompt(prompt=p, base_url=server_args.pe_server_url)
                for p in prompts
            ]


        results = []
        for prompt, height, width, seed in _per_prompt_work(batch, prompts):
            height, width = int(height), int(width)

            ref_paths = None
            if ref_images_batch:
                ref_paths = ([ref_images_batch]
                             if isinstance(ref_images_batch, str)
                             else [p for p in ref_images_batch if p])
                if not ref_paths:
                    ref_paths = None

            # TI2I (single reference) keeps the reference aspect by default:
            # the target size derives from the reference resized to 2048
            # (patch-aligned). An explicit width AND height in the request
            # wins (snapped); T2I always snaps the requested size.
            preresized_ref_pil = None
            explicit_fields = getattr(
                sampling_params, "_explicit_fields", None) or ()
            explicit_size = ("width" in explicit_fields
                             and "height" in explicit_fields)
            if ref_paths and len(ref_paths) == 1 and not explicit_size:
                preresized_ref_pil = resize_pilimage(
                    Image.open(ref_paths[0]).convert("RGB"), 2048, PATCH_SIZE)
                width, height = preresized_ref_pil.size
                logger.info("TI2I: target size %dx%d from reference aspect",
                            width, height)
            else:
                best_w, best_h = find_closest_resolution(width, height)
                if best_w != width or best_h != height:
                    logger.info("Resolution snapped from %dx%d to %dx%d",
                                width, height, best_w, best_h)
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

                layout_data = None
                if layout_bboxes and preresized_ref_pil is None:
                    try:
                        layout_data = load_layout_bboxes(layout_bboxes)
                        K += 1
                    except Exception as e:
                        logger.warning("Incorrect layout_bboxes: %s, %s",
                                       layout_bboxes, e)

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

                if layout_data is not None:
                    ref_pils = create_layout_reference_images(
                        ref_pils, layout_data, width, height,
                        ref_max_size=max_size, patch_size=PATCH_SIZE)

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
                    ref_images.append(
                        _patchify_image_tensor(transform(pil_r), PATCH_SIZE))
                ref_image_lens = [x.shape[0] for x in ref_images]
                ref_patches = torch.cat(ref_images, dim=0).unsqueeze(0).to(
                    device, dtype)

                if K <= 4:
                    cond_img_size = CONDITION_IMAGE_SIZE
                elif K <= 8:
                    cond_img_size = CONDITION_IMAGE_SIZE * 48 // 64
                else:
                    cond_img_size = CONDITION_IMAGE_SIZE // 2
                ref_pils_vlm = []
                for p in ref_pils_resized:
                    cond_w, cond_h = calculate_dimensions(
                        cond_img_size, p.width / p.height)
                    ref_pils_vlm.append(
                        p.resize((cond_w, cond_h), Image.LANCZOS))
                logger.info("TI2I: %d reference image(s), %d conditioning patches",
                            K, sum(ref_image_lens))

            if ref_paths:
                cond_sample = build_ti2i_text_sample(
                    prompt, ref_pils_vlm, ref_pils_resized, height, width,
                    ref_image_lens, tokenizer, self.processor, model_config
                )
                uncond_sample = None
                if guidance_scale > 1.0:
                    uncond_sample = build_ti2i_text_sample(
                        " ", ref_pils_vlm, ref_pils_resized, height, width,
                        ref_image_lens, tokenizer, self.processor, model_config
                    )
            else:
                cond_sample = build_t2i_text_sample(
                    prompt, height, width, tokenizer, self.processor, model_config
                )
                uncond_sample = None
                if guidance_scale > 1.0:
                    uncond_sample = build_t2i_text_sample(
                        " ", height, width, tokenizer, self.processor, model_config
                    )

            def to_device(s):
                return {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in s.items()}

            cond_sample = to_device(cond_sample)
            if uncond_sample is not None:
                uncond_sample = to_device(uncond_sample)

            is_editing = ref_paths is not None and len(ref_paths) == 1
            # official --editing_scheduler applies to DEV only ("Ignored for
            # full model"); the FULL model keeps UniPC for every task
            if (is_editing and model_type != "full"
                    and "scheduler_name" not in set(
                        getattr(sampling_params, "_explicit_fields", ()) or ())):
                if scheduler_name != "flow_match":
                    logger.info(
                        "Editing: scheduler %s -> flow_match (official "
                        "editing default)", scheduler_name)
                # official --editing_scheduler default is flow_match
                scheduler_name = "flow_match"
            if model_type == "full":
                sched = FlowUniPCScheduler(shift=shift)
                sched.set_timesteps(num_steps, device)
                inject_noise = False
            elif is_editing and scheduler_name == "flow_match":
                sched = FlowMatchScheduler(shift=shift)
                sched.set_timesteps_list(DEFAULT_TIMESTEPS, device)
                inject_noise = False
            else:
                if scheduler_name not in ("flash", "flow_match"):
                    logger.warning(
                        "Unsupported scheduler_name=%s for the DEV recipe; "
                        "using flash.", scheduler_name)
                sched = FlashFlowMatchScheduler()
                sched.set_timesteps_list(DEFAULT_TIMESTEPS, device)
                inject_noise = True
            num_steps = len(sched.timesteps)
            if inject_noise and num_steps > 1:
                noise_scale_schedule = [
                    noise_scale_start
                    + (noise_scale_end - noise_scale_start) * i / (num_steps - 1)
                    for i in range(num_steps)
                ]
            else:
                noise_scale_schedule = [noise_scale_start]

            init_noise_scale = noise_scale_start

            noise_note = (
                f" (per-step noise={noise_scale_start}->{noise_scale_end} "
                f"clip={noise_clip_std})" if inject_noise else ""
            )
            logger.info(
                "HiDream %s/%s: cfg=%s shift=%s steps=%d z_init=%s "
                "inject=%s%s seed=%s weights=%s",
                model_type, scheduler_name, guidance_scale, shift, num_steps,
                init_noise_scale,
                "per-step" if inject_noise else "off",
                noise_note, batch.seed, w_dtype,
            )

            tgt_image_len = h_patches * w_patches

            # All noise is drawn through a private CPU generator seeded with
            # this image's seed (official pipeline.py:315; CPU is the
            # reference-quality RNG on NPU).
            generator = torch.Generator("cpu")
            generator.manual_seed(int(seed))
            noise = init_noise_scale * torch.randn(
                (1, 3, height, width),
                generator=generator,
                device=generator.device,
                dtype=torch.float32,
            ).to(device, dtype)
            z = noise.reshape(1, 3, h_patches, PATCH_SIZE, w_patches, PATCH_SIZE)
            z = z.permute(0, 2, 4, 1, 3, 5).contiguous().reshape(
                1, h_patches * w_patches, -1)

            precomputed_image_embeds = None
            precomputed_deepstack_image_embeds = None
            for step_idx, step_t in enumerate(
                tqdm(sched.timesteps, total=num_steps, desc="Denoising")
            ):
                t_pixeldit = 1.0 - step_t.float() / 1000.0
                sigma = (step_t.float() / 1000.0).clamp_min(T_EPS)

                vinputs = (torch.cat([z, ref_patches], dim=1)
                           if ref_patches is not None else z.clone())

                with torch.autocast(
                    device_type=device.type, dtype=dtype, cache_enabled=False,
                ):
                    x_pred_cond, embeds = self._forward_once(
                        cond_sample, vinputs, t_pixeldit.to(device),
                        precomputed_image_embeds,
                        precomputed_deepstack_image_embeds,
                    )
                if embeds is not None:
                    precomputed_image_embeds, \
                        precomputed_deepstack_image_embeds = embeds
                x_tgt = x_pred_cond[:, :tgt_image_len]

                v_cond = (x_tgt.float() - z.float()) / sigma

                if uncond_sample is not None:
                    with torch.autocast(
                        device_type=device.type, dtype=dtype,
                        cache_enabled=False,
                    ):
                        x_pred_uncond, _ = self._forward_once(
                            uncond_sample, vinputs, t_pixeldit.to(device),
                            precomputed_image_embeds,
                            precomputed_deepstack_image_embeds,
                        )
                    x_tgt_uncond = x_pred_uncond[:, :tgt_image_len]
                    v_uncond = (x_tgt_uncond.float() - z.float()) / sigma
                    v_guided = v_uncond + guidance_scale * (v_cond - v_uncond)
                else:
                    v_guided = v_cond

                model_output = -v_guided
                if inject_noise:
                    z = sched.step(
                        model_output.float(), step_t.to(dtype=torch.float32),
                        z.float(), s_noise=noise_scale_schedule[step_idx],
                        noise_clip_std=noise_clip_std,
                        generator=generator).to(dtype)
                else:
                    z = sched.step(model_output.float(),
                                   step_t.to(dtype=torch.float32),
                                   z.float()).to(dtype)

            img = decode_patches_to_image(z, h_patches, w_patches)
            results.append(img)

        output = OutputBatch()
        if len(results) == 1:
            output.output = results[0]
        else:
            output.output = torch.cat(results, dim=0)
        return output

    def _forward_once(
        self, sample: dict, z_in: torch.Tensor, t_pixeldit: torch.Tensor,
        precomputed_image_embeds=None,
        precomputed_deepstack_image_embeds=None,
    ):
        """Run one forward pass. Returns (masked x_pred, cached vision embeds)."""
        device = z_in.device
        kwargs = {}
        if "pixel_values" in sample:
            kwargs["pixel_values"] = sample["pixel_values"].to(device)
            kwargs["image_grid_thw"] = sample["image_grid_thw"].to(device)
            if precomputed_image_embeds is not None:
                kwargs["precomputed_image_embeds"] = precomputed_image_embeds
                kwargs["precomputed_deepstack_image_embeds"] = (
                    precomputed_deepstack_image_embeds)
        outputs = self.model(
            input_ids=sample["input_ids"].to(device),
            position_ids=sample["position_ids"].to(device),
            vinputs=z_in,
            timestep=t_pixeldit.reshape(-1).to(device),
            token_types=sample["token_types"].to(device),
            use_flash_attn=True,  # official flash two-pass attention path
            **kwargs,
        )
        x_pred = outputs["x_pred"]
        vinput_mask = sample["vinput_mask"]
        x_masked = x_pred[0, vinput_mask[0]].unsqueeze(0)
        embeds = None
        if "cond_image_embeds" in outputs:
            embeds = (outputs["cond_image_embeds"],
                      outputs["cond_deepstack_image_embeds"])
        return x_masked, embeds
