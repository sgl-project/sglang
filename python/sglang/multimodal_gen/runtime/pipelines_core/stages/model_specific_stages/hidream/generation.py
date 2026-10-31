"""HiDream-O1-Image generation stage: preprocessing, denoising, and decoding."""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import torch
from PIL import Image
from tqdm.auto import tqdm

from sglang.multimodal_gen import envs
from sglang.multimodal_gen.runtime.cache.cache_dit_integration import (
    CacheDitConfig,
    cache_dit_overrides_key,
    disable_cache_on_transformer,
    enable_cache_on_transformer,
    get_scm_mask,
    refresh_context_on_transformer,
    resolve_cache_dit_request_overrides,
)
from sglang.multimodal_gen.runtime.disaggregation.roles import RoleType
from sglang.multimodal_gen.runtime.distributed import (
    get_sp_group,
    get_tp_group,
    get_world_size,
)
from sglang.multimodal_gen.runtime.models.dits.hidream import O1CacheBlockAdapter
from sglang.multimodal_gen.runtime.pipelines_core.schedule_batch import (
    OutputBatch,
    Req,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.base import PipelineStage
from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.hidream.layout import (
    create_layout_reference_images,
    load_layout_bboxes,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.hidream.prompt_enhance import (
    enhance_prompt,
)
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
        """One Euler step with per-step noise injection.

        ``noise`` (already drawn and clipped, e.g. per-row generators in a
        batched trajectory) replaces the internal draw; the legacy path is
        unchanged for direct callers and the parity suite.
        """
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
            device=device, dtype=torch.long
        )
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

    def multistep_uni_c_bh_update(
        self, this_model_output, last_sample, this_sample, order
    ):
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
            rhos_c = torch.stack(
                [
                    (b[0] - b[1]) / det,
                    (b[1] - r1 * b[0]) / det,
                ]
            ).to(dtype=x.dtype, device=device)

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

        use_corrector = self.step_index > 0 and self.last_sample is not None

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
            this_order = min(self.solver_order, len(self.timesteps) - self.step_index)
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


def draw_initial_noise(
    noise_scale: float,
    height: int,
    width: int,
    h_patches: int,
    w_patches: int,
    device: torch.device,
    dtype: torch.dtype,
    generator: torch.Generator,
) -> torch.Tensor:
    """(1, 3, H, W) scaled noise -> patch layout (1, h*w, 3*patch_size^2)."""
    noise = noise_scale * torch.randn(
        (1, 3, height, width),
        generator=generator,
        device=generator.device,
        dtype=torch.float32,
    ).to(device, dtype)
    z = noise.reshape(1, 3, h_patches, PATCH_SIZE, w_patches, PATCH_SIZE)
    return (
        z.permute(0, 2, 4, 1, 3, 5).contiguous().reshape(1, h_patches * w_patches, -1)
    )


def resize_pilimage(pil_image, image_size, patch_size=16, resampler=None):
    """Aspect-preserving resize + center crop (official utils.py)."""
    from PIL import Image as _Image

    if resampler is None:
        resampler = _Image.BICUBIC
    while min(*pil_image.size) >= 2 * image_size:
        pil_image = pil_image.resize(
            tuple(x // 2 for x in pil_image.size), resample=_Image.BOX
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
    import torchvision.transforms.v2 as transforms

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
    """Work items per image, prompt-major x num_outputs_per_prompt.

    Seeds come from ``batch.seeds`` (populated by InputValidationStage from
    per-request seeds or dynamic_batch_seeds); each prompt expands to
    ``num_outputs_per_prompt`` consecutive seeds. The list order is
    request-major so grouped-output splitting maps rows back correctly.
    """
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
    """Stack per-row text samples into one batched sample.

    Rows with equal text length stack directly (the unpadded fast path;
    bit-identical to single-row inference). Rows with different prompt
    lengths are rearranged to ``[text_wo_tms | pads | tms | vision]`` so
    the generation region (tms token + target patches) aligns across
    rows at identical offsets; a pad mask lets the model exclude padded
    keys from attention.
    """
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

        # [text_wo_tms | pads | tms] for the text region; the tms token
        # (last text token) moves after the pads so the gen region starts
        # at txt_max - 1 for every row
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
    """Partition work items into trajectory chunks of at most ``cap`` rows.

    Items sharing a group key (resolution + reference signature) are kept
    contiguous so each chunk holds same-shape rows.
    """
    groups: dict[tuple, list[dict]] = {}
    for item in items:
        groups.setdefault(item["group_key"], []).append(item)
    chunks = []
    for group in groups.values():
        for start in range(0, len(group), max(1, cap)):
            chunks.append(group[start : start + max(1, cap)])
    return chunks


def _register_o1_cache_dit_spec() -> None:
    """Teach cache-dit where the HiDream adapter keeps its blocks.

    The adapter class name deliberately matches no cache-dit built-in family
    prefix, so the registry has to be told about it explicitly.
    """
    from cache_dit import ForwardPattern

    from sglang.multimodal_gen.runtime.cache.cache_dit_integration import (
        _CUSTOM_BLOCK_ADAPTER_SPECS,
        CustomBlockAdapterSpec,
    )

    _CUSTOM_BLOCK_ADAPTER_SPECS.setdefault(
        O1CacheBlockAdapter.__name__,
        CustomBlockAdapterSpec(
            blocks_attr="blocks", forward_pattern=ForwardPattern.Pattern_3
        ),
    )


class HiDreamGenerationStage(PipelineStage):
    """Full generation stage: preprocessing -> denoising -> decoding."""

    def __init__(self, model: torch.nn.Module, processor: Any):
        super().__init__()
        self.model = model
        self.processor = processor
        # cache-dit mount state: the adapter wrapper is built once and stays
        # mounted while consecutive requests want the same knobs; the key is
        # the frozen per-request override set it was mounted with.
        self._cache_dit_adapter: O1CacheBlockAdapter | None = None
        self._cache_dit_enabled = False
        self._cache_dit_active_key: tuple | None = None
        self._cache_dit_request_overrides: dict = {}

    @property
    def role_affinity(self) -> RoleType:
        return RoleType.DENOISER

    def forward(self, batch: Req, server_args: ServerArgs) -> OutputBatch:
        prompts = batch.prompt if isinstance(batch.prompt, list) else [batch.prompt]
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
        elif not ref_images_batch:
            # Short prompts under-condition the model: output may come out
            # as uncolored line art. Official remedy: the Prompt Agent.
            for p in prompts:
                n = len(p.strip())
                if n < 20:  # arbitrary threshold
                    logger.warning(
                        "Short T2I prompt (%d chars): output may be "
                        "uncolored line art; refine it or enable "
                        "--pe-server-url.",
                        n,
                    )

        ctx = self._build_ctx(batch, server_args)
        work = _generation_work(batch, prompts)
        items = [
            self._prepare_item(item, batch, ctx, ref_images_batch) for item in work
        ]
        for i, item in enumerate(items):
            item["index"] = i

        chunks = _chunk_items(items, ctx["max_batched_outputs"])
        results: list = [None] * len(items)
        for chunk in chunks:
            for item, img in zip(chunk, self._run_trajectory(chunk, ctx)):
                results[item["index"]] = img

        # always 4-D [num_outputs, 3, H, W]: the grouped save/split paths
        # count outputs via the leading dim (the old serial contract kept
        # a size-1 batch dim for single images)
        output = OutputBatch()
        output.output = torch.stack(results, dim=0)
        return output

    def _build_ctx(self, batch: Req, server_args: ServerArgs) -> dict:
        """Shared per-request settings consumed by every work item."""
        first_param = next(self.model.parameters())
        sampling_params = getattr(batch, "sampling_params", None)
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
                    num_steps,
                    requested_steps,
                )
        guidance_scale = batch.guidance_scale
        if guidance_scale is None:
            guidance_scale = 0.0
        tokenizer = (
            self.processor.tokenizer
            if hasattr(self.processor, "tokenizer")
            else self.processor
        )
        pad_token_id = (
            getattr(tokenizer, "pad_token_id", None)
            or getattr(tokenizer, "eos_token_id", None)
            or 0
        )
        return {
            "device": first_param.device,
            "dtype": torch.bfloat16,
            "w_dtype": first_param.dtype,
            "num_steps": num_steps,
            "model_type": model_type,
            "guidance_scale": float(guidance_scale),
            "scheduler_name": (
                batch.scheduler_name if batch.scheduler_name is not None else "flash"
            ),
            "noise_scale_start": float(
                batch.noise_scale_start
                if batch.noise_scale_start is not None
                else NOISE_SCALE
            ),
            "noise_scale_end": float(
                batch.noise_scale_end
                if batch.noise_scale_end is not None
                else NOISE_SCALE
            ),
            "noise_clip_std": float(
                batch.noise_clip_std if batch.noise_clip_std is not None else 0.0
            ),
            "shift": float(batch.shift) if batch.shift is not None else 1.0,
            "layout_bboxes": batch.layout_bboxes,
            "explicit_fields": set(
                getattr(sampling_params, "_explicit_fields", ()) or ()
            ),
            "tokenizer": tokenizer,
            "model_config": self.model.config,
            "pad_token_id": int(pad_token_id),
            "max_batched_outputs": int(
                getattr(server_args.pipeline_config, "max_batched_outputs", 4) or 1
            ),
            "is_warmup": bool(getattr(batch, "is_warmup", False)),
            "enable_torch_compile": bool(
                getattr(server_args, "enable_torch_compile", False)
            ),
            "enable_breakable_cuda_graph": bool(
                getattr(server_args, "enable_breakable_cuda_graph", False)
            ),
            # cache-dit: per-request switch (env default when unset) plus knobs;
            # unknown knob names fail the request here, at the batch boundary
            "cache_dit_requested": self._cache_dit_requested_for_batch(batch),
            "cache_dit_overrides": resolve_cache_dit_request_overrides(
                getattr(sampling_params, "cache_dit_params", None)
            ),
        }

    def _prepare_item(
        self, item: dict, batch: Req, ctx: dict, ref_images_batch
    ) -> dict:
        """Per-image preparation: resolution snap, references, cond sample."""
        device = ctx["device"]
        dtype = ctx["dtype"]
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

            layout_data = None
            if ctx["layout_bboxes"] and preresized_ref_pil is None:
                try:
                    layout_data = load_layout_bboxes(ctx["layout_bboxes"])
                    K += 1
                except Exception as e:
                    logger.warning(
                        "Incorrect layout_bboxes: %s, %s", ctx["layout_bboxes"], e
                    )

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
                    ref_pils,
                    layout_data,
                    width,
                    height,
                    ref_max_size=max_size,
                    patch_size=PATCH_SIZE,
                )

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
                ctx["tokenizer"],
                self.processor,
                ctx["model_config"],
            )
        else:
            cond_sample = build_t2i_text_sample(
                prompt,
                height,
                width,
                ctx["tokenizer"],
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
                "group_key": (
                    height,
                    width,
                    tuple(ref_image_lens) if ref_image_lens else (),
                    tuple(ref_paths or ()),
                ),
            }
        )
        return item

    def _run_trajectory(self, chunk: list[dict], ctx: dict) -> list[torch.Tensor]:
        """Run one batched denoising trajectory for same-shape work items."""
        device = ctx["device"]
        dtype = ctx["dtype"]
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

        # scheduler selection (references are identical within a chunk)
        scheduler_name = ctx["scheduler_name"]
        model_type = ctx["model_type"]
        is_editing = ref_paths is not None and len(ref_paths) == 1
        # official --editing_scheduler applies to DEV only ("Ignored for
        # full model"); the FULL model keeps UniPC for every task
        if (
            is_editing
            and model_type != "full"
            and "scheduler_name" not in ctx["explicit_fields"]
        ):
            if scheduler_name != "flow_match":
                logger.info(
                    "Editing: scheduler %s -> flow_match (official editing default)",
                    scheduler_name,
                )
            # official --editing_scheduler default is flow_match
            scheduler_name = "flow_match"
        if model_type == "full":
            sched = FlowUniPCScheduler(shift=ctx["shift"])
            sched.set_timesteps(ctx["num_steps"], device)
            inject_noise = False
        elif is_editing and scheduler_name == "flow_match":
            sched = FlowMatchScheduler(shift=ctx["shift"])
            sched.set_timesteps_list(DEFAULT_TIMESTEPS, device)
            inject_noise = False
        else:
            if scheduler_name not in ("flash", "flow_match"):
                logger.warning(
                    "Unsupported scheduler_name=%s for the DEV recipe; using flash.",
                    scheduler_name,
                )
            sched = FlashFlowMatchScheduler()
            sched.set_timesteps_list(DEFAULT_TIMESTEPS, device)
            inject_noise = True
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
            "inject=%s%s rows=%d seeds=%s weights=%s",
            model_type,
            scheduler_name,
            guidance_scale,
            ctx["shift"],
            num_steps,
            init_noise_scale,
            "per-step" if inject_noise else "off",
            noise_note,
            len(chunk),
            [it["seed"] for it in chunk],
            ctx["w_dtype"],
        )

        # one private generator per row: draws stay per-sample so every
        # output keeps its own RNG stream (bit-identical to serial mode)
        gens = []
        z_rows = []
        for it in chunk:
            gen = torch.Generator(device)
            gen.manual_seed(int(it["seed"]))
            gens.append(gen)
            z_rows.append(
                draw_initial_noise(
                    init_noise_scale,
                    height,
                    width,
                    h_patches,
                    w_patches,
                    device,
                    dtype,
                    gen,
                )
            )
        z = torch.cat(z_rows, dim=0)

        # Deterministic TI2I (full-model UniPC, DEV flow_match) re-
        # injects no per-step noise, so the trajectory is fixed by
        # the initial draw; a bad draw must be re-rolled.
        reroll_enabled = not inject_noise and ref_paths is not None
        use_cache_dit = self._maybe_enable_cache_dit(ctx, num_steps, reroll_enabled)
        cache_adapter = self._cache_dit_adapter if use_cache_dit else None

        precomputed = (None, None)
        for step_idx, step_t in enumerate(
            tqdm(sched.timesteps, total=num_steps, desc=f"Denoising x{len(chunk)}")
        ):
            t_pixeldit = 1.0 - step_t.float() / 1000.0
            sigma = (step_t.float() / 1000.0).clamp_min(T_EPS)

            vinputs = (
                torch.cat([z, ref_patches], dim=1)
                if ref_patches is not None
                else z.clone()
            )

            with torch.autocast(
                device_type=device.type,
                dtype=dtype,
                cache_enabled=False,
            ):
                x_pred_cond, embeds = self._forward_batch(
                    cond, vinputs, t_pixeldit.to(device), precomputed, cache_adapter
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
                    dtype=dtype,
                    cache_enabled=False,
                ):
                    x_pred_uncond, _ = self._forward_batch(
                        uncond,
                        vinputs,
                        t_pixeldit.to(device),
                        precomputed,
                        cache_adapter,
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
                ).to(dtype)
            else:
                z = sched.step(
                    model_output.float(),
                    step_t.to(dtype=torch.float32),
                    z.float(),
                ).to(dtype)

        img = decode_patches_to_image(z, h_patches, w_patches)
        return [img[i] for i in range(len(chunk))]

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
                reroll_gen = torch.Generator(z.device)
                reroll_gen.manual_seed(reroll_seeds[i])
                z[i] = draw_initial_noise(
                    init_noise_scale,
                    height,
                    width,
                    h_patches,
                    w_patches,
                    z.device,
                    z.dtype,
                    reroll_gen,
                )[0]
            vinputs = (
                torch.cat([z, ref_patches], dim=1)
                if ref_patches is not None
                else z.clone()
            )
            with torch.autocast(
                device_type=z.device.type,
                dtype=z.dtype,
                cache_enabled=False,
            ):
                x_pred_sub, _ = self._forward_batch(
                    cond, vinputs, t_pixeldit, precomputed
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
        """Per-row flash-step noise; preserves each row's generator stream."""
        rows = []
        for gen in gens:
            n_i = torch.randn(
                (1, z.shape[1], z.shape[2]),
                device=z.device,
                dtype=torch.float32,
                generator=gen,
            )
            if noise_clip_std > 0:
                clip_val = noise_clip_std * n_i.std().item()
                n_i = n_i.clamp(min=-clip_val, max=clip_val)
            rows.append(n_i)
        return torch.cat(rows, dim=0)

    def _build_uncond_sample(self, first: dict, ctx: dict, batch_rows: int) -> dict:
        """One uncond sample for the chunk's refs/size, expanded to all rows."""
        if first["ref_paths"]:
            sample = build_ti2i_text_sample(
                " ",
                first["ref_pils_vlm"],
                first["ref_pils_resized"],
                first["height"],
                first["width"],
                first["ref_image_lens"],
                ctx["tokenizer"],
                self.processor,
                ctx["model_config"],
            )
        else:
            sample = build_t2i_text_sample(
                " ",
                first["height"],
                first["width"],
                ctx["tokenizer"],
                self.processor,
                ctx["model_config"],
            )
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

    def _cache_dit_requested(self) -> bool:
        """Request-independent server default (init-time decisions only)."""
        return envs.SGLANG_CACHE_DIT_ENABLED

    def _cache_dit_requested_for_batch(self, batch: Req) -> bool:
        """Per-request Cache-DiT switch; the server default applies when unset."""
        enable = batch.sampling_params.enable_cache_dit
        if enable is None:
            return self._cache_dit_requested()
        return bool(enable)

    def _cache_dit_has_separate_cfg(self, ctx: dict) -> bool:
        """Whether CFG runs as separate cond/uncond forwards per step."""
        return ctx["guidance_scale"] > 1.0

    def _unmount_cache_dit(self) -> None:
        """Remove Cache-DiT hooks so subsequent batches run the native forward."""
        if self._cache_dit_adapter is not None:
            disable_cache_on_transformer(self._cache_dit_adapter)
        self._cache_dit_adapter = None
        self._cache_dit_enabled = False
        self._cache_dit_active_key = None

    def _parse_cache_dit_scm_bins(
        self,
    ) -> tuple[list[int] | None, list[int] | None, str]:
        overrides = self._cache_dit_request_overrides
        scm_preset = overrides.get("scm_preset", envs.SGLANG_CACHE_DIT_SCM_PRESET)
        request_compute_bins = overrides.get("scm_compute_bins")
        request_cache_bins = overrides.get("scm_cache_bins")
        if request_compute_bins is not None or request_cache_bins is not None:
            if request_compute_bins is None or request_cache_bins is None:
                raise ValueError(
                    "cache_dit_params SCM custom bins require both "
                    "scm_compute_bins and scm_cache_bins."
                )
            compute_bins = [int(x) for x in request_compute_bins]
            cache_bins = [int(x) for x in request_cache_bins]
            return compute_bins, cache_bins, scm_preset
        compute_bins_str = envs.SGLANG_CACHE_DIT_SCM_COMPUTE_BINS
        cache_bins_str = envs.SGLANG_CACHE_DIT_SCM_CACHE_BINS
        compute_bins = None
        cache_bins = None
        if compute_bins_str and cache_bins_str:
            try:
                compute_bins = [int(x.strip()) for x in compute_bins_str.split(",")]
                cache_bins = [int(x.strip()) for x in cache_bins_str.split(",")]
            except ValueError as exc:
                logger.warning("Failed to parse SCM bins: %s. SCM disabled.", exc)
                scm_preset = "none"
        elif compute_bins_str or cache_bins_str:
            logger.warning(
                "SCM custom bins require both compute_bins and cache_bins. "
                "Only one was provided (compute=%s, cache=%s). Falling back to preset '%s'.",
                compute_bins_str,
                cache_bins_str,
                scm_preset,
            )
        return compute_bins, cache_bins, scm_preset

    def _cache_dit_scm_masks(
        self, num_inference_steps: int
    ) -> tuple[str, str, list[int] | None]:
        scm_compute_bins, scm_cache_bins, scm_preset = self._parse_cache_dit_scm_bins()
        scm_policy = self._cache_dit_request_overrides.get(
            "scm_policy", envs.SGLANG_CACHE_DIT_SCM_POLICY
        )
        steps_computation_mask = get_scm_mask(
            preset=scm_preset,
            num_inference_steps=num_inference_steps,
            compute_bins=scm_compute_bins,
            cache_bins=scm_cache_bins,
        )
        return scm_preset, scm_policy, steps_computation_mask

    def _cache_dit_knob(self, key: str, env_value: Any) -> Any:
        """One DBCache knob: request > env."""
        return self._cache_dit_request_overrides.get(key, env_value)

    def _build_cache_dit_config(
        self,
        num_inference_steps: int,
        steps_computation_mask: list[int] | None,
        scm_policy: str,
    ) -> CacheDitConfig:
        knob = self._cache_dit_knob
        return CacheDitConfig(
            enabled=True,
            Fn_compute_blocks=knob("Fn_compute_blocks", envs.SGLANG_CACHE_DIT_FN),
            Bn_compute_blocks=knob("Bn_compute_blocks", envs.SGLANG_CACHE_DIT_BN),
            max_warmup_steps=knob("max_warmup_steps", envs.SGLANG_CACHE_DIT_WARMUP),
            residual_diff_threshold=knob(
                "residual_diff_threshold", envs.SGLANG_CACHE_DIT_RDT
            ),
            max_continuous_cached_steps=knob(
                "max_continuous_cached_steps", envs.SGLANG_CACHE_DIT_MC
            ),
            enable_taylorseer=knob(
                "enable_taylorseer", envs.SGLANG_CACHE_DIT_TAYLORSEER
            ),
            taylorseer_order=knob("taylorseer_order", envs.SGLANG_CACHE_DIT_TS_ORDER),
            enable_dmd=knob("enable_dmd", envs.SGLANG_CACHE_DIT_DMD),
            dmd_history=knob("dmd_history", envs.SGLANG_CACHE_DIT_DMD_HISTORY),
            dmd_rank=knob("dmd_rank", envs.SGLANG_CACHE_DIT_DMD_RANK),
            dmd_ridge=knob("dmd_ridge", envs.SGLANG_CACHE_DIT_DMD_RIDGE),
            dmd_svd_precision=knob(
                "dmd_svd_precision", envs.SGLANG_CACHE_DIT_DMD_SVD_PRECISION
            ),
            num_inference_steps=num_inference_steps,
            steps_computation_mask=steps_computation_mask,
            steps_computation_policy=scm_policy,
        )

    def _maybe_enable_cache_dit(
        self, ctx: dict, num_steps: int, reroll_enabled: bool
    ) -> bool:
        """Enable cache-dit on the decoder loop for this trajectory (idempotent).

        Returns whether the step loop should route its forwards through the
        adapter. HiDream is not a diffusers DiT, so the block loop is exposed
        through O1CacheBlockAdapter (see its docstring for the naming
        constraint) and SGLANG_CACHE_DIT_* apply as for the other models.

        Not mounted while the deterministic TI2I reroll is active: the reroll
        issues extra step-0 forwards, which would desynchronise cache-dit's
        step counter.
        """
        requested = ctx["cache_dit_requested"]
        if ctx["enable_breakable_cuda_graph"]:
            # cache-dit wraps the block loop with step-skipping control flow
            # that must not be baked into a captured CUDA graph
            if requested:
                logger.warning_once(
                    "Cache-DiT was requested but is disabled because breakable "
                    "CUDA graphs are enabled."
                )
            return False
        self._cache_dit_request_overrides = ctx["cache_dit_overrides"]
        has_separate_cfg = requested and self._cache_dit_has_separate_cfg(ctx)
        desired_key = (
            (
                cache_dit_overrides_key(self._cache_dit_request_overrides),
                has_separate_cfg,
            )
            if requested
            else None
        )
        # opt-out or knob change: unmount, then remount below (or return)
        if self._cache_dit_enabled and desired_key != self._cache_dit_active_key:
            self._unmount_cache_dit()
        # NOTE: every chunk runs its own num_steps loop, so a mounted adapter
        # only needs its step counter refreshed per request.
        if self._cache_dit_enabled:
            _, scm_policy, steps_computation_mask = self._cache_dit_scm_masks(num_steps)
            refresh_context_on_transformer(
                self._cache_dit_adapter,
                num_steps,
                steps_computation_mask=steps_computation_mask,
                steps_computation_policy=scm_policy,
            )
            return True

        if not requested:
            return False
        # Keep cache-dit disabled for ordinary warmup, but allow torch.compile
        # warmup to mount cache-dit before Dynamo traces the transformer.
        if ctx["is_warmup"] and not ctx["enable_torch_compile"]:
            return False
        if reroll_enabled:
            logger.info(
                "cache-dit skipped for this trajectory: the deterministic TI2I "
                "seed reroll adds step-0 forwards."
            )
            return False

        world_size = get_world_size()
        parallelized = world_size > 1

        sp_group = None
        tp_group = None
        if parallelized:
            sp_group_candidate = get_sp_group()
            tp_group_candidate = get_tp_group()

            sp_world_size = sp_group_candidate.world_size if sp_group_candidate else 1
            tp_world_size = tp_group_candidate.world_size if tp_group_candidate else 1

            has_sp = sp_world_size > 1
            has_tp = tp_world_size > 1

            sp_group = sp_group_candidate.device_group if has_sp else None
            tp_group = tp_group_candidate.device_group if has_tp else None

            logger.info(
                "cache-dit enabled in distributed environment (world_size=%d, has_sp=%s, has_tp=%s)",
                world_size,
                has_sp,
                has_tp,
            )

        _, scm_policy, steps_computation_mask = self._cache_dit_scm_masks(num_steps)
        config = self._build_cache_dit_config(
            num_steps,
            steps_computation_mask=steps_computation_mask,
            scm_policy=scm_policy,
        )

        _register_o1_cache_dit_spec()
        adapter = O1CacheBlockAdapter(self.model)
        self._cache_dit_adapter = enable_cache_on_transformer(
            adapter,
            config,
            model_name="hidream_o1",
            sp_group=sp_group,
            tp_group=tp_group,
            has_separate_cfg=has_separate_cfg,
        )
        logger.info(
            "cache-dit enabled on transformer (steps=%d, Fn=%d, Bn=%d, rdt=%.3f)",
            num_steps,
            config.Fn_compute_blocks,
            config.Bn_compute_blocks,
            config.residual_diff_threshold,
        )
        self._cache_dit_enabled = True
        self._cache_dit_active_key = desired_key
        return True

    def _forward_batch(
        self,
        sample: dict,
        vinputs: torch.Tensor,
        t_pixeldit: torch.Tensor,
        precomputed: tuple,
        cache_adapter: O1CacheBlockAdapter | None = None,
    ) -> tuple[torch.Tensor, tuple | None]:
        """Run one (possibly batched) forward; returns (x_pred, embeds).

        ``cache_adapter`` routes just the decoder block loop through cache-dit;
        the embeddings, final norm and patch head still run in the model, so
        step-skipping never touches the request-static conditioning.
        """
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

        if cache_adapter is None:
            outputs = self.model(
                **common_kwargs,
                use_flash_attn=True,  # official flash two-pass attention path
                pad_mask=sample.get("pad_mask"),
                **vision_kwargs,
            )
        else:
            inputs_embeds, conds, embeds_pair = self.model.forward_pre(
                **common_kwargs,
                **vision_kwargs,
            )
            hidden_states = cache_adapter(
                inputs_embeds,
                conds["cos"],
                conds["sin"],
                conds["idx_ar"],
                sample.get("pad_mask"),
                conds["visual_pos_masks"],
                conds["deepstack_visual_embeds"],
            )
            outputs = self.model.forward_post(hidden_states, apply_norm=True)
            if pixel_values is not None:
                outputs["cond_image_embeds"] = embeds_pair[0]
                outputs["cond_deepstack_image_embeds"] = embeds_pair[1]

        embeds = None
        if "cond_image_embeds" in outputs:
            embeds = (
                outputs["cond_image_embeds"],
                outputs["cond_deepstack_image_embeds"],
            )
        return outputs["x_pred"], embeds

    def run_grouped_requests(self, batches: list[Req], server_args: ServerArgs) -> list:
        """Co-batch trajectory groups across requests (offline grouped path).

        Partial grouped reuse per StageDedupMixin: requests sharing a
        trajectory fingerprint run through one batched forward each step,
        while single-request partitions keep the plain per-request path.
        """
        if len(batches) <= 1:
            return [self(batch, server_args) for batch in batches]
        groups = self._group_requests_by_fingerprint(
            batches,
            lambda batch: self.build_dedup_fingerprint(batch, server_args),
        )
        results: list = [None] * len(batches)
        for _, group in groups:
            indices = [index for index, _ in group]
            if len(indices) == 1:
                results[indices[0]] = self(batches[indices[0]], server_args)
                continue
            self._forward_multi(batches, indices, server_args, results)
        return results

    def build_dedup_fingerprint(self, batch: Req, server_args: ServerArgs) -> tuple:
        """Fields that must match for two requests to share a trajectory."""
        sampling_params = getattr(batch, "sampling_params", None)
        requested_steps = int(
            getattr(sampling_params, "num_inference_steps", None)
            or getattr(batch, "num_inference_steps", None)
            or getattr(server_args.pipeline_config, "num_inference_steps", None)
            or len(DEFAULT_TIMESTEPS)
        )
        model_type = batch.model_type if batch.model_type is not None else "dev"
        num_steps = requested_steps if model_type == "full" else len(DEFAULT_TIMESTEPS)
        return (
            int(batch.height),
            int(batch.width),
            model_type,
            num_steps,
            float(batch.guidance_scale or 0.0),
            batch.scheduler_name,
            float(batch.shift or 1.0),
            float(
                batch.noise_scale_start
                if batch.noise_scale_start is not None
                else NOISE_SCALE
            ),
            float(
                batch.noise_scale_end
                if batch.noise_scale_end is not None
                else NOISE_SCALE
            ),
            float(batch.noise_clip_std if batch.noise_clip_std is not None else 0.0),
            bool(batch.image_path),
            self.freeze_for_dedup(batch.layout_bboxes),
            tuple(sorted(getattr(sampling_params, "_explicit_fields", ()) or ())),
            int(getattr(server_args.pipeline_config, "max_batched_outputs", 4) or 1),
        )

    def _forward_multi(
        self,
        batches: list[Req],
        indices: list[int],
        server_args: ServerArgs,
        results: list,
    ) -> None:
        """Run co-batched trajectories for same-fingerprint requests."""
        first = batches[indices[0]]
        ctx = self._build_ctx(first, server_args)
        items = []
        owner = []
        for idx in indices:
            batch = batches[idx]
            prompts = batch.prompt if isinstance(batch.prompt, list) else [batch.prompt]
            ref_images_batch = getattr(batch, "image_path", None)
            if (
                server_args.pe_server_url
                and not ref_images_batch
                and not getattr(batch, "is_warmup", False)
            ):
                prompts = [
                    enhance_prompt(prompt=p, base_url=server_args.pe_server_url)
                    for p in prompts
                ]
            for item in _generation_work(batch, prompts):
                items.append(self._prepare_item(item, batch, ctx, ref_images_batch))
                owner.append(idx)
        for i, item in enumerate(items):
            item["index"] = i
        imgs: list = [None] * len(items)
        for chunk in _chunk_items(items, ctx["max_batched_outputs"]):
            for item, img in zip(chunk, self._run_trajectory(chunk, ctx)):
                imgs[item["index"]] = img
        per_batch: dict[int, list] = {}
        for idx, img in zip(owner, imgs):
            per_batch.setdefault(idx, []).append(img)
        for idx in indices:
            rows = per_batch[idx]
            output = OutputBatch()
            # always 4-D [num_outputs, 3, H, W] (see forward())
            output.output = torch.stack(rows, dim=0)
            results[idx] = output
