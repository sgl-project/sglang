import inspect
import re
import time
from typing import List, Optional, Tuple, Union

import numpy as np
import PIL
import requests
import torch
from diffusers.image_processor import VaeImageProcessor
from diffusers.utils.torch_utils import randn_tensor
from torch.nn.utils.rnn import pad_sequence

from sglang.multimodal_gen.runtime.distributed import get_local_torch_device
from sglang.multimodal_gen.runtime.managers.forward_context import set_forward_context
from sglang.multimodal_gen.runtime.managers.memory_managers.component_manager import (
    ComponentUse,
)
from sglang.multimodal_gen.runtime.models.dits.glm_image import GlmImageKVCache
from sglang.multimodal_gen.runtime.pipelines_core.schedule_batch import Req
from sglang.multimodal_gen.runtime.pipelines_core.stages.base import (
    PipelineStage,
    StageParallelismType,
)
from sglang.multimodal_gen.runtime.server_args import ServerArgs
from sglang.multimodal_gen.runtime.utils.logging_utils import init_logger
from sglang.multimodal_gen.runtime.utils.precision import (
    align_tensor_to_module_dtype,
    get_module_dtype,
)
from sglang.multimodal_gen.runtime.utils.vision import load_image

logger = init_logger(__name__)


def calculate_shift(
    image_seq_len,
    base_seq_len: int = 256,
    base_shift: float = 0.25,
    max_shift: float = 0.75,
) -> float:
    m = (image_seq_len / base_seq_len) ** 0.5
    mu = m * max_shift + base_shift
    return mu


def retrieve_timesteps(
    scheduler,
    num_inference_steps: Optional[int] = None,
    device: Optional[Union[str, torch.device]] = None,
    timesteps: Optional[List[int]] = None,
    sigmas: Optional[List[float]] = None,
    **kwargs,
):
    r"""
    Calls the scheduler's `set_timesteps` method and retrieves timesteps from the scheduler after the call. Handles
    custom timesteps. Any kwargs will be supplied to `scheduler.set_timesteps`.
    """
    accepts_timesteps = "timesteps" in set(
        inspect.signature(scheduler.set_timesteps).parameters.keys()
    )
    accepts_sigmas = "sigmas" in set(
        inspect.signature(scheduler.set_timesteps).parameters.keys()
    )

    if timesteps is not None and sigmas is not None:
        if not accepts_timesteps and not accepts_sigmas:
            raise ValueError(
                f"The current scheduler class {scheduler.__class__}'s `set_timesteps` does not support custom"
                f" timestep or sigma schedules. Please check whether you are using the correct scheduler."
            )
        scheduler.set_timesteps(
            timesteps=timesteps, sigmas=sigmas, device=device, **kwargs
        )
        timesteps = scheduler.timesteps
        num_inference_steps = len(timesteps)
    elif timesteps is not None and sigmas is None:
        if not accepts_timesteps:
            raise ValueError(
                f"The current scheduler class {scheduler.__class__}'s `set_timesteps` does not support custom"
                f" timestep schedules. Please check whether you are using the correct scheduler."
            )
        scheduler.set_timesteps(timesteps=timesteps, device=device, **kwargs)
        timesteps = scheduler.timesteps
        num_inference_steps = len(timesteps)
    elif timesteps is None and sigmas is not None:
        if not accepts_sigmas:
            raise ValueError(
                f"The current scheduler class {scheduler.__class__}'s `set_timesteps` does not support custom"
                f" sigmas schedules. Please check whether you are using the correct scheduler."
            )
        scheduler.set_timesteps(sigmas=sigmas, device=device, **kwargs)
        timesteps = scheduler.timesteps
        num_inference_steps = len(timesteps)
    else:
        scheduler.set_timesteps(num_inference_steps, device=device, **kwargs)
        timesteps = scheduler.timesteps
    return timesteps, num_inference_steps


# Copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion_img2img.retrieve_latents
def retrieve_latents(
    encoder_output: torch.Tensor,
    generator: Optional[torch.Generator] = None,
    sample_mode: str = "sample",
):
    if hasattr(encoder_output, "latent_dist") and sample_mode == "sample":
        return encoder_output.latent_dist.sample(generator)
    elif hasattr(encoder_output, "latent_dist") and sample_mode == "argmax":
        return encoder_output.latent_dist.mode()
    elif hasattr(encoder_output, "latents"):
        return encoder_output.latents
    else:
        raise AttributeError("Could not access latents of provided encoder_output")


def image_path_to_list(image_path: Union[str, List[str]]) -> List[str]:
    return image_path if isinstance(image_path, list) else [image_path]


def pooled_image_features_to_tensor(image_features) -> torch.Tensor:
    pooler_output = getattr(image_features, "pooler_output", None)
    if pooler_output is not None:
        image_features = pooler_output
    if isinstance(image_features, torch.Tensor):
        return image_features
    return torch.cat(tuple(image_features), dim=0)


def _expand_prompts_and_seeds(batch: Req) -> tuple[list[str], list[int]]:
    prompts = batch.prompt if isinstance(batch.prompt, list) else [batch.prompt]
    num_outputs = int(batch.num_outputs_per_prompt)
    dynamic_seeds = batch.extra.get("dynamic_batch_seeds")

    if dynamic_seeds is not None:
        if len(dynamic_seeds) != len(prompts):
            raise ValueError("dynamic_batch_seeds must contain one seed per prompt")
        base_seeds = [int(seed) for seed in dynamic_seeds]
    elif isinstance(batch.seed, list):
        if len(prompts) != 1 or len(batch.seed) != num_outputs:
            raise ValueError(
                "seed list must contain one seed per output for a single prompt"
            )
        return [prompts[0]] * num_outputs, [int(seed) for seed in batch.seed]
    else:
        base_seed = int(batch.seed)
        base_seeds = [
            base_seed + prompt_index * num_outputs
            for prompt_index in range(len(prompts))
        ]

    expanded_prompts = []
    expanded_seeds = []
    for prompt, base_seed in zip(prompts, base_seeds, strict=True):
        for output_index in range(num_outputs):
            expanded_prompts.append(prompt)
            expanded_seeds.append(base_seed + output_index)
    return expanded_prompts, expanded_seeds


class GlmImageAR(PipelineStage):
    r"""
    Pipeline for text-to-image generation using GLM-Image.

    This stage for the AR (autoregressive) model for token generation.

    Args:
        processor (`AutoProcessor`):
            Processor for the AR model to handle chat templates and tokenization.
        vision_language_encoder ([`GlmImageForConditionalGeneration`]):
            The AR model that generates image tokens from text prompts.
    """

    def __init__(
        self,
        processor,
        vision_language_encoder,
    ):
        super().__init__()
        self.processor = processor
        self.vision_language_encoder = vision_language_encoder

    @property
    def parallelism_type(self) -> StageParallelismType:
        return StageParallelismType.MAIN_RANK_ONLY_AND_SEND_TO_OTHERS

    @staticmethod
    def _compute_generation_params(
        image_grid_thw,
        is_text_to_image: bool,
    ):
        grid_sizes = []
        grid_hw = []

        for i in range(image_grid_thw.shape[0]):
            t, h, w = image_grid_thw[i].tolist()
            grid_sizes.append(int(h * w))
            grid_hw.append((int(h), int(w)))

        if not is_text_to_image:
            max_new_tokens = grid_sizes[-1] + 1
            large_image_start_offset = 0
            target_grid_h, target_grid_w = grid_hw[-1]
        else:
            total_tokens = sum(grid_sizes)
            max_new_tokens = total_tokens + 1
            large_image_start_offset = sum(grid_sizes[1:])
            target_grid_h, target_grid_w = grid_hw[0]
        return max_new_tokens, large_image_start_offset, target_grid_h, target_grid_w

    @staticmethod
    def _upsample_token_ids(
        token_ids: torch.Tensor, token_h: int, token_w: int
    ) -> torch.Tensor:
        token_ids = token_ids.view(1, 1, token_h, token_w)
        token_ids = torch.nn.functional.interpolate(
            token_ids.float(), scale_factor=2, mode="nearest"
        ).to(dtype=torch.long)
        token_ids = token_ids.view(1, -1)
        return token_ids

    def generate_prior_tokens(
        self,
        prompt: str,
        height: int,
        width: int,
        server_args: ServerArgs,
        image: Optional[List[PIL.Image.Image]] = None,
        factor: int = 32,
        seed: Optional[int] = None,
    ) -> Tuple[torch.Tensor, int, int]:
        """
        Generate prior tokens using the AR (vision_language_encoder) model.

        Args:
            prompt: The text prompt with shape info (e.g., "description<sop>36 24<eop>")
            condition_images: Optional list of condition images for i2i

        Returns:
            Tuple of (prior_token_ids, pixel_height, pixel_width)
            - prior_token_ids: Upsampled to d16 format, shape [1, token_h*token_w*4]
            - pixel_height: Image height in pixels
            - pixel_width: Image width in pixels
        """
        device = get_local_torch_device()
        height = (height // factor) * factor
        width = (width // factor) * factor

        is_text_to_image = image is None or len(image) == 0
        # Build messages for processor
        content = []
        if image is not None:
            for img in image:
                content.append({"type": "image", "image": img})
        content.append({"type": "text", "text": prompt})
        messages = [{"role": "user", "content": content}]

        inputs = self.processor.apply_chat_template(
            messages,
            tokenize=True,
            target_h=height,
            target_w=width,
            return_dict=True,
            return_tensors="pt",
        ).to(device)

        image_grid_thw = inputs.get("image_grid_thw")
        max_new_tokens, large_image_offset, token_h, token_w = (
            self._compute_generation_params(
                image_grid_thw=image_grid_thw, is_text_to_image=is_text_to_image
            )
        )

        prior_token_image_ids = None

        # For GLM-Image, greedy decoding is not allowed; it may cause repetitive outputs.
        # max_new_tokens must be exactly grid_h * grid_w + 1 (the +1 is for EOS).
        if server_args.srt_encoder_url is not None:
            if image is not None:
                logger.error(
                    "Image-to-Image tasks is not supported yet when using an external SGLang encoder server."
                )
                raise NotImplementedError(
                    "I2I mode is not supported yet via external SGLang encoder URL."
                )

            payload = {
                "input_ids": inputs["input_ids"][0].tolist(),
                "image_data": [{"image_grid_thw": image_grid_thw.tolist()}],
                "sampling_params": {
                    "temperature": 1.0,
                    "max_new_tokens": max_new_tokens,
                    "ignore_eos": True,
                    "sampling_seed": seed,
                },
            }
            try:
                response = requests.post(
                    server_args.srt_encoder_url + "/generate",
                    json=payload,
                    timeout=(
                        server_args.srt_encoder_connect_timeout,
                        server_args.srt_encoder_timeout,
                    ),
                )
            except requests.ConnectionError as e:
                logger.error(
                    "Failed to establish a connection to SGLang encoder server at %s. "
                    "Verify that the AR model server is running and accessible. Error details: %s",
                    server_args.srt_encoder_url,
                    e,
                )
                raise
            except requests.ConnectTimeout as e:
                logger.error(
                    "Connection timeout to SGLang encoder (%s). Try to increase --srt-encoder-connection-timeout (current: %s sec). Details: %s",
                    server_args.srt_encoder_url,
                    server_args.srt_encoder_connect_timeout,
                    e,
                )
                raise
            except requests.ReadTimeout as e:
                logger.error(
                    "Read timeout from SGLang encoder (%s). Try to increase --srt-encoder-timeout (current: %s sec). Details: %s",
                    server_args.srt_encoder_url,
                    server_args.srt_encoder_timeout,
                    e,
                )
                raise
            except requests.RequestException as e:
                logger.error(
                    "An error occurred during communication with SGLang encoder server at %s. "
                    "The server is reachable, but the request failed. Error type: %s, Details: %s",
                    server_args.srt_encoder_url,
                    type(e).__name__,
                    e,
                )
                raise

            data = response.json()
            generated_ids = data.get("output_ids")
        else:
            if image is not None:
                source_grids = image_grid_thw[:-1]
                prior_token_image_embed = pooled_image_features_to_tensor(
                    self.vision_language_encoder.get_image_features(
                        inputs["pixel_values"], source_grids
                    )
                )
                prior_token_image_ids_d32 = (
                    self.vision_language_encoder.get_image_tokens(
                        prior_token_image_embed, source_grids
                    )
                )
                prior_token_image_ids = []
                prior_ids_per_source = torch.split(
                    prior_token_image_ids_d32,
                    source_grids.prod(dim=-1).tolist(),
                )
                for prior_ids, source_grid in zip(prior_ids_per_source, source_grids):
                    _, source_h, source_w = source_grid.tolist()
                    prior_token_image_ids.append(
                        self._upsample_token_ids(
                            prior_ids,
                            int(source_h),
                            int(source_w),
                        ).squeeze(0)
                    )
            outputs = self.vision_language_encoder.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=True,
            )
            input_len = inputs["input_ids"].shape[-1]
            generated_ids = outputs[0][input_len:]

        expected_output_len = large_image_offset + token_h * token_w
        actual_output_len = 0 if generated_ids is None else len(generated_ids)
        if actual_output_len < expected_output_len:
            raise RuntimeError(
                "GLM-Image AR returned too few output_ids: "
                f"got {actual_output_len}, need at least {expected_output_len} "
                f"(large_image_offset={large_image_offset}, "
                f"token_h={token_h}, token_w={token_w})."
            )

        # Extract large image tokens + upsample D32→D16
        prior_token_ids_d32 = torch.tensor(
            generated_ids[large_image_offset : large_image_offset + token_h * token_w],
            device=device,
        )
        prior_token_ids = self._upsample_token_ids(
            prior_token_ids_d32, token_h, token_w
        )

        return prior_token_ids, prior_token_image_ids

    def generate_prior_tokens_batch(
        self,
        prompts: list[str],
        seeds: list[int],
        height: int,
        width: int,
        server_args: ServerArgs,
        factor: int = 32,
    ) -> list[torch.Tensor]:
        device = get_local_torch_device()
        height = (height // factor) * factor
        width = (width // factor) * factor

        input_ids = []
        image_data = []
        sampling_params = []
        generation_shapes = []
        for prompt, seed in zip(prompts, seeds, strict=True):
            messages = [
                {
                    "role": "user",
                    "content": [{"type": "text", "text": prompt}],
                }
            ]
            inputs = self.processor.apply_chat_template(
                messages,
                tokenize=True,
                target_h=height,
                target_w=width,
                return_dict=True,
                return_tensors="pt",
            ).to(device)
            image_grid_thw = inputs.get("image_grid_thw")
            max_new_tokens, large_image_offset, token_h, token_w = (
                self._compute_generation_params(
                    image_grid_thw=image_grid_thw,
                    is_text_to_image=True,
                )
            )
            input_ids.append(inputs["input_ids"][0].tolist())
            image_data.append([{"image_grid_thw": image_grid_thw.tolist()}])
            sampling_params.append(
                {
                    "temperature": 1.0,
                    "max_new_tokens": max_new_tokens,
                    "ignore_eos": True,
                    "sampling_seed": int(seed),
                }
            )
            generation_shapes.append((large_image_offset, token_h, token_w))

        payload = {
            "input_ids": input_ids,
            "image_data": image_data,
            "sampling_params": sampling_params,
        }
        try:
            response = requests.post(
                server_args.srt_encoder_url + "/generate",
                json=payload,
                timeout=(
                    server_args.srt_encoder_connect_timeout,
                    server_args.srt_encoder_timeout,
                ),
            )
            response.raise_for_status()
        except requests.RequestException as e:
            logger.error(
                "GLM-Image batched AR request to %s failed: %s",
                server_args.srt_encoder_url,
                e,
            )
            raise

        data = response.json()
        if not isinstance(data, list) or len(data) != len(prompts):
            raise RuntimeError(
                "GLM-Image AR batch returned an unexpected response: "
                f"expected {len(prompts)} outputs, got "
                f"{len(data) if isinstance(data, list) else type(data).__name__}."
            )

        prior_token_ids = []
        for item, (large_image_offset, token_h, token_w) in zip(
            data, generation_shapes, strict=True
        ):
            generated_ids = item.get("output_ids")
            expected_output_len = large_image_offset + token_h * token_w
            actual_output_len = 0 if generated_ids is None else len(generated_ids)
            if actual_output_len < expected_output_len:
                raise RuntimeError(
                    "GLM-Image AR returned too few output_ids: "
                    f"got {actual_output_len}, need at least {expected_output_len}."
                )
            prior_token_ids_d32 = torch.tensor(
                generated_ids[
                    large_image_offset : large_image_offset + token_h * token_w
                ],
                device=device,
            )
            prior_token_ids.append(
                self._upsample_token_ids(prior_token_ids_d32, token_h, token_w)
            )
        return prior_token_ids

    @torch.no_grad()
    def forward(
        self,
        batch: Req,
        server_args: ServerArgs,
    ) -> Req:

        prompts, seeds = _expand_prompts_and_seeds(batch)
        height = batch.height
        width = batch.width
        if batch.image_path is not None:
            ar_condition_images = [
                load_image(img_path)
                for img_path in image_path_to_list(batch.image_path)
            ]
        else:
            ar_condition_images = None

        device = get_local_torch_device()

        if ar_condition_images is not None:
            height = height or ar_condition_images[0].height
            width = width or ar_condition_images[0].width

        time_start = time.time()
        if server_args.srt_encoder_url is not None and ar_condition_images is None:
            prior_token_ids = self.generate_prior_tokens_batch(
                prompts=prompts,
                seeds=seeds,
                height=height,
                width=width,
                server_args=server_args,
            )
            prior_token_image_ids = [None] * len(prior_token_ids)
        else:
            prior_token_ids = []
            prior_token_image_ids = []
            for prompt, seed in zip(prompts, seeds, strict=True):
                if seed is None:
                    prior_token_id, image_token_ids = self.generate_prior_tokens(
                        prompt=prompt,
                        image=ar_condition_images,
                        height=height,
                        width=width,
                        server_args=server_args,
                    )
                else:
                    rng_devices = []
                    if device.type == "cuda":
                        rng_devices.append(torch.cuda.current_device())
                    with torch.random.fork_rng(devices=rng_devices, enabled=True):
                        torch.manual_seed(int(seed))
                        prior_token_id, image_token_ids = self.generate_prior_tokens(
                            prompt=prompt,
                            image=ar_condition_images,
                            height=height,
                            width=width,
                            server_args=server_args,
                            seed=seed,
                        )
                prior_token_ids.append(prior_token_id)
                prior_token_image_ids.append(image_token_ids)

        prior_token_id = torch.cat(prior_token_ids, dim=0).to(device=device)
        if all(item is None for item in prior_token_image_ids):
            prior_token_image_ids = None
        elif len(prior_token_image_ids) == 1:
            prior_token_image_ids = prior_token_image_ids[0]
        time_end = time.time()
        logger.info(f"generate_prior_tokens time: {time_end - time_start}")

        batch.prior_token_id = prior_token_id
        batch.prior_token_image_ids = prior_token_image_ids
        batch.height = height
        batch.width = width

        return batch


class GlmImageBeforeDenoisingStage(PipelineStage):
    r"""
    Pipeline for text-to-image generation using GLM-Image.

    This stage for preparations before denoising stage like encoding, latents, timesteps

    Args:
        vae ([`AutoencoderKL`]):
            Variational Auto-Encoder (VAE) Model to encode and decode images to and from latent representations.
        text_encoder ([`T5EncoderModel`]):
            Frozen text-encoder for glyph embeddings.
        tokenizer (`PreTrainedTokenizer`):
            Tokenizer for the text encoder.
        transformer ([`GlmImageTransformer2DModel`]):
            A text conditioned transformer to denoise the encoded image latents (DiT).
        scheduler ([`SchedulerMixin`]):
            A scheduler to be used in combination with `transformer` to denoise the encoded image latents.
    """

    def __init__(
        self,
        tokenizer,
        text_encoder,
        vae,
        transformer,
        scheduler,
    ):
        super().__init__()

        self.tokenizer = tokenizer
        self.text_encoder = text_encoder
        self.vae = vae
        self.transformer = transformer
        self.scheduler = scheduler

        self.vae_scale_factor = (
            2 ** (len(self.vae.config.block_out_channels) - 1)
            if getattr(self, "vae", None)
            else 8
        )
        self.image_processor = VaeImageProcessor(vae_scale_factor=self.vae_scale_factor)

        self.default_sample_size = (
            self.transformer.config.sample_size
            if hasattr(self, "transformer")
            and self.transformer is not None
            and hasattr(self.transformer.config, "sample_size")
            else 128
        )

    def component_uses(
        self, server_args: ServerArgs, stage_name: str | None = None
    ) -> list[ComponentUse]:
        stage_name = self._component_stage_name(stage_name)
        uses: list[ComponentUse] = []
        if self.transformer is not None:
            uses.append(
                ComponentUse(
                    stage_name=stage_name,
                    component_name="transformer",
                    phase="reference_image",
                    memory_intensive=True,
                )
            )
        return uses

    def get_glyph_texts(self, prompt):
        prompt = prompt[0] if isinstance(prompt, list) else prompt
        ocr_texts = (
            re.findall(r"'([^']*)'", prompt)
            + re.findall(r"“([^“”]*)”", prompt)
            + re.findall(r'"([^"]*)"', prompt)
            + re.findall(r"「([^「」]*)」", prompt)
        )
        return ocr_texts

    def _get_glyph_embeds(
        self,
        prompt: Union[str, List[str]] = None,
        max_sequence_length: int = 2048,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ):
        device = device or self._execution_device
        dtype = dtype or self.text_encoder.dtype

        prompts = [prompt] if isinstance(prompt, str) else prompt
        all_input_ids = []
        glyph_counts = []
        for prompt_item in prompts:
            glyph_texts = self.get_glyph_texts(prompt_item)
            input_ids = self.tokenizer(
                glyph_texts if len(glyph_texts) > 0 else [""],
                max_length=max_sequence_length,
                truncation=True,
            ).input_ids
            input_ids = [
                [self.tokenizer.pad_token_id] * ((len(input_ids) + 1) % 2) + input_ids_
                for input_ids_ in input_ids
            ]
            glyph_counts.append(len(input_ids))
            all_input_ids.extend(input_ids)

        padded_length = max(len(input_ids) for input_ids in all_input_ids)
        attention_mask = torch.tensor(
            [
                [1] * len(input_ids) + [0] * (padded_length - len(input_ids))
                for input_ids in all_input_ids
            ],
            device=device,
        )
        input_ids = torch.tensor(
            [
                input_ids
                + [self.tokenizer.pad_token_id] * (padded_length - len(input_ids))
                for input_ids in all_input_ids
            ],
            device=device,
        )
        outputs = self.text_encoder(input_ids, attention_mask=attention_mask)

        glyph_embeds = []
        start = 0
        for glyph_count in glyph_counts:
            end = start + glyph_count
            glyph_embeds.append(
                outputs.last_hidden_state[start:end][attention_mask[start:end].bool()]
            )
            start = end

        prompt_embeds = pad_sequence(glyph_embeds, batch_first=True)
        sequence_lengths = [item.shape[0] for item in glyph_embeds]
        prompt_embeds_mask = None
        if len(set(sequence_lengths)) > 1:
            sequence_lengths_tensor = torch.tensor(sequence_lengths, device=device)
            prompt_embeds_mask = torch.arange(
                prompt_embeds.shape[1], device=device
            ).unsqueeze(0) < sequence_lengths_tensor.unsqueeze(1)
        return prompt_embeds.to(device=device, dtype=dtype), prompt_embeds_mask

    def encode_prompt(
        self,
        prompt: Union[str, List[str]],
        do_classifier_free_guidance: bool = True,
        prompt_embeds: Optional[torch.Tensor] = None,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
        max_sequence_length: int = 2048,
    ):
        r"""
        Encodes the prompt into text encoder hidden states.

        Args:
            prompt (`str` or `List[str]`, *optional*):
                prompt to be encoded
            do_classifier_free_guidance (`bool`, *optional*, defaults to `True`):
                Whether to use classifier free guidance or not.
            prompt_embeds (`torch.Tensor`, *optional*):
                Pre-generated text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt weighting. If not
                provided, text embeddings will be generated from `prompt` input argument.
            device: (`torch.device`, *optional*):
                torch device
            dtype: (`torch.dtype`, *optional*):
                torch dtype
            max_sequence_length (`int`, defaults to `2048`):
                Maximum sequence length in encoded prompt. Can be set to other values but may lead to poorer results.
        """
        device = device or self._execution_device

        prompt = [prompt] if isinstance(prompt, str) else prompt
        if prompt is not None:
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        if prompt_embeds is None:
            prompt_embeds, prompt_embeds_mask = self._get_glyph_embeds(
                prompt, max_sequence_length, device, dtype
            )
        else:
            prompt_embeds_mask = torch.ones(
                prompt_embeds.shape[:2],
                device=prompt_embeds.device,
                dtype=torch.bool,
            )

        negative_prompt_embeds = None
        negative_prompt_embeds_mask = None
        if do_classifier_free_guidance:
            negative_prompt = ""
            negative_prompt = (
                batch_size * [negative_prompt]
                if isinstance(negative_prompt, str)
                else negative_prompt
            )

            if prompt is not None and type(prompt) is not type(negative_prompt):
                raise TypeError(
                    f"`negative_prompt` should be the same type to `prompt`, but got {type(negative_prompt)} !="
                    f" {type(prompt)}."
                )
            elif batch_size != len(negative_prompt):
                raise ValueError(
                    f"`negative_prompt`: {negative_prompt} has batch size {len(negative_prompt)}, but `prompt`:"
                    f" {prompt} has batch size {batch_size}. Please make sure that passed `negative_prompt` matches"
                    " the batch size of `prompt`."
                )

            negative_prompt_embeds, negative_prompt_embeds_mask = (
                self._get_glyph_embeds(
                    negative_prompt, max_sequence_length, device, dtype
                )
            )

        return (
            prompt_embeds,
            negative_prompt_embeds,
            prompt_embeds_mask,
            negative_prompt_embeds_mask,
        )

    def prepare_latents(
        self,
        batch_size,
        num_channels_latents,
        height,
        width,
        dtype,
        device,
        generator,
    ):

        shape = (
            batch_size,
            num_channels_latents,
            int(height) // self.vae_scale_factor,
            int(width) // self.vae_scale_factor,
        )
        if isinstance(generator, list) and len(generator) != batch_size:
            raise ValueError(
                f"You have passed a list of generators of length {len(generator)}, but requested an effective batch"
                f" size of {batch_size}. Make sure the batch size matches the length of the generators."
            )
        latents = randn_tensor(shape, generator=generator, device=device, dtype=dtype)
        return latents

    def check_inputs(
        self,
        prompt,
        height,
        width,
        callback_on_step_end_tensor_inputs,
        prompt_embeds=None,
    ):
        if (
            height is not None
            and height % (self.vae_scale_factor * self.transformer.config.patch_size)
            != 0
            or width is not None
            and width % (self.transformer.config.patch_size) != 0
        ):
            logger.warning(
                f"`height` and `width` have to be divisible by {self.vae_scale_factor * 2} but are {height} and {width}. Dimensions will be resized accordingly"
            )

        if callback_on_step_end_tensor_inputs is not None and not all(
            k in self._callback_tensor_inputs
            for k in callback_on_step_end_tensor_inputs
        ):
            raise ValueError(
                f"`callback_on_step_end_tensor_inputs` has to be in {self._callback_tensor_inputs}, but found {[k for k in callback_on_step_end_tensor_inputs if k not in self._callback_tensor_inputs]}"
            )

        if prompt is not None and prompt_embeds is not None:
            raise ValueError(
                f"Cannot forward both `prompt`: {prompt} and `prompt_embeds`: {prompt_embeds}. Please make sure to"
                " only forward one of the two."
            )
        elif prompt is None and prompt_embeds is None:
            raise ValueError(
                "Provide either `prompt` or `prompt_embeds`. Cannot leave both `prompt` and `prompt_embeds` undefined."
            )
        elif prompt is not None and (
            not isinstance(prompt, str) and not isinstance(prompt, list)
        ):
            raise ValueError(
                f"`prompt` has to be of type `str` or `list` but is {type(prompt)}"
            )

    @property
    def guidance_scale(self):
        return self._guidance_scale

    @property
    def do_classifier_free_guidance(self):
        return self._guidance_scale > 1

    @property
    def num_timesteps(self):
        return self._num_timesteps

    @property
    def attention_kwargs(self):
        return self._attention_kwargs

    @property
    def current_timestep(self):
        return self._current_timestep

    @property
    def interrupt(self):
        return self._interrupt

    @torch.no_grad()
    def forward(
        self,
        batch: Req,
        server_args: ServerArgs,
    ) -> Req:

        guidance_scale = batch.guidance_scale
        prompts, seeds = _expand_prompts_and_seeds(batch)
        batch_size = len(prompts)
        num_inference_steps = batch.num_inference_steps
        if batch.image_path is not None:
            ar_condition_images = [
                load_image(img_path)
                for img_path in image_path_to_list(batch.image_path)
            ]
        else:
            ar_condition_images = None

        height = batch.height
        width = batch.width

        device = get_local_torch_device()
        max_sequence_length = 1024
        generator = [torch.Generator(device=device).manual_seed(seed) for seed in seeds]
        attention_kwargs = {}
        prompt_embeds = None
        do_classifier_free_guidance = True
        dtype = get_module_dtype(self.transformer, torch.bfloat16)

        self._guidance_scale = guidance_scale
        self._current_timestep = None
        self._interrupt = False

        if ar_condition_images is not None:
            height = height or ar_condition_images[0].height
            width = width or ar_condition_images[0].width

        prior_token_id = batch.prior_token_id
        prior_token_image_ids = batch.prior_token_image_ids
        prior_token_id = prior_token_id.to(device)

        # 3. Encode input prompt
        (
            prompt_embeds,
            negative_prompt_embeds,
            prompt_embeds_mask,
            negative_prompt_embeds_mask,
        ) = self.encode_prompt(
            prompts,
            do_classifier_free_guidance,
            prompt_embeds=prompt_embeds,
            max_sequence_length=max_sequence_length,
            device=device,
            dtype=dtype,
        )

        # 4. process images
        if ar_condition_images is not None:
            preprocessed_condition_images = []
            for img in ar_condition_images:
                image_height, image_width = (
                    img.size[::-1]
                    if isinstance(img, PIL.Image.Image)
                    else img.shape[:2]
                )
                multiple_of = self.vae_scale_factor * self.transformer.config.patch_size
                image_height = (image_height // multiple_of) * multiple_of
                image_width = (image_width // multiple_of) * multiple_of
                img = self.image_processor.preprocess(
                    img, height=image_height, width=image_width
                )
                preprocessed_condition_images.append(img)
            ar_condition_images = preprocessed_condition_images

        # 5. Prepare latents and (optional) condition_images kv cache
        latent_channels = self.transformer.config.in_channels
        latents = self.prepare_latents(
            batch_size=batch_size,
            num_channels_latents=latent_channels,
            height=height,
            width=width,
            dtype=torch.float32,
            device=device,
            generator=generator,
        )

        kv_caches = GlmImageKVCache(num_layers=self.transformer.config.num_layers)

        if ar_condition_images is not None:
            latents_mean = torch.tensor(self.vae.config.latents_mean).view(
                1, self.vae.config.latent_channels, 1, 1
            )
            latents_std = torch.tensor(self.vae.config.latents_std).view(
                1, self.vae.config.latent_channels, 1, 1
            )

            vae_dtype = get_module_dtype(self.vae, prompt_embeds.dtype)
            latents_mean = latents_mean.to(device=device, dtype=vae_dtype)
            latents_std = latents_std.to(device=device, dtype=vae_dtype)

            for condition_image, condition_image_prior_token_id in zip(
                ar_condition_images, prior_token_image_ids
            ):
                condition_image = align_tensor_to_module_dtype(
                    condition_image, self.vae, device=device
                )

                condition_latent = retrieve_latents(
                    self.vae.encode(condition_image),
                    generator=generator,
                    sample_mode="argmax",
                )
                condition_latent = (condition_latent - latents_mean) / latents_std

                # Do not remove.
                # It would be use to run the reference image through a
                # forward pass at timestep 0 and keep the KV cache.
                with self.use_declared_component(
                    component_name="transformer",
                    module=self.transformer,
                    phase="reference_image",
                ) as transformer:
                    assert transformer is not None
                    self.transformer = transformer
                    with set_forward_context(current_timestep=1, attn_metadata=None):
                        _ = transformer(
                            hidden_states=condition_latent,
                            encoder_hidden_states=torch.zeros_like(prompt_embeds)[
                                :1, :0, ...
                            ],
                            prior_token_id=condition_image_prior_token_id,
                            prior_token_drop=torch.full_like(
                                condition_image_prior_token_id, False, dtype=torch.bool
                            ),
                            timestep=torch.zeros((1,), device=device),
                            target_size=torch.tensor(
                                [condition_image.shape[-2:]], device=device
                            ),
                            crop_coords=torch.zeros((1, 2), device=device),
                            attention_kwargs=attention_kwargs,
                            kv_caches=kv_caches,
                            kv_caches_mode="write",
                        )

        # 6. Prepare additional timestep conditions
        target_size = (height, width)
        target_size = torch.tensor(
            [target_size] * batch_size,
            dtype=prompt_embeds.dtype,
            device=device,
        )
        crops_coords_top_left = torch.tensor(
            [(0, 0)] * batch_size,
            dtype=prompt_embeds.dtype,
            device=device,
        )

        # Prepare timesteps
        scheduler = self.scheduler
        image_seq_len = (
            (height // self.vae_scale_factor) * (width // self.vae_scale_factor)
        ) // (self.transformer.config.patch_size**2)
        timesteps = np.linspace(
            scheduler.config.num_train_timesteps, 1.0, num_inference_steps + 1
        )[:-1]
        timesteps = timesteps.astype(np.int64).astype(np.float32)
        sigmas = timesteps / scheduler.config.num_train_timesteps
        mu = calculate_shift(
            image_seq_len,
            scheduler.config.get("base_image_seq_len", 256),
            scheduler.config.get("base_shift", 0.25),
            scheduler.config.get("max_shift", 0.75),
        )
        timesteps, num_inference_steps = retrieve_timesteps(
            scheduler, num_inference_steps, device, timesteps, sigmas, mu=mu
        )
        self._num_timesteps = len(timesteps)

        # 7. Prepare for denoising loop

        batch.prompt_embeds = [prompt_embeds]
        batch.negative_prompt_embeds = [negative_prompt_embeds]
        batch.prompt_embeds_mask = [prompt_embeds_mask]
        batch.negative_prompt_embeds_mask = [negative_prompt_embeds_mask]
        batch.latents = latents
        batch.timesteps = timesteps
        batch.scheduler = scheduler
        batch.num_inference_steps = num_inference_steps
        batch.sigmas = sigmas.tolist()  # Convert numpy array to list for validation
        batch.generator = generator
        batch.raw_latent_shape = latents.shape

        batch.prior_token_id = prior_token_id
        batch.prior_token_drop_cond = torch.full_like(
            prior_token_id, False, dtype=torch.bool
        )
        batch.prior_token_drop_uncond = torch.full_like(
            prior_token_id, True, dtype=torch.bool
        )
        batch.target_size = target_size
        batch.crop_coords = crops_coords_top_left

        batch.kv_caches = kv_caches

        batch.height = height
        batch.width = width

        return batch
