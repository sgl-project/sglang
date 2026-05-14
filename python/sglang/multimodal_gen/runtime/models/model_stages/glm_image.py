import inspect
import re
import time
from math import sqrt
from typing import List, Optional, Tuple, Union

import numpy as np
import PIL
import torch
from diffusers.image_processor import VaeImageProcessor
from diffusers.utils.torch_utils import randn_tensor

from sglang.multimodal_gen.runtime.distributed import get_local_torch_device
from sglang.multimodal_gen.runtime.managers.forward_context import set_forward_context
from sglang.multimodal_gen.runtime.models.dits.glm_image import GlmImageKVCache
from sglang.multimodal_gen.runtime.models.vision_utils import load_image
from sglang.multimodal_gen.runtime.pipelines_core.schedule_batch import Req
from sglang.multimodal_gen.runtime.pipelines_core.stages.base import PipelineStage
from sglang.multimodal_gen.runtime.server_args import ServerArgs
from sglang.multimodal_gen.runtime.utils.logging_utils import init_logger

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


class GlmImageBeforeDenoisingStage(PipelineStage):
    r"""
    Pipeline for text-to-image generation using GLM-Image.

    This pipeline integrates both the AR (autoregressive) model for token generation and the DiT (diffusion
    transformer) model for image decoding.

    Args:
        vae ([`AutoencoderKL`]):
            Variational Auto-Encoder (VAE) Model to encode and decode images to and from latent representations.
        text_encoder ([`T5EncoderModel`]):
            Frozen text-encoder for glyph embeddings.
        tokenizer (`PreTrainedTokenizer`):
            Tokenizer for the text encoder.
        processor (`AutoProcessor`):
            Processor for the AR model to handle chat templates and tokenization.
        vision_language_encoder ([`GlmImageForConditionalGeneration`]):
            The AR model that generates image tokens from text prompts.
        transformer ([`GlmImageTransformer2DModel`]):
            A text conditioned transformer to denoise the encoded image latents (DiT).
        scheduler ([`SchedulerMixin`]):
            A scheduler to be used in combination with `transformer` to denoise the encoded image latents.
    """

    def __init__(
        self,
        tokenizer,
        processor,
        text_encoder,
        vision_language_encoder,
        vae,
        transformer,
        scheduler,
    ):
        super().__init__()

        self.tokenizer = tokenizer
        self.processor = processor
        self.text_encoder = text_encoder
        self.vision_language_encoder = vision_language_encoder
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

    def _parse_and_expand_shape_info(
        self, prompt: str
    ) -> Tuple[str, int, int, int, int]:
        """
        Parse the shape info from prompt and expand it for AR model.

        Args:
            prompt: The prompt containing <sop>H W<eop> shape specification

        Returns:
            Tuple of (expanded_prompt, token_h, token_w, prev_token_h, prev_token_w)
        """
        match = re.search(r"<sop>(\d+)\s+(\d+)<eop>", prompt)
        if match is None:
            raise ValueError(
                f"Prompt must contain shape info in format '<sop>H W<eop>', got: {prompt}"
            )

        token_h, token_w = int(match.group(1)), int(match.group(2))
        ratio = token_h / token_w
        prev_token_h = int(sqrt(ratio) * 16)
        prev_token_w = int(sqrt(1 / ratio) * 16)

        old_shape = f"<sop>{token_h} {token_w}<eop>"
        new_shape = (
            f"<sop>{token_h} {token_w}<eop><sop>{prev_token_h} {prev_token_w}<eop>"
        )
        expanded_prompt = prompt.replace(old_shape, new_shape)

        return expanded_prompt, token_h, token_w, prev_token_h, prev_token_w

    def _build_image_grid_thw(
        self,
        token_h: int,
        token_w: int,
        prev_token_h: int,
        prev_token_w: int,
        existing_grid: Optional[torch.Tensor] = None,
        device: Optional[torch.device] = None,
    ) -> torch.Tensor:
        """
        Build image grid tensor for AR model.

        For text-to-image: creates grid for large image + small image For image-to-image: appends new image to existing
        grid
        """
        if existing_grid is None or existing_grid.numel() == 0:
            # Text-to-image: large image + small image
            return torch.tensor(
                [
                    [1, token_h, token_w],
                    [1, prev_token_h, prev_token_w],
                ],
                device=device,
            )
        else:
            # Image-to-image: append to existing
            return torch.cat(
                [existing_grid, torch.tensor([[1, token_h, token_w]], device=device)],
                dim=0,
            )

    def _calculate_ar_generation_params(
        self,
        token_h: int,
        token_w: int,
        prev_token_h: int,
        prev_token_w: int,
        is_text_to_image: bool,
    ) -> Tuple[int, int]:
        """
        Calculate max_new_tokens and large_image_start_offset for AR generation.
        """
        large_image_tokens = token_h * token_w
        small_image_tokens = prev_token_h * prev_token_w

        if is_text_to_image:
            max_new_tokens = small_image_tokens + large_image_tokens + 1
            large_image_start_offset = small_image_tokens
        else:
            max_new_tokens = large_image_tokens + 1
            large_image_start_offset = 0

        return max_new_tokens, large_image_start_offset

    def _extract_large_image_tokens(
        self,
        outputs: torch.Tensor,
        input_length: int,
        large_image_start_offset: int,
        large_image_tokens: int,
    ) -> torch.Tensor:
        """
        Extract the large image tokens from AR model output.
        """
        generated_tokens = outputs[0][input_length:]
        large_image_start = large_image_start_offset
        large_image_end = large_image_start + large_image_tokens
        return generated_tokens[large_image_start:large_image_end]

    def _upsample_d32_to_d16(
        self, token_ids: torch.Tensor, token_h: int, token_w: int
    ) -> torch.Tensor:
        """
        Upsample token IDs from d32 format to d16 format.

        AR model generates tokens at d32 resolution (each token = 32x32 pixels). DiT expects tokens at d16 resolution
        (each token = 16x16 pixels). This function performs 2x nearest-neighbor upsampling.

        Args:
            token_ids: Token IDs of shape [N] where N = token_h * token_w
            token_h: Height in d32 token units
            token_w: Width in d32 token units

        Returns:
            Upsampled token IDs of shape [1, N*4] where N*4 = (token_h*2) * (token_w*2)
        """
        # Reshape to spatial format: [1, 1, H, W]
        token_ids = token_ids.view(1, 1, token_h, token_w)

        # 2x nearest-neighbor upsampling
        token_ids = torch.nn.functional.interpolate(
            token_ids.float(), scale_factor=2, mode="nearest"
        ).to(dtype=torch.long)

        # Flatten back to [1, H*W*4]
        token_ids = token_ids.view(1, -1)

        return token_ids

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
        image: Optional[List[PIL.Image.Image]] = None,
        factor: int = 32,
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
        device = self.vision_language_encoder.device
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
        if image is not None:
            prior_token_image_embed = self.vision_language_encoder.get_image_features(
                inputs["pixel_values"], image_grid_thw[:-1]
            )
            prior_token_image_embed = torch.cat(prior_token_image_embed, dim=0)
            prior_token_image_ids = self.vision_language_encoder.get_image_tokens(
                prior_token_image_embed, image_grid_thw[:-1]
            )

        # For GLM-Image, greedy decoding is not allowed; it may cause repetitive outputs.
        # max_new_tokens must be exactly grid_h * grid_w + 1 (the +1 is for EOS).
        outputs = self.vision_language_encoder.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
        )

        prior_token_ids_d32 = self._extract_large_image_tokens(
            outputs,
            inputs["input_ids"].shape[-1],
            large_image_offset,
            token_h * token_w,
        )
        prior_token_ids = self._upsample_token_ids(
            prior_token_ids_d32, token_h, token_w
        )

        return prior_token_ids, prior_token_image_ids

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

        glyph_texts = self.get_glyph_texts(prompt)
        input_ids = self.tokenizer(
            glyph_texts if len(glyph_texts) > 0 else [""],
            max_length=max_sequence_length,
            truncation=True,
        ).input_ids
        input_ids = [
            [self.tokenizer.pad_token_id] * ((len(input_ids) + 1) % 2) + input_ids_
            for input_ids_ in input_ids
        ]
        max_length = max(len(input_ids_) for input_ids_ in input_ids)
        attention_mask = torch.tensor(
            [
                [1] * len(input_ids_) + [0] * (max_length - len(input_ids_))
                for input_ids_ in input_ids
            ],
            device=device,
        )
        input_ids = torch.tensor(
            [
                input_ids_
                + [self.tokenizer.pad_token_id] * (max_length - len(input_ids_))
                for input_ids_ in input_ids
            ],
            device=device,
        )
        outputs = self.text_encoder(input_ids, attention_mask=attention_mask)
        glyph_embeds = outputs.last_hidden_state[attention_mask.bool()].unsqueeze(0)

        return glyph_embeds.to(device=device, dtype=dtype)

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
            prompt_embeds = self._get_glyph_embeds(
                prompt, max_sequence_length, device, dtype
            )

        seq_len = prompt_embeds.size(1)
        prompt_embeds = prompt_embeds.repeat(1, 1, 1)
        prompt_embeds = prompt_embeds.view(1, seq_len, -1)

        negative_prompt_embeds = None
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

            negative_prompt_embeds = self._get_glyph_embeds(
                negative_prompt, max_sequence_length, device, dtype
            )

            seq_len = negative_prompt_embeds.size(1)
            negative_prompt_embeds = negative_prompt_embeds.repeat(1, 1, 1)
            negative_prompt_embeds = negative_prompt_embeds.view(1, seq_len, -1)

        return prompt_embeds, negative_prompt_embeds

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
        prompt = batch.prompt
        num_inference_steps = batch.num_inference_steps
        if batch.image_path is not None:
            ar_condition_images = [
                load_image(img_path) for img_path in batch.image_path
            ]
        else:
            ar_condition_images = None

        height = batch.height
        width = batch.width

        device = get_local_torch_device()
        max_sequence_length = 1024
        generator = torch.Generator(device=device).manual_seed(batch.seed)
        attention_kwargs = {}
        prompt_embeds = None
        do_classifier_free_guidance = True
        dtype = torch.bfloat16

        self._guidance_scale = guidance_scale
        self._current_timestep = None
        self._interrupt = False

        batch_size = 1

        device = get_local_torch_device()

        if ar_condition_images is not None:
            height = height or ar_condition_images[0].height
            width = width or ar_condition_images[0].width
        time_start = time.time()
        prior_token_id, prior_token_image_ids = self.generate_prior_tokens(
            prompt=prompt,
            image=ar_condition_images,
            height=height,
            width=width,
        )
        prior_token_id = prior_token_id.to(device=device)
        time_end = time.time()
        logger.info(f"generate_prior_tokens time: {time_end - time_start}")

        # 3. Encode input prompt
        prompt_embeds, negative_prompt_embeds = self.encode_prompt(
            prompt,
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
            batch_size=1,
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

            latents_mean = latents_mean.to(device=device, dtype=prompt_embeds.dtype)
            latents_std = latents_std.to(device=device, dtype=prompt_embeds.dtype)

            for condition_image, condition_image_prior_token_id in zip(
                ar_condition_images, prior_token_image_ids
            ):
                condition_image = condition_image.to(
                    device=device, dtype=prompt_embeds.dtype
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
                with set_forward_context(current_timestep=1, attn_metadata=None):
                    _ = self.transformer(
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
            [target_size], dtype=prompt_embeds.dtype, device=device
        )
        crops_coords_top_left = torch.tensor(
            [(0, 0)], dtype=prompt_embeds.dtype, device=device
        )

        # Prepare timesteps
        image_seq_len = (
            (height // self.vae_scale_factor) * (width // self.vae_scale_factor)
        ) // (self.transformer.config.patch_size**2)
        timesteps = np.linspace(
            self.scheduler.config.num_train_timesteps, 1.0, num_inference_steps + 1
        )[:-1]
        timesteps = timesteps.astype(np.int64).astype(np.float32)
        sigmas = timesteps / self.scheduler.config.num_train_timesteps
        mu = calculate_shift(
            image_seq_len,
            self.scheduler.config.get("base_image_seq_len", 256),
            self.scheduler.config.get("base_shift", 0.25),
            self.scheduler.config.get("max_shift", 0.75),
        )
        timesteps, num_inference_steps = retrieve_timesteps(
            self.scheduler, num_inference_steps, device, timesteps, sigmas, mu=mu
        )
        self._num_timesteps = len(timesteps)

        # 7. Prepare for denoising loop

        batch.prompt_embeds = [prompt_embeds]
        batch.negative_prompt_embeds = [negative_prompt_embeds]
        batch.latents = latents
        batch.timesteps = timesteps
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
