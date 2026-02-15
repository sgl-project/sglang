import inspect
import re
import time
from typing import List, Optional, Union

import numpy as np
import torch
from diffusers.image_processor import VaeImageProcessor
from diffusers.utils.torch_utils import randn_tensor

from sglang.multimodal_gen.runtime.distributed import get_local_torch_device
from sglang.multimodal_gen.runtime.models.dits.glm_image import GlmImageKVCache
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
        vision_language_encoder:
            SGLang Engine instance for the AR model that generates image tokens from text prompts.
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
        self.ar_engine = vision_language_encoder
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
        image=None,
        factor: int = 32,
    ):
        """
        Generate prior tokens using the AR model via SGLang Engine.

        Only text-to-image is supported.

        Args:
            prompt: The text prompt
            height: Target image height in pixels
            width: Target image width in pixels
            image: Unused, kept for API compatibility
            factor: Rounding factor for height/width

        Returns:
            Tuple of (prior_token_ids, None)
            - prior_token_ids: Upsampled to d16 format, shape [1, token_h*token_w*4]
        """
        device = get_local_torch_device()
        height = (height // factor) * factor
        width = (width // factor) * factor

        # Build chat template inputs via HF processor
        content = [{"type": "text", "text": prompt}]
        messages = [{"role": "user", "content": content}]

        inputs = self.processor.apply_chat_template(
            messages,
            tokenize=True,
            target_h=height,
            target_w=width,
            return_dict=True,
            return_tensors="pt",
        )

        image_grid_thw = inputs.get("image_grid_thw")
        max_new_tokens, large_image_offset, token_h, token_w = (
            self._compute_generation_params(
                image_grid_thw=image_grid_thw, is_text_to_image=True
            )
        )

        # SGLang Engine call
        output = self.ar_engine.generate(
            input_ids=inputs["input_ids"][0].tolist(),
            image_data=[{"image_grid_thw": image_grid_thw}],
            sampling_params={"temperature": 1.0, "max_new_tokens": max_new_tokens},
        )
        generated_ids = output["output_ids"]

        # Extract large image tokens + upsample D32→D16
        prior_token_ids_d32 = torch.tensor(
            generated_ids[large_image_offset : large_image_offset + token_h * token_w],
            device=device,
        )
        prior_token_ids = self._upsample_token_ids(
            prior_token_ids_d32, token_h, token_w
        )

        return prior_token_ids, None

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

        time_start = time.time()
        prior_token_id, _ = self.generate_prior_tokens(
            prompt=prompt,
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

        # 4. Prepare latents
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

        # 5. Prepare additional timestep conditions
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

        # 6. Prepare for denoising loop

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
