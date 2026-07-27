# Copyright 2025 ByteDance Ltd. and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0
"""BAGEL request validation, LLM prefill, noise, and schedule preparation.

Source: https://github.com/ByteDance-Seed/Bagel/blob/a2fa77dd8caeefc41e6607ae0ec17408d3f4ee9f/inferencer.py
"""

import math

import torch
from PIL import Image
from torchvision.transforms import functional as TF

from sglang.multimodal_gen.runtime.distributed import get_local_torch_device
from sglang.multimodal_gen.runtime.managers.memory_managers.component_manager import (
    ComponentUse,
)
from sglang.multimodal_gen.runtime.pipelines_core.diffusion_scheduler_utils import (
    get_or_create_request_scheduler,
)
from sglang.multimodal_gen.runtime.pipelines_core.schedule_batch import Req
from sglang.multimodal_gen.runtime.pipelines_core.stages.base import PipelineStage
from sglang.multimodal_gen.runtime.pipelines_core.stages.input_validation import (
    InputValidationStage,
)
from sglang.multimodal_gen.runtime.server_args import ServerArgs

_BAGEL_CONTEXT_KEY = "bagel_context"
_SPECIAL_TOKENS = (
    "<|im_start|>",
    "<|im_end|>",
    "<|vision_start|>",
    "<|vision_end|>",
)
_EXPECTED_SPECIAL_TOKEN_IDS = {
    "<|im_start|>": 151644,
    "<|im_end|>": 151645,
    "<|vision_start|>": 151652,
    "<|vision_end|>": 151653,
}


class BagelInputValidationStage(InputValidationStage):
    """Normalize BAGEL requests to the official CPU RNG before seed creation."""

    def _generate_seeds(self, batch: Req, server_args: ServerArgs) -> None:
        # The generic Images API defaults generator_device to CUDA, so merely
        # setting the pipeline default is insufficient. BAGEL's T2I contract
        # fixes the official CPU noise stream for reproducibility.
        batch.generator_device = server_args.pipeline_config.generator_device
        super()._generate_seeds(batch, server_args)


class BagelEditInputValidationStage(BagelInputValidationStage):
    """Validate and resize exactly one BAGEL Editing source image."""

    def preprocess_condition_image(
        self,
        batch: Req,
        server_args: ServerArgs,
        condition_image_width,
        condition_image_height,
    ):
        """Reject multi-image requests before applying the official resize."""
        images = (
            batch.condition_image
            if isinstance(batch.condition_image, list)
            else [batch.condition_image]
        )
        if len(images) != 1:
            raise ValueError(
                f"BAGEL Editing supports exactly one input image; got {len(images)}"
            )
        return super().preprocess_condition_image(
            batch,
            server_args,
            condition_image_width,
            condition_image_height,
        )

    def forward(self, batch: Req, server_args: ServerArgs) -> Req:
        """Run generic loading, then enforce official output-size parity."""
        batch = super().forward(batch, server_args)
        if (
            not isinstance(batch.condition_image, list)
            or len(batch.condition_image) != 1
        ):
            raise ValueError("BAGEL Editing requires exactly one decoded input image")
        image = batch.condition_image[0]
        if not isinstance(image, Image.Image):
            raise ValueError("BAGEL Editing input must decode to a PIL image")
        explicit_fields = set(batch.extra.get("explicit_fields", []))
        if {"width", "height"} & explicit_fields:
            if (int(batch.width), int(batch.height)) != image.size:
                raise ValueError(
                    "BAGEL Editing size must match the official preprocessed input "
                    f"size {image.width}x{image.height}; got "
                    f"{batch.width}x{batch.height}"
                )
        return batch


def validate_bagel_special_tokens(tokenizer) -> dict[str, int]:
    """Resolve BAGEL special-token IDs without modifying the tokenizer.

    Args:
        tokenizer: A Hugging Face compatible tokenizer.

    Returns:
        Mapping from each required token string to its existing vocabulary ID.

    Raises:
        ValueError: If a token is missing, aliases to UNK, or does not encode as
            exactly one vocabulary item.
    """
    token_ids: dict[str, int] = {}
    unknown_id = getattr(tokenizer, "unk_token_id", None)
    unknown_token = getattr(tokenizer, "unk_token", None)
    for token in _SPECIAL_TOKENS:
        token_id = tokenizer.convert_tokens_to_ids(token)
        if token_id is None or int(token_id) < 0:
            raise ValueError(f"BAGEL tokenizer is missing required token {token!r}")
        if (
            unknown_id is not None
            and int(token_id) == int(unknown_id)
            and token != unknown_token
        ):
            raise ValueError(
                f"BAGEL tokenizer maps required token {token!r} to unk_token_id"
            )
        encoded = tokenizer.encode(token, add_special_tokens=False)
        if encoded != [int(token_id)]:
            raise ValueError(
                f"BAGEL token {token!r} must encode to one existing ID; got {encoded!r}"
            )
        expected_id = _EXPECTED_SPECIAL_TOKEN_IDS[token]
        if int(token_id) != expected_id:
            raise ValueError(
                f"BAGEL token {token!r} must use checkpoint ID {expected_id}; "
                f"got {token_id}"
            )
        token_ids[token] = int(token_id)
    return token_ids


class BagelBeforeDenoisingStage(PipelineStage):
    """Prepare a single BAGEL T2I request for the standard denoising stage."""

    def __init__(self, transformer, tokenizer, scheduler) -> None:
        """Initialize the stage with immutable component templates.

        Args:
            transformer: Stateless ``BagelTransformer`` instance.
            tokenizer: Official checkpoint tokenizer.
            scheduler: Scheduler template cloned for every request.
        """
        super().__init__()
        self.transformer = transformer
        self.tokenizer = tokenizer
        self.scheduler = scheduler
        self._special_token_ids: dict[str, int] | None = None

    def component_uses(
        self, server_args: ServerArgs, stage_name: str | None = None
    ) -> list[ComponentUse]:
        """Declare the Transformer prefill residency window."""
        return [
            ComponentUse(
                self._component_stage_name(stage_name),
                "transformer",
                phase="prefill",
                target_dtype=torch.bfloat16,
                memory_intensive=True,
            )
        ]

    def _ensure_special_tokens(self) -> dict[str, int]:
        if self._special_token_ids is None:
            self._special_token_ids = validate_bagel_special_tokens(self.tokenizer)
        return self._special_token_ids

    def _tokenize_prompt(
        self, prompt: str, special_token_ids: dict[str, int]
    ) -> torch.Tensor:
        prompt_ids = self.tokenizer.encode(prompt, add_special_tokens=False)
        return torch.tensor(
            [
                special_token_ids["<|im_start|>"],
                *prompt_ids,
                special_token_ids["<|im_end|>"],
            ],
            dtype=torch.long,
        )

    requires_image: bool = False

    def _validate_request(self, batch: Req, server_args: ServerArgs) -> None:
        """Reject request features outside the current BAGEL T2I contract."""
        if not isinstance(batch.prompt, str) or not batch.prompt.strip():
            raise ValueError("BAGEL T2I requires exactly one non-empty string prompt")
        if not isinstance(batch.seed, int) or isinstance(batch.seed, bool):
            raise ValueError("BAGEL T2I requires one scalar seed")
        if batch.num_outputs_per_prompt != 1:
            raise ValueError("BAGEL T2I supports num_outputs_per_prompt=1 only")
        if batch.num_frames != 1:
            raise ValueError("BAGEL T2I supports num_frames=1 only")
        if batch.negative_prompt not in (None, ""):
            raise ValueError(
                "BAGEL performs internal CFG and does not support negative_prompt"
            )
        has_prompt_embeds = (
            batch.prompt_embeds.numel() > 0
            if isinstance(batch.prompt_embeds, torch.Tensor)
            else bool(batch.prompt_embeds)
        )
        if has_prompt_embeds:
            raise ValueError("BAGEL T2I does not accept precomputed prompt_embeds")
        has_negative_prompt_embeds = (
            batch.negative_prompt_embeds.numel() > 0
            if isinstance(batch.negative_prompt_embeds, torch.Tensor)
            else bool(batch.negative_prompt_embeds)
        )
        if has_negative_prompt_embeds:
            raise ValueError(
                "BAGEL T2I does not accept precomputed negative_prompt_embeds"
            )
        if self.requires_image:
            if batch.true_cfg_scale is not None and (
                not math.isfinite(float(batch.true_cfg_scale))
                or float(batch.true_cfg_scale) < 0
            ):
                raise ValueError(
                    "BAGEL Editing true_cfg_scale must be a finite non-negative value"
                )
            images = (
                batch.condition_image
                if isinstance(batch.condition_image, list)
                else [batch.condition_image]
            )
            if batch.image_path is None or len(images) != 1 or images[0] is None:
                raise ValueError("BAGEL Editing requires exactly one input image")
        else:
            if batch.true_cfg_scale is not None:
                raise ValueError(
                    "BAGEL T2I uses guidance_scale; true_cfg_scale is not supported"
                )
            if batch.image_path is not None or batch.condition_image is not None:
                raise ValueError("BAGEL T2I does not accept image input")
        if batch.extra.get("dynamic_batch_seeds") is not None:
            raise ValueError("BAGEL dynamic batching is not supported")
        if batch.rollout:
            raise ValueError("BAGEL rollout mode is not supported")
        if batch.return_trajectory_latents or batch.return_trajectory_decoded:
            raise ValueError("BAGEL trajectory output is not supported")

        if not isinstance(batch.height, int) or not isinstance(batch.width, int):
            raise ValueError("BAGEL height and width must be integers")
        height, width = batch.height, batch.width
        latent_downsample = int(
            server_args.pipeline_config.dit_config.arch_config.latent_downsample
        )
        if height <= 0 or width <= 0:
            raise ValueError("BAGEL height and width must be positive")
        if height % latent_downsample or width % latent_downsample:
            raise ValueError(
                "BAGEL height and width must be divisible by "
                f"{latent_downsample}; got {width}x{height}"
            )
        max_latent_size = int(
            server_args.pipeline_config.dit_config.arch_config.max_latent_size
        )
        max_output_size = latent_downsample * max_latent_size
        if height > max_output_size or width > max_output_size:
            raise ValueError(
                f"BAGEL supports dimensions up to {max_output_size}; "
                f"got {width}x{height}"
            )

        if not isinstance(batch.generator, list) or len(batch.generator) != 1:
            raise ValueError(
                "BAGEL requires InputValidationStage to create exactly one generator"
            )
        if not math.isfinite(float(batch.guidance_scale)) or batch.guidance_scale < 0:
            raise ValueError("BAGEL guidance_scale must be a finite non-negative value")

    def _build_request_context(
        self,
        batch: Req,
        server_args: ServerArgs,
        transformer,
        special_token_ids: dict[str, int],
        device: torch.device,
        **_context_inputs,
    ):
        """Build the two-way T2I context for one request."""
        del server_args
        conditional_ids = self._tokenize_prompt(batch.prompt, special_token_ids).to(
            device
        )
        return transformer.build_context(
            conditional_ids,
            None,
            height=int(batch.height),
            width=int(batch.width),
            start_of_image_token_id=special_token_ids["<|vision_start|>"],
            end_of_image_token_id=special_token_ids["<|vision_end|>"],
        )

    def _prepare_context_inputs(
        self,
        batch: Req,
        server_args: ServerArgs,
        special_token_ids: dict[str, int],
        device: torch.device,
    ) -> dict[str, object]:
        """Return no extra inputs for the text-only T2I context."""
        del batch, server_args, special_token_ids, device
        return {}

    @staticmethod
    def build_shifted_schedule(
        num_inference_steps: int,
        flow_shift: float,
        device: torch.device,
    ) -> torch.Tensor:
        """Create N shifted raw sigmas from N+1 endpoints.

        Args:
            num_inference_steps: Exact number of requested denoising updates.
            flow_shift: Positive BAGEL rational flow shift.
            device: Device on which to materialize the schedule.

        Returns:
            A float32 tensor of N raw model sigmas in descending order.

        Raises:
            ValueError: If the step count or flow shift is invalid.
        """
        if not isinstance(num_inference_steps, int) or num_inference_steps <= 0:
            raise ValueError("BAGEL num_inference_steps must be a positive integer")
        if not math.isfinite(flow_shift) or flow_shift <= 0:
            raise ValueError("BAGEL flow_shift must be a finite positive value")
        endpoints = torch.linspace(
            1.0,
            0.0,
            num_inference_steps + 1,
            dtype=torch.float32,
            device=device,
        )
        shifted = flow_shift * endpoints / (1 + (flow_shift - 1) * endpoints)
        return shifted[:-1].contiguous()

    @torch.no_grad()
    def forward(self, batch: Req, server_args: ServerArgs) -> Req:
        """Prefill request KV, generate noise, and install an isolated schedule.

        Args:
            batch: Request after ``InputValidationStage``.
            server_args: Runtime configuration.

        Returns:
            The request populated for the standard ``DenoisingStage``.

        Raises:
            ValueError: If the request uses an unsupported feature or invalid
                schedule/generator setting.
        """
        self._validate_request(batch, server_args)
        device = get_local_torch_device()
        generator = batch.generator[0]
        generator_device = torch.device(generator.device)
        if generator_device.type not in {"cpu", torch.device(device).type}:
            raise ValueError(
                "BAGEL generator must use CPU or the denoising device type; got "
                f"{generator.device}, expected cpu or {device}"
            )
        if generator.initial_seed() != int(batch.seed):
            raise ValueError("BAGEL request generator seed does not match batch.seed")

        special_token_ids = self._ensure_special_tokens()
        context_inputs = self._prepare_context_inputs(
            batch,
            server_args,
            special_token_ids,
            device,
        )
        with self.use_declared_component(
            component_name="transformer",
            module=self.transformer,
            phase="prefill",
        ) as transformer:
            assert transformer is not None
            self.transformer = transformer
            context = self._build_request_context(
                batch,
                server_args,
                transformer,
                special_token_ids,
                device,
                **context_inputs,
            )

        arch = server_args.pipeline_config.dit_config.arch_config
        token_height = int(batch.height) // int(arch.latent_downsample)
        token_width = int(batch.width) // int(arch.latent_downsample)
        patch_width = int(arch.latent_patch_size) ** 2 * int(arch.latent_channel)
        latents = torch.randn(
            token_height * token_width,
            patch_width,
            generator=generator,
            device=generator_device,
            dtype=torch.float32,
        ).to(device)

        flow_shift = (
            batch.flow_shift
            if batch.flow_shift is not None
            else server_args.pipeline_config.flow_shift
        )
        raw_sigmas = self.build_shifted_schedule(
            int(batch.num_inference_steps), float(flow_shift), device
        )
        scheduler = get_or_create_request_scheduler(batch, self.scheduler, isolate=True)
        scheduler.set_shift(1.0)
        if scheduler.shift != 1.0:
            raise RuntimeError("BAGEL scheduler template must use shift=1.0")
        schedule_values = raw_sigmas.tolist()
        scheduler.set_timesteps(
            sigmas=schedule_values,
            timesteps=schedule_values,
            device=device,
        )
        if len(scheduler.timesteps) != int(batch.num_inference_steps):
            raise RuntimeError(
                "BAGEL scheduler did not preserve the requested step count"
            )
        if len(scheduler.sigmas) != int(batch.num_inference_steps) + 1:
            raise RuntimeError("BAGEL scheduler must append exactly one terminal sigma")

        batch.extra[_BAGEL_CONTEXT_KEY] = context
        batch.latents = latents
        # The model consumes a 2D token matrix, but the standard denoising
        # skeleton reads axis 0 as logical batch size when expanding timestep.
        batch.raw_latent_shape = (1, *latents.shape)
        batch.n_tokens = token_height * token_width
        batch.scheduler = scheduler
        batch.timesteps = scheduler.timesteps
        batch.sigmas = scheduler.sigmas.tolist()
        batch.prompt_embeds = [torch.empty(0, device=device, dtype=torch.bfloat16)]
        batch.negative_prompt_embeds = []
        batch.do_classifier_free_guidance = False
        return batch


class BagelEditBeforeDenoisingStage(BagelBeforeDenoisingStage):
    """Encode one source image and build BAGEL's three Editing CFG prefixes."""

    requires_image = True

    def __init__(
        self,
        transformer,
        vae,
        image_encoder,
        tokenizer,
        scheduler,
    ) -> None:
        """Initialize the Editing stage with immutable model components.

        Args:
            transformer: Request-stateless BAGEL mixture-of-transformers.
            vae: Full BAGEL VAE with encoder and decoder.
            image_encoder: SigLIP NaViT, connector, and LLM position table.
            tokenizer: Official checkpoint tokenizer.
            scheduler: Scheduler template cloned for every request.
        """
        super().__init__(transformer, tokenizer, scheduler)
        self.vae = vae
        self.image_encoder = image_encoder

    def component_uses(
        self, server_args: ServerArgs, stage_name: str | None = None
    ) -> list[ComponentUse]:
        """Declare VAE/ViT encoding followed by Transformer prefill."""
        component_stage = self._component_stage_name(stage_name)
        return [
            ComponentUse(
                component_stage,
                "vae",
                phase="encode",
                target_dtype=torch.bfloat16,
                memory_intensive=True,
            ),
            ComponentUse(
                component_stage,
                "image_encoder",
                phase="encode",
                target_dtype=torch.bfloat16,
                memory_intensive=True,
            ),
            ComponentUse(
                component_stage,
                "transformer",
                phase="prefill",
                target_dtype=torch.bfloat16,
                memory_intensive=True,
            ),
        ]

    @staticmethod
    def _patchify_vae_latents(
        latents: torch.Tensor,
        *,
        patch_size: int,
        max_latent_size: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Patchify one encoded source image and build its 2D position IDs."""
        if latents.ndim != 4 or latents.shape[0] != 1:
            raise ValueError(
                "BAGEL Editing VAE must return one NCHW latent; "
                f"got shape {tuple(latents.shape)}"
            )
        _, channels, latent_height, latent_width = latents.shape
        if latent_height % patch_size or latent_width % patch_size:
            raise ValueError(
                "BAGEL Editing VAE latent dimensions must be divisible by "
                f"patch_size={patch_size}"
            )
        token_height = latent_height // patch_size
        token_width = latent_width // patch_size
        if token_height > max_latent_size or token_width > max_latent_size:
            raise ValueError(
                "BAGEL Editing source image exceeds the latent position table"
            )
        patches = latents[0].reshape(
            channels,
            token_height,
            patch_size,
            token_width,
            patch_size,
        )
        patches = torch.einsum("chpwq->hwpqc", patches).reshape(
            -1, patch_size * patch_size * channels
        )
        rows = torch.arange(token_height, device=latents.device).unsqueeze(1)
        columns = torch.arange(token_width, device=latents.device).unsqueeze(0)
        position_ids = (rows * max_latent_size + columns).reshape(-1)
        return patches, position_ids

    def _prepare_context_inputs(
        self,
        batch: Req,
        server_args: ServerArgs,
        special_token_ids: dict[str, int],
        device: torch.device,
    ) -> dict[str, object]:
        """Encode the source image before the Transformer prefill phase."""
        image = batch.condition_image[0]
        if not isinstance(image, Image.Image):
            raise ValueError("BAGEL Editing requires one PIL source image")

        with self.use_declared_component(
            component_name="vae",
            module=self.vae,
            phase="encode",
        ) as vae:
            assert vae is not None
            self.vae = vae
            pixels = TF.to_tensor(image.convert("RGB")).mul_(2.0).sub_(1.0)
            pixels = pixels.unsqueeze(0).to(device=vae.device, dtype=vae.dtype)
            posterior_generator = torch.Generator(device=vae.device).manual_seed(
                int(batch.seed)
            )
            encoded_latents = vae.encode(
                pixels,
                generator=posterior_generator,
            )

        arch = server_args.pipeline_config.dit_config.arch_config
        vae_patches, vae_position_ids = self._patchify_vae_latents(
            encoded_latents,
            patch_size=int(arch.latent_patch_size),
            max_latent_size=int(arch.max_latent_size),
        )
        with self.use_declared_component(
            component_name="image_encoder",
            module=self.image_encoder,
            phase="encode",
        ) as image_encoder:
            assert image_encoder is not None
            self.image_encoder = image_encoder
            vision_embeddings = image_encoder.encode_image(image)

        text_input_ids = self._tokenize_prompt(batch.prompt, special_token_ids).to(
            device
        )
        return {
            "vae_patches": vae_patches,
            "vae_position_ids": vae_position_ids,
            "vision_embeddings": vision_embeddings,
            "text_input_ids": text_input_ids,
        }

    def _build_request_context(
        self,
        batch: Req,
        server_args: ServerArgs,
        transformer,
        special_token_ids: dict[str, int],
        device: torch.device,
        *,
        vae_patches: torch.Tensor,
        vae_position_ids: torch.Tensor,
        vision_embeddings: torch.Tensor,
        text_input_ids: torch.Tensor,
    ):
        """Construct request-owned three-way prefixes during Transformer use."""
        del server_args, device
        return transformer.build_editing_context(
            vae_patches,
            vae_position_ids,
            vision_embeddings,
            text_input_ids,
            height=int(batch.height),
            width=int(batch.width),
            start_of_image_token_id=special_token_ids["<|vision_start|>"],
            end_of_image_token_id=special_token_ids["<|vision_end|>"],
        )
