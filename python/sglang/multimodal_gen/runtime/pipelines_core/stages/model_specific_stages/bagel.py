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
from sglang.multimodal_gen.runtime.managers.forward_context import set_forward_context
from sglang.multimodal_gen.runtime.managers.memory_managers.component_manager import (
    ComponentUse,
)
from sglang.multimodal_gen.runtime.models.dits.bagel_taylorseer import (
    BagelTaylorSeerContext,
)
from sglang.multimodal_gen.runtime.pipelines_core.diffusion_scheduler_utils import (
    get_or_create_request_scheduler,
)
from sglang.multimodal_gen.runtime.pipelines_core.schedule_batch import (
    OutputBatch,
    Req,
    TextGenerationOutput,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.base import PipelineStage
from sglang.multimodal_gen.runtime.pipelines_core.stages.input_validation import (
    InputValidationStage,
)
from sglang.multimodal_gen.runtime.server_args import ServerArgs

_BAGEL_CONTEXT_KEY = "bagel_context"
_BAGEL_TAYLORSEER_KEY = "bagel_taylorseer_context"
_REVISED_PROMPT_KEY = "revised_prompt"
GEN_THINK_SYSTEM_PROMPT = (
    "You should first think about the planning process in the mind and then "
    "generate the image.\n"
    "The planning process is enclosed within <think> </think> tags, i.e. "
    "<think> planning process here </think> image here"
)
VLM_THINK_SYSTEM_PROMPT = (
    "You should first think about the reasoning process in the mind and then "
    "provide the user with the answer.\n"
    "The reasoning process is enclosed within <think> </think> tags, i.e. "
    "<think> reasoning process here </think> answer here"
)
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


class BagelUnderstandingInputValidationStage(BagelInputValidationStage):
    """Load exactly one image and apply BAGEL's outer VAE resize transform."""

    def forward(self, batch: Req, server_args: ServerArgs) -> Req:
        """Normalize one Understanding image before its independent ViT resize.

        Args:
            batch: Request containing one image path and one text prompt.
            server_args: Runtime configuration with BAGEL resize policy.

        Returns:
            Request containing one resized RGB PIL image.

        Raises:
            ValueError: If the request does not decode to exactly one PIL image.
        """
        batch = super().forward(batch, server_args)
        images = (
            batch.condition_image
            if isinstance(batch.condition_image, list)
            else [batch.condition_image]
        )
        if len(images) != 1 or not isinstance(images[0], Image.Image):
            raise ValueError(
                "BAGEL Understanding requires exactly one decoded PIL image"
            )
        image = images[0]
        config = server_args.pipeline_config
        target_width, target_height = config.calculate_condition_image_size(
            image, image.width, image.height
        )
        image, _ = config.preprocess_condition_image(
            image,
            target_width,
            target_height,
            self.vae_image_processor,
        )
        batch.condition_image = [image]
        batch.width = target_width
        batch.height = target_height
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
    """Prepare compatible BAGEL T2I requests for the standard denoising stage."""

    allows_dynamic_batching: bool = True

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

    def _request_prompts(self, batch: Req) -> list[str]:
        """Return validated prompts in request order.

        Args:
            batch: One request or a scheduler-merged request.

        Returns:
            A non-empty list of normalized prompt strings.

        Raises:
            ValueError: If prompt shape/content is invalid or the stage variant
                does not support scheduler-level dynamic batching.
        """
        if isinstance(batch.prompt, str):
            if not batch.prompt.strip():
                raise ValueError("BAGEL T2I requires non-empty string prompts")
            return [batch.prompt]

        if isinstance(batch.prompt, list):
            if not self.allows_dynamic_batching:
                raise ValueError("BAGEL dynamic batching is supported by pure T2I only")
            if not batch.prompt or any(
                not isinstance(prompt, str) or not prompt.strip()
                for prompt in batch.prompt
            ):
                raise ValueError("BAGEL T2I requires non-empty string prompts")
            return list(batch.prompt)

        raise ValueError("BAGEL T2I requires non-empty string prompts")

    def _validate_request(self, batch: Req, server_args: ServerArgs) -> None:
        """Reject request features outside the current BAGEL T2I contract."""
        prompts = self._request_prompts(batch)
        request_count = len(prompts)
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
        dynamic_batch_seeds = batch.extra.get("dynamic_batch_seeds")
        if request_count > 1 and dynamic_batch_seeds is None:
            raise ValueError(
                "BAGEL batched prompts require dynamic_batch_seeds metadata"
            )
        if dynamic_batch_seeds is not None:
            if (
                not isinstance(dynamic_batch_seeds, list)
                or len(dynamic_batch_seeds) != request_count
                or any(
                    not isinstance(seed, int) or isinstance(seed, bool)
                    for seed in dynamic_batch_seeds
                )
            ):
                raise ValueError(
                    "BAGEL dynamic_batch_seeds must contain one integer per prompt"
                )
            if dynamic_batch_seeds[0] != batch.seed:
                raise ValueError(
                    "BAGEL first dynamic seed must match the merged request seed"
                )
        if batch.rollout:
            raise ValueError("BAGEL rollout mode is not supported")
        if batch.return_trajectory_latents or batch.return_trajectory_decoded:
            raise ValueError("BAGEL trajectory output is not supported")
        if not isinstance(batch.enable_taylorseer, bool):
            raise ValueError("BAGEL enable_taylorseer must be a boolean")

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

        if (
            not isinstance(batch.seeds, list)
            or len(batch.seeds) != request_count
            or any(
                not isinstance(seed, int) or isinstance(seed, bool)
                for seed in batch.seeds
            )
        ):
            raise ValueError("BAGEL requires one validated seed per prompt")
        expected_seeds = (
            dynamic_batch_seeds if dynamic_batch_seeds is not None else [batch.seed]
        )
        if batch.seeds != expected_seeds:
            raise ValueError("BAGEL validated seeds do not match request seeds")
        if (
            not isinstance(batch.generator, list)
            or len(batch.generator) != request_count
        ):
            raise ValueError("BAGEL requires one generator per prompt")
        if not math.isfinite(float(batch.guidance_scale)) or batch.guidance_scale < 0:
            raise ValueError("BAGEL guidance_scale must be a finite non-negative value")

    def _build_text_to_image_context(
        self,
        batch: Req,
        transformer,
        prompt: str,
        special_token_ids: dict[str, int],
        device: torch.device,
    ):
        """Prefill one pure T2I prompt without mutating the merged request."""
        conditional_ids = self._tokenize_prompt(prompt, special_token_ids).to(device)
        return transformer.build_context(
            conditional_ids,
            None,
            height=int(batch.height),
            width=int(batch.width),
            start_of_image_token_id=special_token_ids["<|vision_start|>"],
            end_of_image_token_id=special_token_ids["<|vision_end|>"],
        )

    def _build_request_context(
        self,
        batch: Req,
        server_args: ServerArgs,
        transformer,
        special_token_ids: dict[str, int],
        device: torch.device,
        **_context_inputs,
    ):
        del server_args
        return self._build_text_to_image_context(
            batch,
            transformer,
            batch.prompt,
            special_token_ids,
            device,
        )

    def _prepare_context_inputs(
        self,
        batch: Req,
        server_args: ServerArgs,
        special_token_ids: dict[str, int],
        device: torch.device,
    ) -> dict[str, object]:
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
        prompts = self._request_prompts(batch)
        device = get_local_torch_device()
        generator_devices: list[torch.device] = []
        for request_index, (generator, seed) in enumerate(
            zip(batch.generator, batch.seeds, strict=True)
        ):
            generator_device = torch.device(generator.device)
            if generator_device.type not in {"cpu", torch.device(device).type}:
                raise ValueError(
                    "BAGEL generator must use CPU or the denoising device type; got "
                    f"{generator.device}, expected cpu or {device}"
                )
            if generator.initial_seed() != seed:
                raise ValueError(
                    "BAGEL generator seed does not match its request seed at index "
                    f"{request_index}"
                )
            generator_devices.append(generator_device)

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
            if len(prompts) == 1 and isinstance(batch.prompt, str):
                context = self._build_request_context(
                    batch,
                    server_args,
                    transformer,
                    special_token_ids,
                    device,
                    **context_inputs,
                )
            elif len(prompts) == 1:
                # Offline/direct callers may provide a one-item prompt list even
                # though scheduler merges always contain at least two requests.
                context = self._build_text_to_image_context(
                    batch,
                    transformer,
                    prompts[0],
                    special_token_ids,
                    device,
                )
            else:
                # Prefix lengths vary with prompt tokenization. Prefill each
                # request independently, then pack the request-major KV caches.
                contexts = [
                    self._build_text_to_image_context(
                        batch,
                        transformer,
                        prompt,
                        special_token_ids,
                        device,
                    )
                    for prompt in prompts
                ]
                context = transformer.pack_contexts(contexts)
                # Packing concatenates each request-local cache. Release the
                # source contexts before allocating initial noise and schedules.
                del contexts

        arch = server_args.pipeline_config.dit_config.arch_config
        token_height = int(batch.height) // int(arch.latent_downsample)
        token_width = int(batch.width) // int(arch.latent_downsample)
        patch_width = int(arch.latent_patch_size) ** 2 * int(arch.latent_channel)
        # Draw each request from its own RNG stream before stacking. This keeps
        # batched initial noise exactly equal to independent sequential requests.
        request_latents = [
            torch.randn(
                token_height * token_width,
                patch_width,
                generator=generator,
                device=generator_device,
                dtype=torch.float32,
            ).to(device)
            for generator, generator_device in zip(
                batch.generator, generator_devices, strict=True
            )
        ]
        latents = (
            request_latents[0]
            if len(request_latents) == 1
            else torch.stack(request_latents)
        )

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
        if batch.enable_taylorseer:
            # Taylor caches can retain several GiB of layer outputs. Keep them
            # request-owned so concurrent and sequential requests never share
            # derivatives through the resident transformer.
            batch.extra[_BAGEL_TAYLORSEER_KEY] = BagelTaylorSeerContext.create(
                num_layers=int(arch.num_hidden_layers),
                num_steps=int(batch.num_inference_steps),
                has_secondary=context.has_three_way_cfg,
            )
        else:
            batch.extra.pop(_BAGEL_TAYLORSEER_KEY, None)
        batch.latents = latents
        # Preserve the established 2D single-request contract while exposing a
        # real batch dimension to the shared denoising loop for merged requests.
        batch.raw_latent_shape = (
            (1, *latents.shape) if latents.ndim == 2 else tuple(latents.shape)
        )
        batch.n_tokens = token_height * token_width
        batch.scheduler = scheduler
        batch.timesteps = scheduler.timesteps
        batch.sigmas = scheduler.sigmas.tolist()
        batch.prompt_embeds = [torch.empty(0, device=device, dtype=torch.bfloat16)]
        batch.negative_prompt_embeds = []
        batch.do_classifier_free_guidance = False
        return batch


class BagelThinkingBeforeDenoisingStage(BagelBeforeDenoisingStage):
    """Generate a plan, rewrap it, and build official three-way T2I context."""

    allows_dynamic_batching: bool = False

    @staticmethod
    def _decode_thought(
        tokenizer,
        generated_ids: torch.Tensor,
        special_token_ids: dict[str, int],
    ) -> str:
        """Decode official BOS-prefixed, EOS-excluded planning tokens.

        Args:
            tokenizer: BAGEL checkpoint tokenizer.
            generated_ids: One-dimensional IDs returned by ``generate_text``.
            special_token_ids: Validated BAGEL token mapping.

        Returns:
            Clean planning text without message-boundary tokens.

        Raises:
            ValueError: If the generated sequence violates the BOS contract.
        """
        if generated_ids.ndim != 1 or generated_ids.numel() == 0:
            raise ValueError("BAGEL Thinking returned an empty token sequence")
        bos_token_id = special_token_ids["<|im_start|>"]
        if int(generated_ids[0].item()) != bos_token_id:
            raise ValueError("BAGEL Thinking output must begin with <|im_start|>")
        decoded = tokenizer.decode(
            generated_ids.detach().cpu().tolist(), skip_special_tokens=False
        )
        if "<|im_start|>" in decoded:
            decoded = decoded.split("<|im_start|>", 1)[1]
        return decoded.split("<|im_end|>", 1)[0]

    def _build_request_context(
        self,
        batch: Req,
        server_args: ServerArgs,
        transformer,
        special_token_ids: dict[str, int],
        device: torch.device,
        **_context_inputs,
    ):
        """Plan in a forked cache, then append the clean thought as a message."""
        system_ids = self._tokenize_prompt(
            GEN_THINK_SYSTEM_PROMPT, special_token_ids
        ).to(device)
        user_ids = self._tokenize_prompt(batch.prompt, special_token_ids).to(device)
        system_prefix, user_prefix = transformer.prepare_thinking_prefixes(
            system_ids, user_ids
        )
        # Startup warmup should exercise text decode without paying the public
        # 1000-token cap. This is request-local and does not mutate defaults.
        max_length = 2 if batch.is_warmup else int(batch.max_think_tokens)
        generated_ids = transformer.generate_text(
            user_prefix,
            bos_token_id=special_token_ids["<|im_start|>"],
            eos_token_id=special_token_ids["<|im_end|>"],
            max_length=max_length,
            do_sample=bool(batch.think_do_sample),
            temperature=float(batch.think_temperature),
            seed=int(batch.seed),
        )
        thought = self._decode_thought(self.tokenizer, generated_ids, special_token_ids)
        thought_ids = self._tokenize_prompt(thought, special_token_ids).to(device)
        context = transformer.build_thinking_context(
            system_prefix,
            user_prefix,
            thought_ids,
            height=int(batch.height),
            width=int(batch.width),
            start_of_image_token_id=special_token_ids["<|vision_start|>"],
            end_of_image_token_id=special_token_ids["<|vision_end|>"],
        )
        batch.extra[_REVISED_PROMPT_KEY] = f"{batch.prompt}\n{thought}"
        return context


class BagelUnderstandingStage(PipelineStage):
    """Encode image semantics and return BAGEL's autoregressive text answer."""

    def __init__(self, transformer, image_encoder, tokenizer) -> None:
        """Initialize immutable Understanding components.

        Args:
            transformer: Slim BAGEL UND transformer with ``lm_head``.
            image_encoder: BAGEL ViT, connector, and position embeddings.
            tokenizer: Official checkpoint tokenizer.
        """
        super().__init__()
        self.transformer = transformer
        self.image_encoder = image_encoder
        self.tokenizer = tokenizer
        self._special_token_ids: dict[str, int] | None = None

    def component_uses(
        self, server_args: ServerArgs, stage_name: str | None = None
    ) -> list[ComponentUse]:
        """Declare image encoding followed by transformer prefill/decode."""
        del server_args
        component_stage = self._component_stage_name(stage_name)
        return [
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
                phase="prefill_decode",
                target_dtype=torch.bfloat16,
                memory_intensive=True,
            ),
        ]

    def _ensure_special_tokens(self) -> dict[str, int]:
        if self._special_token_ids is None:
            self._special_token_ids = validate_bagel_special_tokens(self.tokenizer)
        return self._special_token_ids

    def _tokenize_message(
        self, text: str, special_token_ids: dict[str, int]
    ) -> torch.Tensor:
        token_ids = self.tokenizer.encode(text, add_special_tokens=False)
        return torch.tensor(
            [
                special_token_ids["<|im_start|>"],
                *token_ids,
                special_token_ids["<|im_end|>"],
            ],
            dtype=torch.long,
        )

    @staticmethod
    def _decode_response(
        tokenizer,
        generated_ids: torch.Tensor,
        special_token_ids: dict[str, int],
    ) -> str:
        """Decode official BOS-prefixed, EOS-excluded response tokens."""
        return BagelThinkingBeforeDenoisingStage._decode_thought(
            tokenizer, generated_ids, special_token_ids
        )

    @staticmethod
    def _validate_request(batch: Req) -> Image.Image:
        if not isinstance(batch.prompt, str) or not batch.prompt.strip():
            raise ValueError(
                "BAGEL Understanding requires exactly one non-empty string prompt"
            )
        if not isinstance(batch.seed, int) or isinstance(batch.seed, bool):
            raise ValueError("BAGEL Understanding requires one scalar seed")
        if batch.num_outputs_per_prompt != 1:
            raise ValueError(
                "BAGEL Understanding supports num_outputs_per_prompt=1 only"
            )
        if batch.num_frames != 1:
            raise ValueError("BAGEL Understanding supports num_frames=1 only")
        if batch.negative_prompt not in (None, ""):
            raise ValueError("BAGEL Understanding does not support negative_prompt")
        if batch.true_cfg_scale is not None:
            raise ValueError("BAGEL Understanding does not support true_cfg_scale")
        images = (
            batch.condition_image
            if isinstance(batch.condition_image, list)
            else [batch.condition_image]
        )
        if batch.image_path is None or len(images) != 1:
            raise ValueError("BAGEL Understanding requires exactly one input image")
        image = images[0]
        if not isinstance(image, Image.Image):
            raise ValueError("BAGEL Understanding input must decode to a PIL image")
        if batch.extra.get("dynamic_batch_seeds") is not None:
            raise ValueError("BAGEL Understanding dynamic batching is not supported")
        if batch.rollout:
            raise ValueError("BAGEL Understanding rollout mode is not supported")
        if batch.return_trajectory_latents or batch.return_trajectory_decoded:
            raise ValueError("BAGEL Understanding trajectory output is not supported")
        if (
            batch.save_output
            or batch.return_file_paths_only
            or batch.return_frames
            or batch.return_raw_frames
        ):
            raise ValueError(
                "BAGEL Understanding returns text directly and does not support "
                "media or file output controls"
            )
        return image

    @torch.no_grad()
    def forward(self, batch: Req, server_args: ServerArgs) -> OutputBatch:
        """Run official image-to-text Understanding without diffusion stages.

        Args:
            batch: Validated request with one resized PIL image.
            server_args: Runtime configuration.

        Returns:
            Text-only output with usage and finish metadata.

        Raises:
            ValueError: If the request violates the one-image contract.
            RuntimeError: If required model components are unavailable.
        """
        del server_args
        image = self._validate_request(batch)
        special_token_ids = self._ensure_special_tokens()

        with self.use_declared_component(
            component_name="image_encoder",
            module=self.image_encoder,
            phase="encode",
        ) as image_encoder:
            assert image_encoder is not None
            self.image_encoder = image_encoder
            # Native attention layers read their backend metadata from the
            # standard forward context, even though this stage does not denoise.
            with set_forward_context(current_timestep=0, attn_metadata=None):
                vision_embeddings = image_encoder.encode_image(image)

        with self.use_declared_component(
            component_name="transformer",
            module=self.transformer,
            phase="prefill_decode",
        ) as transformer:
            assert transformer is not None
            self.transformer = transformer
            device = transformer.device
            user_input_ids = self._tokenize_message(batch.prompt, special_token_ids).to(
                device
            )
            system_input_ids = None
            if batch.enable_thinking:
                system_input_ids = self._tokenize_message(
                    VLM_THINK_SYSTEM_PROMPT, special_token_ids
                ).to(device)
            prefix = transformer.build_understanding_prefix(
                vision_embeddings,
                user_input_ids,
                system_input_ids=system_input_ids,
                start_of_image_token_id=special_token_ids["<|vision_start|>"],
                end_of_image_token_id=special_token_ids["<|vision_end|>"],
            )
            max_length = 2 if batch.is_warmup else int(batch.max_new_tokens)
            generated_ids, finish_reason = transformer.generate_text(
                prefix,
                bos_token_id=special_token_ids["<|im_start|>"],
                eos_token_id=special_token_ids["<|im_end|>"],
                max_length=max_length,
                do_sample=bool(batch.do_sample),
                temperature=float(batch.temperature),
                seed=int(batch.seed),
                return_finish_reason=True,
            )

        response_text = self._decode_response(
            self.tokenizer, generated_ids, special_token_ids
        )
        return OutputBatch(
            text_outputs=[
                TextGenerationOutput(
                    text=response_text,
                    prompt_tokens=prefix.kv_cache.sequence_length,
                    completion_tokens=max(0, int(generated_ids.numel()) - 1),
                    finish_reason=finish_reason,
                )
            ],
            metrics=batch.metrics,
        )


class BagelEditBeforeDenoisingStage(BagelBeforeDenoisingStage):
    """Encode one source image and build BAGEL's three Editing CFG prefixes."""

    allows_dynamic_batching: bool = False
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
            # Keep Editing's custom conditioning stage compatible with the same
            # native attention contract used by generic image-encoding stages.
            with set_forward_context(current_timestep=0, attn_metadata=None):
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
