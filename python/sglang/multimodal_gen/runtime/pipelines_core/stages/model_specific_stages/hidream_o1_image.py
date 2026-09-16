# SPDX-License-Identifier: Apache-2.0
"""Model-specific stages for HiDream-O1-Image.

HiDream-O1-Image is a Qwen3-VL backbone that predicts clean 32x32 raw pixel
patches, so it has neither a VAE nor a separate text encoder. The
before-denoising stage therefore produces token ids, interleaved-mrope positions
and a mixed causal/bidirectional attention mask instead of prompt embeddings,
the denoising stage converts the predicted clean image into a flow velocity, and
decoding is a pure unpatchify.
"""

import torch

from sglang.multimodal_gen.configs.pipeline_configs.hidream_o1_image import (
    HIDREAM_O1_COND_KEY,
    HIDREAM_O1_NEG_COND_KEY,
)
from sglang.multimodal_gen.runtime.disaggregation.roles import RoleType
from sglang.multimodal_gen.runtime.pipelines_core.schedule_batch import OutputBatch, Req
from sglang.multimodal_gen.runtime.pipelines_core.stages.base import PipelineStage
from sglang.multimodal_gen.runtime.pipelines_core.stages.denoising import (
    DenoisingContext,
    DenoisingStage,
    DenoisingStepState,
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

HIDREAM_O1_BOI_TOKEN = "<|boi_token|>"
HIDREAM_O1_TMS_TOKEN = "<|tms_token|>"

# Reference models/pipeline.py:15 NOISE_SCALE. The reference CLI's 7.5 default is
# a flash-scheduler-only override and never reaches this FlowUniPC path.
HIDREAM_O1_NOISE_SCALE = 8.0
# Sigma floor so the clean-image -> velocity conversion never divides by zero.
HIDREAM_O1_T_EPS = 0.001
# Image mrope positions start at a fixed offset instead of continuing the text
# positions, so the grid does not shift when the prompt length changes.
HIDREAM_O1_MROPE_FIX_POINT = 4096


def _resolve_single_prompt(prompt, *, field_name: str) -> str:
    if isinstance(prompt, str):
        return prompt
    if isinstance(prompt, list) and len(prompt) == 1 and isinstance(prompt[0], str):
        return prompt[0]
    raise ValueError(
        f"HiDream-O1-Image serves one prompt per request; {field_name} must be a "
        f"single string, got {prompt!r}"
    )


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


class HiDreamO1ImageBeforeDenoisingStage(PipelineStage):
    def __init__(
        self,
        tokenizer,
        processor,
        scheduler,
        *,
        vision_start_token_id: int,
        tms_token_id: int,
    ) -> None:
        super().__init__()
        self.tokenizer = tokenizer
        self.processor = processor
        self.scheduler = scheduler
        self.vision_start_token_id = vision_start_token_id
        self.tms_token_id = tms_token_id

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
    ) -> torch.Tensor:
        # CPU generators seeded with seed + 1 keep the noise bit-identical to the
        # reference implementation regardless of the accelerator in use.
        noise = torch.cat(
            [
                HIDREAM_O1_NOISE_SCALE
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
        arch_config = server_args.pipeline_config.dit_config.arch_config
        patch_size = arch_config.patch_size
        device = self.device
        height, width = batch.height, batch.width
        if height % patch_size != 0 or width % patch_size != 0:
            raise ValueError(
                f"height and width must be divisible by {patch_size}; got "
                f"{height} and {width}"
            )
        h_patches, w_patches = height // patch_size, width // patch_size
        batch_size = len(batch.seeds)
        dit_dtype = resolve_precision(
            server_args, "dit", precision_attr="dit_precision"
        )

        # These post-process the prediction non-linearly, and this stage's
        # denoising loop guides on x_pred rather than on the velocity, so the
        # two orderings are not equivalent for them.
        if batch.guidance_rescale > 0.0 or (
            batch.cfg_normalization and float(batch.cfg_normalization) > 0
        ):
            raise ValueError(
                "HiDream-O1-Image applies classifier-free guidance to the "
                "predicted clean image, so --guidance-rescale and "
                "--cfg-normalization would act on x_pred instead of on the "
                "flow velocity they were designed for. Leave both at 0."
            )

        prompt = _resolve_single_prompt(batch.prompt, field_name="prompt")
        cond = self._build_conditioning(
            prompt,
            batch_size=batch_size,
            h_patches=h_patches,
            w_patches=w_patches,
            mask_dtype=dit_dtype,
            device=device,
        )
        neg_cond = None
        if batch.do_classifier_free_guidance:
            negative_prompt = _resolve_single_prompt(
                batch.negative_prompt, field_name="negative_prompt"
            )
            neg_cond = self._build_conditioning(
                negative_prompt,
                batch_size=batch_size,
                h_patches=h_patches,
                w_patches=w_patches,
                mask_dtype=dit_dtype,
                device=device,
            )

        latents = self._prepare_latents(
            seeds=batch.seeds,
            height=height,
            width=width,
            patch_size=patch_size,
            channels=arch_config.in_channels,
            dtype=dit_dtype,
            device=device,
        )

        scheduler = self.scheduler
        scheduler.set_timesteps(batch.num_inference_steps, device=device)

        batch.extra[HIDREAM_O1_COND_KEY] = cond
        batch.extra[HIDREAM_O1_NEG_COND_KEY] = neg_cond
        # The backbone consumes token ids, so these only exist to satisfy the
        # denoising stage's contract; the DiT ignores encoder_hidden_states.
        batch.prompt_embeds = [cond["input_ids"]]
        batch.negative_prompt_embeds = (
            [] if neg_cond is None else [neg_cond["input_ids"]]
        )
        batch.latents = latents
        batch.timesteps = scheduler.timesteps
        batch.scheduler = scheduler
        batch.sigmas = scheduler.sigmas.tolist()
        batch.raw_latent_shape = latents.shape
        batch.height = height
        batch.width = width
        return batch

    def verify_output(self, batch: Req, server_args: ServerArgs) -> VerificationResult:
        result = VerificationResult()
        result.add_check("latents", batch.latents, [V.is_tensor, V.with_dims(3)])
        result.add_check("timesteps", batch.timesteps, V.is_tensor)
        return result


class HiDreamO1ImageDenoisingStage(DenoisingStage):
    """Denoising loop for a model that predicts the clean image, not the noise.

    The DiT returns ``x_pred``, so classifier-free guidance is applied to
    ``x_pred`` and only then converted to the flow velocity the scheduler
    expects. That reordering is exact because the conversion is affine in
    ``x_pred`` and both branches share ``sample`` and ``sigma``; it holds only
    while ``sigma`` is the scheduler's own sigma for the step and while the
    non-linear CFG post-processing knobs stay off, which the before-denoising
    stage enforces.
    """

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
        # The pixel head is conditioned on the flow position (1 -> 0), which runs
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
        if server_args.comfyui_mode:
            batch.noise_pred = x_pred

        with maybe_nvtx_range("scheduler_step", use_nvtx):
            sample = ctx.latents.float()
            # Exact inverse of the scheduler's own convert_model_output, which
            # recovers x_pred as ``sample - sigma * model_output``.
            model_output = (sample - x_pred.float()) / sigma.clamp_min(HIDREAM_O1_T_EPS)
            ctx.latents = ctx.scheduler.step(
                model_output=model_output,
                timestep=step.t_device,
                sample=sample,
                **ctx.extra_step_kwargs,
                return_dict=False,
            )[0].to(latents_dtype)


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
