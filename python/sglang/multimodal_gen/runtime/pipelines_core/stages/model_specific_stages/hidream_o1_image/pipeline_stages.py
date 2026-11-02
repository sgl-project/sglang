# Adapted from the official HiDream-O1-Image repository (MIT License):
# https://github.com/HiDream-ai/HiDream-O1-Image
"""Model-specific stages for HiDream-O1-Image.

HiDream-O1-Image is a Qwen3-VL backbone that predicts clean 32x32 raw pixel
patches, so it has neither a VAE nor a separate text encoder. The
before-denoising stage therefore produces token ids, interleaved-mrope positions
and reference conditioning instead of prompt embeddings, the denoising stage
converts the predicted clean image into a flow velocity, and decoding is a pure
unpatchify.

All checkpoint variants share this construction; the variant resolved from the
served model path only selects the recipe (scheduler, steps, noise, dtype).
"""

from __future__ import annotations

import os
from typing import Any

import torch
from cache_dit import ForwardPattern
from PIL import Image
from sglang.multimodal_gen import envs
from sglang.multimodal_gen.configs.sample.hidream_o1_image import (
    _VARIANT_DEFAULTS,
    _resolve_variant,
)
from sglang.multimodal_gen.runtime.cache.cache_dit_integration import (
    _CUSTOM_BLOCK_ADAPTER_SPECS,
    CacheDitConfig,
    CustomBlockAdapterSpec,
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
from sglang.multimodal_gen.runtime.models.dits.hidream_o1_image import (
    O1CacheBlockAdapter,
)
from sglang.multimodal_gen.runtime.models.schedulers.scheduling_flow_unipc_multistep import (
    FlowUniPCMultistepScheduler,
)
from sglang.multimodal_gen.runtime.pipelines_core.diffusion_scheduler_utils import (
    clone_scheduler_runtime,
)
from sglang.multimodal_gen.runtime.pipelines_core.schedule_batch import (
    OutputBatch,
    Req,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.base import PipelineStage
from sglang.multimodal_gen.runtime.pipelines_core.stages.denoising import (
    DenoisingStage,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.hidream_o1_image.generation import (
    _HIDREAM_CACHE_DIT_RDT,
    DEFAULT_TIMESTEPS,
    NOISE_SCALE,
    PATCH_SIZE,
    REROLL_MAX_ATTEMPTS,
    REROLL_OUT_OF_RANGE_THRESHOLD,
    T_EPS,
    FlashFlowMatchScheduler,
    _chunk_items,
    _generation_work,
    _max_forward_rows,
    _patchify_image_tensor,
    _ref_tensor_transform,
    _stack_cond_samples,
    build_scheduler,
    build_t2i_text_sample,
    build_ti2i_text_sample,
    calculate_dimensions,
    find_closest_resolution,
    patchify_pixels,
    resize_pilimage,
    unpatchify_pixels,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.hidream_o1_image.layout import (
    create_layout_reference_images,
    load_layout_bboxes,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.hidream_o1_image.prompt_enhance import (
    enhance_prompt,
)
from sglang.multimodal_gen.runtime.server_args import ServerArgs
from sglang.multimodal_gen.runtime.utils.logging_utils import init_logger
from sglang.multimodal_gen.runtime.utils.precision import resolve_precision

logger = init_logger(__name__)

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


def _resolve_trajectory_scheduler(ctx: dict, device: torch.device, template=None):
    """Three-layer scheduler strategy: recipe default -> editing override
    -> explicit request choice. Returns (scheduler, inject_noise).

    The pipeline-built scheduler module is the recipe default and serves as
    the template; a fresh copy per trajectory resets the step counter.
    Task-specific Dev schedulers (editing flow_match, fallbacks, or an
    explicit shift override) are built via the official build_scheduler.
    """
    scheduler_name = ctx["scheduler_name"]
    if ctx["model_type"] == "full":
        # Official: the editing override is "Ignored for full model".
        scheduler_name = "default"
    elif (
        ctx["is_editing"]
        and "scheduler_name" not in ctx["explicit_fields"]
        and scheduler_name != "flow_match"
    ):
        logger.info(
            "Editing: scheduler %s -> flow_match (official editing default)",
            scheduler_name,
        )
        scheduler_name = "flow_match"
    elif scheduler_name not in ("flash", "flow_match"):
        logger.warning(
            "Unsupported scheduler_name=%s for the DEV recipe; using flash.",
            scheduler_name,
        )
        scheduler_name = "flash"
    inject_noise = scheduler_name == "flash"
    template_kind = (
        "default"
        if isinstance(template, FlowUniPCMultistepScheduler)
        else "flash"
        if isinstance(template, FlashFlowMatchScheduler)
        else None
    )
    if template_kind == scheduler_name and (
        scheduler_name != "default" or ctx["shift"] == ctx["recipe_shift"]
    ):
        sched = clone_scheduler_runtime(template)
    else:
        sched = build_scheduler(
            ctx["num_steps"], DEFAULT_TIMESTEPS, ctx["shift"], device, scheduler_name
        )
    if scheduler_name == "default":
        sched.set_timesteps(ctx["num_steps"], device=device)
    else:
        sched.set_timesteps_list(DEFAULT_TIMESTEPS, device)
    return sched, inject_noise


class HiDreamO1ImageBeforeDenoisingStage(PipelineStage):
    """Prompt conditioning, reference conditioning and patchified initial noise."""

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
        self._cache_dit_adapter: O1CacheBlockAdapter | None = None
        self._cache_dit_enabled = False
        self._cache_dit_active_key: tuple | None = None
        self._cache_dit_request_overrides: dict = {}

    @property
    def role_affinity(self) -> RoleType:
        return RoleType.DENOISER

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
        noise_scale: float | None = None,
    ) -> torch.Tensor:
        """Per-row patchified noise; CPU generators seeded with seed + 1 keep the
        draw bit-identical to the reference implementation on any accelerator."""
        scale = NOISE_SCALE if noise_scale is None else noise_scale
        noise = torch.cat(
            [
                scale
                * torch.randn(
                    (1, channels, height, width),
                    generator=torch.Generator("cpu").manual_seed(int(seed) + 1),
                )
                for seed in seeds
            ]
        )
        noise = noise.to(device=device, dtype=dtype)
        return patchify_pixels(noise, patch_size)

    def _warn_short_prompts(self, prompts: list[str]) -> None:
        # Short prompts under-condition the model (uncolored line art);
        # remedy: the Prompt Agent.
        for p in prompts:
            n = len(p.strip())
            if n < 20:  # arbitrary threshold
                logger.warning(
                    "Short T2I prompt (%d chars): output may be uncolored "
                    "line art; refine it or enable --pe-server-url.",
                    n,
                )

    def _enhance_prompts(self, batch: Req, server_args: ServerArgs) -> list[str]:
        prompts = batch.prompt if isinstance(batch.prompt, list) else [batch.prompt]
        ref_images_batch = getattr(batch, "image_path", None)
        if (
            server_args.pe_server_url
            and not ref_images_batch
            and not getattr(batch, "is_warmup", False)
        ):
            return [
                enhance_prompt(prompt=p, base_url=server_args.pe_server_url)
                for p in prompts
            ]
        if not ref_images_batch:
            self._warn_short_prompts(prompts)
        return prompts

    def _build_ctx(self, batch: Req, server_args: ServerArgs) -> dict:
        """Shared per-request settings consumed by every work item."""
        sampling_params = getattr(batch, "sampling_params", None)
        requested_steps = int(
            getattr(sampling_params, "num_inference_steps", None)
            or getattr(batch, "num_inference_steps", None)
            or getattr(server_args.pipeline_config, "num_inference_steps", None)
            or len(DEFAULT_TIMESTEPS)
        )
        # Dev checkpoints always run the fixed 28-step schedule; the full
        # checkpoint honors the requested (reference 50-step) schedule.
        variant = _resolve_variant(server_args.model_path)
        model_type = "full" if variant == "full" else "dev"
        if model_type == "full":
            num_steps = requested_steps
        else:
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
            # Recipe shift the pipeline-built scheduler template carries; an
            # explicit request shift different from it rebuilds the scheduler.
            "recipe_shift": _VARIANT_DEFAULTS[variant]["shift"],
            "layout_bboxes": batch.layout_bboxes,
            "explicit_fields": set(
                getattr(sampling_params, "_explicit_fields", ()) or ()
            ),
            "tokenizer": tokenizer,
            "model_config": server_args.pipeline_config.dit_config.arch_config,
            "pad_token_id": int(pad_token_id),
            "max_forward_rows": _max_forward_rows(server_args),
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
                cond_img_size = 384
            elif K <= 8:
                cond_img_size = 384 * 48 // 64
            else:
                cond_img_size = 384 // 2
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
        sched, inject_noise = _resolve_trajectory_scheduler(ctx, device, self.scheduler)

        # Main's noise construction: CPU generators seeded with seed + 1, one
        # per row, scaled by the recipe's initial noise scale.
        first = items[0]
        height, width = first["height"], first["width"]
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

        cond_sample = first["cond_sample"]
        batch.extra["hidream_o1_ctx"] = ctx
        batch.extra["hidream_o1_items"] = items
        batch.extra["hidream_o1_inject_noise"] = inject_noise
        # The backbone consumes token ids, so these only exist to satisfy the
        # denoising stage's contract; the DiT ignores encoder_hidden_states.
        batch.prompt_embeds = [cond_sample["input_ids"]]
        batch.latents = latents
        batch.timesteps = sched.timesteps
        batch.scheduler = sched
        batch.sigmas = sched.sigmas.tolist()
        batch.raw_latent_shape = latents.shape
        batch.height = height
        batch.width = width
        return batch

    def verify_output(self, batch: Req, server_args: ServerArgs):
        from sglang.multimodal_gen.runtime.pipelines_core.stages.validators import (
            V,
            VerificationResult,
        )

        result = VerificationResult()
        result.add_check("latents", batch.latents, [V.is_tensor, V.with_dims(3)])
        result.add_check("timesteps", batch.timesteps, V.is_tensor)
        return result

    # ---- cache-dit wiring (shared with the denoising stage via ctx) ----

    def _cache_dit_requested(self) -> bool:
        """Request-independent server default (init-time decisions only)."""
        return envs.SGLANG_CACHE_DIT_ENABLED

    def _cache_dit_requested_for_batch(self, batch: Req) -> bool:
        """Per-request Cache-DiT switch; the server default applies when unset."""
        enable = batch.sampling_params.enable_cache_dit
        if enable is None:
            return self._cache_dit_requested()
        return bool(enable)


class HiDreamO1ImageDenoisingStage(DenoisingStage):
    """Denoising loop for a model that predicts the clean image, not the noise.

    The DiT returns ``x_pred`` per row; classifier-free guidance is applied to
    the velocity the scheduler expects. Co-batching: same-fingerprint requests
    share one trajectory; rows keep private CPU generator streams.
    """

    def __init__(self, model: torch.nn.Module, scheduler, processor: Any) -> None:
        # Route the model through the base init so the framework's compile/
        # offload and attention-backend wiring see the transformer.
        super().__init__(transformer=model, scheduler=scheduler)
        self.model = model
        self.processor = processor

    @property
    def role_affinity(self) -> RoleType:
        return RoleType.DENOISER

    @torch.no_grad()
    def forward(self, batch: Req, server_args: ServerArgs) -> Req:
        ctx = batch.extra["hidream_o1_ctx"]
        items = batch.extra["hidream_o1_items"]
        chunks = _chunk_items(items, ctx["max_forward_rows"])
        z_rows: list = [None] * len(items)
        for chunk in chunks:
            for item, z in zip(chunk, self._run_trajectory(chunk, ctx)):
                z_rows[item["index"]] = z
        batch.latents = torch.cat(z_rows, dim=0)
        return batch

    def run_grouped_requests(self, batches: list[Req], server_args: ServerArgs) -> list:
        """Co-batch same-fingerprint requests into shared trajectories."""
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
        ctx = batch.extra["hidream_o1_ctx"]
        return (
            int(batch.height),
            int(batch.width),
            ctx["num_steps"],
            ctx["guidance_scale"],
            ctx["scheduler_name"],
            ctx["shift"],
            ctx["noise_scale_start"],
            ctx["noise_scale_end"],
            ctx["noise_clip_std"],
            bool(batch.image_path),
            self.freeze_for_dedup(batch.layout_bboxes),
            tuple(sorted(ctx["explicit_fields"])),
            ctx["max_forward_rows"],
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
        ctx = first.extra["hidream_o1_ctx"]
        items = []
        owner = []
        for idx in indices:
            batch = batches[idx]
            for item in batch.extra["hidream_o1_items"]:
                items.append(item)
                owner.append(idx)
        for i, item in enumerate(items):
            item["index"] = i
        z_rows: list = [None] * len(items)
        for chunk in _chunk_items(items, ctx["max_forward_rows"]):
            for item, z in zip(chunk, self._run_trajectory(chunk, ctx)):
                z_rows[item["index"]] = z
        per_batch: dict[int, list] = {}
        for idx, z in zip(owner, z_rows):
            per_batch.setdefault(idx, []).append(z)
        for idx in indices:
            batches[idx].latents = torch.cat(per_batch[idx], dim=0)
        results.update({idx: batches[idx] for idx in indices})

    def _run_trajectory(self, chunk: list[dict], ctx: dict) -> list[torch.Tensor]:
        """Run one batched denoising trajectory for same-shape work items."""
        device = self.device
        dit_dtype = chunk[0]["ref_patches"].dtype if chunk[0]["ref_patches"] is not None else torch.bfloat16
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

        sched, inject_noise = _resolve_trajectory_scheduler(ctx, device, self.scheduler)
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
            patches = noise.reshape(
                1, 3, h_patches, PATCH_SIZE, w_patches, PATCH_SIZE
            )
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
        use_cache_dit = self._maybe_enable_cache_dit(ctx, num_steps, reroll_enabled)
        cache_adapter = self._cache_dit_adapter if use_cache_dit else None

        precomputed = (None, None)
        for step_idx, step_t in enumerate(sched.timesteps):
            t_pixeldit = 1.0 - step_t.float() / 1000.0
            sigma = (step_t.float() / 1000.0).clamp_min(T_EPS)

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
                    dtype=torch.bfloat16,
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
                ).to(dit_dtype)
            else:
                step_out = sched.step(
                    model_output.float(),
                    step_t.to(dtype=torch.float32),
                    z.float(),
                    return_dict=False,
                )
                # the framework UniPC returns (sample,); the Dev schedulers
                # return the sample tensor directly
                z = (
                    step_out[0] if isinstance(step_out, tuple) else step_out
                ).to(dit_dtype)

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

    # ---- cache-dit (moved verbatim from the merged stage) ----

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
        # HiDream-specific fallback: tighter residual threshold unless the
        # operator set SGLANG_CACHE_DIT_RDT explicitly (presence check follows
        # the secondary-env inheritance pattern in envs.py).
        rdt_default = (
            envs.SGLANG_CACHE_DIT_RDT
            if os.getenv("SGLANG_CACHE_DIT_RDT") is not None
            else _HIDREAM_CACHE_DIT_RDT
        )
        return CacheDitConfig(
            enabled=True,
            Fn_compute_blocks=knob("Fn_compute_blocks", envs.SGLANG_CACHE_DIT_FN),
            Bn_compute_blocks=knob("Bn_compute_blocks", envs.SGLANG_CACHE_DIT_BN),
            max_warmup_steps=knob("max_warmup_steps", envs.SGLANG_CACHE_DIT_WARMUP),
            residual_diff_threshold=knob("residual_diff_threshold", rdt_default),
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
        """Mount cache-dit; returns whether forwards route through the adapter.
        Not mounted during TI2I reroll: extra step-0 forwards desync the counter."""
        requested = ctx["cache_dit_requested"]
        if ctx["enable_breakable_cuda_graph"]:
            # cache-dit's step-skipping flow must not be baked into a CUDA graph.
            if requested:
                logger.warning_once(
                    "Cache-DiT was requested but is disabled because breakable "
                    "CUDA graphs are enabled."
                )
            return False
        if reroll_enabled:
            # TI2I reroll adds step-0 forwards that desync cache-dit's step counter.
            logger.info(
                "cache-dit skipped for this trajectory: the deterministic TI2I "
                "seed reroll adds step-0 forwards."
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
        # Each chunk runs its own step loop; refresh the counter per request.
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
        # Skip ordinary warmup, but let torch.compile warmup mount before Dynamo traces.
        if ctx["is_warmup"] and not ctx["enable_torch_compile"]:
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

        self._register_o1_cache_dit_spec()
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

    def _register_o1_cache_dit_spec(self) -> None:
        """Register the adapter explicitly: its name matches no cache-dit family prefix."""
        _CUSTOM_BLOCK_ADAPTER_SPECS.setdefault(
            O1CacheBlockAdapter.__name__,
            CustomBlockAdapterSpec(
                blocks_attr="blocks", forward_pattern=ForwardPattern.Pattern_3
            ),
        )

    def _forward_batch(
        self,
        sample: dict,
        vinputs: torch.Tensor,
        t_pixeldit: torch.Tensor,
        precomputed: tuple,
        cache_adapter: O1CacheBlockAdapter | None = None,
    ) -> tuple[torch.Tensor, tuple | None]:
        """Run one (possibly batched) forward; returns (x_pred, embeds).
        cache_adapter wraps only the decoder block loop, not the conditioning."""
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


class HiDreamO1ImageDecodingStage(PipelineStage):
    """Unpatchify the denoised pixel patches; the model ships no VAE."""

    @property
    def role_affinity(self) -> RoleType:
        return RoleType.DECODER

    def _decode(
        self, patches: torch.Tensor, batch: Req, server_args: ServerArgs
    ) -> torch.Tensor:
        from sglang.multimodal_gen.configs.pipeline_configs.hidream_o1_image import (
            arch_config_of,
        )

        arch_config = arch_config_of(server_args)
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
        return OutputBatch(output=frames)
