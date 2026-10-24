# SPDX-License-Identifier: Apache-2.0
"""Nomas stages: event planning, text conditioning, segment denoising, streaming decode.

Offline requests run every event in one pass; realtime chunks run one event
against the session state kept on the worker.
"""

import math

import torch
import torch.distributed as dist
from msgspec.structs import replace

from sglang.multimodal_gen.runtime.disaggregation.roles import RoleType
from sglang.multimodal_gen.runtime.distributed.parallel_state import (
    get_local_torch_device,
    get_replica_group,
    model_parallel_is_initialized,
)
from sglang.multimodal_gen.runtime.managers.memory_managers.component_manager import (
    ComponentUse,
)
from sglang.multimodal_gen.runtime.models.vaes.nomas import NomasStreamingDecoder
from sglang.multimodal_gen.runtime.pipelines_core.schedule_batch import OutputBatch
from sglang.multimodal_gen.runtime.pipelines_core.stages import TextEncodingStage
from sglang.multimodal_gen.runtime.pipelines_core.stages.base import PipelineStage
from sglang.multimodal_gen.runtime.pipelines_core.stages.decoding import (
    DecodingStage,
    _to_unit_range,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.latent_preparation import (
    LatentPreparationStage,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.nomas.denoise_loop import (
    NomasLatentSession,
    NomasSegmentCondition,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.nomas.request_validation import (
    action_metadata,
    cache_budget_bytes,
    cache_bytes_per_token,
    canonical_caption,
    check_cache_budget,
    plan_segments,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.realtime.input_validation import (
    RealtimeInputValidationStage,
)
from sglang.multimodal_gen.runtime.realtime.nomas import NOMAS_FLUSH_CONDITION
from sglang.multimodal_gen.runtime.realtime.states.nomas import NomasRealtimeState
from sglang.multimodal_gen.runtime.utils.logging_utils import init_logger
from sglang.multimodal_gen.runtime.utils.precision import resolve_decode_precision

logger = init_logger(__name__)


def _is_flush(batch) -> bool:
    return bool(batch.condition_inputs.get(NOMAS_FLUSH_CONDITION))


class NomasInputValidationStage(RealtimeInputValidationStage):
    """Offline: plan all events. Realtime: one event per chunk, generator reused."""

    def forward(self, batch, server_args):
        batch = super().forward(batch, server_args)
        for name, requested in (("height", batch.height), ("width", batch.width)):
            aligned = ((requested + 15) // 16 * 16) // 32 * 32
            if aligned < 32:
                raise ValueError("Nomas dimensions must align to at least 32 pixels")
            if aligned != requested:
                logger.info("Nomas %s aligned from %d to %d", name, requested, aligned)
            setattr(batch, name, aligned)
        flush = _is_flush(batch)
        plan = plan_segments(
            batch.num_frames,
            batch.prompt,
            batch.segments,
            batch.continuation_prompts,
            server_args.pipeline_config.segment_latent_frames,
        )
        prompt_format = "plain" if flush else batch.prompt_format
        plan = [
            replace(p, caption=canonical_caption(p.caption, prompt_format))
            for p in plan
        ]
        if batch.session is not None and len(plan) != 1:
            raise ValueError("a Nomas realtime chunk carries exactly one event segment")
        batch.extra["nomas_segments"] = plan
        batch.prompt = plan[0].caption
        if prompt_format == "plain" and not flush:
            logger.warning(
                "Nomas plain captions are an experimental shortcut; pretrained validation requires prompt_format=json"
            )
        return batch


class NomasLatentPreparationStage(LatentPreparationStage):
    """Use shared shape/dtype preparation; online noise comes from the session bank."""

    def forward(self, batch, server_args):
        if batch.session is None:
            return super().forward(batch, server_args)
        frames = self.get_forward_latent_num_frames(batch, server_args)
        spec = self.get_latent_preparation_spec(
            batch, server_args, batch.batch_size, frames, get_local_torch_device()
        )
        # Only shape, device and dtype are consumed before _noise_slice replaces
        # this tensor. A scalar view avoids both a throwaway RNG draw and a full
        # latent allocation; it must never be used as generated noise.
        batch.latents = torch.zeros((), device=spec.device, dtype=spec.dtype).expand(
            spec.shape
        )
        batch.raw_latent_shape = batch.latents.shape
        return batch


class NomasTextEncodingStage(TextEncodingStage):
    """Use the shared tokenizer/encoder/cache for each event caption."""

    deduplicated_extra_output_keys = ("nomas_conditions",)

    def build_dedup_fingerprint(self, batch, server_args):
        return (
            super().build_dedup_fingerprint(batch, server_args),
            self.freeze_for_dedup(batch.continuation_prompts),
            self.freeze_for_dedup(batch.segments),
            batch.prompt_format,
            batch.fps,
            batch.num_frames,
            _is_flush(batch),
            server_args.pipeline_config.segment_latent_frames,
        )

    def forward(self, batch, server_args):
        config = server_args.pipeline_config
        plan = batch.extra["nomas_segments"]
        structured = batch.prompt_format == "json" and not _is_flush(batch)
        batch = super().forward(batch, server_args)
        conditions = []
        for index, segment in enumerate(plan):
            if index == 0:
                embeddings, mask = batch.prompt_embeds[0], batch.prompt_embeds_mask[0]
            else:
                encoded, _, _, masks, _ = self.encode_text(
                    segment.caption,
                    server_args,
                    encoder_index=0,
                    return_attention_mask=True,
                    max_length=batch.max_sequence_length,
                )
                embeddings, mask = encoded[0], masks[0]
            condition = NomasSegmentCondition(embeddings, mask.bool())
            if structured:
                action_start, offsets, windows = action_metadata(
                    segment.caption,
                    config,
                    self.tokenizers[0],
                    condition.text_mask,
                    segment.end - segment.start,
                    batch.fps,
                )
                condition = replace(
                    condition,
                    action_start=action_start,
                    text_offsets=offsets,
                    video_windows=windows,
                )
            conditions.append(condition)
        batch.extra["nomas_conditions"] = conditions
        return batch


def segment_camera_condition(condition, camera_embedding, start, end):
    """Slice the whole-clip camera tokens; never gate a camera-on segment.

    Upstream passes has_camera_mask=None unless a segment disables camera.
    A ones mask is not equivalent: the gate multiplies by a float tensor and
    promotes the residual stream to FP32 for every later block.
    """
    if camera_embedding is None:
        return condition
    return replace(
        condition,
        camera_embedding=camera_embedding[:, start:end],
        camera_has_mask=None,
    )


def _canonical_camera(batch, latent_frames):
    """Finite [(L-1)*4+1, 4, 4] normalized poses for the latents of this request, or None."""
    if batch.norm_extrinsics is None:
        return None
    camera = torch.as_tensor(
        batch.norm_extrinsics, device=batch.latents.device, dtype=torch.float32
    )
    expected = (latent_frames - 1) * 4 + 1
    if camera.shape != (expected, 4, 4) or not torch.isfinite(camera).all():
        raise ValueError(
            f"norm_extrinsics must be finite [{expected},4,4] on the canonical (L-1)*4+1 timeline"
        )
    return camera


def _request_generator(batch):
    return batch.generator[0] if isinstance(batch.generator, list) else batch.generator


class NomasDenoisingStage(PipelineStage):
    def __init__(self, transformer):
        super().__init__()
        self.transformer = transformer

    @property
    def role_affinity(self):
        return RoleType.DENOISER

    def component_uses(self, server_args, stage_name=None):
        return [
            ComponentUse(
                self._component_stage_name(stage_name),
                "transformer",
                target_dtype=torch.bfloat16,
            )
        ]

    @torch.inference_mode()
    def forward(self, batch, server_args):
        if batch.session is not None:
            return self._forward_session(batch, server_args)
        config = server_args.pipeline_config
        latent = batch.latents
        frames = latent.shape[2]
        plan = batch.extra["nomas_segments"]
        conditions = batch.extra.pop("nomas_conditions")
        # The final offline segment does not need persistent KV storage.
        capacity = sum(
            (segment.end - segment.start) * math.prod(latent.shape[3:])
            + condition.text_mask.shape[1]
            for segment, condition in zip(plan[:-1], conditions[:-1], strict=True)
        )
        camera = _canonical_camera(batch, frames)
        generator = _request_generator(batch)
        with self.use_declared_component(
            component_name="transformer", module=self.transformer
        ) as model:
            check_cache_budget(
                config,
                capacity,
                free_bytes=_free_bytes(latent.device),
                head_shards=model.parallel.head_shards,
            )
            # Pad three first poses once for the whole clip, then slice embeddings.
            camera_embedding = (
                model.embed_camera(camera[None]) if camera is not None else None
            )
            session = self._new_session(model, batch, config, capacity, generator)
            segments = []
            for index, (condition, segment) in enumerate(
                zip(conditions, plan, strict=True)
            ):
                initial = latent[:, :, segment.start : segment.end]
                condition = segment_camera_condition(
                    condition, camera_embedding, segment.start, segment.end
                )
                segments.append(
                    session.generate_segment(
                        tuple(initial.shape[2:]),
                        condition,
                        initial_latents=initial,
                        cache_segment=index < len(plan) - 1,
                    )
                )
            batch.latents = torch.cat(segments, dim=2)
            logger.info(
                "Nomas: %d segments, %d cached tokens, %d latent frames",
                len(segments),
                session.cache.valid_length,
                frames,
            )
        return batch

    # ---- realtime session: one event segment per chunk ----

    def _forward_session(self, batch, server_args):
        state = batch.session.get_or_create_state(NomasRealtimeState)
        conditions = batch.extra.pop("nomas_conditions")
        if _is_flush(batch):
            return batch
        condition = conditions[0]
        latent = batch.latents
        frames = latent.shape[2]
        with self.use_declared_component(
            component_name="transformer", module=self.transformer
        ) as model:
            if batch.block_idx == 0 or state.session is None:
                self._start_session(state, model, batch, server_args, condition)
            noise = self._noise_slice(state, server_args.pipeline_config, latent, batch)
            camera = _canonical_camera(batch, frames)
            if camera is not None:
                condition = segment_camera_condition(
                    condition, model.embed_camera(camera[None]), 0, frames
                )
            batch.latents = state.session.generate_segment(
                tuple(latent.shape[2:]),
                condition,
                initial_latents=noise,
                cache_segment=True,
                evict=True,
            )
        state.generated_latents += frames
        logger.info(
            "Nomas session %s chunk %d: %d latent frames, %d cached segments, %d/%d tokens",
            batch.realtime_session_id,
            batch.block_idx,
            frames,
            len(state.session.cache.segment_lengths),
            state.session.cache.valid_length,
            state.session.cache.capacity,
        )
        return batch

    def _start_session(self, state, model, batch, server_args, condition):
        config = server_args.pipeline_config
        latent = batch.latents
        state.dispose()
        shards = model.parallel.head_shards
        bytes_per_token = cache_bytes_per_token(config, shards)
        free_bytes = _free_bytes(latent.device)
        first_tokens = math.prod(latent.shape[2:]) + condition.text_mask.shape[1]
        # Raises when even one segment exceeds the configured or free budget.
        check_cache_budget(
            config, first_tokens, free_bytes=free_bytes, head_shards=shards
        )
        limit = cache_budget_bytes(config, free_bytes)
        max_capacity = max(first_tokens, int(limit // bytes_per_token))
        capacity = min(
            max_capacity, max(first_tokens, config.interactive_kv_cache_initial_tokens)
        )
        # Upstream draws the noise bank and every cache noise from one generator
        # seeded with the request seed: bank first, then per segment any bank
        # growth followed by that segment's cache noise.
        state.noise_generator = torch.Generator(device="cpu").manual_seed(batch.seed)
        state.session = self._new_session(
            model,
            batch,
            config,
            capacity,
            state.noise_generator,
            max_cache_segments=config.max_cache_segments,
            max_cache_capacity=max_capacity,
            cache_growth_factor=config.interactive_kv_cache_growth_factor,
        )
        logger.info(
            "Nomas session %s: cache %d tokens initial, %d max (%.2f GiB), max_cache_segments=%s",
            batch.realtime_session_id,
            capacity,
            max_capacity,
            max_capacity * bytes_per_token / 2**30,
            config.max_cache_segments,
        )

    @staticmethod
    def _new_session(model, batch, config, capacity, generator, **options):
        return NomasLatentSession(
            model,
            cache_capacity=capacity,
            seed=batch.seed,
            steps=batch.num_inference_steps,
            shift=config.flow_shift,
            cache_noise_sigma=config.cache_noise_sigma,
            generator=generator,
            sigmas=batch.scheduler.sigmas,
            **options,
        )

    @staticmethod
    def _noise_slice(state, config, latent, batch):
        """Slice the per-session noise bank by absolute latent frame.

        Like upstream's interactive noise timeline, the bank holds
        interactive_noise_max_seconds of latents up front and grows by
        interactive_noise_growth_seconds without changing its prefix, so a
        segment's noise depends only on where it starts.
        """
        _, channels, frames, height, width = latent.shape
        fps = batch.fps or 24

        def draw(latent_frames):
            return torch.randn(
                (1, channels, latent_frames, height, width),
                generator=state.noise_generator,
                dtype=latent.dtype,
            )

        if state.noise_bank is None:
            state.noise_bank = draw(
                _seconds_to_latents(config.interactive_noise_max_seconds, fps)
            )
        bank = state.noise_bank
        if bank.shape[3:] != (height, width):
            raise ValueError("video size cannot change within a Nomas realtime session")
        start = state.generated_latents
        end = start + frames
        if bank.shape[2] < end:
            growth = _seconds_to_latents(
                max(config.interactive_noise_growth_seconds, 1.0), fps
            )
            target = max(end, bank.shape[2] + growth)
            bank = torch.cat((bank, draw(target - bank.shape[2])), dim=2)
            state.noise_bank = bank
        return bank[:, :, start:end].to(latent.device)


def _seconds_to_latents(seconds, fps):
    return (max(1, round(seconds * fps)) - 1) // 4 + 1


def _free_bytes(device):
    if device.type != "cuda":
        return None
    free_bytes, _ = torch.cuda.mem_get_info(device)
    if model_parallel_is_initialized():
        group = get_replica_group()
        if group.world_size > 1:
            # All head shards must retain the same events. Decide the budget
            # before any rank can reject or allocate, using the least free rank.
            available = torch.tensor(free_bytes, dtype=torch.int64)
            dist.all_reduce(available, op=dist.ReduceOp.MIN, group=group.cpu_group)
            free_bytes = int(available.item())
    return free_bytes


class NomasDecodingStage(DecodingStage):
    """Offline: shared decode then crop. Realtime: one stable-prefix timeline.

    Realtime chunks push their latents onto the session's PixelUMM timeline
    (upstream stable-prefix streaming) and emit only frames no later window
    can change; the ``end`` flush emits the rest. Concatenated, the emitted
    frames equal one full decode of the session with each event's padded
    tail frames dropped.
    """

    def forward(self, batch, server_args):
        if batch.session is not None:
            return self._forward_session(batch, server_args)
        output = super().forward(batch, server_args)
        output.output = output.output[:, :, : batch.num_frames]
        return output

    def _decode_raw(self, latents, frame_offset, batch, server_args):
        """Merged, unnormalized pixels of a timeline slice starting at ``frame_offset``."""
        self.load_model()
        vae_dtype = resolve_decode_precision(
            server_args, self.component_name, quality=batch.sampling_params.quality
        )
        with self.use_declared_component(
            component_name=self.component_name, module=self.vae
        ) as vae:
            self.vae = vae
            with vae.timeline_offset(frame_offset):
                return self.decode_raw(latents, server_args, vae_dtype=vae_dtype)

    @torch.no_grad()
    def _forward_session(self, batch, server_args):
        state = batch.session.get_or_create_state(NomasRealtimeState)

        def decode_fn(latents, frame_offset):
            return self._decode_raw(latents, frame_offset, batch, server_args)

        if _is_flush(batch):
            raw = None if state.decoder is None else state.decoder.finish(decode_fn)
        else:
            if state.decoder is None:
                arch = server_args.pipeline_config.vae_config.arch_config
                if arch.num_steps != 1:
                    # Multi-step sampling blends whole-clip state between steps,
                    # so windows are not local; upstream refuses it too.
                    raise ValueError("Nomas realtime decoding requires num_steps=1")
                state.decoder = NomasStreamingDecoder(
                    arch.temporal_window_latents, arch.temporal_overlap_latents
                )
            raw = state.decoder.push(decode_fn, batch.latents, batch.num_frames)
        if raw is None:
            return self._empty_output(batch)
        output = server_args.pipeline_config.post_decoding(
            _to_unit_range(raw), server_args
        )
        return OutputBatch(output=output, metrics=batch.metrics, usage=batch.usage)

    @staticmethod
    def _empty_output(batch):
        device = batch.latents.device if batch.latents is not None else "cpu"
        output = torch.empty(
            (1, 3, 0, batch.height, batch.width), dtype=torch.float32, device=device
        )
        return OutputBatch(output=output, metrics=batch.metrics, usage=batch.usage)
