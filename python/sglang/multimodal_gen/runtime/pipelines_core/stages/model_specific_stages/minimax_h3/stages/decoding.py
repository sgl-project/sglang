# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import functools
from collections.abc import Mapping
from concurrent.futures import Future, ThreadPoolExecutor
from contextlib import contextmanager, nullcontext

import torch

from sglang.multimodal_gen.configs.sample.sampling_params import (
    quality_allows,
)
from sglang.multimodal_gen.runtime.disaggregation.roles import RoleType
from sglang.multimodal_gen.runtime.distributed import (
    get_decode_parallel_group_coordinator,
    get_decode_parallel_world_size,
    get_replica_group,
    model_parallel_is_initialized,
)
from sglang.multimodal_gen.runtime.managers.forward_context import set_forward_context
from sglang.multimodal_gen.runtime.managers.memory_managers.component_manager import (
    ComponentUse,
)
from sglang.multimodal_gen.runtime.managers.memory_managers.component_residency import (
    RESIDENT,
)
from sglang.multimodal_gen.runtime.models.vaes.fast_path_gate import use_vae_fast_path
from sglang.multimodal_gen.runtime.pipelines_core.schedule_batch import OutputBatch, Req
from sglang.multimodal_gen.runtime.pipelines_core.stages.base import (
    StageParallelismType,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.decoding import DecodingStage
from sglang.multimodal_gen.runtime.pipelines_core.stages.encode_while_decoding import (
    EncodeWhileDecoding,
    streamed_video_output_path,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.validators import (
    StageValidators as V,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.validators import (
    VerificationResult,
)
from sglang.multimodal_gen.runtime.platforms import current_platform
from sglang.multimodal_gen.runtime.server_args import ServerArgs
from sglang.multimodal_gen.runtime.utils.logging_utils import init_logger
from sglang.multimodal_gen.runtime.utils.precision import (
    autocast_context,
    autocast_enabled_for_device,
    resolve_decode_precision,
    resolve_precision,
)
from sglang.multimodal_gen.runtime.utils.torch_compile import (
    ActiveTargetCompiledCallable,
)

from ..constants import MINIMAX_H3_DWD_PIXELS_EXTRA_KEY

logger = init_logger(__name__)


def _required_tensor(value, path: str) -> torch.Tensor:
    if not isinstance(value, torch.Tensor):
        raise ValueError(f"{path} must be a torch.Tensor")
    return value


@contextmanager
def _deterministic_audio_decode_context():
    """Deterministic-algorithm scope for the fp32 audio-VAE decode.

    Without it, cuDNN picks conv algorithms from free-workspace state, so the
    same audio latent decodes to different bytes on a server process's first
    request than on every later one. Deterministic algorithms with TF32 off
    keep cuDNN speed (unlike the encode-side context, which disables cuDNN);
    if first-request divergence ever reappears, escalate to
    reference_encoding._AudioVAEDeterminismContext.
    """
    b = torch.backends
    saved = (
        b.cudnn.allow_tf32,
        b.cuda.matmul.allow_tf32,
        b.cudnn.deterministic,
        b.cudnn.benchmark,
    )
    b.cudnn.allow_tf32 = False
    b.cuda.matmul.allow_tf32 = False
    b.cudnn.deterministic = True
    b.cudnn.benchmark = False
    try:
        yield
    finally:
        (
            b.cudnn.allow_tf32,
            b.cuda.matmul.allow_tf32,
            b.cudnn.deterministic,
            b.cudnn.benchmark,
        ) = saved


@functools.lru_cache(maxsize=None)
def _cached_decode_mean_std(
    mean_values: tuple[float, ...],
    std_values: tuple[float, ...],
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Device/dtype-keyed mean/std tensors, built once per distinct combination.

    mean_values/std_values come from the loaded arch_config and are fixed
    for the process lifetime, so the same (values, device, dtype) always
    reconstructs an identical tensor; cache it instead of rebuilding it on
    every decode call.
    """
    mean = torch.as_tensor(mean_values, device=device, dtype=dtype)
    std = torch.as_tensor(std_values, device=device, dtype=dtype)
    return mean, std


def _reverse_normalize_latents(
    latents: torch.Tensor,
    *,
    mean_values,
    std_values,
    name: str,
) -> torch.Tensor:
    mean, std = _cached_decode_mean_std(
        tuple(mean_values), tuple(std_values), latents.device, latents.dtype
    )
    if mean.ndim != 1:
        raise ValueError(f"{name}.latents_mean must be 1-D, got {tuple(mean.shape)}")
    if std.ndim != 1:
        raise ValueError(f"{name}.latents_std must be 1-D, got {tuple(std.shape)}")
    if mean.shape != std.shape:
        raise ValueError(
            f"{name} latent normalization shape mismatch: "
            f"mean={tuple(mean.shape)} std={tuple(std.shape)}"
        )
    if latents.ndim < 2:
        raise ValueError(f"{name} latents must have a channel dimension")
    if int(latents.shape[1]) != int(mean.shape[0]):
        raise ValueError(
            f"{name} latent normalization channel mismatch: "
            f"latents.shape[1]={int(latents.shape[1])} mean_len={int(mean.shape[0])}"
        )
    view_shape = [1] * latents.ndim
    view_shape[1] = int(mean.shape[0])
    # Out of place on purpose. batch.latents / batch.audio_latents are
    # inference tensors allocated inside the denoising stage's InferenceMode,
    # while --vae-cpu-offload runs this stage under torch.inference_mode(False)
    # (PipelineExecutor._stage_needs_version_counters), where writing to one in
    # place raises. Keep the mul-then-add rounding order rather than addcmul so
    # the result does not shift with FMA contraction.
    return latents * std.view(*view_shape) + mean.view(*view_shape)


def _crop_to_target_canvas(batch: Req, frames: torch.Tensor) -> torch.Tensor:
    """Crop decoded frames [B,C,T,H,W] back to the target canvas.

    The visual VAE pads the latent grid to its tile multiples (padding lands
    at the bottom/right), so a non-tile-aligned geometry decodes larger than the
    requested canvas (e.g. 1344x768 for a 1280x704 target). Target dims come
    from the direct-mode denoise state (latent_h/w * 16); requests without
    that state keep the raw decode.
    """
    from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.minimax_h3.constants import (
        MINIMAX_H3_DENOISE_STATE_EXTRA_KEY,
    )

    state = batch.extra.get(MINIMAX_H3_DENOISE_STATE_EXTRA_KEY)
    if state is None:
        return frames
    target_h = int(state["latent_h"]) * 16
    target_w = int(state["latent_w"]) * 16
    h, w = int(frames.shape[-2]), int(frames.shape[-1])
    if h < target_h or w < target_w:
        raise ValueError(
            f"decoded frames {h}x{w} smaller than target canvas {target_h}x{target_w}"
        )
    if h == target_h and w == target_w:
        return frames
    return frames[..., :target_h, :target_w]


def _canonical_visual_video_frames(
    frames: torch.Tensor, *, batch_size: int
) -> torch.Tensor:
    if frames.ndim == 4:
        if int(frames.shape[0]) % batch_size != 0:
            raise ValueError(
                f"Decoded visual video shape {tuple(frames.shape)} is incompatible "
                f"with batch_size={batch_size}"
            )
        frames = frames.reshape(
            batch_size, int(frames.shape[0]) // batch_size, *frames.shape[1:]
        )
        frames = frames.transpose(1, 2)
    elif frames.ndim == 5:
        if int(frames.shape[0]) != batch_size:
            raise ValueError(
                f"Decoded visual video batch mismatch: frames.shape[0]={int(frames.shape[0])} "
                f"batch_size={batch_size}"
            )
    else:
        raise ValueError(
            f"Decoded visual video shape {tuple(frames.shape)} is not supported"
        )
    return frames


def _canonical_output_audio_waveform(
    audio_waveform: torch.Tensor, *, batch_size: int
) -> torch.Tensor:
    """Project audio-VAE-native ``[C, 1, L]`` audio to output ``[1, C, L]``.

    The audio VAE treats stereo channels as its decoder batch and returns
    ``[2, 1, samples]`` for MiniMax H3's one generated sample.  The generic output
    path instead selects generated samples along dimension zero.  Keep the audio VAE
    tensor unchanged for decoder artifacts, then make the singleton generated-
    sample dimension explicit only at the ``OutputBatch`` boundary.
    """
    if audio_waveform.ndim != 3:
        raise ValueError(
            "Decoded audio VAE waveform must be [C, 1, L], got "
            f"{tuple(audio_waveform.shape)}"
        )
    if batch_size != 1:
        raise ValueError(
            "MiniMax H3 audio VAE output only supports one generated sample, "
            f"got visual batch_size={batch_size}"
        )
    if int(audio_waveform.shape[1]) != 1:
        raise ValueError(
            "Decoded audio VAE waveform must have shape [C, 1, L], got "
            f"{tuple(audio_waveform.shape)}"
        )
    return audio_waveform.permute(1, 0, 2).contiguous()


def _trim_audio_to_video_duration(
    audio_waveform: torch.Tensor, batch: Req, sample_rate: int
) -> torch.Tensor:
    """Cut ``[1, C, L]`` audio to the video's duration, as the reference does.

    The audio latents cover whole hops, so the decoded waveform runs past the
    last video frame (e.g. 165600 samples for 124 frames at 24 fps and 32 kHz,
    whose duration is 165333 samples). Same rule as the Diffusers Pika H3 Flex
    audio decode step: ``round(num_frames / fps * sample_rate)``. The base
    MiniMax-H3 reference keeps the full waveform.
    """
    if not batch.num_frames or not batch.fps:
        return audio_waveform
    num_samples = round(int(batch.num_frames) / float(batch.fps) * int(sample_rate))
    if 0 < num_samples < int(audio_waveform.shape[-1]):
        audio_waveform = audio_waveform[..., :num_samples].contiguous()
    return audio_waveform


_MINIMAX_H3_DECODER_TASKS = frozenset({"t2va", "fl2va", "ref2va"})
# Video tiles of the audio decoder's rank that the other ranks decode while it
# decodes the audio beside them (8 x B300, t2va 16:9 768p at 5 / 10 / 15 s:
# 12 beat 6 and 9; the audio decode alone takes about ten tiles' time)
_OVERLAP_AUDIO_TILES = 12
_MINIMAX_H3_CANONICAL_REQUEST_EXTRA_KEY = "minimax_h3_canonical_request"
_MINIMAX_H3_RESOLVED_PLAN_EXTRA_KEY = "minimax_h3_resolved_plan"


def _minimax_h3_decoder_task(batch: Req) -> str | None:
    """Return the validated request task used for output-decoder routing.

    Debug requests have no canonical task and retain the
    generic decoder.
    """

    extra = getattr(batch, "extra", None)
    if not isinstance(extra, Mapping):
        return None
    canonical = extra.get(_MINIMAX_H3_CANONICAL_REQUEST_EXTRA_KEY)
    if canonical is not None and not isinstance(canonical, Mapping):
        raise ValueError("minimax_h3_canonical_request must be a mapping")
    canonical_task = canonical.get("task") if isinstance(canonical, Mapping) else None
    resolved = extra.get(_MINIMAX_H3_RESOLVED_PLAN_EXTRA_KEY)
    resolved_task = getattr(resolved, "task", None) if resolved is not None else None
    if canonical_task is not None and resolved_task is not None:
        if str(canonical_task) != str(resolved_task):
            raise ValueError(
                "MiniMax H3 decoder task mismatch between canonical request and "
                "resolved plan"
            )
    task_value = resolved_task if resolved_task is not None else canonical_task
    if task_value is None:
        return None
    if not isinstance(task_value, str) or task_value not in _MINIMAX_H3_DECODER_TASKS:
        raise ValueError(f"unsupported MiniMax H3 decoder task {task_value!r}")
    return task_value


@contextmanager
def _lighten_tile_groups(video_vae, rank: int | None, budget: int, keep: int):
    """Hand ``budget`` of ``rank``'s tiles to the other ranks, leaving it
    ``keep`` per tiled decode (see ``tile_owners``)."""
    video_vae.light_tile_rank = rank
    video_vae.light_tile_budget, video_vae.light_tile_keep = budget, keep
    try:
        yield
    finally:
        video_vae.light_tile_rank = None
        video_vae.light_tile_budget, video_vae.light_tile_keep = 0, 1


class MiniMaxH3DecodingStage(DecodingStage):
    def __init__(self, video_vae, audio_vae) -> None:
        """``audio_vae=None`` selects the audio-free image release output."""
        super().__init__(vae=video_vae, component_name="video_vae")
        self.video_vae = video_vae
        self.audio_vae = audio_vae
        self._compiled_audio_vae_decode = ActiveTargetCompiledCallable()
        # thread and stream of an audio decode that overlaps the video decode
        self._audio_thread: ThreadPoolExecutor | None = None
        self._audio_stream: torch.cuda.Stream | None = None

    @property
    def role_affinity(self) -> RoleType:
        return RoleType.DECODER

    @property
    def parallelism_type(self) -> StageParallelismType:
        # Every decode-group rank owns a subset of visual VAE tiles. The GPU
        # worker only materializes/saves the final OutputBatch on world rank 0.
        return StageParallelismType.REPLICATED

    def component_uses(
        self, server_args: ServerArgs, stage_name: str | None = None
    ) -> list[ComponentUse]:
        stage_name = self._component_stage_name(stage_name)
        video_vae_dtype = resolve_precision(
            server_args, "video_vae", precision_attr="vae_precision"
        )
        if self.audio_vae is None:
            return [ComponentUse(stage_name, "video_vae", target_dtype=video_vae_dtype)]
        audio_vae_dtype = resolve_precision(
            server_args, "audio_vae", precision_attr="audio_vae_precision"
        )
        # forward decodes the audio first so the MP4 encoder can start with it
        return [
            ComponentUse(stage_name, "audio_vae", target_dtype=audio_vae_dtype),
            ComponentUse(stage_name, "video_vae", target_dtype=video_vae_dtype),
        ]

    def verify_input(self, batch: Req, server_args: ServerArgs) -> VerificationResult:
        result = VerificationResult()
        result.add_check("latents", batch.latents, [V.is_tensor, V.with_dims(5)])
        if self.audio_vae is not None:
            result.add_check(
                "audio_latents",
                batch.audio_latents,
                [V.is_tensor, V.with_dims(3)],
            )
        return result

    def verify_output(
        self, batch: OutputBatch, server_args: ServerArgs
    ) -> VerificationResult:
        result = VerificationResult()
        if self.audio_vae is None:
            result.add_check("output", batch.output, [V.is_tensor, V.with_dims(4)])
            return result
        if not batch.output_file_paths:
            result.add_check("output", batch.output, [V.is_tensor, V.with_dims(5)])
        result.add_check("audio", batch.audio, [V.is_tensor, V.with_dims(3)])
        result.add_check("audio_sample_rate", batch.audio_sample_rate, V.positive_int)
        return result

    def _decode_audio(
        self,
        audio_latent: torch.Tensor,
        server_args: ServerArgs,
    ) -> dict:
        with self.use_declared_component(
            component_name="audio_vae",
            module=self.audio_vae,
        ) as audio_vae:
            assert audio_vae is not None
            self.audio_vae = audio_vae
            if audio_vae.training:
                audio_vae.eval()
            return self._run_audio_vae(audio_vae, audio_latent, server_args)

    def _run_audio_vae(
        self, audio_vae, audio_latent: torch.Tensor, server_args: ServerArgs
    ) -> dict:
        audio_arch_config = server_args.pipeline_config.audio_vae_config.arch_config
        audio_decode_latent = _reverse_normalize_latents(
            audio_latent,
            mean_values=audio_arch_config.latents_mean,
            std_values=audio_arch_config.latents_std,
            name="audio_vae",
        )
        audio_vae_dtype = resolve_precision(
            server_args, "audio_vae", precision_attr="audio_vae_precision"
        )
        audio_autocast_enabled = autocast_enabled_for_device(
            audio_latent, audio_vae_dtype, server_args.disable_autocast
        )
        autocast_context = (
            torch.autocast(
                device_type="cuda",
                dtype=audio_vae_dtype,
                enabled=audio_autocast_enabled,
            )
            if audio_latent.is_cuda
            else nullcontext()
        )
        with _deterministic_audio_decode_context(), autocast_context:
            audio_decode = self._get_vae_decode_fn(
                audio_vae,
                server_args,
                decode_fn=audio_vae.decode,
                compiled_callable=self._compiled_audio_vae_decode,
            )
            waveform = _required_tensor(
                audio_decode(audio_decode_latent), "audio_vae.decode"
            )
        return {
            "waveform": waveform,
            "sample_rate": int(audio_vae.sample_rate),
        }

    def _decode_owner_audio(
        self, audio_latent: torch.Tensor, server_args: ServerArgs
    ) -> tuple[dict | None, Exception | None]:
        """Decode the audio on replica rank 0 (the weights are replicated);
        other ranks return (None, None) and receive it from ``_share_audio``."""
        if model_parallel_is_initialized() and get_replica_group().rank_in_group != 0:
            return None, None
        try:
            return self._decode_audio(audio_latent, server_args), None
        except Exception as exc:
            return None, exc

    def _share_audio(
        self, audio_payload: dict | None, owner_exception: Exception | None
    ) -> tuple[torch.Tensor, int]:
        """Broadcast replica rank 0's audio (or its error) within the replica,
        excluding independent DP replicas."""
        replica_group = get_replica_group() if model_parallel_is_initialized() else None
        owner_error = None
        if owner_exception is not None:
            owner_error = f"{type(owner_exception).__name__}: {owner_exception}"
        if replica_group is not None:
            owner_error = replica_group.broadcast_object(owner_error, src=0)
        if owner_error is not None:
            if owner_exception is not None:
                raise owner_exception
            raise RuntimeError(
                f"MiniMax H3 audio decode failed on rank 0: {owner_error}"
            )
        if replica_group is not None:
            audio_payload = replica_group.broadcast_tensor_dict(audio_payload, src=0)
        if not isinstance(audio_payload, dict):
            raise RuntimeError("MiniMax H3 audio decode produced no output payload")
        audio_waveform = _required_tensor(
            audio_payload.get("waveform"), "audio_vae.decode"
        )
        return audio_waveform, int(audio_payload["sample_rate"])

    @staticmethod
    def _overlaps_audio_decode(server_args: ServerArgs) -> bool:
        """Whether the audio decodes on a side stream alongside the video.

        CUDA only, with other decode ranks to take over the audio rank's
        tiles, and only for a resident audio VAE: the decode runs on another
        thread, outside the residency manager's sequential uses.
        """
        return (
            current_platform.is_cuda()
            and model_parallel_is_initialized()
            and get_decode_parallel_world_size() > 1
            and server_args.residency_mode("audio_vae") == RESIDENT
        )

    def _start_owner_audio(
        self, audio_latent: torch.Tensor, server_args: ServerArgs, finish
    ) -> Future | None:
        """Start replica rank 0's audio decode on a side stream and thread;
        other ranks return None.

        The future gives the decoded payload, ``finish`` of its waveform (the
        output audio) and that on the host. The stream has the higher priority:
        the MP4 encoder waits for the audio, and the other ranks take over most
        of this rank's first video tiles meanwhile (``_audio_tile_plan``).
        """
        if model_parallel_is_initialized() and get_replica_group().rank_in_group != 0:
            return None
        with self.use_declared_component(
            component_name="audio_vae",
            module=self.audio_vae,
        ) as audio_vae:
            assert audio_vae is not None
            self.audio_vae = audio_vae
            if audio_vae.training:
                audio_vae.eval()
        if self._audio_thread is None:
            self._audio_thread = ThreadPoolExecutor(
                1, thread_name_prefix="sglang-audio-decode"
            )
            self._audio_stream = torch.cuda.Stream(priority=-1)
        stream = self._audio_stream
        stream.wait_stream(torch.cuda.current_stream())
        audio_latent.record_stream(stream)
        device = torch.cuda.current_device()
        inference = torch.is_inference_mode_enabled()

        def run():
            with (
                torch.cuda.device(device),
                torch.cuda.stream(stream),
                torch.inference_mode(inference),
                torch.no_grad(),
            ):
                payload = self._run_audio_vae(audio_vae, audio_latent, server_args)
                waveform = finish(payload["waveform"], payload["sample_rate"])
                return payload, waveform, waveform[0].cpu()

        return self._audio_thread.submit(run)

    def _join_owner_audio(
        self, job: Future | None
    ) -> tuple[dict | None, torch.Tensor | None, Exception | None]:
        """The started audio decode's payload and output audio, ready on the
        current stream, or its error; (None, None, None) without one."""
        if job is None:
            return None, None, None
        try:
            payload, waveform, _ = job.result()
        except Exception as exc:
            return None, None, exc
        current = torch.cuda.current_stream()
        current.wait_stream(self._audio_stream)
        payload["waveform"].record_stream(current)
        waveform.record_stream(current)
        return payload, waveform, None

    @staticmethod
    def _audio_tile_plan(overlap: bool) -> tuple[int | None, int, int]:
        """Decode-group rank of the replica rank that decodes the audio, how
        many of its video tiles the other ranks take over meanwhile, and how
        many it keeps per tiled decode until then.

        ROCm decodes the audio before the video and lightens the first tile
        group, keeping one tile (measured on MI350X: the audio decode costs
        about six tiles). CUDA decodes it alongside the video, where sharing
        the GPU with the rank's own tiles gains nothing over running them in
        turn, so the rank keeps none until the tiles that fit beside the audio
        are spent (``_OVERLAP_AUDIO_TILES``).
        """
        if not model_parallel_is_initialized() or not (
            overlap or current_platform.is_rocm()
        ):
            return None, 0, 1
        owner = get_replica_group().ranks[0]
        decode_ranks = get_decode_parallel_group_coordinator().ranks
        if owner not in decode_ranks:
            return None, 0, 1
        if not overlap:
            return decode_ranks.index(owner), 1, 1
        return decode_ranks.index(owner), _OVERLAP_AUDIO_TILES, 0

    @staticmethod
    def _finish_visual_frames(
        batch: Req,
        video_vae,
        frames: torch.Tensor,
        *,
        batch_size: int,
        runtime_owned: bool = False,
    ) -> torch.Tensor:
        frames = _required_tensor(
            video_vae.processor.revert_tensor(frames, runtime_owned=runtime_owned),
            "video_vae.processor.revert_tensor",
        )
        frames = _canonical_visual_video_frames(frames, batch_size=batch_size)
        frames = _crop_to_target_canvas(batch, frames)
        if frames.dtype != torch.float32 or not frames.is_contiguous():
            canonical_frames = torch.empty_like(
                frames,
                dtype=torch.float32,
                memory_format=torch.contiguous_format,
            )
            canonical_frames.copy_(frames)
            frames = canonical_frames
        return frames

    def _decode_image(self, batch: Req, server_args: ServerArgs) -> OutputBatch:
        """Image release: one FP32 T=1 latent through ``decode_image_raw``
        (BF16 decoder weights under BF16 autocast), then the resize from the
        generation canvas to the requested size (minimax_h3/image_canvas.py)."""
        from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.minimax_h3.image_canvas import (
            minimax_h3_resize_image_output,
        )
        from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.minimax_h3.resolved_plan import (
            minimax_h3_plan_from_batch,
        )

        plan = minimax_h3_plan_from_batch(batch)
        if plan is None:
            raise ValueError("MiniMax H3 image decode requires a resolved request plan")
        with self.use_declared_component(
            component_name="video_vae", module=self.video_vae
        ) as video_vae:
            self.video_vae = video_vae
            arch = server_args.pipeline_config.vae_config.arch_config
            raw = _reverse_normalize_latents(
                batch.latents.to(self.device),
                mean_values=arch.latents_mean,
                std_values=arch.latents_std,
                name="video_vae",
            )
            # The ViT decoder's attention layer reads the forward context on
            # CUDA, as in the video decode path.
            with (
                set_forward_context(current_timestep=0, attn_metadata=None),
                use_vae_fast_path(video_vae, quality_allows(batch.quality, "lossless")),
            ):
                pixels = video_vae.decode_image_raw(raw)
        # decode_image_raw already applies the ImageNet inverse normalization.
        pixels = pixels.detach().float()
        shape = plan.shape
        # The resize quantizes (and, when verified exact, resizes) on the device
        # and returns a host tensor, so only 8-bit pixels cross to the host.
        if shape["output_width"] is None:
            pixels = pixels.cpu()
        else:
            width, height = int(shape["output_width"]), int(shape["output_height"])
            logger.info(
                "MiniMax H3 image: generated %dx%d (min_generation_area), resized "
                "to the requested %dx%d with LANCZOS",
                int(shape["width"]),
                int(shape["height"]),
                width,
                height,
            )
            pixels = minimax_h3_resize_image_output(pixels, width, height)
        return OutputBatch(output=pixels, metrics=batch.metrics)

    @torch.no_grad()
    def forward(self, batch: Req, server_args: ServerArgs) -> OutputBatch:
        if self.audio_vae is None:
            return self._decode_image(batch, server_args)
        _minimax_h3_decoder_task(batch)
        visual_latent = _required_tensor(batch.latents, "batch.latents")
        audio_latent = _required_tensor(batch.audio_latents, "batch.audio_latents")
        if visual_latent.ndim != 5:
            raise ValueError("batch.latents must be [B, C, T, H, W]")
        if audio_latent.ndim != 3:
            raise ValueError(
                "batch.audio_latents must be [audio_channel, latent_dim, T]"
            )

        if self.video_vae is None:
            raise RuntimeError("MiniMax H3 tasks require the video_vae output decoder")
        batch_size = int(visual_latent.shape[0])

        def finish_audio(waveform: torch.Tensor, sample_rate: int) -> torch.Tensor:
            waveform = _canonical_output_audio_waveform(waveform, batch_size=batch_size)
            if server_args.pipeline_config.trim_audio_to_video:
                waveform = _trim_audio_to_video_duration(waveform, batch, sample_rate)
            return waveform

        # Replica rank 0 decodes the audio, alongside its video tiles on CUDA
        # or before them while the other ranks start theirs; everyone else
        # receives it after the video decode.
        overlap_audio = self._overlaps_audio_decode(server_args)
        owner_audio, owner_exception, audio_job = None, None, None
        if overlap_audio:
            audio_job = self._start_owner_audio(
                audio_latent,
                server_args,
                finish_audio,
            )
        else:
            owner_audio, owner_exception = self._decode_owner_audio(
                audio_latent, server_args
            )
        output_audio_waveform = None
        streamed_output_path = None
        encoder_audio = None
        if owner_audio is not None:
            audio_sample_rate = int(owner_audio["sample_rate"])
            output_audio_waveform = finish_audio(
                _required_tensor(owner_audio.get("waveform"), "audio_vae.decode"),
                audio_sample_rate,
            )
            encoder_audio = output_audio_waveform[0]
        elif audio_job is not None:
            audio_sample_rate = int(self.audio_vae.sample_rate)
            encoder_audio = lambda: audio_job.result()[2]  # noqa: E731
        if encoder_audio is not None:
            streamed_output_path = streamed_video_output_path(
                batch, server_args, batch_size
            )

        visual_frames = None
        output_file_paths = None
        with self.use_declared_component(
            component_name="video_vae",
            module=self.video_vae,
        ) as selected_video_vae:
            if selected_video_vae is None:
                raise RuntimeError("video_vae became unavailable during decode")
            self.video_vae = selected_video_vae
            if selected_video_vae.training:
                selected_video_vae.eval()
            visual_arch_config = server_args.pipeline_config.vae_config.arch_config
            visual_decode_latent = _reverse_normalize_latents(
                visual_latent,
                mean_values=visual_arch_config.latents_mean,
                std_values=visual_arch_config.latents_std,
                name="video_vae",
            )
            video_vae_dtype = resolve_decode_precision(server_args, "video_vae")
            visual_autocast_enabled = autocast_enabled_for_device(
                visual_latent, video_vae_dtype, server_args.disable_autocast
            )
            if visual_autocast_enabled:
                selected_video_vae.prepare_decoder_autocast_weights(video_vae_dtype)
            with autocast_context(
                video_vae_dtype,
                server_args.disable_autocast,
                enabled=visual_autocast_enabled,
            ):
                video_decode = self._get_vae_decode_fn(
                    selected_video_vae,
                    server_args,
                    decode_fn=selected_video_vae.decode_base,
                )
                finish_frames = functools.partial(
                    self._finish_visual_frames,
                    batch,
                    selected_video_vae,
                    batch_size=batch_size,
                )
                stream = None
                decode_kwargs = {}
                if streamed_output_path is not None:
                    stream = EncodeWhileDecoding(
                        streamed_output_path,
                        batch,
                        finish_frames,
                        encoder_audio,
                        audio_sample_rate,
                    )
                    decode_kwargs["on_frames"] = stream
                try:
                    with (
                        use_vae_fast_path(
                            selected_video_vae,
                            quality_allows(batch.sampling_params.quality, "lossless"),
                        ),
                        _lighten_tile_groups(
                            selected_video_vae, *self._audio_tile_plan(overlap_audio)
                        ),
                        set_forward_context(current_timestep=0, attn_metadata=None),
                    ):
                        dwd_pixels = batch.extra.pop(
                            MINIMAX_H3_DWD_PIXELS_EXTRA_KEY, None
                        )
                        if dwd_pixels is not None:
                            # DWD release: the DiT's pixel head already decoded
                            # the frames; the VAE stays idle
                            decoded = dwd_pixels.float()
                            if stream is not None:
                                stream(decoded)
                        else:
                            decoded = video_decode(
                                visual_decode_latent, **decode_kwargs
                            )
                    # before closing the encoder, whose x264 and mux tail it overlaps
                    if overlap_audio:
                        owner_audio, output_audio_waveform, owner_exception = (
                            self._join_owner_audio(audio_job)
                        )
                    audio_waveform, audio_sample_rate = self._share_audio(
                        owner_audio, owner_exception
                    )
                    if output_audio_waveform is None:
                        output_audio_waveform = finish_audio(
                            audio_waveform, audio_sample_rate
                        )
                    if stream is not None:
                        output_file_paths = stream.finish()
                except BaseException:
                    if stream is not None:
                        stream.abort()
                    raise
                if output_file_paths is None:
                    # callbacks borrow slices of decoded; only the final output owns it
                    visual_frames = finish_frames(decoded, runtime_owned=True)
                del decoded

        if visual_frames is not None:
            visual_frames = server_args.pipeline_config.post_decoding(
                visual_frames, server_args
            )
        return OutputBatch(
            output=visual_frames,
            output_file_paths=output_file_paths,
            audio=output_audio_waveform,
            audio_sample_rate=audio_sample_rate,
            trajectory_timesteps=batch.trajectory_timesteps,
            trajectory_latents=batch.trajectory_latents,
            rollout_trajectory_data=batch.rollout_trajectory_data,
            trajectory_decoded=None,
            metrics=batch.metrics,
            noise_pred=None,
        )


__all__ = [
    "MiniMaxH3DecodingStage",
]
