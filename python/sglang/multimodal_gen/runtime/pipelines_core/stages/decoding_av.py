import time

import torch

from sglang.multimodal_gen import envs
from sglang.multimodal_gen.runtime.distributed import get_local_torch_device
from sglang.multimodal_gen.runtime.managers.memory_managers.component_manager import (
    ComponentUse,
)
from sglang.multimodal_gen.runtime.pipelines_core.schedule_batch import OutputBatch, Req
from sglang.multimodal_gen.runtime.pipelines_core.stages.decoding import DecodingStage
from sglang.multimodal_gen.runtime.platforms import current_platform
from sglang.multimodal_gen.runtime.server_args import ServerArgs
from sglang.multimodal_gen.runtime.utils.logging_utils import init_logger
from sglang.multimodal_gen.utils import PRECISION_TO_TYPE

logger = init_logger(__name__)

# Fallback LTX-2 compression ratios used only if the VAE does not expose them.
_DEFAULT_SPATIAL_COMPRESSION = 32
_DEFAULT_TEMPORAL_COMPRESSION = 8
# Safety factors for the untiled-decode free-memory estimate. The decoder's
# transient activation workspace dwarfs the final frame buffer, so the
# decoded-frame byte estimate is scaled up before comparing against free memory.
_UNTILED_DECODE_WORKSPACE_FACTOR = 8  # peak transient activations vs. output frames
_UNTILED_DECODE_SAFETY_MARGIN = 1.3  # extra headroom on top of the estimate
_UNTILED_DECODE_MIN_FREE_FRACTION = 0.3  # require >30% of total memory still free


class LTX2AVDecodingStage(DecodingStage):
    """
    LTX-2 specific decoding stage that handles both video and audio decoding.
    """

    def __init__(self, vae, audio_vae, vocoder, pipeline=None):
        super().__init__(vae, pipeline)
        self.audio_vae = audio_vae
        self.vocoder = vocoder
        # Add video processor for postprocessing
        from diffusers.video_processor import VideoProcessor

        self.video_processor = VideoProcessor(vae_scale_factor=32)
        self._vae_decode_untiled_failed = False
        self._vae_decode_untiled_logged = False

    def component_uses(
        self, server_args: ServerArgs, stage_name: str | None = None
    ) -> list[ComponentUse]:
        stage_name = self._component_stage_name(stage_name)
        return [
            ComponentUse(stage_name, "vae", target_dtype=torch.bfloat16),
            ComponentUse(stage_name, "audio_vae"),
            ComponentUse(stage_name, "vocoder"),
        ]

    @staticmethod
    def _ltx2_should_externally_denorm_video_latents(server_args: ServerArgs) -> bool:
        arch_config = server_args.pipeline_config.vae_config.arch_config
        return str(getattr(arch_config, "video_decoder_variant", "ltx_2")) != "ltx_2_3"

    def _vae_decode_untiled_mode(self) -> str:
        """Resolve eager untiled VAE-decode policy: "off" | "auto" | "force"."""
        mode = str(envs.SGLANG_DIFFUSION_LTX2_UNTILED_VAE_DECODE).strip().lower()
        if mode in ("0", "off", "false", "no", "disable", "disabled", ""):
            return "off"
        if mode in ("1", "on", "true", "yes", "force", "always"):
            return "force"
        return "auto"

    def _untiled_decode_fits(self, latents: torch.Tensor) -> bool:
        try:
            free, total = torch.cuda.mem_get_info()
            spatial = int(
                getattr(
                    self.vae,
                    "spatial_compression_ratio",
                    _DEFAULT_SPATIAL_COMPRESSION,
                )
            )
            temporal = int(
                getattr(
                    self.vae,
                    "temporal_compression_ratio",
                    _DEFAULT_TEMPORAL_COMPRESSION,
                )
            )
            channels = int(latents.shape[1]) if latents.dim() >= 2 else 1
            # latent positions -> decoded RGB pixels.
            decoded_numel = latents.numel() / max(channels, 1) * 3
            decoded_numel *= spatial * spatial * temporal
            # bf16 output frames scaled by the transient-activation workspace.
            peak_bytes = decoded_numel * 2 * _UNTILED_DECODE_WORKSPACE_FACTOR
            return (
                free > peak_bytes * _UNTILED_DECODE_SAFETY_MARGIN
                and (free / max(total, 1)) > _UNTILED_DECODE_MIN_FREE_FRACTION
            )
        except Exception:
            return False

    def _eager_untiled_decode(self, latents: torch.Tensor):
        try:
            if hasattr(self.vae, "disable_tiling"):
                self.vae.disable_tiling()
        except Exception:
            pass
        if not self._vae_decode_untiled_logged:
            logger.info("Using untiled eager LTX-2 VAE decode.")
            self._vae_decode_untiled_logged = True
        return self.vae.decode(latents)

    def _decode_video_latents(self, latents: torch.Tensor, server_args: ServerArgs):
        untiled_mode = self._vae_decode_untiled_mode()
        use_untiled_eager = (
            untiled_mode != "off"
            and not self._vae_decode_untiled_failed
            and (untiled_mode == "force" or self._untiled_decode_fits(latents))
        )
        if use_untiled_eager:
            try:
                return self._eager_untiled_decode(latents)
            except torch.cuda.OutOfMemoryError:
                logger.warning(
                    "Untiled eager VAE decode hit OOM; falling back to tiled "
                    "eager decode for the rest of this run."
                )
            except Exception:  # noqa: BLE001
                # Log the full traceback so a genuine decode bug (shape/dtype/
                # NaN) is not silently masked by the tiled-decode fallback.
                logger.exception(
                    "Untiled eager VAE decode failed; falling back to tiled "
                    "eager decode for the rest of this run."
                )
            self._vae_decode_untiled_failed = True
            try:
                torch.cuda.empty_cache()
            except Exception:
                pass

        try:
            if server_args.pipeline_config.vae_tiling:
                self.vae.enable_tiling()
        except Exception:
            pass
        return self.vae.decode(latents)

    @staticmethod
    def _should_profile_decode_substages(batch: Req) -> bool:
        return bool(batch.perf_dump_path or envs.SGLANG_DIFFUSION_STAGE_LOGGING)

    @staticmethod
    def _sync_for_decode_substage_profile() -> None:
        if torch.cuda.is_available():
            torch.cuda.synchronize()

    def _record_decode_substage(
        self,
        batch: Req,
        stage_name: str,
        start_time: float,
        *,
        sync: bool,
    ) -> float:
        if sync:
            self._sync_for_decode_substage_profile()
        now = time.perf_counter()
        if batch.metrics is not None:
            batch.metrics.record_stage(stage_name, now - start_time)
        return now

    @staticmethod
    def _postprocess_video_to_uint8_np(video: torch.Tensor):
        """Convert decoded [B, C, T, H, W] video to final uint8 [B, T, H, W, C] frames.

        This replaces ``VideoProcessor.postprocess_video(output_type="np")`` (which
        returned float32 [0, 1]) and does the denormalize + uint8 cast on-GPU so only
        uint8 (not float32) is copied to the host. It is numerically identical to the
        previous pipeline end-to-end: ``save_outputs`` already cast the float output
        to uint8 via ``(np.clip(x, 0, 1) * 255).astype(np.uint8)`` (truncation), which
        matches ``.to(torch.uint8)`` here. NOTE: the stage output is now uint8 [0, 255]
        rather than float [0, 1]; ``save_outputs`` handles both, but any consumer that
        reads ``OutputBatch.output`` directly as float must account for this.
        """
        if not isinstance(video, torch.Tensor) or video.dim() != 5:
            raise TypeError(
                "Expected decoded video tensor with shape [B, C, T, H, W], "
                f"got {type(video)}"
            )
        video = ((video / 2 + 0.5).clamp(0, 1) * 255).to(torch.uint8)
        return video.permute(0, 2, 3, 4, 1).contiguous().cpu().numpy()

    def forward(self, batch: Req, server_args: ServerArgs) -> OutputBatch:
        self.load_model()

        vae_dtype = PRECISION_TO_TYPE[server_args.pipeline_config.vae_precision]
        vae_autocast_enabled = (
            vae_dtype != torch.float32
        ) and not server_args.disable_autocast

        original_dtype = vae_dtype
        profile_decode_substages = self._should_profile_decode_substages(batch)
        substage_start = time.perf_counter()
        if profile_decode_substages:
            self._sync_for_decode_substage_profile()
            substage_start = time.perf_counter()
        with self.use_declared_component(component_name="vae", module=self.vae) as vae:
            assert vae is not None
            self.vae = vae
            self.vae.eval()
            latents = batch.latents.to(get_local_torch_device(), dtype=torch.bfloat16)
            if self._ltx2_should_externally_denorm_video_latents(server_args):
                std = self.vae.latents_std.view(1, -1, 1, 1, 1).to(latents)
                mean = self.vae.latents_mean.view(1, -1, 1, 1, 1).to(latents)
                latents = latents * std + mean
            latents = server_args.pipeline_config.preprocess_decoding(
                latents, server_args, vae=self.vae
            )

            with torch.autocast(
                device_type=current_platform.device_type,
                dtype=vae_dtype,
                enabled=vae_autocast_enabled,
            ):
                decode_output = self._decode_video_latents(latents, server_args)
                if isinstance(decode_output, tuple):
                    video = decode_output[0]
                elif hasattr(decode_output, "sample"):
                    video = decode_output.sample
                else:
                    video = decode_output

            if profile_decode_substages:
                substage_start = self._record_decode_substage(
                    batch,
                    "LTX2AVDecodingStage.video_decode",
                    substage_start,
                    sync=True,
                )
            self.vae.to(original_dtype)
            if profile_decode_substages:
                substage_start = self._record_decode_substage(
                    batch,
                    "LTX2AVDecodingStage.vae_dtype_restore",
                    substage_start,
                    sync=False,
                )
        video = self._postprocess_video_to_uint8_np(video)
        if profile_decode_substages:
            substage_start = self._record_decode_substage(
                batch,
                "LTX2AVDecodingStage.video_postprocess",
                substage_start,
                sync=False,
            )

        output_batch = OutputBatch(
            output=video,
            trajectory_timesteps=batch.trajectory_timesteps,
            trajectory_latents=batch.trajectory_latents,
            trajectory_decoded=None,
            metrics=batch.metrics,
        )

        # 2. Decode Audio
        try:
            audio_latents = batch.audio_latents
        except AttributeError:
            audio_latents = None
        if audio_latents is not None:
            # Ensure device/dtype
            device = get_local_torch_device()
            with self.use_declared_component(
                component_name="audio_vae",
                module=self.audio_vae,
            ) as audio_vae:
                assert audio_vae is not None
                self.audio_vae = audio_vae
                self.audio_vae.eval()
                try:
                    dtype = self.audio_vae.dtype
                except AttributeError:
                    dtype = None
                if dtype is None:
                    try:
                        dtype = next(self.audio_vae.parameters()).dtype
                    except StopIteration:
                        dtype = torch.float32
                audio_latents = audio_latents.to(device, dtype=dtype)
                try:
                    latents_std = self.audio_vae.latents_std
                except AttributeError:
                    latents_std = None
                if isinstance(latents_std, torch.Tensor) and torch.all(
                    latents_std == 0
                ):
                    logger.warning(
                        "audio_vae.latents_std is all zeros; audio denorm may be incorrect."
                    )
                try:
                    latents_mean = self.audio_vae.latents_mean
                except AttributeError:
                    latents_mean = None
                if isinstance(latents_mean, torch.Tensor) and isinstance(
                    latents_std, torch.Tensor
                ):
                    latents_mean = latents_mean.to(device=device, dtype=dtype)
                    latents_std = latents_std.to(device=device, dtype=dtype)
                    if audio_latents.ndim == 4:
                        latents_mean = latents_mean.view(
                            1, audio_latents.shape[1], 1, audio_latents.shape[3]
                        )
                        latents_std = latents_std.view(
                            1, audio_latents.shape[1], 1, audio_latents.shape[3]
                        )
                    audio_latents = audio_latents * latents_std + latents_mean

                with torch.no_grad():
                    # Decode latents to spectrogram
                    spectrogram = self.audio_vae.decode(
                        audio_latents, return_dict=False
                    )[0]
                if profile_decode_substages:
                    substage_start = self._record_decode_substage(
                        batch,
                        "LTX2AVDecodingStage.audio_decode",
                        substage_start,
                        sync=True,
                    )

            with self.use_declared_component(
                component_name="vocoder",
                module=self.vocoder,
            ) as vocoder:
                assert vocoder is not None
                self.vocoder = vocoder
                self.vocoder.eval()
                if hasattr(self.vocoder, "conv_in") and hasattr(
                    self.vocoder.conv_in, "in_channels"
                ):
                    expected_in = int(self.vocoder.conv_in.in_channels)
                    actual_in = int(spectrogram.shape[1]) * int(spectrogram.shape[3])
                    if actual_in != expected_in:
                        raise ValueError(
                            f"Vocoder expects channels*mel_bins={expected_in}, got {actual_in} from spectrogram shape {tuple(spectrogram.shape)}"
                        )
                # Decode spectrogram to waveform
                with torch.no_grad():
                    waveform = self.vocoder(spectrogram)
                if profile_decode_substages:
                    substage_start = self._record_decode_substage(
                        batch,
                        "LTX2AVDecodingStage.vocoder",
                        substage_start,
                        sync=True,
                    )
            output_batch.audio = waveform.cpu().float()
            if profile_decode_substages:
                substage_start = self._record_decode_substage(
                    batch,
                    "LTX2AVDecodingStage.audio_to_cpu",
                    substage_start,
                    sync=False,
                )
            try:
                pipeline_audio_cfg = server_args.pipeline_config.audio_vae_config
            except AttributeError:
                pipeline_audio_cfg = None
            try:
                pipeline_audio_arch = pipeline_audio_cfg.arch_config  # type: ignore[union-attr]
            except AttributeError:
                pipeline_audio_arch = None
            try:
                pipeline_audio_sr = pipeline_audio_arch.sample_rate  # type: ignore[union-attr]
            except AttributeError:
                pipeline_audio_sr = None

            try:
                vocoder_sr = self.vocoder.sample_rate
            except AttributeError:
                vocoder_sr = None
            try:
                audio_vae_sr = self.audio_vae.sample_rate
            except AttributeError:
                audio_vae_sr = None
            output_batch.audio_sample_rate = (
                vocoder_sr or audio_vae_sr or pipeline_audio_sr
            )

        return output_batch
