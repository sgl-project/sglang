import torch

from sglang.multimodal_gen.runtime.distributed import get_local_torch_device
from sglang.multimodal_gen.runtime.managers.memory_managers.component_manager import (
    ComponentUse,
)
from sglang.multimodal_gen.runtime.pipelines_core.schedule_batch import OutputBatch, Req
from sglang.multimodal_gen.runtime.pipelines_core.stages.decoding import DecodingStage
from sglang.multimodal_gen.runtime.platforms import current_platform
from sglang.multimodal_gen.runtime.server_args import ServerArgs
from sglang.multimodal_gen.runtime.utils.logging_utils import init_logger
from sglang.multimodal_gen import envs
from sglang.multimodal_gen.utils import PRECISION_TO_TYPE

logger = init_logger(__name__)


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
        # Lazily-compiled untiled VAE decode (see _decode_video_latents).
        self._vae_decode_fn = None
        self._vae_decode_compile_failed = False

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

    def _vae_decode_compile_mode(self, server_args: ServerArgs) -> str:
        """Resolve VAE-decode compile policy: "off" | "auto" | "force"."""
        if not server_args.enable_torch_compile:
            return "off"
        mode = str(envs.SGLANG_DIFFUSION_COMPILE_VAE_DECODE).strip().lower()
        if mode in ("0", "off", "false", "no", "disable", "disabled", ""):
            return "off"
        if mode in ("1", "on", "true", "yes", "force", "always"):
            return "force"
        return "auto"

    @staticmethod
    def _untiled_decode_fits(latents: torch.Tensor) -> bool:
        """Best-effort memory gate for an untiled VAE decode.

        Tiling bounds VAE peak memory; only skip it (to obtain a
        compile-friendly fixed-shape graph) when there is clearly enough free
        GPU memory. This only needs to be roughly right because the caller has
        an OOM fallback to tiled eager decode. The decoded volume is estimated
        from the latent volume using LTX spatial(32x)/temporal(8x) upscales and
        a conservative peak multiplier.
        """
        try:
            free, total = torch.cuda.mem_get_info()
            channels = int(latents.shape[1]) if latents.dim() >= 2 else 1
            decoded_numel = latents.numel() / max(channels, 1) * 3
            decoded_numel *= 32 * 32 * 8  # H*W*T expansion vs latent grid
            peak_bytes = decoded_numel * 2 * 8  # bf16 working dtype, ~8x peak
            return free > peak_bytes * 1.3 and (free / max(total, 1)) > 0.3
        except Exception:
            return False

    def _compiled_untiled_decode(self, latents: torch.Tensor):
        try:
            if hasattr(self.vae, "disable_tiling"):
                self.vae.disable_tiling()
        except Exception:
            pass
        if self._vae_decode_fn is None:
            import os

            mode = os.environ.get(
                "SGLANG_TORCH_COMPILE_MODE", "max-autotune-no-cudagraphs"
            )
            logger.info("Compiling LTX-2 VAE decode with mode: %s", mode)
            self._vae_decode_fn = torch.compile(
                self.vae.decode, mode=mode, fullgraph=False, dynamic=None
            )
        return self._vae_decode_fn(latents)

    def _decode_video_latents(self, latents: torch.Tensor, server_args: ServerArgs):
        """Decode video latents (tiled eager by default).

        When torch.compile is enabled and SGLANG_DIFFUSION_COMPILE_VAE_DECODE
        allows it, run an untiled fixed-shape decode wrapped in torch.compile
        (aligns with FastVideo's compiled VAE path). Always falls back to the
        tiled eager decode on OOM or any compile/runtime failure, so the
        default and high-resolution behavior never regresses.
        """
        mode = self._vae_decode_compile_mode(server_args)
        use_compiled = (
            mode != "off"
            and not self._vae_decode_compile_failed
            and (mode == "force" or self._untiled_decode_fits(latents))
        )
        if use_compiled:
            try:
                return self._compiled_untiled_decode(latents)
            except torch.cuda.OutOfMemoryError:
                logger.warning(
                    "Compiled untiled VAE decode hit OOM; falling back to "
                    "tiled eager decode for the rest of this run."
                )
            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    "Compiled untiled VAE decode failed (%s); falling back to "
                    "tiled eager decode.",
                    type(exc).__name__,
                )
            self._vae_decode_compile_failed = True
            self._vae_decode_fn = None
            try:
                torch.cuda.empty_cache()
            except Exception:
                pass
        # Default / fallback: tiled eager decode.
        try:
            if server_args.pipeline_config.vae_tiling:
                self.vae.enable_tiling()
        except Exception:
            pass
        return self.vae.decode(latents)

    def forward(self, batch: Req, server_args: ServerArgs) -> OutputBatch:
        self.load_model()

        vae_dtype = PRECISION_TO_TYPE[server_args.pipeline_config.vae_precision]
        vae_autocast_enabled = (
            vae_dtype != torch.float32
        ) and not server_args.disable_autocast

        original_dtype = vae_dtype
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

            self.vae.to(original_dtype)
        video = self.video_processor.postprocess_video(video, output_type="np")

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
            output_batch.audio = waveform.cpu().float()
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
