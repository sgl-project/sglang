from dataclasses import dataclass

from sglang.multimodal_gen.configs.pipeline_configs.base import (
    ModelTaskType,
    PipelineConfig,
)


@dataclass
class LongCatAudioDiTPipelineConfig(PipelineConfig):
    """Pipeline configuration for LongCat-AudioDiT.

    LongCat-AudioDiT is a text-to-speech (TTS) / voice-cloning model built on a
    Conditional Flow Matching DiT backbone with a WAV-VAE audio codec.

    Unlike image/video pipelines the entire generation loop (ODE solve + VAE
    decode) is handled inside the model's own ``forward`` method, so most of the
    standard ``DenoisingStage`` / ``DecodingStage`` callbacks are not used.
    The pipeline uses a single monolithic ``LongCatAudioDiTInferenceStage`` that
    drives the full inference and returns the waveform directly.
    """

    task_type: ModelTaskType = ModelTaskType.T2A

    # ── precision ─────────────────────────────────────────────────────────────
    # The DiT transformer runs in bfloat16; the WAV-VAE encoder/decoder run in
    # float16.
    dit_precision: str = "bf16"
    vae_precision: str = "fp16"
