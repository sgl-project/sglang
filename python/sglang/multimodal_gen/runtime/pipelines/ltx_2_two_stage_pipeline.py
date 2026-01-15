# ================================================================
# LTX2TwoStagePipeline WIP Now
# ================================================================


import inspect
import json
import os

import torch
from huggingface_hub import snapshot_download
from safetensors.torch import load_file as load_safetensors

from sglang.multimodal_gen.runtime.models.upsamplers.ltx_2_upsampler import (
    LatentUpsampler,
)
from sglang.multimodal_gen.runtime.models.vaes.ltx_2_audio import LTX2AudioDecoder
from sglang.multimodal_gen.runtime.models.vocoder.ltx_2_vocoder import LTX2Vocoder
from sglang.multimodal_gen.runtime.pipelines_core.composed_pipeline_base import (
    ComposedPipelineBase,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages import (
    InputValidationStage,
    LTX2AVDecodingStage,
    LTX2AVDenoisingStage,
    LTX2AVLatentPreparationStage,
    TextEncodingStage,
    TimestepPreparationStage,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.denoising_av import (
    LTX2RefinementStage,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.upsampling import (
    LTX2UpsamplingStage,
)
from sglang.multimodal_gen.runtime.server_args import ServerArgs
from sglang.multimodal_gen.runtime.utils.logging_utils import init_logger

logger = init_logger(__name__)


def _filter_kwargs_for_cls(cls, cfg: dict) -> dict:
    sig = inspect.signature(cls.__init__)
    valid = set(sig.parameters.keys()) - {"self"}
    return {k: v for k, v in cfg.items() if k in valid}


def _load_component_config(model_path: str, subfolder: str) -> dict:
    if os.path.isdir(model_path):
        cfg_path = os.path.join(model_path, subfolder, "config.json")
        if os.path.exists(cfg_path):
            with open(cfg_path, "r", encoding="utf-8") as f:
                return json.load(f)
        return {}

    local_dir = snapshot_download(
        repo_id=model_path,
        allow_patterns=[f"{subfolder}/*"],
        local_files_only=False,
    )
    cfg_path = os.path.join(local_dir, subfolder, "config.json")
    if os.path.exists(cfg_path):
        with open(cfg_path, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


# From ltx_pipelines/utils/constants.py
STAGE_2_DISTILLED_SIGMA_VALUES = [0.909375, 0.725, 0.421875, 0.0]


def load_model_weights(
    model, model_path, subfolder, filename="diffusion_pytorch_model.safetensors"
):
    """Helper to load weights from local path or HF Hub."""
    # Check if model_path is a local directory
    if os.path.isdir(model_path):
        file_path = os.path.join(model_path, subfolder, filename)
    else:
        # Assume HF Hub ID
        try:
            file_path = snapshot_download(
                repo_id=model_path,
                allow_patterns=[f"{subfolder}/*"],
                local_files_only=False,
            )
            file_path = os.path.join(file_path, subfolder, filename)
        except Exception as e:
            logger.warning(f"Failed to download {subfolder} from {model_path}: {e}")
            return

    if not os.path.exists(file_path):
        # Try .bin if .safetensors not found
        if filename.endswith(".safetensors"):
            bin_filename = filename.replace(".safetensors", ".bin")
            return load_model_weights(model, model_path, subfolder, bin_filename)
        logger.warning(
            f"Weight file not found: {file_path}. Using random initialization."
        )
        return

    logger.info(f"Loading {subfolder} weights from {file_path}")
    if file_path.endswith(".safetensors"):
        state_dict = load_safetensors(file_path)
    else:
        state_dict = torch.load(file_path, map_location="cpu")

    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing:
        logger.warning(f"Missing keys for {subfolder}: {missing}")
    if unexpected:
        logger.warning(f"Unexpected keys for {subfolder}: {unexpected}")


class LTX2TwoStagePipeline(ComposedPipelineBase):
    # NOTE: must match `model_index.json`'s `_class_name` for native dispatch.
    pipeline_name = "LTXVideoTwoStagePipeline"

    _required_config_modules = [
        "transformer",
        "text_encoder",
        "tokenizer",
        "scheduler",
        "vae",  # Video VAE
    ]

    def create_pipeline_stages(self, server_args: ServerArgs):
        """Set up pipeline stages with proper dependency injection."""

        # Get loaded modules
        transformer = self.get_module("transformer")
        text_encoder = self.get_module("text_encoder")
        tokenizer = self.get_module("tokenizer")
        scheduler = self.get_module("scheduler")
        vae = self.get_module("vae")

        # Initialize Audio Components (prefer config.json if present)
        audio_cfg = _load_component_config(self.model_path, "audio_vae")
        vocoder_cfg = _load_component_config(self.model_path, "vocoder")
        upsampler_cfg = _load_component_config(self.model_path, "upsampler")

        audio_cfg = (
            audio_cfg.get("ddconfig")
            or audio_cfg.get("model", {}).get("params", {}).get("ddconfig")
            or audio_cfg.get("audio_vae", {})
            .get("model", {})
            .get("params", {})
            .get("ddconfig")
            or audio_cfg
        )
        vocoder_cfg = vocoder_cfg.get("vocoder") or vocoder_cfg
        upsampler_cfg = upsampler_cfg.get("upsampler") or upsampler_cfg

        audio_vae = LTX2AudioDecoder(
            **_filter_kwargs_for_cls(LTX2AudioDecoder, audio_cfg)
        )
        vocoder = LTX2Vocoder(**_filter_kwargs_for_cls(LTX2Vocoder, vocoder_cfg))

        # Initialize Upsampler
        upsampler = LatentUpsampler(
            **_filter_kwargs_for_cls(LatentUpsampler, upsampler_cfg)
        )

        # Load weights
        # We assume the model_path (local or HF ID) contains these subfolders
        load_model_weights(audio_vae, self.model_path, "audio_vae")
        load_model_weights(vocoder, self.model_path, "vocoder")
        load_model_weights(upsampler, self.model_path, "upsampler")

        # Move to device
        if hasattr(transformer, "device"):
            device = transformer.device
            dtype = transformer.dtype
            audio_vae = audio_vae.to(device=device, dtype=dtype)
            vocoder = vocoder.to(device=device, dtype=dtype)
            upsampler = upsampler.to(device=device, dtype=dtype)

        self.add_module("audio_vae", audio_vae)
        self.add_module("vocoder", vocoder)
        self.add_module("upsampler", upsampler)

        # 1. Input Validation
        self.add_stage(
            stage_name="input_validation_stage", stage=InputValidationStage()
        )

        # 2. Text Encoding
        self.add_stage(
            stage_name="text_encoding_stage",
            stage=TextEncodingStage(
                text_encoders=[text_encoder],
                tokenizers=[tokenizer],
            ),
        )

        # 3. Timestep Preparation (Stage 1)
        self.add_stage(
            stage_name="timestep_preparation_stage",
            stage=TimestepPreparationStage(
                scheduler=scheduler,
            ),
        )

        # 4. Latent Preparation (Stage 1 - Low Res)
        self.add_stage(
            stage_name="latent_preparation_stage",
            stage=LTX2AVLatentPreparationStage(
                scheduler=scheduler,
                transformer=transformer,
            ),
        )

        # 5. Denoising (Stage 1)
        self.add_stage(
            stage_name="denoising_stage_1",
            stage=LTX2AVDenoisingStage(
                transformer=transformer,
                scheduler=scheduler,
                vae=vae,
                audio_vae=audio_vae,
            ),
        )

        # 6. Upsampling
        self.add_stage(
            stage_name="upsampling_stage",
            stage=LTX2UpsamplingStage(
                upsampler=upsampler,
                video_encoder_stats=vae.per_channel_statistics,
            ),
        )

        # 7. Refinement (Stage 2)
        self.add_stage(
            stage_name="denoising_stage_2",
            stage=LTX2RefinementStage(
                transformer=transformer,
                scheduler=scheduler,
                distilled_sigmas=STAGE_2_DISTILLED_SIGMA_VALUES,
                vae=vae,
                audio_vae=audio_vae,
            ),
        )

        # 8. Decoding
        self.add_stage(
            stage_name="decoding_stage",
            stage=LTX2AVDecodingStage(
                vae=vae, audio_vae=audio_vae, vocoder=vocoder, pipeline=self
            ),
        )


EntryClass = LTX2TwoStagePipeline
