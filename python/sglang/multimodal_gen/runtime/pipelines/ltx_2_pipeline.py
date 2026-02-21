import inspect
import json
import os

from sglang.multimodal_gen.runtime.pipelines_core.composed_pipeline_base import (
    ComposedPipelineBase,
)
from sglang.multimodal_gen.runtime.pipelines_core.schedule_batch import Req
from sglang.multimodal_gen.runtime.pipelines_core.stages import (
    InputValidationStage,
    LTX2AVDecodingStage,
    LTX2AVDenoisingStage,
    LTX2AVLatentPreparationStage,
    LTX2TextConnectorStage,
    TextEncodingStage,
    TimestepPreparationStage,
)
from sglang.multimodal_gen.runtime.server_args import ServerArgs
from sglang.multimodal_gen.runtime.utils.logging_utils import init_logger

logger = init_logger(__name__)


def calculate_shift(
    image_seq_len,
    base_seq_len: int = 256,
    max_seq_len: int = 4096,
    base_shift: float = 0.5,
    max_shift: float = 1.15,
):
    m = (max_shift - base_shift) / (max_seq_len - base_seq_len)
    b = base_shift - m * base_seq_len
    mu = image_seq_len * m + b
    return mu


def prepare_mu(batch: Req, server_args: ServerArgs):
    height = batch.height
    width = batch.width
    num_frames = batch.num_frames

    vae_arch = getattr(
        getattr(server_args.pipeline_config, "vae_config", None), "arch_config", None
    )
    vae_scale_factor = (
        getattr(vae_arch, "spatial_compression_ratio", None)
        or getattr(vae_arch, "vae_scale_factor", None)
        or getattr(server_args.pipeline_config, "vae_scale_factor", None)
    )
    vae_temporal_compression = getattr(
        vae_arch, "temporal_compression_ratio", None
    ) or getattr(server_args.pipeline_config, "vae_temporal_compression", None)

    latent_num_frames = (int(num_frames) - 1) // int(vae_temporal_compression) + 1
    latent_height = int(height) // int(vae_scale_factor)
    latent_width = int(width) // int(vae_scale_factor)
    video_sequence_length = latent_num_frames * latent_height * latent_width

    # Values from LTX2Pipeline in diffusers
    mu = calculate_shift(
        video_sequence_length,
        base_seq_len=1024,
        max_seq_len=4096,
        base_shift=0.95,
        max_shift=2.05,
    )
    return "mu", mu


def _load_component_config(model_path: str, component_name: str):
    """Helper to load component config from model_index.json or config.json"""
    try:
        # Try loading model_index.json first
        index_path = os.path.join(model_path, "model_index.json")
        if os.path.exists(index_path):
            with open(index_path, "r") as f:
                index = json.load(f)

            if component_name in index:
                # It's a subfolder
                subfolder = index[component_name][1]
                config_path = os.path.join(model_path, subfolder, "config.json")
                if os.path.exists(config_path):
                    with open(config_path, "r") as f:
                        return json.load(f)

        # Fallback to direct config.json in subfolder if standard structure
        config_path = os.path.join(model_path, component_name, "config.json")
        if os.path.exists(config_path):
            with open(config_path, "r") as f:
                return json.load(f)

    except Exception as e:
        logger.warning(f"Failed to load config for {component_name}: {e}")

    return {}


def _filter_kwargs_for_cls(cls, kwargs):
    """Filter kwargs to only include those accepted by cls.__init__"""
    sig = inspect.signature(cls.__init__)
    return {k: v for k, v in kwargs.items() if k in sig.parameters}


class LTX2Pipeline(ComposedPipelineBase):
    # NOTE: must match `model_index.json`'s `_class_name` for native dispatch.
    pipeline_name = "LTX2Pipeline"

    _required_config_modules = [
        "transformer",
        "text_encoder",
        "tokenizer",
        "scheduler",
        "vae",
        "audio_vae",
        "vocoder",
        "connectors",
    ]

    def create_pipeline_stages(self, server_args: ServerArgs):
        """Set up pipeline stages with proper dependency injection."""

        # 1. Input Validation
        self.add_stage(
            stage_name="input_validation_stage", stage=InputValidationStage()
        )

        # 2. Text Encoding
        self.add_stage(
            stage_name="text_encoding_stage",
            stage=TextEncodingStage(
                # LTX-2 needs two contexts (video/audio). We reuse the same
                # underlying Gemma encoder/tokenizer twice.
                text_encoders=[
                    self.get_module("text_encoder"),
                ],
                tokenizers=[
                    self.get_module("tokenizer"),
                ],
            ),
        )

        # 3. connector stage
        self.add_stage(
            stage_name="text_connector_stage",
            stage=LTX2TextConnectorStage(connectors=self.get_module("connectors")),
        )

        # 4. Timestep Preparation
        self.add_stage(
            stage_name="timestep_preparation_stage",
            stage=TimestepPreparationStage(
                scheduler=self.get_module("scheduler"),
                prepare_extra_set_timesteps_kwargs=[prepare_mu],
            ),
        )

        # 4. Latent Preparation
        self.add_stage(
            stage_name="latent_preparation_stage",
            stage=LTX2AVLatentPreparationStage(
                scheduler=self.get_module("scheduler"),
                transformer=self.get_module("transformer"),
                audio_vae=self.get_module("audio_vae"),
            ),
        )

        # 5. Denoising
        self.add_stage(
            stage_name="denoising_stage",
            stage=LTX2AVDenoisingStage(
                transformer=self.get_module("transformer"),
                scheduler=self.get_module("scheduler"),
                vae=self.get_module("vae"),
                audio_vae=self.get_module("audio_vae"),
            ),
        )

        # 6. Decoding
        self.add_stage(
            stage_name="decoding_stage",
            stage=LTX2AVDecodingStage(
                vae=self.get_module("vae"),
                audio_vae=self.get_module("audio_vae"),
                vocoder=self.get_module("vocoder"),
                pipeline=self,
            ),
        )


EntryClass = LTX2Pipeline
