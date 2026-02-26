import inspect
import json
import math
import os

import numpy as np
import torch
from diffusers import FlowMatchEulerDiscreteScheduler

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

    # Values from LTX2Pipeline in diffusers
    mu = calculate_shift(
        4096,
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


class LTX2FlowMatchScheduler(FlowMatchEulerDiscreteScheduler):
    """Override ``_time_shift_exponential`` to use torch f32 instead of numpy f64."""

    def _time_shift_exponential(self, mu, sigma, t):
        if isinstance(t, np.ndarray):
            t_torch = torch.from_numpy(t).to(torch.float32)
            result = math.exp(mu) / (math.exp(mu) + (1 / t_torch - 1) ** sigma)
            return result.numpy()
        return math.exp(mu) / (math.exp(mu) + (1 / t - 1) ** sigma)


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

    def initialize_pipeline(self, server_args: ServerArgs):
        orig = self.get_module("scheduler")
        self.modules["scheduler"] = LTX2FlowMatchScheduler.from_config(orig.config)

    def create_pipeline_stages(self, server_args: ServerArgs):
        self.add_stages(
            [
                InputValidationStage(),
                TextEncodingStage(
                    text_encoders=[self.get_module("text_encoder")],
                    tokenizers=[self.get_module("tokenizer")],
                ),
                LTX2TextConnectorStage(connectors=self.get_module("connectors")),
            ]
        )

        self.add_standard_timestep_preparation_stage(prepare_extra_kwargs=[prepare_mu])

        self.add_stages(
            [
                LTX2AVLatentPreparationStage(
                    scheduler=self.get_module("scheduler"),
                    transformer=self.get_module("transformer"),
                    audio_vae=self.get_module("audio_vae"),
                ),
                LTX2AVDenoisingStage(
                    transformer=self.get_module("transformer"),
                    scheduler=self.get_module("scheduler"),
                    vae=self.get_module("vae"),
                    audio_vae=self.get_module("audio_vae"),
                ),
                LTX2AVDecodingStage(
                    vae=self.get_module("vae"),
                    audio_vae=self.get_module("audio_vae"),
                    vocoder=self.get_module("vocoder"),
                    pipeline=self,
                ),
            ]
        )


EntryClass = LTX2Pipeline
