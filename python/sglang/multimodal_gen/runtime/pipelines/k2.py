"""Krea-2 (K2) text-to-image pipeline.

K2 ships a single MMDiT safetensors (no diffusers ``model_index.json``); the
text encoder (Qwen3-VL-4B) and the autoencoder (Qwen-Image VAE) are pulled from
their own Hugging Face repos. ``load_modules`` assembles the three sources, then
the Hybrid stage chain runs: Krea2BeforeDenoisingStage -> DenoisingStage ->
DecodingStage.
"""

import os
from typing import Any

import torch
from transformers import (
    AutoTokenizer,
    Qwen2TokenizerFast,
    Qwen3VLForConditionalGeneration,
)

from sglang.multimodal_gen.configs.models.dits.k2 import K2DitConfig
from sglang.multimodal_gen.runtime.distributed import get_local_torch_device
from sglang.multimodal_gen.runtime.loader.component_loaders.vae_loader import VAELoader
from sglang.multimodal_gen.runtime.loader.fsdp_load import (
    load_model_from_full_model_state_dict,
)
from sglang.multimodal_gen.runtime.loader.utils import (
    get_param_names_mapping,
    set_default_torch_dtype,
)
from sglang.multimodal_gen.runtime.loader.weight_utils import (
    safetensors_weights_iterator,
)
from sglang.multimodal_gen.runtime.models.registry import ModelRegistry
from sglang.multimodal_gen.runtime.models.schedulers.scheduling_k2_flow import (
    K2FlowMatchScheduler,
)
from sglang.multimodal_gen.runtime.pipelines_core import LoRAPipeline
from sglang.multimodal_gen.runtime.pipelines_core.composed_pipeline_base import (
    ComposedPipelineBase,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages import DenoisingStage
from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.k2 import (
    Krea2BeforeDenoisingStage,
)
from sglang.multimodal_gen.runtime.server_args import ServerArgs
from sglang.multimodal_gen.runtime.utils.hf_diffusers_utils import maybe_download_model
from sglang.multimodal_gen.runtime.utils.logging_utils import init_logger
from sglang.multimodal_gen.runtime.utils.precision import resolve_precision

logger = init_logger(__name__)

TEXT_ENCODER_REPO = "Qwen/Qwen3-VL-4B-Instruct"
VAE_REPO = "Qwen/Qwen-Image"
_TEXT_MAX_LENGTH = 512


class K2Pipeline(LoRAPipeline, ComposedPipelineBase):
    pipeline_name = "K2Pipeline"

    from sglang.multimodal_gen.configs.pipeline_configs.k2 import K2PipelineConfig
    from sglang.multimodal_gen.configs.sample.k2 import K2SamplingParams

    pipeline_config_cls = K2PipelineConfig
    sampling_params_cls = K2SamplingParams

    _required_config_modules = [
        "text_encoder",
        "tokenizer",
        "processor",
        "vae",
        "transformer",
        "scheduler",
    ]

    def initialize_pipeline(self, server_args: ServerArgs):
        self.modules["scheduler"] = K2FlowMatchScheduler()
        vae_config = server_args.pipeline_config.vae_config
        if hasattr(vae_config, "post_init"):
            vae_config.post_init()

    def _resolve_dit_weights(self) -> str:
        if os.path.isfile(self.model_path) and self.model_path.endswith(".safetensors"):
            return self.model_path
        # Directory layout: default to the distilled turbo checkpoint.
        for name in ("turbo.safetensors", "raw.safetensors"):
            candidate = os.path.join(self.model_path, name)
            if os.path.isfile(candidate):
                return candidate
        raise FileNotFoundError(f"No K2 MMDiT safetensors found at {self.model_path}")

    def _load_transformer(self, server_args: ServerArgs):
        dit_weights = self._resolve_dit_weights()
        logger.info("Loading K2 MMDiT from %s", dit_weights)

        dit_config = server_args.pipeline_config.dit_config
        if not isinstance(dit_config, K2DitConfig):
            dit_config = K2DitConfig()
            server_args.pipeline_config.dit_config = dit_config

        model_cls, _ = ModelRegistry.resolve_model_cls("K2Transformer2DModel")
        default_dtype = resolve_precision(
            server_args, "dit", precision_attr="dit_precision"
        )
        server_args.model_paths["transformer"] = os.path.dirname(dit_weights) or "."

        with set_default_torch_dtype(default_dtype), torch.device("meta"):
            model = model_cls(config=dit_config, hf_config={})

        load_model_from_full_model_state_dict(
            model,
            safetensors_weights_iterator([dit_weights]),
            get_local_torch_device(),
            default_dtype,
            strict=True,
            cpu_offload=server_args.dit_cpu_offload,
            param_names_mapping=get_param_names_mapping(
                dit_config.arch_config.param_names_mapping
            ),
        )
        for n, p in model.named_parameters():
            p.requires_grad = False
        return model

    def load_modules(
        self,
        server_args: ServerArgs,
        loaded_modules: dict[str, torch.nn.Module] | None = None,
    ) -> dict[str, Any]:
        device = get_local_torch_device()
        transformer = self._load_transformer(server_args)

        logger.info("Loading text encoder %s", TEXT_ENCODER_REPO)
        te_dtype = torch.bfloat16
        text_encoder = (
            Qwen3VLForConditionalGeneration.from_pretrained(
                TEXT_ENCODER_REPO, torch_dtype=te_dtype
            )
            .to(device=device, dtype=te_dtype)
            .eval()
            .requires_grad_(False)
        )
        tokenizer = AutoTokenizer.from_pretrained(
            TEXT_ENCODER_REPO, max_length=_TEXT_MAX_LENGTH
        )
        processor = Qwen2TokenizerFast.from_pretrained(
            TEXT_ENCODER_REPO, max_length=_TEXT_MAX_LENGTH
        )

        logger.info("Loading Qwen-Image VAE from %s", VAE_REPO)
        vae_path = os.path.join(maybe_download_model(VAE_REPO), "vae")
        vae = VAELoader().load_customized(vae_path, server_args, "vae")

        return {
            "transformer": transformer,
            "text_encoder": text_encoder,
            "tokenizer": tokenizer,
            "processor": processor,
            "vae": vae,
            "scheduler": self.modules.get("scheduler"),
        }

    def create_pipeline_stages(self, server_args: ServerArgs):
        self.add_stage(
            Krea2BeforeDenoisingStage(
                vae=self.get_module("vae"),
                text_encoder=self.get_module("text_encoder"),
                tokenizer=self.get_module("tokenizer"),
                processor=self.get_module("processor"),
                transformer=self.get_module("transformer"),
                scheduler=self.get_module("scheduler"),
            ),
            "k2_before_denoising_stage",
        )
        self.add_stage(
            DenoisingStage(
                transformer=self.get_module("transformer"),
                scheduler=self.get_module("scheduler"),
            ),
        )
        self.add_standard_decoding_stage()


EntryClass = [K2Pipeline]
