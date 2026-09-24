# SPDX-License-Identifier: Apache-2.0
"""FLUX 3 Action robot policy pipeline (``black-forest-labs/flux-3-action-*``)."""

from __future__ import annotations

import os

import torch
from safetensors.torch import load_file

from sglang.multimodal_gen.configs.pipeline_configs.flux3_action import (
    Flux3ActionPipelineConfig,
    resolve_flux3_action_package,
    verify_flux3_action_manifest,
)
from sglang.multimodal_gen.configs.sample.flux3_action import Flux3ActionSamplingParams
from sglang.multimodal_gen.runtime.disaggregation.roles import RoleType
from sglang.multimodal_gen.runtime.distributed import get_local_torch_device
from sglang.multimodal_gen.runtime.loader.fsdp_load import maybe_load_fsdp_model
from sglang.multimodal_gen.runtime.loader.utils import (
    get_memory_usage_of_component,
    set_default_torch_dtype,
)
from sglang.multimodal_gen.runtime.models.dits.flux3 import (
    Flux3Transformer,
    load_fp8r_checkpoint,
)
from sglang.multimodal_gen.runtime.models.encoders.flux3_text_encoder import (
    Flux3TextEncoder,
    parse_weight_spec,
)
from sglang.multimodal_gen.runtime.models.vaes.flux3_video_vae import Flux3VideoVAE
from sglang.multimodal_gen.runtime.pipelines_core.composed_pipeline_base import (
    ComposedPipelineBase,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.flux3_action import (
    Flux3ActionConditioningStage,
    Flux3ActionDenoisingStage,
    Flux3ActionPreprocessStage,
    make_cosmos_unipc_scheduler,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.vla import (
    VLAActionPostprocessStage,
)
from sglang.multimodal_gen.runtime.server_args import ServerArgs
from sglang.multimodal_gen.runtime.utils.hf_diffusers_utils import hf_hub_download
from sglang.multimodal_gen.runtime.utils.logging_utils import init_logger
from sglang.multimodal_gen.runtime.utils.precision import resolve_component_precision

logger = init_logger(__name__)

VIDEO_VAE_FILENAME = "video_vae.safetensors"


def _resolve_file(spec: str, default_filename: str) -> str:
    """A local file as is, otherwise ``repo_id[:filename][@revision]`` from the Hub."""
    if os.path.exists(spec):
        return spec
    repo_id, filename, revision = parse_weight_spec(spec)
    return hf_hub_download(repo_id, filename or default_filename, revision=revision)


class Flux3ActionPipeline(ComposedPipelineBase):
    pipeline_name = "Flux3ActionPipeline"
    pipeline_config_cls = Flux3ActionPipelineConfig
    sampling_params_cls = Flux3ActionSamplingParams
    _required_config_modules: list[str] = []

    def validate_disagg_role(self, role: RoleType) -> None:
        if role != RoleType.MONOLITHIC:
            raise ValueError("Flux3ActionPipeline supports same-process execution only")

    def load_modules(
        self,
        server_args: ServerArgs,
        loaded_modules: dict[str, torch.nn.Module] | None = None,
    ) -> dict[str, torch.nn.Module]:
        if loaded_modules is not None:
            return loaded_modules
        config: Flux3ActionPipelineConfig = server_args.pipeline_config
        if config.quantization not in (None, "fp8r"):
            raise NotImplementedError(
                f"FLUX 3 Action {config.quantization} packages are not supported"
            )
        package = resolve_flux3_action_package(
            self.model_path, config.policy_variant, revision=config.policy_revision
        )
        verify_flux3_action_manifest(package, include_weights=True)
        modules = {
            "transformer": self._load_transformer(server_args, config, package),
            "vae": self._load_vae(server_args, config),
            "text_encoder": self._load_text_encoder(server_args, config),
        }
        for name, module in modules.items():
            module.requires_grad_(False).eval()
            self.memory_usages[name] = get_memory_usage_of_component(module)
        modules["scheduler"] = make_cosmos_unipc_scheduler()
        return modules

    @staticmethod
    def _load_transformer(
        server_args: ServerArgs, config: Flux3ActionPipelineConfig, package
    ) -> Flux3Transformer:
        logger.info("Loading FLUX 3 Action DiT from %s", package)
        weights = str(package / "model.safetensors")
        device = get_local_torch_device()
        if config.quantization == "fp8r":
            # Native FP8r payloads load as they are (no requantization).
            with torch.device("meta"), set_default_torch_dtype(torch.bfloat16):
                transformer = Flux3Transformer(config=config.dit_config, hf_config={})
            load_fp8r_checkpoint(transformer, load_file(weights, device=str(device)))
            return transformer
        return maybe_load_fsdp_model(
            model_cls=Flux3Transformer,
            init_params={"config": config.dit_config, "hf_config": {}},
            weight_dir_list=[weights],
            device=device,
            hsdp_replicate_dim=server_args.hsdp_replicate_dim,
            hsdp_shard_dim=server_args.hsdp_shard_dim,
            param_dtype=resolve_component_precision(server_args, "transformer"),
            reduce_dtype=torch.float32,
            component_starts_on_cpu=server_args.should_start_component_on_cpu(
                "transformer"
            ),
            fsdp_inference=server_args.should_use_fsdp_for_component("transformer"),
            pin_cpu_memory=server_args.pin_cpu_memory,
            strict=True,
        )

    @staticmethod
    def _load_vae(
        server_args: ServerArgs, config: Flux3ActionPipelineConfig
    ) -> Flux3VideoVAE:
        spec = server_args.component_paths.get("vae") or config.video_vae_id
        logger.info("Loading FLUX 3 video VAE from %s", spec)
        device = get_local_torch_device()
        with torch.device("meta"):
            vae = Flux3VideoVAE(config.vae_config)
        vae.load_checkpoint(
            load_file(_resolve_file(spec, VIDEO_VAE_FILENAME), device=str(device))
        )
        return vae.to(resolve_component_precision(server_args, "vae"))

    @staticmethod
    def _load_text_encoder(
        server_args: ServerArgs, config: Flux3ActionPipelineConfig
    ) -> Flux3TextEncoder:
        spec = server_args.component_paths.get("text_encoder") or config.text_encoder_id
        logger.info("Loading FLUX 3 text encoder from %s", spec)
        text_encoder = Flux3TextEncoder(
            spec,
            output_layers=config.text_output_layers,
            pad_multiple=config.text_pad_multiple,
            max_length=config.text_max_length,
        )
        on_cpu = server_args.should_start_component_on_cpu("text_encoder")
        return text_encoder.to("cpu" if on_cpu else get_local_torch_device())

    def create_pipeline_stages(self, server_args: ServerArgs):
        config: Flux3ActionPipelineConfig = server_args.pipeline_config
        transformer = self.get_module("transformer")
        self.add_stage(Flux3ActionPreprocessStage(config), "flux3_action_preprocess")
        self.add_stage(
            Flux3ActionConditioningStage(
                config,
                transformer,
                self.get_module("vae"),
                self.get_module("text_encoder"),
            ),
            "flux3_action_conditioning",
        )
        self.add_stage(
            Flux3ActionDenoisingStage(
                config, transformer, scheduler=self.get_module("scheduler")
            ),
            "flux3_action_denoise",
        )
        self.add_stage(VLAActionPostprocessStage(), "flux3_action_postprocess")


EntryClass = Flux3ActionPipeline
