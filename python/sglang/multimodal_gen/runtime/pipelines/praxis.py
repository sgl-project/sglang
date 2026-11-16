# SPDX-License-Identifier: Apache-2.0
"""Praxis-1 robot policy pipeline: (``runwayml/praxis-1-*``)"""

from __future__ import annotations

import os
from pathlib import Path

import torch

from sglang.multimodal_gen.configs.pipeline_configs.praxis import (
    PraxisPipelineConfig,
    native_checkpoint_dir,
    read_praxis_config,
    resolve_praxis_package,
)
from sglang.multimodal_gen.configs.sample.praxis import PraxisSamplingParams
from sglang.multimodal_gen.runtime.disaggregation.roles import RoleType
from sglang.multimodal_gen.runtime.distributed import get_local_torch_device
from sglang.multimodal_gen.runtime.distributed.parallel_state import (
    get_cfg_group,
    get_classifier_free_guidance_rank,
)
from sglang.multimodal_gen.runtime.loader.fsdp_load import maybe_load_fsdp_model
from sglang.multimodal_gen.runtime.loader.utils import get_memory_usage_of_component
from sglang.multimodal_gen.runtime.models.dits.praxis import PraxisTransformer
from sglang.multimodal_gen.runtime.models.encoders.praxis_text_encoder import (
    PraxisTextEncoder,
)
from sglang.multimodal_gen.runtime.models.vaes.praxis_vae import PraxisVAE
from sglang.multimodal_gen.runtime.pipelines_core.composed_pipeline_base import (
    ComposedPipelineBase,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.praxis import (
    PraxisPolicyStage,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.vla import (
    VLAActionPostprocessStage,
)
from sglang.multimodal_gen.runtime.server_args import ServerArgs
from sglang.multimodal_gen.runtime.utils.hf_diffusers_utils import maybe_download_model
from sglang.multimodal_gen.runtime.utils.logging_utils import init_logger
from sglang.multimodal_gen.runtime.utils.precision import resolve_component_precision
from sglang.multimodal_gen.runtime.vla.praxis_policy import PolicyLayout, PraxisPolicy
from sglang.multimodal_gen.runtime.vla.praxis_sampler import CfgParallel

logger = init_logger(__name__)


def _component_dir(
    *,
    server_args: ServerArgs,
    name: str,
    spec: str,
    allow_patterns: list[str] | None = None,
) -> Path:
    """A component directory: ``--component-paths.<name>``, or a local path or Hub repo id from the config."""
    spec = server_args.component_paths.get(name) or spec
    if os.path.exists(spec):
        return Path(spec)
    return Path(maybe_download_model(spec, allow_patterns=allow_patterns))


class PraxisPipeline(ComposedPipelineBase):
    pipeline_name = "PraxisPipeline"
    pipeline_config_cls = PraxisPipelineConfig
    sampling_params_cls = PraxisSamplingParams
    _required_config_modules: list[str] = []

    def validate_disagg_role(self, role: RoleType) -> None:
        if role != RoleType.MONOLITHIC:
            raise ValueError("PraxisPipeline supports same-process execution only")

    def load_modules(
        self,
        server_args: ServerArgs,
        loaded_modules: dict[str, torch.nn.Module] | None = None,
    ) -> dict[str, torch.nn.Module]:
        if loaded_modules is not None:
            return loaded_modules
        config: PraxisPipelineConfig = server_args.pipeline_config
        package = resolve_praxis_package(self.model_path, revision=server_args.revision)
        logger.info("Loading Praxis-1 from %s", package)
        model_config = read_praxis_config(package)["model_config"]
        device = get_local_torch_device()
        transformer = self._load_transformer(server_args, config, package)
        vae_dir = _component_dir(
            server_args=server_args,
            name="vae",
            spec=config.vae_id,
            allow_patterns=[f"vae/*"],
        )
        # A repo id gives the diffusers repo; a path may name the repo or its vae folder.
        vae = PraxisVAE(
            vae_dir / "vae" if (vae_dir / "vae").is_dir() else vae_dir,
            device=device,
        )
        text_encoder = PraxisTextEncoder(
            _component_dir(
                server_args=server_args,
                name="text_encoder",
                spec=config.text_encoder_id,
            ),
            device=device,
        )
        self._policy = PraxisPolicy(
            transformer=transformer,
            vae=vae,
            text_encoder=text_encoder,
            layout=PolicyLayout.from_model_config(model_config),
            cfg_parallel=self._cfg_parallel(server_args),
        )
        modules = {
            "transformer": transformer,
            "vae": vae.model,
            "text_encoder": text_encoder.model,
        }
        for name, module in modules.items():
            self.memory_usages[name] = get_memory_usage_of_component(module)
        return modules

    @staticmethod
    def _cfg_parallel(server_args: ServerArgs) -> CfgParallel | None:
        """The split of a guided step over the two CFG ranks, when CFG parallel is on."""
        if not server_args.enable_cfg_parallel:
            return None
        group = get_cfg_group()
        return CfgParallel(
            rank=get_classifier_free_guidance_rank(),
            share=lambda tensor, src: group.broadcast(tensor, src=src),
        )

    @staticmethod
    def _load_transformer(
        server_args: ServerArgs, config: PraxisPipelineConfig, package: Path
    ) -> PraxisTransformer:
        weights = sorted(
            str(p) for p in native_checkpoint_dir(package).glob("*.safetensors")
        )
        return maybe_load_fsdp_model(
            model_cls=PraxisTransformer,
            init_params={"config": config.dit_config, "hf_config": {}},
            weight_dir_list=weights,
            device=get_local_torch_device(),
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

    def create_pipeline_stages(self, server_args: ServerArgs):
        self.add_stage(
            PraxisPolicyStage(config=server_args.pipeline_config, policy=self._policy),
            "praxis_policy",
        )
        self.add_stage(VLAActionPostprocessStage(), "praxis_postprocess")


EntryClass = PraxisPipeline
