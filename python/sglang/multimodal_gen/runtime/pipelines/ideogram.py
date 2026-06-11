# SPDX-License-Identifier: Apache-2.0

import os
from dataclasses import dataclass
from functools import lru_cache
from typing import Any, cast

from sglang.multimodal_gen.runtime.disaggregation.roles import RoleType
from sglang.multimodal_gen.runtime.pipelines_core import LoRAPipeline
from sglang.multimodal_gen.runtime.pipelines_core.composed_pipeline_base import (
    ComposedPipelineBase,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.input_validation import (
    InputValidationStage,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.ideogram import (
    Ideogram4DecodingStage,
    Ideogram4DenoisingStage,
    Ideogram4TextEncodingStage,
)
from sglang.multimodal_gen.runtime.server_args import ServerArgs
from sglang.multimodal_gen.runtime.utils.hf_diffusers_utils import (
    maybe_download_model,
    verify_model_config_and_directory,
)
from sglang.multimodal_gen.runtime.utils.logging_utils import init_logger

logger = init_logger(__name__)

_IDEOGRAM4_BASE_MODEL = "ideogram-ai/ideogram-4-fp8"
_IDEOGRAM4_NVFP4_COND_FILE = "diffusion_models/ideogram4_nvfp4_mixed.safetensors"
_IDEOGRAM4_NVFP4_UNCOND_FILE = (
    "diffusion_models/ideogram4_unconditional_nvfp4_mixed.safetensors"
)


@dataclass(frozen=True)
class Ideogram4Nvfp4ModelResolution:
    base_model_name: str
    base_model_path: str
    transformer_weights_path: str
    unconditional_transformer_weights_path: str | None


@lru_cache(maxsize=1)
def _resolve_ideogram4_base_model_path() -> str:
    return maybe_download_model(_IDEOGRAM4_BASE_MODEL, force_diffusers_model=True)


def _resolve_ideogram4_unconditional_transformer_weights_path(
    transformer_weights_path: str,
) -> str | None:
    if os.path.basename(transformer_weights_path) != os.path.basename(
        _IDEOGRAM4_NVFP4_COND_FILE
    ):
        return None
    return os.path.join(
        os.path.dirname(transformer_weights_path),
        os.path.basename(_IDEOGRAM4_NVFP4_UNCOND_FILE),
    )


def _resolve_ideogram4_nvfp4_transformer_weights_paths(
    server_args: ServerArgs, model_path: str
) -> tuple[str, str | None]:
    if server_args.transformer_weights_path is not None:
        transformer_weights_path = server_args.transformer_weights_path
        return (
            transformer_weights_path,
            _resolve_ideogram4_unconditional_transformer_weights_path(
                transformer_weights_path
            ),
        )

    local_nvfp4_path = maybe_download_model(
        model_path,
        allow_patterns=[
            _IDEOGRAM4_NVFP4_COND_FILE,
            _IDEOGRAM4_NVFP4_UNCOND_FILE,
        ],
    )
    return (
        os.path.join(local_nvfp4_path, _IDEOGRAM4_NVFP4_COND_FILE),
        os.path.join(local_nvfp4_path, _IDEOGRAM4_NVFP4_UNCOND_FILE),
    )


def resolve_ideogram4_nvfp4_model(
    server_args: ServerArgs, model_path: str
) -> Ideogram4Nvfp4ModelResolution:
    (
        transformer_weights_path,
        unconditional_transformer_weights_path,
    ) = _resolve_ideogram4_nvfp4_transformer_weights_paths(
        server_args,
        model_path,
    )
    return Ideogram4Nvfp4ModelResolution(
        base_model_name=_IDEOGRAM4_BASE_MODEL,
        base_model_path=_resolve_ideogram4_base_model_path(),
        transformer_weights_path=transformer_weights_path,
        unconditional_transformer_weights_path=unconditional_transformer_weights_path,
    )


class Ideogram4Pipeline(LoRAPipeline, ComposedPipelineBase):
    pipeline_name = "Ideogram4Pipeline"

    _required_config_modules = [
        "text_encoder",
        "tokenizer",
        "vae",
        "transformer",
        "unconditional_transformer",
        "scheduler",
    ]

    def create_pipeline_stages(self, server_args: ServerArgs):
        self.add_stage(InputValidationStage())
        self.add_stage_factory(
            RoleType.ENCODER,
            lambda: Ideogram4TextEncodingStage(
                text_encoder=self.get_module("text_encoder"),
                tokenizer=self.get_module("tokenizer"),
            ),
            "ideogram4_text_encoding_stage",
        )
        self.add_standard_latent_preparation_stage()
        self.add_stage_factory(
            RoleType.DENOISER,
            lambda: Ideogram4DenoisingStage(
                transformer=self.get_module("transformer"),
                unconditional_transformer=self.get_module("unconditional_transformer"),
                pipeline=self,
            ),
            "ideogram4_denoising_stage",
        )
        self.add_stage_factory(
            RoleType.DECODER,
            lambda: Ideogram4DecodingStage(vae=self.get_module("vae")),
            "ideogram4_decoding_stage",
        )


class Ideogram4Nvfp4Pipeline(Ideogram4Pipeline):
    pipeline_name = "Ideogram4Nvfp4Pipeline"
    _model_resolution: Ideogram4Nvfp4ModelResolution | None = None

    def _get_model_resolution(
        self,
        server_args: ServerArgs | None = None,
    ) -> Ideogram4Nvfp4ModelResolution:
        if self._model_resolution is None:
            if server_args is None:
                raise ValueError(
                    "server_args is required to resolve Ideogram4 NVFP4 paths"
                )
            self._model_resolution = resolve_ideogram4_nvfp4_model(
                server_args,
                self.model_path,
            )
        return self._model_resolution

    def _load_config(self) -> dict[str, Any]:
        model_resolution = self._get_model_resolution(self.server_args)
        logger.info("Model path: %s", self.model_path)
        logger.info(
            "Using base model '%s' at %s for config and non-transformer components",
            model_resolution.base_model_name,
            model_resolution.base_model_path,
        )
        config = verify_model_config_and_directory(model_resolution.base_model_path)
        return cast(dict[str, Any], config)

    def _resolve_component_path(
        self,
        server_args: ServerArgs,
        module_name: str,
        load_module_name: str,
    ) -> str:
        override_path = server_args.component_paths.get(module_name)
        if override_path is not None:
            return maybe_download_model(override_path)

        component_model_path = os.path.join(
            self._get_model_resolution(server_args).base_model_path,
            load_module_name,
        )
        logger.debug("Resolved component path: %s", component_model_path)
        return component_model_path

    def load_modules(
        self,
        server_args: ServerArgs,
        loaded_modules: dict | None = None,
    ) -> dict:
        model_resolution = self._get_model_resolution(server_args)
        server_args.transformer_weights_path = model_resolution.transformer_weights_path
        if model_resolution.unconditional_transformer_weights_path is not None:
            # The loader treats transformer_weights_path as the base DiT override.
            # Route the sibling unconditional DiT weights through the generic
            # per-component override map instead of hard-coding Ideogram there.
            component_transformer_weights_paths = dict(
                getattr(server_args, "component_transformer_weights_paths", {})
            )
            component_transformer_weights_paths.setdefault(
                "unconditional_transformer",
                model_resolution.unconditional_transformer_weights_path,
            )
            server_args.component_transformer_weights_paths = (
                component_transformer_weights_paths
            )
        logger.info(
            "NVFP4 transformer weights: %s",
            model_resolution.transformer_weights_path,
        )
        logger.info(
            "NVFP4 unconditional transformer weights: %s",
            server_args.component_transformer_weights_paths.get(
                "unconditional_transformer"
            ),
        )
        return super().load_modules(server_args, loaded_modules)


EntryClass = [Ideogram4Pipeline, Ideogram4Nvfp4Pipeline]
