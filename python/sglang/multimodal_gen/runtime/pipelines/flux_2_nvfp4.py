# SPDX-License-Identifier: Apache-2.0
import glob
import os
from dataclasses import dataclass
from functools import lru_cache
from typing import Any, cast

from sglang.multimodal_gen.runtime.pipelines.flux_2 import Flux2Pipeline
from sglang.multimodal_gen.runtime.server_args import ServerArgs
from sglang.multimodal_gen.runtime.utils.hf_diffusers_utils import (
    maybe_download_model,
    verify_model_config_and_directory,
)
from sglang.multimodal_gen.runtime.utils.logging_utils import init_logger

logger = init_logger(__name__)


@dataclass(frozen=True)
class Flux2Nvfp4ModelResolution:
    base_model_name: str
    base_model_path: str
    transformer_weights_path: str


_FLUX2_BASE_MODEL = "black-forest-labs/FLUX.2-dev"


@lru_cache(maxsize=1)
def _resolve_flux2_base_model_path() -> str:
    return maybe_download_model(_FLUX2_BASE_MODEL, force_diffusers_model=True)


def _find_mixed_safetensors(local_dir: str) -> str | None:
    mixed_files = sorted(glob.glob(os.path.join(local_dir, "*-mixed.safetensors")))
    return mixed_files[0] if mixed_files else None


def _resolve_nvfp4_transformer_weights_path(
    server_args: ServerArgs, model_path: str
) -> str:
    if server_args.transformer_weights_path is not None:
        return server_args.transformer_weights_path

    local_nvfp4_path = maybe_download_model(model_path)
    mixed_file = _find_mixed_safetensors(local_nvfp4_path)
    if mixed_file is not None:
        logger.info("Using mixed-precision NVFP4 weights: %s", mixed_file)
        return mixed_file

    logger.warning(
        "No *-mixed.safetensors found in %s; falling back to full directory",
        local_nvfp4_path,
    )
    return local_nvfp4_path


def resolve_flux2_nvfp4_model(
    server_args: ServerArgs, model_path: str
) -> Flux2Nvfp4ModelResolution:
    transformer_weights_path = _resolve_nvfp4_transformer_weights_path(
        server_args, model_path
    )
    return Flux2Nvfp4ModelResolution(
        base_model_name=_FLUX2_BASE_MODEL,
        base_model_path=_resolve_flux2_base_model_path(),
        transformer_weights_path=transformer_weights_path,
    )


class Flux2NvfpPipeline(Flux2Pipeline):
    pipeline_name = "Flux2NvfpPipeline"
    _model_resolution: Flux2Nvfp4ModelResolution | None = None

    def _get_model_resolution(
        self, server_args: ServerArgs | None = None
    ) -> Flux2Nvfp4ModelResolution:
        if self._model_resolution is None:
            if server_args is None:
                raise ValueError(
                    "server_args is required to resolve FLUX.2 NVFP4 paths"
                )
            self._model_resolution = resolve_flux2_nvfp4_model(
                server_args, self.model_path
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
        self, server_args: ServerArgs, module_name: str, load_module_name: str
    ) -> str:
        override_path = server_args.component_paths.get(module_name)
        if override_path is not None:
            return maybe_download_model(override_path)

        # get non-transformer components from the base FLUX.2 repo explicitly.
        # e.g.:
        #   transformer weights: ...FLUX.2-dev-NVFP4/.../flux2-dev-nvfp4-mixed.safetensors
        #   text_encoder path:  ...FLUX.2-dev/.../text_encoder
        component_model_path = os.path.join(
            self._get_model_resolution(server_args).base_model_path, load_module_name
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
        logger.info(
            "NVFP4 transformer weights: %s",
            model_resolution.transformer_weights_path,
        )
        return super().load_modules(server_args, loaded_modules)


EntryClass = Flux2NvfpPipeline
