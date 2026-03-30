# SPDX-License-Identifier: Apache-2.0

import glob
import os
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

_FLUX2_BASE_MODEL = "black-forest-labs/FLUX.2-dev"


def _find_mixed_safetensors(local_dir: str) -> str | None:
    """Return the path to the *-mixed.safetensors file in a directory, or None."""
    mixed_files = sorted(glob.glob(os.path.join(local_dir, "*-mixed.safetensors")))
    return mixed_files[0] if mixed_files else None


@lru_cache(maxsize=1)
def _resolve_flux2_base_model_path() -> str:
    # The NVFP4 repo only provides the quantized transformer weights.
    # We still load model_index.json and the non-transformer components from the base repo.
    return maybe_download_model(_FLUX2_BASE_MODEL, force_diffusers_model=True)


class Flux2NvfpPipeline(Flux2Pipeline):
    pipeline_name = "Flux2NvfpPipeline"
    _base_model_path: str | None = None

    def _get_base_model_path(self) -> str:
        if self._base_model_path is None:
            self._base_model_path = _resolve_flux2_base_model_path()
        return self._base_model_path

    def _load_config(self) -> dict[str, Any]:
        base_model_path = self._get_base_model_path()
        logger.info("Model path: %s", self.model_path)
        logger.info(
            "Using base model '%s' at %s for config and non-transformer components",
            _FLUX2_BASE_MODEL,
            base_model_path,
        )
        config = verify_model_config_and_directory(base_model_path)
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
            self._get_base_model_path(), load_module_name
        )
        logger.debug("Resolved component path: %s", component_model_path)
        return component_model_path

    def load_modules(
        self,
        server_args: ServerArgs,
        loaded_modules: dict | None = None,
    ) -> dict:
        if server_args.transformer_weights_path is None:
            local_nvfp4_path = maybe_download_model(self.model_path)
            mixed_file = _find_mixed_safetensors(local_nvfp4_path)
            if mixed_file:
                logger.info("Using mixed-precision NVFP4 weights: %s", mixed_file)
                server_args.transformer_weights_path = mixed_file
            else:
                logger.warning(
                    "No *-mixed.safetensors found in %s; falling back to full directory",
                    local_nvfp4_path,
                )
                server_args.transformer_weights_path = local_nvfp4_path

        logger.info(
            "NVFP4 transformer weights: %s", server_args.transformer_weights_path
        )
        return super().load_modules(server_args, loaded_modules)


EntryClass = Flux2NvfpPipeline
