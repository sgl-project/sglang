# SPDX-License-Identifier: Apache-2.0

"""Helpers for FLUX.2 quantized checkpoint path resolution.

This module keeps FLUX.2 quant-specific repo and local-path resolution out of
the pipeline class so the pipeline can stay focused on loading modules.
"""

import glob
import os
from dataclasses import dataclass
from functools import lru_cache

from sglang.multimodal_gen.runtime.server_args import ServerArgs
from sglang.multimodal_gen.runtime.utils.hf_diffusers_utils import maybe_download_model
from sglang.multimodal_gen.runtime.utils.logging_utils import init_logger

logger = init_logger(__name__)

_FLUX2_BASE_MODEL = "black-forest-labs/FLUX.2-dev"


@dataclass(frozen=True)
class Flux2Nvfp4ModelResolution:
    base_model_name: str
    base_model_path: str
    transformer_weights_path: str


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


def _find_mixed_safetensors(local_dir: str) -> str | None:
    mixed_files = sorted(glob.glob(os.path.join(local_dir, "*-mixed.safetensors")))
    return mixed_files[0] if mixed_files else None


@lru_cache(maxsize=1)
def _resolve_flux2_base_model_path() -> str:
    return maybe_download_model(_FLUX2_BASE_MODEL, force_diffusers_model=True)
