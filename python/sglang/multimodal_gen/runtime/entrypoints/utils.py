# Copied and adapted from: https://github.com/hao-ai-lab/FastVideo

# SPDX-License-Identifier: Apache-2.0
"""
DiffGenerator module for sglang-diffusion.

This module provides a consolidated interface for generating videos using
diffusion models.
"""

import dataclasses

from sglang.multimodal_gen.configs.sample.sampling_params import SamplingParams
from sglang.multimodal_gen.runtime.pipelines_core.schedule_batch import Req
from sglang.multimodal_gen.runtime.server_args import ServerArgs
from sglang.multimodal_gen.runtime.utils.logging_utils import init_logger
from sglang.multimodal_gen.utils import shallow_asdict

logger = init_logger(__name__)


def prepare_request(
    server_args: ServerArgs,
    sampling_params: SamplingParams,
) -> Req:
    params_dict = shallow_asdict(sampling_params)

    diffusers_kwargs = params_dict.pop("diffusers_kwargs", None)
    extra = params_dict.get("extra") or {}
    if diffusers_kwargs:
        extra["diffusers_kwargs"] = diffusers_kwargs
        params_dict["extra"] = extra

    req = Req(
        **params_dict,
        VSA_sparsity=server_args.VSA_sparsity,
    )

    req.adjust_size(server_args)

    if (req.width is not None and req.width <= 0) or (
        req.height is not None and req.height <= 0
    ):
        raise ValueError(
            f"Height and width must be positive, got height={req.height}, width={req.width}"
        )

    return req
