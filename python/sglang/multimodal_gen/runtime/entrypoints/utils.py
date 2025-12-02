# Copied and adapted from: https://github.com/hao-ai-lab/FastVideo

# SPDX-License-Identifier: Apache-2.0
"""
DiffGenerator module for sglang-diffusion.

This module provides a consolidated interface for generating videos using
diffusion models.
"""


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
    """
    Settle SamplingParams according to ServerArgs

    """
    # Create a copy of inference args to avoid modifying the original
    req = Req(
        **shallow_asdict(sampling_params),
        VSA_sparsity=server_args.VSA_sparsity,
    )
    req.adjust_size(server_args)

    if req.width <= 0 or req.height <= 0:
        raise ValueError(
            f"Height, width must be positive integers, got "
            f"height={req.height}, width={req.width}"
        )

    return req
