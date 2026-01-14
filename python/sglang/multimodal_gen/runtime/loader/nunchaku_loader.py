# SPDX-License-Identifier: Apache-2.0
from sglang.multimodal_gen.runtime.layers.quantization.nunchaku_config import (
    NunchakuConfig,
)
from sglang.multimodal_gen.runtime.utils.logging_utils import init_logger

logger = init_logger(__name__)


def create_nunchaku_config_from_server_args(server_args) -> NunchakuConfig:
    """
    Create NunchakuConfig from server arguments.

    Args:
        server_args: Server configuration arguments

    Returns:
        NunchakuConfig instance
    """
    return NunchakuConfig(
        precision=server_args.nunchaku_config.quantization_precision,
        rank=server_args.nunchaku_config.quantization_rank,
        act_unsigned=server_args.nunchaku_config.quantization_act_unsigned,
        quantized_model_path=server_args.nunchaku_config.quantized_model_path,
    )
