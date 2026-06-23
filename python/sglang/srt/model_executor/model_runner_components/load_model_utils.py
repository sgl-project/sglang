from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from sglang.srt.configs.model_config import ModelConfig
    from sglang.srt.server_args import ServerArgs

logger = logging.getLogger(__name__)


def maybe_downgrade_dtype_for_legacy_gpu(
    *, server_args: ServerArgs, model_config: ModelConfig
) -> None:
    if torch.cuda.get_device_capability()[0] < 8:
        logger.info(
            "Compute capability below sm80. Use float16 due to lack of bfloat16 support."
        )
        from sglang.srt.arg_groups.overrides import declare_load_time_override

        declare_load_time_override(
            "ModelRunner._sm80_dtype_fallback", {"dtype": "float16"}
        )
        model_config.dtype = torch.float16
        if torch.cuda.get_device_capability()[1] < 5:
            raise RuntimeError("SGLang only supports sm75 and above.")
