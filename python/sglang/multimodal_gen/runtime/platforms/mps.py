# Copied and adapted from: https://github.com/hao-ai-lab/FastVideo
from functools import lru_cache

import torch

from sglang.multimodal_gen.runtime.platforms.interface import (
    AttentionBackendEnum,
    MMPlatform,
)
from sglang.multimodal_gen.runtime.utils.logging_utils import init_logger
from sglang.srt.platforms.mps import MpsDeviceMixin

# SPDX-License-Identifier: Apache-2.0


logger = init_logger(__name__)


class MpsPlatform(MpsDeviceMixin, MMPlatform):
    dispatch_key: str = "MPS"
    device_control_env_var: str = "MPS_VISIBLE_DEVICES"

    @classmethod
    @lru_cache(maxsize=1)
    def is_amp_supported(cls) -> bool:
        return False

    @classmethod
    @lru_cache(maxsize=1)
    def is_float64_supported(cls) -> bool:
        return False

    @classmethod
    def is_async_output_supported(cls, enforce_eager: bool | None) -> bool:
        if enforce_eager:
            logger.warning(
                "To see benefits of async output processing, enable MPS "
                "graph. Since, enforce-eager is enabled, async output "
                "processor cannot be used"
            )
            return False
        return True

    @classmethod
    def get_attn_backend_cls_str(
        cls,
        selected_backend: AttentionBackendEnum | None,
        head_size: int,
        dtype: torch.dtype,
    ) -> str:
        # MPS supports SDPA (Scaled Dot-Product Attention) which is the most compatible
        logger.info("Using Torch SDPA backend for MPS.")
        return (
            "sglang.multimodal_gen.runtime.layers.attention.backends.sdpa.SDPABackend"
        )

    def get_all_to_all_communicator_class(self) -> type:
        from sglang.multimodal_gen.runtime.distributed.device_communicators.cpu_communicator import (
            CpuCommunicator,
        )

        return CpuCommunicator
