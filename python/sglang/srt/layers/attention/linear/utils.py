from __future__ import annotations

import logging
from enum import Enum
from typing import TYPE_CHECKING, Optional

from sglang.srt.utils.common import rank0_log

if TYPE_CHECKING:
    from sglang.srt.server_args import ServerArgs

logger = logging.getLogger(
    __name__
)  # Used for warning in get_linear_attn_kernel_backend


class LinearAttnKernelBackend(Enum):
    AUTO = "auto"
    TRITON = "triton"
    CUTEDSL = "cutedsl"

    def is_auto(self):
        return self == LinearAttnKernelBackend.AUTO

    def is_triton(self):
        return self == LinearAttnKernelBackend.TRITON

    def is_cutedsl(self):
        return self == LinearAttnKernelBackend.CUTEDSL


LINEAR_ATTN_KERNEL_BACKEND: Optional[LinearAttnKernelBackend] = None


def initialize_linear_attn_config(server_args: ServerArgs):
    global LINEAR_ATTN_KERNEL_BACKEND
    LINEAR_ATTN_KERNEL_BACKEND = LinearAttnKernelBackend(
        server_args.linear_attn_kernel_backend
    )
    rank0_log(f"Linear attention kernel backend: {LINEAR_ATTN_KERNEL_BACKEND.value}")


def get_linear_attn_kernel_backend() -> LinearAttnKernelBackend:
    global LINEAR_ATTN_KERNEL_BACKEND
    if LINEAR_ATTN_KERNEL_BACKEND is None:
        logger.warning(
            "LINEAR_ATTN_KERNEL_BACKEND is not initialized, using triton backend"
        )
        LINEAR_ATTN_KERNEL_BACKEND = LinearAttnKernelBackend.TRITON
    return LINEAR_ATTN_KERNEL_BACKEND
