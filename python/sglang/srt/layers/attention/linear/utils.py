from __future__ import annotations

import logging
from enum import Enum
from typing import TYPE_CHECKING, Optional

from sglang.srt.utils.common import rank0_log

if TYPE_CHECKING:
    from sglang.srt.server_args import ServerArgs

logger = logging.getLogger(__name__)


class LinearAttnKernelBackend(Enum):
    TRITON = "triton"
    HIP = "hip"
    CUTEDSL = "cutedsl"
    FLASHINFER = "flashinfer"
    FLASHKDA = "flashkda"
    CUSTOM = "custom"

    @classmethod
    def _missing_(cls, value):
        return cls.CUSTOM

    def is_triton(self):
        return self == LinearAttnKernelBackend.TRITON

    def is_hip(self):
        return self == LinearAttnKernelBackend.HIP

    def is_cutedsl(self):
        return self == LinearAttnKernelBackend.CUTEDSL

    def is_flashinfer(self):
        return self == LinearAttnKernelBackend.FLASHINFER

    def is_flashkda(self):
        return self == LinearAttnKernelBackend.FLASHKDA

    def is_custom(self):
        return self == LinearAttnKernelBackend.CUSTOM


LINEAR_ATTN_DECODE_BACKEND: Optional[LinearAttnKernelBackend] = None
LINEAR_ATTN_PREFILL_BACKEND: Optional[LinearAttnKernelBackend] = None


def initialize_linear_attn_config(server_args: ServerArgs, *, is_gdn: bool = False):
    global LINEAR_ATTN_DECODE_BACKEND
    global LINEAR_ATTN_PREFILL_BACKEND

    base = server_args.linear_attn_backend
    decode = server_args.linear_attn_decode_backend or base
    prefill = server_args.linear_attn_prefill_backend or base

    if base == "hip":
        raise ValueError(
            "--linear-attn-backend hip is not supported: HIP is GDN decode-only. "
            "Use --linear-attn-decode-backend hip instead."
        )
    if prefill == "hip":
        raise ValueError(
            "--linear-attn-prefill-backend hip is not supported: HIP is "
            "GDN decode-only."
        )
    if decode == "hip" and not is_gdn:
        raise ValueError(
            "--linear-attn-decode-backend hip is supported only for GDN models."
        )

    LINEAR_ATTN_DECODE_BACKEND = LinearAttnKernelBackend(decode)
    LINEAR_ATTN_PREFILL_BACKEND = LinearAttnKernelBackend(prefill)

    rank0_log(f"Linear attention kernel backend: decode={decode}, prefill={prefill}")


def get_linear_attn_decode_backend() -> LinearAttnKernelBackend:
    global LINEAR_ATTN_DECODE_BACKEND
    if LINEAR_ATTN_DECODE_BACKEND is None:
        logger.warning(
            "LINEAR_ATTN_DECODE_BACKEND is not initialized, using triton backend"
        )
        LINEAR_ATTN_DECODE_BACKEND = LinearAttnKernelBackend.TRITON
    return LINEAR_ATTN_DECODE_BACKEND


def get_linear_attn_prefill_backend() -> LinearAttnKernelBackend:
    global LINEAR_ATTN_PREFILL_BACKEND
    if LINEAR_ATTN_PREFILL_BACKEND is None:
        logger.warning(
            "LINEAR_ATTN_PREFILL_BACKEND is not initialized, using triton backend"
        )
        LINEAR_ATTN_PREFILL_BACKEND = LinearAttnKernelBackend.TRITON
    return LINEAR_ATTN_PREFILL_BACKEND
