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
    CUTEDSL = "cutedsl"
    FLASHINFER = "flashinfer"
    CUSTOM = "custom"

    @classmethod
    def _missing_(cls, value):
        return cls.CUSTOM

    def is_triton(self):
        return self == LinearAttnKernelBackend.TRITON

    def is_cutedsl(self):
        return self == LinearAttnKernelBackend.CUTEDSL

    def is_flashinfer(self):
        return self == LinearAttnKernelBackend.FLASHINFER

    def is_custom(self):
        return self == LinearAttnKernelBackend.CUSTOM


LINEAR_ATTN_DECODE_BACKEND: Optional[LinearAttnKernelBackend] = None
LINEAR_ATTN_PREFILL_BACKEND: Optional[LinearAttnKernelBackend] = None


def initialize_linear_attn_config(server_args: ServerArgs):
    global LINEAR_ATTN_DECODE_BACKEND
    global LINEAR_ATTN_PREFILL_BACKEND

    base = server_args.linear_attn_backend
    decode = server_args.linear_attn_decode_backend or base
    prefill = server_args.linear_attn_prefill_backend or base

    LINEAR_ATTN_DECODE_BACKEND = LinearAttnKernelBackend(decode)
    LINEAR_ATTN_PREFILL_BACKEND = LinearAttnKernelBackend(prefill)

    rank0_log(f"Linear attention kernel backend: decode={decode}, prefill={prefill}")


def gdn_chunk_tree_verify_enabled(server_args: ServerArgs) -> bool:
    """Whether GDN target verify uses the block-quadratic tree kernel.

    Single source of truth shared by GDNAttnBackend (forward/commit paths) and
    MambaPool (intermediate SSM cache sizing).
    """
    from sglang.srt.layers.attention.fla.chunk_tree_verify import TREE_CHUNK_SIZE
    from sglang.srt.utils import is_cuda

    num_draft_tokens = server_args.speculative_num_draft_tokens or 0
    uses_tree_spec_inputs = (
        server_args.speculative_eagle_topk or 0
    ) > 1 or gdn_dflash_tfm_tree_verify_enabled(server_args)
    fused = gdn_fused_tree_verify_enabled(server_args)
    return (
        server_args.speculative_gdn_verify_kernel == "chunk"
        and is_cuda()
        and uses_tree_spec_inputs
        and 0 < num_draft_tokens
        and (fused or num_draft_tokens <= TREE_CHUNK_SIZE)
    )


def gdn_fused_tree_verify_enabled(server_args: ServerArgs) -> bool:
    """Whether chunk-mode GDN verify should use the fused tree kernel.

    The block-quadratic chunk kernel is capped by TREE_CHUNK_SIZE. For larger
    DFLASH_TFM tree budgets, fused tree verify is the chunk-mode implementation
    and avoids a silent recurrent fallback.
    """
    from sglang.srt.layers.attention.fla.chunk_tree_verify import TREE_CHUNK_SIZE

    num_draft_tokens = server_args.speculative_num_draft_tokens or 0
    return (
        gdn_dflash_tfm_tree_verify_enabled(server_args)
        and num_draft_tokens > TREE_CHUNK_SIZE
    )


def gdn_dflash_tfm_tree_verify_enabled(server_args: ServerArgs) -> bool:
    if server_args.speculative_algorithm != "DFLASH_TFM":
        return False
    num_draft_tokens = server_args.speculative_num_draft_tokens or 0
    block_size = server_args.speculative_dflash_block_size or 0
    return block_size > 0 and num_draft_tokens > block_size


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
