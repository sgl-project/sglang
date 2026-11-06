"""The communicator of Kimi-K3's gfx950 TP8 LL collectives: the column-parallel
latent-MoE tail's gather fused with the next attention read.

``get_comm()`` is a collective over the TP group on its first call, so the
model makes it at construction, where every rank builds the same layers in the
same order and nothing is captured.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional

import torch

from sglang.kernels.jit.utils import cache_once
from sglang.srt.utils import is_gfx95_supported

if TYPE_CHECKING:
    from sglang.kernels.ops.communication.k3_ll_comm_hip import LLComm


@cache_once
def get_comm() -> Optional[LLComm]:
    """None off gfx950 TP8, or under --disable-custom-all-reduce (the scratch is a
    dedicated aiter CustomAllreduce)."""
    from sglang.srt.runtime_context import get_exec, get_parallel

    if not is_gfx95_supported() or get_exec().comm.disable_custom_all_reduce:
        return None
    from sglang.kernels.ops.communication.k3_ll_comm_hip import (
        WORLD_SIZE,
        create_ll_comm,
    )

    parallel = get_parallel()
    if parallel.tp_size != WORLD_SIZE or parallel.attn_tp_size != parallel.tp_size:
        return None
    return create_ll_comm(
        parallel.tp_group.cpu_group,
        torch.device("cuda", torch.cuda.current_device()),
    )
