from __future__ import annotations

from typing import Any, Optional

from sglang.srt.distributed.parallel_state import in_the_same_node_as
from sglang.srt.mem_cache.shared_kv import SharedHostTensorAllocator
from sglang.srt.utils import is_hip


def should_enable_dsa_cp_shared_kvcache(
    *,
    enable_hisparse: bool,
    enabled: bool,
    is_hip_platform: Optional[bool] = None,
) -> bool:
    if not enabled:
        return False
    if enable_hisparse:
        return False
    if is_hip_platform is None:
        is_hip_platform = is_hip()
    return not is_hip_platform


class DsaCpSharedHostTensorAllocator(SharedHostTensorAllocator):
    def __init__(self, cpu_group, owner_rank: int, kind: str):
        super().__init__(
            cpu_group,
            owner_rank=owner_rank,
            kind=kind,
            log_label="DSA CP shared L2 host memory",
        )


def maybe_create_dsa_cp_shared_l2_allocator(
    *,
    params: Any,
    server_args: Any,
    kv_pool: Any,
    kind: str,
):
    if not (
        getattr(server_args, "enable_dsa_cp_shared_kv_cache", False)
        and getattr(kv_pool, "enable_cp_shared_kvcache", False)
    ):
        return None

    cp_group = params.attn_cp_cache_group or params.tp_cache_group
    if cp_group is None:
        raise RuntimeError("DSA CP shared L2 requires a CP cache group.")
    if not all(in_the_same_node_as(cp_group, source_rank=0)):
        raise RuntimeError(
            "DSA CP shared L2 requires all CP ranks to be on the same node."
        )
    return DsaCpSharedHostTensorAllocator(
        cpu_group=cp_group,
        owner_rank=0,
        kind=kind,
    )
