from __future__ import annotations

import functools
import os

from sglang.srt.environ import envs
from sglang.srt.utils import is_hip


@functools.lru_cache(maxsize=1)
def is_unified_kv_triton() -> bool:
    # unified_kv_triton is only implemented on HIP (ROCm)
    return is_hip() and envs.SGLANG_HACK_FLASHMLA_BACKEND.get() == "unified_kv_triton"


@functools.lru_cache(maxsize=1)
def is_dcp_physical() -> bool:
    """Opt-in physical DCP sharding of the unified_kv pool.

    When enabled (and dcp_size>1), each DCP rank physically stores only its
    1/dcp shard of the unified_kv rows (owner = unified_slot % dcp_size, local
    row = unified_slot // dcp_size); the SWA/compressed WRITE locations are
    remapped to the local shard and decode reads only owned rows then merges via
    cp_lse_ag_out_rs. Default off -> read-only DCP / baseline byte-for-byte
    unchanged. HIP + unified_kv only.
    """
    return is_unified_kv_triton() and os.environ.get(
        "SGLANG_DSV4_DCP_PHYSICAL", "0"
    ) not in ("0", "", "false", "False")
