"""Single import gate for the `sgl_kernel.kvcacheio` KV-transfer ops.

CUDA/ROCm get these ops from the in-tree sgl-kernel; XPU gets them from the
out-of-tree sgl-kernel-xpu wheel. pool_host sits on the scheduler import path, so
a build without the ops must fail at first offload rather than at
`import scheduler` -- hence the stubs. CUDA/ROCm still re-raise the ImportError at
import time, since an sgl-kernel that cannot supply them is broken rather than
merely unbuilt.

Every symbol below is defined on every device type, so a pool that reaches one of
these ops where none of them exist gets a RuntimeError naming the op and why,
rather than a NameError raised inside a write-back thread. `unavailable_reason` is
None exactly when the real ops are bound, so a pool that cannot work without them
can read it and refuse at construction instead.

`unavailable_stub` is exported for one op that is not in `sgl_kernel.kvcacheio`
and so cannot be gated here: memory_pool_host takes `transfer_cache_dsv4_mla`
from the `sgl_kernel` top level on XPU, and keeps that import next to
DeepSeekV4PagedHostPool, its only caller, rather than coupling every pool_host
module to the HiSparse ops module the other devices build it from.
"""

from __future__ import annotations

from sglang.srt.utils import is_cuda, is_hip, is_xpu

_is_cuda = is_cuda()
_is_hip = is_hip()
_is_xpu = is_xpu()


def unavailable_stub(name: str, reason: str):
    """A stand-in for one missing op, naming it so the traceback needs no lookup.

    Both io backends land here -- the *_direct* ops serve --hicache-io-backend
    direct -- so the message names neither, or it would send half the callers to
    change the one flag that cannot help.
    """

    def _raise(*args, **kwargs):
        raise RuntimeError(
            f"HiCache host<->device KV transfer needs {name}, which is not "
            f"available here: {reason}"
        )

    return _raise


unavailable_reason: str | None = None
if _is_cuda or _is_hip or _is_xpu:
    try:
        from sgl_kernel.kvcacheio import (
            transfer_kv_all_layer,
            transfer_kv_all_layer_direct_lf_pf,
            transfer_kv_all_layer_lf_pf,
            transfer_kv_all_layer_lf_ph,
            transfer_kv_all_layer_mla,
            transfer_kv_all_layer_mla_lf_pf,
            transfer_kv_direct,
            transfer_kv_per_layer,
            transfer_kv_per_layer_direct_pf_lf,
            transfer_kv_per_layer_mla,
            transfer_kv_per_layer_mla_pf_lf,
            transfer_kv_per_layer_pf_lf,
            transfer_kv_per_layer_ph_lf,
        )
    except ImportError as e:
        # Not `not _is_xpu`: XPU next to a CUDA/ROCm GPU is still a broken build.
        if _is_cuda or _is_hip:
            raise
        unavailable_reason = (
            f"sgl_kernel.kvcacheio failed to import ({e}); XPU takes it from the "
            "out-of-tree sgl-kernel-xpu wheel"
        )
else:
    unavailable_reason = (
        "sgl_kernel.kvcacheio ships only in the CUDA, ROCm and XPU sgl-kernel builds"
    )

if unavailable_reason is not None:
    transfer_kv_all_layer = unavailable_stub(
        "transfer_kv_all_layer", unavailable_reason
    )
    transfer_kv_all_layer_direct_lf_pf = unavailable_stub(
        "transfer_kv_all_layer_direct_lf_pf", unavailable_reason
    )
    transfer_kv_all_layer_lf_pf = unavailable_stub(
        "transfer_kv_all_layer_lf_pf", unavailable_reason
    )
    transfer_kv_all_layer_lf_ph = unavailable_stub(
        "transfer_kv_all_layer_lf_ph", unavailable_reason
    )
    transfer_kv_all_layer_mla = unavailable_stub(
        "transfer_kv_all_layer_mla", unavailable_reason
    )
    transfer_kv_all_layer_mla_lf_pf = unavailable_stub(
        "transfer_kv_all_layer_mla_lf_pf", unavailable_reason
    )
    transfer_kv_direct = unavailable_stub("transfer_kv_direct", unavailable_reason)
    transfer_kv_per_layer = unavailable_stub(
        "transfer_kv_per_layer", unavailable_reason
    )
    transfer_kv_per_layer_direct_pf_lf = unavailable_stub(
        "transfer_kv_per_layer_direct_pf_lf", unavailable_reason
    )
    transfer_kv_per_layer_mla = unavailable_stub(
        "transfer_kv_per_layer_mla", unavailable_reason
    )
    transfer_kv_per_layer_mla_pf_lf = unavailable_stub(
        "transfer_kv_per_layer_mla_pf_lf", unavailable_reason
    )
    transfer_kv_per_layer_pf_lf = unavailable_stub(
        "transfer_kv_per_layer_pf_lf", unavailable_reason
    )
    transfer_kv_per_layer_ph_lf = unavailable_stub(
        "transfer_kv_per_layer_ph_lf", unavailable_reason
    )
