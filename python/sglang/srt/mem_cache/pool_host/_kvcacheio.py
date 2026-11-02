"""`sgl_kernel.kvcacheio` KV-transfer ops for XPU, from the sgl-kernel-xpu wheel.

CUDA/ROCm keep importing `sgl_kernel.kvcacheio` directly in each pool module;
only XPU imports the ops from here. The XPU wheel can lag the CUDA op set, so
each op is bound on its own: a missing one becomes a stub naming it, is recorded
in `missing_ops`, and host pools refuse at construction via
`require_kvcacheio_ops`. `unavailable_reason` is set when the module itself
does not import. Off XPU nothing is bound and `missing_ops` stays empty.
`hisparse_memory_pool.py` still imports `transfer_kv_all_layer_mla` directly,
so on XPU a wheel without that op or without the module fails at import.
"""

from __future__ import annotations

from collections.abc import Mapping
from types import MappingProxyType

from sglang.srt.utils import is_xpu

_is_xpu = is_xpu()


def unavailable_stub(name: str, reason: str):
    """A stand-in for one missing op that names it when called."""

    def _raise(*args, **kwargs):
        raise RuntimeError(
            f"HiCache host<->device KV transfer needs {name}, which is not "
            f"available here: {reason}"
        )

    return _raise


# The table shared by the pools that move one latent buffer per layer
# (MLA, DSA indexer, DeepSeek-V4). Read-only, since four classes share it.
MLA_KVCACHEIO_OPS = MappingProxyType(
    {
        "kernel": MappingProxyType(
            {
                "layer_first": (
                    "transfer_kv_per_layer_mla",
                    "transfer_kv_all_layer_mla",
                ),
                "page_first": (
                    "transfer_kv_per_layer_mla_pf_lf",
                    "transfer_kv_all_layer_mla_lf_pf",
                ),
            }
        ),
        "direct": MappingProxyType(
            {
                "layer_first": ("transfer_kv_direct",),
                "page_first_direct": (
                    "transfer_kv_per_layer_direct_pf_lf",
                    "transfer_kv_all_layer_direct_lf_pf",
                ),
            }
        ),
    }
)


def require_kvcacheio_ops(
    *, pool: str, layout: str, ops: Mapping[str, Mapping[str, tuple[str, ...]]]
) -> None:
    """Fail at host-pool construction if this build lacks an op the pool will call.

    `ops` is the pool's own `{io_backend: {layout: op names}}` table, declared
    next to the transfer methods that dispatch to those ops.
    """
    if not missing_ops:
        return
    # Only read when an op is missing, so pools stay constructible without
    # published server args wherever the build is complete.
    from sglang.srt.runtime_context import get_memory

    io_backend = get_memory().hicache_io_backend
    missing = [
        name for name in ops.get(io_backend, {}).get(layout, ()) if name in missing_ops
    ]
    if missing:
        if unavailable_reason is None:
            fix = "Pick another io backend or layout, or launch without HiCache"
        else:
            fix = "Launch without HiCache"
        raise ValueError(
            f"HiCache cannot build the {pool} host pool for --hicache-io-backend "
            f"{io_backend} --hicache-mem-layout {layout} here: "
            f"{missing_ops[missing[0]]}. {fix} (--enable-hierarchical-cache)."
        )


missing_ops: dict[str, str] = {}
unavailable_reason: str | None = None
if _is_xpu:
    try:
        import sgl_kernel.kvcacheio as _kvcacheio
    except ImportError as e:
        _kvcacheio = None
        unavailable_reason = (
            f"sgl_kernel.kvcacheio failed to import ({e}); XPU takes it from "
            "the out-of-tree sgl-kernel-xpu wheel"
        )

    def _bind(name: str):
        # getattr, not a from-import: the op set varies by sgl-kernel-xpu wheel,
        # and an absent op must stub only itself.
        op = None if _kvcacheio is None else getattr(_kvcacheio, name, None)
        if op is not None:
            return op
        reason = unavailable_reason or (
            f"this sgl-kernel-xpu build has no sgl_kernel.kvcacheio.{name}"
        )
        missing_ops[name] = reason
        return unavailable_stub(name, reason)

    transfer_kv_all_layer = _bind("transfer_kv_all_layer")
    transfer_kv_all_layer_direct_lf_pf = _bind("transfer_kv_all_layer_direct_lf_pf")
    transfer_kv_all_layer_lf_pf = _bind("transfer_kv_all_layer_lf_pf")
    transfer_kv_all_layer_lf_ph = _bind("transfer_kv_all_layer_lf_ph")
    transfer_kv_all_layer_mla = _bind("transfer_kv_all_layer_mla")
    transfer_kv_all_layer_mla_lf_pf = _bind("transfer_kv_all_layer_mla_lf_pf")
    transfer_kv_direct = _bind("transfer_kv_direct")
    transfer_kv_per_layer = _bind("transfer_kv_per_layer")
    transfer_kv_per_layer_direct_pf_lf = _bind("transfer_kv_per_layer_direct_pf_lf")
    transfer_kv_per_layer_mla = _bind("transfer_kv_per_layer_mla")
    transfer_kv_per_layer_mla_pf_lf = _bind("transfer_kv_per_layer_mla_pf_lf")
    transfer_kv_per_layer_pf_lf = _bind("transfer_kv_per_layer_pf_lf")
    transfer_kv_per_layer_ph_lf = _bind("transfer_kv_per_layer_ph_lf")
