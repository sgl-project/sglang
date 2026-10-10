"""Single import gate for the `sgl_kernel.kvcacheio` KV-transfer ops.

CUDA/ROCm import all ops at once and re-raise, since there a missing op is a
broken build. Other builds that ship kvcacheio (XPU, via the out-of-tree
sgl-kernel-xpu wheel) can lag that op set, so each op is bound on its own, the
missing ones are recorded in `missing_ops`, and host pools refuse at
construction via `require_kvcacheio_ops`. Where no build ships kvcacheio, every
op is a stub that raises at first transfer. `unavailable_reason` is None exactly
when the module itself imported.
"""

from __future__ import annotations

from sglang.srt.utils import is_cuda, is_hip, is_xpu

_is_cuda = is_cuda()
_is_hip = is_hip()
_is_xpu = is_xpu()
_ships_kvcacheio = _is_cuda or _is_hip or _is_xpu


def unavailable_stub(name: str, reason: str):
    """A stand-in for one missing op that names it when called."""

    def _raise(*args, **kwargs):
        raise RuntimeError(
            f"HiCache host<->device KV transfer needs {name}, which is not "
            f"available here: {reason}"
        )

    return _raise


def require_kvcacheio_ops(
    *, pool: str, layout: str, ops: dict[str, dict[str, tuple[str, ...]]]
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
# Checked first: XPU next to a CUDA/ROCm GPU is still a broken build.
if _is_cuda or _is_hip:
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
else:
    _kvcacheio = None
    if _ships_kvcacheio:
        try:
            import sgl_kernel.kvcacheio as _kvcacheio
        except ImportError as e:
            unavailable_reason = (
                f"sgl_kernel.kvcacheio failed to import ({e}); XPU takes it from "
                "the out-of-tree sgl-kernel-xpu wheel"
            )
    else:
        unavailable_reason = (
            "sgl_kernel.kvcacheio ships only in the CUDA, ROCm and XPU sgl-kernel "
            "builds"
        )

    def _bind(name: str):
        # getattr, not a from-import: an absent op must stub only itself.
        op = None if _kvcacheio is None else getattr(_kvcacheio, name, None)
        if op is not None:
            return op
        reason = unavailable_reason or (
            f"this sgl-kernel build has no sgl_kernel.kvcacheio.{name}"
        )
        # Where no build ships kvcacheio, the stub is the contract, not a refusal.
        if _ships_kvcacheio:
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
