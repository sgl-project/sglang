"""Load every Triton module a captured step can launch before serving.

A graph replay is prepared out of graph: the attention backend's replay metadata prep
launches Triton kernels eagerly (for example DSA metadata kernels, which compile per
power-of-2 ``BLOCK_BS`` / ``BLOCK_EXPANDED`` tile bucket even when batch size ``bs`` is
a runtime argument). Capture does not run that replay prep, nor does it run sampled
post-verify graphs until the first sampled request arrives. Thus the first launch of a
replay or sampled kernel can happen mid-serving, invoking ``cuModuleLoadData``. Loading
a CUDA module can synchronize/block on the device; if cross-GPU spin-wait collectives
(e.g., custom all-reduce / symmetric-memory multimem all-gather) are already in flight
on peer GPUs waiting for this rank while this rank blocks in ``cuModuleLoadData``, the
ranks can deadlock until the scheduler watchdog fires.

* ``warm_replay_prep`` (``SGLANG_WARM_REPLAY_PREP``): right after a shape is captured,
  run that shape's replay prep once on the capture batch, with the device idle. The
  prep only writes the shape's static metadata buffers, which every replay rewrites
  before it runs.
* ``arm_lazy_load_guard`` (``SGLANG_TRITON_LAZY_LOAD_SYNC``): once the graphs are
  captured, a Triton module that still loads synchronizes the device first (outside a
  stream capture; a capture that may load one synchronizes before it begins, see
  ``sync_before_lazy_capture``) and is logged once per kernel.
"""

from __future__ import annotations

import logging
from typing import Callable, Optional

import torch

from sglang.srt.environ import envs

logger = logging.getLogger(__name__)

_GUARD_ARMED = False
_GUARD_LOGGED: set = set()


def warm_replay_prep(what: str, bs: int, prep: Callable[[], None]) -> bool:
    """Run ``prep`` (a captured shape's out-of-graph replay prep) once now, between two
    device synchronizations; a failure only logs (the kernels then load at first use)."""
    if not envs.SGLANG_WARM_REPLAY_PREP.get():
        return False
    try:
        torch.cuda.synchronize()
        prep()
        torch.cuda.synchronize()
    except Exception as e:  # noqa: BLE001 - best effort; replay stays correct
        logger.warning(
            "Replay prep warm-up of the %s graph at batch size %d failed (%r); its "
            "kernels load at their first replay",
            what,
            bs,
            e,
        )
        return False
    return True


def log_warmed(what: str, sizes) -> None:
    if sizes:
        logger.info(
            "Warmed replay prep of the %s graphs at batch sizes %s (SGLANG_WARM_REPLAY_PREP)",
            what,
            sorted(sizes),
        )


def _install_guard() -> bool:
    try:
        from triton.compiler import compiler as triton_compiler
    except Exception:  # noqa: BLE001 - no Triton, nothing to guard
        return False
    kernel_cls = getattr(triton_compiler, "CompiledKernel", None)
    if kernel_cls is None or not hasattr(kernel_cls, "_init_handles"):
        return False
    if getattr(kernel_cls, "_sglang_lazy_load_guard", False):
        return True
    original = kernel_cls._init_handles

    def _init_handles(self):
        if not _GUARD_ARMED or getattr(self, "module", None) is not None:
            return original(self)
        capturing = torch.cuda.is_current_stream_capturing()
        if not capturing:
            torch.cuda.synchronize()
        name = getattr(self, "name", None) or "?"
        if name not in _GUARD_LOGGED:
            _GUARD_LOGGED.add(name)
            logger.info(
                "Triton module %s first loaded after graph capture (%s)",
                name,
                "inside a stream capture" if capturing else "device synchronized first",
            )
        return original(self)

    kernel_cls._init_handles = _init_handles
    kernel_cls._sglang_lazy_load_guard = True
    return True


def arm_lazy_load_guard() -> None:
    """From now on a Triton module load first synchronizes the device (and is logged)."""
    global _GUARD_ARMED
    if _GUARD_ARMED or not envs.SGLANG_TRITON_LAZY_LOAD_SYNC.get():
        return
    if _install_guard():
        _GUARD_ARMED = True


def sync_before_lazy_capture(device: Optional[torch.device] = None) -> None:
    """Before a capture that may load modules while serving (a lazily captured graph):
    let the device go idle, so a load inside the capture cannot wait on in-flight
    spin-wait kernels."""
    if _GUARD_ARMED and not torch.cuda.is_current_stream_capturing():
        torch.cuda.synchronize(device)
