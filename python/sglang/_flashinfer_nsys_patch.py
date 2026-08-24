"""Profiling-only FlashInfer adjustments for very high-overhead NSYS runs."""

from __future__ import annotations

import logging
import os
import sys
from typing import Any, Callable


_LOGGER = logging.getLogger(__name__)
_ENV_NAME = "SGLANG_NSYS_DISABLE_FLASHINFER_MOE_A2A_TIMEOUT"
_PATCHED_SPEC_NAME = "mnnvl_moe_alltoall_nsys_no_timeout"


def _enabled() -> bool:
    return os.getenv(_ENV_NAME, "0").strip().lower() in {"1", "true", "yes"}


def _make_no_timeout_spec(
    original_factory: Callable[[], Any],
) -> Any:
    spec = original_factory()
    spec.name = _PATCHED_SPEC_NAME
    flags = list(spec.extra_cuda_cflags or [])
    if "-DDISABLE_TIMEOUT=1" not in flags:
        flags.append("-DDISABLE_TIMEOUT=1")
    spec.extra_cuda_cflags = flags
    return spec


def apply_flashinfer_nsys_patch() -> None:
    """Force only FlashInfer MoE A2A through a no-timeout JIT build.

    NSYS CUDA Graph node tracing can hold one rank inside a graph replay for
    several minutes.  FlashInfer's production MoE A2A kernel deliberately
    traps after a 300-second peer wait, which aborts an otherwise valid trace.
    This opt-in changes only that profiling guard: the communication algorithm,
    data path, and successful execution path are identical.
    """

    if not _enabled():
        return

    from flashinfer import jit as flashinfer_jit
    from flashinfer.jit import comm as flashinfer_jit_comm

    original_factory = flashinfer_jit_comm.gen_moe_alltoall_module

    def no_timeout_factory() -> Any:
        return _make_no_timeout_spec(original_factory)

    flashinfer_jit_comm.gen_moe_alltoall_module = no_timeout_factory
    flashinfer_jit.gen_moe_alltoall_module = no_timeout_factory

    loaded_a2a = sys.modules.get("flashinfer.comm.trtllm_moe_alltoall")
    if loaded_a2a is not None:
        get_module = loaded_a2a.get_moe_alltoall_module
        if get_module.cache_info().currsize:
            raise RuntimeError(
                f"{_ENV_NAME} was enabled after the FlashInfer MoE A2A module "
                "had already been loaded"
            )
        loaded_a2a.gen_moe_alltoall_module = no_timeout_factory

    _LOGGER.warning(
        "NSYS profiling override enabled: FlashInfer MoE A2A will JIT as %s "
        "with DISABLE_TIMEOUT=1",
        _PATCHED_SPEC_NAME,
    )

