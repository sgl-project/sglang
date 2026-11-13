"""Memory guard for the MC2 scratch that CANN allocates on the first DeepEP
low-latency dispatch on Ascend A2, outside the torch_npu caching allocator.

The allocation lands inside decode graph capture warmup, after the KV pool is
carved, so it must fit in the pre-capture slack together with the rest of the
capture. Two opt-in levers: a pre-capture check that fails fast when free memory
is below the scratch plus an optional capture headroom, and a reservation of the
scratch alone out of the KV budget.
"""

from __future__ import annotations

import functools
import logging
import os

from sglang.srt.environ import envs
from sglang.srt.layers.moe.utils import get_moe_a2a_backend
from sglang.srt.platforms import current_platform
from sglang.srt.runtime_context import get_parallel, get_schedule
from sglang.srt.utils import get_available_gpu_memory, is_npu, log_info_on_rank0

logger = logging.getLogger(__name__)

_is_npu = is_npu()

_DEFAULT_HCCL_BUFFSIZE_MB = 200

# A2 SoC names are Ascend910B1/B2/B2C/B3/B4/B4_1; A3 is Ascend910_93xx, so the
# prefix needs the "B".
_A2_SOC_PREFIX = "Ascend910B"


def _resolve_deepep_hccl_buffsize() -> tuple[int, str]:
    # Vendor vars read raw; an empty string falls through like an unset one.
    for name in ("DEEPEP_HCCL_BUFFSIZE", "HCCL_BUFFSIZE"):
        value = os.environ.get(name)
        if value:
            return int(value), name
    return _DEFAULT_HCCL_BUFFSIZE_MB, "default"


def resolve_deepep_hccl_buffsize_mb() -> int:
    return _resolve_deepep_hccl_buffsize()[0]


def estimate_mc2_scratch_mb(*, hccl_buffsize_mb: int, margin_mb: int) -> int:
    # Single measured point on A2 (1536 MB buffer): the scratch was 2 x buffer
    # + 1 MB; the margin covers the +1 MB and is a conservative pick.
    return 2 * hccl_buffsize_mb + margin_mb


@functools.lru_cache(maxsize=1)
def _get_npu_soc_name() -> str:
    return current_platform.get_device_name()


def _mc2_scratch_applies() -> bool:
    # A2 only: A3 takes a different CANN MC2 path and has not been measured.
    return (
        _is_npu
        and get_moe_a2a_backend().is_deepep()
        and _get_npu_soc_name().startswith(_A2_SOC_PREFIX)
    )


def preflight_check_mc2_scratch(
    *, device: str, gpu_id: int, is_draft_worker: bool
) -> None:
    """Raise before decode capture if free memory cannot hold the MC2 scratch
    plus the capture headroom."""
    # The draft shares the target's communicator, whose scratch is already built.
    if (
        is_draft_worker
        or not envs.SGLANG_NPU_MC2_PREFLIGHT_CHECK.get()
        or not _mc2_scratch_applies()
    ):
        return

    hccl_buffsize_mb, buffsize_source = _resolve_deepep_hccl_buffsize()
    margin_mb = envs.SGLANG_NPU_MC2_SCRATCH_MARGIN_MB.get()
    scratch_mb = estimate_mc2_scratch_mb(
        hccl_buffsize_mb=hccl_buffsize_mb, margin_mb=margin_mb
    )
    headroom_mb = envs.SGLANG_NPU_MC2_CAPTURE_HEADROOM_MB.get()
    required_mb = scratch_mb + headroom_mb
    # Min across ranks, so every rank takes the same branch below.
    free_mb = (
        get_available_gpu_memory(
            device,
            gpu_id,
            distributed=get_parallel().launch_world_size > 1,
            cpu_group=get_parallel().world_group.cpu_group,
        )
        * 1024
    )
    if free_mb < required_mb:
        raise RuntimeError(
            f"Free NPU memory before decode graph capture ({free_mb:.0f} MB, min "
            f"across ranks) is below the {required_mb} MB the capture is "
            f"estimated to need: MC2 scratch {scratch_mb} MB (2 x hccl_buffer_size "
            f"{hccl_buffsize_mb} MB [{buffsize_source}] + {margin_mb} MB margin via "
            f"SGLANG_NPU_MC2_SCRATCH_MARGIN_MB), which CANN allocates outside the "
            f"torch_npu allocator on the first DeepEP low-latency dispatch, plus "
            f"capture headroom {headroom_mb} MB for HCCL windows and graph buffers "
            f"(SGLANG_NPU_MC2_CAPTURE_HEADROOM_MB). Lower --mem-fraction-static "
            f"(currently {get_schedule().mem_fraction_static}) or lower "
            f"DEEPEP_HCCL_BUFFSIZE/HCCL_BUFFSIZE. {_reservation_hint()}"
        )
    log_info_on_rank0(
        logger,
        f"MC2 scratch pre-flight passed: {free_mb:.0f} MB free (min across ranks) "
        f">= {required_mb} MB required = scratch {scratch_mb} MB (hccl_buffer_size "
        f"{hccl_buffsize_mb} MB [{buffsize_source}], margin {margin_mb} MB) + "
        f"capture headroom {headroom_mb} MB.",
    )


def _reservation_hint() -> str:
    if envs.SGLANG_NPU_RESERVE_MC2_SCRATCH.get():
        return (
            "SGLANG_NPU_RESERVE_MC2_SCRATCH is on, so the scratch is already "
            "reserved out of the KV cache budget."
        )
    return (
        "Alternatively, set SGLANG_NPU_RESERVE_MC2_SCRATCH=1 to reserve the "
        "scratch out of the KV cache budget."
    )


def mc2_scratch_reservation_gb(*, is_draft_worker: bool) -> float:
    """KV budget to hold back for the MC2 scratch; reserved once, on the target."""
    # Scratch only; the capture headroom is checked, not reserved, so
    # --mem-fraction-static still sizes it.
    if (
        is_draft_worker
        or not envs.SGLANG_NPU_RESERVE_MC2_SCRATCH.get()
        or not _mc2_scratch_applies()
    ):
        return 0.0

    reserved_mb = estimate_mc2_scratch_mb(
        hccl_buffsize_mb=resolve_deepep_hccl_buffsize_mb(),
        margin_mb=envs.SGLANG_NPU_MC2_SCRATCH_MARGIN_MB.get(),
    )
    logger.info(
        "Reserving %.2f GB of the KV budget for the MC2 scratch allocated on the "
        "first DeepEP low-latency dispatch.",
        reserved_mb / 1024,
    )
    return reserved_mb / 1024
