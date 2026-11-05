"""Optional initialization of the native NPU pinned-host allocator."""

import logging
import os
import re

import torch

from sglang.srt.environ import envs

logger = logging.getLogger(__name__)


def maybe_prewarm_pinned_host(*, device: str, is_draft_worker: bool) -> None:
    if (
        device != "npu"
        or is_draft_worker
        or not envs.SGLANG_NPU_PREWARM_PINNED_HOST.get()
    ):
        return

    config = os.environ.get("PYTORCH_NPU_ALLOC_CONF", "")
    reserves = re.findall(
        r"(?:^|,)\s*pinned_reserve_segment_size_mb\s*:\s*([^,]+)", config
    )
    if len(reserves) != 1 or not reserves[0].strip().isdigit() or int(reserves[0]) <= 0:
        raise ValueError(
            "SGLANG_NPU_PREWARM_PINNED_HOST requires one positive "
            "pinned_reserve_segment_size_mb in PYTORCH_NPU_ALLOC_CONF "
            "before process startup, for example pinned_reserve_segment_size_mb:64."
        )

    # torch_npu initializes and caches the configured host segment on this
    # first tiny allocation. Do this before the large HiCache/L3 pools attach.
    warm_host = torch.empty(1, dtype=torch.uint8, device="cpu", pin_memory=True)
    if not warm_host.is_pinned():
        raise RuntimeError("NPU pinned-host warmup returned unpinned memory")
    del warm_host
    logger.info(
        "Initialized NPU pinned-host allocator with %s MiB reserve segments",
        reserves[0].strip(),
    )
