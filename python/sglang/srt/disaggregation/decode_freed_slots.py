"""PD decode admission that refills a finished request's slot in the same pass.

A finished running request is released when its result is processed, but it
leaves the running batch only in the next pass's `update_running_batch`, after
that pass's admission. With SGLANG_OPT_PD_DECODE_ADMIT_FREED_SLOTS the overlap
loop's admission counts its slot as free, so a waiting request starts one pass
earlier. Which requests are admitted, and in what order, is unchanged.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from sglang.srt.environ import envs
from sglang.srt.runtime_context import get_disagg

if TYPE_CHECKING:
    from sglang.srt.managers.schedule_batch import ScheduleBatch

logger = logging.getLogger(__name__)


def freed_slot_admission_on() -> bool:
    """Resolve the switch for the PD decode overlap loop. Server settings only,
    so every rank of a decode instance decides alike."""
    if not envs.SGLANG_OPT_PD_DECODE_ADMIT_FREED_SLOTS.get():
        return False
    if get_disagg().disaggregation_decode_enable_radix_cache:
        logger.info(
            "SGLANG_OPT_PD_DECODE_ADMIT_FREED_SLOTS declined: the decode radix cache "
            "is on"
        )
        return False
    logger.info(
        "SGLANG_OPT_PD_DECODE_ADMIT_FREED_SLOTS engaged: decode admission counts "
        "finished requests' slots as free in the pass that filters them out"
    )
    return True


def num_freed_slots(running_batch: ScheduleBatch) -> int:
    """Finished requests still in the running batch; a request whose last step
    is in flight is not finished yet and keeps its slot."""
    return sum(1 for req in running_batch.reqs if req.finished())
