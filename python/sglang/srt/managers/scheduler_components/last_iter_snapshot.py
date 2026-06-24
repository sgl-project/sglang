from __future__ import annotations

from typing import TYPE_CHECKING, List, Optional

import msgspec

if TYPE_CHECKING:
    from sglang.srt.managers.schedule_batch import Req
    from sglang.srt.model_executor.forward_batch_info import ForwardMode


class LastIterSnapshot(msgspec.Struct):
    """Snapshot of the last forward iteration's batch for external readers.

    The event loop holds ``cur_batch`` / ``last_batch`` as local variables, so
    out-of-loop readers (request receiver, pool stats observer, invariant
    checker, idle/health checks) cannot see them. The loop publishes this
    snapshot at the end of each iteration instead.

    It keeps ``reqs`` (light: per-req tensors live on the ScheduleBatch, not on
    Req) but not the ScheduleBatch object, so the batch's GPU input tensors can
    be released once the loop drops its local reference.
    """

    forward_mode: Optional[ForwardMode]
    reqs: List[Req]
    is_empty: bool
