from __future__ import annotations

from typing import TYPE_CHECKING

from sglang.srt.kv_canary.config import CanaryMode
from sglang.srt.kv_canary.perturb.detokenizer_swap import (
    maybe_perturb_detokenizer_swap,
)
from sglang.srt.runtime_context import get_observability

if TYPE_CHECKING:
    from sglang.srt.managers.io_struct import BatchTokenIDOutput


def maybe_perturb_batch_token_id_out(recv_obj: BatchTokenIDOutput) -> None:
    if get_observability().kv_canary == CanaryMode.NONE:
        return
    maybe_perturb_detokenizer_swap(recv_obj)
