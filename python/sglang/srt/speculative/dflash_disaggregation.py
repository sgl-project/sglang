from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from sglang.srt.managers.overlap_utils import RelayPayload
from sglang.srt.speculative.draft_worker_common import make_draft_input_v2

if TYPE_CHECKING:
    from sglang.srt.managers.overlap_utils import FutureMap
    from sglang.srt.managers.schedule_batch import ScheduleBatch
    from sglang.srt.server_args import ServerArgs
    from sglang.srt.speculative.dflash_info_v2 import DFlashDraftInputV2


def build_dflash_disagg_draft_input(
    batch: ScheduleBatch,
    server_args: ServerArgs,
    last_tokens_tensor: torch.Tensor,
    future_map: FutureMap,
) -> "DFlashDraftInputV2":
    """Seed the first decode's draft state for DFLASH/DSPARK on a disaggregated
    decode node.

    The prefill happened on another node, so -- unlike the normal path -- there
    is no local forward that installs `batch.spec_info` before the first
    `prepare_for_decode`. We build the initial `DFlashDraftInputV2` the same way
    the prefill node's `_forward_extend` does (empty topk/hidden + bonus +
    seq_lens), then wire up the overlap relay exactly like the EAGLE path:
    `future_indices`, `publish` (so `resolve_seq_lens_cpu` can pull seq_lens),
    and `stash` (so `_resolve_spec_extras` reads the prefill bonus back out of
    `output_tokens_buf` instead of overwriting it with garbage).

    DFLASH/DSPARK only relay `bonus_tokens` through the FutureMap (topk/hidden
    are regenerated every step), so the `RelayPayload` carries nothing else.
    """
    spec_info = make_draft_input_v2(
        bonus_tokens=last_tokens_tensor,
        new_seq_lens=batch.seq_lens,
    )

    if batch.enable_overlap:
        spec_info.future_indices = batch.req_pool_indices
        # Seed the relay buf with the known seq_lens; publish's chained record
        # keeps the in-flight forward's fence intact (see FutureMap.publish).
        future_map.publish(spec_info.future_indices, batch.seq_lens)
        future_map.stash(
            spec_info.future_indices, RelayPayload.from_draft_input(spec_info)
        )

    return spec_info
