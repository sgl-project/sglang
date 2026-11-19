"""Agree on adapter mutation outcomes before committing frontend state."""

from typing import Callable

import torch.distributed as dist

from sglang.srt.managers.io_struct import LoRAUpdateOutput
from sglang.srt.runtime_context import get_parallel


def run_lora_update(update: Callable[[], LoRAUpdateOutput]) -> LoRAUpdateOutput:
    try:
        local = update()
    except Exception as error:
        # Deserialization and checksum checks can fail before the LoRA loader
        # returns its normal result. Peers must still reach the collective.
        local = LoRAUpdateOutput(success=False, error_message=str(error))

    parallel = get_parallel()
    # Supported attention-DP LoRA configurations have attention TP/CP size 1:
    # every rank already replies to the frontend's control fan-out. Keep those
    # independently scheduled DP groups out of a new full-TP collective.
    if parallel.tp_size == 1 or parallel.attn_dp_enabled:
        return local
    results = [None] * parallel.tp_size
    dist.all_gather_object(results, local, group=parallel.tp_group.cpu_group)
    if all(result.success for result in results):
        return local

    errors = [
        f"TP rank {rank}: {result.error_message or 'LoRA update failed'}"
        for rank, result in enumerate(results)
        if not result.success
    ]
    # Keep every rank's resident adapters visible: the frontend needs the new
    # identity for cleanup even when its own I/O rank was the one that failed.
    # None means a rank threw before it could report its resident adapters.
    adapters = None
    if all(result.loaded_adapters is not None for result in results):
        adapters = {
            name: adapter
            for result in results
            for name, adapter in result.loaded_adapters.items()
        }
    return LoRAUpdateOutput(
        success=False, error_message=" | ".join(errors), loaded_adapters=adapters
    )
