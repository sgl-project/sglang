"""Agree on adapter mutation outcomes before committing frontend state."""

from typing import Callable, Optional

import torch.distributed as dist

from sglang.srt.managers.io_struct import LoRAUpdateOutput
from sglang.srt.runtime_context import get_parallel


def check_lora_update_parallelism() -> None:
    if get_parallel().pp_size > 1:
        raise ValueError(
            "Dynamic LoRA updates are not supported with pipeline parallelism."
        )


def _run_local_update(update: Callable[[], LoRAUpdateOutput]) -> LoRAUpdateOutput:
    try:
        return update()
    except Exception as error:
        # Deserialization and checksum checks can fail before the LoRA loader
        # returns its normal result. Peers must still reach the collective.
        return LoRAUpdateOutput(success=False, error_message=str(error))


def _gather_updates(local: LoRAUpdateOutput) -> list[LoRAUpdateOutput]:
    parallel = get_parallel()
    # Supported attention-DP LoRA configurations have attention TP/CP size 1:
    # every rank already replies to the frontend's control fan-out. Keep those
    # independently scheduled DP groups out of a new full-TP collective.
    if parallel.tp_size == 1 or parallel.attn_dp_enabled:
        return [local]
    results = [None] * parallel.tp_size
    dist.all_gather_object(results, local, group=parallel.tp_group.cpu_group)
    return results


def run_lora_update(
    update: Callable[[], LoRAUpdateOutput],
    *,
    rollback: Optional[Callable[[], LoRAUpdateOutput]] = None,
) -> LoRAUpdateOutput:
    try:
        check_lora_update_parallelism()
    except ValueError as error:
        return LoRAUpdateOutput(success=False, error_message=str(error))

    local = _run_local_update(update)
    results = _gather_updates(local)
    if all(result.success for result in results):
        return local

    errors = [
        f"TP rank {rank}: {result.error_message or 'LoRA update failed'}"
        for rank, result in enumerate(results)
        if not result.success
    ]
    if rollback is not None:
        # A pinned load reserves a slot before frontend registration. Restore
        # every rank before another batch can observe different capacities.
        results = _gather_updates(_run_local_update(rollback))
        rollback_errors = [
            f"TP rank {rank}: {result.error_message or 'LoRA rollback failed'}"
            for rank, result in enumerate(results)
            if not result.success
        ]
        if rollback_errors:
            # Serving with different slot counts can send TP ranks into
            # different forwards. Do not resume the scheduler in that state.
            raise RuntimeError(
                "LoRA load rollback failed: " + " | ".join(rollback_errors)
            )
    # Report adapters still resident after the operation or rollback. An unload
    # refusal may leave an adapter on a peer even when the I/O rank removed it.
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
