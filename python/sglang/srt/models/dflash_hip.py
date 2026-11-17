"""The DFLASH draft's ROCm paths. ``dflash`` calls these behind ``_is_hip``."""

import torch

from sglang.srt.runtime_context import get_parallel

# ROCm prefill: each rank projects its 1/TP of the rows, then all-gathers;
# CUDA keeps the replicated fc, which its DFLASH aux graph overlaps on a side stream
_FC_ROW_SHARD_MIN_ROWS = 4096


def projects_row_sharded(model, target_hidden: torch.Tensor) -> bool:
    """Whether DFlashDraftModel.project_target_hidden takes project_row_sharded."""
    return (
        model.fc_tp_columns is None
        and target_hidden.shape[0] >= _FC_ROW_SHARD_MIN_ROWS
        and get_parallel().tp_group.world_size > 1
    )


def project_row_sharded(model, target_hidden: torch.Tensor) -> torch.Tensor:
    """The replicated fc and the norm on this rank's 1/TP of the rows, then
    an all-gather, instead of every rank projecting every row."""
    group = get_parallel().tp_group
    num_rows = target_hidden.shape[0]
    rows = -(-num_rows // group.world_size)
    start = group.rank_in_group * rows
    shard = target_hidden[start : start + rows]
    if shard.shape[0] < rows:
        shard = torch.nn.functional.pad(shard, (0, 0, 0, rows - shard.shape[0]))
    projected = model.fc(shard)
    if model.quantizable:
        projected = projected[0]
    normed = model.hidden_norm(projected)
    gathered = normed.new_empty(rows * group.world_size, normed.shape[1])
    group.all_gather_into_tensor(gathered, normed.contiguous())
    return gathered[:num_rows]
