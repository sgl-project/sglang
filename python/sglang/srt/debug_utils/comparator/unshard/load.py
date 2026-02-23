from pathlib import Path
from typing import Optional

import torch

from sglang.srt.debug_utils.comparator.dims import parse_dims
from sglang.srt.debug_utils.comparator.unshard.execute import execute_unshard_plan
from sglang.srt.debug_utils.comparator.unshard.parallel_info import (
    normalize_parallel_info,
)
from sglang.srt.debug_utils.comparator.unshard.plan import compute_unshard_plan
from sglang.srt.debug_utils.dump_loader import ValueWithMeta


def load_and_unshard(
    *,
    rows: list[dict],
    base_path: Path,
    dims_str: str,
    preloaded_first: Optional[ValueWithMeta] = None,
) -> Optional[torch.Tensor]:
    """Load all rank tensors for one side and unshard them.

    Returns None if any tensor fails to load.
    """
    dim_specs = parse_dims(dims_str)
    loaded: list[ValueWithMeta] = []

    for i, row in enumerate(rows):
        if i == 0 and preloaded_first is not None:
            loaded.append(preloaded_first)
        else:
            path = base_path / row["filename"]
            loaded.append(ValueWithMeta.load(path))

    parallel_infos = [normalize_parallel_info(item.meta) for item in loaded]

    plan = compute_unshard_plan(
        dim_specs=dim_specs,
        parallel_infos=parallel_infos,
    )

    tensors_by_index: dict[int, torch.Tensor] = {}
    for i, item in enumerate(loaded):
        if not isinstance(item.value, torch.Tensor):
            return None
        tensors_by_index[i] = item.value

    return execute_unshard_plan(plan, tensors_by_index)
