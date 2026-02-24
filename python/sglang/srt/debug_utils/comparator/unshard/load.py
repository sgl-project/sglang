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
    rows: list[dict],
    base_path: Path,
) -> Optional[torch.Tensor]:
    """Load all rank tensors for one side and unshard them.

    Reads dims from the first file's embedded metadata.
    When dims is missing: returns the raw tensor if there is exactly one row,
    otherwise returns None (ambiguous — cannot determine how to unshard).
    """
    if not rows:
        return None

    loaded = [ValueWithMeta.load(base_path / row["filename"]) for row in rows]

    dims_str = loaded[0].meta.get("dims")
    if dims_str is None:
        if len(loaded) == 1:
            return _as_tensor(loaded[0].value)
        return None

    dim_specs = parse_dims(dims_str)
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


def _as_tensor(value: object) -> Optional[torch.Tensor]:
    if not isinstance(value, torch.Tensor):
        return None
    return value
