from collections import defaultdict
from pathlib import Path
from typing import List, Optional

import torch
from tqdm import tqdm

from sglang.srt.eplb.expert_distribution import (
    _convert_global_physical_count_to_logical_count,
)

convert_global_physical_count_to_logical_count = (
    _convert_global_physical_count_to_logical_count
)


def read_mode_per_pass(dir_data: Path):
    """Read data from ExpertDistributionRecorder when recorded with mode `per_pass`"""

    # gpc := global_physical_count
    gpc_of_forward_pass_and_rank = defaultdict(lambda: defaultdict())
    for path in tqdm(list(dir_data.glob("*.pt"))):
        data_pack = torch.load(path, weights_only=True)
        last_physical_to_logical_map = data_pack["last_physical_to_logical_map"]
        for record in data_pack["records"]:
            forward_pass_id = record["forward_pass_id"]
            rank = record["rank"]
            assert (
                gpc_of_forward_pass_and_rank[forward_pass_id].get(rank) is None
            ), f"Duplicated {forward_pass_id=} {rank=}"
            gpc_of_forward_pass_and_rank[forward_pass_id][rank] = record[
                "global_physical_count"
            ]

    forward_pass_ids = sorted(gpc_of_forward_pass_and_rank.keys())
    print(f"Make {forward_pass_ids=} into array")

    items = []
    for forward_pass_id, gpc_of_rank in sorted(gpc_of_forward_pass_and_rank.items()):
        gpc_of_rank_tensor = torch.stack(
            [gpc for rank, gpc in sorted(gpc_of_rank.items())]
        ).sum(dim=0)
        items.append(gpc_of_rank_tensor)

    gpc_of_forward_pass = torch.stack(items)
    print(f"{gpc_of_forward_pass.shape=}")

    return dict(
        global_physical_count_of_forward_pass=gpc_of_forward_pass,
        last_physical_to_logical_map=last_physical_to_logical_map,
        forward_pass_ids=forward_pass_ids,
    )


def read_mode_stat(dir_data: Path):
    """Read data from ExpertDistributionRecorder when recorded with mode
    ``stat``/``stat_approx``.

    Stat-mode dumps are written by rank 0 only (one file per ``dump_record``
    call, named ``expert_distribution_recorder_{time}.pt``). Each file holds
    the buffered ``logical_count`` of shape
    ``[num_buffered_steps, num_layers, num_logical_experts]`` and the scalar
    ``average_utilization_rate_over_window`` recorded at that dump. This reader
    reconstructs the full time series by stacking the per-file buffers along
    the step (leading) dimension, in chronological (filename) order.
    """

    # Files are named with ``time.time()`` so a lexicographic sort matches the
    # chronological dump order, giving a step-ordered time series.
    logical_count_of_step: List[torch.Tensor] = []
    average_utilization_rate_of_step: List[Optional[float]] = []
    for path in tqdm(sorted(dir_data.glob("*.pt"))):
        data_pack = torch.load(path, weights_only=True)
        # Stat-mode dumps are rank-0 only; guard against accidentally mixing in
        # a per-pass/per-token dump (which has no ``logical_count`` key).
        assert data_pack.get("rank", 0) == 0, (
            f"stat-mode dumps are written by rank 0 only, but {path.name} has "
            f"rank={data_pack.get('rank')!r}"
        )
        logical_count_of_step.append(data_pack["logical_count"])
        average_utilization_rate_of_step.append(
            data_pack.get("average_utilization_rate_over_window")
        )

    if logical_count_of_step:
        logical_count_of_step = torch.cat(logical_count_of_step, dim=0)
    else:
        logical_count_of_step = torch.empty(0)

    print(f"{logical_count_of_step.shape=}")

    return dict(
        logical_count_of_step=logical_count_of_step,
        average_utilization_rate_of_step=average_utilization_rate_of_step,
    )
