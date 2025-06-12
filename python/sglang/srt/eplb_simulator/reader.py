import torch
from collections import defaultdict
from pathlib import Path

from tqdm import tqdm


def read_physical_count_of_forward_pass_id_and_rank(dir_data: Path):
    physical_count_of_forward_pass_id_and_rank = defaultdict(lambda: defaultdict())
    for path in tqdm(list(dir_data.glob("*.pt"))):
        data_pack = torch.load(path, weights_only=True)
        last_physical_to_logical_map = data_pack["last_physical_to_logical_map"]
        for record in data_pack["records"]:
            assert (
                physical_count_of_forward_pass_id_and_rank[
                    record["forward_pass_id"]
                ].get(record["rank"])
                is None
            )
            physical_count_of_forward_pass_id_and_rank[record["forward_pass_id"]][
                record["rank"]
            ] = record["physical_count"]
    # print(len(physical_count_of_forward_pass_id_and_rank))
    return physical_count_of_forward_pass_id_and_rank, last_physical_to_logical_map


def read_physical_count_of_forward_pass(dir_data: Path):
    physical_count_of_forward_pass_id_and_rank, last_physical_to_logical_map = (
        read_physical_count_of_forward_pass_id_and_rank(dir_data)
    )

    items = []
    for forward_pass_id, physical_count_of_rank in sorted(
        physical_count_of_forward_pass_id_and_rank.items()
    ):
        physical_count_of_rank_tensor = torch.stack(
            [
                physical_count
                for rank, physical_count in sorted(physical_count_of_rank.items())
            ]
        ).sum(dim=0)
        items.append(physical_count_of_rank_tensor)

    physical_count_of_forward_pass = torch.stack(items)
    print(f"{physical_count_of_forward_pass.shape=}")

    return physical_count_of_forward_pass, last_physical_to_logical_map
