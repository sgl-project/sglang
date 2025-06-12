import torch
from collections import defaultdict
from pathlib import Path

from tqdm import tqdm


def read_mode_per_pass(dir_data: Path):
    global_physical_count_of_forward_pass_and_rank = defaultdict(lambda: defaultdict())
    for path in tqdm(list(dir_data.glob("*.pt"))):
        data_pack = torch.load(path, weights_only=True)
        last_physical_to_logical_map = data_pack["last_physical_to_logical_map"]
        for record in data_pack["records"]:
            forward_pass_id = record["forward_pass_id"]
            rank = record["rank"]
            assert global_physical_count_of_forward_pass_and_rank[forward_pass_id].get(rank) is None
            global_physical_count_of_forward_pass_and_rank[forward_pass_id][rank] = record["global_physical_count"]

    items = []
    for forward_pass_id, global_physical_count_of_rank in sorted(
        global_physical_count_of_forward_pass_and_rank.items()):
        global_physical_count_of_rank_tensor = torch.stack(
            [
                global_physical_count
                for rank, global_physical_count in sorted(global_physical_count_of_rank.items())
            ]
        ).sum(dim=0)
        items.append(global_physical_count_of_rank_tensor)

    global_physical_count_of_forward_pass = torch.stack(items)
    print(f"{global_physical_count_of_forward_pass.shape=}")

    return global_physical_count_of_forward_pass, last_physical_to_logical_map
