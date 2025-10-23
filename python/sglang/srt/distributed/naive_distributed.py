import pickle
import time
from pathlib import Path
from typing import Any, List, Optional

import pybase64
import torch

from sglang.srt.utils import MultiprocessingSerializer


class NaiveDistributed:
    def __init__(self, rank: int, world_size: int, rendezvous: str):
        self._rank = rank
        self._world_size = world_size
        self._operation_index = 0
        self._directory = Path(rendezvous)
        self._directory.mkdir(parents=True, exist_ok=True)
        assert 0 <= rank < world_size

        # both barrier to be safe, and as a sanity check
        self.barrier()

    def get_rank(self):
        return self._rank

    def get_world_size(self):
        return self._world_size

    def scatter(
        self, tensor: torch.Tensor, scatter_list: List[torch.Tensor], src: int = 0
    ):
        if self._rank == src:
            assert len(scatter_list) == self._world_size
        else:
            assert scatter_list is None

        gathered_objects = self.all_gather_object(
            dict(
                serialized_scatter_list=[
                    (
                        None
                        if item_rank == src
                        else MultiprocessingSerializer.serialize(item)
                    )
                    for item_rank, item in enumerate(scatter_list)
                ]
            )
            if self._rank == src
            else dict()
        )

        remote_serialized_tensor = gathered_objects[src]["serialized_scatter_list"][
            self._rank
        ]
        if self._rank == src:
            assert remote_serialized_tensor is None
            remote_tensor = scatter_list[self._rank]
        else:
            remote_tensor = MultiprocessingSerializer.deserialize(
                remote_serialized_tensor
            )
        tensor.copy_(remote_tensor)

        # avoid src tensor be deleted too early
        self.barrier()

    def all_gather_object(self, obj: Any) -> List[Any]:
        self._operation_index += 1

        text_postfix = "\n"

        def _get_path(interesting_rank: int):
            return (
                self._directory
                / f"rank{interesting_rank}_op{self._operation_index}.txt"
            )

        _get_path(self._rank).write_text(
            pybase64.b64encode(pickle.dumps(obj)).decode("utf-8") + text_postfix
        )

        def _read_one(interesting_rank: int):
            p = _get_path(interesting_rank)
            while True:
                if p.exists() and (text := p.read_text()).endswith(text_postfix):
                    return pickle.loads(
                        pybase64.b64decode(text[: -len(text_postfix)], validate=True)
                    )
                time.sleep(0.001)

        return [
            _read_one(interesting_rank) for interesting_rank in range(self._world_size)
        ]

    def barrier(self):
        actual_objs = self.all_gather_object(self._rank)
        assert actual_objs == list(range(self._world_size)), f"{actual_objs=}"


# Can have multi instances if needed
_instance: Optional[NaiveDistributed] = None


def get_naive_distributed():
    assert _instance is not None
    return _instance


def set_naive_distributed(instance: NaiveDistributed):
    global _instance
    assert _instance is None
    _instance = instance
