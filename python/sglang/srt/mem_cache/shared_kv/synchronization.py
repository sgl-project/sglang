"""Mandatory device-side publication for owner-only shared writes."""

import torch


class SharedWritePublisher:
    def __init__(self, attention_cp_group: object) -> None:
        self._attention_cp_group = attention_cp_group
        self._fence = torch.ones(
            (1,),
            dtype=torch.int32,
            device=torch.device("cuda", torch.cuda.current_device()),
        )

    def publish(self) -> None:
        self._fence.fill_(1)
        self._attention_cp_group._all_reduce_in_place(self._fence)
