from __future__ import annotations

import torch


class DsparkTpSync:
    def __init__(self, tp_group) -> None:
        self._tp_group = tp_group
        self._enabled = tp_group.world_size > 1

    def sync(self, tensor: torch.Tensor) -> torch.Tensor:
        if not self._enabled:
            return tensor
        pynccl_comm = self._tp_group.pynccl_comm
        if (
            tensor.device.type == "cuda"
            and pynccl_comm is not None
            and pynccl_comm.available
        ):
            with pynccl_comm.change_state(enable=True):
                pynccl_comm.broadcast(tensor, src=0)
        else:
            self._tp_group.broadcast(tensor, src=0)
        return tensor
