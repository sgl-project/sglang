from __future__ import annotations

from dataclasses import dataclass
from threading import Lock
from typing import Optional

import torch

from sglang.srt.utils import is_cpu, is_cuda


@dataclass
class ElasticEPState:
    using_elastic_ep: bool
    active_ranks: Optional[torch.Tensor]
    last_active_ranks: Optional[torch.Tensor]
    active_ranks_cpu: Optional[torch.Tensor]

    def is_active_equal_last(self) -> bool:
        if self.active_ranks is None or self.last_active_ranks is None:
            return False
        return torch.equal(self.active_ranks, self.last_active_ranks)

    def sync_active_to_cpu(self):
        if self.active_ranks is not None:
            self.active_ranks_cpu = self.active_ranks.detach().cpu().clone()

    def snapshot_active_to_last(self):
        if self.active_ranks is not None:
            self.last_active_ranks = self.active_ranks.clone()


__elastic_ep_state: Optional[ElasticEPState] = None
__state_lock = Lock()


def get_elastic_ep_state():
    global __elastic_ep_state
    if __elastic_ep_state is None:
        with __state_lock:
            if __elastic_ep_state is None:
                __elastic_ep_state = _build_default_state()
    return __elastic_ep_state


def _build_default_state() -> ElasticEPState:
    # TODO check server args to enable elastic_ep
    return _build_state(ep_size=None, device=None, using_elastic_ep=False)


def _select_device() -> torch.device:
    # cuda or cpu for now
    if is_cuda():
        return torch.device("cuda")
    elif is_cpu():
        return torch.device("cpu")
    else:
        raise NotImplementedError("Only CUDA and CPU are supported now.")


def _build_state(
    *,
    ep_size: Optional[int],
    device: Optional[torch.device],
    using_elastic_ep: bool,
) -> ElasticEPState:
    ep = ep_size if ep_size is not None else torch.distributed.get_world_size()
    dev = device if device is not None else _select_device()

    active = torch.ones(ep, dtype=torch.int32, device=dev)
    state = ElasticEPState(
        using_elastic_ep=using_elastic_ep,
        active_ranks=active,
        last_active_ranks=active.clone(),
        active_ranks_cpu=active.detach().cpu().clone(),
    )
    return state
