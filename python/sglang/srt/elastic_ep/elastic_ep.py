from __future__ import annotations

from curses import use_env
from dataclasses import dataclass
from typing import Optional

import torch


@dataclass
class ElasticEpMetadata:
    using_elastic_ep: bool
    active_ranks: Optional[torch.Tensor]
    last_active_ranks: Optional[torch.Tensor]


_global_elastic_ep_metadata: Optional[ElasticEpMetadata] = None


def get_global_elastic_ep_metadata():
    return _global_elastic_ep_metadata


def set_global_elastic_ep_metadata(value):
    global _global_elastic_ep_metadata
    assert _global_elastic_ep_metadata is None
    _global_elastic_ep_metadata = value


def _init_global_elastic_ep_metadata():
    global _global_elastic_ep_metadata
    if _global_elastic_ep_metadata is not None:
        return

    ep_size = torch.distributed.get_world_size()
    active_ranks = torch.ones(ep_size, dtype=torch.int32, device="cuda")
    last_active_ranks = active_ranks.clone()

    _global_elastic_ep_metadata = ElasticEpMetadata(
        using_elastic_ep=False,  # TODO pr elastic_ep to add args decide whether use elastic ep
        active_ranks=active_ranks,
        last_active_ranks=last_active_ranks,
    )
