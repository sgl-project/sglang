# Adapted from https://github.com/vllm-project/vllm/blob/a6221a144af772fd1a68fe7e627935dc53e81738/vllm/distributed/parallel_state.py

import torch
import vllm.distributed.parallel_state as _ps

logger = logging.getLogger(__name__)


def init_distributed_environment_via_existing(
    local_rank: int,
    backend: str,
):
    assert _ps._WORLD is None
    ranks = list(range(torch.distributed.get_world_size()))
    _ps._WORLD = init_world_group(ranks, local_rank, backend)
