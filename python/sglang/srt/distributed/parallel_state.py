# Adapted from https://github.com/vllm-project/vllm/blob/a6221a144af772fd1a68fe7e627935dc53e81738/vllm/distributed/parallel_state.py
from typing import Optional

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


def initialize_model_parallel_via_existing(
    backend: Optional[str] = None,
) -> None:
    assert _ps._TP is None, "tensor model parallel group is already initialized"
    _ps._TP = init_model_parallel_group(group_ranks,
                                        get_world_group().local_rank,
                                        backend,
                                        use_message_queue_broadcaster=True,
                                        group_name="tp")
    # Not handle PP yet
