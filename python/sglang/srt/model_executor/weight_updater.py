from __future__ import annotations

import logging

import torch

from sglang.srt.utils import init_custom_process_group
from sglang.srt.utils.network import NetworkAddress

logger = logging.getLogger(__name__)


def init_weights_update_group(
    *,
    _model_update_group,
    tp_rank,
    master_address,
    master_port,
    rank_offset,
    world_size,
    group_name,
    backend="nccl",
):
    """Initialize the Torch process group for model parameter updates.

    `_model_update_group` is used in the RLHF workflow, where rank
    0 is the actor model in the training engine, and the other ranks are
    the inference engine, which is used for rollout.

    In the RLHF workflow, the training engine updates the model
    weights/parameters online, and broadcasts them to the inference
    engine through the `_model_update_group` process group.
    """
    assert (
        torch.distributed.is_initialized()
    ), "Default torch process group must be initialized"
    assert group_name != "", "Group name cannot be empty"

    rank = rank_offset + tp_rank

    logger.info(
        f"init custom process group: master_address={master_address}, master_port={master_port}, "
        f"rank_offset={rank_offset}, rank={rank}, world_size={world_size}, group_name={group_name}, backend={backend}"
    )

    try:
        na = NetworkAddress(master_address, master_port)
        _model_update_group[group_name] = init_custom_process_group(
            backend=backend,
            init_method=na.to_tcp(),
            world_size=world_size,
            rank=rank,
            group_name=group_name,
        )
        return True, "Succeeded to initialize custom process group."
    except Exception as e:
        message = f"Failed to initialize custom process group: {e}."
        logger.error(message)
        return False, message


def destroy_weights_update_group(*, _model_update_group, group_name):
    try:
        if group_name in _model_update_group:
            pg = _model_update_group.pop(group_name)
            torch.distributed.destroy_process_group(pg)
            return True, "Succeeded to destroy custom process group."
        else:
            return False, "The group to be destroyed does not exist."
    except Exception as e:
        message = f"Failed to destroy custom process group: {e}."
        logger.error(message)
        return False, message
