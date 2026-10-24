"""Communication groups for EPLB collectives and NPU weight migration."""

import logging

import torch.distributed as dist

from sglang.srt.runtime_context import get_device

logger = logging.getLogger(__name__)
_group_world = None
_device_group = None


def get_eplb_device_group():
    """Return a cached HCCL group with WORLD membership for NPU EPLB."""
    global _group_world, _device_group
    if get_device().device != "npu" or not dist.is_initialized():
        return None

    world = dist.group.WORLD
    if _group_world is not world:
        import torch_npu

        options = torch_npu._C._distributed_c10d.ProcessGroupHCCL.Options()
        # Keep migration off the forward graph's device-expanded HCCL group.
        # EPLB needs a smaller buffer than the MoE dispatch group.
        options.hccl_config = {
            "hccl_buffer_size": 32,
            "hccl_op_expansion_mode": 1,
        }
        _device_group = dist.new_group(backend="hccl", pg_options=options)
        _group_world = world
        logger.info(
            "[NPU EPLB host group] world=%d buffer=32MiB", dist.get_world_size()
        )
    return _device_group
