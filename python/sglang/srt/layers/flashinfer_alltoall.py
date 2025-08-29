import logging
import torch
import torch.distributed as dist
from torch.distributed import ProcessGroup

from sglang.srt.distributed import (
    get_moe_expert_parallel_world_size,
    get_moe_expert_parallel_rank,
    get_moe_ep_group,
)
from sglang.srt.utils import is_flashinfer_available

logger = logging.getLogger(__name__)

if is_flashinfer_available():
    from flashinfer.comm.mnnvl import CommBackend
    from flashinfer.comm.trtllm_alltoall import (
        Mapping,
        MnnvlConfig,
        MnnvlMemory,
        MnnvlMoe,
    )

class TorchDistributedCommBackend(CommBackend):
    """
    Use torch distributed instead of MPI to set up flashinfer MNNVL workspaces during initialization
    """
    def __init__(self, group: ProcessGroup):
        self._group = group

    def Get_rank(self) -> int:
        return self._group.rank()

    def Get_size(self) -> int:
        return self._group.size()

    def allgather(self, data: int):
        gathered = [None] * self.Get_size()
        dist.all_gather_object(gathered, data, group=self._group)
        return gathered

    def Split(self, color: int, key: int):
        # No need to split, we already use the proper group
        return self

_alltoall_workspaces = None

def initialize_flashinfer_alltoall_workspaces():
    global _alltoall_workspaces
    if _alltoall_workspaces is not None:
        return _alltoall_workspaces

    # Initialize workspaces
    ep_size = get_moe_expert_parallel_world_size()
    ep_rank = get_moe_expert_parallel_rank()
    mapping = Mapping(
        ep_size,
        ep_rank,
        gpus_per_node=torch.cuda.device_count(),
        tp_size=ep_size,
    )
    config = MnnvlConfig(
        comm_backend=TorchDistributedCommBackend(get_moe_ep_group().cpu_group),
        fabric_page_size=1 << 29,  # 512MB
        allocation_granularity=0,  # Auto-detect
    )
    MnnvlMemory.initialize()
    alltoall_workspace = MnnvlMoe.get_moe_workspaces(mapping, config)
    alltoall_prepare_workspace = MnnvlMoe.get_moe_prepare_workspace(
        mapping, config
    )
    _alltoall_workspaces = (alltoall_workspace, alltoall_prepare_workspace)
    return _alltoall_workspaces
