# Adapted from https://github.com/vllm-project/vllm/blob/a6221a144af772fd1a68fe7e627935dc53e81738/vllm/distributed/parallel_state.py
# Currently uses monkey patching, but there are other ways like copy-paste-modify
import dataclasses
import logging
from typing import Optional, List, Union, Any

import torch
import vllm.distributed.parallel_state as _ps

logger = logging.getLogger(__name__)


def monkey_patch_vllm_distributed_parallel_state():
    _ps.GroupCoordinator.__init__ = _group_coordinator_init


def init_distributed_environment_via_existing(
        local_rank: int,
        backend: str,
):
    assert _ps._WORLD is None
    ranks = list(range(torch.distributed.get_world_size()))
    _ps._WORLD = _init_world_group(ranks, local_rank, backend)


def initialize_model_parallel_via_existing() -> None:
    assert _ps._TP is None, "tensor model parallel group is already initialized"
    _ps._TP = _init_model_parallel_group(
        group_ranks=None,
        local_rank=_ps.get_world_group().local_rank,
        backend=None,
        use_message_queue_broadcaster=True,
        group_name="tp",
    )
    # Not handle PP yet


# Only thing added: `**kwargs`
def _init_world_group(ranks: List[int], local_rank: int,
                      backend: str,
                      **kwargs) -> _ps.GroupCoordinator:
    return _ps.GroupCoordinator(
        group_ranks=[ranks],
        local_rank=local_rank,
        torch_distributed_backend=backend,
        use_pynccl=False,
        use_custom_allreduce=False,
        use_tpu_communicator=False,
        use_hpu_communicator=False,
        use_xpu_communicator=False,
        group_name="world",
        **kwargs,
    )


# Only thing added: `**kwargs`
def _init_model_parallel_group(
        group_ranks: List[List[int]],
        local_rank: int,
        backend: str,
        use_custom_allreduce: Optional[bool] = None,
        use_message_queue_broadcaster: bool = False,
        group_name: Optional[str] = None,
        **kwargs,
) -> _ps.GroupCoordinator:
    if use_custom_allreduce is None:
        use_custom_allreduce = _ps._ENABLE_CUSTOM_ALL_REDUCE
    return _ps.GroupCoordinator(
        group_ranks=group_ranks,
        local_rank=local_rank,
        torch_distributed_backend=backend,
        use_pynccl=True,
        use_custom_allreduce=use_custom_allreduce,
        use_tpu_communicator=True,
        use_hpu_communicator=True,
        use_xpu_communicator=True,
        use_message_queue_broadcaster=use_message_queue_broadcaster,
        group_name=group_name,
        **kwargs,
    )


@dataclasses
class GroupCoordinatorSourceExisting:
    ranks: List[int]
    device_group: Any
    cpu_group: Any


def _group_coordinator_init(
        self,
        group_ranks: List[List[int]],
        local_rank: int,
        torch_distributed_backend: Union[str, _ps.Backend],
        use_pynccl: bool,
        use_custom_allreduce: bool,
        use_tpu_communicator: bool,
        use_hpu_communicator: bool,
        use_xpu_communicator: bool,
        use_message_queue_broadcaster: bool = False,
        group_name: Optional[str] = None,
        # NOTE MODIFIED add
        existing: Optional[GroupCoordinatorSourceExisting] = None,
):
    group_name = group_name or "anonymous"
    self.unique_name = _ps._get_unique_name(group_name)
    _ps._register_group(self)

    self.rank = torch.distributed.get_rank()
    self.local_rank = local_rank
    self.device_group = None
    self.cpu_group = None

    # NOTE MODIFIED add this branch
    if existing is not None:
        assert torch_distributed_backend is None and group_ranks is None
        self.ranks = existing.ranks
        self.world_size = len(existing.ranks)
        self.rank_in_group = existing.ranks.index(self.rank)
        self.device_group = existing.device_group
        self.cpu_group = existing.cpu_group
    else:
        for ranks in group_ranks:
            device_group = torch.distributed.new_group(
                ranks, backend=torch_distributed_backend)
            # a group with `gloo` backend, to allow direct coordination between
            # processes through the CPU.
            cpu_group = torch.distributed.new_group(ranks, backend="gloo")
            if self.rank in ranks:
                self.ranks = ranks
                self.world_size = len(ranks)
                self.rank_in_group = ranks.index(self.rank)
                self.device_group = device_group
                self.cpu_group = cpu_group

    assert self.cpu_group is not None
    assert self.device_group is not None

    if _ps.current_platform.is_cuda_alike():
        self.device = torch.device(f"cuda:{local_rank}")
    else:
        self.device = torch.device("cpu")

    self.use_pynccl = use_pynccl
    self.use_custom_allreduce = use_custom_allreduce
    self.use_tpu_communicator = use_tpu_communicator
    self.use_hpu_communicator = use_hpu_communicator
    self.use_xpu_communicator = use_xpu_communicator

    # lazy import to avoid documentation build error
    from vllm.distributed.device_communicators.custom_all_reduce import (
        CustomAllreduce)
    from vllm.distributed.device_communicators.pynccl import (
        PyNcclCommunicator)

    self.pynccl_comm: Optional[PyNcclCommunicator] = None
    if use_pynccl and self.world_size > 1:
        self.pynccl_comm = PyNcclCommunicator(
            group=self.cpu_group,
            device=self.device,
        )

    self.ca_comm: Optional[CustomAllreduce] = None
    if use_custom_allreduce and self.world_size > 1:
        # Initialize a custom fast all-reduce implementation.
        self.ca_comm = CustomAllreduce(
            group=self.cpu_group,
            device=self.device,
        )

    from vllm.distributed.device_communicators.tpu_communicator import (
        TpuCommunicator)
    self.tpu_communicator: Optional[TpuCommunicator] = None
    if use_tpu_communicator and self.world_size > 1:
        self.tpu_communicator = TpuCommunicator(group=self.cpu_group)

    from vllm.distributed.device_communicators.hpu_communicator import (
        HpuCommunicator)
    self.hpu_communicator: Optional[HpuCommunicator]
    if use_hpu_communicator and self.world_size > 1:
        self.hpu_communicator = HpuCommunicator(group=self.device_group)

    from vllm.distributed.device_communicators.xpu_communicator import (
        XpuCommunicator)
    self.xpu_communicator: Optional[XpuCommunicator]
    if use_xpu_communicator and self.world_size > 1:
        self.xpu_communicator = XpuCommunicator(group=self.device_group)

    from vllm.distributed.device_communicators.shm_broadcast import (
        MessageQueue)
    self.mq_broadcaster: Optional[MessageQueue] = None
    if use_message_queue_broadcaster and self.world_size > 1:
        self.mq_broadcaster = MessageQueue.create_from_process_group(
            self.cpu_group, 1 << 22, 6)
