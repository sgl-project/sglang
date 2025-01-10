# Adapted from https://github.com/vllm-project/vllm/blob/a6221a144af772fd1a68fe7e627935dc53e81738/vllm/distributed/parallel_state.py
# Currently uses monkey patching, but there are other ways like copy-paste-modify

from typing import Optional, List, Union

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
):
    group_name = group_name or "anonymous"
    self.unique_name = _ps._get_unique_name(group_name)
    _ps._register_group(self)

    self.rank = torch.distributed.get_rank()
    self.local_rank = local_rank
    self.device_group = None
    self.cpu_group = None

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
