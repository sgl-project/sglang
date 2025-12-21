# Copied and adapted from: https://github.com/hao-ai-lab/FastVideo

# Copyright 2024 xDiT team.
# Adapted from
# https://github.com/vllm-project/vllm/blob/main/vllm/distributed/parallel_state.py
# Copyright 2023 The vLLM team.
# Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.
import pickle
from collections import namedtuple
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.distributed
from torch.cuda import synchronize
from torch.distributed import Backend, ProcessGroup

from sglang.multimodal_gen import envs
from sglang.multimodal_gen.runtime.distributed.device_communicators.base_device_communicator import (
    DeviceCommunicatorBase,
)
from sglang.multimodal_gen.runtime.distributed.device_communicators.cpu_communicator import (
    CpuCommunicator,
)
from sglang.multimodal_gen.runtime.utils.logging_utils import (
    init_logger,
    suppress_stdout,
)

try:
    import torch_musa  # noqa: F401
    from torch_musa.core.device import synchronize
except ModuleNotFoundError:
    pass

logger = init_logger(__name__)

TensorMetadata = namedtuple("TensorMetadata", ["device", "dtype", "size"])


_group_name_counter: dict[str, int] = {}


def get_local_torch_device() -> torch.device:
    """Return the torch device for the current rank."""
    from sglang.multimodal_gen.runtime.platforms import current_platform

    return (
        torch.device(f"cuda:{envs.LOCAL_RANK}")
        if current_platform.is_cuda_alike()
        else torch.device("mps")
    )


def _get_unique_name(name: str) -> str:
    """Get a unique name for the group.
    Example:
    _get_unique_name("tp") -> "tp:0"
    _get_unique_name("tp") -> "tp:1"
    """
    if name not in _group_name_counter:
        _group_name_counter[name] = 0
    newname = f"{name}:{_group_name_counter[name]}"
    _group_name_counter[name] += 1
    return newname


def _split_tensor_dict(
    tensor_dict: Dict[str, Union[torch.Tensor, Any]], prefix: str = ""
) -> Tuple[List[Tuple[str, Any]], List[torch.Tensor]]:
    """Split the tensor dictionary into two parts:
    1. A list of (key, value) pairs. If the value is a tensor, it is replaced
         by its metadata.
    2. A list of tensors.

    If the Tensor is nested under `tensor_dict["key1"]["key2"]`, the key of its
    metadata will be "key1%key2".
    """
    metadata_list: List[Tuple[str, Any]] = []
    tensor_list = []
    for key, value in tensor_dict.items():
        assert "%" not in key, (
            "Avoid having '%' in key "
            "as it is used as a separator for nested entries."
        )
        if isinstance(value, torch.Tensor):
            # Note: we cannot use `value.device` here,
            # because it contains not only the device type but also the device
            # index (e.g. "cuda:0"). We only need the device type.
            # receiving side will set the device index.
            device = value.device.type
            metadata_list.append(
                (
                    prefix + key,
                    TensorMetadata(device, value.dtype, value.size()),
                )
            )
            tensor_list.append(value)
        elif isinstance(value, dict):
            if len(value) == 0:
                metadata_list.append((prefix + key, value))
            inner_metadata_list, inner_tensor_list = _split_tensor_dict(
                value, prefix + key + "%"
            )
            metadata_list.extend(inner_metadata_list)
            tensor_list.extend(inner_tensor_list)
        else:
            metadata_list.append((prefix + key, value))
    return metadata_list, tensor_list


def _update_nested_dict(nested_dict, flattened_key, value):
    key_splits = flattened_key.split("%")
    cur_dict = nested_dict
    for k in key_splits[:-1]:
        if k not in cur_dict:
            cur_dict[k] = {}
        cur_dict = cur_dict[k]
    cur_dict[key_splits[-1]] = value


@dataclass
class GraphCaptureContext:
    stream: torch.cuda.Stream | None


class GroupCoordinator:
    """
    PyTorch ProcessGroup wrapper for a group of processes.
    PyTorch ProcessGroup is bound to one specific communication backend,
        e.g. NCCL, Gloo, MPI, etc.
    GroupCoordinator takes charge of all the communication operations among
        the processes in the group. It can route the communication to
        a specific implementation (e.g. switch allreduce implementation
        based on the tensor size and cuda graph mode).
    """

    # available attributes:
    rank: int  # global rank
    ranks: List[int]  # global ranks in the group
    world_size: int  # size of the group
    # difference between `local_rank` and `rank_in_group`:
    # if we have a group of size 4 across two nodes:
    # Process | Node | Rank | Local Rank | Rank in Group
    #   0     |   0  |  0   |     0      |       0
    #   1     |   0  |  1   |     1      |       1
    #   2     |   1  |  2   |     0      |       2
    #   3     |   1  |  3   |     1      |       3
    local_rank: int  # local rank in the current node, used to assign devices
    rank_in_group: int  # rank inside the group
    cpu_group: ProcessGroup  # group for CPU communication
    device_group: ProcessGroup  # group for device communication
    use_device_communicator: bool  # whether to use device communicator
    device_communicator: DeviceCommunicatorBase  # device communicator

    def __init__(
        self,
        group_ranks: List[List[int]],
        local_rank: int,
        torch_distributed_backend: Union[str, Backend],
        use_device_communicator: bool = True,
        use_message_queue_broadcaster: bool = False,
        group_name: str | None = None,
    ):
        self.unique_name = _get_unique_name(group_name)
        self.rank = torch.distributed.get_rank()
        self.local_rank = local_rank
        self.device_group = None
        self.cpu_group = None

        for ranks in group_ranks:
            device_group = torch.distributed.new_group(
                ranks, backend=torch_distributed_backend
            )
            # a group with `gloo` backend, to allow direct coordination between
            # processes through the CPU.
            with suppress_stdout():
                cpu_group = torch.distributed.new_group(ranks, backend="gloo")
            if self.rank in ranks:
                self.ranks = ranks
                self.world_size = len(ranks)
                self.rank_in_group = ranks.index(self.rank)
                self.device_group = device_group
                self.cpu_group = cpu_group

        assert self.cpu_group is not None, f"{group_ranks=}, {local_rank=}"
        assert self.device_group is not None

        # TODO: fix it for other platforms
        self.device = get_local_torch_device()

        from sglang.multimodal_gen.runtime.platforms import current_platform

        self.use_device_communicator = use_device_communicator

        self.device_communicator: DeviceCommunicatorBase = None  # type: ignore
        if use_device_communicator and self.world_size > 1:
            # Platform-aware device communicator selection
            if current_platform.is_cuda_alike():
                from sglang.multimodal_gen.runtime.distributed.device_communicators.cuda_communicator import (
                    CudaCommunicator,
                )

                self.device_communicator = CudaCommunicator(
                    cpu_group=self.cpu_group,
                    device=self.device,
                    device_group=self.device_group,
                    unique_name=self.unique_name,
                )
            else:
                # For MPS and CPU, use the CPU communicator
                self.device_communicator = CpuCommunicator(
                    cpu_group=self.cpu_group,
                    device=self.device,
                    device_group=self.device_group,
                    unique_name=self.unique_name,
                )

        self.mq_broadcaster = None

        # TODO(will): check if this is needed
        # self.use_custom_op_call = current_platform.is_cuda_alike()
        self.use_custom_op_call = False

    @property
    def first_rank(self):
        """Return the global rank of the first process in the group"""
        return self.ranks[0]

    @property
    def last_rank(self):
        """Return the global rank of the last process in the group"""
        return self.ranks[-1]

    @property
    def is_first_rank(self):
        """Return whether the caller is the first process in the group"""
        return self.rank == self.first_rank

    @property
    def is_last_rank(self):
        """Return whether the caller is the last process in the group"""
        return self.rank == self.last_rank

    @property
    def next_rank(self):
        """Return the global rank of the process that follows the caller"""
        rank_in_group = self.rank_in_group
        world_size = self.world_size
        return self.ranks[(rank_in_group + 1) % world_size]

    @property
    def prev_rank(self):
        """Return the global rank of the process that precedes the caller"""
        rank_in_group = self.rank_in_group
        world_size = self.world_size
        return self.ranks[(rank_in_group - 1) % world_size]

    @property
    def group_next_rank(self):
        """Return the group rank of the process that follows the caller"""
        rank_in_group = self.rank_in_group
        world_size = self.world_size
        return (rank_in_group + 1) % world_size

    @property
    def group_prev_rank(self):
        """Return the group rank of the process that precedes the caller"""
        rank_in_group = self.rank_in_group
        world_size = self.world_size
        return (rank_in_group - 1) % world_size

    @property
    def skip_rank(self):
        """Return the global rank of the process that skip connects with the caller"""
        rank_in_group = self.rank_in_group
        world_size = self.world_size
        return self.ranks[(world_size - rank_in_group - 1) % world_size]

    @property
    def group_skip_rank(self):
        """Return the group rank of the process that skip connects with the caller"""
        rank_in_group = self.rank_in_group
        world_size = self.world_size
        return (world_size - rank_in_group - 1) % world_size

    @contextmanager
    def graph_capture(self, graph_capture_context: GraphCaptureContext | None = None):
        # Platform-aware graph capture
        from sglang.multimodal_gen.runtime.platforms import current_platform

        if current_platform.is_cuda_alike():
            if graph_capture_context is None:
                stream = torch.cuda.Stream()
                graph_capture_context = GraphCaptureContext(stream)
            else:
                stream = graph_capture_context.stream

            # ensure all initialization operations complete before attempting to
            # capture the graph on another stream
            curr_stream = torch.cuda.current_stream()
            if curr_stream != stream:
                stream.wait_stream(curr_stream)

            with torch.cuda.stream(stream):
                yield graph_capture_context
        else:
            # For non-CUDA platforms (MPS, CPU), just yield the context without stream management
            if graph_capture_context is None:
                # Create a dummy context for non-CUDA platforms
                graph_capture_context = GraphCaptureContext(None)
            yield graph_capture_context

    def all_to_all_4D(
        self, input_: torch.Tensor, scatter_dim: int = 2, gather_dim: int = 1
    ) -> torch.Tensor:
        if self.world_size == 1:
            return input_
        return self.device_communicator.all_to_all_4D(input_, scatter_dim, gather_dim)

    def all_reduce(
        self,
        input_: torch.Tensor,
        op=torch._C._distributed_c10d.ReduceOp.SUM,
        async_op: bool = False,
    ) -> torch.Tensor:
        """
        NOTE: This operation will be applied in-place or out-of-place.
        Always assume this function modifies its input, but use the return
        value as the output.
        """
        # Bypass the function if we are using only 1 GPU.
        if self.world_size == 1:
            return input_
        else:
            torch.distributed.all_reduce(
                input_, op=op, group=self.device_group, async_op=async_op
            )
        return input_

    def all_gather(
        self, input_: torch.Tensor, dim: int = 0, separate_tensors: bool = False
    ) -> Union[torch.Tensor, List[torch.Tensor]]:
        world_size = self.world_size
        # Bypass the function if we are using only 1 GPU.
        if world_size == 1:
            return input_
        assert (
            -input_.dim() <= dim < input_.dim()
        ), f"Invalid dim ({dim}) for input tensor with shape {input_.size()}"
        if dim < 0:
            # Convert negative dim to positive.
            dim += input_.dim()
        # Allocate output tensor.
        input_size = list(input_.size())
        input_size[0] *= world_size
        output_tensor = torch.empty(
            input_size, dtype=input_.dtype, device=input_.device
        )
        # All-gather.
        torch.distributed.all_gather_into_tensor(
            output_tensor, input_, group=self.device_group
        )
        if dim != 0:
            input_size[0] //= world_size
            output_tensor = output_tensor.reshape(
                [
                    world_size,
                ]
                + input_size
            )
            output_tensor = output_tensor.movedim(0, dim)

        if separate_tensors:
            tensor_list = [
                output_tensor.reshape(-1)
                .narrow(0, input_.numel() * i, input_.numel())
                .view_as(input_)
                for i in range(world_size)
            ]
            return tensor_list
        else:
            input_size = list(input_.size())
            input_size[dim] = input_size[dim] * world_size
            # Reshape
            output_tensor = output_tensor.reshape(input_size)
            return output_tensor

    def gather(self, input_: torch.Tensor, dst: int = 0, dim: int = -1) -> torch.Tensor:
        """
        NOTE: We assume that the input tensor is on the same device across
        all the ranks.
        NOTE: `dst` is the local rank of the destination rank.
        """
        world_size = self.world_size
        # Bypass the function if we are using only 1 GPU.
        if world_size == 1:
            return input_
        assert (
            -input_.dim() <= dim < input_.dim()
        ), f"Invalid dim ({dim}) for input tensor with shape {input_.size()}"
        if dim < 0:
            # Convert negative dim to positive.
            dim += input_.dim()
        # Allocate output tensor.
        if self.rank_in_group == dst:
            gather_list = [torch.empty_like(input_) for _ in range(world_size)]
        else:
            gather_list = None
        # Gather.
        torch.distributed.gather(
            input_, gather_list, dst=self.ranks[dst], group=self.device_group
        )
        if self.rank_in_group == dst:
            output_tensor = torch.cat(gather_list, dim=dim)
        else:
            output_tensor = None
        return output_tensor

    def broadcast(self, input_: torch.Tensor, src: int = 0, async_op: bool = False):
        """Broadcast the input tensor.
        NOTE: `src` is the local rank of the source rank.
        """
        assert src < self.world_size, f"Invalid src rank ({src})"

        # Bypass the function if we are using only 1 GPU.
        if self.world_size == 1:
            return input_
        # Broadcast.
        torch.distributed.broadcast(
            input_,
            src=self.ranks[src],
            group=self.device_group,
            async_op=async_op,
        )
        return input_

    def broadcast_object(self, obj: Optional[Any] = None, src: int = 0):
        """Broadcast the input object.
        NOTE: `src` is the local rank of the source rank.
        """
        assert src < self.world_size, f"Invalid src rank ({src})"

        # Bypass the function if we are using only 1 GPU.
        if self.world_size == 1:
            return obj
        if self.shm_broadcaster is not None:
            assert src == 0, "Shared memory broadcaster only supports src=0"
            return self.shm_broadcaster.broadcast_object(obj)
        if self.rank_in_group == src:
            torch.distributed.broadcast_object_list(
                [obj], src=self.ranks[src], group=self.cpu_group
            )
            return obj
        else:
            recv = [None]
            torch.distributed.broadcast_object_list(
                recv, src=self.ranks[src], group=self.cpu_group
            )
            return recv[0]

    def broadcast_object_list(
        self,
        obj_list: List[Any],
        src: int = 0,
        group: Optional[ProcessGroup] = None,
    ):
        """Broadcast the input object list.
        NOTE: `src` is the local rank of the source rank.
        """
        assert src < self.world_size, f"Invalid src rank ({src})"

        # Bypass the function if we are using only 1 GPU.
        if self.world_size == 1:
            return obj_list
        # Broadcast.
        torch.distributed.broadcast_object_list(
            obj_list, src=self.ranks[src], group=self.device_group
        )
        return obj_list

    def send_object(self, obj: Any, dst: int) -> None:
        """Send the input object list to the destination rank."""
        """NOTE: `dst` is the local rank of the destination rank."""

        assert dst < self.world_size, f"Invalid dst rank ({dst})"

        assert dst != self.rank, (
            "Invalid destination rank. Destination rank is the same "
            "as the current rank."
        )

        # Serialize object to tensor and get the size as well
        object_tensor = torch.frombuffer(pickle.dumps(obj), dtype=torch.uint8)

        size_tensor = torch.tensor(
            [object_tensor.numel()], dtype=torch.long, device="cpu"
        )

        # Send object size

        torch.distributed.send(size_tensor, dst=self.ranks[dst], group=self.cpu_group)

        # Send object
        torch.distributed.send(object_tensor, dst=self.ranks[dst], group=self.cpu_group)

        return None

    def recv_object(self, src: int) -> Any:
        """Receive the input object list from the source rank."""
        """NOTE: `src` is the local rank of the source rank."""

        assert src < self.world_size, f"Invalid src rank ({src})"

        assert (
            src != self.rank
        ), "Invalid source rank. Source rank is the same as the current rank."

        size_tensor = torch.empty(1, dtype=torch.long, device="cpu")

        # Receive object size
        rank_size = torch.distributed.recv(
            size_tensor, src=self.ranks[src], group=self.cpu_group
        )

        # Tensor to receive serialized objects into.
        object_tensor = torch.empty(  # type: ignore[call-overload]
            size_tensor.item(),  # type: ignore[arg-type]
            dtype=torch.uint8,
            device="cpu",
        )

        rank_object = torch.distributed.recv(
            object_tensor, src=self.ranks[src], group=self.cpu_group
        )

        assert (
            rank_object == rank_size
        ), "Received object sender rank does not match the size sender rank."

        obj = pickle.loads(object_tensor.numpy().tobytes())

        return obj

    def broadcast_tensor_dict(
        self,
        tensor_dict: Optional[Dict[str, Union[torch.Tensor, Any]]] = None,
        src: int = 0,
        group: Optional[ProcessGroup] = None,
        metadata_group: Optional[ProcessGroup] = None,
    ) -> Optional[Dict[str, Union[torch.Tensor, Any]]]:
        """Broadcast the input tensor dictionary.
        NOTE: `src` is the local rank of the source rank.
        """
        # Bypass the function if we are using only 1 GPU.
        if not torch.distributed.is_initialized() or self.world_size == 1:
            return tensor_dict

        group = self.device_group
        metadata_group = self.cpu_group
        assert src < self.world_size, f"Invalid src rank ({src})"
        src = self.ranks[src]

        rank = self.rank
        if rank == src:
            metadata_list: List[Tuple[Any, Any]] = []
            assert isinstance(
                tensor_dict, dict
            ), f"Expecting a dictionary, got {type(tensor_dict)}"
            metadata_list, tensor_list = _split_tensor_dict(tensor_dict)
            # `metadata_list` lives in CPU memory.
            # `broadcast_object_list` has serialization & deserialization,
            # all happening on CPU. Therefore, we can use the CPU group.
            self.broadcast_object(metadata_list, src=src)
            async_handles = []
            for tensor in tensor_list:
                if tensor.numel() == 0:
                    # Skip broadcasting empty tensors.
                    continue
                if tensor.is_cpu:
                    # use metadata_group for CPU tensors
                    handle = torch.distributed.broadcast(
                        tensor, src=src, group=metadata_group, async_op=True
                    )
                else:
                    # use group for GPU tensors
                    handle = torch.distributed.broadcast(
                        tensor, src=src, group=group, async_op=True
                    )
                async_handles.append(handle)
            for async_handle in async_handles:
                async_handle.wait()

        else:
            metadata_list = self.broadcast_object(None, src=src)
            tensor_dict = {}
            async_handles = []
            for key, value in metadata_list:
                if isinstance(value, TensorMetadata):
                    tensor = torch.empty(
                        value.size, dtype=value.dtype, device=value.device
                    )
                    if tensor.numel() == 0:
                        # Skip broadcasting empty tensors.
                        _update_nested_dict(tensor_dict, key, tensor)
                        continue
                    if tensor.is_cpu:
                        # use metadata_group for CPU tensors
                        handle = torch.distributed.broadcast(
                            tensor, src=src, group=metadata_group, async_op=True
                        )
                    else:
                        # use group for GPU tensors
                        handle = torch.distributed.broadcast(
                            tensor, src=src, group=group, async_op=True
                        )
                    async_handles.append(handle)
                    _update_nested_dict(tensor_dict, key, tensor)
                else:
                    _update_nested_dict(tensor_dict, key, value)
            for async_handle in async_handles:
                async_handle.wait()
        return tensor_dict

    def send_tensor_dict(
        self,
        tensor_dict: Dict[str, Union[torch.Tensor, Any]],
        dst: Optional[int] = None,
    ) -> Optional[Dict[str, Union[torch.Tensor, Any]]]:
        """Send the input tensor dictionary.
        NOTE: `dst` is the local rank of the source rank.
        """
        # Bypass the function if we are using only 1 GPU.
        if not torch.distributed.is_initialized() or self.world_size == 1:
            return tensor_dict

        group = self.device_group
        metadata_group = self.cpu_group

        if dst is None:
            dst = self.group_next_rank
        assert dst < self.world_size, f"Invalid dst rank ({dst})"

        metadata_list: List[Tuple[Any, Any]] = []
        assert isinstance(
            tensor_dict, dict
        ), f"Expecting a dictionary, got {type(tensor_dict)}"
        metadata_list, tensor_list = _split_tensor_dict(tensor_dict)
        # `metadata_list` lives in CPU memory.
        # `send_object_list` has serialization & deserialization,
        # all happening on CPU. Therefore, we can use the CPU group.
        self.send_object(metadata_list, dst=dst)
        for tensor in tensor_list:
            if tensor.numel() == 0:
                # Skip sending empty tensors.
                continue
            if tensor.is_cpu:
                # use metadata_group for CPU tensors
                torch.distributed.send(
                    tensor, dst=self.ranks[dst], group=metadata_group
                )
            else:
                # use group for GPU tensors
                torch.distributed.send(tensor, dst=self.ranks[dst], group=group)
        return None

    def recv_tensor_dict(
        self, src: Optional[int] = None
    ) -> Optional[Dict[str, Union[torch.Tensor, Any]]]:
        """Recv the input tensor dictionary.
        NOTE: `src` is the local rank of the source rank.
        """
        # Bypass the function if we are using only 1 GPU.
        if not torch.distributed.is_initialized() or self.world_size == 1:
            return None

        group = self.device_group
        metadata_group = self.cpu_group

        if src is None:
            src = self.group_prev_rank
        assert src < self.world_size, f"Invalid src rank ({src})"

        recv_metadata_list = self.recv_object(src=src)
        tensor_dict: Dict[str, Any] = {}
        for key, value in recv_metadata_list:
            if isinstance(value, TensorMetadata):
                tensor = torch.empty(value.size, dtype=value.dtype, device=value.device)
                if tensor.numel() == 0:
                    # Skip broadcasting empty tensors.
                    _update_nested_dict(tensor_dict, key, tensor)
                    continue
                if tensor.is_cpu:
                    # use metadata_group for CPU tensors
                    torch.distributed.recv(
                        tensor, src=self.ranks[src], group=metadata_group
                    )
                else:
                    # use group for GPU tensors
                    torch.distributed.recv(tensor, src=self.ranks[src], group=group)
                _update_nested_dict(tensor_dict, key, tensor)
            else:
                _update_nested_dict(tensor_dict, key, value)
        return tensor_dict

    def barrier(self):
        """Barrier synchronization among the group.
        NOTE: don't use `device_group` here! `barrier` in NCCL is
        terrible because it is internally a broadcast operation with
        secretly created GPU tensors. It is easy to mess up the current
        device. Use the CPU group instead.
        """
        torch.distributed.barrier(group=self.cpu_group)

    def send(self, tensor: torch.Tensor, dst: Optional[int] = None) -> None:
        """Sends a tensor to the destination rank in a non-blocking way"""
        """NOTE: `dst` is the rank_in_group of the destination rank."""
        if dst is None:
            dst = self.group_next_rank

        torch.distributed.send(
            tensor,
            self.ranks[dst],
            group=(
                self.device_groups[self.rank_in_group % 2]
                if self.world_size == 2
                else self.device_group
            ),
        )

    def recv(
        self, size: torch.Size, dtype: torch.dtype, src: Optional[int] = None
    ) -> torch.Tensor:
        """Receives a tensor from the src rank."""
        """NOTE: `src` is the rank_in_group of the source rank."""
        if src is None:
            src = self.group_prev_rank

        tensor = torch.empty(size, dtype=dtype, device=self.device)
        torch.distributed.recv(
            tensor,
            self.ranks[src],
            (
                self.device_groups[(self.rank_in_group + 1) % 2]
                if self.world_size == 2
                else self.device_group
            ),
        )
        return tensor

    def destroy(self) -> None:
        if self.device_group is not None:
            torch.distributed.destroy_process_group(self.device_group)
            self.device_group = None
        if self.cpu_group is not None:
            torch.distributed.destroy_process_group(self.cpu_group)
            self.cpu_group = None
        if self.device_communicator is not None:
            self.device_communicator.destroy()
        if self.mq_broadcaster is not None:
            self.mq_broadcaster = None


class PipelineGroupCoordinator(GroupCoordinator):
    """
    available attributes:
    rank: int  # global rank
    ranks: List[int]  # global ranks in the group
    world_size: int  # size of the group
    difference between `local_rank` and `rank_in_group`:
    if we have a group of size 4 across two nodes:
    Process | Node | Rank | Local Rank | Rank in Group
      0     |   0  |  0   |     0      |       0
      1     |   0  |  1   |     1      |       1
      2     |   1  |  2   |     0      |       2
      3     |   1  |  3   |     1      |       3
    local_rank: int  # local rank used to assign devices
    rank_in_group: int  # rank inside the group
    cpu_group: ProcessGroup  # group for CPU communication
    device_group: ProcessGroup  # group for device communication
    """

    def __init__(
        self,
        group_ranks: List[List[int]],
        local_rank: int,
        torch_distributed_backend: Union[str, Backend],
        group_name: str | None = None,
    ):
        super().__init__(
            group_ranks=group_ranks,
            local_rank=local_rank,
            torch_distributed_backend=torch_distributed_backend,
            group_name=group_name,
        )
        self.rank = torch.distributed.get_rank()
        self.local_rank = local_rank
        self.device_group = None
        self.cpu_group = None
        self.cpu_groups = []
        self.device_groups = []
        if len(group_ranks[0]) > 2 or len(group_ranks[0]) == 1:
            for ranks in group_ranks:
                device_group = torch.distributed.new_group(
                    ranks, backend=torch_distributed_backend
                )
                # a group with `gloo` backend, to allow direct coordination between
                # processes through the CPU.
                with suppress_stdout():
                    cpu_group = torch.distributed.new_group(ranks, backend="gloo")
                if self.rank in ranks:
                    self.ranks = ranks
                    self.world_size = len(ranks)
                    self.rank_in_group = ranks.index(self.rank)
                    self.device_group = device_group
                    self.cpu_group = cpu_group
        # when pipeline parallelism is 2, we need to create two groups to avoid
        #   communication stall.
        # *_group_0_1 represents the group for communication from device 0 to
        #   device 1.
        # *_group_1_0 represents the group for communication from device 1 to
        #   device 0.
        elif len(group_ranks[0]) == 2:
            for ranks in group_ranks:
                device_group_0_1 = torch.distributed.new_group(
                    ranks, backend=torch_distributed_backend
                )
                device_group_1_0 = torch.distributed.new_group(
                    ranks, backend=torch_distributed_backend
                )
                # a group with `gloo` backend, to allow direct coordination between
                # processes through the CPU.
                with suppress_stdout():
                    cpu_group_0_1 = torch.distributed.new_group(ranks, backend="gloo")
                    cpu_group_1_0 = torch.distributed.new_group(ranks, backend="gloo")
                if self.rank in ranks:
                    self.ranks = ranks
                    self.world_size = len(ranks)
                    self.rank_in_group = ranks.index(self.rank)
                    self.device_groups = [device_group_0_1, device_group_1_0]
                    self.cpu_groups = [cpu_group_0_1, cpu_group_1_0]
                    self.device_group = device_group_0_1
                    self.cpu_group = cpu_group_0_1

        assert self.cpu_group is not None
        assert self.device_group is not None

        self.device = envs.get_device(local_rank)

        self.recv_buffer_set: bool = False
        self.recv_tasks_queue: List[Tuple[str, int]] = []
        self.receiving_tasks: List[Tuple[torch.distributed.Work, str, int]] = []
        self.dtype: Optional[torch.dtype] = None
        self.num_pipefusion_patches: Optional[int] = None

        self.recv_shape: Dict[str, Dict[int, torch.Size]] = {}
        self.send_shape: Dict[str, Dict[int, torch.Size]] = {}
        self.recv_buffer: Dict[str, Dict[int, torch.Size]] = {}

        self.skip_tensor_recv_buffer_set: bool = False
        self.recv_skip_tasks_queue: List[Union[int, Tuple[str, int]]] = []
        self.receiving_skip_tasks: List[Tuple[torch.distributed.Work, str, int]] = []
        self.skip_tensor_recv_buffer: Optional[
            Union[List[torch.Tensor], torch.Tensor]
        ] = None
        self.skip_device_group = None
        for ranks in group_ranks:
            skip_device_group = torch.distributed.new_group(
                ranks, backend=torch_distributed_backend
            )
            if self.rank in ranks:
                self.skip_device_group = skip_device_group
        assert self.skip_device_group is not None

    def reset_buffer(self):
        self.recv_tasks_queue = []
        self.receiving_tasks = []
        self.recv_shape = {}
        self.send_shape = {}
        self.recv_buffer = {}

        self.recv_skip_tasks_queue = []
        self.receiving_skip_tasks = []
        self.skip_tensor_recv_buffer = {}

    def set_config(self, dtype: torch.dtype):
        self.dtype = dtype

    def set_recv_buffer(
        self,
        num_pipefusion_patches: int,
        patches_shape_list: List[List[int]],
        feature_map_shape: List[int],
        dtype: torch.dtype,
    ):
        assert isinstance(dtype, torch.dtype), "dtype must be a torch.dtype object"
        assert (
            isinstance(num_pipefusion_patches, int) and num_pipefusion_patches >= 1
        ), "num_pipefusion_patches must be greater than or equal to 1"
        self.dtype = dtype
        self.num_pipefusion_patches = num_pipefusion_patches
        self.recv_buffer = [
            torch.zeros(*shape, dtype=self.dtype, device=self.device)
            for shape in patches_shape_list
        ]
        self.recv_buffer.append(
            torch.zeros(*feature_map_shape, dtype=self.dtype, device=self.device)
        )
        self.recv_buffer_set = True

    def set_extra_tensors_recv_buffer(
        self,
        name: str,
        shape: List[int],
        num_buffers: int = 1,
        dtype: torch.dtype = torch.float16,
    ):
        self.extra_tensors_recv_buffer[name] = [
            torch.zeros(*shape, dtype=dtype, device=self.device)
            for _ in range(num_buffers)
        ]

    def _check_shape_and_buffer(
        self,
        tensor_send_to_next=None,
        recv_prev=False,
        name: Optional[str] = None,
        segment_idx: int = 0,
    ):
        send_flag = False
        name = name or "latent"
        if tensor_send_to_next is not None:
            shape_list = self.send_shape.get(name, None)
            if shape_list is None:
                self.send_shape[name] = {segment_idx: tensor_send_to_next.shape}
                send_flag = True
            elif shape_list.get(segment_idx, None) is None:
                self.send_shape[name][segment_idx] = tensor_send_to_next.shape
                send_flag = True

        recv_flag = False
        if recv_prev:
            shape_list = self.recv_shape.get(name, None)
            if shape_list is None:
                recv_flag = True
            elif shape_list.get(segment_idx, None) is None:
                recv_flag = True

        recv_prev_shape = self._communicate_shapes(
            tensor_send_to_next=tensor_send_to_next if send_flag else None,
            recv_prev=recv_flag,
        )

        if recv_flag:
            if self.recv_shape.get(name, None) is None:
                self.recv_shape[name] = {segment_idx: recv_prev_shape}
            else:
                self.recv_shape[name][segment_idx] = recv_prev_shape

            if self.recv_buffer.get(name, None) is None:
                self.recv_buffer[name] = {
                    segment_idx: torch.zeros(
                        recv_prev_shape, device=self.device, dtype=self.dtype
                    )
                }
            else:
                if self.recv_buffer[name].get(segment_idx, None) is not None:
                    logger.warning(
                        f"Recv buffer [name: {name}, segment_idx: {segment_idx}] already exist. updating..."
                    )
                self.recv_buffer[name][segment_idx] = torch.zeros(
                    recv_prev_shape, device=self.device, dtype=self.dtype
                )

    def _communicate_shapes(self, tensor_send_to_next=None, recv_prev=False):
        """Communicate tensor shapes between stages. Used to communicate
        tensor shapes before the actual tensor communication happens.

        Args:
            tensor_send_next: tensor to send to next rank (no tensor sent if
                              set to None).
            recv_prev: boolean for whether tensor should be received from
                       previous rank.
        """

        ops = []
        if recv_prev:
            recv_prev_dim_tensor = torch.empty(
                (1), device=self.device, dtype=torch.int64
            )
            recv_prev_dim_op = torch.distributed.P2POp(
                torch.distributed.irecv,
                recv_prev_dim_tensor,
                self.prev_rank,
                self.device_group,
            )
            ops.append(recv_prev_dim_op)

        if tensor_send_to_next is not None:
            send_next_dim_tensor = torch.tensor(
                tensor_send_to_next.dim(), device=self.device, dtype=torch.int64
            )
            send_next_dim_op = torch.distributed.P2POp(
                torch.distributed.isend,
                send_next_dim_tensor,
                self.next_rank,
                self.device_group,
            )
            ops.append(send_next_dim_op)

        if len(ops) > 0:
            reqs = torch.distributed.batch_isend_irecv(ops)
            for req in reqs:
                req.wait()

        # To protect against race condition when using batch_isend_irecv().
        # should take this out once the bug with batch_isend_irecv is resolved.
        synchronize()

        ops = []
        recv_prev_shape_tensor = None
        if recv_prev:
            recv_prev_shape_tensor = torch.empty(
                torch.Size(recv_prev_dim_tensor),
                device=self.device,
                dtype=torch.int64,
            )
            recv_prev_shape_op = torch.distributed.P2POp(
                torch.distributed.irecv,
                recv_prev_shape_tensor,
                self.prev_rank,
                self.device_group,
            )
            ops.append(recv_prev_shape_op)

        if tensor_send_to_next is not None:
            send_next_shape_tensor = torch.tensor(
                tensor_send_to_next.size(),
                device=self.device,
                dtype=torch.int64,
            )
            send_next_shape_op = torch.distributed.P2POp(
                torch.distributed.isend,
                send_next_shape_tensor,
                self.next_rank,
                self.device_group,
            )
            ops.append(send_next_shape_op)

        if len(ops) > 0:
            reqs = torch.distributed.batch_isend_irecv(ops)
            for req in reqs:
                req.wait()

        synchronize()

        recv_prev_shape = [0, 0, 0]
        if recv_prev_shape_tensor is not None:
            recv_prev_shape = recv_prev_shape_tensor
        return torch.Size(recv_prev_shape)

    def pipeline_send(
        self, tensor: torch.Tensor, name: str = "latent", segment_idx: int = -1
    ) -> None:
        tensor = tensor.contiguous()
        self._check_shape_and_buffer(
            tensor_send_to_next=tensor, name=name, segment_idx=segment_idx
        )
        self._pipeline_isend(tensor).wait()

    def pipeline_isend(
        self, tensor: torch.Tensor, name: str = "latent", segment_idx: int = -1
    ) -> None:
        tensor = tensor.contiguous()
        self._check_shape_and_buffer(
            tensor_send_to_next=tensor, name=name, segment_idx=segment_idx
        )
        self._pipeline_isend(tensor)

    def pipeline_recv(self, idx: int = -1, name: str = "latent") -> torch.Tensor:
        name = name or "latent"
        self._check_shape_and_buffer(recv_prev=True, name=name, segment_idx=idx)
        self._pipeline_irecv(self.recv_buffer[name][idx]).wait()
        return self.recv_buffer[name][idx]

    def add_pipeline_recv_task(self, idx: int = -1, name: str = "latent"):
        name = name or "latent"
        self.recv_tasks_queue.append((name, idx))

    def recv_next(self):
        if len(self.recv_tasks_queue) == 0:
            raise ValueError("No more tasks to receive")
        elif len(self.recv_tasks_queue) > 0:
            name, idx = self.recv_tasks_queue.pop(0)
            self._check_shape_and_buffer(recv_prev=True, name=name, segment_idx=idx)
            self.receiving_tasks.append(
                (self._pipeline_irecv(self.recv_buffer[name][idx]), name, idx)
            )

    def get_pipeline_recv_data(
        self, idx: int = -1, name: str = "latent"
    ) -> torch.Tensor:
        assert (
            len(self.receiving_tasks) > 0
        ), "No tasks to receive, call add_pipeline_recv_task first"
        receiving_task = self.receiving_tasks.pop(0)
        receiving_task[0].wait()
        assert (
            receiving_task[1] == name and receiving_task[2] == idx
        ), "Received tensor does not match the requested"
        return self.recv_buffer[name][idx]

    def _pipeline_irecv(self, tensor: torch.tensor):
        return torch.distributed.irecv(
            tensor,
            src=self.prev_rank,
            group=(
                self.device_groups[(self.rank_in_group + 1) % 2]
                if self.world_size == 2
                else self.device_group
            ),
        )

    def _pipeline_isend(self, tensor: torch.tensor):
        return torch.distributed.isend(
            tensor,
            dst=self.next_rank,
            group=(
                self.device_groups[self.rank_in_group % 2]
                if self.world_size == 2
                else self.device_group
            ),
        )

    def set_skip_tensor_recv_buffer(
        self,
        patches_shape_list: List[List[int]],
        feature_map_shape: List[int],
    ):
        self.skip_tensor_recv_buffer = [
            torch.zeros(*shape, dtype=self.dtype, device=self.device)
            for shape in patches_shape_list
        ]
        self.skip_tensor_recv_buffer.append(
            torch.zeros(*feature_map_shape, dtype=self.dtype, device=self.device)
        )
        self.skip_tensor_recv_buffer_set = True

    def pipeline_send_skip(self, tensor: torch.Tensor) -> None:
        tensor = tensor.contiguous()
        self._pipeline_isend_skip(tensor).wait()

    def pipeline_isend_skip(self, tensor: torch.Tensor) -> None:
        tensor = tensor.contiguous()
        self._pipeline_isend_skip(tensor)

    def pipeline_recv_skip(self, idx: int = -1) -> torch.Tensor:
        self._pipeline_irecv_skip(self.skip_tensor_recv_buffer[idx]).wait()
        return self.skip_tensor_recv_buffer[idx]

    def add_pipeline_recv_skip_task(self, idx: int = -1):
        self.recv_skip_tasks_queue.append(idx)

    def get_pipeline_recv_skip_data(self, idx: int = -1) -> torch.Tensor:
        assert (
            len(self.receiving_skip_tasks) > 0
        ), "No tasks to receive, call add_pipeline_recv_skip_task first"
        receiving_skip_task = self.receiving_skip_tasks.pop(0)
        receiving_skip_task[0].wait()
        assert (
            receiving_skip_task[2] == idx
        ), "Received tensor does not match the requested"
        return self.skip_tensor_recv_buffer[idx]

    def recv_skip_next(self):
        if len(self.recv_skip_tasks_queue) == 0:
            raise ValueError("No more tasks to receive")
        elif len(self.recv_skip_tasks_queue) > 0:
            task = self.recv_skip_tasks_queue.pop(0)
            idx = task
            self.receiving_skip_tasks.append(
                (
                    self._pipeline_irecv_skip(self.skip_tensor_recv_buffer[idx]),
                    None,
                    idx,
                )
            )

    def _pipeline_irecv_skip(self, tensor: torch.tensor):
        return torch.distributed.irecv(
            tensor, src=self.skip_rank, group=self.skip_device_group
        )

    def _pipeline_isend_skip(self, tensor: torch.tensor):
        return torch.distributed.isend(
            tensor, dst=self.skip_rank, group=self.skip_device_group
        )


class SequenceParallelGroupCoordinator(GroupCoordinator):
    def __init__(
        self,
        group_ranks: List[List[int]],
        local_rank: int,
        torch_distributed_backend: Union[str, Backend],
        group_name: str | None = None,
        **kwargs,
    ):
        super().__init__(
            group_ranks=group_ranks,
            local_rank=local_rank,
            torch_distributed_backend=torch_distributed_backend,
            group_name=group_name,
        )
        ulysses_group = kwargs.get("ulysses_group", None)
        ring_group = kwargs.get("ring_group", None)
        if ulysses_group is None:
            raise RuntimeError(
                f"Please pass argument 'ulysses_group' when calling init func of SequenceParallelGroupCoordinator"
            )
        if ring_group is None:
            raise RuntimeError(
                f"Please pass argument 'ring_group' when calling init func of SequenceParallelGroupCoordinator"
            )
        self.ulysses_group = ulysses_group
        self.ring_group = ring_group

        self.ulysses_world_size = torch.distributed.get_world_size(self.ulysses_group)
        self.ulysses_rank = torch.distributed.get_rank(self.ulysses_group)
        self.ring_world_size = torch.distributed.get_world_size(self.ring_group)
        self.ring_rank = torch.distributed.get_rank(self.ring_group)
