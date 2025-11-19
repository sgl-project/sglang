import logging
from typing import Union

import torch
import torch.distributed as dist
from torch.distributed import ProcessGroup

from sglang.srt.distributed import (
    get_moe_ep_group,
    get_moe_expert_parallel_rank,
    get_moe_expert_parallel_world_size,
    get_tensor_model_parallel_world_size,
    get_tensor_model_parallel_rank,
    get_pp_group,
)
from sglang.srt.layers.dp_attention import (
    get_attention_tp_rank,
    get_attention_tp_size,
    is_dp_attention_enabled,
)
from sglang.srt.distributed.parallel_state import get_tp_group
from sglang.srt.utils import is_flashinfer_available

from sglang.srt.utils import (
    direct_register_custom_op,
    is_flashinfer_available,
    supports_custom_op,
)
from sglang.srt.utils import is_cuda
from sglang.srt.distributed.parallel_state import in_the_same_node_as
logger = logging.getLogger(__name__)

if is_flashinfer_available():
    try:
        from flashinfer.comm.mnnvl import CommBackend
        import flashinfer.comm.trtllm_mnnvl_ar as trtllm_mnnvl_ar

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


            def bcast(self, data, root: int = 0):
                """
                Broadcast a picklable Python object from `root` to all ranks.
                Uses torch.distributed.broadcast_object_list under the hood.

                Returns the broadcasted object on every rank.
                """
                obj_list = [data]
                # broadcast_object_list mutates obj_list in-place
                dist.broadcast_object_list(obj_list, src=root, group=self._group)
                return obj_list[0]

            def barrier(self):
                """
                Synchronize all ranks in this communicator.
                """
                dist.barrier(group=self._group)

            def Split(self, color: int, key: int):
                # No need to split, we already use the proper group
                return self._group
        trtllm_mnnvl_available = True
    except ImportError:
        trtllm_mnnvl_available = False


# _allreduce_workspaces = None

# from flashinfer.comm.mnnvl import MnnvlConfig 
# # config = MnnvlConfig(
# #     comm_backend=TorchDistributedCommBackend(get_tp_group().cpu_group),
# #     fabric_page_size=-1,
# #     allocation_granularity=0,
# # )
    
# def initialize_flashinfer_allreduce_metas(dtype: torch.dtype):
#     from flashinfer.comm import Mapping
#     import flashinfer.comm.trtllm_mnnvl_ar as trtllm_mnnvl_ar
#     # from flashinfer.comm.trtllm_alltoall import (
#     #     Mapping,
#     #     MnnvlConfig,
#     #     MnnvlMemory,
#     #     MnnvlMoe,
#     # )
#     # Initialize workspaces
#     # TODO(shuw): real size of 
#     tp_size = get_tensor_model_parallel_world_size()
#     tp_rank = get_tensor_model_parallel_rank()
#     # print(f"I am rank:{tp_rank}")
#     # print(torch.cuda.device_count())
#     mapping = Mapping(
#         tp_size,
#         tp_rank,
#         gpus_per_node=torch.cuda.device_count(),
#         tp_size=tp_size,
#     )

#     global _allreduce_workspaces
#     if _allreduce_workspaces is not None:
#         return _allreduce_workspaces, mapping

#     torch.cuda.set_device(mapping.local_rank)

#     print(
#         f"[Node {mapping.node_rank}] Running MNNVL AllReduce test with {tp_size} ranks"
#     )
#     print(
#         f"[Node {mapping.node_rank}] Rank {tp_rank} using GPU {torch.cuda.current_device()}"
#     )

#     #TODO(shuw): make dtype configurable
#     comm = TorchDistributedCommBackend(get_tp_group().cpu_group)
#     # comm = TorchDistributedCommBackend(dist.group.WORLD)
#     _allreduce_workspaces = (
#         trtllm_mnnvl_ar.get_allreduce_mnnvl_workspace(mapping, dtype, comm)
#     )
#     return _allreduce_workspaces, mapping



# def run_flashinfer_mnnvl_allreduce(input: torch.Tensor) -> torch.Tensor:
#     from flashinfer.comm.trtllm_alltoall import MnnvlMemory
#     import flashinfer.comm.trtllm_mnnvl_ar as trtllm_mnnvl_ar
#     tp_size = get_tensor_model_parallel_world_size()
#     tp_rank = get_tensor_model_parallel_rank()
#     (mcast_buffer_mnnvl, buffer_flags_mnnvl, max_num_elements_mnnvl), mapping = initialize_flashinfer_allreduce_metas(
#         input.dtype
#     )
#     try:
#         multicast_ptr = mcast_buffer_mnnvl.get_multicast_ptr()
#         buffer_ptrs_dev = mcast_buffer_mnnvl.get_buffer_ptrs_dev()
#         unicast_ptr = mcast_buffer_mnnvl.mcast_device_memory.get_unicast_ptr(
#             mapping.tp_rank
#             # tp_rank
#         )
#     except Exception as e:
#         failure_message = f"FAILED[rank={tp_rank}]: failed: {e}"
#         print(failure_message)
#     hidden_size = input.size(-1)
#     assert max_num_elements_mnnvl % hidden_size == 0
#     # print(f"max_num_elements_mnnvl:{max_num_elements_mnnvl}")
#     buffer_M = max_num_elements_mnnvl // hidden_size
    


#     # print(f"calling trtllm_mnnvl_all_reduce with hs:{hidden_size}")
#     torch.cuda.set_device(mapping.local_rank)
#     input = input.view(-1, input.shape[-1])
#     output = torch.empty_like(input)
#     trtllm_mnnvl_ar.trtllm_mnnvl_all_reduce(
#         input,
#         multicast_ptr,
#         buffer_ptrs_dev,
#         buffer_M,
#         buffer_flags_mnnvl,
#         tp_size,
#         tp_rank,
#         True,  # wait_for_results
#         True,  # launch_with_pdl
#         output,  # Need to provide output tensor since we are writing them out.
#     )

#     # print("end trtllm_mnnvl_all_reduce")
#     # del mcast_buffer_mnnvl
#     # _allreduce_workspaces = None
#     return output

# def fake_run_flashinfer_mnnvl_allreduce(nput: torch.Tensor) -> torch.Tensor:
#     output = torch.empty_like(input)
#     return output

# if supports_custom_op():
#     direct_register_custom_op(
#         "run_flashinfer_mnnvl_allreduce",
#         run_flashinfer_mnnvl_allreduce,
#         mutates_args=["input"],
#         fake_impl=fake_run_flashinfer_mnnvl_allreduce,
#     )


# /////
_is_cuda_ = is_cuda()
class MnnvlAllReduce:
    def __init__(self, group: ProcessGroup, device: Union[int, str, torch.device]):
        """
        Args:
            group: Torch process group used for rendezvous and naming.
            device: Target CUDA device (index, 'cuda:X', or torch.device).
        """
        self.disabled = True
        if not trtllm_mnnvl_available:
            logger.info(
                "MNNVL all-reduce is disabled because flashinfer mnnvl is not available"
            )
            return

        if not _is_cuda_:
            logger.info(
                "TRTLLM all-reduce is disabled because it requires CUDA platform"
            )
            return
        self.device = device
        self.device_capability = torch.cuda.get_device_capability(device)[0]
        self.group = group
        self.world_size = dist.get_world_size(self.group)
        self.rank = dist.get_rank(self.group)
        if self.world_size == 1:
            return

        if self.device_capability < 10:
            logger.warning(
                "MNNVLMemCommunicator: Device capability %s not supported, "
                "communicator is not available.",
                self.device_capability,
            )
            return

        if not self._is_mnnvl():
            logger.info(
                "TRTLLM all-reduce is disabled because "
                "it is only used under Multi-Node NVLINK setup"
            )
            return

        self._initialize_workspace()
        self.disabled = False

    def _is_mnnvl(self) -> bool:
        """
        Check if current environment is a Multi-Node NVLINK setup.
        """
        return True
        all_on_same_node = all(in_the_same_node_as(self.group, source_rank=0))
        # Do not use TRTLLM all-reduce for single-node setup.
        if all_on_same_node:
            return False

        # from vllm.distributed.device_communicators.custom_all_reduce import (
        #     get_physical_device_ids,
        # )

        # # Check if the GPUs are fully connected by NVLINK
        # physical_device_ids = get_physical_device_ids(self.group, self.device)
        # fully_connected = current_platform.is_fully_connected(physical_device_ids)
        # TODO（shuw):
        fully_connected = True

        return fully_connected
    

    def destroy(self):
        if not self.disabled:
            try:
                del self.mcast_buffer_mnnvl
                del self.buffer_flags_mnnvl
                del self.max_num_elements_mnnvl
            except Exception as e:
                logger.warning(f"Error during MNNVL cleanup: {e}")


    def _initialize_workspace(self):
        from flashinfer.comm import Mapping
        gpus_per_node = sum(in_the_same_node_as(self.group, source_rank=0))

        mapping = Mapping(
            world_size=self.world_size,
            tp_size=self.world_size,
            rank=self.rank,
            gpus_per_node=gpus_per_node,
        )

        # Create custom communicator backend for flashinfer
        comm = TorchDistributedCommBackend(self.group)

        (
            self.mcast_buffer_mnnvl,
            self.buffer_flags_mnnvl,
            self.max_num_elements_mnnvl,
        ) = trtllm_mnnvl_ar.get_allreduce_mnnvl_workspace(mapping, torch.bfloat16, comm)

    def should_use_trtllm_mnnvl_ar(self, input_tensor: torch.Tensor) -> bool:
        if self.disabled:
            return False

        if not input_tensor.is_cuda:
            return False

        if input_tensor.dtype != torch.bfloat16:
            return False

        if not input_tensor.is_contiguous():
            return False

        tensor_numel = input_tensor.numel()
        if tensor_numel > self.max_num_elements_mnnvl:
            return False

        hidden_size = input_tensor.shape[-1]
        if self.max_num_elements_mnnvl % hidden_size != 0:
            return False

        return True
    

    def all_reduce(self, input: torch.Tensor) -> torch.Tensor:
        output = torch.empty_like(input)
        try:
            multicast_ptr = self.mcast_buffer_mnnvl.get_multicast_ptr()
            buffer_ptrs_dev = self.mcast_buffer_mnnvl.get_buffer_ptrs_dev()
            # unicast_ptr for rmsnorm fusion
            # unicast_ptr = self.mcast_buffer_mnnvl.mcast_device_memory.get_unicast_ptr()

            hidden_size = input.size(-1)
            assert self.max_num_elements_mnnvl % hidden_size == 0
            # print(f"max_num_elements_mnnvl:{max_num_elements_mnnvl}")
            buffer_M = self.max_num_elements_mnnvl // hidden_size
            

            # print(f"calling trtllm_mnnvl_all_reduce with hs:{hidden_size}")
            # torch.cuda.set_device(mapping.local_rank)
            # input = input.view(-1, input.shape[-1])
            trtllm_mnnvl_ar.trtllm_mnnvl_all_reduce(
                input,
                multicast_ptr,
                buffer_ptrs_dev,
                buffer_M,
                self.buffer_flags_mnnvl,
                self.world_size,
                self.rank,
                True,  # wait_for_results
                False,  # launch_with_pdl
                output,  # Need to provide output tensor since we are writing them out.
            )            

        except Exception as e:
            failure_message = f"FAILED[rank={self.rank}]: failed: {e}"
            print(failure_message)


        # print("end trtllm_mnnvl_all_reduce")
        # del mcast_buffer_mnnvl
        # _allreduce_workspaces = None
        return output


        # from flashinfer.comm import Mapping
        # import flashinfer.comm.trtllm_mnnvl_ar as trtllm_mnnvl_ar
        # # from flashinfer.comm.trtllm_alltoall import (
        # #     Mapping,
        # #     MnnvlConfig,
        # #     MnnvlMemory,
        # #     MnnvlMoe,
        # # )
        # # Initialize workspaces
        # # TODO(shuw): real size of 
        # tp_size = get_tensor_model_parallel_world_size()
        # tp_rank = get_tensor_model_parallel_rank()
        # # print(f"I am rank:{tp_rank}")
        # # print(torch.cuda.device_count())
        # mapping = Mapping(
        #     tp_size,
        #     tp_rank,
        #     gpus_per_node=torch.cuda.device_count(),
        #     tp_size=tp_size,
        # )

        # global _allreduce_workspaces
        # if _allreduce_workspaces is not None:
        #     return _allreduce_workspaces, mapping

        # torch.cuda.set_device(mapping.local_rank)

        # print(
        #     f"[Node {mapping.node_rank}] Running MNNVL AllReduce test with {tp_size} ranks"
        # )
        # print(
        #     f"[Node {mapping.node_rank}] Rank {tp_rank} using GPU {torch.cuda.current_device()}"
        # )

        # #TODO(shuw): make dtype configurable
        # comm = TorchDistributedCommBackend(get_tp_group().cpu_group)
        # # comm = TorchDistributedCommBackend(dist.group.WORLD)
        # _allreduce_workspaces = (
        #     trtllm_mnnvl_ar.get_allreduce_mnnvl_workspace(mapping, dtype, comm)
        # )
        # return _allreduce_workspaces, mapping