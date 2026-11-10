from typing import Optional

import torch
import torch.distributed as dist
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.tensor import DTensor

from sglang.srt.entrypoints.engine import Engine
from sglang.srt.managers.io_struct import UpdateWeightsFromTensorReqInput
from sglang.srt.model_executor.model_runner_components.weight_updater import (
    LocalSerializedTensor,
)
from sglang.srt.runtime_context import get_parallel
from sglang.srt.utils import MultiprocessingSerializer


async def update_weights(
    engine: Engine,
    params_batch: list[tuple[str, torch.Tensor]],
    device_mesh_key: str,
    device_mesh: DeviceMesh,
    load_format: Optional[str] = None,
):
    """
    Update weights for the inference engine.
    This function is designed to be stateless, so that the caller process could keep the stateful engine.
    The caller must hold an open weight-update session (engine.begin_weight_update() ...
    engine.end_weight_update()) around one or more calls.
    Example Use Case:
        - Multiple Producer Process will call this function in a SPMD style

    Args:
        engine: The inference engine created by the caller process.
        params_batch: A list of (name, tensor) tuples. We batched the tensors to avoid the overhead of cpu call.
        device_mesh_key: The key of the device mesh. Typically "tp" or "infer_tp"
        device_mesh: The device mesh.
        load_format: The format of the weights.
    """
    infer_tp_size = device_mesh[device_mesh_key].mesh.size()[0]
    infer_tp_rank = device_mesh[device_mesh_key].get_local_rank()
    tp_group = device_mesh[device_mesh_key].get_group()
    tp_rank_zero = device_mesh[device_mesh_key].mesh.tolist()[0]
    from sglang.srt.utils.patch_torch import monkey_patch_torch_reductions

    monkey_patch_torch_reductions()

    pp_size = [get_parallel().pp_size if infer_tp_rank == 0 else None]
    dist.broadcast_object_list(pp_size, src=tp_rank_zero, group=tp_group)
    pp_size = pp_size[0]

    # [
    #   (name0, [ipc_tensor0_tp0_pp0, ipc_tensor0_tp0_pp1]),
    #   (name1, [ipc_tensor1_tp0_pp0, ipc_tensor1_tp0_pp1]),
    # ]
    named_tensors_batch = []
    for name, tensor in params_batch:
        tensor = _preprocess_tensor_for_update_weights(tensor.detach())
        named_tensors_batch.append(
            (
                name,
                [MultiprocessingSerializer.serialize(tensor) for _ in range(pp_size)],
            )
        )

    if infer_tp_rank == 0:
        gathered_serialized_batches = [None for _ in range(infer_tp_size)]
    else:
        gathered_serialized_batches = None

    # [
    #   [ (name0, [ipc_tensor0_tp0_pp0, ...]), (name1, [ipc_tensor1_tp0_pp0, ...]) ],
    #   [ (name0, [ipc_tensor0_tp1_pp0, ...]), (name1, [ipc_tensor1_tp1_pp0, ...]) ],
    # ]
    dist.gather_object(
        obj=named_tensors_batch,
        object_gather_list=gathered_serialized_batches,
        dst=tp_rank_zero,
        group=tp_group,
    )

    if infer_tp_rank == 0:
        # Use zip(*) to "transpose" the data structure.
        # After transpose, the data structure is like:
        # [
        #   ( (name0, [ipc_tensor0_tp0_pp0, ...]), (name0, [ipc_tensor0_tp1_pp0, ...]) ),
        #   ( (name1, [ipc_tensor1_tp0_pp0, ...]), (name1, [ipc_tensor1_tp1_pp0, ...]) ),
        # ]
        logical_tensors = zip(*gathered_serialized_batches, strict=True)

        named_tensors = [
            # [
            #   (name0, LocalSerializedTensor(values=[ipc_tensor0_tp0_pp0, ipc_tensor0_tp1_pp0,
            #                                         ipc_tensor0_tp0_pp1, ipc_tensor0_tp1_pp1])),
            #   (name1, LocalSerializedTensor(values=[...])),
            # ]
            (
                tensor_group[0][0],
                LocalSerializedTensor(
                    values=[
                        rank_part[1][pp_rank]
                        for pp_rank in range(pp_size)
                        for rank_part in tensor_group
                    ]
                ),
            )
            for tensor_group in logical_tensors
        ]

        update_weights_request = UpdateWeightsFromTensorReqInput(
            serialized_named_tensors=engine.serialize_tensors_per_rank(
                named_tensors, load_format
            ),
            load_format=load_format,
        )

        return await engine.update_weights_from_tensor(update_weights_request)


def _preprocess_tensor_for_update_weights(tensor: torch.Tensor):
    """
    Preprocess the tensor for update weights.
    Example Use Case:
        - FSDP: we gather tensor by calling full_tensor in _preprocess_tensor_for_update_weights
        - Megatron: we do nothing here, assuming it is gathered when feed into this func

    Args:
        tensor: The tensor to be preprocessed.

    Returns:
        The full tensor if it is a DTensor, otherwise the original tensor.
    """
    if isinstance(tensor, DTensor):
        return tensor.full_tensor()
    return tensor
