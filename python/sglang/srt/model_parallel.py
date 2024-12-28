"""
Common utilities for torch model parallelism.
"""

from typing import Optional, Sequence

import torch
import torch.nn as nn
from torch.distributed.device_mesh import DeviceMesh

try:
    import torch.distributed.tensor as dt
except ImportError:
    # torch 2.4 or older
    import torch.distributed._tensor as dt

from torch.distributed.tensor.parallel import (
    ColwiseParallel,
    RowwiseParallel,
    parallelize_module,
)


def _shard_tensor(
    full_tensor: torch.Tensor,
    device_mesh: DeviceMesh,
    placements: Sequence[dt.Shard],
) -> "dt.DTensor":
    """
    Locally shards a full tensor based on indicated sharding arrangement, and
    returns a DTensor containing the local shard.

    .. warning:: This is a private API that is subject to change. It skips the
        communication otherwise required by `distribute_tensor`. It is only
        applicable to cases where all ranks have the same `full_tensor`. For
        example, in distributed inference all ranks load from the same
        checkpoint. This API will not check for data equality between ranks, it
        is thus user's responsibility to ensure the `full_tensor` is the same
        across ranks.

    Args:
        full_tensor (torch.Tensor): the full tensor to be sharded.
        device_mesh (:class:`DeviceMesh`): DeviceMesh to place the
            DTensor.  Must have same dimension as the number of placements.
        placements (Sequence[:class:`Shard`]): the placements that
            describes how to place the local tensor on DeviceMesh.

    Returns:
        A :class:`DTensor` object with the shard as its local tensor.

    Examples:
        >>> # xdoctest: +SKIP("need world_size and rank")
        >>> device_mesh = dist.init_device_mesh("cuda", (world_size,))
        >>> full_tensor = torch.arange(world_size, device=f"cuda:{rank}")
        >>> dtensor = _shard_tensor(full_tensor, device_mesh, [Shard(1)])
    """
    shape, offset = dt._utils.compute_local_shape_and_global_offset(
        full_tensor.shape, device_mesh, placements
    )
    slices = [
        slice(cur_offset, cur_offset + cur_shape)
        for cur_shape, cur_offset in zip(shape, offset)
    ]
    local_tensor = full_tensor[slices]
    return dt.DTensor.from_local(local_tensor, device_mesh, placements)


class ColwiseParallelSharded(ColwiseParallel):
    """
    A version of ColwiseParallel where the local weight has been already
    sharded.  This is used for the fused wqkv case, where during loading, we
    already sharded wq, wk, wv before fusing them.
    """

    # Override the _partition_linear_fn in ColwiseParallel
    def _partition_linear_fn(self, name, module, device_mesh):
        # colwise shard weight/bias to Shard(0), weight be Shard(0)
        # means Colwise as Linear is input * weight^T + bias, where
        # weight would become Shard(1)
        for name, param in module.named_parameters():
            dtensor = dt.DTensor.from_local(param, device_mesh, [dt.Shard(0)])
            dist_param = torch.nn.Parameter(dtensor, requires_grad=False)
            module.register_parameter(name, dist_param)


class RowwiseParallelMaybeWait(RowwiseParallel):
    """
    A version of RowwiseParallel that waits for the output (establish dependency
    between comm stream and compute stream in CUDA sense) before going into the
    next op. This is needed to workaround the current interaction between
    AsyncCollectiveTensor and custom ops, such as `class RMSNorm(CustomOp)`.
    """

    def _partition_linear_fn(self, name, module, device_mesh):
        # Rowwise shard weight to Shard(1), bias to Replicate(), weight be Shard(1)
        # means Rowwise as nn.Linear is input * weight^T + bias, where
        # weight would become Shard(0)
        module.register_parameter(
            "weight",
            nn.Parameter(_shard_tensor(module.weight, device_mesh, [dt.Shard(1)])),
        )
        if getattr(module, "bias", None) is not None:
            # The Linear module has bias
            module.register_parameter(
                "bias",
                nn.Parameter(
                    dt.distribute_tensor(module.bias, device_mesh, [dt.Replicate()])
                ),
            )

    @staticmethod
    def _prepare_output_fn(output_layouts, use_local_output, mod, outputs, device_mesh):
        outputs = super(
            RowwiseParallelMaybeWait, RowwiseParallelMaybeWait
        )._prepare_output_fn(
            output_layouts, use_local_output, mod, outputs, device_mesh
        )
        return torch.distributed._functional_collectives.wait_tensor(outputs)


def tensor_parallel(
    module: torch.nn.Module,
    device_mesh: Optional[DeviceMesh] = None,
):
    """
    Tensor parallelize the model across the given device mesh.
    Args:
        module (`torch.nn.Module`):
            The module to tensor parallelize.
        device_mesh (`torch.distributed.DeviceMesh`):
            The device mesh to use for tensor parallelism.
    """

    # Tensor parallelize a nn.Module based on the `_tp_plan` attribute of the module.
    # No op if `_tp_plan` attribute does not exist under the module.
    # This is a helper function to be used with `model.apply` to recursively
    # parallelize a model.
    def tplize(mod: torch.nn.Module) -> None:
        tp_plan = getattr(mod, "_tp_plan", None)
        if tp_plan is None:
            return
        for child_name, tp_style in tp_plan.items():
            submod = mod.get_submodule(child_name)
            if tp_style == "Colwise":
                parallelize_module(submod, device_mesh, ColwiseParallel())
            elif tp_style == "Rowwise":
                parallelize_module(submod, device_mesh, RowwiseParallelMaybeWait())
            elif tp_style == "Colwise_Sharded":
                parallelize_module(submod, device_mesh, ColwiseParallelSharded())
            else:
                raise ValueError(f"Unknown TP style {tp_style}")

    # `apply` is a native method of `nn.Module` that recursively applies a
    # function to every submodule.
    module.apply(tplize)
