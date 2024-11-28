"""
Common utilities for torch model parallelism.
"""

from typing import Optional

import torch
from torch.distributed.device_mesh import DeviceMesh

try:
    from torch.distributed.tensor import DTensor, Shard
except ImportError:
    # torch 2.4 or older
    from torch.distributed._tensor import DTensor, Shard

from torch.distributed._functional_collectives import AsyncCollectiveTensor
from torch.distributed.tensor.parallel import (
    ColwiseParallel,
    RowwiseParallel,
    parallelize_module,
)


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
            dtensor = DTensor.from_local(param, device_mesh, [Shard(0)])
            dist_param = torch.nn.Parameter(dtensor, requires_grad=False)
            module.register_parameter(name, dist_param)


class RowwiseParallelMaybeWait(RowwiseParallel):
    """
    A version of RowwiseParallel that waits for the output (establish dependency
    between comm stream and compute stream in CUDA sense) before going into the
    next op. This is needed to workaround the current interaction between
    AsyncCollectiveTensor and custom ops, such as `class RMSNorm(CustomOp)`.
    """

    @staticmethod
    def _prepare_output_fn(output_layouts, use_local_output, mod, outputs, device_mesh):
        outputs = super(
            RowwiseParallelMaybeWait, RowwiseParallelMaybeWait
        )._prepare_output_fn(
            output_layouts, use_local_output, mod, outputs, device_mesh
        )
        # wait for the output to be ready
        if isinstance(outputs, AsyncCollectiveTensor):
            return outputs.wait()
        else:
            return outputs


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
