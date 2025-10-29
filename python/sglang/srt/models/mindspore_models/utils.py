# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the SGLang project
import mindspore as ms
import torch
import torch_npu
from mindspore.utils.dlpack import from_dlpack as ms_from_dlpack
from mindspore.utils.dlpack import to_dlpack as ms_to_dlpack

from sglang.srt.distributed import get_tp_group, get_world_group


def tensor_torch2ms(x: torch.Tensor):
    if x is None or not isinstance(x, torch.Tensor):
        return x

    if x.device.type == "cpu":
        # TODO: dlpack support CPU, for now will slow down the weight loading
        if x.dtype == torch.bfloat16:
            return ms.Tensor(
                x.contiguous().to(torch.float32).numpy(), dtype=ms.bfloat16
            )
        return ms.Tensor(x.contiguous().numpy())

    # torch tensor -> dlpack -> mindspore tensor
    pt_dlpack = torch.utils.dlpack.to_dlpack(x)
    ms_tensor = ms_from_dlpack(pt_dlpack)
    return ms_tensor


def tensor_ms2torch(x: ms.Tensor):
    if x is None or not isinstance(x, ms.Tensor):
        return x

    if x.device == "CPU":  # TODO: dlpack support CPU
        if x.dtype == ms.bfloat16:
            return torch.tensor(
                x.contiguous().to(ms.float32).asnumpy(), dtype=torch.bfloat16
            )
        return torch.tensor(x.contiguous().asnumpy())

    # ms tensor -> dlpack -> torch tensor
    ms_dlpack = ms_to_dlpack(x)
    torch_tensor = torch.utils.dlpack.from_dlpack(ms_dlpack)
    torch_npu.npu.synchronize()
    return torch_tensor


def split_loaded_weight(loaded_weight, shard_dim, start_idx, shard_size):
    if shard_dim is None:
        loaded_weight = loaded_weight[:]
        return loaded_weight

    end_idx = start_idx + shard_size
    if shard_dim == 0:
        loaded_weight = loaded_weight[start_idx:end_idx]
    elif shard_dim == 1:
        loaded_weight = loaded_weight[:, start_idx:end_idx]
    elif shard_dim == 2:
        loaded_weight = loaded_weight[:, :, start_idx:end_idx]
    else:
        raise ValueError("shard_dim:{} is not supported.".format(shard_dim))
    return loaded_weight


def _get_tp_group_name():
    return get_tp_group().unique_name


def _get_world_group_name():
    return get_world_group().unique_name


def set_weight_attrs(weight, weight_attrs):
    if not weight_attrs:
        return
    for key, value in weight_attrs.items():
        setattr(weight, key, value)
