# Copyright 2023-2024 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
from typing import Callable

import torch


def monkey_patch_torch_reductions():
    """Monkey patching before Torch https://github.com/pytorch/pytorch/pull/149248 is fixed"""

    from torch.multiprocessing import reductions

    if hasattr(reductions, "_reduce_tensor_original"):
        return

    reductions._reduce_tensor_original = reductions.reduce_tensor
    reductions._rebuild_cuda_tensor_original = reductions.rebuild_cuda_tensor

    reductions.reduce_tensor = _reduce_tensor_modified
    reductions.rebuild_cuda_tensor = _rebuild_cuda_tensor_modified

    reductions.init_reductions()


# The signature has not been changed for years, and we will not need this when the next version is released,
# so it looks safe to use a constant.
_REDUCE_TENSOR_ARG_DEVICE_INDEX = 6


def _reduce_tensor_modified(*args, **kwargs):
    original_fn, original_args = (
        torch.multiprocessing.reductions._reduce_tensor_original(*args, **kwargs)
    )
    modified_args = list(original_args)
    modified_args[_REDUCE_TENSOR_ARG_DEVICE_INDEX] = _device_to_uuid(
        modified_args[_REDUCE_TENSOR_ARG_DEVICE_INDEX]
    )
    return original_fn, tuple(modified_args)


def _rebuild_cuda_tensor_modified(*args):
    args = list(args)
    args[_REDUCE_TENSOR_ARG_DEVICE_INDEX] = _device_from_uuid(
        args[_REDUCE_TENSOR_ARG_DEVICE_INDEX]
    )
    return torch.multiprocessing.reductions._rebuild_cuda_tensor_original(*args)


def _device_to_uuid(device: int) -> str:
    return str(torch.cuda.get_device_properties(device).uuid)


def _device_from_uuid(device_uuid: str) -> int:
    assert isinstance(
        device_uuid, str
    ), "The reduction function is probably not patched"
    for device in range(torch.cuda.device_count()):
        if str(torch.cuda.get_device_properties(device).uuid) == device_uuid:
            return device
    raise Exception("Invalid device_uuid=" + device_uuid)


def _modify_tuple(t, index: int, modifier: Callable):
    return *t[:index], modifier(t[index]), *t[index:]
