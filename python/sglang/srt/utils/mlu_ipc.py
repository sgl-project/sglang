# Copyright 2023-2026 SGLang Team
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
"""Same-host MLU IPC for ``update_weights_from_tensor``.

The platform initializes this path in standalone senders and receivers without
loading general plugins. Exporting tensors must remain alive until receivers
finish using them. This is separate from CheckpointEngine's
``update_weights_from_ipc`` protocol.
"""

from multiprocessing.reduction import ForkingPickler
from typing import Callable

import torch


def _device_from_uuid(device_uuid: str) -> int:
    for device in range(torch.mlu.device_count()):
        if str(torch.mlu.get_device_properties(device).uuid) == device_uuid:
            return device
    raise RuntimeError(f"MLU IPC device {device_uuid!r} is not visible to this process")


def _rebuild_mlu_tensor_modified(device_uuid: str, tensor_args: tuple):
    from torch_mlu.mlu.reductions import rebuild_mlu_tensor

    # torch_mlu stores the exporting process's device index at slot 6. Resolve
    # the physical card under the receiving process's visible-device ordering.
    device = _device_from_uuid(device_uuid)
    return rebuild_mlu_tensor(*tensor_args[:6], device, *tensor_args[7:])


def reduce_tensor(tensor):
    from torch_mlu.mlu.reductions import _reduce_tensor, rebuild_mlu_tensor

    rebuild, args = _reduce_tensor(tensor)
    if rebuild is not rebuild_mlu_tensor:
        return rebuild, args
    device_uuid = str(torch.mlu.get_device_properties(tensor.device.index).uuid)
    return _rebuild_mlu_tensor_modified, (device_uuid, args)


def register_mlu_reductions() -> None:
    """Register MLU reducers without modifying torch's CUDA reduction functions."""
    ForkingPickler.register(torch.Tensor, reduce_tensor)
    ForkingPickler.register(torch.nn.Parameter, reduce_tensor)


def get_tensor_ipc_rebuild_fn(
    module: str, name: str
) -> Callable[..., torch.Tensor] | None:
    """Resolve only the UUID-aware and native MLU tensor rebuild functions."""
    if (module, name) == (__name__, "_rebuild_mlu_tensor_modified"):
        return _rebuild_mlu_tensor_modified
    if (module, name) == ("torch_mlu.mlu.reductions", "rebuild_mlu_tensor"):
        from torch_mlu.mlu.reductions import rebuild_mlu_tensor

        return rebuild_mlu_tensor
    return None
