# Copyright (c) 2026 by FlashInfer team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Framework-side facilities shared by production Kernel wrappers."""

import cuda.bindings.driver as cuda
import cutlass.cute as cute
import torch
from cutlass.cute.runtime import from_dlpack, make_fake_compact_tensor


class _GraphSafeDLPack:
    __slots__ = ("tensor",)

    def __init__(self, tensor: torch.Tensor) -> None:
        self.tensor = tensor

    def __dlpack__(self, stream=None):
        # stream=-1 skips producer sync; CuTe launches on the current captured stream.
        return self.tensor.__dlpack__(stream=-1)

    def __dlpack_device__(self):
        return self.tensor.__dlpack_device__()


def to_cute(tensor: torch.Tensor, alignment: int) -> cute.Tensor:
    return from_dlpack(
        _GraphSafeDLPack(tensor.detach()),
        assumed_align=alignment,
    )


def to_cute_dynamic(
    tensor: torch.Tensor,
    alignment: int,
    *,
    divisibility: int,
) -> cute.Tensor:
    return to_cute(tensor, alignment).mark_compact_shape_dynamic(
        mode=0,
        divisibility=divisibility,
    )


def make_fake_dynamic_compact_tensor(
    dtype,
    *,
    alignment: int,
    divisibility: int,
) -> cute.Tensor:
    return make_fake_compact_tensor(
        dtype,
        (cute.sym_int32(divisibility=divisibility),),
        assumed_align=alignment,
    )


def current_cu_stream() -> cuda.CUstream:
    return cuda.CUstream(torch.cuda.current_stream().cuda_stream)
