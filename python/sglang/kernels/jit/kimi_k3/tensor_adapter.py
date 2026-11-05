# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the SGLang project
"""SGLang-owned Torch-to-CuTe tensor adapter for graph-compatible launches."""

import cutlass.cute as cute
import torch
from cutlass.cute.runtime import from_dlpack


class CUDAGraphCompatibleWrapper:
    """DLPack view which deliberately avoids producer-stream synchronization."""

    def __init__(self, tensor: torch.Tensor):
        self.tensor = tensor

    def __dlpack__(self, stream=None):
        return self.tensor.__dlpack__(stream=-1)

    def __dlpack_device__(self):
        return self.tensor.__dlpack_device__()


def to_cute(tensor: torch.Tensor, assumed_align: int = 16) -> cute.Tensor:
    return from_dlpack(
        CUDAGraphCompatibleWrapper(tensor.detach()), assumed_align=assumed_align
    )
