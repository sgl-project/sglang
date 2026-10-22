#
# Copyright (c) 2026 Cambricon Technologies Co., Ltd. All Rights Reserved.
# This file is a part of the SGLang-MLU project.
#
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
#

from __future__ import annotations

import torch
import torch_mlu


def weak_ref_tensor(tensor: torch.Tensor) -> torch.Tensor:
    """Create a non-owning tensor view over an MLU allocation."""
    if tensor.device.type != "mlu":
        raise ValueError(f"weak_ref_tensor expects an MLU tensor, got {tensor.device}")
    if tensor.layout != torch.strided:
        raise ValueError(
            f"weak_ref_tensor expects a strided tensor, got {tensor.layout}"
        )

    storage = tensor.untyped_storage()
    metadata = {
        "nbytes": storage.nbytes(),
        "data_ptr": storage.data_ptr(),
        "size": tensor.shape,
        "stride": tensor.stride(),
        "dtype": tensor.dtype,
        "device": tensor.device,
        "storage_offset": tensor.storage_offset(),
    }
    weak_storage = torch._C._construct_storage_from_data_pointer(
        metadata["data_ptr"], metadata["device"], metadata["nbytes"]
    )
    return torch_mlu._MLUC._construct_MLU_Tensor_From_Storage_And_Metadata(
        metadata, weak_storage
    )
