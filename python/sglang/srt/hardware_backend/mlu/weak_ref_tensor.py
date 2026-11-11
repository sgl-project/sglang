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
