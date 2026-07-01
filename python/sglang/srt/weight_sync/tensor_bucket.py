import math
from dataclasses import dataclass
from typing import List, Tuple, Union

import torch


@dataclass
class FlattenedTensorMetadata:
    """Metadata for a tensor in a flattened bucket"""

    name: str
    shape: torch.Size
    dtype: torch.dtype
    start_idx: int
    end_idx: int
    numel: int


class FlattenedTensorBucket:
    """
    A bucket that flattens multiple tensors into a single tensor for efficient processing
    while preserving all metadata needed for reconstruction.
    """

    # This field is solely for users of to check whether the class supports this feature
    supports_multi_dtypes = True

    def __init__(
        self,
        flattened_tensor: torch.Tensor,
        metadata: List[FlattenedTensorMetadata],
    ):
        """
        Initialize a tensor bucket from pre-flattened data.
        Args:
            flattened_tensor: Pre-flattened tensor (for reconstruction)
            metadata: Pre-computed metadata (for reconstruction)
        """
        # Initialize from pre-flattened data
        if flattened_tensor is None or metadata is None:
            raise ValueError("Must provide both flattened_tensor and metadata")
        self.flattened_tensor = flattened_tensor
        self.metadata = metadata

    def get_flattened_tensor(self) -> torch.Tensor:
        """Get the flattened tensor containing all bucket tensors"""
        return self.flattened_tensor

    def get_metadata(self) -> List[FlattenedTensorMetadata]:
        """Get metadata for all tensors in the bucket"""
        return self.metadata

    def reconstruct_tensors(self) -> List[Tuple[str, torch.Tensor]]:
        """
        Reconstruct original tensors from flattened tensor with optimized performance.
        Uses memory-efficient operations to minimize allocations and copies.
        """
        # preallocate the result list
        reconstructed = [None] * len(self.metadata)

        for i, meta in enumerate(self.metadata):
            tensor = (
                self.flattened_tensor[meta.start_idx : meta.end_idx]
                .view(meta.dtype)
                .reshape(meta.shape)
            )

            reconstructed[i] = (meta.name, tensor)

        return reconstructed

    @classmethod
    def from_tensors(cls, named_tensors: List[Tuple[str, torch.Tensor]]):
        # Create bucket from named tensors
        if not named_tensors:
            raise ValueError("Cannot create empty tensor bucket")

        current_idx = 0
        metadata: List[FlattenedTensorMetadata] = [None] * len(named_tensors)
        # Collect metadata and flatten tensors
        flattened_tensors: List[torch.Tensor] = [None] * len(named_tensors)

        for i, (name, tensor) in enumerate(named_tensors):
            flattened = tensor.flatten().view(torch.uint8)
            flattened_tensors[i] = flattened

            # Store metadata
            numel = flattened.numel()
            metadata_obj = FlattenedTensorMetadata(
                name=name,
                shape=tensor.shape,
                dtype=tensor.dtype,
                start_idx=current_idx,
                end_idx=current_idx + numel,
                numel=numel,
            )
            metadata[i] = metadata_obj
            current_idx += numel

        # Concatenate all flattened tensors
        flattened_tensor = torch.cat(flattened_tensors, dim=0)
        return cls(flattened_tensor=flattened_tensor, metadata=metadata)

    @classmethod
    def from_specs(
        cls,
        names: List[str],
        dtypes: List[Union[str, torch.dtype]],
        shapes: List[torch.Size],
        device: Union[torch.device, str],
    ):
        if not names or not (len(names) == len(dtypes) == len(shapes)):
            raise ValueError("Cannot create empty specs")
        current_idx = 0
        metadata: List[FlattenedTensorMetadata] = [None] * len(names)
        for i, (name, dtype, shape) in enumerate(zip(names, dtypes, shapes)):
            target_dtype = (
                dtype if isinstance(dtype, torch.dtype) else getattr(torch, dtype)
            )
            numel = math.prod(shape) * target_dtype.itemsize
            metadata_obj = FlattenedTensorMetadata(
                name=name,
                shape=shape,
                dtype=target_dtype,
                start_idx=current_idx,
                end_idx=current_idx + numel,
                numel=numel,
            )
            metadata[i] = metadata_obj
            current_idx += numel
        flattened_tensor = torch.empty(
            metadata[-1].end_idx, dtype=torch.uint8, device=device
        )
        return cls(flattened_tensor=flattened_tensor, metadata=metadata)
