from dataclasses import dataclass
from typing import List, Tuple

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
        named_tensors: List[Tuple[str, torch.Tensor]] = None,
        flattened_tensor: torch.Tensor = None,
        metadata: List[FlattenedTensorMetadata] = None,
    ):
        """
        Initialize a tensor bucket from a list of named tensors OR from pre-flattened data.
        Args:
            named_tensors: List of (name, tensor) tuples (for creating new bucket)
            flattened_tensor: Pre-flattened tensor (for reconstruction)
            metadata: Pre-computed metadata (for reconstruction)
        """
        if named_tensors is not None:
            # Create bucket from named tensors
            self.metadata: List[FlattenedTensorMetadata] = [None] * len(named_tensors)
            self.flattened_tensor: torch.Tensor = None

            if not named_tensors:
                raise ValueError("Cannot create empty tensor bucket")

            # Collect metadata and flatten tensors
            current_idx = 0
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
                self.metadata[i] = metadata_obj
                current_idx += numel

            # Concatenate all flattened tensors
            self.flattened_tensor = torch.cat(flattened_tensors, dim=0)
        else:
            # Initialize from pre-flattened data
            if flattened_tensor is None or metadata is None:
                raise ValueError(
                    "Must provide either named_tensors or both flattened_tensor and metadata"
                )
            self.flattened_tensor = flattened_tensor
            self.metadata = metadata

    def get_flattened_tensor(self) -> torch.Tensor:
        """Get the flattened tensor containing all bucket tensors"""
        return self.flattened_tensor

    @classmethod
    def empty(cls, names, dtypes, shapes, device):
        """Allocate a receive bucket without constructing or copying its tensors."""
        metadata = []
        offset = 0
        for name, dtype, shape in zip(names, dtypes, shapes, strict=True):
            dtype = dtype if isinstance(dtype, torch.dtype) else getattr(torch, dtype)
            shape = torch.Size(shape)
            if any(dim < 0 for dim in shape):
                raise ValueError(
                    f"Negative dimension in received tensor {name}: {shape}"
                )
            if offset % dtype.itemsize:
                raise ValueError(
                    f"Unaligned received tensor {name}: byte offset {offset}, dtype {dtype}"
                )
            size = shape.numel() * dtype.itemsize
            metadata.append(
                FlattenedTensorMetadata(name, shape, dtype, offset, offset + size, size)
            )
            offset += size
        if not metadata:
            raise ValueError("Cannot create empty tensor bucket")
        return cls(
            flattened_tensor=torch.empty(offset, dtype=torch.uint8, device=device),
            metadata=metadata,
        )

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
