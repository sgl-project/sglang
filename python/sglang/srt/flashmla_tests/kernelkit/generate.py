import torch


def _get_new_non_contiguous_tensor_shape(shape):
    """
    Get the expanded shape for a non-contiguous tensor.
    The last dimension is increased by 128 (for alignment), and all other dimensions are increased by 1
    """
    return [
        dim + 128 if dim_idx == len(shape) - 1 else dim + 1
        for dim_idx, dim in enumerate(shape)
    ]


def gen_non_contiguous_randn_tensor(shape, *args, **kwargs):
    new_shape = _get_new_non_contiguous_tensor_shape(shape)
    base_tensor = torch.randn(new_shape, *args, **kwargs)
    slices = [slice(0, dim) for dim in shape]
    return base_tensor[slices]


def gen_non_contiguous_tensor(shape, *args, **kwargs):
    new_shape = _get_new_non_contiguous_tensor_shape(shape)
    base_tensor = torch.empty(new_shape, *args, **kwargs)
    slices = [slice(0, dim) for dim in shape]
    return base_tensor[slices]


def non_contiguousify(tensor: torch.Tensor) -> torch.Tensor:
    new_tensor = gen_non_contiguous_tensor(
        tensor.shape, dtype=tensor.dtype, device=tensor.device
    )
    new_tensor[:] = tensor
    return new_tensor
