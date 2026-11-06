import torch


def pack_qkv_destination_major_cpu(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    world_size: int,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    """Pack CPU Q/K/V tensors into destination-major Ulysses layout."""
    if q.ndim != 3 or world_size < 1:
        raise ValueError("q must be 3D and world_size must be positive")

    rows, global_heads, head_size = q.shape
    expected_shape = (
        world_size,
        rows,
        global_heads // world_size,
        3 * head_size,
    )
    if out is None:
        out = torch.empty(expected_shape, dtype=q.dtype, device=q.device)

    import sgl_kernel  # noqa: F401

    return torch.ops.sgl_kernel.pack_qkv_destination_major_cpu(
        q, k, v, world_size, out
    )