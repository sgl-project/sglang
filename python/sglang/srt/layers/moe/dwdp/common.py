"""Platform-neutral helpers shared by DWDP backends."""


def restore_storage_rank(original, expert_view):
    """Restore a gathered logical expert view to the layer's storage rank."""
    if original.ndim >= expert_view.ndim:
        return expert_view.contiguous()
    tail = tuple(original.shape[1:])
    tail_numel = 1
    for dim in tail:
        tail_numel *= dim
    if expert_view.numel() % tail_numel != 0:
        raise ValueError(
            f"Cannot restore logical shape {tuple(expert_view.shape)} to storage "
            f"tail {tail}"
        )
    return expert_view.reshape((-1,) + tail).contiguous()


def align_up(value: int, alignment: int) -> int:
    if alignment <= 0 or (alignment & (alignment - 1)) != 0:
        raise ValueError(f"alignment must be a positive power of 2, got {alignment}")
    return ((value + alignment - 1) // alignment) * alignment


def align_down(value: int, alignment: int) -> int:
    if alignment <= 0 or (alignment & (alignment - 1)) != 0:
        raise ValueError(f"alignment must be a positive power of 2, got {alignment}")
    return (value // alignment) * alignment
