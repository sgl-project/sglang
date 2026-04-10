import torch
import torch.distributed as dist
import torch.nn.functional as F

from sglang.multimodal_gen.runtime.distributed import parallel_state as _ps
from sglang.multimodal_gen.runtime.utils.logging_utils import init_logger

logger = init_logger(__name__)


def get_vae_group():
    if not dist.is_available() or not dist.is_initialized():
        return None
    if getattr(_ps, "_VAE", None) is None:
        # Fallback to global group if no explicit VAE group is created.
        if not getattr(get_vae_group, "_warned_fallback", False):
            logger.warning(
                "VAE group not initialized; falling back to global process group."
            )
            get_vae_group._warned_fallback = True
        return dist.group.WORLD
    return _ps.get_vae_parallel_group()


def get_vae_parallel_world_size() -> int:
    group = get_vae_group()
    if group is None:
        return 1
    return dist.get_world_size(group=group)


def get_vae_parallel_rank() -> int:
    group = get_vae_group()
    if group is None:
        return 0
    return dist.get_rank(group=group)


def _pad_on_dim(x: torch.Tensor, dim: int, pad_size: int) -> torch.Tensor:
    if pad_size <= 0:
        return x
    pad = [0] * (2 * x.dim())
    pad[(x.dim() - 1 - dim) * 2 + 1] = pad_size
    return F.pad(x, pad)


def _normalize_dim(dim: int, ndim: int) -> int:
    if dim < 0:
        dim += ndim
    if dim < 0 or dim >= ndim:
        raise IndexError(f"dim {dim} out of range for tensor with {ndim} dims")
    return dim


def _get_shard_range(length: int, world_size: int, rank: int) -> tuple[int, int]:
    base = length // world_size
    remainder = length % world_size
    start = rank * base + min(rank, remainder)
    end = start + base + (1 if rank < remainder else 0)
    return start, end


def _slice_on_dim(x: torch.Tensor, dim: int, start: int, end: int) -> torch.Tensor:
    slices = [slice(None)] * x.dim()
    slices[dim] = slice(start, end)
    return x[tuple(slices)].contiguous()


def gather_tensor(
    x: torch.Tensor,
    dim: int = 3,
    sizes: list[int] | None = None,
    return_sizes: bool = False,
) -> torch.Tensor | tuple[torch.Tensor, list[int] | None]:
    group = get_vae_group()
    world_size = get_vae_parallel_world_size()
    if world_size <= 1 or group is None:
        return (x, sizes) if return_sizes else x

    dim = _normalize_dim(dim, x.dim())

    if sizes is None:
        device = x.device
        size_value = torch.tensor([x.shape[dim]], device=device, dtype=torch.int64)
        size_list = [torch.zeros_like(size_value) for _ in range(world_size)]
        dist.all_gather(size_list, size_value, group=group)
        sizes = [int(s.item()) for s in size_list]

    max_size = max(sizes)
    if x.shape[dim] < max_size:
        x = _pad_on_dim(x, dim, max_size - x.shape[dim])

    shards = [torch.empty_like(x) for _ in range(world_size)]
    dist.all_gather(shards, x.contiguous(), group=group)
    trimmed = [_slice_on_dim(t, dim, 0, sizes[i]) for i, t in enumerate(shards)]
    gathered = torch.cat(trimmed, dim=dim)
    return (gathered, sizes) if return_sizes else gathered


def split_tensor(
    x: torch.Tensor,
    dim: int = 3,
    sizes: list[int] | None = None,
    return_sizes: bool = False,
) -> torch.Tensor | tuple[torch.Tensor, list[int] | None]:
    group = get_vae_group()
    rank = get_vae_parallel_rank()
    world_size = get_vae_parallel_world_size()
    if world_size <= 1 or group is None:
        return (x, sizes) if return_sizes else x

    dim = _normalize_dim(dim, x.dim())
    if sizes is None:
        start, end = _get_shard_range(x.shape[dim], world_size, rank)
        length = x.shape[dim]
        base = length // world_size
        remainder = length % world_size
        sizes = [base + (1 if i < remainder else 0) for i in range(world_size)]
    else:
        start = sum(sizes[:rank])
        end = start + sizes[rank]
    shard = _slice_on_dim(x, dim, start, end)
    return (shard, sizes) if return_sizes else shard
