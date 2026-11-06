"""Group metadata fixtures for tests that do not initialize distributed groups."""

from contextlib import contextmanager
from types import SimpleNamespace

from sglang.srt.runtime_context import SpawnRanks, get_parallel
from sglang.srt.runtime_context import publish as publish_context


def _group_metadata(values):
    parallel = get_parallel()
    for name in ("tp", "attn_tp"):
        rank_key, size_key, group_key = (
            f"{name}_rank",
            f"{name}_size",
            f"{name}_group",
        )
        rank = values.get(rank_key, getattr(parallel, rank_key))
        size = values.get(size_key, getattr(parallel, size_key))
        if group_key not in values and (rank_key in values or size_key in values):
            values[group_key] = SimpleNamespace(rank_in_group=rank, world_size=size)
        group = values.get(group_key)
        if group is not None:
            # Execution mocks also need placement metadata when used to build layers.
            if not isinstance(getattr(group, "rank_in_group", None), int):
                group.rank_in_group = rank
            if not isinstance(getattr(group, "world_size", None), int):
                group.world_size = size
    return values


def publish(*args, **kwargs):
    if kwargs.get("ranks") is None:
        kwargs["ranks"] = SpawnRanks(world_rank=0)
    context = publish_context(*args, **kwargs)
    parallel = get_parallel()
    groups = {
        f"{name}_group": SimpleNamespace(
            rank_in_group=getattr(parallel, f"{name}_rank"),
            world_size=getattr(parallel, f"{name}_size"),
        )
        for name in ("tp", "attn_tp")
    }
    parallel.override_permanently(**groups)
    return context


@contextmanager
def parallel_scope(**values):
    with get_parallel().override(**_group_metadata(values)) as parallel:
        yield parallel


def rank_size(layer, *, kv=False):
    """Read the retained query or KV partition used by a native linear layer."""
    group = layer.kv_tp_group if kv else layer.tp_group
    return (0, 1) if group is None else (group.rank_in_group, group.world_size)
