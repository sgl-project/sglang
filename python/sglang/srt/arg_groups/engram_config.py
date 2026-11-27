"""Values of --engram-config; its help says what each one means. Stdlib only:
server-argument resolution imports it without torch."""

import dataclasses
from typing import Any, Dict, Optional


class EngramPlacement:
    GPU = "gpu"
    CPU = "cpu"
    ALL = (GPU, CPU)


class EngramReplication:
    NONE = "none"
    PER_NODE = "per_node"
    GROUP_SHARDS = "group_shards"
    ALL = (NONE, PER_NODE, GROUP_SHARDS)


class EngramLookup:
    AUTO = "auto"
    ALL_REDUCE = "all_reduce"
    REDUCE_SCATTER = "reduce_scatter"
    ALL_TO_ALL = "all_to_all"
    PEER_READ = "peer_read"
    ALL = (AUTO, ALL_REDUCE, REDUCE_SCATTER, ALL_TO_ALL, PEER_READ)
    # Read as AUTO.
    COLLECTIVE = "collective"


def cols_per_rank(columns: int, tp_size: int) -> Optional[int]:
    """Hash columns each TP rank owns when they divide evenly, else None."""
    return columns // tp_size if columns % tp_size == 0 else None


def _choice(default: str, choices: tuple):
    return dataclasses.field(default=default, metadata={"choices": choices})


@dataclasses.dataclass(frozen=True)
class EngramPoolConfig:
    store_url: str = ""
    world_size: int = 0
    rank_offset: int = 0
    owner_ranks: Optional[list[int]] = None

    @classmethod
    def from_dict(cls, raw: Dict[str, Any]) -> "EngramPoolConfig":
        fields = {f.name: f for f in dataclasses.fields(cls)}
        if not isinstance(raw, dict) or any(
            k not in fields
            or (
                not isinstance(v, list) or any(type(r) is not int for r in v)
                if k == "owner_ranks"
                else type(v) is not type(fields[k].default)
            )
            for k, v in raw.items()
        ):
            raise ValueError("--engram-config: invalid pool fields or types")
        config = cls(**raw)
        if config.world_size < 0 or config.rank_offset < 0:
            raise ValueError(
                "--engram-config: pool world_size/rank_offset must be nonnegative"
            )
        if config.world_size and config.rank_offset >= config.world_size:
            raise ValueError(
                "--engram-config: pool rank_offset must be below world_size"
            )
        owners = config.owner_ranks
        if owners is not None and (
            not owners
            or len(set(owners)) != len(owners)
            or any(
                r < 0 or (config.world_size and r >= config.world_size) for r in owners
            )
        ):
            raise ValueError(
                "--engram-config: pool owner_ranks must be unique ranks within world_size"
            )
        return config


@dataclasses.dataclass(frozen=True)
class EngramConfig:
    placement: str = _choice(EngramPlacement.GPU, EngramPlacement.ALL)
    replication: str = _choice(EngramReplication.NONE, EngramReplication.ALL)
    lookup: str = _choice(EngramLookup.AUTO, EngramLookup.ALL)
    # Replication group_shards only: the GPU count of the servers sharing the
    # tables, and a directory all of them see, empty for each deployment.
    group_size: int = 0
    group_store: str = ""
    pool: Optional[EngramPoolConfig] = None

    @classmethod
    def from_dict(cls, raw: Dict[str, Any]) -> "EngramConfig":
        raw = dict(raw)
        pool = EngramPoolConfig.from_dict(raw.pop("pool")) if "pool" in raw else None
        if raw.get("lookup") == EngramLookup.COLLECTIVE:
            raw = {**raw, "lookup": EngramLookup.AUTO}
        fields = {f.name: f for f in dataclasses.fields(cls)}
        choices = {n: f.metadata["choices"] for n, f in fields.items() if f.metadata}
        invalid = [
            f"{k}={v!r}"
            for k, v in raw.items()
            if k not in fields
            or (k in choices and v not in choices[k])
            or (k not in choices and type(v) is not type(fields[k].default))
        ]
        if invalid:
            raise ValueError(
                f"--engram-config: invalid {', '.join(invalid)}; choices: {choices}"
            )
        config = cls(**raw, pool=pool)
        if pool is not None:
            if config.placement != EngramPlacement.CPU:
                raise ValueError("--engram-config: pool requires NPU host offload")
            if config.replication == EngramReplication.GROUP_SHARDS or (
                config.replication == EngramReplication.PER_NODE
                and (pool.world_size or pool.rank_offset)
            ):
                raise ValueError(
                    "--engram-config: an explicit pool topology requires replication 'none'"
                )
        if config.replication == EngramReplication.GROUP_SHARDS:
            if config.lookup not in (EngramLookup.AUTO, EngramLookup.PEER_READ):
                raise ValueError(
                    "--engram-config: replication 'group_shards' reads the shards "
                    "in place; lookup must be 'auto' or 'peer_read'."
                )
            if config.group_size <= 0 or not config.group_store:
                raise ValueError(
                    "--engram-config: replication 'group_shards' needs 'group_size', "
                    "the GPU count of the servers sharing the tables, and "
                    "'group_store', a directory all of them see, empty for each "
                    "deployment."
                )
            return dataclasses.replace(config, lookup=EngramLookup.PEER_READ)
        if config.group_size or config.group_store:
            raise ValueError(
                "--engram-config: 'group_size' and 'group_store' apply to "
                "replication 'group_shards' only."
            )
        if (
            config.lookup == EngramLookup.PEER_READ
            and config.placement != EngramPlacement.GPU
        ):
            raise ValueError(
                "--engram-config: lookup 'peer_read' reads GPU shards; it needs "
                "placement 'gpu'."
            )
        if (
            config.replication == EngramReplication.PER_NODE
            and config.placement != EngramPlacement.CPU
        ):
            raise ValueError(
                "--engram-config: replication 'per_node' needs placement 'cpu'."
            )
        return config
