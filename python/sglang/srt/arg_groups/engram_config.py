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
    ALL = (NONE, PER_NODE)


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
class EngramConfig:
    placement: str = _choice(EngramPlacement.GPU, EngramPlacement.ALL)
    replication: str = _choice(EngramReplication.NONE, EngramReplication.ALL)
    lookup: str = _choice(EngramLookup.AUTO, EngramLookup.ALL)

    @classmethod
    def from_dict(cls, raw: Dict[str, Any]) -> "EngramConfig":
        if raw.get("lookup") == EngramLookup.COLLECTIVE:
            raw = {**raw, "lookup": EngramLookup.AUTO}
        choices = {f.name: f.metadata["choices"] for f in dataclasses.fields(cls)}
        invalid = [f"{k}={v!r}" for k, v in raw.items() if v not in choices.get(k, ())]
        if invalid:
            raise ValueError(
                f"--engram-config: invalid {', '.join(invalid)}; choices: {choices}"
            )
        config = cls(**raw)
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
