"""--engram-config: where the DeepSeek-V4.1 Engram tables live and how a rank
reads rows that another rank holds. The --engram-config help says what each
value means.

Stdlib only, so server-argument resolution can import it without torch.
"""

import dataclasses
from typing import Any, Dict


class EngramPlacement:
    GPU = "gpu"
    CPU = "cpu"
    ALL = (GPU, CPU)


class EngramReplication:
    NONE = "none"
    PER_NODE = "per_node"
    ALL = (NONE, PER_NODE)


class EngramLookup:
    COLLECTIVE = "collective"
    PEER_READ = "peer_read"
    ALL = (COLLECTIVE, PEER_READ)


def _choice(default: str, choices: tuple):
    return dataclasses.field(default=default, metadata={"choices": choices})


@dataclasses.dataclass(frozen=True)
class EngramConfig:
    placement: str = _choice(EngramPlacement.GPU, EngramPlacement.ALL)
    replication: str = _choice(EngramReplication.NONE, EngramReplication.ALL)
    lookup: str = _choice(EngramLookup.COLLECTIVE, EngramLookup.ALL)

    @classmethod
    def from_dict(cls, raw: Dict[str, Any]) -> "EngramConfig":
        """Validate a partial --engram-config; omitted keys keep their defaults."""
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
