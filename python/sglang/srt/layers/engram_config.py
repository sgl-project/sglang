"""--engram-config: where the DeepSeek-V4.1 Engram tables live and how a rank
reads rows that another rank holds.

Stdlib only, so server-argument resolution can import it without torch.
"""

import dataclasses
from typing import Any, Dict, Optional

# Each key's allowed values; the first is the default. The --engram-config help
# (arg_groups/fields/exec_.py) says what each one means.
_CHOICES = {
    "placement": ("gpu", "cpu"),
    "replication": ("none", "per_node"),
    "lookup": ("collective", "peer_read"),
}


@dataclasses.dataclass(frozen=True)
class EngramConfig:
    placement: str = _CHOICES["placement"][0]
    replication: str = _CHOICES["replication"][0]
    lookup: str = _CHOICES["lookup"][0]

    @classmethod
    def from_dict(cls, raw: Optional[Dict[str, Any]]) -> "EngramConfig":
        """Validate a (partial) --engram-config; defaults fill the rest."""
        raw = {} if raw is None else raw
        if not isinstance(raw, dict):
            raise ValueError("--engram-config must be a JSON object")
        for key, value in raw.items():
            if key not in _CHOICES:
                raise ValueError(
                    f"--engram-config: unknown key {key!r}, expected one of "
                    f"{tuple(_CHOICES)}"
                )
            if value not in _CHOICES[key]:
                raise ValueError(
                    f"--engram-config: {key}={value!r}, expected one of {_CHOICES[key]}"
                )
        config = cls(**raw)
        if config.replication == "per_node" and config.placement != "cpu":
            raise ValueError(
                "--engram-config: replication 'per_node' needs placement 'cpu'"
            )
        if config.replication != "none" and "lookup" in raw:
            raise ValueError(
                "--engram-config: lookup applies only with replication 'none'; "
                "a full copy holds every row"
            )
        if config.lookup == "peer_read" and config.placement != "gpu":
            raise ValueError(
                "--engram-config: lookup 'peer_read' reads the peers' GPU shards "
                "and needs placement 'gpu'"
            )
        return config

    def to_dict(self) -> Dict[str, str]:
        return dataclasses.asdict(self)

    def __str__(self) -> str:
        if self.replication != "none":
            return f"placement={self.placement}, replication={self.replication}"
        return f"placement={self.placement}, lookup={self.lookup}"
