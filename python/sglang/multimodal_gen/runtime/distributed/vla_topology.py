# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class VLAParallelTopology:
    prefix_strategy: str
    action_strategy: str
    layout_version: str

    @classmethod
    def from_config(cls, config: Any) -> VLAParallelTopology:
        return cls(
            prefix_strategy=config.prefix_parallel_strategy,
            action_strategy=config.action_parallel_strategy,
            layout_version=config.parallel_layout_version,
        )

    @property
    def cache_layout_tag(self) -> str:
        return (
            f"prefix={self.prefix_strategy};"
            f"action={self.action_strategy};"
            f"version={self.layout_version}"
        )

    def validate(self) -> None:
        if self.prefix_strategy == self.action_strategy == "tp":
            raise ValueError(
                "VLA action expert should not share the prefix TP layout. "
                "Use SP, Ulysses, Ring, DP, or monolithic fallback for the "
                "action path."
            )
