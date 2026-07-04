# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from dataclasses import dataclass

from sglang.multimodal_gen.configs.pipeline_configs.pi05 import Pi05PipelineConfig


@dataclass(frozen=True)
class Pi05ParallelTopology:
    prefix_strategy: str
    action_strategy: str
    layout_version: str

    @classmethod
    def from_config(cls, config: Pi05PipelineConfig) -> Pi05ParallelTopology:
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
                "Pi05 action expert should not share the prefix TP layout. "
                "Use SP, Ulysses, Ring, DP, or monolithic fallback for the "
                "action path."
            )
