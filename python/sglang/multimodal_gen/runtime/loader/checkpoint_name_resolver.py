# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from collections import defaultdict
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class SourceParamShard:
    source_name: str
    merge_index: int | None
    num_params_to_merge: int | None


class CheckpointNameResolver:
    """Small wrapper around checkpoint-to-runtime parameter name mapping."""

    def __init__(self, mapping_fn: Callable[[str], tuple[str, Any, Any]]):
        self.mapping_fn = mapping_fn
        self.reverse_param_names_mapping: dict[str, tuple[str, Any, Any]] = {}
        self.source_param_shards_by_target: dict[str, list[SourceParamShard]] = (
            defaultdict(list)
        )
        self._fused_sources_by_target: dict[str, dict[int, str]] = defaultdict(dict)
        self._fused_expected_by_target: dict[str, int] = {}

    def map_source_param(self, source_name: str) -> tuple[str, Any, Any]:
        return self.mapping_fn(source_name)

    def record_source_param(
        self,
        source_name: str,
        target_name: str,
        merge_index: int | None,
        num_params_to_merge: int | None,
    ) -> None:
        # Legacy single-entry mapping kept for backward compatibility.
        # Use source_param_shards_by_target when all source shards are needed.
        self.reverse_param_names_mapping[target_name] = (
            source_name,
            merge_index,
            num_params_to_merge,
        )
        self.source_param_shards_by_target[target_name].append(
            SourceParamShard(source_name, merge_index, num_params_to_merge)
        )
        if merge_index is None:
            return
        self._fused_sources_by_target[target_name][merge_index] = source_name
        self._fused_expected_by_target[target_name] = int(num_params_to_merge)

    def validate_complete_fused_params(self) -> None:
        incomplete: list[str] = []
        for target_name, expected in sorted(self._fused_expected_by_target.items()):
            seen = self._fused_sources_by_target.get(target_name, {})
            if len(seen) == expected:
                continue
            missing = [idx for idx in range(expected) if idx not in seen]
            present = ", ".join(f"{idx}:{seen[idx]}" for idx in sorted(seen))
            incomplete.append(
                f"{target_name} missing shard(s) {missing}; present [{present}]"
            )
        if incomplete:
            raise ValueError(
                "Incomplete fused checkpoint parameter merge: " + "; ".join(incomplete)
            )

    def metadata_key_map(
        self,
        source_keys: Iterable[str],
        suffixes: Iterable[str],
    ) -> dict[str, dict[str, str]]:
        suffix_set = set(suffixes)
        maps: dict[str, dict[str, str]] = {suffix: {} for suffix in suffix_set}
        for key in source_keys:
            for suffix in suffix_set:
                marker = f".{suffix}"
                if not key.endswith(marker):
                    continue
                source_module = key[: -len(marker)]
                # Keep checkpoint-side lookup available for callers that already
                # have source names; the first key wins if duplicates appear.
                maps[suffix].setdefault(source_module, key)
                for param_suffix in ("weight", "bias"):
                    target_name, _, _ = self.map_source_param(
                        f"{source_module}.{param_suffix}"
                    )
                    if not target_name:
                        continue
                    runtime_module = target_name.rsplit(".", 1)[0]
                    maps[suffix].setdefault(runtime_module, key)
        return maps
