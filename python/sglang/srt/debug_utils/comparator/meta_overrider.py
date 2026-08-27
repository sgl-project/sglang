"""Meta overrider: replace metadata fields without re-running dumps.

Currently only overrides 'dims', but the design supports overriding
additional meta fields (e.g. parallel_info) in the future.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Literal, Optional

import yaml

from sglang.srt.debug_utils.comparator.utils import _StrictBase


class MetaOverrideRule(_StrictBase):
    """Single override rule: regex match on tensor name → replacement meta field(s).

    Currently only 'dims' is supported; more fields may be added in the future.
    """

    match: str
    dims: str
    side: Literal["both", "baseline", "target"] = "both"


class MetaOverrideConfig(_StrictBase):
    """YAML top-level config for overriding comparator behavior."""

    overrides: list[MetaOverrideRule] = []


class MetaOverrider:
    """Holds override rules and applies first-match-wins replacement."""

    def __init__(self, rules: list[MetaOverrideRule]) -> None:
        self._rules: list[MetaOverrideRule] = rules

    @property
    def is_empty(self) -> bool:
        return len(self._rules) == 0

    @classmethod
    def from_args_and_config(
        cls,
        *,
        override_dims: list[str],
        override_baseline_dims: list[str],
        override_target_dims: list[str],
        override_config: Optional[Path],
    ) -> "MetaOverrider":
        per_side_args: list[tuple[list[str], Literal["both", "baseline", "target"]]] = [
            (override_dims, "both"),
            (override_baseline_dims, "baseline"),
            (override_target_dims, "target"),
        ]
        cli_rules: list[MetaOverrideRule] = [
            MetaOverrideRule(match=name, dims=dims_str, side=side)
            for raw_args, side in per_side_args
            for name, dims_str in [_parse_cli_override_arg(raw) for raw in raw_args]
        ]

        yaml_rules: list[MetaOverrideRule] = (
            _load_yaml_rules(override_config) if override_config is not None else []
        )

        return cls(rules=cli_rules + yaml_rules)

    def apply_to_meta(
        self,
        *,
        name: str,
        meta: dict[str, Any],
        side: Literal["baseline", "target"],
    ) -> dict[str, Any]:
        """First-match-wins: return meta with dims replaced by the first matching rule for this side."""
        for rule in self._rules:
            if rule.side not in ("both", side):
                continue
            if re.search(rule.match, name):
                return {**meta, "dims": rule.dims}

        return meta


def _parse_cli_override_arg(raw: str) -> tuple[str, str]:
    """Parse 'name:dims_string' from a CLI --override-* argument."""
    parts: list[str] = raw.split(":", maxsplit=1)
    if len(parts) != 2 or not parts[0].strip() or not parts[1].strip():
        raise ValueError(
            f"Invalid override format: {raw!r}; expected 'name:dims_string'"
        )
    return parts[0].strip(), parts[1].strip()


def _load_yaml_rules(path: Path) -> list[MetaOverrideRule]:
    """Load override rules from a YAML config file."""
    with open(path) as f:
        raw_data: Any = yaml.safe_load(f)

    if raw_data is None:
        return []

    config: MetaOverrideConfig = MetaOverrideConfig.model_validate(raw_data)
    return config.overrides
