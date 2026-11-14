# SPDX-License-Identifier: Apache-2.0
"""Per-layer format selection for online diffusion quantization.

``--quantization <format>`` gives every linear in a transformer the same
format. This config keeps that as the base format and routes named layers to a
different one, so a model can run most of its blocks in a narrow format while
the layers carrying the most quality run in a wider one.

A rule is ``<pattern>=<format>``. Patterns match against the layer prefix
(``blocks.12.ffn.fc_out``): a pattern without a glob character matches on
dotted path boundaries, the match ``--quantization-ignored-layers`` uses, and
a pattern holding ``*``, ``?`` or ``[`` is matched as a glob against the whole
prefix. Rules are tried in order and the first match wins, so an explicit rule
placed ahead of a broader one overrides it. A layer no rule claims takes the
base format, and an ignored layer stays unquantized whichever rule claims it.

Routing covers only formats quantized after load, since a pre-quantized
checkpoint fixes each tensor's format on disk.
"""

from __future__ import annotations

import logging
from collections import Counter
from fnmatch import fnmatchcase

import msgspec
import torch

from sglang.multimodal_gen.runtime.layers.quantization.configs.base_config import (
    QuantizationConfig,
    QuantizeMethodBase,
)
from sglang.srt.layers.quantization.utils import is_layer_skipped

logger = logging.getLogger(__name__)

_GLOB_CHARS = ("*", "?", "[")


class PrecisionRule(msgspec.Struct, frozen=True):
    """One ``<pattern>=<format>`` routing rule."""

    pattern: str
    format: str


def _is_glob(pattern: str) -> bool:
    return any(char in pattern for char in _GLOB_CHARS)


def _matches(pattern: str, prefix: str, fused_mapping: dict[str, list[str]]) -> bool:
    if _is_glob(pattern):
        return fnmatchcase(prefix, pattern)
    return is_layer_skipped(prefix, [pattern], fused_mapping=fused_mapping)


def parse_precision_rules(
    entries: list[str],
    *,
    supported_formats: frozenset[str],
) -> tuple[PrecisionRule, ...]:
    """Turn ``--quantization-overrides`` entries into ordered routing rules."""
    rules: list[PrecisionRule] = []
    for entry in entries:
        pattern, separator, format_name = (
            part.strip() for part in entry.partition("=")
        )
        if not separator or not pattern or not format_name:
            raise ValueError(
                f"--quantization-overrides entry {entry!r} must be written as "
                "<layer-pattern>=<format>"
            )
        format_name = format_name.lower()
        if format_name not in supported_formats:
            raise ValueError(
                f"--quantization-overrides names format {format_name!r}, which is "
                "not quantized after load; online formats are "
                f"{', '.join(sorted(supported_formats))}"
            )
        rules.append(PrecisionRule(pattern=pattern, format=format_name))
    return tuple(dict.fromkeys(rules))


class MixedPrecisionConfig(QuantizationConfig):
    """Hands each linear to the sub-config its first matching rule names."""

    def __init__(
        self,
        *,
        base_format: str,
        rules: tuple[PrecisionRule, ...],
        ignored_layers: list[str] | None = None,
        packed_modules_mapping: dict[str, list[str]] | None = None,
    ) -> None:
        super().__init__()
        from sglang.multimodal_gen.runtime.layers.quantization import (
            get_quantization_config,
        )

        self.base_format = base_format
        self.rules = rules
        self.ignored_layers = ignored_layers or []
        self.packed_modules_mapping = packed_modules_mapping or {}
        # Each sub-config takes its own copy of the ignored list, since a config
        # may extend the list it is given with its own environment overrides.
        self._configs: dict[str, QuantizationConfig] = {
            name: get_quantization_config(name)(
                ignored_layers=list(self.ignored_layers),
                packed_modules_mapping=dict(self.packed_modules_mapping),
            )
            for name in dict.fromkeys([base_format] + [rule.format for rule in rules])
        }
        self._layer_formats: Counter[str] = Counter()
        self._rule_hits: Counter[int] = Counter()

    @classmethod
    def get_name(cls) -> str:
        return "mixed_precision"

    @classmethod
    def get_supported_act_dtypes(cls) -> list[torch.dtype]:
        return [torch.bfloat16, torch.float16]

    @classmethod
    def get_min_capability(cls) -> int:
        # Each sub-config screens its own hardware when its method is built.
        return 80

    @classmethod
    def get_config_filenames(cls) -> list[str]:
        return []

    @classmethod
    def from_config(cls, config: dict) -> MixedPrecisionConfig:
        raise NotImplementedError(
            "mixed precision is selected with --quantization-overrides, not "
            "declared by a checkpoint"
        )

    def _resolve(self, prefix: str) -> tuple[str, int | None]:
        """Format for this layer, and the index of the rule that chose it."""
        for index, rule in enumerate(self.rules):
            if _matches(rule.pattern, prefix, self.packed_modules_mapping):
                return rule.format, index
        return self.base_format, None

    def format_for(self, prefix: str) -> str:
        return self._resolve(prefix)[0]

    def get_quant_method(
        self, layer: torch.nn.Module, prefix: str
    ) -> QuantizeMethodBase | None:
        from sglang.multimodal_gen.runtime.layers.linear import (
            LinearBase,
            UnquantizedLinearMethod,
        )

        format_name, rule_index = self._resolve(prefix)
        method = self._configs[format_name].get_quant_method(layer, prefix)
        if isinstance(layer, LinearBase):
            if isinstance(method, UnquantizedLinearMethod):
                self._layer_formats["unquantized"] += 1
            else:
                self._layer_formats[format_name] += 1
                if rule_index is not None:
                    self._rule_hits[rule_index] += 1
        return method

    def supports_input_partition(
        self, prefix: str, input_size_per_partition: int
    ) -> bool:
        return self._configs[self.format_for(prefix)].supports_input_partition(
            prefix, input_size_per_partition
        )

    def quantizes_embedding(self, prefix: str) -> bool:
        return self._configs[self.format_for(prefix)].quantizes_embedding(prefix)

    def log_resolved_plan(self) -> None:
        """Report the format every linear actually received.

        Every rank resolves the same plan, so only one of them reports it.
        """
        if torch.distributed.is_initialized() and torch.distributed.get_rank() != 0:
            return
        total = sum(self._layer_formats.values())
        breakdown = ", ".join(
            f"{name}={count}" for name, count in sorted(self._layer_formats.items())
        )
        logger.info(
            "Mixed-precision quantization: base=%s, %d linear layers (%s)",
            self.base_format,
            total,
            breakdown or "none",
        )
        for index, rule in enumerate(self.rules):
            logger.info(
                "  %s -> %s (%d layers)",
                rule.pattern,
                rule.format,
                self._rule_hits[index],
            )
        unmatched = sorted(
            {
                rule.pattern
                for index, rule in enumerate(self.rules)
                if not self._rule_hits[index]
            }
        )
        if unmatched:
            logger.warning(
                "No layer matched these --quantization-overrides patterns: %s",
                ", ".join(unmatched),
            )
