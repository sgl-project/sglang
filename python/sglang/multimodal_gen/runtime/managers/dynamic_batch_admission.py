# SPDX-License-Identifier: Apache-2.0
"""Admission control for native diffusion batching.

Native diffusion batching is model, resolution, device, and implementation
dependent. The scheduler treats `--batching-max-size` as the public ceiling;
`--batching-config` can apply stricter caps for specific model and shape
combinations.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from difflib import get_close_matches
from typing import TYPE_CHECKING, Any

from sglang.multimodal_gen.runtime.loader.utils import BYTES_PER_GB
from sglang.multimodal_gen.runtime.pipelines_core import Req
from sglang.multimodal_gen.runtime.platforms import current_platform
from sglang.multimodal_gen.runtime.utils.logging_utils import init_logger

if TYPE_CHECKING:
    from sglang.multimodal_gen.runtime.server_args import ServerArgs

logger = init_logger(__name__)

_BATCHING_RULE_KEYS = frozenset(
    {
        "model",
        "model_contains",
        "resolution",
        "device_memory_gb_min",
        "device_memory_gb_max",
        "offload",
        "max_batch_size",
        "max_cost",
        # Free-form provenance/benchmark metadata. It is intentionally ignored
        # by admission, but accepted so production configs can explain caps.
        "calibration",
    }
)


@dataclass(frozen=True)
class AdmissionLimit:
    """Effective batch size and cost caps after matching batching rules."""

    max_batch_size: int
    max_cost: float | None = None
    cap_reason: str | None = None

    def reject_reason(self, *, batch_size: int, batch_cost: float) -> str | None:
        if batch_size > self.max_batch_size:
            return self.cap_reason or f"config_cap:{self.max_batch_size}"
        if self.max_cost is not None and batch_cost > self.max_cost:
            return f"cost_budget:{batch_cost:.0f}>{self.max_cost:.0f}"
        return None

    def stop_reason_for_next_cost(self, next_batch_cost: float) -> str | None:
        if self.max_cost is not None and next_batch_cost > self.max_cost:
            return f"cost_budget_next:{next_batch_cost:.0f}>{self.max_cost:.0f}"
        return None


@dataclass(frozen=True)
class BatchingRule:
    """One user-provided batching admission rule loaded from batching config."""

    model: str | None = None
    model_contains: str | None = None
    resolution: str | None = None
    device_memory_gb_min: float | None = None
    device_memory_gb_max: float | None = None
    offload: bool | None = None
    max_batch_size: int = 1
    max_cost: float | None = None
    source: str = "user"

    @classmethod
    def from_dict(cls, data: dict[str, Any], *, source: str) -> "BatchingRule":
        if not isinstance(data, dict):
            raise ValueError(
                f"batching config rule from {source} must be an object, "
                f"got {type(data).__name__}"
            )
        _validate_rule_keys(data, source=source)
        if "max_batch_size" not in data:
            raise ValueError("batching config rule requires max_batch_size")

        rule = cls(
            model=_optional_str(data.get("model")),
            model_contains=_optional_str(data.get("model_contains")),
            resolution=_optional_str(data.get("resolution")),
            device_memory_gb_min=_optional_float(data.get("device_memory_gb_min")),
            device_memory_gb_max=_optional_float(data.get("device_memory_gb_max")),
            offload=_optional_bool(data.get("offload")),
            max_batch_size=int(data["max_batch_size"]),
            max_cost=_optional_float(data.get("max_cost")),
            source=source,
        )
        rule.validate()
        return rule

    def validate(self) -> None:
        if self.model is not None and self.model_contains is not None:
            raise ValueError(
                "batching config rule cannot set both model and model_contains"
            )
        if self.model is None and self.model_contains is None:
            raise ValueError("batching config rule requires model or model_contains")
        if self.max_batch_size < 1:
            raise ValueError("batching config rule max_batch_size must be >= 1")
        if self.max_cost is not None and self.max_cost <= 0.0:
            raise ValueError("batching config rule max_cost must be > 0")
        if (
            self.device_memory_gb_min is not None
            and self.device_memory_gb_max is not None
            and self.device_memory_gb_min > self.device_memory_gb_max
        ):
            raise ValueError(
                "batching config rule device_memory_gb_min must be <= device_memory_gb_max"
            )

    def matches(
        self,
        *,
        model_path: str,
        resolution: str | None,
        device_memory_gb: float | None,
        offload: bool,
    ) -> bool:
        if self.model is not None and self.model != model_path:
            return False
        if self.model_contains is not None and self.model_contains not in model_path:
            return False
        if self.resolution not in (None, "*") and self.resolution != resolution:
            return False
        if self.offload is not None and self.offload != offload:
            return False
        if device_memory_gb is None:
            return True
        if (
            self.device_memory_gb_min is not None
            and device_memory_gb < self.device_memory_gb_min
        ):
            return False
        if (
            self.device_memory_gb_max is not None
            and device_memory_gb > self.device_memory_gb_max
        ):
            return False
        return True


class BatchAdmissionController:
    """Applies configured caps before adding requests to a batch."""

    def __init__(self, server_args: "ServerArgs", gpu_id: int):
        self._mode = getattr(server_args, "batching_mode", "dynamic")
        self._user_max_batch_size = max(1, int(server_args.batching_max_size))
        self._model_path = server_args.model_path
        self._offload = bool(server_args.dit_layerwise_offload)
        self._device_memory_gb = self._get_device_memory_gb(gpu_id)
        self._rules = load_batching_config(server_args.batching_config)
        self._pipeline_config = server_args.pipeline_config

        if self.enabled:
            logger.info(
                "Batch admission enabled: user_max=%d, device_memory=%.1fGiB, rules=%d",
                self._user_max_batch_size,
                self._device_memory_gb or 0.0,
                len(self._rules),
            )

    @property
    def enabled(self) -> bool:
        return self._mode == "dynamic" and self._user_max_batch_size > 1

    def reject_reason_for_candidate(
        self, current_reqs: list[Req], candidate_req: Req
    ) -> str | None:
        if not self.enabled:
            return None
        proposed = current_reqs + [candidate_req]
        limit = self.limit_for(proposed[0])
        return limit.reject_reason(
            batch_size=len(proposed),
            batch_cost=self.estimate_batch_cost(proposed),
        )

    def batch_is_full(self, reqs: list[Req]) -> bool:
        """Return whether another roughly similar request would exceed the cap."""
        if not self.enabled or not reqs:
            return len(reqs) >= self._user_max_batch_size

        limit = self.limit_for(reqs[0])
        if len(reqs) >= limit.max_batch_size:
            return True

        next_cost = self.estimate_batch_cost(reqs + [reqs[0]])
        return limit.max_cost is not None and next_cost > limit.max_cost

    def limit_reason_for_batch(self, reqs: list[Req]) -> str | None:
        if not self.enabled or not reqs:
            return None

        limit = self.limit_for(reqs[0])
        if len(reqs) >= limit.max_batch_size:
            return limit.cap_reason or f"config_cap:{limit.max_batch_size}"

        next_cost = self.estimate_batch_cost(reqs + [reqs[0]])
        return limit.stop_reason_for_next_cost(next_cost)

    def max_admissible_batch_size(self, req: Req) -> int:
        return self.limit_for(req).max_batch_size

    def limit_for(self, req: Req) -> AdmissionLimit:
        """Return the effective admission limit for the request's model and shape."""
        rules = self._matching_rules(req)
        if not rules:
            return AdmissionLimit(max_batch_size=self._user_max_batch_size)

        config_cap = min(rule.max_batch_size for rule in rules)
        max_batch_size = min(self._user_max_batch_size, config_cap)
        cap_reason = (
            f"config_cap:{max_batch_size}"
            if max_batch_size < self._user_max_batch_size
            else None
        )
        costs = [rule.max_cost for rule in rules if rule.max_cost is not None]
        return AdmissionLimit(
            max_batch_size=max(1, max_batch_size),
            max_cost=min(costs) if costs else None,
            cap_reason=cap_reason,
        )

    def estimate_batch_cost(self, reqs: list[Req]) -> float:
        return sum(
            float(self._pipeline_config.estimate_request_cost(req)) for req in reqs
        )

    def _matching_rules(self, req: Req) -> list[BatchingRule]:
        return [
            rule
            for rule in self._rules
            if rule.matches(
                model_path=self._model_path,
                resolution=req.resolution_key,
                device_memory_gb=self._device_memory_gb,
                offload=self._offload,
            )
        ]

    @staticmethod
    def _get_device_memory_gb(gpu_id: int) -> float | None:
        try:
            return current_platform.get_device_total_memory(gpu_id) / BYTES_PER_GB
        except Exception:
            return None


def load_batching_config(path: str | None) -> list[BatchingRule]:
    if path is None:
        return []

    with open(path, encoding="utf-8") as f:
        payload = json.load(f)

    source = os.path.abspath(path)
    entries = _config_entries(payload)
    rules = [BatchingRule.from_dict(entry, source=source) for entry in entries]
    if not rules:
        raise ValueError(f"batching config {source} does not contain any rules")
    return rules


def _config_entries(payload: Any) -> list[dict[str, Any]]:
    if isinstance(payload, dict) and payload.get("schema_version") not in (None, 1):
        raise ValueError("batching config schema_version must be 1")
    if isinstance(payload, dict) and isinstance(payload.get("rules"), list):
        return payload["rules"]
    if isinstance(payload, list):
        return payload
    if isinstance(payload, dict):
        entries: list[dict[str, Any]] = []
        for key, value in payload.items():
            if key == "schema_version" or not isinstance(value, dict):
                continue
            model, _sep, resolution = key.partition("|")
            entry = dict(value)
            if model:
                entry.setdefault("model", model)
            if resolution:
                entry.setdefault("resolution", resolution)
            entries.append(entry)
        return entries
    raise ValueError(
        "batching config must be a {'schema_version': 1, 'rules': [...]} object, "
        "a list of rules, or a mapping keyed by model|resolution"
    )


def _validate_rule_keys(data: dict[str, Any], *, source: str) -> None:
    unknown = sorted(set(data) - _BATCHING_RULE_KEYS)
    if not unknown:
        return

    hints = []
    for key in unknown:
        matches = get_close_matches(key, _BATCHING_RULE_KEYS, n=1)
        if matches:
            hints.append(f"{key!r} (did you mean {matches[0]!r}?)")
        else:
            hints.append(repr(key))
    raise ValueError(
        f"batching config rule from {source} contains unknown key(s): "
        f"{', '.join(hints)}"
    )


def _optional_str(value: Any) -> str | None:
    if value is None:
        return None
    return str(value)


def _optional_float(value: Any) -> float | None:
    if value is None:
        return None
    return float(value)


def _optional_bool(value: Any) -> bool | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in ("1", "true", "yes", "y", "on"):
            return True
        if lowered in ("0", "false", "no", "n", "off"):
            return False
    raise ValueError(f"cannot parse boolean batching config value: {value!r}")
