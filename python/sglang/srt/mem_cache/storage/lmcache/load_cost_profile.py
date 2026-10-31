"""Measured costs for exact LMCache request shapes; no request-time I/O."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, fields
from pathlib import Path


@dataclass(frozen=True)
class LoadCostKey:
    model_name: str
    world_size: int
    chunk_size: int
    lookup_tokens: int
    engine_computed_tokens: int
    external_tokens: int


def _number(value: object, name: str) -> float:
    if type(value) not in (int, float):
        raise ValueError(f"{name} must be a finite number")
    try:
        number = float(value)
    except OverflowError as exc:
        raise ValueError(f"{name} must be a finite number") from exc
    if not math.isfinite(number):
        raise ValueError(f"{name} must be a finite number")
    return number


class LoadCostProfile:
    """Decline only an exact measured shape with sufficient predicted savings.

    Costs must compare the same metric, workload and already-issued lookup
    path. The file's metric labels the objective; it does not change how costs
    are compared. Unmeasured shapes retain normal loading behavior.
    """

    def __init__(self, data: dict) -> None:
        required = {"version", "metric", "profiles"}
        if not isinstance(data, dict) or not required <= data.keys():
            raise ValueError("Cost profile requires version, metric and profiles")
        if data.keys() - required - {"min_savings", "calibration"}:
            raise ValueError("Unknown cost profile fields")
        if type(data["version"]) is not int or data["version"] != 1:
            raise ValueError("Cost profile version must be 1")
        if data["metric"] not in ("ttft_ms", "request_elapsed_ms", "wave_elapsed_ms"):
            raise ValueError("Unknown cost profile metric")
        self.metric = data["metric"]
        self.min_savings = _number(data.get("min_savings", 0.05), "min_savings")
        if not 0 <= self.min_savings < 1:
            raise ValueError("min_savings must be in [0, 1)")
        rows = data["profiles"]
        if not isinstance(rows, list) or not rows:
            raise ValueError("profiles must be a non-empty list")
        names = {field.name for field in fields(LoadCostKey)}
        self._costs: dict[LoadCostKey, tuple[float, float]] = {}
        for row in rows:
            if not isinstance(row, dict) or row.keys() != names | {
                "load_ms",
                "recompute_ms",
            }:
                raise ValueError("Each profile must contain exact geometry and costs")
            if not isinstance(row["model_name"], str) or not row["model_name"]:
                raise ValueError("model_name must be a non-empty string")
            for name in names - {"model_name"}:
                minimum = 0 if name == "engine_computed_tokens" else 1
                if type(row[name]) is not int or row[name] < minimum:
                    raise ValueError(f"{name} must be an integer >= {minimum}")
            if (
                row["engine_computed_tokens"] + row["external_tokens"]
                > row["lookup_tokens"]
            ):
                raise ValueError("Local and external tokens exceed lookup_tokens")
            key = LoadCostKey(**{name: row[name] for name in names})
            if key in self._costs:
                raise ValueError("Duplicate cost profile geometry")
            load = _number(row["load_ms"], "load_ms")
            recompute = _number(row["recompute_ms"], "recompute_ms")
            if min(load, recompute) <= 0:
                raise ValueError("Measured costs must be positive")
            self._costs[key] = (load, recompute)

    @classmethod
    def from_file(cls, path: str) -> LoadCostProfile:
        return cls(json.loads(Path(path).read_text(encoding="utf-8")))

    def should_load(self, key: LoadCostKey) -> bool:
        costs = self._costs.get(key)
        if costs is None:
            return True
        load, recompute = costs
        return not (recompute < load and recompute <= load * (1 - self.min_savings))
