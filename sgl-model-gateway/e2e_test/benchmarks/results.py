"""Benchmark result dataclasses for parsing genai-bench and GPU monitor output."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path


@dataclass
class BenchmarkResult:
    """Parsed benchmark metrics from genai-bench output."""

    ttft_mean: float
    e2e_latency_mean: float
    input_throughput_mean: float
    output_throughput_mean: float
    file_name: str

    @classmethod
    def from_json(cls, path: Path) -> "BenchmarkResult":
        """Parse benchmark results from JSON file."""
        with path.open() as f:
            data = json.load(f)
        stats = data.get("aggregated_metrics", {}).get("stats", {})
        return cls(
            ttft_mean=float(stats.get("ttft", {}).get("mean", float("inf"))),
            e2e_latency_mean=float(
                stats.get("e2e_latency", {}).get("mean", float("inf"))
            ),
            input_throughput_mean=float(
                stats.get("input_throughput", {}).get("mean", 0.0)
            ),
            output_throughput_mean=float(
                stats.get("output_throughput", {}).get("mean", 0.0)
            ),
            file_name=path.name,
        )

    def log(self, experiment: str, logger) -> None:
        """Log benchmark results."""
        logger.info(
            "genai-bench[%s] %s ttft=%.3fs e2e=%.3fs input=%.1f tok/s output=%.1f tok/s",
            experiment,
            self.file_name,
            self.ttft_mean,
            self.e2e_latency_mean,
            self.input_throughput_mean,
            self.output_throughput_mean,
        )

    def validate(self, thresholds: dict) -> None:
        """Validate metrics against thresholds."""
        checks = [
            ("ttft_mean_max", self.ttft_mean, "<=", "TTFT"),
            ("e2e_latency_mean_max", self.e2e_latency_mean, "<=", "E2E latency"),
            (
                "input_throughput_mean_min",
                self.input_throughput_mean,
                ">=",
                "Input throughput",
            ),
            (
                "output_throughput_mean_min",
                self.output_throughput_mean,
                ">=",
                "Output throughput",
            ),
        ]
        for key, value, op, name in checks:
            if key not in thresholds:
                continue
            threshold = thresholds[key]
            if op == "<=" and value > threshold:
                raise AssertionError(f"{name}: {value:.2f} > {threshold}")
            if op == ">=" and value < threshold:
                raise AssertionError(f"{name}: {value:.2f} < {threshold}")


@dataclass
class GPUUtilization:
    """Parsed GPU utilization metrics from gpu_monitor output."""

    overall_mean: float
    per_gpu: dict[str, dict[str, float]]

    @classmethod
    def from_json(cls, path: Path) -> "GPUUtilization | None":
        """Parse GPU utilization from JSON file."""
        try:
            with path.open() as f:
                data = json.load(f)
            return cls(
                overall_mean=float(data.get("overall", {}).get("mean", 0)),
                per_gpu=data.get("per_gpu", {}),
            )
        except Exception:
            return None
