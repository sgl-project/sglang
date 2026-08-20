#!/usr/bin/env python3

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from summarize_msa_repetitions import (
    CONCURRENCIES,
    SERVING_METRICS,
    build_summary,
    expected_order,
)


class MSARepetitionSummaryTest(unittest.TestCase):
    def write_json(self, path: Path, payload: dict) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload))

    def make_fixture(self, root: Path) -> None:
        for repetition in range(1, 4):
            repetition_dir = root / f"rep{repetition:02d}"
            self.write_json(
                repetition_dir / "order.json",
                {"order": expected_order(repetition)},
            )
            comparison = {
                "accuracy": {
                    "gpqa": {"baseline": 0.5, "candidate": 0.5, "delta": 0},
                    "longbench_v2": {
                        "baseline": 0.4,
                        "candidate": 0.4,
                        "delta": 0,
                    },
                },
                "serving": {},
            }
            for concurrency in CONCURRENCIES:
                metrics = {}
                for metric, higher_is_better in SERVING_METRICS.items():
                    baseline = float(100 + repetition)
                    candidate = baseline * (1.1 if higher_is_better else 0.9)
                    metrics[metric] = {
                        "baseline": baseline,
                        "candidate": candidate,
                        "gain": 0,
                    }
                comparison["serving"][str(concurrency)] = metrics
            self.write_json(repetition_dir / "comparison.json", comparison)

            for provider in ("baseline", "candidate"):
                provider_dir = repetition_dir / provider
                provider_dir.mkdir()
                (provider_dir / "gpqa_dataset.sha256").write_text("a" * 64 + "\n")
                self.write_json(
                    provider_dir / "longbench_v2_subset_manifest.json",
                    {"subset_sha256": "b" * 64},
                )
                self.write_json(
                    provider_dir / "fixed_parity.json",
                    {
                        "records": [
                            {"name": name, "response_sha256": digest}
                            for name, digest in (
                                ("short", "c" * 64),
                                ("long_32768", "d" * 64),
                                ("long_65536", "e" * 64),
                            )
                        ]
                    },
                )

    def test_three_run_medians_and_order(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self.make_fixture(root)
            summary = build_summary(root)

        self.assertEqual(
            summary["orders"],
            [
                ["baseline", "candidate"],
                ["candidate", "baseline"],
                ["baseline", "candidate"],
            ],
        )
        output = summary["serving"]["32"]["output_throughput"]
        self.assertEqual(output["baseline_median"], 102.0)
        self.assertAlmostEqual(output["candidate_median"], 112.2)
        self.assertAlmostEqual(output["gain_from_backend_medians"], 0.1)
        latency = summary["serving"]["32"]["median_itl_ms"]
        self.assertGreater(latency["gain_from_backend_medians"], 0)

    def test_rejects_order_drift(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self.make_fixture(root)
            self.write_json(
                root / "rep02" / "order.json",
                {"order": ["baseline", "candidate"]},
            )
            with self.assertRaisesRegex(ValueError, "order"):
                build_summary(root)


if __name__ == "__main__":
    unittest.main()
