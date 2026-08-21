#!/usr/bin/env python3

from __future__ import annotations

import http.client
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from precompile_fmha_sm100 import runtime_variants
from probe_msa_e2e_dependencies import (
    REQUIRED_TP4_SM103_ROUTES,
    probe_blackwell_msa_route_manifest,
)
from run_msa_ab_repetitions import (
    OFFLINE_THROUGHPUT_ARGS,
    OFFLINE_THROUGHPUT_DATASET,
    server_healthy,
    validate_resume_manifest,
)
from summarize_msa_repetitions import (
    CONCURRENCIES,
    SERVING_METRICS,
    accuracy_noninferiority_failures,
    build_summary,
    expected_order,
)


class MSARepetitionSummaryTest(unittest.TestCase):
    def make_route_manifest(self, root: Path) -> Path:
        routes = [
            {
                "id": route_id,
                "architectures": ["sm100", "sm103"],
                "source_units": sorted(source_units),
            }
            for route_id, source_units in REQUIRED_TP4_SM103_ROUTES.items()
        ]
        units = sorted(
            {
                source_unit
                for source_units in REQUIRED_TP4_SM103_ROUTES.values()
                for source_unit in source_units
            }
        )
        hashes = {
            field: "a" * 64
            for field in (
                "generated_input_sha256",
                "vendored_sha256",
                "binding_sha256",
                "arg_plan_sha256",
            )
        }
        manifest = {
            "operation": "blackwell_msa",
            "attention_topk": 16,
            "reachable_specialization_count": len(routes),
            "reachable_specializations": routes,
            "source_inventory": {
                "entries": [
                    {"target": "sm103a", "source_unit": unit, **hashes}
                    for unit in units
                ]
            },
        }
        path = root / "csrc" / "blackwell_msa" / "route_manifest.json"
        self.write_json(path, manifest)
        return path

    def test_route_manifest_covers_tp4_gb300_source_routes(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            path = self.make_route_manifest(root)
            result = probe_blackwell_msa_route_manifest(root)

        self.assertEqual(result["path"], str(path.resolve()))
        self.assertEqual(
            result["required_routes"], sorted(REQUIRED_TP4_SM103_ROUTES)
        )

    def test_route_manifest_rejects_missing_decode_source(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            path = self.make_route_manifest(root)
            manifest = json.loads(path.read_text())
            manifest["source_inventory"]["entries"] = [
                entry
                for entry in manifest["source_inventory"]["entries"]
                if entry["source_unit"] != "decode_m16_bf16_paged"
            ]
            self.write_json(path, manifest)
            with self.assertRaisesRegex(RuntimeError, "decode_m16_bf16_paged"):
                probe_blackwell_msa_route_manifest(root)

    def test_throughput_dataset_is_offline_and_deterministic(self) -> None:
        self.assertEqual(OFFLINE_THROUGHPUT_DATASET, "random-ids")
        self.assertEqual(
            OFFLINE_THROUGHPUT_ARGS,
            ("--dataset-name", "random-ids", "--tokenize-prompt"),
        )

    def test_transient_http_protocol_error_is_not_healthy(self) -> None:
        with mock.patch(
            "run_msa_ab_repetitions.urllib.request.urlopen",
            side_effect=http.client.BadStatusLine("GET /health_generate HTTP/1.1"),
        ):
            self.assertFalse(server_healthy("http://127.0.0.1:30000"))

    def test_precompile_variants_cover_tp4_sparse_paged_routes(self) -> None:
        dtype_code = 42
        variants = runtime_variants(dtype_code)

        self.assertEqual(len(variants), 9)
        self.assertEqual(len(set(variants)), 9)
        self.assertEqual(
            set(variants),
            {
                (dtype_code, 128, single_wg, 0, 128, split_kv, pack_factor)
                for single_wg in (True, False)
                for split_kv in (False, True)
                for pack_factor in (1, 16)
            }
            | {(dtype_code, 256, False, 0, 128, False, 1)},
        )

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
                            {
                                "name": name,
                                "expected": answer,
                                "exact_expected": True,
                                "content": answer,
                                "response_sha256": digest,
                            }
                            for name, answer, digest in (
                                ("short", "MSA-SHORT-4B19", "c" * 64),
                                ("long_32768", "MSA-32768-C7F29A", "d" * 64),
                                ("long_65536", "MSA-65536-C7F29A", "e" * 64),
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

    def test_reasoning_hash_drift_does_not_change_exact_fixed_answer(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self.make_fixture(root)
            parity_path = root / "rep02" / "candidate" / "fixed_parity.json"
            parity = json.loads(parity_path.read_text())
            parity["records"][0]["response_sha256"] = "f" * 64
            self.write_json(parity_path, parity)

            build_summary(root)

    def test_rejects_fixed_answer_drift(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self.make_fixture(root)
            parity_path = root / "rep02" / "candidate" / "fixed_parity.json"
            parity = json.loads(parity_path.read_text())
            parity["records"][0]["content"] = "WRONG"
            parity["records"][0]["exact_expected"] = False
            self.write_json(parity_path, parity)

            with self.assertRaisesRegex(ValueError, "failed its expected answer"):
                build_summary(root)

    def test_accuracy_margins_are_metric_specific(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self.make_fixture(root)
            for repetition in range(1, 4):
                comparison_path = root / f"rep{repetition:02d}" / "comparison.json"
                comparison = json.loads(comparison_path.read_text())
                comparison["accuracy"]["longbench_v2"]["candidate"] = 0.39
                self.write_json(comparison_path, comparison)
            summary = build_summary(root)

        self.assertEqual(
            accuracy_noninferiority_failures(
                summary,
                {"gpqa": 0.0, "longbench_v2": 0.01},
            ),
            [],
        )
        self.assertRegex(
            accuracy_noninferiority_failures(
                summary,
                {"gpqa": 0.0, "longbench_v2": 0.009},
            )[0],
            "longbench_v2",
        )

    def test_resume_manifest_rejects_immutable_input_drift(self) -> None:
        expected = {
            "schema_version": 1,
            "model": "/model",
            "base_url": "http://127.0.0.1:30000",
            "flashinfer_source_dir": "/flashinfer",
            "expected_flashinfer_head": "a" * 40,
            "expected_tvm_ffi_version": "0.1.9",
            "server_command": ["python", "-m", "sglang.launch_server"],
            "repetitions": 3,
        }
        validate_resume_manifest(expected, expected, start_repetition=2)
        drifted = dict(expected, model="/different-model")
        with self.assertRaisesRegex(ValueError, "model"):
            validate_resume_manifest(drifted, expected, start_repetition=2)


if __name__ == "__main__":
    unittest.main()
