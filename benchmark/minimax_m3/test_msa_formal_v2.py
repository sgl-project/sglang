#!/usr/bin/env python3

from __future__ import annotations

import http.client
import io
import json
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest import mock

from precompile_fmha_sm100 import runtime_variants
from probe_msa_e2e_dependencies import (
    REQUIRED_TP4_SM103_ROUTES,
    probe_blackwell_msa_route_manifest,
)
from run_msa_formal_v2 import run_test_only as run_formal_v2_self_tests
from run_msa_formal_v2 import (
    repetitions_for_mode,
    server_healthy,
)


class MSAFormalV2Test(unittest.TestCase):
    def write_json(self, path: Path, payload: dict) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload))

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
        self.assertEqual(result["required_routes"], sorted(REQUIRED_TP4_SM103_ROUTES))

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

    def test_transient_http_protocol_error_is_not_healthy(self) -> None:
        with mock.patch(
            "run_msa_formal_v2.urllib.request.urlopen",
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

    def test_formal_mode_repetition_counts(self) -> None:
        self.assertEqual(repetitions_for_mode("accuracy"), 3)
        self.assertEqual(repetitions_for_mode("external-speed"), 1)
        self.assertEqual(repetitions_for_mode("triton-speed"), 1)
        with self.assertRaisesRegex(ValueError, "unsupported mode"):
            repetitions_for_mode("unknown")

    def test_formal_v2_fail_closed_contract(self) -> None:
        required = {
            "alternating_order_contract",
            "mode_specific_repetition_counts",
            "loopback_api_key_default_and_preservation_contract",
            "speed_modes_have_no_accuracy_requests",
            "wrong_external_route_rejected",
            "wrong_flashinfer_route_rejected",
            "wrong_triton_route_rejected",
            "measured_post_count_rejected",
            "measured_generic_error_rejected",
            "client_log_retry_rejected",
            "client_log_error_rejected",
            "client_log_exhausted_rejected",
            "client_log_timeout_rejected",
            "duplicate_serving_records_rejected",
            "serving_completed_count_rejected",
            "serving_total_input_tokens_tamper_rejected",
            "serving_output_throughput_nan_rejected",
            "speed_fixed_workload_contract",
            "exactly_one_unmeasured_warmup_contract",
            "fresh_output_claim_receipt_positive",
            "stale_output_mtime_rejected",
            "fixed_parity_real_producer_schema_positive",
            "full_model_manifest_positive",
            "stale_model_manifest_hash_rejected",
            "stale_model_aggregate_rejected",
            "model_file_tamper_rejected",
            "model_file_set_drift_rejected",
            "accuracy_per_rep_and_aggregate_gates_rejected",
            "longbench_exact_float_boundary_positive",
            "longbench_beyond_margin_rejected",
            "speed_nan_threshold_rejected",
            "speed_negative_threshold_rejected",
        }
        with tempfile.TemporaryDirectory() as temporary:
            receipt_path = Path(temporary) / "receipt.json"
            with redirect_stdout(io.StringIO()):
                run_formal_v2_self_tests(receipt_path)
            receipt = json.loads(receipt_path.read_text())

        passed = {
            row["id"] for row in receipt["test_results"] if row.get("status") == "pass"
        }
        self.assertEqual(receipt["status"], "pass")
        self.assertEqual(receipt["test_count"], 88)
        self.assertEqual(len(receipt["test_results"]), 88)
        self.assertEqual(len(passed), 88)
        self.assertTrue(required <= passed, sorted(required - passed))


if __name__ == "__main__":
    unittest.main()
