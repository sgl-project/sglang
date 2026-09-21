"""Evidence gates must reject incomplete runs and metadata/data mismatches."""

import importlib.util
import json
import os
import tempfile
import unittest
from pathlib import Path

if os.environ.get("SGLANG_COMPRESSION_STANDALONE_TEST") == "1":
    # CI registration is a runtime no-op; avoid package initialization here.
    def register_cpu_ci(**kwargs):
        pass
else:
    from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

ROOT = Path(__file__).resolve().parents[4]
spec = importlib.util.spec_from_file_location(
    "pd_validation", ROOT / "test/manual/kv_transfer/validate_pd_compression.py"
)
validation = importlib.util.module_from_spec(spec)
spec.loader.exec_module(validation)


class ValidationTests(unittest.TestCase):
    def test_execution_contract_requires_pinned_real_values(self):
        template = json.loads(
            (
                ROOT / "test/manual/kv_transfer/execution_contract.example.json"
            ).read_text()
        )
        with self.assertRaises(AssertionError):
            validation.validate_execution_contract(template)
        template.update(
            source_manifest_sha256="a" * 64,
            image_digest="sha256:" + "b" * 64,
            model_revision="model-files-sha256",
        )
        validation.validate_execution_contract(template)
        for key, value in (
            ("model_revision", ""),
            ("page_size", 16),
            ("source_manifest_sha256", "old"),
            ("topology", {}),
        ):
            with self.assertRaises(AssertionError):
                validation.validate_execution_contract(dict(template, **{key: value}))

    def test_gpu_inventory_rejects_skips_missing_and_duplicate_cases(self):
        spec = importlib.util.spec_from_file_location(
            "image_check",
            ROOT / "test/manual/kv_transfer/check_pd_compression_image.py",
        )
        check = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(check)
        manifest = {"gpu_test_nodeids": ["test_gpu.py::test_a", "test_gpu.py::test_b"]}
        path = self.root / "gpu.xml"
        path.write_text(
            '<testsuite><testcase name="test_a"/><testcase name="test_b"/></testsuite>'
        )
        self.assertEqual(check.validate_gpu_report(manifest, path), 2)
        for bad in (
            '<testcase name="test_a"/>',
            '<testcase name="test_a"/><testcase name="test_a"/>',
            '<testcase name="test_a"><skipped/></testcase><testcase name="test_b"/>',
            '<testcase name="test_a"><failure/></testcase><testcase name="test_b"/>',
        ):
            path.write_text("<testsuite>" + bad + "</testsuite>")
            with self.assertRaises(AssertionError):
                check.validate_gpu_report(manifest, path)

    def test_restore_gate_requires_full_frozen_range(self):
        row = {
            "group": "l2",
            "case": "restore",
            "rid": "r",
            "input_tokens": 8192,
            "result": {
                "meta_info": {
                    "cached_tokens": 8191,
                    "cached_tokens_details": {"device": 0, "host": 8191, "storage": 0},
                }
            },
        }
        event = dict(
            native_missing_pages=8191,
            adopted_pages=8191,
            verified_pages=8191,
            lz4_pages=8191,
        )
        validation.validate_restore_event(row, event, "force-l2")
        for count in (1, 7172):
            broken = dict(event, adopted_pages=count, verified_pages=count)
            with self.assertRaises(AssertionError):
                validation.validate_restore_event(row, broken, "force-l2")
        for detail in (
            None,
            {},
            {"device": 8190, "host": 1},
            {"device": 0, "host": 7172},
        ):
            row["result"]["meta_info"]["cached_tokens_details"] = detail
            with self.assertRaises(AssertionError):
                validation.validate_restore_response(row)

    def test_drain_requires_three_observations_not_claimed_counter(self):
        rows = [
            dict(label="final", elapsed=i * 5, idle=True, consecutive=i)
            for i in (1, 2, 3)
        ]
        self.assertEqual(validation.completed_drains(rows), {"final"})
        for broken in (
            [rows[-1]],
            [rows[0], rows[0], rows[-1]],
            [dict(rows[0], elapsed=181)],
        ):
            with self.assertRaises(AssertionError):
                validation.completed_drains(broken)

    def test_illegal_pairs_self_comparison_and_model_changes_rejected(self):
        a, b = self.fixture("a"), self.fixture("b")
        with self.assertRaisesRegex(AssertionError, "Self-comparison"):
            validation.compare(b, b)
        config = json.loads((b / "config.json").read_text())
        config["model"] = "other"
        (b / "config.json").write_text(json.dumps(config))
        with self.assertRaisesRegex(AssertionError, "configuration"):
            validation.compare(a, b)
        config.pop("model")
        config["phase"] = "force-l2"
        (b / "config.json").write_text(json.dumps(config))
        with self.assertRaisesRegex(AssertionError, "phase pair"):
            validation.compare(a, b)

    def test_null_missing_and_device_only_are_host_misses(self):
        for meta in (
            {},
            {"cached_tokens_details": None},
            {"cached_tokens_details": {"device": 8191}},
        ):
            self.assertEqual(validation.host_cached_tokens(meta), 0)
        self.assertEqual(
            validation.host_cached_tokens({"cached_tokens_details": {"host": 8191}}),
            8191,
        )

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)

    def tearDown(self):
        self.temp.cleanup()

    def fixture(self, name, token=7, cached=0):
        root = self.root / name
        root.mkdir()
        config = {
            "phase": "off" if name == "a" else "force",
            "workload_sha256": "same",
            "expected_cases": [["lengths", 0, "32"]],
        }
        row = {
            "group": "lengths",
            "case": "32",
            "cycle": 0,
            "rid": "test",
            "input_sha256": "same",
            "input_tokens": 32,
            "result": {
                "meta_info": {
                    "cached_tokens": cached,
                    "finish_reason": {"type": "length"},
                    "prompt_tokens": 32,
                    "completion_tokens": 64,
                    "output_token_logprobs": [[0, token, None]] * 64,
                }
            },
        }
        (root / "config.json").write_text(json.dumps(config))
        (root / "requests.jsonl").write_text(json.dumps(row) + "\n")

        def trace(stage, **extra):
            return (
                "KV_COMPRESSION_HANDOFF "
                + json.dumps(
                    dict(stage=stage, rid="test", room=17, token_id=token, **extra)
                )
                + "\n"
            )

        send = {
            "room": 17,
            "chunk": 0,
            "raw_bytes": 100,
            "wire_bytes": 110,
            "objects": 32,
            "lz4_objects": 32,
            "raw_objects": 0,
        }
        recv = {
            "room": 17,
            "page_start": 0,
            "raw_bytes": 100,
            "wire_bytes": 110,
            "lz4_objects": 32,
            "verified": True,
        }
        (root / "prefill.log").write_text(
            trace("prefill_sampled")
            + trace("metadata_written")
            + "PD_KV_COMPRESSION_SEND "
            + json.dumps(send)
            + "\n"
        )
        (root / "decode.log").write_text(
            trace("decode_received", received_room=17)
            + "PD_KV_COMPRESSION_RECV "
            + json.dumps(recv)
            + "\n"
        )
        config_log = (
            "PD_KV_COMPRESSION_CONFIG "
            + json.dumps(
                {
                    "mode": "lz4",
                    "force": True,
                    "verify": True,
                    "shared_l2": False,
                    "chunk_tokens": 1024,
                    "workspace_bytes": 512 * 1024**2,
                }
            )
            + "\n"
        )
        for name in ("prefill.log", "decode.log"):
            path = root / name
            path.write_text(config_log + path.read_text())
        return root

    def test_verified_transfer_and_complete_output_comparison(self):
        a, b = self.fixture("a"), self.fixture("b")
        validation.audit(b)
        validation.compare(a, b)
        self.assertTrue(json.loads((b / "comparison.json").read_text())["passed"])

    def test_incomplete_cases_cannot_pass_audit_or_comparison(self):
        a, b = self.fixture("a"), self.fixture("b")
        (b / "requests.jsonl").write_text("")
        with self.assertRaisesRegex(AssertionError, "incomplete"):
            validation.audit(b)
        with self.assertRaisesRegex(AssertionError, "Missing"):
            validation.compare(a, b)
        (a / "requests.jsonl").write_text("")
        with self.assertRaisesRegex(AssertionError, "incomplete"):
            validation.compare(a, b)

    def test_aborted_or_partial_output_cannot_pass_offline_audit(self):
        a = self.fixture("a")
        row = validation.result_rows(a)[0]
        row["result"]["meta_info"]["finish_reason"]["type"] = "abort"
        (a / "requests.jsonl").write_text(json.dumps(row) + "\n")
        with self.assertRaisesRegex(AssertionError, "termination"):
            validation.audit(a)

    def test_first_token_difference_is_preserved(self):
        a, b = self.fixture("a", 821), self.fixture("b", 279)
        with self.assertRaisesRegex(AssertionError, "token mismatch"):
            validation.compare(a, b)
        report = json.loads((b / "comparison.json").read_text())
        self.assertEqual(report["differences"][0]["first_difference"], 0)

    def test_different_cache_paths_are_not_an_accuracy_comparison(self):
        a, b = self.fixture("a"), self.fixture("b", cached=1)
        with self.assertRaisesRegex(AssertionError, "cache execution paths"):
            validation.compare(a, b)

    def test_metadata_and_kv_verification_failures_are_rejected(self):
        for old, new, message in [
            ('"token_id": 7', '"token_id": 8', "Handoff"),
            ('"received_room": 17', '"received_room": 18', "room"),
            ('"verified": true', '"verified": false', "writeback"),
            ('"lz4_objects": 32', '"lz4_objects": 0', "decompress"),
            ('"force": true', '"force": false', "Effective config"),
        ]:
            root = self.fixture(str(len(list(self.root.iterdir()))))
            p = root / "decode.log"
            p.write_text(p.read_text().replace(old, new))
            with (
                self.subTest(message=message),
                self.assertRaisesRegex(AssertionError, message),
            ):
                validation.audit(root)

    def test_idle_requires_all_resource_categories_drained(self):
        state = {
            "pending_backups": 0,
            "active_backup": None,
            "active_restore": None,
            "quarantined": 0,
            "runtime": {
                "queued_tasks": 0,
                "running_tasks": 0,
                "inflight_objects": 0,
                "quarantined": 0,
                "resident_bytes": 0,
            },
            "l2": {
                "reserved_bytes": 0,
                "retired_bytes": 0,
                "active_readers": 0,
                "active_writers": 0,
                "stage_waiters": 0,
                "stage_users": 0,
                "quarantined_objects": 0,
            },
        }

        def is_idle():
            return validation.latest_state("KV_COMPRESSION_STATS " + json.dumps(state))[
                1
            ]

        self.assertTrue(is_idle())
        state["active_restore"] = "request"
        self.assertFalse(is_idle())
        state["active_restore"] = None
        for group, key in [
            ("runtime", "resident_bytes"),
            ("l2", "retired_bytes"),
            ("l2", "reserved_bytes"),
            *[
                ("l2", k)
                for k in (
                    "active_readers",
                    "active_writers",
                    "stage_waiters",
                    "stage_users",
                    "quarantined_objects",
                )
            ],
        ]:
            state[group][key] = 1
            self.assertFalse(is_idle())
            state[group][key] = 0


if __name__ == "__main__":
    unittest.main()
