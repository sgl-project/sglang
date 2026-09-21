"""Evidence gates must reject incomplete runs and metadata/data mismatches."""

import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
spec = importlib.util.spec_from_file_location(
    "pd_validation", ROOT / "test/manual/kv_transfer/validate_pd_compression.py"
)
validation = importlib.util.module_from_spec(spec)
spec.loader.exec_module(validation)


class ValidationTests(unittest.TestCase):
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
            "phase": "force",
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
