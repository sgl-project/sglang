"""Real single-GPU PD bad-seed rejection and natural allocator reuse.

    python test/manual/spec/test_spec_standalone_pd_recovery.py -f

Uses the same public model overrides as test_spec_standalone_disaggregation.
The test-only forward hook corrupts one producer seed after set_buf and before
final send. Mooncake TCP, Decode validation/abort, and all allocators run normally.
This exercises a malformed producer value, not a transport failure or TP/PP
consensus. D2H index observations affect timing; no performance claim is made.
"""

import json
import math
import os
import secrets
import shutil
import tempfile
import time
import unittest
from pathlib import Path

from standalone_pd_fixture import StandalonePDServerBase
from transformers import AutoTokenizer


class TestStandaloneSeedRecovery(StandalonePDServerBase):
    # B's prompt uses more than half this pool. Its real allocation must reuse
    # some pages freed by A, regardless of FIFO vs sorted-free allocation.
    extra_common_args = (
        "--context-length",
        "256",
        "--max-total-tokens",
        "256",
        "--chunked-prefill-size",
        "256",
        "--num-reserved-decode-tokens",
        "32",
        "--disable-radix-cache",
    )

    @classmethod
    def setUpClass(cls):
        cls.bad_rid = "standalone-pd-bad-" + secrets.token_hex(12)
        output = os.environ.get("SGLANG_TEST_STANDALONE_PD_OUTPUT_DIR")
        if output:
            # The base creates the fresh root; only plan paths before its setup.
            cls.fault_dir = Path(output).resolve() / "fault-events"
        else:
            cls.fault_dir = Path(tempfile.mkdtemp(prefix="standalone-pd-recovery-"))
            cls.addClassCleanup(shutil.rmtree, cls.fault_dir)
        cls.event_paths = {
            role: cls.fault_dir / f"{role}.jsonl" for role in ("prefill", "decode")
        }
        for role in ("prefill", "decode"):
            hooks = json.dumps(
                [
                    {
                        "name": "standalone-seed-recovery",
                        "target_modules": ["*logits_processor*"],
                        "hook_factory": "standalone_pd_fault_hook:make_hook",
                        "config": {
                            "role": role,
                            "log": str(cls.event_paths[role]),
                            "bad_rid": cls.bad_rid,
                        },
                    }
                ]
            )
            args = ["--forward-hooks", hooks]
            if role == "decode":
                args += ["--disaggregation-decode-extra-slots", "0"]
            setattr(cls, f"extra_{role}_args", tuple(args))
        super().setUpClass()
        cls.fault_dir.mkdir(exist_ok=True)

    def events(self, role):
        path = self.event_paths[role]
        if not path.exists():
            return []
        raw = path.read_bytes()
        # A reader can run while the scheduler appends one complete JSON line.
        rows = [
            json.loads(line)
            for line in raw.splitlines(keepends=True)
            if line.endswith(b"\n")
        ]
        self.assertEqual(
            [row["sequence"] for row in rows], list(range(1, len(rows) + 1))
        )
        self.assertLessEqual(len(rows), 256)
        self.assertEqual(len({row["pid"] for row in rows}), 1 if rows else 0)
        return rows

    def wait_idle_snapshot(self, completed):
        deadline = time.monotonic() + 45
        while True:
            matches = [
                row
                for row in self.events("decode")
                if row["phase"] == "idle" and row["completed"] == completed
            ]
            if matches:
                self.assertEqual(len(matches), 1)
                return matches[0]["resources"]
            if time.monotonic() >= deadline:
                self.fail(f"Missing real idle allocator census after {completed}")
            time.sleep(0.1)

    def assert_full_capacity(self, snapshot):
        self.assertEqual(snapshot["allocator"], "TokenToKVPoolAllocator")
        self.assertEqual(snapshot["request_pool"], "DecodeReqToTokenPool")
        self.assertEqual(snapshot["page_size"], 1)
        self.assertEqual(snapshot["token_capacity"], 256)
        self.assertEqual(snapshot["token_available"], 256)
        self.assertEqual(set(snapshot["token_free_ids"]), set(range(1, 257)))
        self.assertEqual(len(snapshot["token_free_ids"]), 256)
        self.assertIsNone(snapshot["free_group_pending"])
        self.assertEqual(snapshot["request_capacity"], 1)
        self.assertEqual(snapshot["request_available"], 1)
        self.assertEqual(len(set(snapshot["request_free_slots"])), 1)
        self.assertEqual(snapshot["metadata_capacity"], 2)
        self.assertEqual(snapshot["metadata_available"], 2)
        self.assertEqual(sorted(snapshot["metadata_free_slots"]), [0, 1])

    def one_event(self, phase, rid, *, role="decode"):
        matches = [
            row
            for row in self.events(role)
            if row["phase"] == phase
            and row.get("rid", row.get("request", {}).get("rid")) == rid
        ]
        self.assertEqual(len(matches), 1, (phase, rid, matches))
        return matches[0]

    def test_bad_seed_aborts_then_real_capacity_is_reused(self):
        target = AutoTokenizer.from_pretrained(self.target_model)
        draft = AutoTokenizer.from_pretrained(self.draft_model)
        self.assertEqual(target.get_vocab(), draft.get_vocab())
        self.assertEqual(target.all_special_ids, draft.all_special_ids)
        inputs = [
            target.encode(text * 40, add_special_tokens=False)[:192]
            for text in (
                "Explain how sunlight helps plants grow, using simple factual language. ",
                "Write a cheerful story about a cat exploring a quiet garden. ",
                "List the positive integers in order and explain how counting works. ",
            )
        ]
        self.assertTrue(all(len(ids) == 192 for ids in inputs))
        self.assertEqual(len({tuple(ids) for ids in inputs}), 3)
        self.start_server("ordinary")
        references = []
        for ids in inputs:
            self.assert_idle("ordinary")
            reference_request = self.new_payload(ids, pd=False)
            reference = self.http("ordinary", "POST", "/generate", reference_request)
            references.append(self.assert_completion(reference, reference_request)[0])
            self.assert_idle("ordinary")
        self.stop_server("ordinary")
        self.start_server("prefill")
        self.start_server("decode")
        self.assert_idle("prefill")
        self.assert_idle("decode")
        # Reject an unexpected allocator/configuration before any PD POST.
        self.assert_full_capacity(self.wait_idle_snapshot([]))

        bad = self.new_payload(inputs[0])
        bad["rid"] = self.bad_rid
        responses = self.post_pair(bad)
        first, _ = self.assert_completion(responses["prefill"], bad, prefill=True)
        self.assertEqual(first, references[0][:1])
        self.assertEqual(responses["decode"].status_code, 500)
        self.assertIn(
            "Invalid or unwritten STANDALONE PD draft seed",
            str(responses["decode"].body),
        )
        self.assert_idle("prefill")
        self.assert_idle("decode")
        self.assert_full_capacity(self.wait_idle_snapshot([self.bad_rid]))

        producer = self.one_event("producer", self.bad_rid, role="prefill")
        self.assertTrue(producer["injected"])
        self.assertTrue(math.isfinite(producer["probability_before"][0]))
        self.assertGreater(producer["probability_before"][0], 0)
        self.assertEqual(producer["probability_sent"], [0.0])
        received = self.one_event("transfer_success", self.bad_rid)
        self.assertEqual(received["room"], bad["bootstrap_room"])
        self.assertEqual(producer["room"], received["room"])
        self.assertEqual(received["probability_received"], [0.0])
        committed = self.one_event("commit_after", self.bad_rid)["request"]
        self.assertEqual(committed["finish"], "FINISH_ABORT")
        self.assertEqual(committed["status_code"], 500)
        self.assertEqual(committed["output_ids"], [])
        self.assertTrue(committed["receiver_cleared"])
        allocated = self.one_event("allocated", self.bad_rid)["request"]
        released = self.one_event("release_after", self.bad_rid)
        self.assertFalse(released["request"]["holds_kv"])
        self.assertTrue(released["request"]["kv_released"])
        self.assertTrue(
            set(allocated["kv_ids"]) <= set(released["resources"]["token_free_ids"])
        )
        self.one_event("release_before", self.bad_rid)
        concluded = [
            r
            for r in self.events("decode")
            if r["phase"] == "pop_after" and self.bad_rid in r["outcomes"]
        ]
        self.assertEqual(len(concluded), 1)
        self.assertEqual(concluded[0]["outcomes"], {self.bad_rid: True})
        self.assertNotIn(self.bad_rid, concluded[0]["admitted"])
        self.assertEqual(concluded[0]["cleared_rooms"][self.bad_rid], 0)
        self.assert_full_capacity(concluded[0]["resources"])

        good_requests = []
        accepted = 0
        for ids, expected in zip(inputs[1:], references[1:]):
            body = self.new_payload(ids)
            good_requests.append(body)
            replies = self.post_pair(body)
            first, _ = self.assert_completion(replies["prefill"], body, prefill=True)
            actual, meta = self.assert_completion(replies["decode"], body)
            self.assertEqual(actual, expected)
            self.assertEqual(first, expected[:1])
            self.assertGreaterEqual(meta["spec_verify_ct"], 2)
            accepted += meta["spec_num_correct_drafts"]
            self.assert_idle("prefill")
            self.assert_idle("decode")
            self.assert_full_capacity(
                self.wait_idle_snapshot(
                    [self.bad_rid] + [r["rid"] for r in good_requests]
                )
            )
        self.assertGreater(accepted, 0)
        good_allocations = [
            self.one_event("allocated", r["rid"])["request"] for r in good_requests
        ]
        # Only observe normal allocator decisions; never reorder or reserve slots.
        self.assertEqual(good_allocations[0]["request_slot"], allocated["request_slot"])
        self.assertGreater(good_allocations[0]["generation"], allocated["generation"])
        self.assertTrue(set(allocated["kv_ids"]) & set(good_allocations[0]["kv_ids"]))
        self.assertEqual(
            good_allocations[1]["metadata_slot"], allocated["metadata_slot"]
        )
        for body in good_requests:
            sent = self.one_event("producer", body["rid"], role="prefill")
            self.assertFalse(sent["injected"])
            self.assertGreater(sent["probability_sent"][0], 0)
            self.assertEqual(
                self.one_event("transfer_success", body["rid"])["probability_received"],
                sent["probability_sent"],
            )
        for row in self.events("decode"):
            if row["phase"] in ("factory", "draft", "verify"):
                self.assertNotIn(self.bad_rid, row["rids"])
        for body in good_requests:
            for phase in ("factory", "draft", "verify"):
                self.assertTrue(
                    any(
                        row["phase"] == phase and body["rid"] in row["rids"]
                        for row in self.events("decode")
                    )
                )
        self.record(
            "recovery-summary.json",
            {
                "bad_rid": self.bad_rid,
                "bad_http_status": responses["decode"].status_code,
                "good_rids": [body["rid"] for body in good_requests],
                "ordinary_output_ids": references,
                "accepted_draft_tokens": accepted,
                "rejected_allocation": allocated,
                "good_allocations": good_allocations,
                "final_resources": self.wait_idle_snapshot(
                    [self.bad_rid] + [body["rid"] for body in good_requests]
                ),
                "exact_ordinary_token_match": True,
                "natural_request_metadata_and_kv_reuse": True,
            },
        )


if __name__ == "__main__":
    unittest.main()
