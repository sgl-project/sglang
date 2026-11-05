"""Manual ordinary-PD and EAGLE3 regressions for shared metadata paths.

Run each class separately on one dedicated Hopper GPU (Mooncake CUDA/NVML/FA3)::

    python test/manual/spec/test_spec_pd_shared_paths.py TestOrdinaryPD -f
    python test/manual/spec/test_spec_pd_shared_paths.py TestEagle3PD -f

The EAGLE3 default is the trained Llama-3.1 target/draft pair used by SGLang's
registered PD tests, not a vanilla small language model. Both peers must use
the same target and the same trained draft assets. The existing fixture's
SGLANG_TEST_STANDALONE_PD_TARGET/DRAFT overrides also apply here; set them
separately for each class. OUTPUT_DIR must be fresh per complete unittest run.
Read-only observer D2H fingerprints affect timing; no performance claim is made.
"""

import json
import os
import shutil
import tempfile
import unittest
from pathlib import Path

from standalone_pd_fixture import StandalonePDServerBase
from transformers import AutoConfig, AutoTokenizer

from sglang.test.test_utils import (
    DEFAULT_DRAFT_MODEL_EAGLE3,
    DEFAULT_TARGET_MODEL_EAGLE3,
)


class SharedPDBase(StandalonePDServerBase):
    # No test methods on this base: only the two concrete classes are collected.
    algorithm = None

    @classmethod
    def setUpClass(cls):
        output = os.environ.get("SGLANG_TEST_STANDALONE_PD_OUTPUT_DIR")
        if output:
            cls.observer_dir = Path(output).resolve() / "fault-events"
        else:
            cls.observer_dir = Path(tempfile.mkdtemp(prefix="shared-pd-observer-"))
            cls.addClassCleanup(shutil.rmtree, cls.observer_dir)
        cls.event_paths = {
            role: cls.observer_dir / f"{role}.jsonl" for role in ("prefill", "decode")
        }
        for role in ("prefill", "decode"):
            hooks = json.dumps(
                [
                    {
                        "name": "shared-pd-observer",
                        "target_modules": ["*logits_processor*"],
                        "hook_factory": "shared_pd_observer:make_hook",
                        "config": {"role": role, "log": str(cls.event_paths[role])},
                    }
                ]
            )
            setattr(cls, f"extra_{role}_args", ("--forward-hooks", hooks))
        super().setUpClass()
        cls.observer_dir.mkdir(exist_ok=True)

    @classmethod
    def server_args(cls, role):
        args = super().server_args(role)
        if role == "ordinary":
            return args
        if cls.algorithm is None:
            # Remove only the original fixture's explicit speculative options;
            # keep its real PD transport, bootstrap and lifecycle parameters.
            spec_keys = {
                "--speculative-algorithm",
                "--speculative-draft-model-path",
                "--speculative-num-steps",
                "--speculative-eagle-topk",
                "--speculative-num-draft-tokens",
            }
            result = []
            i = 0
            while i < len(args):
                if args[i] in spec_keys:
                    i += 2
                else:
                    result.append(args[i])
                    i += 1
            return result
        index = args.index("--speculative-algorithm")
        args[index + 1] = cls.algorithm
        return args

    def events(self, role):
        raw = self.event_paths[role].read_bytes()
        self.assertTrue(raw.endswith(b"\n"), "Incomplete observer log tail")
        rows = [json.loads(line) for line in raw.splitlines()]
        self.assertEqual(
            [row["sequence"] for row in rows], list(range(1, len(rows) + 1))
        )
        self.assertLessEqual(len(rows), 1024)
        self.assertEqual(len({row["pid"] for row in rows}), 1)
        return rows

    def one_event(self, role, phase, rid):
        rows = [
            row
            for row in self.events(role)
            if row["phase"] == phase and row.get("rid") == rid
        ]
        self.assertEqual(len(rows), 1, (role, phase, rid, rows))
        return rows[0]

    def check_metadata(self, request, handoff_token):
        rid = request["rid"]
        producer = self.one_event("prefill", "producer", rid)
        received = self.one_event("decode", "received", rid)
        committed = self.one_event("decode", "committed", rid)
        for row in (producer, received, committed):
            self.assertEqual(row["room"], request["bootstrap_room"])
        self.assertEqual(producer["token"], handoff_token)
        self.assertEqual(received["token"], handoff_token)
        self.assertEqual(committed["output_ids"], [handoff_token])
        self.assertTrue(committed["receiver_cleared"])
        self.assertFalse(committed["finished"])
        for key in ("topk", "indices", "hidden"):
            self.assertEqual(producer[key], received[key], key)
        if self.algorithm is None:
            for key in ("active_topk", "active_indices", "active_hidden"):
                self.assertIsNone(producer[key])
            for key in ("topk", "indices", "hidden"):
                self.assertEqual(received[key]["nonzero"], 0)
            self.assertIsNone(committed["hidden"])
            self.assertFalse(
                any(
                    row["phase"] in ("eagle_factory", "draft", "verify")
                    for row in self.events("decode")
                )
            )
        else:
            self.assertEqual(producer["active_topk"], received["active_topk"])
            self.assertEqual(producer["active_indices"], received["active_indices"])
            self.assertGreater(producer["active_topk"]["nonzero"], 0)
            self.assertGreater(producer["hidden"]["nonzero"], 0)
            self.assertEqual(producer["active_hidden"], producer["hidden"])
            self.assertEqual(committed["hidden"], received["hidden"])
            factory = self.one_event("decode", "eagle_factory", rid)
            self.assertEqual(factory["hidden"], received["hidden"])
            self.assertEqual(factory["topk"], received["active_topk"])
            self.assertEqual(factory["indices"], received["active_indices"])
            self.assertEqual(factory["capture_hidden_mode"], "LAST")
            for phase in ("draft", "verify", "inner_draft_decode"):
                self.assertGreaterEqual(
                    sum(
                        row["phase"] == phase and rid in row.get("rids", [])
                        for row in self.events("decode")
                    ),
                    2,
                )
            models = {
                row["model_class"]
                for row in self.events("decode")
                if row["phase"] == "inner_draft_decode" and rid in row["rids"]
            }
            self.assertTrue(models)
            self.assertTrue(all("eagle3" in name.lower() for name in models), models)

    def run_shared_regression(self):
        tokenizer = AutoTokenizer.from_pretrained(self.target_model)
        short = [
            "Explain why the sky appears blue in one short paragraph.",
            "Continue this list in words: one, two, three, four, five,",
        ]
        inputs = [tokenizer.encode(text, add_special_tokens=False) for text in short]
        long_ids = tokenizer.encode(
            "Describe how sunlight, water and healthy soil help a garden grow. " * 100,
            add_special_tokens=False,
        )[:700]
        self.assertEqual(len(long_ids), 700)
        inputs.append(long_ids)
        model_configs = {
            "target": AutoConfig.from_pretrained(self.target_model).to_dict()
        }
        if self.algorithm is not None:
            model_configs["draft"] = AutoConfig.from_pretrained(
                self.draft_model
            ).to_dict()
        self.record("shared-model-configs.json", model_configs)
        self.start_server("ordinary")
        references = []
        for ids in inputs:
            self.assert_idle("ordinary")
            request = self.new_payload(ids, pd=False)
            response = self.http("ordinary", "POST", "/generate", request)
            references.append(self.assert_completion(response, request)[0])
            self.assert_idle("ordinary")
        self.stop_server("ordinary")
        self.start_server("prefill")
        self.start_server("decode")
        accepted = 0
        cases = []
        for ids, expected in zip(inputs, references):
            self.assert_idle("prefill")
            self.assert_idle("decode")
            request = self.new_payload(ids)
            responses = self.post_pair(request)
            first, _ = self.assert_completion(
                responses["prefill"], request, prefill=True
            )
            actual, meta = self.assert_completion(responses["decode"], request)
            self.assertEqual(first, expected[:1])
            self.assertEqual(actual, expected)
            self.assert_idle("prefill")
            self.assert_idle("decode")
            self.check_metadata(request, first[0])
            if self.algorithm is not None:
                self.assertGreaterEqual(meta["spec_verify_ct"], 2)
                self.assertGreaterEqual(meta["spec_num_correct_drafts"], 0)
                accepted += meta["spec_num_correct_drafts"]
            else:
                self.assertIn(meta.get("spec_verify_ct"), (None, 0))
                self.assertIn(meta.get("spec_num_correct_drafts"), (None, 0))
            chunks = [
                row
                for row in self.events("prefill")
                if row["phase"] == "chunk_sent" and row["rid"] == request["rid"]
            ]
            self.assertEqual(sum(row["last_chunk"] for row in chunks), 1)
            if len(ids) == 700:
                self.assertTrue(
                    any(
                        not row["last_chunk"] and row["end"] > row["start"]
                        for row in chunks
                    )
                )
            cases.append(
                {
                    "rid": request["rid"],
                    "prompt_tokens": len(ids),
                    "output_ids": actual,
                    "spec_verify_ct": meta.get("spec_verify_ct"),
                    "accepted_draft_tokens": meta.get("spec_num_correct_drafts"),
                }
            )
        if self.algorithm is not None:
            self.assertGreater(accepted, 0)
        self.record(
            "shared-summary.json",
            {
                "algorithm": self.algorithm,
                "cases": cases,
                "accepted_draft_tokens": accepted,
                "exact_ordinary_token_match": True,
                "metadata_observations_checked": True,
            },
        )


class TestOrdinaryPD(SharedPDBase):
    def test_three_greedy_completions_match_ordinary(self):
        self.run_shared_regression()


class TestEagle3PD(SharedPDBase):
    algorithm = "EAGLE3"
    target_model = DEFAULT_TARGET_MODEL_EAGLE3
    draft_model = DEFAULT_DRAFT_MODEL_EAGLE3

    def test_three_greedy_completions_match_ordinary(self):
        self.run_shared_regression()


if __name__ == "__main__":
    unittest.main()
