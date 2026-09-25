"""CPU tests for the deterministic PPU Answer evaluator."""

import copy
import json
import os
import sys
import tempfile
import unittest
import xml.etree.ElementTree as ET
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from sglang.test.ci.ci_register import register_cpu_ci, register_ppu_ci
from sglang.test.kits.answer_eval_kit import (
    AnswerEvalError,
    CandidateRequestError,
    _evaluate_regression_gate,
    _repeated_ngram_coverage,
    answer_expected_hardware,
    answer_node_count,
    answer_server_environment,
    apply_cross_case_checks,
    build_answer_server_args,
    build_report,
    canonical_digest,
    default_provenance,
    evaluate_case,
    load_json,
    normalize_answer,
    parse_thinking_blocks,
    redact_report,
    render_candidates,
    render_junit,
    render_summary,
    request_chat_completion,
    resolve_distributed_runtime,
    resolve_evaluation_paths,
    validate_annotation_record,
    validate_baseline,
    validate_dataset,
    validate_profile,
    validate_test_config,
    write_report_files,
)

# The on-machine driver, imported for the multi-node exchange tests at the end of
# this file.  It needs torch, which the evaluator deliberately does not, so a
# host without it skips those tests rather than failing to collect this file.
try:
    from sglang.test.kits import answer_suite_kit
except ImportError:
    answer_suite_kit = None

# Hardware-free (pure stdlib evaluator), so the CPU suite owns it. It is also
# registered on the PPU per-commit chain: this file is the only guard for the
# evaluator that gates the nightly-answer-*-ppu suites, and a PR that breaks a
# threshold or a redaction rule should turn the PPU gate red immediately rather
# than surface hours later in the nightly Answer run.
register_cpu_ci(est_time=3, suite="base-a-test-cpu")
register_ppu_ci(est_time=10, suite="stage-b-test-1-gpu-ppu")

DATA_ROOT = Path(__file__).parent
CONFIG_DIR = DATA_ROOT / "configs"
DATASET_DIR = DATA_ROOT / "dataset"


class TestPPUAnswerEval(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.test_config = load_json(CONFIG_DIR / "qwen3.5" / "397b-a17b-w8a8-int8.json")
        cls.dataset = load_json(DATASET_DIR / "answer_cases_zh_v1.json")
        cls.profile = load_json(DATASET_DIR / "quality_profile.json")
        cls.cases = {case["id"]: case for case in cls.dataset["cases"]}

    def evaluate(self, case_id, answer, finish_reason="stop"):
        return evaluate_case(
            self.cases[case_id],
            answer,
            finish_reason,
            self.profile,
            self.dataset["revision"],
        )

    def reason_codes(self, result):
        return {finding["reason_code"] for finding in result["findings"]}

    @staticmethod
    def _gate_results(passed, failed_model, failed_critical):
        """Synthesize the minimal per-case results the gate reads.

        Only case_id/verdict/findings matter to _evaluate_regression_gate, so a
        test can name exactly which cases passed, missed for model-capability
        reasons, or broke on infrastructure without driving a full corpus.
        """

        results = []
        for case_id in passed:
            results.append({"case_id": case_id, "verdict": "passed", "findings": []})
        for case_id in failed_model:
            results.append(
                {
                    "case_id": case_id,
                    "verdict": "failed",
                    "findings": [
                        {
                            "action": "hard_fail",
                            "severity": "model",
                            "reason_code": "fact_rule_failed",
                        }
                    ],
                }
            )
        for case_id in failed_critical:
            results.append(
                {
                    "case_id": case_id,
                    "verdict": "failed",
                    "findings": [
                        {
                            "action": "hard_fail",
                            "severity": "critical",
                            "reason_code": "finish_reason_length",
                        }
                    ],
                }
            )
        return results

    def test_baseline_tolerates_only_declared_model_failures(self):
        # A known model miss stays green; the per-case verdict is still failed,
        # but the run does not regress and the tolerated case is surfaced.
        results = self._gate_results(
            passed=[f"c{i}" for i in range(9)],
            failed_model=["deepseek-letter-count"],
            failed_critical=[],
        )
        gate = _evaluate_regression_gate(
            results,
            {"known_failures": ["deepseek-letter-count"], "min_score": 9},
        )
        self.assertFalse(gate["regressed"])
        self.assertEqual(gate["known_failures_hit"], ["deepseek-letter-count"])
        self.assertEqual(gate["new_regressions"], [])
        self.assertEqual(gate["passed"], 9)

    def test_new_model_miss_reddens_even_at_the_score_floor(self):
        # A miss on a case outside the allowlist is a new regression, and it
        # reddens even though the passing count still meets min_score -- the
        # allowlist, not the raw score, is the primary signal.
        results = self._gate_results(
            passed=[f"c{i}" for i in range(9)],
            failed_model=["henan-bordering-provinces"],
            failed_critical=[],
        )
        gate = _evaluate_regression_gate(
            results,
            {"known_failures": ["deepseek-letter-count"], "min_score": 9},
        )
        self.assertTrue(gate["regressed"])
        self.assertEqual(gate["new_regressions"], ["henan-bordering-provinces"])
        self.assertFalse(gate["score_below_baseline"])

    def test_infrastructure_failure_is_never_tolerated(self):
        # Even a case in known_failures reddens when it breaks on a critical
        # (infrastructure) finding rather than a model-capability miss.
        results = self._gate_results(
            passed=[f"c{i}" for i in range(9)],
            failed_model=[],
            failed_critical=["deepseek-letter-count"],
        )
        gate = _evaluate_regression_gate(
            results,
            {"known_failures": ["deepseek-letter-count"], "min_score": 9},
        )
        self.assertTrue(gate["regressed"])
        self.assertEqual(gate["critical_failures"], ["deepseek-letter-count"])
        self.assertEqual(gate["known_failures_hit"], [])

    def test_score_floor_reddens_without_a_new_miss(self):
        # Two tolerated misses, no new case, but the passing count falls below
        # min_score -- the floor guard still reddens.
        results = self._gate_results(
            passed=[f"c{i}" for i in range(8)],
            failed_model=["deepseek-letter-count", "red-ball-probability"],
            failed_critical=[],
        )
        gate = _evaluate_regression_gate(
            results,
            {
                "known_failures": ["deepseek-letter-count", "red-ball-probability"],
                "min_score": 9,
            },
        )
        self.assertTrue(gate["regressed"])
        self.assertTrue(gate["score_below_baseline"])
        self.assertEqual(gate["new_regressions"], [])

    def test_absent_baseline_keeps_strict_behaviour(self):
        # With no baseline any miss reddens, matching the pre-baseline gate.
        results = self._gate_results(
            passed=[f"c{i}" for i in range(9)],
            failed_model=["deepseek-letter-count"],
            failed_critical=[],
        )
        gate = _evaluate_regression_gate(results, None)
        self.assertTrue(gate["regressed"])
        self.assertFalse(gate["baseline_applied"])
        self.assertEqual(gate["new_regressions"], ["deepseek-letter-count"])

    def test_case_that_beats_its_baseline_is_flagged_not_failed(self):
        # A known-failing case that now passes is an unexpected pass: green, but
        # surfaced so the baseline can be tightened.
        results = self._gate_results(
            passed=["deepseek-letter-count"] + [f"c{i}" for i in range(9)],
            failed_model=[],
            failed_critical=[],
        )
        gate = _evaluate_regression_gate(
            results, {"known_failures": ["deepseek-letter-count"], "min_score": 7}
        )
        self.assertFalse(gate["regressed"])
        self.assertEqual(gate["unexpected_passes"], ["deepseek-letter-count"])

    def test_fact_rule_findings_carry_model_severity(self):
        # A wrong fact answer is a model-severity miss; a bad finish_reason on
        # the same case is a critical finding.
        wrong = self.evaluate("deepseek-letter-count", "答案是 5")
        fact = next(
            finding
            for finding in wrong["findings"]
            if finding["reason_code"] == "fact_rule_failed"
        )
        self.assertEqual(fact["severity"], "model")
        truncated = self.evaluate("deepseek-letter-count", "答案是 4", "length")
        finish = next(
            finding
            for finding in truncated["findings"]
            if finding["reason_code"] == "finish_reason_length"
        )
        self.assertEqual(finish["severity"], "critical")

    def test_validate_baseline_rejects_malformed_declarations(self):
        validate_baseline(None)
        validate_baseline({"known_failures": ["a"], "min_score": 7})
        with self.assertRaises(AnswerEvalError):
            validate_baseline({"known_failures": "a"})
        with self.assertRaises(AnswerEvalError):
            validate_baseline({"known_failures": ["a", "a"]})
        with self.assertRaises(AnswerEvalError):
            validate_baseline({"known_failures": [1]})
        with self.assertRaises(AnswerEvalError):
            validate_baseline({"min_score": -1})
        with self.assertRaises(AnswerEvalError):
            validate_baseline({"min_score": True})

    @staticmethod
    def rendezvous_for(config):
        """A stand-in for what the launcher injects, for configs that need one."""

        return resolve_distributed_runtime(
            config,
            {"NODE_RANK": "0", "MASTER_ADDR": "rank0.example", "MASTER_PORT": "29500"},
        )

    def test_data_contract_and_digest_are_stable(self):
        validate_test_config(self.test_config)
        validate_dataset(self.dataset)
        validate_profile(self.profile)
        self.assertEqual(
            self.test_config["server"]["parameters"]["tp_size"],
            len(self.test_config["hardware"]["visible_devices"]),
        )
        # The contract under test is the flag spelling and the argument order.
        # The values belong to the reviewed configuration, so they are read from
        # it rather than copied here, where a copy would silently become a
        # second, competing source of truth.
        parameters = self.test_config["server"]["parameters"]
        self.assertEqual(
            build_answer_server_args(self.test_config),
            [
                "--trust-remote-code",
                "--tp-size",
                str(parameters["tp_size"]),
                "--attention-backend",
                parameters["attention_backend"],
                "--mem-fraction-static",
                str(parameters["mem_fraction_static"]),
                "--quantization",
                parameters["quantization"],
                "--reasoning-parser",
                parameters["reasoning_parser"],
                "--dist-timeout",
                str(parameters["dist_timeout"]),
                "--watchdog-timeout",
                str(parameters["watchdog_timeout"]),
                "--served-model-name",
                self.test_config["model"]["served_model_name"],
            ],
        )
        # Likewise, provenance depends on the shape of the hardware string, not
        # on one particular accelerator generation; the rendering rule itself is
        # pinned by test_expected_hardware_renders_the_declared_topology.
        self.assertRegex(
            answer_expected_hardware(self.test_config), r"^[0-9a-z.]+-\d+x[0-9.]+g$"
        )
        self.assertEqual(len(canonical_digest(self.dataset)), 64)
        self.assertEqual(len({case["id"] for case in self.dataset["cases"]}), 10)

    def test_every_reviewed_test_config_is_executable(self):
        # One config per nightly Answer job, and only the workflow names the ones
        # this class does not load, so they are validated here instead of first
        # failing on the machine after a checkpoint has been warmed. The walk is
        # recursive because the configs are filed per model family, and it also
        # asserts that layout: a config that lands outside a family directory
        # would be tested here but invisible to a reader of the tree.
        config_paths = sorted(CONFIG_DIR.rglob("*.json"))
        self.assertGreaterEqual(len(config_paths), 2)
        test_ids = set()
        for config_path in config_paths:
            with self.subTest(config=config_path.relative_to(CONFIG_DIR).as_posix()):
                self.assertEqual(
                    config_path.parent.parent,
                    CONFIG_DIR,
                    "a config must live in configs/<model family>/",
                )
                config = load_json(config_path)
                validate_test_config(config)
                test_ids.add(config["test_id"])
                for field, resolved in zip(
                    ("dataset", "quality_profile"),
                    resolve_evaluation_paths(config, DATA_ROOT),
                ):
                    self.assertTrue(
                        resolved.is_file(),
                        f"evaluation.{field} does not resolve to a file",
                    )
                args = build_answer_server_args(
                    config, distributed=self.rendezvous_for(config)
                )
                # The group a launch claims is tp_size * pp_size, which is what
                # SGLang itself checks against the node count, so the product is
                # what has to account for every device the job holds.
                parameters = config["server"]["parameters"]
                self.assertEqual(
                    int(args[args.index("--tp-size") + 1])
                    * (
                        int(args[args.index("--pp-size") + 1])
                        if "--pp-size" in args
                        else 1
                    ),
                    len(config["hardware"]["visible_devices"])
                    * answer_node_count(config),
                )
                # A config that stays on pure tensor parallelism must not grow an
                # explicit --pp-size 1, so the flag appears exactly when the
                # config asks for the layers to be split.
                self.assertEqual("--pp-size" in args, "pp_size" in parameters)
                # The schema only requires a non-empty string, so a value that
                # argparse would reject would otherwise surface as a server that
                # never starts. server_args is imported here rather than at
                # module scope to keep the evaluator's own import path stdlib
                # only; "unquant" is that module's spelling for an explicit
                # opt-out, which is how a BF16 checkpoint satisfies the field.
                # A null is not a value argparse ever sees: it asks for the flag
                # to be left off, so the checkpoint's own declaration stands,
                # and the assertion is that no flag was emitted.
                from sglang.srt.server_args import QUANTIZATION_CHOICES

                quantization = config["server"]["parameters"]["quantization"]
                if quantization is None:
                    self.assertNotIn("--quantization", args)
                else:
                    self.assertIn(quantization, QUANTIZATION_CHOICES)
                self.assertEqual(
                    args[args.index("--watchdog-timeout") + 1],
                    str(config["server"]["parameters"]["watchdog_timeout"]),
                )
        self.assertEqual(len(test_ids), len(config_paths), "test_id must be unique")

    def test_expected_hardware_renders_the_declared_topology(self):
        config = {
            "hardware": {
                "generation": "zw810e",
                "visible_devices": [0, 1, 2, 3],
                "memory_gib_per_device": 96,
            }
        }
        self.assertEqual(answer_expected_hardware(config), "zw810e-4x96g")

        # A capacity that JSON carries as a float must not leak a ".0" into
        # provenance, while a genuinely fractional capacity must survive.
        config["hardware"]["memory_gib_per_device"] = 96.0
        self.assertEqual(answer_expected_hardware(config), "zw810e-4x96g")
        config["hardware"]["memory_gib_per_device"] = 97.5
        self.assertEqual(answer_expected_hardware(config), "zw810e-4x97.5g")

        config["hardware"]["generation"] = "btv1.5"
        config["hardware"]["visible_devices"] = [0, 1, 2, 3, 4, 5, 6, 7]
        config["hardware"]["memory_gib_per_device"] = 144
        self.assertEqual(answer_expected_hardware(config), "btv1.5-8x144g")

        # A multi-node contract has to state both dimensions: eight devices on
        # each of four nodes is not the same machine as thirty-two on one, and a
        # report that rendered them alike could not be used to tell them apart.
        config["hardware"]["nnodes"] = 4
        self.assertEqual(answer_expected_hardware(config), "btv1.5-4nx8x144g")
        # An explicitly declared single node keeps the original rendering, so the
        # contract already recorded by the existing entries does not move.
        config["hardware"]["nnodes"] = 1
        self.assertEqual(answer_expected_hardware(config), "btv1.5-8x144g")

    def test_multi_node_config_binds_the_rendezvous_it_is_handed(self):
        single_node = self.test_config
        self.assertIsNone(resolve_distributed_runtime(single_node, {}))
        self.assertEqual(answer_node_count(single_node), 1)

        config = copy.deepcopy(single_node)
        config["hardware"]["nnodes"] = 4
        config["server"]["parameters"]["tp_size"] = 4 * len(
            config["hardware"]["visible_devices"]
        )
        validate_test_config(config)
        self.assertEqual(answer_node_count(config), 4)

        # The rank and the address are runtime facts, so they come from the
        # variables the action injects per pod rather than from the config.
        environ = {
            "NODE_RANK": "2",
            "MASTER_ADDR": "rank0.headless.svc.cluster.local",
            "MASTER_PORT": "29500",
        }
        distributed = resolve_distributed_runtime(config, environ)
        self.assertEqual(
            distributed,
            {
                "nnodes": 4,
                "node_rank": 2,
                "dist_init_addr": "rank0.headless.svc.cluster.local:29500",
            },
        )

        args = build_answer_server_args(config, distributed=distributed)
        self.assertEqual(
            args[-6:],
            [
                "--nnodes",
                "4",
                "--node-rank",
                "2",
                "--dist-init-addr",
                "rank0.headless.svc.cluster.local:29500",
            ],
        )
        # The topology flags are purely additive: everything a single-node launch
        # would pass is still passed, in the same order.
        without_topology = copy.deepcopy(config)
        del without_topology["hardware"]["nnodes"]
        self.assertEqual(args[:-6], build_answer_server_args(without_topology))

        # The measured defect this override exists for: the action asks for host
        # networking without setting dnsPolicy, so MASTER_ADDR does not resolve
        # inside the pods and the caller has to supply rank 0's address itself.
        environ["SGLANG_PPU_ANSWER_DIST_INIT_ADDR"] = "215.193.196.51:29500"
        environ["SGLANG_PPU_ANSWER_NODE_RANK"] = "3"
        overridden = resolve_distributed_runtime(config, environ)
        self.assertEqual(overridden["dist_init_addr"], "215.193.196.51:29500")
        self.assertEqual(overridden["node_rank"], 3)

        # A launcher that states its group size has to agree with the reviewed
        # one. The action states it, as NNODES, and a single-node launch of it
        # states 1, which is why the disagreement is worth catching here rather
        # than after the boards have been claimed.
        agreeing = dict(environ, NNODES="4")
        self.assertEqual(resolve_distributed_runtime(config, agreeing), overridden)
        for launched, message in (
            ("2", "started 2 node"),
            ("1", "started 1 node"),
            ("four", "must be an integer"),
        ):
            with self.subTest(NNODES=launched):
                with self.assertRaisesRegex(AnswerEvalError, message):
                    resolve_distributed_runtime(config, dict(environ, NNODES=launched))
        # An empty value is a launcher that says nothing, not one that says zero.
        self.assertEqual(
            resolve_distributed_runtime(config, dict(environ, NNODES="")),
            overridden,
        )
        # And a single-node config is never asked about its group size, so the
        # NNODES=1 the action injects for every single-board entry cannot make
        # one of those fail.
        self.assertIsNone(resolve_distributed_runtime(single_node, {"NNODES": "1"}))

    def test_multi_node_launch_refuses_an_unusable_rendezvous(self):
        config = copy.deepcopy(self.test_config)
        config["hardware"]["nnodes"] = 4
        config["server"]["parameters"]["tp_size"] = 4 * len(
            config["hardware"]["visible_devices"]
        )
        address = {"MASTER_ADDR": "rank0.example", "MASTER_PORT": "29500"}
        for environ, message in (
            ({}, "rank of this node"),
            ({"NODE_RANK": "4", **address}, "outside the"),
            ({"NODE_RANK": "-1", **address}, "outside the"),
            ({"NODE_RANK": "first", **address}, "must be an integer"),
            ({"NODE_RANK": "0"}, "rendezvous address"),
            ({"NODE_RANK": "0", "MASTER_ADDR": "rank0.example"}, "rendezvous address"),
            (
                {"NODE_RANK": "0", "MASTER_ADDR": "rank0.example", "MASTER_PORT": "n"},
                "port must be an integer",
            ),
            (
                {"NODE_RANK": "0", "MASTER_ADDR": "rank0.example", "MASTER_PORT": "0"},
                "out of range",
            ),
            (
                {"NODE_RANK": "0", "SGLANG_PPU_ANSWER_DIST_INIT_ADDR": "10.0.0.1"},
                "host:port",
            ),
        ):
            with self.subTest(environ=environ):
                with self.assertRaisesRegex(AnswerEvalError, message):
                    resolve_distributed_runtime(config, environ)

        # A caller that forgets the rendezvous must not get four independent
        # rank 0 servers, each trying to fit the whole checkpoint on eight
        # devices, and a single-node config must not be handed one either.
        distributed = self.rendezvous_for(config)
        with self.assertRaisesRegex(AnswerEvalError, "must be launched with"):
            build_answer_server_args(config)
        with self.assertRaisesRegex(AnswerEvalError, "must not be handed"):
            build_answer_server_args(self.test_config, distributed=distributed)
        with self.assertRaisesRegex(AnswerEvalError, "spans 2 nodes"):
            build_answer_server_args(config, distributed={**distributed, "nnodes": 2})

    def test_pipeline_parallel_config_splits_the_group_it_declares(self):
        # Measured on the cluster: tensor parallelism alone cannot serve the
        # 2.4T FP8 checkpoint across 32 devices.  Its moe_intermediate_size is
        # 2048 and its FP8 scales are blocked at 128, so 2048 // 32 == 64 is
        # rejected by the FP8 weight loader, and its linear_num_key_heads is 16,
        # which 32 does not divide either.  Splitting the layers across the nodes
        # and keeping tensor parallelism inside each one is the shape the vendor
        # documents for that checkpoint, and it is the reason pp_size exists
        # here at all.
        config = copy.deepcopy(self.test_config)
        config["hardware"]["nnodes"] = 4
        config["server"]["parameters"]["pp_size"] = 4
        validate_test_config(config)

        tp_size = config["server"]["parameters"]["tp_size"]
        args = build_answer_server_args(config, distributed=self.rendezvous_for(config))
        index = args.index("--tp-size")
        self.assertEqual(
            args[index : index + 4],
            ["--tp-size", str(tp_size), "--pp-size", "4"],
        )
        # The parameter is optional, and an entry that does not name it keeps the
        # command line it had before pp_size was a field at all.
        self.assertNotIn("--pp-size", build_answer_server_args(self.test_config))

        # A degree that leaves devices unaccounted for, or claims more than the
        # job holds, is the failure this check exists to catch before the boards
        # are claimed: the server would otherwise wait for ranks that never come.
        for pp_size in (2, 8):
            invalid = copy.deepcopy(config)
            invalid["server"]["parameters"]["pp_size"] = pp_size
            with self.subTest(pp_size=pp_size):
                with self.assertRaisesRegex(AnswerEvalError, "visible device count"):
                    validate_test_config(invalid)

        for pp_size in (0, -1, 2.0, True, "4", None):
            invalid = copy.deepcopy(config)
            invalid["server"]["parameters"]["pp_size"] = pp_size
            with self.subTest(pp_size=pp_size):
                with self.assertRaisesRegex(AnswerEvalError, "pp_size"):
                    validate_test_config(invalid)

        # Admitting one optional parameter must not turn the schema into an open
        # set: a parameter the kit does not translate would be read as honoured
        # and silently dropped from the command line.
        invalid = copy.deepcopy(config)
        invalid["server"]["parameters"]["ep_size"] = 8
        with self.assertRaisesRegex(AnswerEvalError, "exactly the supported"):
            validate_test_config(invalid)

        invalid = copy.deepcopy(config)
        del invalid["server"]["parameters"]["tp_size"]
        with self.assertRaisesRegex(AnswerEvalError, "exactly the supported"):
            validate_test_config(invalid)

    def test_execution_config_rejects_inconsistent_or_unsafe_values(self):
        invalid = copy.deepcopy(self.test_config)
        invalid["server"]["parameters"]["tp_size"] = 4
        with self.assertRaisesRegex(AnswerEvalError, "visible device count"):
            validate_test_config(invalid)

        # tp_size spans the whole group, so a node count that appears without a
        # matching tp_size is the same inconsistency.
        invalid = copy.deepcopy(self.test_config)
        invalid["hardware"]["nnodes"] = 4
        with self.assertRaisesRegex(AnswerEvalError, "visible device count"):
            validate_test_config(invalid)

        for nnodes in (0, -1, 2.0, True, "4"):
            invalid = copy.deepcopy(self.test_config)
            invalid["hardware"]["nnodes"] = nnodes
            with self.subTest(nnodes=nnodes):
                with self.assertRaisesRegex(AnswerEvalError, "nnodes"):
                    validate_test_config(invalid)

        invalid = copy.deepcopy(self.test_config)
        invalid["evaluation"]["dataset"] = "../private/prompts.json"
        with self.assertRaisesRegex(AnswerEvalError, "relative"):
            validate_test_config(invalid)

        # A null quantization is the one accepted omission, and the flag simply
        # disappears from the command line.  An empty string is not the same
        # thing: it would reach argparse and be rejected there, on the machine,
        # after the checkpoint had been warmed.
        omitted = copy.deepcopy(self.test_config)
        omitted["server"]["parameters"]["quantization"] = None
        validate_test_config(omitted)
        self.assertNotIn("--quantization", build_answer_server_args(omitted))

        invalid = copy.deepcopy(self.test_config)
        invalid["server"]["parameters"]["quantization"] = ""
        with self.assertRaisesRegex(AnswerEvalError, "quantization"):
            validate_test_config(invalid)

        for watchdog_timeout in (0, -1, "600", True, None):
            invalid = copy.deepcopy(self.test_config)
            invalid["server"]["parameters"]["watchdog_timeout"] = watchdog_timeout
            with self.subTest(watchdog_timeout=watchdog_timeout):
                with self.assertRaisesRegex(AnswerEvalError, "watchdog_timeout"):
                    validate_test_config(invalid)

    def test_attention_backends_may_be_named_per_phase(self):
        # The Kimi-K2.6 case serves prefill and decode with different kernels and
        # states no unified backend.  SGLang reads the pair as the override of the
        # unified choice rather than as an addition to it, so a config that named
        # one as well would describe a backend the run does not use.
        config = copy.deepcopy(self.test_config)
        parameters = config["server"]["parameters"]
        parameters["attention_backend"] = None
        parameters["prefill_attention_backend"] = "fa3"
        parameters["decode_attention_backend"] = "flashmla"
        validate_test_config(config)

        args = build_answer_server_args(config)
        self.assertNotIn("--attention-backend", args)
        index = args.index("--prefill-attention-backend")
        self.assertEqual(
            args[index : index + 4],
            [
                "--prefill-attention-backend",
                "fa3",
                "--decode-attention-backend",
                "flashmla",
            ],
        )
        # Both names have to be ones argparse accepts, or the config would only
        # fail on the machine, after the checkpoint had been warmed.
        from sglang.srt.server_args import ATTENTION_BACKEND_CHOICES

        for field in ("prefill_attention_backend", "decode_attention_backend"):
            self.assertIn(parameters[field], ATTENTION_BACKEND_CHOICES)

        # A null with neither phase named is the omission this check exists for:
        # the server would silently fall back to a default no one reviewed.
        invalid = copy.deepcopy(self.test_config)
        invalid["server"]["parameters"]["attention_backend"] = None
        with self.assertRaisesRegex(AnswerEvalError, "may only be null"):
            validate_test_config(invalid)

        # Only the unified backend is nullable; a null in the pair states nothing
        # at all, and an empty string would reach argparse.
        for field in ("prefill_attention_backend", "decode_attention_backend"):
            for value in (None, "", 3):
                invalid = copy.deepcopy(config)
                invalid["server"]["parameters"][field] = value
                with self.subTest(field=field, value=value):
                    with self.assertRaisesRegex(AnswerEvalError, field):
                        validate_test_config(invalid)

    def test_sparse_attention_parameters_need_the_backend_that_reads_them(self):
        # The GLM cases drive the sparse backend with a prefill kernel, a decode
        # kernel and, on the context-parallel entry, a split mode.  All three are
        # read only by that backend, so a config that named them alongside a dense
        # one would state settings the run ignores.
        config = copy.deepcopy(self.test_config)
        parameters = config["server"]["parameters"]
        parameters["attention_backend"] = "dsa"
        parameters["dsa_prefill_backend"] = "flashmla_sparse"
        parameters["dsa_decode_backend"] = "flashmla_kv"
        parameters["dsa_prefill_cp_mode"] = "round-robin-split"
        parameters["enable_dsa_prefill_context_parallel"] = True
        parameters["attn_cp_size"] = len(config["hardware"]["visible_devices"])
        validate_test_config(config)

        args = build_answer_server_args(config)
        index = args.index("--attention-backend")
        self.assertEqual(
            args[index : index + 10],
            [
                "--attention-backend",
                "dsa",
                "--dsa-prefill-backend",
                "flashmla_sparse",
                "--dsa-decode-backend",
                "flashmla_kv",
                "--dsa-prefill-cp-mode",
                "round-robin-split",
                "--attn-cp-size",
                str(parameters["attn_cp_size"]),
            ],
        )
        self.assertIn("--enable-dsa-prefill-context-parallel", args)
        # And the values are the ones argparse accepts under those names, which
        # is what makes the `nsa_*` to `dsa_*` rename visible here rather than on
        # the machine: the old spellings survive only as deprecated aliases.
        from sglang.srt.server_args import (
            DSA_CHOICES,
            DSA_PREFILL_CP_SPLIT_CHOICES,
        )

        self.assertIn(parameters["dsa_prefill_backend"], DSA_CHOICES)
        self.assertIn(parameters["dsa_decode_backend"], DSA_CHOICES)
        self.assertIn(parameters["dsa_prefill_cp_mode"], DSA_PREFILL_CP_SPLIT_CHOICES)

        # `nsa` is accepted as the backend name because SGLang still accepts it,
        # so a config copied verbatim from an internal case is not rejected here.
        alias = copy.deepcopy(config)
        alias["server"]["parameters"]["attention_backend"] = "nsa"
        validate_test_config(alias)

        invalid = copy.deepcopy(config)
        invalid["server"]["parameters"]["attention_backend"] = "fa3"
        with self.assertRaisesRegex(AnswerEvalError, "sparse attention"):
            validate_test_config(invalid)

        # The split mode describes how prefill context parallelism divides a
        # sequence, so without that switch it describes nothing.
        invalid = copy.deepcopy(config)
        del invalid["server"]["parameters"]["enable_dsa_prefill_context_parallel"]
        with self.assertRaisesRegex(
            AnswerEvalError, "enable_dsa_prefill_context_parallel"
        ):
            validate_test_config(invalid)

    def test_flag_parameters_are_only_ever_true(self):
        # These five render as bare flags, so `false` would state an intention the
        # command line cannot carry and the next reader would have to work out
        # whether the default it silently accepted was the reviewed one.
        config = copy.deepcopy(self.test_config)
        parameters = config["server"]["parameters"]
        for field in (
            "disable_piecewise_cuda_graph",
            "disable_shared_experts_fusion",
            "disable_custom_all_reduce",
            "enforce_disable_flashinfer_allreduce_fusion",
        ):
            parameters[field] = True
        validate_test_config(config)
        args = build_answer_server_args(config)
        self.assertEqual(
            args[args.index("--disable-piecewise-cuda-graph") : -2],
            [
                "--disable-piecewise-cuda-graph",
                "--disable-shared-experts-fusion",
                "--disable-custom-all-reduce",
                "--enforce-disable-flashinfer-allreduce-fusion",
            ],
        )

        for field in (
            "disable_piecewise_cuda_graph",
            "disable_shared_experts_fusion",
            "disable_custom_all_reduce",
            "enforce_disable_flashinfer_allreduce_fusion",
            "enable_dsa_prefill_context_parallel",
        ):
            for value in (False, None, 1, "true"):
                invalid = copy.deepcopy(self.test_config)
                invalid["server"]["parameters"][field] = value
                with self.subTest(field=field, value=value):
                    with self.assertRaisesRegex(AnswerEvalError, "only be set to true"):
                        validate_test_config(invalid)

    def test_every_reviewed_parameter_renders_in_a_fixed_order(self):
        # A config that names all of them at once, which none of the reviewed ones
        # do: what is pinned here is the spelling of each flag and the order they
        # appear in, so two runs of the same config produce the same command line
        # in the logs regardless of how the JSON happened to be keyed.
        config = copy.deepcopy(self.test_config)
        config["server"]["parameters"] = {
            "trust_remote_code": True,
            "watchdog_timeout": 600,
            "dist_timeout": 24000,
            "reasoning_parser": "glm45",
            "quantization": "w8a8_int8",
            "mem_fraction_static": 0.9,
            "num_continuous_decode_steps": 1,
            "max_running_requests": 16,
            "chunked_prefill_size": 7488,
            "cuda_graph_max_bs": 16,
            "attn_cp_size": 4,
            "dsa_prefill_cp_mode": "round-robin-split",
            "dsa_decode_backend": "flashmla_kv",
            "dsa_prefill_backend": "flashmla_sparse",
            "decode_attention_backend": "flashmla",
            "prefill_attention_backend": "fa3",
            "attention_backend": "dsa",
            "pp_size": 2,
            "tp_size": 4,
            "enable_dsa_prefill_context_parallel": True,
            "disable_piecewise_cuda_graph": True,
            "disable_shared_experts_fusion": True,
            "disable_custom_all_reduce": True,
            "enforce_disable_flashinfer_allreduce_fusion": True,
        }
        validate_test_config(config)
        self.assertEqual(
            build_answer_server_args(config),
            [
                "--trust-remote-code",
                "--tp-size",
                "4",
                "--pp-size",
                "2",
                "--attention-backend",
                "dsa",
                "--prefill-attention-backend",
                "fa3",
                "--decode-attention-backend",
                "flashmla",
                "--dsa-prefill-backend",
                "flashmla_sparse",
                "--dsa-decode-backend",
                "flashmla_kv",
                "--dsa-prefill-cp-mode",
                "round-robin-split",
                "--attn-cp-size",
                "4",
                "--cuda-graph-max-bs",
                "16",
                "--chunked-prefill-size",
                "7488",
                "--max-running-requests",
                "16",
                "--num-continuous-decode-steps",
                "1",
                "--mem-fraction-static",
                "0.9",
                "--quantization",
                "w8a8_int8",
                "--reasoning-parser",
                "glm45",
                "--dist-timeout",
                "24000",
                "--watchdog-timeout",
                "600",
                # The bare flags follow the kit's declaration order rather than
                # the config's key order, which is what makes the command line
                # reproducible: the config above keys the context-parallel flag
                # first, and it still renders last.
                "--disable-piecewise-cuda-graph",
                "--disable-shared-experts-fusion",
                "--disable-custom-all-reduce",
                "--enforce-disable-flashinfer-allreduce-fusion",
                "--enable-dsa-prefill-context-parallel",
                "--served-model-name",
                config["model"]["served_model_name"],
            ],
        )

        # The counts and the sizes are counts and sizes: a zero or a float would
        # be rejected by argparse on the machine, and a string would be accepted
        # there and then compared against an integer here.
        for field in (
            "cuda_graph_max_bs",
            "chunked_prefill_size",
            "max_running_requests",
            "num_continuous_decode_steps",
            "attn_cp_size",
        ):
            for value in (0, -1, 2.5, True, "16", None):
                invalid = copy.deepcopy(config)
                invalid["server"]["parameters"][field] = value
                with self.subTest(field=field, value=value):
                    with self.assertRaisesRegex(AnswerEvalError, field):
                        validate_test_config(invalid)

        # A timeout is a duration, so unlike the counts it may be fractional --
        # the same rule watchdog_timeout already follows.
        fractional = copy.deepcopy(config)
        fractional["server"]["parameters"]["dist_timeout"] = 0.5
        validate_test_config(fractional)
        for value in (0, -1, True, "24000", None):
            invalid = copy.deepcopy(config)
            invalid["server"]["parameters"]["dist_timeout"] = value
            with self.subTest(dist_timeout=value):
                with self.assertRaisesRegex(AnswerEvalError, "dist_timeout"):
                    validate_test_config(invalid)

    def test_server_environment_is_limited_to_names_this_tree_reads(self):
        # An entry in the whitelist is a name this tree actually reads. The
        # internal GLM cases also export SGLANG_NSA_DUAL_STREAM=0, which nothing
        # here reads -- the dual-stream threshold is a module constant, not an
        # environment lookup -- so accepting it would let a config state a setting
        # no run honours.
        config = copy.deepcopy(self.test_config)
        # The reviewed 397B config declares a warmup timeout for its slow start,
        # so the baseline environment is that one rendered name rather than
        # empty. The value is read from the config, not copied, so this stays a
        # single source of truth with the reviewed configuration.
        self.assertEqual(
            answer_server_environment(config),
            {
                "SGLANG_WARMUP_TIMEOUT": str(
                    config["server"]["env"]["SGLANG_WARMUP_TIMEOUT"]
                )
            },
        )

        config["server"]["env"] = {
            "SGLANG_WARMUP_TIMEOUT": 3600,
            "SGLANG_NSA_FLASHMLA_BACKEND_DECODE_COMPUTE_FP8": "0",
        }
        validate_test_config(config)
        # An environment is strings, and the internal cases state these as JSON
        # numbers, so the number is rendered rather than passed through.
        self.assertEqual(
            answer_server_environment(config),
            {
                "SGLANG_WARMUP_TIMEOUT": "3600",
                "SGLANG_NSA_FLASHMLA_BACKEND_DECODE_COMPUTE_FP8": "0",
            },
        )

        for environment in (
            {},
            {"SGLANG_NSA_DUAL_STREAM": 0},
            {"SGLANG_WARMUP_TIMEOUT": True},
            {"SGLANG_WARMUP_TIMEOUT": ""},
            {"SGLANG_WARMUP_TIMEOUT": [3600]},
            [],
            None,
        ):
            invalid = copy.deepcopy(self.test_config)
            invalid["server"]["env"] = environment
            with self.subTest(env=environment):
                with self.assertRaisesRegex(AnswerEvalError, "server.env"):
                    validate_test_config(invalid)

        # Admitting a third key must not turn the section into an open set.
        invalid = copy.deepcopy(self.test_config)
        invalid["server"]["log_level"] = "info"
        with self.assertRaisesRegex(AnswerEvalError, "reviewed keys"):
            validate_test_config(invalid)

    def test_chat_template_kwargs_accept_the_reviewed_spellings(self):
        # Four shapes, one per way a checkpoint's own template steers its
        # reasoning pass. A checkpoint whose template raises on
        # enable_thinking=false is steered by reasoning_effort instead, so the
        # validator has to accept either spelling while still refusing anything
        # unreviewed.
        effort = copy.deepcopy(self.test_config)
        effort["request"]["generation"]["chat_template_kwargs"] = {
            "reasoning_effort": "low"
        }
        validate_test_config(effort)

        both = copy.deepcopy(self.test_config)
        both["request"]["generation"]["chat_template_kwargs"] = {
            "enable_thinking": True,
            "reasoning_effort": "medium",
        }
        validate_test_config(both)

        # Kimi-K2.6 spells the same switch `thinking`, which is also what
        # SGLang's kimi_k2 detector declares as its reasoning_default, so that
        # name is a third accepted one.
        kimi = copy.deepcopy(self.test_config)
        kimi["request"]["generation"]["chat_template_kwargs"] = {"thinking": False}
        validate_test_config(kimi)

        # The MiniMax-M2.7 template reads no switch at all and opens <think>
        # unconditionally, so a config for it states an empty object and the
        # request carries nothing the template did not ask for. Empty rather than
        # absent: the key stays required, so the config says which of the four
        # shapes it is.
        neither = copy.deepcopy(self.test_config)
        neither["request"]["generation"]["chat_template_kwargs"] = {}
        validate_test_config(neither)

        for kwargs in (
            {"enable_thinking": "false"},
            {"thinking": "false"},
            {"thinking": 0},
            {"reasoning_effort": ""},
            {"reasoning_effort": 1},
            {"enable_thinking": False, "temperature": 0},
            None,
        ):
            invalid = copy.deepcopy(self.test_config)
            invalid["request"]["generation"]["chat_template_kwargs"] = kwargs
            with self.subTest(chat_template_kwargs=kwargs):
                with self.assertRaisesRegex(AnswerEvalError, "chat_template_kwargs"):
                    validate_test_config(invalid)

    def test_provenance_prefers_the_actual_checked_out_revision(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            model_dir = Path(temp_dir) / "Qwen3.5-397B-A17B-W8A8-INT8"
            model_dir.mkdir()
            (model_dir / "config.json").write_text(
                '{"model_type":"qwen3_5_moe"}\n', encoding="utf-8"
            )
            with mock.patch.dict(
                os.environ,
                {
                    "GITHUB_SHA": "dispatcher-sha",
                    "SGLANG_PPU_SOURCE_REVISION": "tested-sha",
                    "PPU_BASE_IMAGE_DIGEST": "sha256:image",
                },
                clear=True,
            ):
                provenance = default_provenance(
                    "Qwen3.5-397B-A17B-W8A8-INT8",
                    str(model_dir),
                    server_config=self.test_config["server"]["parameters"],
                    generation_config=self.test_config["request"]["generation"],
                    expected_hardware=answer_expected_hardware(self.test_config),
                )
        self.assertEqual(provenance["source_revision"], "tested-sha")
        self.assertEqual(provenance["base_image_digest"], "sha256:image")
        self.assertEqual(provenance["checkpoint_name"], model_dir.name)
        self.assertEqual(
            provenance["expected_hardware"],
            answer_expected_hardware(self.test_config),
        )
        self.assertEqual(len(provenance["checkpoint_config_sha256"]), 64)
        self.assertEqual(
            provenance["generation_config"],
            self.test_config["request"]["generation"],
        )

    def test_normalization_and_balanced_thinking_parser(self):
        final, findings = parse_thinking_blocks(
            "<think>这里是推理，不参与事实判分。</think> ４\r\n"
        )
        self.assertEqual(normalize_answer(final), "4")
        self.assertEqual(findings, [])

    def test_malformed_and_empty_thinking_blocks_fail(self):
        for answer in ("</think>答案", "<think>未闭合", "<think>a<think>b</think>"):
            with self.subTest(answer=answer):
                _, findings = parse_thinking_blocks(answer)
                self.assertIn(
                    "malformed_thinking_block",
                    {finding["reason_code"] for finding in findings},
                )
        _, findings = parse_thinking_blocks("<think>只有推理</think>")
        self.assertIn("empty_final_answer", {item["reason_code"] for item in findings})

    def test_unicode_hard_fail_boundaries(self):
        invalid = self.evaluate("deepseek-letter-count", "答案是4\ufffd")
        self.assertIn("invalid_unicode", self.reason_codes(invalid))

        abnormal = self.evaluate("deepseek-letter-count", "答案是4\ue000\ue001\ue002")
        self.assertIn("abnormal_codepoint_ratio", self.reason_codes(abnormal))

        control = self.evaluate("deepseek-letter-count", "答案是4\x01")
        self.assertIn("invalid_unicode", self.reason_codes(control))

        surrogate = self.evaluate("deepseek-letter-count", "答案是4\ud800")
        self.assertIn("invalid_unicode", self.reason_codes(surrogate))
        with tempfile.TemporaryDirectory() as temp_dir:
            report = build_report(
                self.dataset,
                self.profile,
                {
                    case["id"]: {
                        "content": (
                            "答案是4\ud800"
                            if case["id"] == "deepseek-letter-count"
                            else "候选回答"
                        ),
                        "finish_reason": "stop",
                        "model": "Qwen3.5-397B-A17B-W8A8-INT8",
                    }
                    for case in self.dataset["cases"]
                },
                {"served_model_name": "Qwen3.5-397B-A17B-W8A8-INT8"},
            )
            write_report_files(report, Path(temp_dir), include_raw_outputs=True)

    def test_repetition_hard_fail_boundaries(self):
        same = self.evaluate("deepseek-letter-count", "4" + "哈" * 8)
        self.assertIn("same_codepoint_run", self.reason_codes(same))

        periodic = self.evaluate("deepseek-letter-count", "4" + "甲乙丙丁戊己" * 4)
        self.assertIn("periodic_fragment_repeat", self.reason_codes(periodic))

        sentence = self.evaluate(
            "deepseek-letter-count",
            "答案确实就是四个。答案确实就是四个。答案确实就是四个。",
        )
        self.assertIn("exact_sentence_repeat", self.reason_codes(sentence))

        newline_sentence = self.evaluate(
            "deepseek-letter-count",
            "答案就是四个字母e\n答案就是四个字母e\n答案就是四个字母e",
        )
        self.assertIn("exact_sentence_repeat", self.reason_codes(newline_sentence))

    def test_ngram_coverage_thresholds(self):
        settings = self.profile["repetition"]
        repetitive = ("甲乙丙丁" * 20) + "戊己庚辛壬癸天地玄黄宇宙洪荒"
        self.assertGreaterEqual(
            _repeated_ngram_coverage(repetitive, settings),
            settings["ngram_hard_fail_coverage"],
        )
        normal = "这是一段包含不同词汇和自然语序的中文文本，用于确认正常表达不会因为少量常见字词而被判定成异常复读。"
        self.assertEqual(_repeated_ngram_coverage(normal, settings), 0.0)

    def test_repeated_mojibake_marker_is_hard_fail(self):
        result = self.evaluate("deepseek-letter-count", "答案是4，Ã¤ 与 Ã¥ 是异常片段")
        marker = next(
            finding
            for finding in result["findings"]
            if finding["reason_code"] == "mojibake_marker"
        )
        self.assertEqual(marker["action"], "hard_fail")
        self.assertEqual(result["verdict"], "failed")

        latin1 = self.evaluate("deepseek-letter-count", "答案是4，ä½ å¥½")
        self.assertIn("reversible_utf8_mojibake", self.reason_codes(latin1))
        self.assertEqual(latin1["verdict"], "failed")

    def test_finish_reason_length_fails(self):
        result = self.evaluate("deepseek-letter-count", "答案是4", "length")
        self.assertIn("finish_reason_length", self.reason_codes(result))

        for finish_reason in (None, "content_filter", "tool_calls"):
            with self.subTest(finish_reason=finish_reason):
                result = self.evaluate(
                    "deepseek-letter-count", "答案是4", finish_reason
                )
                self.assertIn("finish_reason_incomplete", self.reason_codes(result))

    def test_chat_completion_contract_and_strict_utf8(self):
        captured = {}

        def post(url, **kwargs):
            captured["url"] = url
            captured.update(kwargs)
            body = {
                "model": "Qwen3.5-397B-A17B-W8A8-INT8",
                "choices": [
                    {
                        "finish_reason": "stop",
                        "message": {
                            "content": "答案是 4",
                            "reasoning_content": None,
                        },
                    }
                ],
                "usage": {"completion_tokens": 4},
            }
            return SimpleNamespace(
                status_code=200,
                content=json.dumps(body, ensure_ascii=False).encode("utf-8"),
            )

        fake_requests = SimpleNamespace(RequestException=Exception, post=post)
        with mock.patch.dict(sys.modules, {"requests": fake_requests}):
            result = request_chat_completion(
                "http://127.0.0.1:30000/",
                "Qwen3.5-397B-A17B-W8A8-INT8",
                "deepseek这个单词中有几个e？",
                "请直接回复最终的答案。",
                timeout_seconds=10,
                generation_config=self.test_config["request"]["generation"],
            )

        self.assertEqual(captured["url"], "http://127.0.0.1:30000/v1/chat/completions")
        self.assertEqual(captured["timeout"], 10)
        self.assertEqual(captured["json"]["temperature"], 0)
        self.assertFalse(captured["json"]["chat_template_kwargs"]["enable_thinking"])
        self.assertEqual(result["content"], "答案是 4")

        def invalid_post(*_args, **_kwargs):
            return SimpleNamespace(status_code=200, content=b"\xff")

        fake_requests.post = invalid_post
        with mock.patch.dict(sys.modules, {"requests": fake_requests}):
            with self.assertRaisesRegex(CandidateRequestError, "not UTF-8") as caught:
                request_chat_completion(
                    "http://127.0.0.1:30000",
                    "model",
                    "prompt",
                    "suffix",
                    timeout_seconds=10,
                    generation_config=self.test_config["request"]["generation"],
                )
        self.assertEqual(caught.exception.reason_code, "response_decode_error")

        # A rejected request has to carry the server's own explanation: the
        # status line alone cannot tell a bad sampling parameter apart from a
        # chat template that refuses the call, and the server log records
        # nothing but the status.
        def rejecting_post(*_args, **_kwargs):
            return SimpleNamespace(
                status_code=400,
                text='{"error": "Disabling thinking is not supported."}',
                content=b"{}",
            )

        fake_requests.post = rejecting_post
        with mock.patch.dict(sys.modules, {"requests": fake_requests}):
            with self.assertRaises(CandidateRequestError) as caught:
                request_chat_completion(
                    "http://127.0.0.1:30000",
                    "model",
                    "prompt",
                    "suffix",
                    timeout_seconds=10,
                    generation_config=self.test_config["request"]["generation"],
                )
        self.assertEqual(caught.exception.reason_code, "request_error")
        self.assertIn("HTTP 400", str(caught.exception))
        self.assertIn("Disabling thinking is not supported.", str(caught.exception))

    def test_objective_reference_rules_accept_correct_variants(self):
        answers = {
            "deepseek-letter-count": "deepseek 中共有 4 个字母 e。",
            "spring-dawn-poem": "作者孟浩然：春眠不觉晓，处处闻啼鸟。夜来风雨声，花落知多少。",
            "mount-everest-height": "世界最高峰是珠穆朗玛峰，最新公认高程为 8848.86 米。",
            "henan-bordering-provinces": "河南与河北、山西、陕西、湖北、安徽、山东六省接壤。",
            "baijiaxing-first-ten": "赵、钱、孙、李、周、吴、郑、王、冯、陈。",
            "art-primary-colors": "美术颜料三原色是红色、黄色和蓝色。",
            "chinese-zodiac": "十二生肖为鼠、牛、虎、兔、龙、蛇、马、羊、猴、鸡、狗、猪。",
            "red-ball-probability": "概率为 (3/7)×(2/6)=1/7，约为14.29%。",
        }
        for case_id, answer in answers.items():
            with self.subTest(case_id=case_id):
                result = self.evaluate(case_id, answer)
                self.assertEqual(result["verdict"], "passed", result["findings"])

    def test_number_extraction_respects_chinese_context_and_compound_numerals(self):
        correct = self.evaluate("deepseek-letter-count", "答案是4个字母e。")
        correct_with_context = self.evaluate(
            "deepseek-letter-count", "deepseek共有8个字母，其中4个是字母e。"
        )
        correct_with_asserted_context = self.evaluate(
            "deepseek-letter-count",
            "deepseek 的总字符数为8，其中 e 的数量为4。",
        )
        correct_probability = self.evaluate("red-ball-probability", "答案是1/7。")
        correct_probability_derivation = self.evaluate(
            "red-ball-probability",
            "第一次抽中红球的概率为3/7，第二次为2/6，"
            "所以两次都是红球的概率为1/7，约14.29%。",
        )
        wrong = self.evaluate("deepseek-letter-count", "答案是十四个字母e。")
        conflicting = self.evaluate("deepseek-letter-count", "答案是4个，而实际是3个。")
        also_conflicting = self.evaluate(
            "deepseek-letter-count", "答案是4个，也是3个。"
        )
        self.assertEqual(correct["verdict"], "passed", correct["findings"])
        self.assertEqual(
            correct_with_context["verdict"], "passed", correct_with_context["findings"]
        )
        self.assertEqual(
            correct_probability["verdict"], "passed", correct_probability["findings"]
        )
        self.assertEqual(
            correct_with_asserted_context["verdict"],
            "passed",
            correct_with_asserted_context["findings"],
        )
        self.assertEqual(
            correct_probability_derivation["verdict"],
            "passed",
            correct_probability_derivation["findings"],
        )
        self.assertEqual(wrong["verdict"], "failed", wrong["findings"])
        self.assertEqual(conflicting["verdict"], "failed", conflicting["findings"])
        self.assertEqual(
            also_conflicting["verdict"], "failed", also_conflicting["findings"]
        )

    def test_probability_reads_bare_numbers_in_the_declared_units(self):
        # The red-ball prompt asks for a percentage, so its rule admits a bare
        # number as one. Without that declaration the rule could only ever be
        # satisfied by ``a/b``, ``X%``, or a value already in [0, 1], and a
        # correctly formed bare ``14.29`` would be discarded as if it were wrong.
        bare_percentage = self.evaluate("red-ball-probability", "答案是14.29。")
        bare_probability = self.evaluate("red-ball-probability", "答案是0.1429。")
        # Admitting the percentage reading must not widen the tolerance: a bare
        # integer is 14% here, which is outside it, and a wrong bare percentage
        # still has to fail.
        rounded_percentage = self.evaluate("red-ball-probability", "答案是14。")
        wrong_bare_percentage = self.evaluate("red-ball-probability", "答案是12.5。")
        self.assertEqual(
            bare_percentage["verdict"], "passed", bare_percentage["findings"]
        )
        self.assertEqual(
            bare_probability["verdict"], "passed", bare_probability["findings"]
        )
        for result in (rounded_percentage, wrong_bare_percentage):
            self.assertEqual(result["verdict"], "failed", result["findings"])
            self.assertIn("fact_rule_failed", self.reason_codes(result))

        # A digit inside a fraction is part of it, never a candidate of its own,
        # so the evidence a reviewer reads stays the values the answer claims.
        negated = self.evaluate("red-ball-probability", "概率不是1/7，而是1/2。")
        finding = next(
            item
            for item in negated["findings"]
            if item["reason_code"] == "fact_rule_failed"
        )
        self.assertEqual(
            [
                round(value, 4)
                for value in finding["observed"]["probability_candidates"]
            ],
            [0.1429, 0.5],
        )

        # The declaration is the only way in: the default stays the probability
        # reading alone, so no other case loosens by being silent, and a typo in
        # the unit fails the run instead of quietly restoring the strict default.
        undeclared = copy.deepcopy(self.cases["red-ball-probability"])
        del undeclared["rules"][0]["bare_number_units"]
        strict = evaluate_case(
            undeclared, "答案是14.29。", "stop", self.profile, self.dataset["revision"]
        )
        self.assertEqual(strict["verdict"], "failed", strict["findings"])
        mistyped = copy.deepcopy(self.cases["red-ball-probability"])
        mistyped["rules"][0]["bare_number_units"] = ["percentage"]
        with self.assertRaisesRegex(AnswerEvalError, "unsupported bare number units"):
            evaluate_case(
                mistyped,
                "答案是14.29。",
                "stop",
                self.profile,
                self.dataset["revision"],
            )

    def test_negated_extra_and_interleaved_facts_fail(self):
        poem = "春眠不觉晓，处处闻啼鸟。夜来风雨声，花落知多少。"
        negated_authors = [
            self.evaluate("spring-dawn-poem", f"作者不是孟浩然，而是李白：{poem}"),
            self.evaluate(
                "spring-dawn-poem", f"孟浩然不是《春晓》的作者，李白才是。{poem}"
            ),
            self.evaluate(
                "spring-dawn-poem", f"作者不是唐代诗人孟浩然，而是李白。{poem}"
            ),
        ]
        negated_number = self.evaluate(
            "deepseek-letter-count", "答案不是4个，而是3个。"
        )
        negated_probability = self.evaluate(
            "red-ball-probability", "概率不是1/7，而是1/2。"
        )
        negated_percentage = self.evaluate(
            "red-ball-probability", "答案不是14.29%，而是50%。"
        )
        negated_province = self.evaluate(
            "henan-bordering-provinces",
            "河南不与河北接壤，却与山西、陕西、湖北、安徽、山东接壤。",
        )
        extra_province = self.evaluate(
            "henan-bordering-provinces",
            "河南与河北、山西、陕西、湖北、安徽、山东和广东接壤。",
        )
        self_province = self.evaluate(
            "henan-bordering-provinces",
            "河南与河北、山西、陕西、湖北、安徽、山东以及河南自身接壤。",
        )
        interleaved_surname = self.evaluate(
            "baijiaxing-first-ten",
            "赵、钱、孙、刘、李、周、吴、郑、王、冯、陈。",
        )
        leading_surname = self.evaluate(
            "baijiaxing-first-ten",
            "刘、赵、钱、孙、李、周、吴、郑、王、冯、陈。",
        )
        trailing_surname = self.evaluate(
            "baijiaxing-first-ten",
            "赵、钱、孙、李、周、吴、郑、王、冯、陈、刘。",
        )
        for result in (
            *negated_authors,
            negated_number,
            negated_probability,
            negated_percentage,
            negated_province,
            extra_province,
            self_province,
            interleaved_surname,
            leading_surname,
            trailing_surname,
        ):
            self.assertEqual(result["verdict"], "failed", result["findings"])
            self.assertIn("fact_rule_failed", self.reason_codes(result))

    def test_exact_sequence_allows_numbering_and_item_labels(self):
        surname = self.evaluate(
            "baijiaxing-first-ten",
            "1.赵、2.钱、3.孙、4.李、5.周、6.吴、7.郑、8.王、9.冯、10.陈",
        )
        poem = self.evaluate(
            "spring-dawn-poem",
            "作者孟浩然。第一句：春眠不觉晓；第二句：处处闻啼鸟；"
            "第三句：夜来风雨声；第四句：花落知多少。",
        )
        poem_with_comma = self.evaluate(
            "spring-dawn-poem",
            "作者孟浩然，春眠不觉晓，处处闻啼鸟，夜来风雨声，花落知多少。",
        )
        self.assertEqual(surname["verdict"], "passed", surname["findings"])
        self.assertEqual(poem["verdict"], "passed", poem["findings"])
        self.assertEqual(
            poem_with_comma["verdict"], "passed", poem_with_comma["findings"]
        )

    def test_known_old_henan_golden_is_rejected(self):
        result = self.evaluate(
            "henan-bordering-provinces",
            "河南与山东、安徽、湖北、湖南、山西、陕西、河北等七省接壤。",
        )
        self.assertEqual(result["verdict"], "failed")
        finding = next(
            item
            for item in result["findings"]
            if item["reason_code"] == "fact_rule_failed"
        )
        self.assertEqual(finding["observed"]["unexpected"], ["湖南"])

    def test_wrong_objective_facts_fail(self):
        wrong_answers = {
            "deepseek-letter-count": "共有 3 个 e。",
            "spring-dawn-poem": "作者李白：春眠不觉晓，处处闻啼鸟。",
            "mount-everest-height": "最高峰是泰山，高度 1545 米。",
            "baijiaxing-first-ten": "钱、赵、孙、李、周、吴、郑、王、冯、陈。",
            "art-primary-colors": "三原色是红绿蓝。",
            "chinese-zodiac": "鼠牛虎兔龙蛇马羊猴鸡狗。",
            "red-ball-probability": "概率为 28.57%。",
        }
        for case_id, answer in wrong_answers.items():
            with self.subTest(case_id=case_id):
                result = self.evaluate(case_id, answer)
                self.assertEqual(result["verdict"], "failed")
                self.assertEqual(result["failure_class"], "candidate_failed")
                self.assertIn("fact_rule_failed", self.reason_codes(result))

    def test_open_ended_cases_are_only_hard_constraint_covered(self):
        xian = self.evaluate(
            "xian-three-day-trip",
            "西安三日游：首日逛钟楼、城墙和回民街；次日参观兵马俑与华清池；第三日游大雁塔及陕西历史博物馆，感受古都风貌。",
        )
        tengwang = self.evaluate(
            "tengwangge-reflection",
            "读王勃的《滕王阁序》，我最先感受到的是文字铺陈出的辽阔气象。山川、楼阁与宾客在整饬的骈句中次第展开，落霞与秋水的画面尤其明净。文章并不只写宴游之乐，也把个人际遇、生命短暂和进取之心交织在一起。作者虽感叹关山难越、萍水相逢，却没有停在失意中，而以老当益壮、穷且益坚表达不甘沉沦的选择。这种由盛景转入身世、再由感伤振起精神的结构，使作品既华美又有力量。今天重读它，我理解到真正动人的才华不仅是辞藻繁复，更是人在困顿中仍能确认志向。我们面对不确定的环境，也应珍惜相遇，保持清醒，在有限时间里做值得留下的事。",
        )
        self.assertEqual(xian["verdict"], "passed", xian["findings"])
        self.assertEqual(tengwang["verdict"], "passed", tengwang["findings"])
        self.assertEqual(xian["semantic_coverage"], "hard_constraints_only")
        self.assertEqual(xian["judge"]["status"], "deferred")

    def test_cross_case_duplicate_and_near_duplicate(self):
        base = "这是一段长度足够并且被错误复用于两个不同问题的统一候选回答文本。"
        left = self.evaluate("xian-three-day-trip", base + "钟楼大雁塔三日游")
        right = self.evaluate("tengwangge-reflection", base + "钟楼大雁塔三日游")
        apply_cross_case_checks([left, right], self.profile)
        self.assertIn("cross_case_duplicate", self.reason_codes(left))
        self.assertIn("cross_case_duplicate", self.reason_codes(right))

        common = (
            "甲乙丙丁戊己庚辛壬癸子丑寅卯辰巳午未申酉戌亥天地玄黄宇宙洪荒日月盈昃辰宿列张"
            "寒来暑往秋收冬藏闰余成岁律吕调阳云腾致雨露结为霜金生丽水玉出昆冈"
        )
        left = {
            "case_id": "left",
            "normalized_answer": common + "甲",
            "findings": [],
        }
        right = {
            "case_id": "right",
            "normalized_answer": common + "乙",
            "findings": [],
        }
        apply_cross_case_checks([left, right], self.profile)
        self.assertTrue(
            {"cross_case_duplicate", "cross_case_near_duplicate"}
            & self.reason_codes(left)
        )

    def test_build_report_classifies_l0_failure_and_model_mismatch(self):
        responses = {
            case["id"]: {
                "content": "4" if case["id"] == "deepseek-letter-count" else "临时答案",
                "finish_reason": "stop",
                "model": "unexpected",
            }
            for case in self.dataset["cases"]
        }
        responses["spring-dawn-poem"] = {
            "error": "Timeout",
            "reason_code": "request_error",
        }
        report = build_report(
            self.dataset,
            self.profile,
            responses,
            {"served_model_name": "Qwen3.5-397B-A17B-W8A8-INT8"},
        )
        mismatch = report["cases"][0]
        self.assertIn("unexpected_model_name", self.reason_codes(mismatch))
        self.assertEqual(report["cases"][1]["failure_class"], "server_error")
        self.assertNotIn(
            "spring-dawn-poem",
            {candidate["case_id"] for candidate in report["label_candidates"]},
        )
        public = redact_report(report)
        junit = ET.fromstring(render_junit(public))
        spring_dawn = next(
            case
            for case in junit.findall("testcase")
            if case.attrib["name"] == "spring-dawn-poem"
        )
        self.assertEqual(spring_dawn.find("failure").attrib["type"], "server_error")

    def test_label_candidates_conform_to_public_schema_surface(self):
        responses = {
            case["id"]: {
                "content": "候选回答",
                "finish_reason": "stop",
                "model": "Qwen3.5-397B-A17B-W8A8-INT8",
            }
            for case in self.dataset["cases"]
        }
        report = build_report(
            self.dataset,
            self.profile,
            responses,
            {"served_model_name": "Qwen3.5-397B-A17B-W8A8-INT8"},
        )
        schema = load_json(DATASET_DIR / "annotation_record.schema.json")
        allowed = set(schema["properties"])
        required = set(schema["required"])
        self.assertTrue(report["label_candidates"])
        for candidate in report["label_candidates"]:
            self.assertFalse(set(candidate) - allowed)
            self.assertFalse(required - set(candidate))
            self.assertRegex(candidate["sample_id"], r"^[0-9a-f]{64}$")
            self.assertRegex(candidate["answer_sha256"], r"^[0-9a-f]{64}$")
            self.assertTrue(candidate["selection_reasons"])
            self.assertEqual(candidate["case_revision"], self.dataset["revision"])
            self.assertEqual(
                candidate["quality_profile"]["sha256"],
                canonical_digest(self.profile),
            )
            self.assertIn("candidate_answer", candidate)

        pending_rule = schema["properties"]["annotation"]["allOf"][0]["then"]
        self.assertEqual(pending_rule["properties"]["reviews"]["maxItems"], 0)

        candidate = report["label_candidates"][0]
        validate_annotation_record(candidate)
        review = {
            "reviewer": "reviewer-a",
            "reviewed_at": "2026-09-01T00:00:00Z",
            "verdict": "pass",
            "scores": {
                "factual_correctness": 3,
                "relevance": 3,
                "completeness": 3,
                "coherence": 3,
                "no_repetition": 3,
            },
            "critical_errors": [],
        }
        invalid_pending = copy.deepcopy(candidate)
        invalid_pending["annotation"]["reviews"] = [review]
        with self.assertRaisesRegex(AnswerEvalError, "lifecycle state"):
            validate_annotation_record(invalid_pending)

        duplicate_reviewers = copy.deepcopy(candidate)
        duplicate_reviewers["annotation"].update(
            {
                "status": "reviewed",
                "reviews": [review, {**review, "reviewer": "Reviewer-A"}],
            }
        )
        with self.assertRaisesRegex(AnswerEvalError, "different reviewers"):
            validate_annotation_record(duplicate_reviewers)

    def test_empty_final_answer_is_graded_but_not_a_label_candidate(self):
        # A case whose whole output is a reasoning block leaves an empty final
        # answer.  That is a graded hard failure, but its candidate_answer would
        # be the empty string, which cannot satisfy the annotation invariant that
        # exactly one candidate-answer field be non-empty.  build_report must
        # still return -- the empty answer counts against the verdict and is left
        # out of the label set rather than raising during report build.
        responses = {
            case["id"]: {
                "content": (
                    "<think>只在思考块内作答，正文为空</think>"
                    if case["id"] == "deepseek-letter-count"
                    else "候选回答"
                ),
                "finish_reason": "stop",
                "model": "Qwen3.5-397B-A17B-W8A8-INT8",
            }
            for case in self.dataset["cases"]
        }
        report = build_report(
            self.dataset,
            self.profile,
            responses,
            {"served_model_name": "Qwen3.5-397B-A17B-W8A8-INT8"},
        )
        empty = next(
            case
            for case in report["cases"]
            if case["case_id"] == "deepseek-letter-count"
        )
        self.assertEqual(empty["final_answer"], "")
        self.assertEqual(empty["verdict"], "failed")
        self.assertIn("empty_final_answer", self.reason_codes(empty))
        self.assertGreaterEqual(report["summary"]["failed"], 1)
        self.assertNotIn(
            "deepseek-letter-count",
            {candidate["case_id"] for candidate in report["label_candidates"]},
        )
        for candidate in report["label_candidates"]:
            validate_annotation_record(candidate)

    def test_public_report_and_files_do_not_contain_candidate_text(self):
        responses = {
            case["id"]: {
                "content": f"PRIVATE-CANDIDATE-{case['id']}",
                "finish_reason": "stop",
                "model": "Qwen3.5-397B-A17B-W8A8-INT8",
            }
            for case in self.dataset["cases"]
        }
        report = build_report(
            self.dataset,
            self.profile,
            responses,
            {"served_model_name": "Qwen3.5-397B-A17B-W8A8-INT8"},
        )
        public = redact_report(report)
        serialized = json.dumps(public)
        self.assertNotIn("PRIVATE-CANDIDATE", serialized)
        self.assertNotIn("raw_response", serialized)
        self.assertNotIn("answer_sha256", serialized)
        self.assertNotIn("sample_id", serialized)
        self.assertNotIn("label_candidates", public)
        ET = render_junit(public)
        self.assertNotIn(b"PRIVATE-CANDIDATE", ET)

        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir)
            write_report_files(report, output_dir)
            # This fixture carries no test-config identity, so the trend
            # conversion cannot build a series key and deliberately leaves a
            # fixed, candidate-free marker instead of a trend file.  The main
            # report and gate are unaffected.
            self.assertEqual(
                {path.name for path in output_dir.iterdir()},
                {"result.json", "summary.md", "junit.xml", "trend-error.json"},
            )
            self.assertEqual(
                json.loads((output_dir / "trend-error.json").read_text()),
                {"reason_code": "trend_conversion_failed"},
            )
            for path in output_dir.iterdir():
                self.assertNotIn(b"PRIVATE-CANDIDATE", path.read_bytes())

        report["cases"][0]["returned_model"] = "/private/model/path"
        public = redact_report(report)
        serialized = json.dumps(public)
        self.assertNotIn("/private/model/path", serialized)
        self.assertEqual(
            public["cases"][0]["returned_model"],
            "<redacted-unexpected-model-name>",
        )
        self.assertNotIn("returned_model_sha256", public["cases"][0])

    def test_candidates_render_for_the_job_log(self):
        # Whoever opens a red nightly reads the log first, so the block has to
        # carry the answer text, the prompt it answered, and the observed value
        # that tripped the rule.  A candidate that UTF-8 cannot encode must come
        # out escaped rather than raise while the block is being built.
        responses = {
            case["id"]: {
                "content": (
                    "3个" if case["id"] == "deepseek-letter-count" else "候选回答\ud800"
                ),
                "finish_reason": "stop",
                "model": "Qwen3.5-397B-A17B-W8A8-INT8",
            }
            for case in self.dataset["cases"]
        }
        report = build_report(
            self.dataset,
            self.profile,
            responses,
            {"served_model_name": "Qwen3.5-397B-A17B-W8A8-INT8"},
        )
        rendered = render_candidates(report, self.dataset)
        self.assertIn("3个", rendered)
        self.assertIn("[failed] deepseek-letter-count", rendered)
        self.assertIn("observed=", rendered)
        first_prompt = self.dataset["cases"][0]["prompt"]
        self.assertIn(first_prompt, rendered)
        # The lone surrogate survives as an escape and the block stays writable
        # to a UTF-8 stream.
        self.assertNotIn("\ud800", rendered)
        self.assertIn("\\ud800", rendered)
        rendered.encode("utf-8")
        # Without the dataset the block still renders, only without prompts.
        self.assertNotIn(first_prompt, render_candidates(report))

    def test_summary_names_the_failing_cases_for_annotations(self):
        # Both Answer workflows turn every "- `" line of this document into one
        # GitHub annotation, because a step summary is not always collected --
        # the K8s runner's container hook drops it -- while an annotation is
        # shown on the run page either way. The prefix, the one-bullet-per-
        # failing-case shape, and the presence of the rule sentence are therefore
        # a contract with two bash snippets that cannot assert it themselves.
        responses = {
            case["id"]: {
                "content": (
                    "3个" if case["id"] == "deepseek-letter-count" else "候选回答"
                ),
                "finish_reason": "stop",
                "model": "Qwen3.5-397B-A17B-W8A8-INT8",
            }
            for case in self.dataset["cases"]
        }
        report = redact_report(
            build_report(
                self.dataset,
                self.profile,
                responses,
                {"served_model_name": "Qwen3.5-397B-A17B-W8A8-INT8"},
            )
        )
        rendered = render_summary(report)
        bullets = [line for line in rendered.splitlines() if line.startswith("- `")]
        failing = [case for case in report["cases"] if case["verdict"] == "failed"]
        self.assertTrue(failing, "the fixture must fail at least one case")
        self.assertEqual(
            [bullet.split("`")[1] for bullet in bullets],
            [case["case_id"] for case in failing],
            "one bullet per failing case, in report order, and no other line",
        )
        for bullet, case in zip(bullets, failing):
            for finding in case["findings"]:
                self.assertIn(finding["reason_code"], bullet)
                # The reason code names a class of failure; the rule sentence is
                # what tells a reader which answer was expected.
                if finding.get("rule_description"):
                    self.assertIn(finding["rule_description"], bullet)
        # The workflows also lift this line into a notice, so the counts reach
        # the run page whether or not anything failed.
        self.assertIn(
            f"- Cases: {report['summary']['passed']}/{report['summary']['total']} passed",
            rendered,
        )
        # A case that passes must drop out of the list while its neighbours stay,
        # or a reader would be sent to a question that answered correctly.
        report["cases"][1]["verdict"] = "passed"
        remaining = [
            line.split("`")[1]
            for line in render_summary(report).splitlines()
            if line.startswith("- `")
        ]
        self.assertNotIn(report["cases"][1]["case_id"], remaining)
        self.assertEqual(len(remaining), len(bullets) - 1)
        # A green report must not carry the section at all, or the workflows
        # would annotate a run that has nothing to report.
        green = render_summary(
            {
                "summary": {
                    "verdict": "passed",
                    "passed": 1,
                    "total": 1,
                    "suspect": 0,
                },
                "provenance": {"served_model_name": "Qwen3.8-27B"},
                "cases": [
                    {
                        "case_id": "only-case",
                        "kind": "objective",
                        "verdict": "passed",
                        "findings": [],
                    }
                ],
            }
        )
        self.assertNotIn("### Failing cases", green)
        self.assertEqual(
            [line for line in green.splitlines() if line.startswith("- `")], []
        )

    def test_raw_outputs_land_next_to_the_redacted_report(self):
        # The nightly workflow publishes these two file names as its only record
        # of what the model actually answered, so a rename has to break here
        # rather than quietly ship an artifact with no candidate text in it.
        responses = {
            case["id"]: {
                "content": "答案是3",
                "finish_reason": "stop",
                "model": "Qwen3.5-397B-A17B-W8A8-INT8",
            }
            for case in self.dataset["cases"]
        }
        report = build_report(
            self.dataset,
            self.profile,
            responses,
            {"served_model_name": "Qwen3.5-397B-A17B-W8A8-INT8"},
        )
        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir)
            write_report_files(report, output_dir, include_raw_outputs=True)
            # No test-config identity here either, so the sanitized trend marker
            # sits alongside the raw and redacted artifacts without disturbing
            # them.
            self.assertEqual(
                {path.name for path in output_dir.iterdir()},
                {
                    "result.json",
                    "summary.md",
                    "junit.xml",
                    "result.raw.json",
                    "label_candidates.jsonl",
                    "trend-error.json",
                },
            )
            self.assertEqual(
                json.loads((output_dir / "trend-error.json").read_text()),
                {"reason_code": "trend_conversion_failed"},
            )
            raw = json.loads((output_dir / "result.raw.json").read_text())
            public = json.loads((output_dir / "result.json").read_text())
            candidates = [
                json.loads(line)
                for line in (output_dir / "label_candidates.jsonl")
                .read_text()
                .splitlines()
            ]

        self.assertEqual({case["final_answer"] for case in raw["cases"]}, {"答案是3"})
        self.assertNotIn("final_answer", public["cases"][0])
        self.assertEqual({row["candidate_answer"] for row in candidates}, {"答案是3"})

    def test_reasoning_content_captured_only_for_exhausted_cases(self):
        # A length-truncated response that spent its whole budget reasoning
        # returns an empty answer, so the chain-of-thought is the only surface
        # that can tell a runaway loop from a genuinely long answer cut off at
        # the ceiling.  build_report keeps that text for the exhausted case and
        # only the digest for the healthy ones, and redact_report drops it from
        # the published report exactly like the other raw fields.
        long_reasoning = "让我想想。" * 4000
        responses = {
            case["id"]: {
                "content": "候选回答",
                "reasoning_content": "简短推理",
                "finish_reason": "stop",
                "model": "Qwen3.5-397B-A17B-W8A8-INT8",
            }
            for case in self.dataset["cases"]
        }
        responses["deepseek-letter-count"] = {
            "content": "",
            "reasoning_content": long_reasoning,
            "finish_reason": "length",
            "model": "Qwen3.5-397B-A17B-W8A8-INT8",
            "usage": {"completion_tokens": 16384},
        }
        report = build_report(
            self.dataset,
            self.profile,
            responses,
            {"served_model_name": "Qwen3.5-397B-A17B-W8A8-INT8"},
        )
        by_id = {case["case_id"]: case for case in report["cases"]}
        exhausted = by_id["deepseek-letter-count"]
        # The exhausted case keeps the full reasoning text and its digest.
        self.assertEqual(exhausted["reasoning_content"], long_reasoning)
        self.assertRegex(exhausted["reasoning_sha256"], r"^[0-9a-f]{64}$")
        # A healthy case that finished with a real answer keeps only the digest;
        # carrying every chain-of-thought would bloat the artifact for nothing.
        healthy = next(
            case for cid, case in by_id.items() if cid != "deepseek-letter-count"
        )
        self.assertNotIn("reasoning_content", healthy)
        self.assertRegex(healthy["reasoning_sha256"], r"^[0-9a-f]{64}$")
        # The published report never leaks the chain-of-thought.
        public = redact_report(report)
        self.assertTrue(
            all("reasoning_content" not in case for case in public["cases"])
        )
        # It rides the same disclosure switch as the other raw fields: present
        # in result.raw.json, absent from result.json.
        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir)
            write_report_files(report, output_dir, include_raw_outputs=True)
            raw = json.loads((output_dir / "result.raw.json").read_text())
            published = json.loads((output_dir / "result.json").read_text())
        raw_by_id = {case["case_id"]: case for case in raw["cases"]}
        self.assertEqual(
            raw_by_id["deepseek-letter-count"]["reasoning_content"], long_reasoning
        )
        self.assertTrue(
            all("reasoning_content" not in case for case in published["cases"])
        )


@unittest.skipIf(answer_suite_kit is None, "answer_suite_kit needs torch")
class TestPPUAnswerMultiNodeExchange(unittest.TestCase):
    """The four nodes of a group, as far as one host can stand in for them.

    The suite that serves a checkpoint across four boards cannot be run here, but
    the part of it that is protocol rather than inference can: which node reports
    where, how the nodes publish what they hold, and how a worker learns that rank
    0 is done.  Those are the pieces whose mistakes cost an hour of cluster time
    each, so they are checked against a real directory with a stubbed device.

    `setUpClass` is deliberately not called -- it launches a server -- and the
    class attributes it would set are assigned directly instead.
    """

    RENDEZVOUS = {"nnodes": 4, "node_rank": 0, "dist_init_addr": "10.0.0.1:29500"}

    @classmethod
    def setUpClass(cls):
        cls.four_node = load_json(CONFIG_DIR / "qwen3.8" / "2.4t-a95b-fp8-144g.json")
        cls.single_node = load_json(CONFIG_DIR / "qwen3.5" / "397b-a17b-w8a8-int8.json")

    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        # Eight devices per node, whatever this host has: the exchange is what is
        # under test, and a CPU runner reports none.
        for name, value in (
            ("device_count", lambda: 8),
            ("get_device_name", lambda index: "ZW-M890P"),
            (
                "get_device_properties",
                lambda index: SimpleNamespace(total_memory=147456 * 1024 * 1024),
            ),
        ):
            patcher = mock.patch.object(
                answer_suite_kit.torch.cuda, name, value, create=True
            )
            patcher.start()
            self.addCleanup(patcher.stop)

    def node(self, node_rank, output_dir=None, config=None):
        config = self.four_node if config is None else config
        # From what the config declares, not from which object it is: a test that
        # varies a copy of one of the two must still get the shape it copied.
        multi_node = answer_node_count(config) > 1

        class Node(answer_suite_kit.AnswerSuiteMixin, unittest.TestCase):
            def test_public_answer_suite(self):
                return answer_suite_kit.AnswerSuiteMixin.test_public_answer_suite(self)

        Node.test_config = config
        Node.request_config = config["request"]
        Node.dataset = {"cases": [{"id": f"case-{index}"} for index in range(10)]}
        Node.output_dir = Path(self.root if output_dir is None else output_dir)
        Node.distributed = (
            {**self.RENDEZVOUS, "node_rank": node_rank} if multi_node else None
        )
        Node.node_rank = node_rank if multi_node else 0
        Node.rank_dir = Node._resolve_rank_dir()
        Node.report_dir = (
            Node.output_dir
            if Node.node_rank == 0
            else Node.rank_dir / f"rank-{Node.node_rank}"
        )
        return Node

    def worker_case(self, node_rank, exit_code=None):
        node = self.node(node_rank)
        node.rank_dir.mkdir(parents=True, exist_ok=True)
        case = node("test_public_answer_suite")
        case.process = SimpleNamespace(poll=lambda: exit_code)
        return case

    def test_a_group_refuses_a_results_directory_it_cannot_share(self):
        # Every pod resolves the relative default against its own working
        # directory, so the four would never observe each other's files.
        with self.assertRaises(RuntimeError) as raised:
            self.node(0, output_dir="ppu-answer-artifacts")
        self.assertIn("every node of the group shares", str(raised.exception))
        self.assertIsNone(
            self.node(
                0, output_dir="ppu-answer-artifacts", config=self.single_node
            ).rank_dir
        )

    def test_only_rank_zero_reports_where_the_workflow_collects(self):
        self.assertEqual(self.node(0).report_dir, self.root)
        for node_rank in (1, 2, 3):
            self.assertEqual(
                self.node(node_rank).report_dir,
                self.root / "ranks" / f"rank-{node_rank}",
            )

    def test_each_node_persists_its_server_streams_beside_its_report(self):
        # A server_start crash is the run's cause yet the report carries only a
        # generic reason_code; the traceback survives only because the child's
        # streams are teed to files under the node's own report directory,
        # where the workflow already collects the results.  Each node opens its
        # own pair so a worker's bringup crash is not written over rank 0's.
        for node_rank in range(4):
            node = self.node(node_rank)
            stdout, stderr = node._open_server_logs()
            self.addCleanup(stdout.close)
            self.addCleanup(stderr.close)
            self.assertEqual(
                (Path(stdout.name), Path(stderr.name)),
                (
                    node.report_dir / "server.stdout.log",
                    node.report_dir / "server.stderr.log",
                ),
            )
            # Line-buffered and flushed per line, so a server that dies before
            # the API answers still leaves what it said on disk.
            stdout.write("bringing up\n")
            self.assertEqual(
                (node.report_dir / "server.stdout.log").read_text(encoding="utf-8"),
                "bringing up\n",
            )
        # Rank 0's pair lands where the workflow collects; a worker's under its
        # own rank directory, so the four servers do not share two names.
        self.assertTrue((self.root / "server.stdout.log").is_file())
        self.assertTrue(
            (self.root / "ranks" / "rank-3" / "server.stderr.log").is_file()
        )

    def test_each_node_states_the_group_to_its_own_server(self):
        # The config's own reviewed variables are applied first and the group's
        # description of itself second, so each node launches with server.env
        # rendered plus the three rendezvous variables. MASTER_ADDR comes from
        # the rendezvous in force rather than from the injected variable of the
        # same name, so an override that exists because the injected address does
        # not resolve is not undone here. The value is read from the config to
        # keep one source of truth with the reviewed configuration.
        warmup = str(self.four_node["server"]["env"]["SGLANG_WARMUP_TIMEOUT"])
        for node_rank in range(4):
            with self.subTest(node_rank=node_rank):
                self.assertEqual(
                    self.node(node_rank)._server_environment(),
                    {
                        "SGLANG_WARMUP_TIMEOUT": warmup,
                        "MASTER_ADDR": "10.0.0.1",
                        "NNODES": "4",
                        "RANK": str(node_rank),
                    },
                )
        # A single-node launch inherits its environment plus only the variables
        # its config names; the reviewed 397B config names a warmup timeout, so
        # that one rendered name is what it launches with.
        self.assertEqual(
            self.node(0, config=self.single_node)._server_environment(),
            {
                "SGLANG_WARMUP_TIMEOUT": str(
                    self.single_node["server"]["env"]["SGLANG_WARMUP_TIMEOUT"]
                )
            },
        )

    def test_a_config_that_names_variables_gets_them_on_a_single_node_too(self):
        # server.env is not a multi-node facility: the entries ported from the
        # internal GLM cases set it on one board.  The group's own description of
        # itself is applied last, so a config cannot displace it.
        config = copy.deepcopy(self.single_node)
        config["server"]["env"] = {"SGLANG_WARMUP_TIMEOUT": 3600}
        self.assertEqual(
            self.node(0, config=config)._server_environment(),
            {"SGLANG_WARMUP_TIMEOUT": "3600"},
        )

        config = copy.deepcopy(self.four_node)
        config["server"]["env"] = {
            "SGLANG_NSA_FLASHMLA_BACKEND_DECODE_COMPUTE_FP8": "0"
        }
        self.assertEqual(
            self.node(2, config=config)._server_environment(),
            {
                "SGLANG_NSA_FLASHMLA_BACKEND_DECODE_COMPUTE_FP8": "0",
                "MASTER_ADDR": "10.0.0.1",
                "NNODES": "4",
                "RANK": "2",
            },
        )

    def test_provenance_describes_every_node_the_verdict_was_produced_on(self):
        for node_rank in range(4):
            node = self.node(node_rank)
            node.rank_dir.mkdir(parents=True, exist_ok=True)
            node._write_node_inventory()
        accelerator = self.node(0)._accelerator()
        self.assertEqual(accelerator["visible_device_count"], 8)
        self.assertEqual(accelerator["total_device_count"], 32)
        self.assertEqual([n["node_rank"] for n in accelerator["nodes"]], [0, 1, 2, 3])
        self.assertEqual(accelerator["dist_init_addr"], "10.0.0.1:29500")
        self.assertNotIn("node_ranks_without_inventory", accelerator)
        # Staged writes leave nothing a reader could mistake for an inventory.
        self.assertEqual(
            sorted(path.name for path in (self.root / "ranks").iterdir()),
            [f"rank-{index}-devices.json" for index in range(4)],
        )

    def test_a_node_that_reported_nothing_is_named_rather_than_dropped(self):
        for node_rank in (0, 1, 3):
            node = self.node(node_rank)
            node.rank_dir.mkdir(parents=True, exist_ok=True)
            node._write_node_inventory()
        accelerator = self.node(0)._accelerator()
        self.assertEqual(accelerator["total_device_count"], 24)
        self.assertEqual(accelerator["node_ranks_without_inventory"], [2])

    def test_a_single_node_report_keeps_the_shape_already_collected(self):
        accelerator = self.node(0, config=self.single_node)._accelerator()
        self.assertEqual(set(accelerator), {"visible_device_count", "devices"})

    def test_only_rank_zero_releases_the_group(self):
        for node_rank in (1, 2, 3):
            self.node(node_rank)._release_worker_nodes()
        self.assertFalse((self.root / "ranks" / "rank0-complete").exists())
        self.node(0, config=self.single_node)._release_worker_nodes()
        self.assertFalse((self.root / "ranks").exists())
        rank_zero = self.node(0)
        # Called from the setup failure path and the teardown both, so twice.
        rank_zero._release_worker_nodes()
        rank_zero._release_worker_nodes()
        self.assertTrue((self.root / "ranks" / "rank0-complete").is_file())

    def test_a_worker_returns_once_rank_zero_releases_it(self):
        case = self.worker_case(2)
        self.node(0)._release_worker_nodes()
        case.test_public_answer_suite()

    def test_a_worker_fails_if_its_own_server_dies_first(self):
        case = self.worker_case(1, exit_code=1)
        with self.assertRaises(AssertionError) as raised:
            case.test_public_answer_suite()
        self.assertIn("lost its server with exit code 1", str(raised.exception))

    def test_a_worker_gives_up_on_a_rank_zero_that_never_finishes(self):
        case = self.worker_case(3)
        # The hold is the reviewed request budget for the whole corpus plus the
        # margin; the clock is moved rather than waited on.
        clock = iter([0, 10**9])
        with mock.patch.object(answer_suite_kit.time, "monotonic", lambda: next(clock)):
            with self.assertRaises(AssertionError) as raised:
                case.test_public_answer_suite()
        budget = (
            self.four_node["request"]["timeout_seconds"] * 10
            + answer_suite_kit.WORKER_HOLD_MARGIN_SECONDS
        )
        self.assertIn(str(budget), str(raised.exception))
        self.assertIn("never published its completion", str(raised.exception))

    def test_teardown_releases_the_workers_before_killing_the_server(self):
        # The other order kills rank 0's process, which makes the workers'
        # schedulers exit, and a healthy run then reports three failed pods.
        order = []
        rank_zero = self.node(0)
        rank_zero.process = SimpleNamespace(pid=4321)
        release = answer_suite_kit.AnswerSuiteMixin._release_worker_nodes.__func__

        def traced_release(cls):
            order.append("release")
            release(cls)

        rank_zero._release_worker_nodes = classmethod(traced_release)
        with mock.patch.object(
            answer_suite_kit, "kill_process_tree", lambda pid: order.append("kill")
        ):
            rank_zero.tearDownClass()
        self.assertEqual(order, ["release", "kill"])
        self.assertTrue((self.root / "ranks" / "rank0-complete").is_file())


if __name__ == "__main__":
    unittest.main()
