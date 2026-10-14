"""The guard for the PPU accuracy contract, without a device.

Everything here runs on a laptop: :mod:`accuracy_eval_kit` is deliberately free
of ``torch`` and of EvalScope itself, so every reviewed config can be validated,
every command line rendered and every report shape pinned before a nightly job
spends an hour warming a checkpoint to discover a typo.

Three things it protects that nothing else can:

* the EvalScope report reader.  Report v2 stores a metric's identity as an
  object and derives its display name from a property that is never written to
  the file, so a reader that guessed either would take a diagnostic metric for
  the conclusion and report a plausible wrong number.  The fixtures below are
  that file shape.
* the summary prefixes ``collect_accuracy_evidence.sh`` turns into annotations.
  A workflow cannot assert them.
* the reviewed configs, all of them, including the ones no suite runs yet.
"""

import copy
import json
import tempfile
import unittest
import xml.etree.ElementTree as ET
from pathlib import Path

from sglang.test.ci.ci_register import register_cpu_ci, register_ppu_ci
from sglang.test.kits.accuracy_eval_kit import (
    ACCURACY_REPORT_SCHEMA_VERSION,
    DATASET_CONTRACTS,
    REASON_CODES,
    REPORT_SCHEMA_VERSION,
    SUPPORTED_SERVER_ENVIRONMENT,
    AccuracyEvalError,
    accuracy_expected_hardware,
    accuracy_provenance,
    accuracy_server_environment,
    build_accuracy_server_args,
    build_evalscope_command,
    build_report,
    failed_measurement_record,
    load_json,
    locate_evalscope_report,
    measurement_record,
    read_evalscope_report,
    render_junit,
    render_summary,
    resolve_evaluation_plan,
    validate_test_config,
    write_report_files,
)

# Hardware-free, so the CPU suite owns it.  Also on the PPU per-commit chain for
# the reason the perf unit file is: this is the only guard for the evaluator
# behind the nightly-accuracy-*-ppu suites, and a PR that breaks a config or a
# summary prefix should turn the PPU gate red immediately rather than surface in
# a nightly run hours later.
register_cpu_ci(est_time=3, suite="base-a-test-cpu")
register_ppu_ci(est_time=10, suite="stage-b-test-1-gpu-ppu")

DATA_ROOT = Path(__file__).parent
CONFIG_DIR = DATA_ROOT / "configs"
GSM8K_CONFIG = CONFIG_DIR / "glm5.2" / "fp8-channelwise-144g-gsm8k.json"


def unjudged_config():
    """The reviewed GSM8K config with its baseline taken back off.

    The tests about the *absence* of a baseline used to get that absence by
    loading this config as it stands, which held only until the entry earned one
    — a routine reviewed change, made the moment it had a green full run. Six
    tests went red on the day it did. So the absence is stated here instead of
    borrowed from a file that was always going to stop supplying it.
    """

    config = load_json(GSM8K_CONFIG)
    config["evaluation"]["baseline"] = None
    config["evaluation"].pop("min_ratio", None)
    return config


def evalscope_report(
    *,
    metric_name="accuracy",
    aggregation="mean",
    primary=True,
    samples=1319,
    score=0.94,
    execution_summary=None,
    extra_metrics=(),
):
    """A report of the shape EvalScope's ``Report.model_dump`` produces.

    ``num`` is present and ``score`` is not, at the report level, because the
    first is a computed field and the second a plain property: that asymmetry is
    exactly what the reader has to survive, so the fixture reproduces it rather
    than describing it.
    """

    identity = {"name": metric_name, "aggregation": aggregation, "dimensions": {}}
    metrics = [
        {
            "identity": identity,
            "legacy_name": None,
            "num": samples,
            "score": score,
            "macro_score": score,
            "categories": [
                {
                    "name": ["default"],
                    "num": samples,
                    "score": score,
                    "macro_score": score,
                    "subsets": [
                        {
                            "name": "main",
                            "score": score,
                            "num": samples,
                            "is_aggregate": False,
                        },
                        {
                            "name": "OVERALL",
                            "score": score,
                            "num": samples,
                            "is_aggregate": True,
                        },
                    ],
                }
            ],
            "semantics": {"kind": "quality"},
        }
    ]
    metrics.extend(copy.deepcopy(metric) for metric in extra_metrics)
    return {
        "schema_version": REPORT_SCHEMA_VERSION,
        "name": "gsm8k",
        "dataset_name": "gsm8k",
        "model_name": "GLM-5.2-FP8-Channelwise",
        "metrics": metrics,
        "num": samples,
        "primary_metric_identity": identity if primary else None,
        "primary_metric_unavailable_reason": (
            None if primary else "no scored metric was produced"
        ),
        "execution_summary": execution_summary,
    }


class TestAccuracyConfigs(unittest.TestCase):
    """Every reviewed config, validated and rendered."""

    def _config_paths(self):
        paths = sorted(CONFIG_DIR.rglob("*.json"))
        self.assertTrue(paths, f"no reviewed configs found under {CONFIG_DIR}")
        return paths

    def test_every_reviewed_config_validates(self):
        for path in self._config_paths():
            with self.subTest(config=path.name):
                validate_test_config(load_json(path))

    def test_every_reviewed_config_resolves_a_plan(self):
        for path in self._config_paths():
            with self.subTest(config=path.name):
                config = load_json(path)
                plan = resolve_evaluation_plan(config)
                contract = DATASET_CONTRACTS[plan["dataset"]]
                self.assertEqual(plan["dataset_id"], contract["dataset_id"])
                self.assertEqual(plan["primary_metric"], contract["primary_metric"])
                self.assertEqual(
                    plan["expected_samples"], plan["limit"] or contract["samples"]
                )

    def test_test_ids_are_unique(self):
        seen = {}
        for path in self._config_paths():
            test_id = load_json(path)["test_id"]
            self.assertNotIn(
                test_id,
                seen,
                f"{path.name} reuses the test_id of {seen.get(test_id)}",
            )
            seen[test_id] = path.name


class TestConfigValidation(unittest.TestCase):
    def setUp(self):
        self.config = load_json(GSM8K_CONFIG)

    def _refuses(self, mutate, fragment):
        config = copy.deepcopy(self.config)
        mutate(config)
        with self.assertRaises(AccuracyEvalError) as raised:
            validate_test_config(config)
        self.assertIn(fragment, str(raised.exception))

    def test_unknown_top_level_key_is_refused(self):
        self._refuses(lambda config: config.update(workload={}), "unexpected")

    def test_tp_size_must_match_the_declared_devices(self):
        self._refuses(
            lambda config: config["server"]["parameters"].update(tp_size=4),
            "does not match",
        )

    def test_unknown_server_parameter_is_refused(self):
        self._refuses(
            lambda config: config["server"]["parameters"].update(enable_ep_moe=True),
            "which this schema does not model",
        )

    def test_a_false_bare_flag_is_refused_rather_than_ignored(self):
        self._refuses(
            lambda config: config["server"]["parameters"].update(
                disable_custom_all_reduce=False
            ),
            "bare flag",
        )

    def test_unread_server_environment_is_refused(self):
        self._refuses(
            lambda config: config["server"]["env"].update(SGLANG_USE_MODELSCOPE="true"),
            "nothing in this tree",
        )

    def test_use_modelscope_is_not_a_supported_variable(self):
        # The internal cases all set it; these pods can reach ModelScope, so
        # honouring it can only buy a hub fetch of weights already on the NAS.
        self.assertNotIn("SGLANG_USE_MODELSCOPE", SUPPORTED_SERVER_ENVIRONMENT)

    def test_unified_and_split_attention_backends_are_alternatives(self):
        self._refuses(
            lambda config: config["server"]["parameters"].update(
                prefill_attention_backend="fa3"
            ),
            "alternatives",
        )

    def test_dsa_parameters_need_the_sparse_backend(self):
        self._refuses(
            lambda config: config["server"]["parameters"].update(
                attention_backend=None
            ),
            "attention_backend 'dsa'",
        )

    def test_a_partial_speculative_configuration_is_refused(self):
        self._refuses(
            lambda config: config["server"]["parameters"].update(
                speculative_algorithm="EAGLE", disable_radix_cache=True
            ),
            "also needs",
        )

    def test_speculative_decoding_requires_the_radix_cache_off(self):
        self._refuses(
            lambda config: config["server"]["parameters"].update(
                speculative_algorithm="EAGLE",
                speculative_num_steps=2,
                speculative_eagle_topk=1,
                speculative_num_draft_tokens=3,
            ),
            "disable_radix_cache",
        )

    def test_an_unknown_dataset_is_refused(self):
        self._refuses(
            lambda config: config["evaluation"].update(dataset="mmlu"),
            "is not one of",
        )

    def test_the_primary_metric_must_be_the_datasets_own(self):
        self._refuses(
            lambda config: config["evaluation"].update(primary_metric="pass_at_k"),
            "concludes on",
        )

    def test_a_relative_dataset_dir_is_refused(self):
        self._refuses(
            lambda config: config["evaluation"].update(dataset_dir="datasets/gsm8k"),
            "absolute path",
        )

    def test_a_ratio_band_without_a_baseline_is_refused(self):
        self._refuses(
            lambda config: config["evaluation"].update(baseline=None, min_ratio=0.9),
            "judges nothing",
        )

    def test_a_baseline_outside_the_reported_scale_is_refused(self):
        self._refuses(
            lambda config: config["evaluation"].update(baseline=94.0),
            "0..1 scale",
        )

    def test_a_zero_limit_is_refused(self):
        # The internal cases spell "the whole split" as `limit: 0`; EvalScope
        # spells it as no limit at all, and a config that copied the 0 across
        # would evaluate nothing.
        self._refuses(lambda config: config["evaluation"].update(limit=0), "omit it")


class TestServerCommandLine(unittest.TestCase):
    def setUp(self):
        self.config = load_json(GSM8K_CONFIG)
        self.args = build_accuracy_server_args(self.config)

    def test_trust_remote_code_comes_first(self):
        self.assertEqual(self.args[0], "--trust-remote-code")

    def test_a_value_carrying_parameter_is_rendered_with_its_value(self):
        self.assertIn("--tp-size", self.args)
        self.assertEqual(self.args[self.args.index("--tp-size") + 1], "8")

    def test_a_null_parameter_leaves_its_flag_off(self):
        self.assertIsNone(self.config["server"]["parameters"]["quantization"])
        self.assertNotIn("--quantization", self.args)

    def test_a_bare_flag_carries_no_value(self):
        index = self.args.index("--disable-custom-all-reduce")
        following = self.args[index + 1]
        self.assertTrue(following.startswith("--"), following)

    def test_the_served_model_name_is_rendered_last(self):
        self.assertEqual(self.args[-2], "--served-model-name")
        self.assertEqual(self.args[-1], self.config["model"]["served_model_name"])

    def test_the_rendering_is_stable_across_key_order(self):
        shuffled = copy.deepcopy(self.config)
        parameters = shuffled["server"]["parameters"]
        shuffled["server"]["parameters"] = dict(reversed(list(parameters.items())))
        self.assertEqual(build_accuracy_server_args(shuffled), self.args)

    def test_the_environment_is_rendered_as_strings(self):
        environment = accuracy_server_environment(self.config)
        self.assertEqual(environment["SGLANG_WARMUP_TIMEOUT"], "3600")

    def test_expected_hardware_names_the_board(self):
        self.assertEqual(accuracy_expected_hardware(self.config), "zw-m890p-8x144g")


class TestEvalscopeCommandLine(unittest.TestCase):
    def setUp(self):
        self.config = load_json(GSM8K_CONFIG)
        self.plan = resolve_evaluation_plan(self.config)
        self.command = build_evalscope_command(
            self.config,
            self.plan,
            base_url="http://127.0.0.1:8157",
            work_dir=Path("/tmp/work"),
        )

    def _value(self, flag):
        return self.command[self.command.index(flag) + 1]

    def test_it_evaluates_a_served_model_over_http(self):
        self.assertEqual(self.command[1], "eval")
        self.assertEqual(self._value("--eval-type"), "openai_api")
        self.assertEqual(self._value("--api-url"), "http://127.0.0.1:8157/v1")
        self.assertEqual(
            self._value("--model"), self.config["model"]["served_model_name"]
        )

    def test_the_dataset_is_a_staged_directory_and_not_a_hub_id(self):
        dataset_args = json.loads(self._value("--dataset-args"))
        entry = dataset_args["gsm8k"]
        self.assertEqual(entry["dataset_id"], self.plan["dataset_dir"])
        self.assertEqual(
            entry["few_shot_num"], DATASET_CONTRACTS["gsm8k"]["default_few_shot_num"]
        )

    def test_the_generation_config_is_the_reviewed_one(self):
        self.assertEqual(
            json.loads(self._value("--generation-config")),
            self.config["evaluation"]["generation"],
        )

    def test_the_report_directory_is_named_and_not_timestamped(self):
        self.assertEqual(self._value("--work-dir"), "/tmp/work")
        self.assertIn("--no-timestamp", self.command)

    def test_no_limit_is_passed_when_the_whole_split_is_evaluated(self):
        self.assertIsNone(self.plan["limit"])
        self.assertNotIn("--limit", self.command)

    def test_a_limit_is_passed_when_the_config_states_one(self):
        config = copy.deepcopy(self.config)
        config["evaluation"]["limit"] = 20
        plan = resolve_evaluation_plan(config)
        command = build_evalscope_command(
            config, plan, base_url="http://127.0.0.1:8157", work_dir=Path("/tmp/work")
        )
        self.assertEqual(command[command.index("--limit") + 1], "20")
        self.assertEqual(plan["expected_samples"], 20)

    def test_the_rendering_is_stable_across_key_order(self):
        shuffled = copy.deepcopy(self.config)
        generation = shuffled["evaluation"]["generation"]
        shuffled["evaluation"]["generation"] = dict(reversed(list(generation.items())))
        self.assertEqual(
            build_evalscope_command(
                shuffled,
                resolve_evaluation_plan(shuffled),
                base_url="http://127.0.0.1:8157",
                work_dir=Path("/tmp/work"),
            ),
            self.command,
        )


class TestScorerResources(unittest.TestCase):
    """What a dataset's scorer needs before it scores anything.

    IFEval lost one prompt of 541 on 2026-09-10 because EvalScope fetched the
    sentence tokenizer while scoring rather than before, and charged the failure
    to that prompt's score instead of to the run. The contract names the corpus
    so the suite can resolve it first; these assert the naming, since nothing
    else on a CPU can.
    """

    def test_every_dataset_declares_what_its_scorer_needs(self):
        for dataset, contract in DATASET_CONTRACTS.items():
            with self.subTest(dataset=dataset):
                self.assertIn("nltk_resources", contract)
                for resource in contract["nltk_resources"]:
                    download_id, lookup_path = resource
                    self.assertTrue(download_id)
                    self.assertIn("/", lookup_path)

    def test_ifeval_names_the_tokenizer_that_cost_it_a_prompt(self):
        self.assertEqual(
            DATASET_CONTRACTS["ifeval"]["nltk_resources"],
            (("punkt_tab", "tokenizers/punkt_tab"),),
        )

    def test_the_datasets_that_need_nothing_say_so(self):
        self.assertEqual(DATASET_CONTRACTS["gsm8k"]["nltk_resources"], ())
        self.assertEqual(DATASET_CONTRACTS["ceval"]["nltk_resources"], ())

    def test_the_plan_carries_them_so_the_suite_need_not_know_the_dataset(self):
        for path in sorted(CONFIG_DIR.rglob("*.json")):
            config = load_json(path)
            with self.subTest(config=path.name):
                plan = resolve_evaluation_plan(config)
                self.assertEqual(
                    plan["nltk_resources"],
                    DATASET_CONTRACTS[plan["dataset"]]["nltk_resources"],
                )


class TestReportReader(unittest.TestCase):
    def _written(self, report, dataset="gsm8k", model="GLM-5.2"):
        directory = Path(tempfile.mkdtemp())
        report_path = directory / "reports" / model / f"{dataset}.json"
        report_path.parent.mkdir(parents=True)
        report_path.write_text(json.dumps(report), encoding="utf-8")
        return directory, report_path

    def test_it_finds_the_one_report_for_a_dataset(self):
        directory, report_path = self._written(evalscope_report())
        self.assertEqual(locate_evalscope_report(directory, "gsm8k"), report_path)

    def test_a_missing_report_is_absent_rather_than_an_error(self):
        directory = Path(tempfile.mkdtemp())
        self.assertIsNone(locate_evalscope_report(directory, "gsm8k"))

    def test_two_models_in_one_work_directory_are_ambiguous(self):
        directory, _ = self._written(evalscope_report(), model="first")
        second = directory / "reports" / "second" / "gsm8k.json"
        second.parent.mkdir(parents=True)
        second.write_text("{}", encoding="utf-8")
        with self.assertRaises(AccuracyEvalError):
            locate_evalscope_report(directory, "gsm8k")

    def test_the_score_comes_from_the_named_primary_metric(self):
        diagnostic = {
            "identity": {
                "name": "output_tokens",
                "aggregation": "mean",
                "dimensions": {},
            },
            "legacy_name": "output_tokens",
            "num": 1319,
            "score": 812.5,
            "macro_score": 812.5,
            "categories": [],
            "semantics": {"kind": "diagnostic"},
        }
        report = evalscope_report(extra_metrics=[diagnostic])
        # The diagnostic is first in the list, which is what a reader that took
        # list order would report.
        report["metrics"].reverse()
        _, path = self._written(report)
        reading = read_evalscope_report(path)
        self.assertEqual(reading["score"], 0.94)
        self.assertEqual(reading["metric_name"], "accuracy")
        self.assertEqual(reading["samples"], 1319)
        self.assertIn("output_tokens", reading["metrics"])

    def test_a_report_naming_no_primary_metric_has_no_score(self):
        _, path = self._written(evalscope_report(primary=False))
        reading = read_evalscope_report(path)
        self.assertIsNone(reading["score"])
        self.assertIn("no scored metric", reading["unavailable_reason"])

    def test_a_metric_without_a_legacy_name_is_named_from_its_identity(self):
        report = evalscope_report(metric_name="prompt_level_strict")
        _, path = self._written(report, dataset="ifeval")
        reading = read_evalscope_report(path)
        self.assertEqual(reading["metric_name"], "prompt_level_strict")
        self.assertEqual(reading["metric_display_name"], "prompt_level_strict:mean")

    def test_aggregate_subsets_are_left_out_of_the_breakdown(self):
        _, path = self._written(evalscope_report())
        reading = read_evalscope_report(path)
        self.assertEqual(list(reading["subsets"]), ["main"])

    def test_a_v1_report_is_refused_rather_than_misread(self):
        report = evalscope_report()
        report["schema_version"] = 1
        _, path = self._written(report)
        with self.assertRaises(AccuracyEvalError):
            read_evalscope_report(path)


class TestGrading(unittest.TestCase):
    def setUp(self):
        self.config = load_json(GSM8K_CONFIG)
        self.plan = resolve_evaluation_plan(self.config)

    def _reading(self, **overrides):
        reading = {
            "score": 0.94,
            "metric_name": "accuracy",
            "metric_display_name": "accuracy:mean",
            "samples": self.plan["expected_samples"],
            "execution": {
                "requested": self.plan["expected_samples"],
                "succeeded": self.plan["expected_samples"],
                "errored": 0,
                "incomplete": False,
            },
            "unavailable_reason": None,
            "metrics": {"accuracy:mean": 0.94},
            "subsets": {"main": 0.94},
        }
        reading.update(overrides)
        return reading

    def _plan(self, **overrides):
        config = copy.deepcopy(self.config)
        config["evaluation"].update(overrides)
        validate_test_config(config)
        return resolve_evaluation_plan(config)

    def test_without_a_baseline_a_score_is_measured_and_not_judged(self):
        plan = resolve_evaluation_plan(unjudged_config())
        record = measurement_record(plan, self._reading())
        self.assertEqual(record["status"], "measured")
        self.assertIsNone(record["ratio"])
        self.assertEqual([w["code"] for w in record["warnings"]], ["no_baseline"])

    def test_a_score_within_the_band_passes(self):
        plan = self._plan(baseline=0.95, min_ratio=0.90)
        record = measurement_record(plan, self._reading())
        self.assertEqual(record["status"], "measured")
        self.assertAlmostEqual(record["ratio"], 0.94 / 0.95)
        self.assertEqual(record["warnings"], [])

    def test_a_score_below_the_floor_fails(self):
        plan = self._plan(baseline=0.99, min_ratio=0.98)
        record = measurement_record(plan, self._reading(score=0.80))
        self.assertEqual(record["status"], "failed")
        self.assertEqual(record["reason_code"], "below_baseline")

    def test_a_score_above_the_ceiling_fails_rather_than_being_celebrated(self):
        plan = self._plan(baseline=0.4, min_ratio=0.90)
        record = measurement_record(plan, self._reading(score=0.94))
        self.assertEqual(record["status"], "failed")
        self.assertEqual(record["reason_code"], "above_baseline")

    def test_an_incomplete_run_is_not_compared_against_a_baseline(self):
        plan = self._plan(baseline=0.95, min_ratio=0.90)
        reading = self._reading(
            score=0.50,
            execution={
                "requested": 1319,
                "succeeded": 700,
                "errored": 619,
                "incomplete": True,
            },
        )
        record = measurement_record(plan, reading)
        self.assertEqual(record["reason_code"], "incomplete_samples")

    def test_a_truncated_split_is_visible_as_a_count(self):
        record = measurement_record(self.plan, self._reading(samples=700))
        self.assertEqual(record["reason_code"], "incomplete_samples")
        self.assertIn("700", record["detail"])

    def test_a_report_scoring_another_metric_is_refused(self):
        reading = self._reading(metric_name="prompt_level_strict")
        record = measurement_record(self.plan, reading)
        self.assertEqual(record["reason_code"], "primary_metric_mismatch")

    def test_a_report_without_a_score_is_named_as_such(self):
        reading = self._reading(score=None, unavailable_reason="judge unavailable")
        record = measurement_record(self.plan, reading)
        self.assertEqual(record["reason_code"], "primary_metric_missing")
        self.assertEqual(record["detail"], "judge unavailable")

    def test_a_record_states_what_it_was_asked_to_do_even_when_it_failed(self):
        record = failed_measurement_record(
            self.plan, "server_start_failed", "the checkpoint is missing"
        )
        self.assertEqual(record["dataset"], "gsm8k")
        self.assertEqual(record["expected_samples"], 1319)
        self.assertEqual(record["few_shot_num"], 4)
        self.assertIsNone(record["score"])

    def test_an_unknown_reason_code_is_refused(self):
        with self.assertRaises(AccuracyEvalError):
            failed_measurement_record(self.plan, "it_broke", "somehow")

    def test_every_reason_code_is_declared(self):
        self.assertEqual(len(set(REASON_CODES)), len(REASON_CODES))


class TestReportShape(unittest.TestCase):
    def setUp(self):
        self.config = load_json(GSM8K_CONFIG)
        self.plan = resolve_evaluation_plan(self.config)
        self.provenance = accuracy_provenance(
            self.config, accelerator={"visible_device_count": 8}
        )

    def _report(self, record, config=None):
        return build_report(
            config if config is not None else self.config,
            [record],
            provenance=self.provenance,
        )

    def _measured(self, config=None, score=0.98):
        # 0.98 sits inside this entry's reviewed band; the no-baseline tests below
        # pass an unjudged config, for which any score is recorded rather than
        # compared.
        return measurement_record(
            resolve_evaluation_plan(config if config is not None else self.config),
            {
                "score": score,
                "metric_name": "accuracy",
                "metric_display_name": "accuracy:mean",
                "samples": 1319,
                "execution": None,
                "unavailable_reason": None,
                "metrics": {"accuracy:mean": score},
                "subsets": {"main": score},
            },
        )

    def test_a_measured_report_passes(self):
        report = self._report(self._measured())
        self.assertEqual(report["schema_version"], ACCURACY_REPORT_SCHEMA_VERSION)
        self.assertEqual(report["summary"]["verdict"], "passed")
        self.assertEqual(report["summary"]["measured"], 1)

    def test_a_failed_measurement_turns_the_report_red(self):
        record = failed_measurement_record(self.plan, "evalscope_failed", "exited 1")
        report = self._report(record)
        self.assertEqual(report["summary"]["verdict"], "failed")

    def test_the_config_digest_travels_with_the_numbers(self):
        report = self._report(self._measured())
        other = copy.deepcopy(self.config)
        other["server"]["parameters"]["mem_fraction_static"] = 0.85
        self.assertNotEqual(
            report["config_digest"],
            build_report(other, [], provenance={})["config_digest"],
        )

    def test_the_provenance_records_what_was_evaluated(self):
        report = self._report(self._measured())
        self.assertEqual(report["provenance"]["evaluation"], self.config["evaluation"])
        self.assertEqual(report["provenance"]["test_config_id"], self.config["test_id"])

    def test_the_summary_prefixes_the_workflow_reads_are_stable(self):
        config = unjudged_config()
        summary = render_summary(
            self._report(self._measured(config, score=0.94), config)
        )
        self.assertIn("- MEASURED gsm8k | accuracy=0.9400", summary)
        self.assertIn("- WARN gsm8k | no_baseline", summary)
        self.assertNotIn("- FAIL", summary)

    def test_a_judged_summary_states_what_it_was_judged_against(self):
        summary = render_summary(self._report(self._measured()))
        self.assertIn("- MEASURED gsm8k | accuracy=0.9800", summary)
        self.assertIn("| baseline=0.9803 ratio=", summary)
        self.assertNotIn("- WARN gsm8k | no_baseline", summary)

    def test_a_failure_is_rendered_as_an_error_annotation(self):
        record = failed_measurement_record(
            self.plan, "report_missing", "EvalScope wrote no report"
        )
        summary = render_summary(self._report(record))
        self.assertIn("- FAIL gsm8k | report_missing |", summary)

    def test_no_other_line_of_the_summary_starts_with_a_prefix(self):
        config = unjudged_config()
        summary = render_summary(
            self._report(self._measured(config, score=0.94), config)
        )
        prefixed = [
            line
            for line in summary.splitlines()
            if line.startswith(("- MEASURED ", "- FAIL ", "- WARN "))
        ]
        self.assertEqual(len(prefixed), 2)

    def test_the_junit_file_carries_the_number(self):
        report = self._report(self._measured())
        suite = ET.fromstring(render_junit(report))
        self.assertEqual(suite.attrib["failures"], "0")
        payload = json.loads(suite.find("testcase/system-out").text)
        self.assertEqual(payload["score"], 0.98)

    def test_write_report_files_produces_the_three_artifacts(self):
        report = self._report(self._measured())
        directory = Path(tempfile.mkdtemp()) / "artifacts"
        write_report_files(report, directory)
        for name in ("result.json", "summary.md", "junit.xml"):
            self.assertTrue((directory / name).is_file(), name)
        self.assertEqual(load_json(directory / "result.json"), report)


if __name__ == "__main__":
    unittest.main()
