"""The guard for the PPU serving-performance contract, without a device.

Everything here runs on a laptop: :mod:`perf_eval_kit` is deliberately free of
``torch``, so every reviewed config can be validated, every command line
rendered and every report shape pinned before a nightly job spends an hour
warming a 2.4T checkpoint to discover a typo.

What this file protects is not a threshold -- this suite has none -- but the path
by which a number reaches a reader: the config that describes the run, the
command line it renders, the record shape the report carries, and the three
summary prefixes ``scripts/ci/ppu/collect_perf_evidence.sh`` turns into
annotations.  A regression in any of them produces a green run whose numbers
nobody can see, which is worse than a red one.
"""

import copy
import json
import os
import tempfile
import unittest
import xml.etree.ElementTree as ET
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from sglang.test.ci.ci_register import register_cpu_ci, register_ppu_ci
from sglang.test.kits.answer_eval_kit import canonical_digest
from sglang.test.kits.perf_eval_kit import (
    ALLOWED_SERVER_ENVIRONMENT,
    DECODE_METRIC_FIELDS,
    EXTERNAL_SERVER_ENVIRONMENT,
    INPUT_LENGTH_TOLERANCE,
    MEASUREMENT_REQUIRED_KEYS,
    METRIC_FIELDS,
    REASON_CODES,
    SERVER_PARAMETER_STORE_TRUE,
    TREND_POINT_SCHEMA_VERSION,
    TREND_PROVENANCE_FIELDS,
    MeasurementError,
    PerfEvalError,
    build_perf_server_args,
    build_report,
    extract_metrics,
    failed_measurement_record,
    load_json,
    measurement_record,
    observed_workload,
    perf_expected_hardware,
    perf_node_count,
    perf_provenance,
    perf_server_environment,
    render_junit,
    render_summary,
    render_trend_jsonl,
    resolve_distributed_runtime,
    resolve_measurement_plan,
    trend_points,
    validate_test_config,
    write_report_files,
)

# The on-machine driver, imported for the multi-node exchange tests at the end of
# this file.  It needs torch, which the evaluator deliberately does not, so a
# host without it skips those tests rather than failing to collect this file.
try:
    from sglang.test.kits import perf_suite_kit
except ImportError:
    perf_suite_kit = None

# Hardware-free, so the CPU suite owns it. It is also registered on the PPU
# per-commit chain for the same reason the Answer unit file is: this is the only
# guard for the evaluator behind the nightly-perf-*-ppu suites, and a PR that
# breaks a config or a summary prefix should turn the PPU gate red immediately
# rather than surface in a nightly measurement hours later.
register_cpu_ci(est_time=3, suite="base-a-test-cpu")
register_ppu_ci(est_time=10, suite="stage-b-test-1-gpu-ppu")

DATA_ROOT = Path(__file__).parent
CONFIG_DIR = DATA_ROOT / "configs"


def benchmark_result(**overrides):
    """What ``run_benchmark`` returns, as far as this suite reads it.

    Every metric key is populated, because ``extract_metrics`` treats a missing
    one as this tree's benchmark result shape having moved; a test that wants
    that failure removes a key explicitly.
    """

    raw = {
        "mean_ttft_ms": 1234.5,
        "median_ttft_ms": 1200.0,
        "std_ttft_ms": 30.25,
        "p99_ttft_ms": 1300.0,
        "mean_e2e_latency_ms": 1240.0,
        "median_e2e_latency_ms": 1205.0,
        "p90_e2e_latency_ms": 1290.0,
        "p99_e2e_latency_ms": 1310.0,
        "request_throughput": 0.81,
        "input_throughput": 3240.0,
        "output_throughput": 0.81,
        "max_output_tokens_per_s": 5.0,
        "total_throughput": 3240.81,
        "duration": 12.34,
        "completed": 10,
        "total_input_tokens": 40000,
        "total_output_tokens": 10,
        "concurrency": 1,
        "max_concurrent_requests": 1,
        # The decode-side numbers bench_serving always emits: zero for a
        # single-token case, real once a measurement decodes past the first
        # token.  extract_metrics records them only when asked (include_decode).
        "mean_tpot_ms": 40.0,
        "median_tpot_ms": 39.5,
        "std_tpot_ms": 1.2,
        "p99_tpot_ms": 41.0,
        "mean_itl_ms": 40.1,
        "median_itl_ms": 39.6,
        "std_itl_ms": 1.3,
        "p95_itl_ms": 41.2,
        "p99_itl_ms": 41.5,
        "input_lens": [4000] * 10,
        "output_lens": [1] * 10,
        "errors": [""] * 10,
    }
    raw.update(overrides)
    return raw


class TestPPUPerfEval(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # The single-node config with the plainest parameter set, used wherever a
        # test needs "a valid config" rather than a particular one.
        cls.test_config = load_json(
            CONFIG_DIR / "qwen3.5" / "397b-a17b-mxfp4-fp8-144g-prefill.json"
        )
        # The widest one: sparse attention, expert parallelism, DeepEP and six
        # bare flags, so the CLI order test exercises most of the schema.
        cls.sparse_config = load_json(
            CONFIG_DIR / "glm5.2" / "fp8-channelwise-144g-prefill-64k.json"
        )
        cls.four_node = load_json(
            CONFIG_DIR / "qwen3.8" / "2.4t-a95b-fp8-144g-prefill-4n.json"
        )

    def setUp(self):
        self.plan_entry = resolve_measurement_plan(self.test_config)[0]

    @staticmethod
    def rendezvous_for(config):
        """A stand-in for what the launcher injects, for configs that need one."""

        return resolve_distributed_runtime(
            config,
            {"NODE_RANK": "0", "MASTER_ADDR": "rank0.example", "MASTER_PORT": "29500"},
        )

    def test_every_reviewed_test_config_is_executable(self):
        # One config per nightly performance job, and only the workflow names the
        # ones this class does not load, so they are validated here instead of
        # first failing on the machine after a checkpoint has been warmed. The
        # walk is recursive because the configs are filed per model family, and
        # it also asserts that layout: a config outside a family directory would
        # be tested here but invisible to a reader of the tree.
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
                args = build_perf_server_args(
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
                    * perf_node_count(config),
                )
                # A config that stays on pure tensor parallelism must not grow an
                # explicit --pp-size 1, so the flag appears exactly when the
                # config asks for the layers to be split.
                self.assertEqual("--pp-size" in args, "pp_size" in parameters)
                # The schema only requires a non-empty string, so a value
                # argparse would reject would otherwise surface as a server that
                # never starts. server_args is imported here rather than at
                # module scope to keep the evaluator's own import path stdlib
                # only. A null asks for the flag to be left off so the
                # checkpoint's own declaration stands, and every ported config
                # is null, so the assertion is that no flag was emitted.
                from sglang.srt.server_args import QUANTIZATION_CHOICES

                quantization = parameters["quantization"]
                if quantization is None:
                    self.assertNotIn("--quantization", args)
                else:
                    self.assertIn(quantization, QUANTIZATION_CHOICES)
                # Every measurement of this suite is a prefill of a stated
                # length, and the ratio is what pins the prompts to it.
                self.assertEqual(config["workload"].get("random_range_ratio"), 1.0)
                for entry in resolve_measurement_plan(config):
                    self.assertEqual(entry["random_range_ratio"], 1.0)
                    self.assertLessEqual(entry["concurrency"], entry["num_prompts"])
        self.assertEqual(len(test_ids), len(config_paths), "test_id must be unique")

    def test_measurement_plan_folds_the_workload_defaults(self):
        # A plan entry is what the report records, so it has to carry the values
        # a measurement actually ran with rather than leave a caller to rediscover
        # a default the config left out.
        config = copy.deepcopy(self.test_config)
        for key in ("random_range_ratio", "warmup_requests", "warmup_passes", "seed"):
            config["workload"].pop(key, None)
        entry = resolve_measurement_plan(config)[0]
        self.assertEqual(entry["random_range_ratio"], 1.0)
        self.assertEqual(entry["warmup_requests"], 1)
        # Zero by default: the colocated line discards no pass, so a config that
        # states nothing runs exactly the one recorded pass.
        self.assertEqual(entry["warmup_passes"], 0)
        self.assertEqual(entry["seed"], 0)
        self.assertEqual(entry["flush_cache_timeout_seconds"], 900)
        self.assertEqual(entry["dataset"], config["workload"]["dataset"])
        measurement = config["workload"]["measurements"][0]
        for key in ("id", "input_len", "output_len", "num_prompts", "concurrency"):
            self.assertEqual(entry[key], measurement[key])
        # Where the number came from, so a report can be matched against the
        # internal case without going through the README.
        self.assertEqual(entry["source_case"], measurement["source_case"])
        self.assertEqual(entry["tc_name"], measurement["tc_name"])

    def test_expected_hardware_renders_the_declared_topology(self):
        config = {
            "hardware": {
                "generation": "zw-m890p",
                "visible_devices": [0, 1, 2, 3],
                "memory_gib_per_device": 144,
            }
        }
        self.assertEqual(perf_expected_hardware(config), "zw-m890p-4x144g")

        # A capacity JSON carries as a float must not leak a ".0" into
        # provenance, while a genuinely fractional one must survive.
        config["hardware"]["memory_gib_per_device"] = 144.0
        self.assertEqual(perf_expected_hardware(config), "zw-m890p-4x144g")
        config["hardware"]["memory_gib_per_device"] = 97.5
        self.assertEqual(perf_expected_hardware(config), "zw-m890p-4x97.5g")

        # A multi-node contract has to state both dimensions: eight devices on
        # each of four nodes is not the same machine as thirty-two on one, and a
        # throughput number attributed alike to both is unusable.
        config["hardware"]["memory_gib_per_device"] = 144
        config["hardware"]["visible_devices"] = list(range(8))
        config["hardware"]["nnodes"] = 4
        self.assertEqual(perf_expected_hardware(config), "zw-m890p-4nx8x144g")
        config["hardware"]["nnodes"] = 1
        self.assertEqual(perf_expected_hardware(config), "zw-m890p-8x144g")

    def test_config_rejects_the_shapes_that_would_cost_a_board(self):
        rejected = (
            (
                lambda c: c.__setitem__("schema_version", "ppu-perf-test-config/v2"),
                None,
            ),
            (lambda c: c.__setitem__("test_id", ""), None),
            (lambda c: c.__setitem__("thresholds", {}), "an unreviewed section"),
            (lambda c: c["hardware"].__setitem__("platform", "GPU"), None),
            (lambda c: c["hardware"].__setitem__("visible_devices", []), None),
            (
                lambda c: c["hardware"].__setitem__("visible_devices", [0, 0, 1, 2]),
                None,
            ),
            (lambda c: c["hardware"].__setitem__("nnodes", 0), None),
            (lambda c: c["model"].__setitem__("path", "relative/checkpoint"), None),
            (
                lambda c: c["model"].__setitem__("checkpoint_name", "Something-Else"),
                None,
            ),
            (lambda c: c["model"].__setitem__("accepted_model_types", []), None),
            (lambda c: c["server"].__setitem__("startup_timeout_seconds", 0), None),
            (lambda c: c["server"]["parameters"].__setitem__("tp_size", 3), None),
            (
                lambda c: c["server"]["parameters"].__setitem__(
                    "mem_fraction_static", 1.5
                ),
                None,
            ),
            (
                lambda c: c["server"]["parameters"].__setitem__("watchdog_timeout", 0),
                None,
            ),
            (
                lambda c: c["server"]["parameters"].__setitem__(
                    "unreviewed_flag", True
                ),
                "a parameter no reader has reviewed",
            ),
            (lambda c: c["workload"].__setitem__("dataset", "sharegpt"), None),
            (lambda c: c["workload"].__setitem__("measurements", []), None),
            (
                lambda c: c["workload"].__setitem__("random_range_ratio", 0.0),
                "a ratio that measures a distribution",
            ),
            (
                lambda c: c["workload"].__setitem__("warmup_passes", -1),
                "a negative count of passes to discard",
            ),
            (
                lambda c: c["workload"]["measurements"][0].__setitem__("input_len", 0),
                None,
            ),
            (
                lambda c: c["workload"]["measurements"][0].__setitem__(
                    "concurrency", 11
                ),
                "a parallelism the run never reaches",
            ),
            (
                lambda c: c["workload"]["measurements"][0].pop("tc_name"),
                "a measurement that cannot be traced back",
            ),
            (
                lambda c: c["workload"]["measurements"][1].__setitem__(
                    "id", c["workload"]["measurements"][0]["id"]
                ),
                "two measurements that would overwrite each other",
            ),
        )
        for index, (mutate, reason) in enumerate(rejected):
            config = copy.deepcopy(self.test_config)
            mutate(config)
            with self.subTest(case=reason or index):
                with self.assertRaises(PerfEvalError):
                    validate_test_config(config)

    def test_measurement_states_exactly_the_reviewed_keys(self):
        # Exactly, not at least: a key nobody reads would describe a setting the
        # measurement does not honour, and the reader of a report has no way to
        # tell the difference.
        self.assertEqual(
            MEASUREMENT_REQUIRED_KEYS,
            {
                "id",
                "input_len",
                "output_len",
                "num_prompts",
                "concurrency",
                "source_case",
                "tc_name",
            },
        )
        config = copy.deepcopy(self.test_config)
        config["workload"]["measurements"][0]["request_rate"] = 4
        with self.assertRaises(PerfEvalError):
            validate_test_config(config)

    def test_flag_parameters_are_only_ever_true(self):
        # Each renders as a bare flag, so `false` would state an intention the
        # command line cannot carry and the next reader would have to work out
        # whether the default it silently accepted was the reviewed one.
        for name in SERVER_PARAMETER_STORE_TRUE:
            for value in (False, "true", 1):
                config = copy.deepcopy(self.test_config)
                config["server"]["parameters"][name] = value
                with self.subTest(parameter=name, value=value):
                    with self.assertRaises(PerfEvalError):
                        validate_test_config(config)

    def test_attention_backends_may_be_named_per_phase(self):
        # The GLM 4k cases name a prefill and a decode kernel apart and leave the
        # unified backend null, which SGLang treats as the override of the
        # unified choice rather than as an addition to it.
        config = copy.deepcopy(self.test_config)
        parameters = config["server"]["parameters"]
        parameters["attention_backend"] = None
        parameters["prefill_attention_backend"] = "fa3"
        parameters["decode_attention_backend"] = "flashmla"
        validate_test_config(config)
        args = build_perf_server_args(config)
        self.assertNotIn("--attention-backend", args)
        self.assertEqual(args[args.index("--prefill-attention-backend") + 1], "fa3")
        self.assertEqual(args[args.index("--decode-attention-backend") + 1], "flashmla")

        # Null with neither phase named describes no kernel at all.
        parameters.pop("prefill_attention_backend")
        parameters.pop("decode_attention_backend")
        with self.assertRaises(PerfEvalError):
            validate_test_config(config)

    def test_sparse_and_deepep_parameters_need_the_backends_that_read_them(self):
        validate_test_config(self.sparse_config)
        for mutate in (
            # dsa_* only applies under the sparse backend, so a config that
            # renamed the backend would be describing kernels no run consults.
            lambda c: c["server"]["parameters"].__setitem__("attention_backend", "fa3"),
            # The cp mode describes how prefill context parallelism splits a
            # sequence, which needs the feature enabled.
            lambda c: c["server"]["parameters"].pop(
                "enable_dsa_prefill_context_parallel"
            ),
            # deepep_mode is consulted only on the DeepEP dispatch path.
            lambda c: c["server"]["parameters"].__setitem__("moe_a2a_backend", "none"),
        ):
            config = copy.deepcopy(self.sparse_config)
            mutate(config)
            with self.assertRaises(PerfEvalError):
                validate_test_config(config)

        # "nsa" is accepted because SGLang still accepts it, so a config copied
        # verbatim from an internal case is not rejected for its spelling.
        config = copy.deepcopy(self.sparse_config)
        config["server"]["parameters"]["attention_backend"] = "nsa"
        validate_test_config(config)

    def test_every_reviewed_parameter_renders_in_a_fixed_order(self):
        # The contract under test is the flag spelling and the argument order,
        # which is what makes two runs of one config comparable in the logs and,
        # for this suite, what makes their numbers comparable at all. The values
        # belong to the reviewed config, so they are read from it rather than
        # copied here, where a copy would become a second source of truth.
        parameters = self.sparse_config["server"]["parameters"]
        self.assertEqual(
            build_perf_server_args(self.sparse_config),
            [
                "--trust-remote-code",
                "--tp-size",
                str(parameters["tp_size"]),
                "--ep-size",
                str(parameters["ep_size"]),
                "--moe-a2a-backend",
                parameters["moe_a2a_backend"],
                "--deepep-mode",
                parameters["deepep_mode"],
                "--attention-backend",
                parameters["attention_backend"],
                "--dsa-prefill-backend",
                parameters["dsa_prefill_backend"],
                "--dsa-decode-backend",
                parameters["dsa_decode_backend"],
                "--dsa-prefill-cp-mode",
                parameters["dsa_prefill_cp_mode"],
                "--attn-cp-size",
                str(parameters["attn_cp_size"]),
                "--cuda-graph-max-bs",
                str(parameters["cuda_graph_max_bs"]),
                "--chunked-prefill-size",
                str(parameters["chunked_prefill_size"]),
                "--max-running-requests",
                str(parameters["max_running_requests"]),
                "--num-continuous-decode-steps",
                str(parameters["num_continuous_decode_steps"]),
                "--mem-fraction-static",
                str(parameters["mem_fraction_static"]),
                "--reasoning-parser",
                parameters["reasoning_parser"],
                "--tool-call-parser",
                parameters["tool_call_parser"],
                "--dist-timeout",
                str(parameters["dist_timeout"]),
                "--watchdog-timeout",
                str(parameters["watchdog_timeout"]),
                "--disable-shared-experts-fusion",
                "--disable-custom-all-reduce",
                "--enforce-disable-flashinfer-allreduce-fusion",
                "--enable-dsa-prefill-context-parallel",
                "--disable-radix-cache",
                "--disable-overlap-schedule",
                "--enable-metrics",
                "--served-model-name",
                self.sparse_config["model"]["served_model_name"],
            ],
        )

    def test_server_environment_is_limited_to_the_reviewed_names(self):
        # Numbers are rendered because an environment is strings and the internal
        # cases state several of these as JSON numbers.
        self.assertEqual(
            perf_server_environment(self.four_node),
            {
                "DG_USE_MOE_DYNAMIC_TILE": "1",
                "HGGC_EMBEDDED_COPY_THRESHOLD": "0x10000",
                "SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK": "256",
                "SGLANG_SAIL_DEEPEP_RECV_HOOK": "0",
                "SGLANG_WARMUP_TIMEOUT": "3600",
            },
        )
        # Three of the allowed names have no read point in this tree and are
        # accepted anyway, because the internal cases export them around the
        # server and a number measured without them was measured on a different
        # machine. The set is pinned here so that admitting a fourth is a
        # reviewed act rather than a diff nobody notices.
        self.assertEqual(
            EXTERNAL_SERVER_ENVIRONMENT,
            {
                "DG_USE_MOE_DYNAMIC_TILE",
                "HGGC_EMBEDDED_COPY_THRESHOLD",
                "SGLANG_NSA_DUAL_STREAM",
            },
        )
        self.assertLess(EXTERNAL_SERVER_ENVIRONMENT, ALLOWED_SERVER_ENVIRONMENT)

        for value in ({"PATH": "/usr/bin"}, {}, {"SGLANG_WARMUP_TIMEOUT": ""}):
            config = copy.deepcopy(self.test_config)
            config["server"]["env"] = value
            with self.subTest(env=value):
                with self.assertRaises(PerfEvalError):
                    validate_test_config(config)

        # Omitted rather than empty is how a config states that it needs none.
        config = copy.deepcopy(self.test_config)
        config["server"].pop("env")
        validate_test_config(config)
        self.assertEqual(perf_server_environment(config), {})

    def test_multi_node_config_binds_the_rendezvous_it_is_handed(self):
        self.assertIsNone(resolve_distributed_runtime(self.test_config, {}))
        self.assertEqual(perf_node_count(self.test_config), 1)
        self.assertEqual(perf_node_count(self.four_node), 4)

        # The rank and the address are runtime facts, so they come from the
        # launcher rather than from the reviewed config.
        runtime = resolve_distributed_runtime(
            self.four_node,
            {
                "NNODES": "4",
                "NODE_RANK": "2",
                "MASTER_ADDR": "rank0.example",
                "MASTER_PORT": "29500",
            },
        )
        self.assertEqual(
            runtime,
            {
                "nnodes": 4,
                "node_rank": 2,
                "dist_init_addr": "rank0.example:29500",
            },
        )
        args = build_perf_server_args(self.four_node, distributed=runtime)
        self.assertEqual(
            args[-6:],
            [
                "--nnodes",
                "4",
                "--node-rank",
                "2",
                "--dist-init-addr",
                "rank0.example:29500",
            ],
        )

        # The suite's own variables take precedence over the injected ones, which
        # is what lets the workflow hand it an address it discovered itself when
        # the injected one does not resolve.
        self.assertEqual(
            resolve_distributed_runtime(
                self.four_node,
                {
                    "NODE_RANK": "0",
                    "SGLANG_PPU_PERF_NODE_RANK": "3",
                    "MASTER_ADDR": "unresolvable.cluster.local",
                    "MASTER_PORT": "29500",
                    "SGLANG_PPU_PERF_DIST_INIT_ADDR": "10.0.0.1:29600",
                },
            ),
            {"nnodes": 4, "node_rank": 3, "dist_init_addr": "10.0.0.1:29600"},
        )

    def test_multi_node_launch_refuses_an_unusable_rendezvous(self):
        for environ in (
            # No rank at all: every pod would launch as an independent rank 0.
            {"MASTER_ADDR": "rank0.example", "MASTER_PORT": "29500"},
            # A rank outside the group this config declares.
            {
                "NODE_RANK": "4",
                "MASTER_ADDR": "rank0.example",
                "MASTER_PORT": "29500",
            },
            {"NODE_RANK": "one", "MASTER_ADDR": "a", "MASTER_PORT": "29500"},
            # No address: the group would never rendezvous, after holding four
            # boards for the whole startup budget.
            {"NODE_RANK": "0"},
            {"NODE_RANK": "0", "MASTER_ADDR": "rank0.example"},
            {
                "NODE_RANK": "0",
                "SGLANG_PPU_PERF_DIST_INIT_ADDR": "rank0.example",
            },
            {
                "NODE_RANK": "0",
                "SGLANG_PPU_PERF_DIST_INIT_ADDR": "rank0.example:99999",
            },
            # A launcher that started a different number of nodes than reviewed.
            {
                "NNODES": "2",
                "NODE_RANK": "0",
                "MASTER_ADDR": "rank0.example",
                "MASTER_PORT": "29500",
            },
        ):
            with self.subTest(environ=environ):
                with self.assertRaises(PerfEvalError):
                    resolve_distributed_runtime(self.four_node, environ)

        # A launch that forgot the rendezvous entirely, and one that handed a
        # rendezvous to a config that serves on one node.
        with self.assertRaises(PerfEvalError):
            build_perf_server_args(self.four_node)
        with self.assertRaises(PerfEvalError):
            build_perf_server_args(
                self.test_config,
                distributed={
                    "nnodes": 4,
                    "node_rank": 0,
                    "dist_init_addr": "10.0.0.1:29500",
                },
            )
        with self.assertRaises(PerfEvalError):
            build_perf_server_args(
                self.four_node,
                distributed={
                    "nnodes": 2,
                    "node_rank": 0,
                    "dist_init_addr": "10.0.0.1:29500",
                },
            )

    def test_a_missing_metric_is_an_inability_to_measure(self):
        raw = benchmark_result()
        metrics = extract_metrics(raw)
        self.assertEqual(len(metrics), len(METRIC_FIELDS))
        # The reported names are spelled out because bench_serving's own keys are
        # ambiguous once they leave that module.
        self.assertEqual(metrics["ttft_mean_ms"], raw["mean_ttft_ms"])
        self.assertEqual(
            metrics["output_token_throughput_tok_s"], raw["output_throughput"]
        )
        self.assertEqual(
            metrics["total_token_throughput_tok_s"], raw["total_throughput"]
        )
        # End-to-end p90 and the output-throughput peak are defined from the
        # first token, so they are recorded for every measurement, prefill or
        # not.
        self.assertEqual(metrics["e2e_latency_p90_ms"], raw["p90_e2e_latency_ms"])
        self.assertEqual(
            metrics["output_token_throughput_peak_tok_s"],
            raw["max_output_tokens_per_s"],
        )
        self.assertEqual(metrics["duration_s"], raw["duration"])
        # Time per output token and inter-token latency are undefined for a
        # single output token, so recording the zero bench_serving computes over
        # an empty list would read as a measurement.  Without include_decode they
        # are left out, and TTFT p90 is absent for the separate reason that
        # bench_serving does not emit it.
        self.assertNotIn("tpot_mean_ms", metrics)
        self.assertNotIn("itl_mean_ms", metrics)
        self.assertNotIn("ttft_p90_ms", metrics)

        # A decoding measurement records them, keyed under the unambiguous names.
        decode = extract_metrics(raw, include_decode=True)
        self.assertEqual(len(decode), len(METRIC_FIELDS) + len(DECODE_METRIC_FIELDS))
        self.assertEqual(decode["tpot_mean_ms"], raw["mean_tpot_ms"])
        self.assertEqual(decode["itl_p99_ms"], raw["p99_itl_ms"])
        # A decode source key that went missing is the same inability to measure
        # as any other, but only when the decode set was asked for.
        without_tpot = benchmark_result()
        without_tpot.pop("mean_tpot_ms")
        self.assertEqual(len(extract_metrics(without_tpot)), len(METRIC_FIELDS))
        with self.assertRaises(MeasurementError) as raised_decode:
            extract_metrics(without_tpot, include_decode=True)
        self.assertEqual(raised_decode.exception.reason_code, "metrics_missing")

        raw.pop("median_ttft_ms")
        with self.assertRaises(MeasurementError) as raised:
            extract_metrics(raw)
        self.assertEqual(raised.exception.reason_code, "metrics_missing")

    def test_a_measurement_is_graded_only_on_whether_numbers_exist(self):
        # Measured, whatever the numbers say: no threshold is enforced anywhere.
        record = measurement_record(self.plan_entry, benchmark_result())
        self.assertEqual(record["status"], "measured")
        self.assertEqual(record["warnings"], [])
        self.assertIsNone(record["reason_code"])
        self.assertEqual(record["source_case"], self.plan_entry["source_case"])
        self.assertEqual(record["observed"]["input_lengths"]["mean"], 4000)
        self.assertEqual(record["observed"]["request_errors"], [])
        slow = measurement_record(self.plan_entry, benchmark_result(mean_ttft_ms=10**7))
        self.assertEqual(slow["status"], "measured")

        # A drift the tokenizer decided is worth stating and not worth a red run:
        # a measurement of 3,900 tokens is not a measurement of 4,000.
        drifted = 4000 * (1 - 2 * INPUT_LENGTH_TOLERANCE)
        record = measurement_record(
            self.plan_entry, benchmark_result(input_lens=[int(drifted)] * 10)
        )
        self.assertEqual(record["status"], "measured")
        self.assertEqual(
            [w["code"] for w in record["warnings"]], ["input_length_drift"]
        )

        # Requests that never completed, and requests that failed. Both are an
        # inability to measure rather than a slow result.
        record = measurement_record(self.plan_entry, benchmark_result(completed=9))
        self.assertEqual(record["status"], "failed")
        self.assertEqual(record["reason_code"], "incomplete_requests")
        self.assertIn("9 of 10", record["detail"])
        # The numbers are kept even so: a reader of the report should see what
        # the partial run produced rather than a blank record.
        self.assertIsNotNone(record["metrics"])

        record = measurement_record(
            self.plan_entry,
            benchmark_result(errors=["", "connection reset", ""], completed=9),
        )
        self.assertEqual(record["reason_code"], "request_errors")
        self.assertIn("connection reset", record["detail"])

    def test_only_a_reviewed_reason_can_fail_a_measurement(self):
        self.assertEqual(
            set(REASON_CODES),
            {
                "server_start_failed",
                "benchmark_crashed",
                "metrics_missing",
                "incomplete_requests",
                "request_errors",
                "cache_flush_failed",
            },
        )
        record = failed_measurement_record(
            self.plan_entry, "server_start_failed", "RuntimeError: no config.json"
        )
        self.assertEqual(record["status"], "failed")
        self.assertIsNone(record["metrics"])
        self.assertEqual(record["id"], self.plan_entry["id"])
        with self.assertRaises(PerfEvalError):
            failed_measurement_record(self.plan_entry, "too_slow", "1200ms")

    def test_observed_workload_reports_what_the_requests_actually_were(self):
        observed = observed_workload(
            benchmark_result(input_lens=[3999, 4000, 4001], errors=["", None, "boom"])
        )
        self.assertEqual(
            observed["input_lengths"],
            {"min": 3999, "max": 4001, "mean": 4000, "count": 3},
        )
        self.assertEqual(observed["request_errors"], ["boom"])
        # A benchmark that reported no lengths at all states that, rather than an
        # empty summary a reader would take for zero-length prompts.
        self.assertIsNone(observed_workload({})["input_lengths"])

    def test_report_counts_the_measurements_and_names_the_settings(self):
        plan = resolve_measurement_plan(self.test_config)
        measurements = [
            measurement_record(plan[0], benchmark_result()),
            failed_measurement_record(plan[1], "benchmark_crashed", "SystemExit: 1"),
        ]
        report = build_report(self.test_config, measurements)
        self.assertEqual(
            report["summary"],
            {
                "verdict": "failed",
                "total": 2,
                "measured": 1,
                "failed": 1,
                "warnings": 0,
            },
        )
        self.assertEqual(report["test_id"], self.test_config["test_id"])
        # Two reports whose digests differ were measured on different settings,
        # whatever their file names say, and that is the first thing a comparison
        # has to check.
        self.assertEqual(report["config_digest"], canonical_digest(self.test_config))
        self.assertEqual(
            build_report(self.test_config, measurements[:1])["summary"]["verdict"],
            "passed",
        )

    def test_summary_carries_the_prefixes_the_workflow_reads(self):
        # collect_perf_evidence.sh has no JSON parser, so it greps these three
        # prefixes out of summary.md and turns them into annotations. That is the
        # whole delivery mechanism of a suite that judges nothing: a green run
        # whose numbers nobody can see has measured nothing usable.
        plan = resolve_measurement_plan(self.test_config)
        measurements = [
            measurement_record(plan[0], benchmark_result(input_lens=[3000] * 10)),
            failed_measurement_record(plan[1], "cache_flush_failed", "HTTP 503"),
        ]
        report = build_report(
            self.test_config,
            measurements,
            provenance=perf_provenance(self.test_config),
        )
        summary = render_summary(report)
        lines = summary.splitlines()
        self.assertEqual(
            [line for line in lines if line.startswith("- Measurements: ")],
            ["- Measurements: 1/2 measured"],
        )
        measured = [line for line in lines if line.startswith("- MEASURED ")]
        self.assertEqual(len(measured), 1)
        self.assertIn(plan[0]["id"], measured[0])
        self.assertIn("ttft_mean=1234.50ms", measured[0])
        self.assertIn("total=3240.81tok/s", measured[0])
        failed = [line for line in lines if line.startswith("- FAIL ")]
        self.assertEqual(len(failed), 1)
        self.assertIn("cache_flush_failed", failed[0])
        warned = [line for line in lines if line.startswith("- WARN ")]
        self.assertEqual(len(warned), 1)
        self.assertIn("input_length_drift", warned[0])
        # No threshold is stated anywhere, and the summary says so rather than
        # leaving a reader to infer it from a report with no red lines.
        self.assertIn("- Thresholds: none.", summary)
        self.assertIn(
            "time per output token and inter-token latency are undefined", summary
        )
        self.assertIn(self.test_config["model"]["served_model_name"], lines[0])

    def test_a_decoding_measurement_records_and_shows_the_decode_timings(self):
        # A measurement that decodes past the first token has time per output
        # token and inter-token latency defined, so measurement_record keeps them
        # -- keyed off the measurement's own output_len -- and the summary shows
        # them rather than the note the single-token line carries.
        config = copy.deepcopy(self.test_config)
        config["workload"]["measurements"][0]["output_len"] = 1500
        plan = resolve_measurement_plan(config)
        record = measurement_record(plan[0], benchmark_result())
        self.assertEqual(record["status"], "measured")
        self.assertEqual(record["metrics"]["tpot_mean_ms"], 40.0)
        self.assertEqual(record["metrics"]["itl_p99_ms"], 41.5)
        report = build_report(config, [record], provenance=perf_provenance(config))
        summary = render_summary(report)
        measured = [
            line for line in summary.splitlines() if line.startswith("- MEASURED ")
        ]
        self.assertEqual(len(measured), 1)
        self.assertIn("tpot_mean=40.00ms", measured[0])
        self.assertIn("itl_p99=41.50ms", measured[0])
        self.assertNotIn(
            "time per output token and inter-token latency are undefined", summary
        )

    def test_report_files_carry_the_numbers_and_need_no_redaction(self):
        plan = resolve_measurement_plan(self.test_config)
        report = build_report(
            self.test_config,
            [
                measurement_record(plan[0], benchmark_result()),
                failed_measurement_record(plan[1], "metrics_missing", "no duration"),
            ],
            provenance=perf_provenance(self.test_config),
        )
        with tempfile.TemporaryDirectory() as directory:
            output_dir = Path(directory) / "nested"
            write_report_files(report, output_dir)
            self.assertEqual(
                sorted(path.name for path in output_dir.iterdir()),
                ["junit.xml", "result.json", "summary.md", "trend.jsonl"],
            )
            written = load_json(output_dir / "result.json")
            # Unlike the Answer report there is no redaction pass, because a
            # measurement records lengths and timings and never a generated
            # token; the numbers being present is the point of the artifact.
            self.assertEqual(
                written["measurements"][0]["metrics"]["ttft_mean_ms"], 1234.5
            )
            self.assertEqual(
                written["provenance"]["workload"], self.test_config["workload"]
            )
            # The trend rows are written next to the report rather than derived
            # from it later, so a measurement taken before anything publishes
            # them is still on the data branch once something does.
            self.assertEqual(
                [
                    json.loads(line)
                    for line in (output_dir / "trend.jsonl")
                    .read_text(encoding="utf-8")
                    .splitlines()
                ],
                trend_points(report),
            )

        suite = ET.fromstring(render_junit(report))
        self.assertEqual(suite.attrib["tests"], "2")
        self.assertEqual(suite.attrib["failures"], "1")
        cases = suite.findall("testcase")
        self.assertEqual(
            [case.attrib["name"] for case in cases], [e["id"] for e in plan]
        )
        self.assertEqual(cases[1].find("failure").attrib["type"], "metrics_missing")
        # A reader who has only the test report still has the numbers.
        self.assertEqual(
            json.loads(cases[0].find("system-out").text)["metrics"]["duration_s"], 12.34
        )

    def test_trend_rows_key_a_series_and_carry_what_moves_it(self):
        # One row per measurement, for the branch the nightly numbers accumulate
        # on. What a row owes a reader months later is fixed by what that reader
        # cannot otherwise recover: which series the row belongs to, the numbers,
        # and the inputs that legitimately move them.
        plan = resolve_measurement_plan(self.test_config)
        report = build_report(
            self.test_config,
            [
                measurement_record(plan[0], benchmark_result()),
                failed_measurement_record(plan[1], "server_start_failed", "no port"),
            ],
            provenance=perf_provenance(self.test_config),
        )
        rows = trend_points(report)
        self.assertEqual(len(rows), 2)
        measured, unmeasured = rows
        for row in rows:
            self.assertEqual(row["schema_version"], TREND_POINT_SCHEMA_VERSION)
            self.assertEqual(row["test_id"], report["test_id"])
            self.assertEqual(row["generated_at"], report["generated_at"])
            self.assertEqual(row["config_digest"], canonical_digest(self.test_config))
        self.assertEqual(
            [row["measurement_id"] for row in rows], [entry["id"] for entry in plan]
        )
        self.assertEqual(measured["status"], "measured")
        self.assertEqual(measured["metrics"]["ttft_mean_ms"], 1234.5)
        self.assertIsNone(measured["reason_code"])
        # A night that could not measure stays in the series with no numbers, so
        # a reader can tell it from a night on which nothing ran at all. Dropping
        # it would leave a gap that reads as a scheduling hole.
        self.assertEqual(unmeasured["status"], "failed")
        self.assertEqual(unmeasured["reason_code"], "server_start_failed")
        self.assertIsNone(unmeasured["metrics"])
        # The workload shape is repeated into every row on purpose: a row has to
        # be readable without resolving its digest back to a config file that may
        # since have been edited or deleted.
        for key in ("input_len", "output_len", "num_prompts", "concurrency"):
            self.assertEqual(measured[key], plan[0][key])
        self.assertEqual(measured["tc_name"], plan[0]["tc_name"])
        self.assertEqual(measured["source_case"], plan[0]["source_case"])
        # Exactly the reviewed provenance fields, so widening what a row explains
        # a step change with is a reviewed change rather than whatever the report
        # happened to be carrying that month.
        self.assertEqual(
            sorted(measured["provenance"]), sorted(TREND_PROVENANCE_FIELDS)
        )
        self.assertEqual(
            measured["provenance"]["expected_hardware"],
            perf_expected_hardware(self.test_config),
        )
        self.assertEqual(
            measured["provenance"]["package_versions"],
            report["provenance"]["package_versions"],
        )

    def test_a_config_edit_starts_a_new_trend_series(self):
        # Editing a config ends one series and starts another. The numbers either
        # side of that edit were never comparable, so a comparison keyed on the
        # digest reads the break as a break instead of as a regression -- which
        # is why the digest is part of the key and not a passenger.
        plan = resolve_measurement_plan(self.test_config)
        edited = copy.deepcopy(self.test_config)
        edited["server"]["parameters"]["mem_fraction_static"] = 0.5
        rows = [
            trend_points(
                build_report(config, [measurement_record(plan[0], benchmark_result())])
            )[0]
            for config in (self.test_config, edited)
        ]
        self.assertEqual(rows[0]["measurement_id"], rows[1]["measurement_id"])
        self.assertNotEqual(rows[0]["config_digest"], rows[1]["config_digest"])

    def test_trend_rows_render_as_one_json_object_per_line(self):
        # The series is read back as the concatenation of every run's file, so the
        # unit of the format is a line, not a document. Sorted keys and compact
        # separators are what make a diff on the data branch show only the numbers
        # that moved.
        plan = resolve_measurement_plan(self.test_config)
        report = build_report(
            self.test_config,
            [measurement_record(entry, benchmark_result()) for entry in plan],
            provenance=perf_provenance(self.test_config),
        )
        rendered = render_trend_jsonl(report)
        self.assertTrue(rendered.endswith("\n"))
        lines = rendered.splitlines()
        self.assertEqual(len(lines), len(plan))
        rows = trend_points(report)
        self.assertEqual([json.loads(line) for line in lines], rows)
        self.assertEqual(
            lines[0],
            json.dumps(rows[0], sort_keys=True, separators=(",", ":")),
        )
        # A report with nothing in it renders an empty file rather than a line
        # that would be read back as a measurement.
        self.assertEqual(render_trend_jsonl(build_report(self.test_config, [])), "")

    def test_provenance_describes_the_run_the_numbers_came_from(self):
        with mock.patch.dict(
            os.environ,
            {
                "SGLANG_PPU_SOURCE_REVISION": "c0ffee",
                "GITHUB_REPOSITORY": "flytiger-eco/sglang-for-sail",
                "GITHUB_RUN_ID": "34173202132",
            },
            clear=False,
        ):
            provenance = perf_provenance(self.four_node)
        self.assertEqual(provenance["source_revision"], "c0ffee")
        self.assertIn("34173202132", provenance["github_run_url"])
        self.assertEqual(
            provenance["expected_hardware"], perf_expected_hardware(self.four_node)
        )
        self.assertEqual(
            provenance["server_config"], self.four_node["server"]["parameters"]
        )
        self.assertEqual(
            provenance["server_environment"], perf_server_environment(self.four_node)
        )
        # The workload is what makes two numbers comparable at all, and it is not
        # part of the shared provenance block, so this suite adds it.
        self.assertEqual(provenance["workload"], self.four_node["workload"])


@unittest.skipIf(perf_suite_kit is None, "the on-machine driver needs torch")
class TestPPUPerfMultiNodeExchange(unittest.TestCase):
    """The four nodes of a group, as far as one host can stand in for them.

    The suite that serves a checkpoint across four boards cannot be run here, but
    the part of it that is protocol rather than inference can: which node reports
    where, how the nodes publish what they hold, and how a worker learns that
    rank 0 is done.  Those are the pieces whose mistakes cost an hour of cluster
    time each, so they are checked against a real directory with a stubbed device.

    ``setUpClass`` is deliberately not called -- it launches a server -- and the
    class attributes it would set are assigned directly instead.
    """

    RENDEZVOUS = {"nnodes": 4, "node_rank": 0, "dist_init_addr": "10.0.0.1:29500"}

    @classmethod
    def setUpClass(cls):
        cls.four_node = load_json(
            CONFIG_DIR / "qwen3.8" / "2.4t-a95b-fp8-144g-prefill-4n.json"
        )
        cls.single_node = load_json(
            CONFIG_DIR / "qwen3.5" / "397b-a17b-mxfp4-fp8-144g-prefill.json"
        )

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
                perf_suite_kit.torch.cuda, name, value, create=True
            )
            patcher.start()
            self.addCleanup(patcher.stop)

    def node(self, node_rank, output_dir=None, config=None):
        config = self.four_node if config is None else config
        # From what the config declares, not from which object it is: a test that
        # varies a copy of one of the two must still get the shape it copied.
        multi_node = perf_node_count(config) > 1

        class Node(perf_suite_kit.PerfSuiteMixin, unittest.TestCase):
            def test_public_perf_suite(self):
                return perf_suite_kit.PerfSuiteMixin.test_public_perf_suite(self)

        Node.test_config = config
        Node.model_config = config["model"]
        Node.plan = resolve_measurement_plan(config)
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
        case = node("test_public_perf_suite")
        case.process = SimpleNamespace(poll=lambda: exit_code)
        return case

    def test_a_group_refuses_a_results_directory_it_cannot_share(self):
        # Every pod resolves the relative default against its own working
        # directory, so the four would never observe each other's files.
        with self.assertRaises(RuntimeError) as raised:
            self.node(0, output_dir="ppu-perf-artifacts")
        self.assertIn("every node of the group shares", str(raised.exception))
        self.assertIsNone(
            self.node(
                0, output_dir="ppu-perf-artifacts", config=self.single_node
            ).rank_dir
        )

    def test_only_rank_zero_reports_where_the_workflow_collects(self):
        self.assertEqual(self.node(0).report_dir, self.root)
        for node_rank in (1, 2, 3):
            self.assertEqual(
                self.node(node_rank).report_dir,
                self.root / "ranks" / f"rank-{node_rank}",
            )

    def test_the_raw_benchmark_output_lands_inside_the_collected_report(self):
        # bench_serving appends its raw result unconditionally, deriving a name in
        # the current working directory when it is not told one, which is how a
        # measurement's per-request evidence becomes a stray file next to whatever
        # the runner happened to cd into.
        case = self.node(0)("test_public_perf_suite")
        self.assertEqual(case.raw_dir, self.root / "raw")
        self.assertTrue(case.raw_dir.is_dir())

    def test_each_node_states_the_group_to_its_own_server(self):
        # The variables the internal framework exports around the same launch.
        # MASTER_ADDR comes from the rendezvous in force rather than from the
        # injected variable of the same name, so an override that exists because
        # the injected address does not resolve is not undone here.
        reviewed = perf_server_environment(self.four_node)
        for node_rank in range(4):
            with self.subTest(node_rank=node_rank):
                self.assertEqual(
                    self.node(node_rank)._server_environment(),
                    {
                        **reviewed,
                        "MASTER_ADDR": "10.0.0.1",
                        "NNODES": "4",
                        "RANK": str(node_rank),
                    },
                )
        # A single-node launch that names no variables inherits its environment
        # untouched.
        bare = copy.deepcopy(self.single_node)
        bare["server"].pop("env")
        self.assertIsNone(self.node(0, config=bare)._server_environment())

    def test_a_config_that_names_variables_gets_them_on_a_single_node_too(self):
        # server.env is not a multi-node facility: every ported config sets it,
        # and eight of the twelve run on one board.
        self.assertEqual(
            self.node(0, config=self.single_node)._server_environment(),
            {"SGLANG_WARMUP_TIMEOUT": "3600"},
        )

    def test_provenance_describes_every_node_the_numbers_were_produced_on(self):
        for node_rank in range(4):
            node = self.node(node_rank)
            node.rank_dir.mkdir(parents=True, exist_ok=True)
            node._write_node_inventory()
        accelerator = self.node(0)._accelerator()
        self.assertEqual(accelerator["visible_device_count"], 8)
        # A throughput number attributed to thirty-two devices and one attributed
        # to eight are different pieces of evidence.
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
        case.test_public_perf_suite()

    def test_a_worker_fails_if_its_own_server_dies_first(self):
        case = self.worker_case(1, exit_code=1)
        with self.assertRaises(AssertionError) as raised:
            case.test_public_perf_suite()
        self.assertIn("lost its server with exit code 1", str(raised.exception))

    def test_a_worker_gives_up_on_a_rank_zero_that_never_finishes(self):
        case = self.worker_case(3)
        # A measurement's own duration has no bound in the config -- it is the
        # number being measured -- so the hold budget is built per measurement
        # plus the margin. The clock is moved rather than waited on.
        clock = iter([0, 10**9])
        with mock.patch.object(perf_suite_kit.time, "monotonic", lambda: next(clock)):
            with self.assertRaises(AssertionError) as raised:
                case.test_public_perf_suite()
        budget = (
            perf_suite_kit.MEASUREMENT_HOLD_BUDGET_SECONDS
            * len(self.four_node["workload"]["measurements"])
            + perf_suite_kit.WORKER_HOLD_MARGIN_SECONDS
        )
        self.assertIn(str(budget), str(raised.exception))
        self.assertIn("never published its completion", str(raised.exception))

    def test_a_setup_that_never_measured_still_owes_a_report(self):
        # The artifact of a failed run has the same shape as a successful one, so
        # the run page still names what was supposed to be measured.
        rank_zero = self.node(0)
        rank_zero._write_setup_failure("server_start", "RuntimeError: timed out")
        report = load_json(self.root / "result.json")
        self.assertEqual(report["summary"]["measured"], 0)
        self.assertEqual(
            [record["reason_code"] for record in report["measurements"]],
            ["server_start_failed", "server_start_failed"],
        )
        self.assertEqual(report["provenance"]["setup_stage"], "server_start")

    def test_teardown_releases_the_workers_before_killing_the_server(self):
        # The other order kills rank 0's process, which makes the workers'
        # schedulers exit, and a healthy run then reports three failed pods.
        order = []
        rank_zero = self.node(0)
        rank_zero.process = SimpleNamespace(pid=4321)
        release = perf_suite_kit.PerfSuiteMixin._release_worker_nodes.__func__

        def traced_release(cls):
            order.append("release")
            release(cls)

        rank_zero._release_worker_nodes = classmethod(traced_release)
        with mock.patch.object(
            perf_suite_kit, "kill_process_tree", lambda pid: order.append("kill")
        ):
            rank_zero.tearDownClass()
        self.assertEqual(order, ["release", "kill"])
        self.assertTrue((self.root / "ranks" / "rank0-complete").is_file())


if __name__ == "__main__":
    unittest.main()
