"""The guard for the PPU disaggregated-performance contract, without a device.

Everything here runs on a laptop: :mod:`pd_perf_eval_kit` is deliberately free of
``torch``, so every reviewed PD config can be validated and every one of the three
command lines a disaggregated run is made of can be rendered before a nightly job
spends an hour warming two copies of a 2.4T checkpoint to discover a typo.

What this file protects is the part of a PD run that no colocated test covers: two
roles rendered from one config, the tensor-parallel degree checked against the
boards of a role rather than of the group, the KV path stated once and rendered
twice, and the bootstrap port the router silently assumes.  Getting any of them
wrong produces a group that holds two boards for its whole wait budget and then
reports nothing.
"""

import copy
import unittest
from pathlib import Path

from sglang.test.ci.ci_register import register_cpu_ci, register_ppu_ci
from sglang.test.kits.pd_perf_eval_kit import (
    PD_HIGHEST_BINDABLE_PORT,
    PD_PERF_CONFIG_SCHEMA_VERSION,
    PD_RESERVED_SERVER_PARAMETERS,
    PD_ROLES,
    PD_ROUTER_ASSUMED_BOOTSTRAP_PORT,
    build_pd_server_args,
    build_router_args,
    pd_endpoint_url,
    pd_expected_hardware,
    pd_node_count,
    pd_provenance,
    pd_role_environment,
    pd_role_for_node_rank,
    pd_topology,
    resolve_pd_runtime,
    validate_pd_test_config,
)
from sglang.test.kits.perf_eval_kit import (
    DECODE_METRIC_FIELDS,
    METRIC_FIELDS,
    PerfEvalError,
    build_report,
    failed_measurement_record,
    load_json,
    measurement_record,
    render_summary,
    resolve_measurement_plan,
)

# Hardware-free, so the CPU suite owns it, and registered on the PPU per-commit
# chain as well for the reason the colocated unit file is: this is the only guard
# for the evaluator behind the nightly PD suite, and a PR that breaks a config
# should turn the gate red immediately rather than surface hours into a nightly
# measurement that holds two boards.
register_cpu_ci(est_time=3, suite="base-a-test-cpu")
register_ppu_ci(est_time=10, suite="stage-b-test-1-gpu-ppu")

DATA_ROOT = Path(__file__).parent
CONFIG_DIR = DATA_ROOT / "configs"
GLM52_1P1D = CONFIG_DIR / "glm5.2" / "fp8-channelwise-144g-pd-1p1d-4k-1500.json"


class TestPPUPdPerfEval(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.reference = load_json(GLM52_1P1D)

    def setUp(self):
        self.config = copy.deepcopy(self.reference)

    # --------------------------------------------------------------- the configs

    def test_every_reviewed_pd_config_is_executable(self):
        # One config per nightly PD job, and the workflow names them rather than
        # this file, so they are validated here instead of first failing on the
        # boards. The walk is recursive because the configs are filed per model
        # family, and it also asserts that layout.
        config_paths = sorted(CONFIG_DIR.rglob("*.json"))
        self.assertGreaterEqual(len(config_paths), 1)
        test_ids = set()
        for config_path in config_paths:
            with self.subTest(config=config_path.relative_to(CONFIG_DIR).as_posix()):
                self.assertEqual(len(config_path.relative_to(CONFIG_DIR).parts), 2)
                config = load_json(config_path)
                validate_pd_test_config(config)
                self.assertNotIn(config["test_id"], test_ids)
                test_ids.add(config["test_id"])
                # A config that renders is worth more than one that validates:
                # both roles and the router have to produce a command line, and
                # the plan has to expand into at least one measurement.
                for role in PD_ROLES:
                    self.assertTrue(build_pd_server_args(config, role))
                self.assertTrue(
                    build_router_args(
                        config,
                        host="10.0.0.1",
                        prefill_urls=["http://10.0.0.1:30000"],
                        decode_urls=["http://10.0.0.2:40000"],
                    )
                )
                self.assertTrue(resolve_measurement_plan(config))

    def test_the_ported_case_states_the_red_zone_topology_and_workload(self):
        # The numbers this line will be compared against were measured at these,
        # so a silent edit of any of them makes the comparison meaningless.
        self.assertEqual(pd_topology(self.config), "1p1d")
        self.assertEqual(pd_node_count(self.config), 2)
        plan = resolve_measurement_plan(self.config)
        self.assertEqual(len(plan), 1)
        self.assertEqual(
            (
                plan[0]["input_len"],
                plan[0]["output_len"],
                plan[0]["num_prompts"],
                plan[0]["concurrency"],
            ),
            (4096, 1500, 80, 8),
        )
        # The internal harness reports a second pass over a first, so this line
        # discards one warmup pass at the measurement's own shape before the
        # recorded one; the colocated line's default is zero.
        self.assertEqual(plan[0]["warmup_passes"], 1)

    def test_the_decoding_case_records_the_decode_side_timings(self):
        # Unlike the prefill line, this case decodes 1500 tokens, so time per
        # output token and inter-token latency are defined and recorded; the
        # source case's metric block asks for TPOT and its percentiles.  The set
        # is picked from the measurement's output_len by the shared kit.
        plan = resolve_measurement_plan(self.config)
        raw = {source: 1.0 for _, source in METRIC_FIELDS + DECODE_METRIC_FIELDS}
        raw["completed"] = plan[0]["num_prompts"]
        raw["input_lens"] = [plan[0]["input_len"]] * plan[0]["num_prompts"]
        raw["output_lens"] = [plan[0]["output_len"]] * plan[0]["num_prompts"]
        raw["errors"] = [""] * plan[0]["num_prompts"]
        record = measurement_record(plan[0], raw)
        self.assertEqual(record["status"], "measured")
        for name, _ in DECODE_METRIC_FIELDS:
            self.assertIn(name, record["metrics"])

    def test_an_unknown_schema_or_top_level_key_is_refused(self):
        self.config["schema_version"] = "ppu-perf-test-config/v1"
        with self.assertRaises(PerfEvalError):
            validate_pd_test_config(self.config)
        self.config["schema_version"] = PD_PERF_CONFIG_SCHEMA_VERSION
        self.config["router"] = {"port": 12345}
        with self.assertRaises(PerfEvalError):
            validate_pd_test_config(self.config)

    def test_a_pd_config_states_its_node_count_per_role(self):
        # And not as hardware.nnodes, which the colocated schema admits: two
        # statements of how many boards a group holds could disagree, and the
        # disagreement would surface as a tensor-parallel degree checked against
        # the wrong device count.
        self.config["hardware"]["nnodes"] = 2
        with self.assertRaises(PerfEvalError):
            validate_pd_test_config(self.config)

    # ----------------------------------------------------------------- the roles

    def test_a_role_is_sized_against_its_own_boards(self):
        # The 1p1d case is two servers of eight devices, not one of sixteen.
        self.config["decode"]["parameters"]["tp_size"] = 16
        with self.assertRaises(PerfEvalError):
            validate_pd_test_config(self.config)

    def test_a_role_that_spans_two_boards_may_double_its_tp_size(self):
        self.config["disaggregation"]["decode_nodes"] = 2
        self.config["decode"]["parameters"]["tp_size"] = 16
        validate_pd_test_config(self.config)
        self.assertEqual(pd_topology(self.config), "1p2d")
        self.assertEqual(pd_node_count(self.config), 3)
        self.assertEqual(pd_role_for_node_rank(self.config, 0), "prefill")
        self.assertEqual(pd_role_for_node_rank(self.config, 2), "decode")

    def test_the_kv_path_may_not_be_restated_inside_a_role(self):
        for field in PD_RESERVED_SERVER_PARAMETERS:
            with self.subTest(parameter=field):
                config = copy.deepcopy(self.reference)
                config["prefill"]["parameters"][field] = "prefill"
                # The message matters as much as the refusal: these names are
                # also outside the supported set, and "unknown parameter" would
                # send a reader looking for a typo instead of at the section the
                # value belongs in.
                with self.assertRaisesRegex(PerfEvalError, "disaggregation section"):
                    validate_pd_test_config(config)

    def test_data_parallel_attention_has_to_be_switched_on_to_be_sized(self):
        del self.config["decode"]["parameters"]["enable_dp_attention"]
        with self.assertRaises(PerfEvalError):
            validate_pd_test_config(self.config)

    def test_a_data_parallel_degree_has_to_divide_the_tensor_parallel_one(self):
        self.config["decode"]["parameters"]["dp_size"] = 3
        with self.assertRaises(PerfEvalError):
            validate_pd_test_config(self.config)

    def test_a_speculative_setting_needs_the_algorithm_that_reads_it(self):
        self.config["decode"]["parameters"]["speculative_num_steps"] = 2
        with self.assertRaises(PerfEvalError):
            validate_pd_test_config(self.config)
        self.config["decode"]["parameters"]["speculative_algorithm"] = "EAGLE"
        validate_pd_test_config(self.config)

    def test_an_unreviewed_environment_variable_is_refused(self):
        self.config["prefill"]["env"]["MC_MYSTERY_KNOB"] = 1
        with self.assertRaises(PerfEvalError):
            validate_pd_test_config(self.config)

    # --------------------------------------------------------- the disaggregation

    def test_the_bootstrap_port_is_pinned_to_the_one_the_router_assumes(self):
        self.config["disaggregation"]["bootstrap_port"] = 9998
        with self.assertRaises(PerfEvalError):
            validate_pd_test_config(self.config)
        self.assertEqual(
            self.reference["disaggregation"]["bootstrap_port"],
            PD_ROUTER_ASSUMED_BOOTSTRAP_PORT,
        )

    def test_two_ports_of_one_run_may_not_collide(self):
        self.config["disaggregation"]["decode_port"] = self.config["disaggregation"][
            "prefill_port"
        ]
        with self.assertRaises(PerfEvalError):
            validate_pd_test_config(self.config)

    def test_a_port_the_cluster_hands_out_or_reuses_is_refused(self):
        # 30000 is where the node port range starts and 40000 sits inside the
        # ephemeral one, which is exactly the pair the red-zone commands use: a
        # config written by copying them is the case this check exists for, and
        # it costs two boards and a checkpoint load to learn any other way.
        for port in (30000, 32767, 40000):
            with self.subTest(port=port):
                config = copy.deepcopy(self.reference)
                config["disaggregation"]["prefill_port"] = port
                with self.assertRaisesRegex(PerfEvalError, "at most"):
                    validate_pd_test_config(config)
        for field in ("prefill_port", "decode_port", "router_port"):
            self.assertLessEqual(
                self.reference["disaggregation"][field], PD_HIGHEST_BINDABLE_PORT
            )

    def test_the_kv_path_names_a_backend_sglang_accepts(self):
        self.config["disaggregation"]["transfer_backend"] = "rdma"
        with self.assertRaises(PerfEvalError):
            validate_pd_test_config(self.config)

    def test_the_rdma_devices_are_a_non_empty_list_without_repeats(self):
        for devices in ([], ["mlx5_bond_0", "mlx5_bond_0"], "mlx5_bond_0"):
            with self.subTest(devices=devices):
                config = copy.deepcopy(self.reference)
                config["disaggregation"]["ib_devices"] = devices
                with self.assertRaises(PerfEvalError):
                    validate_pd_test_config(config)

    # ------------------------------------------------------------ the rendering

    def test_each_role_renders_its_own_mode_and_the_shared_kv_path(self):
        rendered = {role: build_pd_server_args(self.config, role) for role in PD_ROLES}
        for role, args in rendered.items():
            with self.subTest(role=role):
                pairs = list(zip(args, args[1:]))
                self.assertIn(("--disaggregation-mode", role), pairs)
                self.assertIn(("--disaggregation-transfer-backend", "mooncake"), pairs)
                self.assertIn(
                    (
                        "--disaggregation-bootstrap-port",
                        str(PD_ROUTER_ASSUMED_BOOTSTRAP_PORT),
                    ),
                    pairs,
                )
                self.assertIn(
                    (
                        "--disaggregation-ib-device",
                        "mlx5_bond_0,mlx5_bond_1,mlx5_bond_2,mlx5_bond_3",
                    ),
                    pairs,
                )
                self.assertIn("--trust-remote-code", args)
        # The two roles are not one server with a flag flipped, which is the
        # reason this schema exists: prefill runs the sparse backend with no
        # graph capture, decode runs dense kernels with data-parallel attention.
        prefill, decode = rendered["prefill"], rendered["decode"]
        self.assertIn("--disable-cuda-graph", prefill)
        self.assertIn(("--attention-backend", "dsa"), list(zip(prefill, prefill[1:])))
        self.assertNotIn("--attention-backend", decode)
        self.assertIn("--enable-dp-attention", decode)
        self.assertIn(("--dp-size", "8"), list(zip(decode, decode[1:])))
        self.assertNotIn("--enable-dp-attention", prefill)

    def test_a_flag_the_config_leaves_out_is_not_rendered(self):
        # `mem_fraction_static` differs between the roles and both are rendered;
        # `ep_size` is stated only by prefill and must appear only there.
        prefill = build_pd_server_args(self.config, "prefill")
        decode = build_pd_server_args(self.config, "decode")
        self.assertIn(
            ("--mem-fraction-static", "0.85"), list(zip(prefill, prefill[1:]))
        )
        self.assertIn(("--mem-fraction-static", "0.8"), list(zip(decode, decode[1:])))
        self.assertIn("--ep-size", prefill)
        self.assertNotIn("--ep-size", decode)

    def test_the_router_is_handed_every_endpoint_of_both_roles(self):
        args = build_router_args(
            self.config,
            host="0.0.0.0",
            prefill_urls=["http://10.0.0.1:30000"],
            decode_urls=["http://10.0.0.2:40000", "http://10.0.0.3:40000"],
        )
        self.assertEqual(
            args[:5],
            [
                "python3",
                "-m",
                "sglang_router.launch_router",
                "--pd-disaggregation",
                "--mini-lb",
            ],
        )
        pairs = list(zip(args, args[1:]))
        self.assertIn(("--host", "0.0.0.0"), pairs)
        self.assertIn(("--port", "12345"), pairs)
        self.assertEqual(args.count("--prefill"), 1)
        self.assertEqual(args.count("--decode"), 2)

    def test_a_router_without_both_roles_is_refused(self):
        with self.assertRaises(PerfEvalError):
            build_router_args(
                self.config,
                host="0.0.0.0",
                prefill_urls=["http://10.0.0.1:30000"],
                decode_urls=[],
            )

    def test_an_unknown_role_is_refused_rather_than_rendered(self):
        for role in ("router", "PREFILL", ""):
            with self.subTest(role=role):
                with self.assertRaises(PerfEvalError):
                    build_pd_server_args(self.config, role)
                with self.assertRaises(PerfEvalError):
                    pd_role_environment(self.config, role)

    def test_an_endpoint_url_is_built_from_a_probed_address(self):
        self.assertEqual(pd_endpoint_url("10.0.0.1", 30000), "http://10.0.0.1:30000")

    # -------------------------------------------------------------- the runtime

    def test_the_node_rank_comes_from_the_launcher(self):
        runtime = resolve_pd_runtime(self.config, {"NODE_RANK": "1", "NNODES": "2"})
        self.assertEqual(runtime["role"], "decode")
        self.assertEqual(runtime["role_rank"], 0)
        self.assertEqual(runtime["nnodes"], 2)
        self.assertEqual(runtime["topology"], "1p1d")
        override = resolve_pd_runtime(
            self.config, {"NODE_RANK": "1", "SGLANG_PPU_PD_PERF_NODE_RANK": "0"}
        )
        self.assertEqual(override["role"], "prefill")

    def test_a_group_of_the_wrong_size_fails_before_the_boards_are_held(self):
        with self.assertRaises(PerfEvalError):
            resolve_pd_runtime(self.config, {"NODE_RANK": "0", "NNODES": "4"})

    def test_a_node_without_a_rank_or_with_an_impossible_one_is_refused(self):
        for environ in ({}, {"NODE_RANK": "2"}, {"NODE_RANK": "rank0"}):
            with self.subTest(environ=environ):
                with self.assertRaises(PerfEvalError):
                    resolve_pd_runtime(self.config, environ)

    def test_a_rank_outside_the_group_has_no_role(self):
        with self.assertRaises(PerfEvalError):
            pd_role_for_node_rank(self.config, 2)

    # ------------------------------------------------------------- the evidence

    def test_the_hardware_contract_names_the_topology(self):
        # A 1p1d of two eight-device boards and a colocated sixteen-device run
        # are not the same machine, and nothing else in the report says so.
        self.assertEqual(pd_expected_hardware(self.config), "zw-m890p-1p1dx8x144g")

    def test_the_provenance_records_both_roles_and_the_kv_path(self):
        provenance = pd_provenance(self.config, accelerator={"visible_device_count": 8})
        self.assertEqual(provenance["topology"], "1p1d")
        self.assertEqual(
            provenance["server_config"]["prefill"]["mem_fraction_static"], 0.85
        )
        self.assertEqual(
            provenance["server_config"]["decode"]["mem_fraction_static"], 0.8
        )
        self.assertEqual(
            provenance["server_config"]["disaggregation"]["transfer_backend"],
            "mooncake",
        )
        self.assertEqual(set(provenance["server_environment"]), {"prefill", "decode"})
        # Rendered as strings, because that is what a subprocess environment is.
        self.assertEqual(
            provenance["server_environment"]["prefill"][
                "SGLANG_DISAGGREGATION_ALL_CP_RANKS_TRANSFER"
            ],
            "1",
        )

    def test_a_pd_report_is_read_by_the_existing_evidence_collector(self):
        # The report half of the harness is shared with the colocated line on
        # purpose: `scripts/ci/ppu/collect_perf_evidence.sh` turns the three
        # summary prefixes into annotations, and it must keep doing so for a PD
        # report without knowing one exists.
        plan = resolve_measurement_plan(self.config)
        report = build_report(
            self.config,
            [
                failed_measurement_record(
                    entry, "server_start_failed", "no boards were held"
                )
                for entry in plan
            ],
            provenance=pd_provenance(self.config, accelerator={}),
        )
        self.assertEqual(report["summary"]["failed"], len(plan))
        summary = render_summary(report)
        self.assertIn("- FAIL", summary)


if __name__ == "__main__":
    unittest.main()
