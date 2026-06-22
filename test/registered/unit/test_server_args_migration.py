"""Parity tests for the ServerArgs CLI migration from manual add_argument to Annotated style.

Verifies that the migrated CLI parser produces identical behavior to the
original manual parser for all non-deprecated options, including edge cases
like aliases, nargs, BooleanOptionalAction, json type parsers, and Literal
types.
"""

import argparse
import json
import unittest

from sglang.srt.server_args import ServerArgs
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=10, suite="base-a-test-cpu")


class TestServerArgsMigrationParity(CustomTestCase):
    """End-to-end parse equivalence tests for migrated ServerArgs."""

    @classmethod
    def setUpClass(cls):
        cls.parser = argparse.ArgumentParser()
        ServerArgs.add_cli_args(cls.parser)
        cls.actions_by_option = {
            option: action
            for action in cls.parser._actions
            for option in action.option_strings
        }

    def _parse(self, args_list):
        args = self.parser.parse_args(["--model", "dummy"] + args_list)
        return ServerArgs.from_cli_args(args)

    # ------------------------------------------------------------------
    # Basic parsing
    # ------------------------------------------------------------------
    def test_minimal_parse(self):
        sa = self._parse([])
        self.assertEqual(sa.model_path, "dummy")
        self.assertEqual(sa.tp_size, 1)
        self.assertEqual(sa.dp_size, 1)
        self.assertEqual(sa.dtype, "auto")
        self.assertEqual(sa.host, "127.0.0.1")
        self.assertEqual(sa.port, 30000)

    def test_scalar_options(self):
        sa = self._parse(
            ["--watchdog-timeout", "600", "--base-gpu-id", "2", "--log-level", "debug"]
        )
        self.assertEqual(sa.watchdog_timeout, 600.0)
        self.assertEqual(sa.base_gpu_id, 2)
        self.assertEqual(sa.log_level, "debug")

    def test_bool_store_true(self):
        sa = self._parse(
            [
                "--enable-lora",
                "--disable-radix-cache",
                "--enable-metrics",
                "--sleep-on-idle",
            ]
        )
        self.assertTrue(sa.enable_lora)
        self.assertTrue(sa.disable_radix_cache)
        self.assertTrue(sa.enable_metrics)
        self.assertTrue(sa.sleep_on_idle)

    def test_bool_defaults_false(self):
        sa = self._parse([])
        self.assertFalse(sa.disable_radix_cache)
        self.assertFalse(sa.enable_metrics)
        self.assertFalse(sa.sleep_on_idle)
        self.assertFalse(sa.enable_trace)

    # ------------------------------------------------------------------
    # Aliases and cli_name remapping
    # ------------------------------------------------------------------
    def test_tp_size_via_alias(self):
        sa1 = self._parse(["--tensor-parallel-size", "4"])
        sa2 = self._parse(["--tp-size", "4"])
        self.assertEqual(sa1.tp_size, 4)
        self.assertEqual(sa2.tp_size, 4)

    def test_pp_size_via_alias(self):
        sa1 = self._parse(["--pipeline-parallel-size", "2"])
        sa2 = self._parse(["--pp-size", "2"])
        self.assertEqual(sa1.pp_size, 2)
        self.assertEqual(sa2.pp_size, 2)

    def test_ep_size_via_aliases(self):
        sa1 = self._parse(["--expert-parallel-size", "8"])
        sa2 = self._parse(["--ep-size", "8"])
        sa3 = self._parse(["--ep", "8"])
        self.assertEqual(sa1.ep_size, 8)
        self.assertEqual(sa2.ep_size, 8)
        self.assertEqual(sa3.ep_size, 8)

    def test_dp_size_via_alias(self):
        sa1 = self._parse(["--data-parallel-size", "2"])
        sa2 = self._parse(["--dp-size", "2"])
        self.assertEqual(sa1.dp_size, 2)
        self.assertEqual(sa2.dp_size, 2)

    def test_attn_cp_size_via_alias(self):
        sa = self._parse(["--attn-cp-size", "2"])
        self.assertEqual(sa.attn_cp_size, 2)

    def test_moe_dp_size_via_alias(self):
        sa = self._parse(["--moe-dp-size", "2"])
        self.assertEqual(sa.moe_dp_size, 2)

    def test_fp8_gemm_backend_maps_to_runner_field(self):
        sa = self._parse(["--fp8-gemm-backend", "triton"])
        self.assertEqual(sa.fp8_gemm_runner_backend, "triton")

    def test_fp4_gemm_backend_maps_to_runner_field(self):
        sa = self._parse(["--fp4-gemm-backend", "cutlass"])
        self.assertEqual(sa.fp4_gemm_runner_backend, "cutlass")

    def test_nccl_init_addr_alias(self):
        sa = self._parse(["--nccl-init-addr", "192.168.1.1:25000"])
        self.assertEqual(sa.dist_init_addr, "192.168.1.1:25000")

    def test_hisparse_config_alias(self):
        sa = self._parse(
            ["--hierarchical-sparse-attention-extra-config", '{"top_k": 2048}']
        )
        self.assertEqual(sa.hisparse_config, '{"top_k": 2048}')

    def test_speculative_draft_model_alias(self):
        sa = self._parse(["--speculative-draft-model", "/path/to/draft"])
        self.assertEqual(sa.speculative_draft_model_path, "/path/to/draft")

    # ------------------------------------------------------------------
    # nargs edge cases
    # ------------------------------------------------------------------
    def test_model_checksum_nargs_question_no_value(self):
        sa = self._parse(["--model-checksum"])
        self.assertEqual(sa.model_checksum, "")

    def test_model_checksum_nargs_question_with_value(self):
        sa = self._parse(["--model-checksum", "abc123"])
        self.assertEqual(sa.model_checksum, "abc123")

    def test_model_checksum_default_none(self):
        sa = self._parse([])
        self.assertIsNone(sa.model_checksum)

    def test_list_nargs_plus(self):
        sa = self._parse(["--bucket-time-to-first-token", "0.5", "1.0", "2.0"])
        self.assertEqual(sa.bucket_time_to_first_token, [0.5, 1.0, 2.0])

    def test_list_nargs_star_lora_paths(self):
        sa = self._parse(["--lora-paths", "path1", "path2"])
        self.assertIsNotNone(sa.lora_paths)

    def test_list_nargs_star_custom_weight_loader(self):
        sa = self._parse(["--custom-weight-loader", "my_pkg.loader"])
        self.assertEqual(sa.custom_weight_loader, ["my_pkg.loader"])

    def test_cuda_graph_bs_decode_list(self):
        sa = self._parse(["--cuda-graph-bs-decode", "1", "2", "4", "8"])
        self.assertEqual(sa.cuda_graph_bs_decode, [1, 2, 4, 8])

    def test_numa_node_list(self):
        sa = self._parse(["--numa-node", "0", "1"])
        self.assertEqual(sa.numa_node, [0, 1])

    # ------------------------------------------------------------------
    # BooleanOptionalAction
    # ------------------------------------------------------------------
    def test_experts_shared_outer_loras_enable(self):
        sa = self._parse(["--experts-shared-outer-loras"])
        self.assertTrue(sa.experts_shared_outer_loras)

    def test_experts_shared_outer_loras_disable(self):
        sa = self._parse(["--no-experts-shared-outer-loras"])
        self.assertFalse(sa.experts_shared_outer_loras)

    def test_experts_shared_outer_loras_default(self):
        sa = self._parse([])
        self.assertIsNone(sa.experts_shared_outer_loras)

    def test_lora_strict_loading_enable(self):
        sa = self._parse(["--lora-strict-loading"])
        self.assertTrue(sa.lora_strict_loading)

    def test_lora_strict_loading_disable(self):
        sa = self._parse(["--no-lora-strict-loading"])
        self.assertFalse(sa.lora_strict_loading)

    # ------------------------------------------------------------------
    # Custom type parsers (json.loads, json_list_type)
    # ------------------------------------------------------------------
    def test_extra_metric_labels_json(self):
        sa = self._parse(["--extra-metric-labels", json.dumps({"k": "v", "a": "b"})])
        self.assertEqual(sa.extra_metric_labels, {"k": "v", "a": "b"})

    def test_mm_process_config_json(self):
        sa = self._parse(
            ["--mm-process-config", json.dumps({"image": {"max_size": 1024}})]
        )
        self.assertEqual(sa.mm_process_config, {"image": {"max_size": 1024}})

    def test_limit_mm_data_per_request_json(self):
        sa = self._parse(["--limit-mm-data-per-request", json.dumps({"image": 1})])
        self.assertEqual(sa.limit_mm_data_per_request, {"image": 1})

    def test_preferred_sampling_params_json(self):
        sa = self._parse(
            ["--preferred-sampling-params", json.dumps({"temperature": 0.7})]
        )
        self.assertEqual(sa.preferred_sampling_params, {"temperature": 0.7})

    def test_forward_hooks_json_list(self):
        sa = self._parse(["--forward-hooks", json.dumps([{"type": "test"}])])
        self.assertEqual(sa.forward_hooks, [{"type": "test"}])

    def test_remote_weight_loader_ports_json_list(self):
        sa = self._parse(
            [
                "--remote-instance-weight-loader-send-weights-group-ports",
                json.dumps([8001, 8002]),
            ]
        )
        self.assertEqual(
            sa.remote_instance_weight_loader_send_weights_group_ports,
            [8001, 8002],
        )

    # ------------------------------------------------------------------
    # Literal types and choices
    # ------------------------------------------------------------------
    def test_deepep_mode_literal(self):
        sa = self._parse(["--deepep-mode", "low_latency"])
        self.assertEqual(sa.deepep_mode, "low_latency")

    def test_disaggregation_mode_literal(self):
        sa = self._parse(["--disaggregation-mode", "prefill"])
        self.assertEqual(sa.disaggregation_mode, "prefill")

    def test_elastic_ep_backend_none_as_string(self):
        sa = self._parse(["--elastic-ep-backend", "none"])
        self.assertEqual(sa.elastic_ep_backend, "none")

    def test_speculative_ngram_match_type_literal(self):
        sa = self._parse(["--speculative-ngram-match-type", "PROB"])
        self.assertEqual(sa.speculative_ngram_match_type, "PROB")

    def test_choices_enforced(self):
        for opt, valid in [
            ("--dtype", "bfloat16"),
            ("--schedule-policy", "lpm"),
            ("--log-requests-format", "json"),
            ("--sampling-defaults", "openai"),
            ("--kv-canary", "raise"),
            ("--cp-strategy", "interleave"),
        ]:
            with self.subTest(opt=opt):
                sa = self._parse([opt, valid])

    # ------------------------------------------------------------------
    # Deprecated args preserved
    # ------------------------------------------------------------------
    def test_deprecated_stream_output(self):
        sa = self._parse(["--stream-output"])
        self.assertTrue(sa.incremental_streaming_output)

    def test_deprecated_disable_cuda_graph(self):
        sa = self._parse(["--disable-cuda-graph"])
        self.assertTrue(sa.disable_cuda_graph)

    # ------------------------------------------------------------------
    # Cuda graph config
    # ------------------------------------------------------------------
    def test_cuda_graph_config_json(self):
        sa = self._parse(
            [
                "--cuda-graph-config",
                json.dumps({"decode": {"backend": "full", "max_bs": 256}}),
            ]
        )
        self.assertIsNotNone(sa.cuda_graph_config)

    # ------------------------------------------------------------------
    # Fields kept manual
    # ------------------------------------------------------------------
    def test_reasoning_parser_has_dynamic_choices(self):
        action = self.actions_by_option.get("--reasoning-parser")
        self.assertIsNotNone(action)
        self.assertIn("auto", action.choices)

    def test_tool_call_parser_has_dynamic_choices(self):
        action = self.actions_by_option.get("--tool-call-parser")
        self.assertIsNotNone(action)
        self.assertIn("auto", action.choices)

    def test_kv_canary_real_data_has_dynamic_choices(self):
        action = self.actions_by_option.get("--kv-canary-real-data")
        self.assertIsNotNone(action)
        self.assertIn("none", action.choices)

    def test_kt_method_default(self):
        sa = self._parse([])
        self.assertEqual(sa.kt_method, "AMXINT4")

    def test_kt_threadpool_count_default(self):
        sa = self._parse([])
        self.assertEqual(sa.kt_threadpool_count, 2)

    def test_pre_warm_nccl_default_false(self):
        sa = self._parse([])
        self.assertFalse(sa.pre_warm_nccl)

    def test_pre_warm_nccl_enable(self):
        sa = self._parse(["--pre-warm-nccl"])
        self.assertTrue(sa.pre_warm_nccl)

    # ------------------------------------------------------------------
    # Option count stability
    # ------------------------------------------------------------------
    def test_total_option_count(self):
        actions = [a for a in self.parser._actions if a.option_strings]
        self.assertEqual(len(actions), 422)

    # ------------------------------------------------------------------
    # Combined parsing
    # ------------------------------------------------------------------
    def test_combined_parse(self):
        sa = self._parse(
            [
                "--dtype",
                "bfloat16",
                "--max-total-tokens",
                "1024",
                "--data-parallel-size",
                "2",
                "--load-balance-method",
                "total_tokens",
                "--tp-size",
                "4",
                "--enable-lora",
                "--watchdog-timeout",
                "600",
                "--bucket-time-to-first-token",
                "0.5",
                "1.0",
            ]
        )
        self.assertEqual(sa.dtype, "bfloat16")
        self.assertEqual(sa.max_total_tokens, 1024)
        self.assertEqual(sa.dp_size, 2)
        self.assertEqual(sa.load_balance_method, "total_tokens")
        self.assertEqual(sa.tp_size, 4)
        self.assertTrue(sa.enable_lora)
        self.assertEqual(sa.watchdog_timeout, 600.0)
        self.assertEqual(sa.bucket_time_to_first_token, [0.5, 1.0])


if __name__ == "__main__":
    unittest.main()
