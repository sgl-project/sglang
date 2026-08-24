import unittest

from sglang.benchmark.dynamic_parallel import (
    DeploymentMode,
    GridCase,
    RunRecord,
    build_inputs,
    build_mode_server_args,
    compare_with_tp,
    parse_args,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


def _record(mode: str, *, output_ids=None, logprobs=None):
    case = GridCase(
        batch_size=2,
        input_length=32,
        output_length=2,
        prefix_hit_ratio=0.5,
        repeat=0,
    )
    return RunRecord(
        mode=mode,
        case={},
        case_key=case.key,
        wall_latency_s=0.1,
        cached_tokens=[16, 16],
        prompt_tokens=[32, 32],
        completion_tokens=[2, 2],
        output_ids=output_ids or [[1, 2], [3, 4]],
        output_logprobs=logprobs or [[-0.1, -0.2], [-0.3, -0.4]],
        mode_metrics_before={},
        mode_metrics_after={},
        server_info={},
    )


class TestDynamicParallelBenchmark(unittest.TestCase):
    def test_build_inputs_has_exact_shared_prefix(self):
        prefix, inputs = build_inputs(
            batch_size=4,
            input_length=100,
            prefix_hit_ratio=0.75,
            seed=7,
            token_id_low=100,
            token_id_high=200,
        )

        self.assertEqual(len(prefix), 75)
        self.assertEqual([len(row) for row in inputs], [100] * 4)
        self.assertTrue(all(row[:75] == prefix for row in inputs))
        self.assertGreater(len({tuple(row[75:]) for row in inputs}), 1)

    def test_mode_specific_server_args(self):
        self.assertEqual(
            build_mode_server_args(
                DeploymentMode.TP,
                cp_size=8,
                dcp_size=8,
                dynamic_include_dcp=False,
            ),
            [],
        )
        cp_args = build_mode_server_args(
            DeploymentMode.PREFILL_CP,
            cp_size=8,
            dcp_size=8,
            dynamic_include_dcp=False,
        )
        self.assertIn("--enable-prefill-cp", cp_args)
        self.assertIn("--enable-cp-decode-attn-tp", cp_args)

        dynamic_args = build_mode_server_args(
            DeploymentMode.DYNAMIC,
            cp_size=8,
            dcp_size=8,
            dynamic_include_dcp=True,
        )
        self.assertIn("--enable-dynamic-attn-parallel", dynamic_args)
        self.assertIn("--dynamic-attn-parallel-enable-dcp", dynamic_args)
        self.assertIn("--dcp-size", dynamic_args)

    def test_parity_comparison_reports_logprob_delta(self):
        baseline = _record(DeploymentMode.TP.value)
        candidate = _record(
            DeploymentMode.PREFILL_CP.value,
            logprobs=[[-0.11, -0.19], [-0.31, -0.39]],
        )

        result = compare_with_tp(baseline, candidate, logprob_tolerance=0.02)

        self.assertTrue(result["exact_output_match"])
        self.assertTrue(result["logprob_shape_match"])
        self.assertAlmostEqual(result["max_logprob_delta"], 0.01)
        self.assertTrue(result["passed"])

    def test_parity_fails_on_output_mismatch(self):
        baseline = _record(DeploymentMode.TP.value)
        candidate = _record(
            DeploymentMode.DCP.value,
            output_ids=[[1, 9], [3, 4]],
        )

        result = compare_with_tp(baseline, candidate, logprob_tolerance=0.1)

        self.assertFalse(result["exact_output_match"])
        self.assertFalse(result["passed"])

    def test_parse_args_requires_tp_first_for_comparison(self):
        with self.assertRaises(SystemExit):
            parse_args(
                [
                    "--model-path",
                    "model",
                    "--modes",
                    "prefill_cp,tp",
                ]
            )


if __name__ == "__main__":
    unittest.main()
