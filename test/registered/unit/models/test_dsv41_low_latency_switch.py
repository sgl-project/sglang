"""--enable-dsv41-low-latency-decode: the deployment check names every unmet
requirement, and the row bounds stay where they were measured."""

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

import unittest
from types import SimpleNamespace
from unittest.mock import patch

from sglang.srt.models.dsv41_low_latency import switch
from sglang.srt.runtime_context import override_platform
from sglang.test.test_utils import CustomTestCase

# (field, value that breaks it, a phrase of the requirement it breaks)
_BREAKS = (
    ("tp_size", 8, "--tp-size 4"),
    ("ep_size", 4, "--ep-size 1"),
    ("moe_a2a_backend", "deepep", "--moe-a2a-backend none"),
    ("pp_size", 2, "pipeline parallelism"),
    ("attn_dp_size", 4, "DP attention"),
    ("attn_cp_size", 2, "context parallelism"),
    ("nnodes", 2, "one node"),
    ("speculative_algorithm", "EAGLE", "DSPARK"),
    ("dsv4_attn_backend", "trtllm", "attention backend"),
    ("disable_cuda_graph", True, "decode CUDA graphs"),
    ("enable_encoder_swa_bounded_replay", True, "encoder SWA"),
    ("enable_decoder_swa_bounded_replay", True, "decoder SWA"),
    ("enable_two_batch_overlap", True, "two-batch overlap"),
    ("disaggregation_mode", "decode", "PD disaggregation"),
    ("enable_lora", True, "LoRA"),
    ("enable_deterministic_inference", True, "deterministic"),
    ("enable_fp32_lm_head", True, "--enable-fp32-lm-head"),
    ("rl_on_policy_target", "fsdp", "--rl-on-policy-target"),
)


def _validated(**overrides):
    cfg = SimpleNamespace(
        tp_size=4,
        ep_size=1,
        moe_a2a_backend="none",
        pp_size=1,
        attn_dp_size=1,
        attn_cp_size=1,
        nnodes=1,
        speculative_algorithm="DSPARK",
        dsv4_attn_backend="auto",
        disable_cuda_graph=False,
        cuda_graph_config=SimpleNamespace(decode=SimpleNamespace(backend="full")),
        enable_encoder_swa_bounded_replay=False,
        enable_decoder_swa_bounded_replay=False,
        enable_two_batch_overlap=False,
        disaggregation_mode="null",
        enable_lora=None,
        enable_fp32_lm_head=False,
        rl_on_policy_target=None,
        enable_deterministic_inference=False,
    )
    for name, value in overrides.items():
        setattr(cfg, name, value)
    return cfg


class TestDeploymentCheck(CustomTestCase):
    def setUp(self):
        # The fake args are their own resolving view; DP attention reads one field.
        for patcher in (
            patch(
                "sglang.srt.arg_groups.overrides.resolving_view",
                side_effect=lambda a: a,
            ),
            patch(
                "sglang.srt.runtime_context.attn_dp_enabled_of",
                side_effect=lambda cfg: cfg.attn_dp_size > 1,
            ),
            override_platform(is_cuda=True, device_sm=103),
        ):
            patcher.__enter__()
            self.addCleanup(patcher.__exit__, None, None, None)

    def test_validated_deployment_passes(self):
        self.assertEqual(switch.deployment_violations(_validated(), "deepseek_v41"), [])
        for backend in ("auto", "flashmla"):
            cfg = _validated(dsv4_attn_backend=backend)
            switch.check_deployment(cfg, "deepseek_v41")

    def test_each_requirement_is_checked_on_its_own(self):
        for field, value, phrase in _BREAKS:
            with self.subTest(field=field):
                violations = switch.deployment_violations(
                    _validated(**{field: value}), "deepseek_v41"
                )
                self.assertEqual(len(violations), 1, violations)
                self.assertIn(phrase, violations[0])
        graphs_off = SimpleNamespace(decode=SimpleNamespace(backend="disabled"))
        violations = switch.deployment_violations(
            _validated(cuda_graph_config=graphs_off), "deepseek_v41"
        )
        self.assertEqual(len(violations), 1)
        self.assertIn("decode CUDA graphs", violations[0])

    def test_model_and_gpu_are_checked(self):
        self.assertEqual(
            len(switch.deployment_violations(_validated(), "deepseek_v4")), 1
        )
        # GB200 / B200 (SM100) are not the validated GPU.
        with override_platform(device_sm=100):
            violations = switch.deployment_violations(_validated(), "deepseek_v41")
        self.assertEqual(len(violations), 1)
        self.assertIn("SM103", violations[0])

    def test_check_names_every_violation_in_one_error(self):
        cfg = _validated(**{field: value for field, value, _ in _BREAKS})
        with override_platform(is_cuda=False):
            violations = switch.deployment_violations(cfg, "deepseek_v4")
            # Every row of _BREAKS, the model and the GPU.
            self.assertEqual(len(violations), len(_BREAKS) + 2)
            with self.assertRaises(ValueError) as raised:
                switch.check_deployment(cfg, "deepseek_v4")
        message = str(raised.exception)
        self.assertIn("--enable-dsv41-low-latency-decode", message)
        for violation in violations:
            self.assertIn(violation, message)


class TestRowBounds(CustomTestCase):
    def test_measured_bounds(self):
        # Measured on GB300 (see switch.py); change them only with new measurements.
        self.assertEqual(switch.MXFP8_MAX_ROWS, 8)
        self.assertEqual(switch.LOW_M_BF16_MAX_ROWS, 8)
        self.assertEqual(switch.ROUTED_EXPERT_MAX_ROWS, 48)
        self.assertEqual(switch.SWAPAB_MAX_ROWS, 24)
        self.assertEqual(switch.LM_HEAD_MAX_ROWS, 16)

    def test_bounds_fit_their_kernels(self):
        from sglang.srt.models.dsv41_low_latency import routed_expert_gemm

        # RoutedExpertGemm rejects a bound outside the route kernels' range.
        self.assertTrue(
            routed_expert_gemm.MAX_ROWS
            <= switch.ROUTED_EXPERT_MAX_ROWS
            <= routed_expert_gemm.WIDE_MAX_ROWS
        )
        # The wo_b mailbox (sized by WO_B_MAILBOX_MAX_TOKENS) must hold every row wo_b publishes.
        self.assertLessEqual(switch.MXFP8_MAX_ROWS, switch.WO_B_MAILBOX_MAX_TOKENS)


if __name__ == "__main__":
    unittest.main()
