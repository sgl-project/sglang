"""Static topology eligibility of the opt-in CP MoE transport."""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

from sglang.srt.environ import envs
from sglang.srt.layers import layernorm_sp
from sglang.srt.layers.cp import moe_fp8_transport as transport
from sglang.srt.layers.moe import utils as moe_utils
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=10, suite="base-a-test-cpu")


class TestCpMoeFp8Policy(unittest.TestCase):
    def test_static_topology_excludes_unsupported_row_variants(self):
        supported = dict(
            enable_prefill_cp=True,
            enable_cp_tp_group_sharing=True,
            cp_strategy="interleave",
            attn_cp_size=4,
            tp_size=4,
            attn_tp_size=1,
            attn_dp_size=1,
            moe_dp_size=1,
            moe_ep_size=1,
            pp_size=1,
            enable_cp_decode_attn_tp=False,
            enable_attn_tp_input_scattered=False,
        )
        runner = SimpleNamespace(is_triton=lambda: True, is_deep_gemm=lambda: False)
        a2a = SimpleNamespace(is_none=lambda: True)
        cases = [({}, True)] + [
            ({key: value}, False)
            for key, value in (
                ("enable_prefill_cp", False),
                ("enable_cp_tp_group_sharing", False),
                ("cp_strategy", "zigzag"),
                ("attn_cp_size", 2),
                ("attn_tp_size", 2),
                ("attn_dp_size", 4),
                ("moe_dp_size", 4),
                ("moe_ep_size", 4),
                ("pp_size", 2),
                ("enable_cp_decode_attn_tp", True),
                ("enable_attn_tp_input_scattered", True),
            )
        ]
        for changes, expected in cases:
            with (
                self.subTest(changes=changes),
                envs.SGLANG_CP_MOE_FP8_TRANSPORT.override(True),
                patch.object(
                    transport,
                    "get_parallel",
                    return_value=SimpleNamespace(**(supported | changes)),
                ),
                patch.object(layernorm_sp, "layernorm_sp_enabled", return_value=False),
                patch.object(moe_utils, "get_moe_runner_backend", return_value=runner),
                patch.object(moe_utils, "get_moe_a2a_backend", return_value=a2a),
            ):
                self.assertEqual(transport.resolve_cp_moe_fp8_transport(), expected)

    def test_disabled_flag_does_not_consult_parallel_state(self):
        with (
            envs.SGLANG_CP_MOE_FP8_TRANSPORT.override(False),
            patch.object(
                transport,
                "get_parallel",
                side_effect=AssertionError("unexpected topology read"),
            ),
        ):
            self.assertFalse(transport.resolve_cp_moe_fp8_transport())


if __name__ == "__main__":
    unittest.main()
