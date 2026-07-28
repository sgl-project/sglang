import unittest
from types import SimpleNamespace

from sglang.srt.arg_groups.overrides import resolved_view
from sglang.srt.server_args import ServerArgs
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class TestFlashInferA2APrefillCP(CustomTestCase):
    def _args(
        self,
        *,
        enable_dp_attention: bool,
        dp_size: int,
        enable_prefill_cp: bool,
        attn_cp_size: int,
    ) -> ServerArgs:
        args = ServerArgs(model_path="dummy")
        args.tp_size = 4
        args.dp_size = dp_size
        args.enable_dp_attention = enable_dp_attention
        args.enable_prefill_cp = enable_prefill_cp
        args.attn_cp_size = attn_cp_size
        args.moe_a2a_backend = "flashinfer"
        args.moe_runner_backend = "flashinfer_trtllm_routed"
        return args

    def test_parallelism_support_truth_table(self):
        for case in (
            dict(
                enable_dp_attention=True,
                dp_size=4,
                enable_prefill_cp=False,
                attn_cp_size=1,
                expected=True,
            ),
            dict(
                enable_dp_attention=False,
                dp_size=1,
                enable_prefill_cp=True,
                attn_cp_size=4,
                expected=True,
            ),
            dict(
                enable_dp_attention=False,
                dp_size=1,
                enable_prefill_cp=True,
                attn_cp_size=2,
                expected=False,
            ),
            dict(
                enable_dp_attention=False,
                dp_size=1,
                enable_prefill_cp=False,
                attn_cp_size=1,
                expected=False,
            ),
        ):
            with self.subTest(case=case):
                args = self._args(**{k: v for k, v in case.items() if k != "expected"})
                self.assertEqual(
                    args._supports_flashinfer_a2a_parallelism(), case["expected"]
                )

    def test_handle_a2a_moe_admits_full_prefill_cp_with_routed_runner(self):
        args = self._args(
            enable_dp_attention=False,
            dp_size=1,
            enable_prefill_cp=True,
            attn_cp_size=4,
        )
        args.get_model_config = lambda: SimpleNamespace(nvfp4_moe_meta=None)

        args._handle_a2a_moe()

        view = resolved_view(args)
        self.assertEqual(view.moe_a2a_backend, "flashinfer")
        self.assertEqual(view.moe_runner_backend, "flashinfer_trtllm_routed")

    def test_handle_a2a_moe_rejects_partial_prefill_cp_with_diagnostic(self):
        args = self._args(
            enable_dp_attention=False,
            dp_size=1,
            enable_prefill_cp=True,
            attn_cp_size=2,
        )

        with self.assertRaisesRegex(
            AssertionError,
            "full prefill context parallelism with attn_cp_size == tp_size",
        ):
            args._handle_a2a_moe()


if __name__ == "__main__":
    unittest.main()
