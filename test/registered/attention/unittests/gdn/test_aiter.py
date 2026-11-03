import importlib.util
import unittest
from dataclasses import replace
from unittest.mock import patch

from sglang.srt.arg_groups.attention_hook import handle_linear_attn_backend
from sglang.srt.environ import envs
from sglang.srt.layers.attention.linear.gdn_backend import GDNKernelDispatcher
from sglang.srt.layers.attention.linear.kernels import gdn_aiter
from sglang.srt.layers.attention.linear.utils import LinearAttnKernelBackend
from sglang.srt.model_executor.forward_mode import ForwardMode
from sglang.srt.runtime_context import override_platform
from sglang.srt.server_args import ServerArgs
from sglang.srt.utils import is_hip
from sglang.test.ci.ci_register import register_amd_ci
from sglang.test.kits.attention_unittest.attention_methods.gdn_attention import (
    GDNAttentionCase,
)
from sglang.test.kits.attention_unittest.runner_modes.speculative_target_verify_runner import (
    run_gdn_eagle_verify_case,
    run_gdn_eagle_verify_cuda_graph_case,
)
from sglang.test.test_utils import CustomTestCase

register_amd_ci(est_time=20, suite="stage-b-test-1-gpu-large-amd")

_has_aiter_flydsl = is_hip() and (
    importlib.util.find_spec("aiter.ops.flydsl.linear_attention_kernels") is not None
)


def _verify_case(name, prefix_lens, extend_lens, num_v_heads):
    return GDNAttentionCase(
        name=name,
        backend="triton",
        forward_mode=ForwardMode.TARGET_VERIFY,
        num_k_heads=2,
        num_v_heads=num_v_heads,
        page_size=16,
        prefix_lens=prefix_lens,
        extend_lens=extend_lens,
        linear_attn_verify_backend="aiter",
    )


@unittest.skipIf(not _has_aiter_flydsl, "ROCm with aiter FlyDSL kernels is required")
class TestAiterGDNVerifyCorrectness(CustomTestCase):
    CHAIN = _verify_case("aiter_verify_gdn_chain", (4, 7, 9), (6, 6, 6), 4)
    TREE = _verify_case("aiter_verify_gdn_tree", (5, 6), (3, 3), 2)
    VERIFY_CASES = (
        (CHAIN, 1, "eagle"),
        (TREE, 2, "eagle"),
        (replace(CHAIN, name="aiter_verify_gdn_frozen_kv_mtp"), 1, "frozen_kv_mtp"),
        (replace(CHAIN, name="aiter_verify_gdn_dflash"), 1, "dflash"),
        (replace(CHAIN, name="aiter_verify_gdn_ngram"), 1, "ngram"),
    )

    def test_eagle_verify_cases(self):
        for case, topk, spec_kind in self.VERIFY_CASES:
            with self.subTest(case=case.name, topk=topk, spec_kind=spec_kind):
                run_gdn_eagle_verify_case(self, case, topk=topk, spec_kind=spec_kind)

    def test_eagle_verify_cuda_graph_cases(self):
        for case, topk, spec_kind in self.VERIFY_CASES:
            with self.subTest(case=case.name, topk=topk, spec_kind=spec_kind):
                run_gdn_eagle_verify_cuda_graph_case(
                    self, case, topk=topk, spec_kind=spec_kind
                )

    def test_chain_verify_runs_aiter_kernels(self):
        from aiter.ops.flydsl import (
            causal_conv1d_update_kernels,
            linear_attention_kernels,
        )

        with (
            patch.object(
                linear_attention_kernels,
                "flydsl_gdr_mtp_sglang",
                wraps=linear_attention_kernels.flydsl_gdr_mtp_sglang,
            ) as gdr_spy,
            patch.object(
                causal_conv1d_update_kernels,
                "causal_conv1d_update_sglang_flydsl",
                wraps=causal_conv1d_update_kernels.causal_conv1d_update_sglang_flydsl,
            ) as conv_spy,
            patch.object(
                gdn_aiter.TritonGDNKernel,
                "target_verify",
                side_effect=AssertionError("chain verify fell back to Triton"),
            ),
        ):
            run_gdn_eagle_verify_case(self, self.CHAIN, topk=1, spec_kind="eagle")
        gdr_spy.assert_called()
        conv_spy.assert_called()


@unittest.skipIf(not _has_aiter_flydsl, "ROCm with aiter FlyDSL kernels is required")
class TestAiterGDNVerifyBackendPolicy(CustomTestCase):
    def _make_dispatcher(self):
        triton = LinearAttnKernelBackend.TRITON
        return GDNKernelDispatcher(triton, triton, LinearAttnKernelBackend.AITER)

    def test_dispatcher_routes_linear_chain_verify_to_aiter(self):
        from aiter.ops.flydsl.causal_conv1d_update_kernels import (
            causal_conv1d_update_sglang_flydsl,
        )

        with envs.SGLANG_USE_AITER.override(True):
            dispatcher = self._make_dispatcher()
        self.assertIsInstance(dispatcher.verify_kernel, gdn_aiter.AiterGDNKernel)
        self.assertNotIsInstance(
            dispatcher.tree_verify_kernel, gdn_aiter.AiterGDNKernel
        )
        self.assertIs(dispatcher.verify_conv_update, causal_conv1d_update_sglang_flydsl)
        self.assertTrue(dispatcher.target_verify_supports_strided_qkv(None))

    def test_dispatcher_requires_use_aiter(self):
        with envs.SGLANG_USE_AITER.override(False):
            with self.assertRaisesRegex(ValueError, "SGLANG_USE_AITER=1"):
                self._make_dispatcher()

    def test_replayssm_spec_rejects_aiter_verify(self):
        args = ServerArgs(
            model_path="dummy",
            linear_attn_verify_backend="aiter",
            enable_linear_replayssm_spec=True,
        )
        with override_platform(is_sm100=False), override_platform(is_cuda=False):
            with self.assertRaisesRegex(ValueError, "verify-backend aiter"):
                handle_linear_attn_backend(args)


if __name__ == "__main__":
    unittest.main()
