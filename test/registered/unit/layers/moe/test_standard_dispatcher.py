import sys
from unittest.mock import patch

import torch

from sglang.srt.layers.moe.fused_moe_triton.layer import FusedMoE
from sglang.srt.layers.moe.moe_runner.base import MoeRunnerConfig
from sglang.srt.layers.moe.token_dispatcher.standard import (
    StandardDispatcher,
)
from sglang.srt.layers.moe.topk import StandardTopKOutput, TopK, TopKOutputFormat
from sglang.srt.layers.moe.utils import MoeA2ABackend, MoeRunnerBackend
from sglang.srt.layers.quantization.fp8 import Fp8MoEMethod
from sglang.srt.layers.quantization.unquant import UnquantizedFusedMoEMethod
from sglang.srt.runtime_context import get_context, get_flags, get_parallel
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=12, suite="base-a-test-cpu")


def test_dispatch_preserves_negative_routing_sentinel():
    dispatcher = object.__new__(StandardDispatcher)
    dispatcher.moe_ep_size = 2
    dispatcher.skip_local_expert_mapping = False
    dispatcher.local_expert_mapping = torch.tensor([-1, -1, 0, 1], dtype=torch.int32)
    dispatcher.use_aiter_moe_runner = False
    dispatcher.expert_mask_gpu = None

    topk_ids = torch.tensor([[-1, 3], [2, 0]], dtype=torch.int32)
    topk_output = StandardTopKOutput(
        topk_weights=torch.ones_like(topk_ids, dtype=torch.float32),
        topk_ids=topk_ids,
        router_logits=None,
    )

    with patch(
        "sglang.srt.layers.moe.token_dispatcher.standard."
        "should_use_flashinfer_cutlass_moe_fp4_allgather",
        return_value=False,
    ):
        output = dispatcher.dispatch(torch.ones((2, 4)), topk_output)

    expected = torch.tensor([[-1, 1], [0, -1]], dtype=torch.int32)
    assert torch.equal(output.topk_output.topk_ids, expected)


class TestLayerRunnerDispatch(CustomTestCase):
    def setUp(self):
        self.enterContext(
            patch(
                "sglang.srt.layers.moe.token_dispatcher.standard.get_device",
                return_value="cpu",
            )
        )
        self.enterContext(
            get_context().override_server_args(model_path="dummy", device="cpu")
        )
        self.enterContext(
            get_flags().moe.override(
                runner_backend=MoeRunnerBackend.TRITON,
                a2a_backend=MoeA2ABackend.NONE,
            )
        )
        self.enterContext(
            get_parallel().override(
                tp_size=2,
                tp_rank=1,
                attn_tp_size=2,
                attn_tp_rank=1,
                moe_ep_size=2,
                moe_ep_rank=1,
                moe_tp_size=1,
                moe_tp_rank=0,
            )
        )

    @staticmethod
    def _config():
        return MoeRunnerConfig(
            num_experts=4,
            num_local_experts=2,
            num_fused_shared_experts=0,
            hidden_size=4,
            intermediate_size_per_partition=96,
            top_k=2,
        )

    def _build_layer(self, method):
        with patch.object(method, "create_weights"):
            return FusedMoE(
                num_experts=4,
                top_k=2,
                hidden_size=4,
                intermediate_size=96,
                layer_id=0,
                quant_method=method,
            )

    def _assert_local_routes(self, config):
        dispatcher = StandardDispatcher(config)
        routes = StandardTopKOutput(
            topk_weights=torch.ones((2, 2)),
            topk_ids=torch.tensor([[-1, 3], [2, 0]], dtype=torch.int32),
            router_logits=None,
        )
        with patch(
            "sglang.srt.layers.moe.token_dispatcher.standard."
            "should_use_flashinfer_cutlass_moe_fp4_allgather",
            return_value=False,
        ):
            output = dispatcher.dispatch(torch.ones((2, 4)), routes)
        self.assertTrue(
            torch.equal(
                output.topk_output.topk_ids,
                torch.tensor([[-1, 1], [0, -1]], dtype=torch.int32),
            )
        )
        topk = TopK(top_k=2, layer_id=0, allow_routed_experts_capture=False)
        topk.bind_runner_backend(config.runner_backend)
        self.assertEqual(topk.topk_config.output_format, TopKOutputFormat.STANDARD)
        self.assertFalse(dispatcher.use_aiter_moe_runner)

    def test_fp8_fallback_uses_local_ids_and_materialized_routes(self):
        """FP8 experts under an FP4 runner request must not receive global IDs."""
        for requested in (
            MoeRunnerBackend.FLASHINFER_CUTLASS,
            MoeRunnerBackend.FLASHINFER_CUTEDSL,
        ):
            with (
                self.subTest(requested=requested),
                get_flags().moe.override(runner_backend=requested),
            ):
                method = object.__new__(Fp8MoEMethod)
                method.is_fp4_expert = False
                method.use_mxfp8 = False
                layer = self._build_layer(method)
                self._assert_local_routes(layer.moe_runner_config)

    def test_unquantized_fallback_is_independent_of_requested_runner(self):
        """A draft BF16 layer may fall back while its target keeps global IDs."""
        for requested in (
            MoeRunnerBackend.FLASHINFER_MEGAMOE,
            MoeRunnerBackend.FLASHINFER_CUTEDSL,
        ):
            with (
                self.subTest(requested=requested),
                get_flags().moe.override(runner_backend=requested),
            ):
                layer = self._build_layer(UnquantizedFusedMoEMethod())
                self._assert_local_routes(layer.moe_runner_config)

    def test_unsupported_aiter_shape_keeps_triton_local_ids(self):
        """Enabling AITER must not suppress local remapping after a shape fallback."""
        with (
            get_flags().moe.override(runner_backend=MoeRunnerBackend.AUTO),
            patch("sglang.srt.layers.quantization.unquant._use_aiter", True),
        ):
            layer = self._build_layer(UnquantizedFusedMoEMethod())
            self._assert_local_routes(layer.moe_runner_config)

    def test_fp8_fallback_keeps_dp_allgather_in_bf16(self):
        """FP8 fallback experts still gather DP-local rows without FP4 quantization."""
        captured = {}

        class Group:
            world_size = 2

            def all_gatherv(self, payloads, sizes):
                captured["payloads"] = payloads
                captured["sizes"] = sizes
                weights, ids, hidden = payloads
                return [
                    torch.cat([weights, weights]),
                    torch.tensor([[0, 3], [2, 1]], dtype=ids.dtype),
                    torch.cat([hidden, hidden + 1]),
                ]

        config = self._config()
        config.runner_backend = MoeRunnerBackend.TRITON
        dispatcher = StandardDispatcher(config)
        hidden = torch.zeros((1, 4), dtype=torch.bfloat16)
        topk = StandardTopKOutput(
            torch.ones((1, 2)), torch.tensor([[0, 3]], dtype=torch.int32), None
        )
        with (
            get_parallel().override(tp_group=Group()),
            patch(
                "sglang.srt.layers.moe.token_dispatcher.standard."
                "should_use_flashinfer_cutlass_moe_fp4_allgather",
                return_value=True,
            ),
            patch(
                "sglang.srt.layers.moe.token_dispatcher.standard.get_dp_global_num_tokens",
                return_value=[1, 1],
            ),
        ):
            output = dispatcher.dispatch(hidden, topk)
        self.assertIs(captured["payloads"][2], hidden)
        self.assertEqual(captured["sizes"], [1, 1])
        self.assertEqual(output.hidden_states.dtype, torch.bfloat16)
        self.assertIsNone(output.hidden_states_scale)
        self.assertTrue(
            torch.equal(
                output.topk_output.topk_ids,
                torch.tensor([[-1, 1], [0, -1]], dtype=torch.int32),
            )
        )

    def test_direct_kernels_declare_their_execution_contract(self):
        """A missing or dormant runner must not change a direct kernel's ID ABI."""
        from sglang.srt.layers.quantization.compressed_tensors.schemes.compressed_tensors_w4a4_mxint4_moe import (
            CompressedTensorsMxInt4MoE,
        )
        from sglang.srt.layers.quantization.modelopt_quant import ModelOptFp8MoEMethod
        from sglang.srt.layers.quantization.mxfp4_flashinfer_trtllm_moe import (
            Mxfp4FlashinferTrtllmMoEMethod,
        )

        cases = [
            (CompressedTensorsMxInt4MoE, MoeRunnerBackend.FLASHINFER_TRTLLM, False),
            (Mxfp4FlashinferTrtllmMoEMethod, MoeRunnerBackend.FLASHINFER_MXFP4, True),
            (ModelOptFp8MoEMethod, MoeRunnerBackend.FLASHINFER_TRTLLM, False),
        ]
        for method_type, backend, fp4 in cases:
            with (
                self.subTest(method=method_type.__name__),
                get_flags().moe.override(runner_backend=backend),
            ):
                method = object.__new__(method_type)
                layer = self._build_layer(method)
                self.assertEqual(layer.runner_backend, backend)
                self.assertTrue(layer.dispatcher.skip_local_expert_mapping)
                topk = TopK(top_k=2, layer_id=0, is_fp4_experts=fp4)
                topk.bind_runner_backend(layer.runner_backend)
                self.assertEqual(
                    topk.topk_config.output_format,
                    TopKOutputFormat.STANDARD if fp4 else TopKOutputFormat.BYPASSED,
                )
                if method_type is ModelOptFp8MoEMethod:
                    self.assertTrue(method.runner.runner_backend.is_triton())
                    self.assertTrue(layer.use_flashinfer_trtllm_moe)
                    self.assertFalse(layer.moe_runner_config.inplace)

    def test_compressed_wrapper_propagates_the_inner_direct_contract(self):
        """Wrapper methods expose their scheme's format and global-ID contract."""
        from sglang.srt.layers.quantization.compressed_tensors.compressed_tensors import (
            CompressedTensorsFusedMoEMethod,
        )
        from sglang.srt.layers.quantization.compressed_tensors.schemes.compressed_tensors_w4a4_mxint4_moe import (
            CompressedTensorsMxInt4MoE,
        )

        scheme = object.__new__(CompressedTensorsMxInt4MoE)
        method = object.__new__(CompressedTensorsFusedMoEMethod)

        def create_weights(*, layer, **kwargs):
            layer.scheme = scheme

        with patch.object(method, "create_weights", side_effect=create_weights):
            layer = FusedMoE(
                num_experts=4,
                top_k=2,
                hidden_size=4,
                intermediate_size=96,
                layer_id=0,
                quant_method=method,
            )
        self.assertEqual(layer.runner_backend, MoeRunnerBackend.FLASHINFER_TRTLLM)
        self.assertTrue(layer.use_flashinfer_trtllm_moe)
        self.assertTrue(layer.dispatcher.skip_local_expert_mapping)

    def test_gfx1250_internal_router_requires_full_expert_weights(self):
        """Internal raw-logit routing cannot address a rank-local expert shard."""
        from sglang.srt.layers.quantization.quark.schemes.quark_w4a8_mxfp4_moe import (
            QuarkW4A8MXFp4MoE,
        )

        with patch(
            "sglang.srt.layers.quantization.quark.schemes.quark_w4a8_mxfp4_moe."
            "_use_gfx1250_w4a8",
            True,
        ):
            with self.assertRaisesRegex(ValueError, "expert parallelism"):
                self._build_layer(object.__new__(QuarkW4A8MXFp4MoE))
            with get_parallel().override(
                tp_size=1,
                tp_rank=0,
                attn_tp_size=1,
                attn_tp_rank=0,
                moe_ep_size=1,
                moe_ep_rank=0,
            ):
                layer = self._build_layer(object.__new__(QuarkW4A8MXFp4MoE))
            self.assertIsNone(layer.runner_backend)
            self.assertFalse(layer.dispatcher.skip_local_expert_mapping)

    def test_topk_does_not_re_read_the_target_backend(self):
        """A constructed router keeps its layer's format across draft scopes."""
        topk = TopK(top_k=2, layer_id=0, allow_routed_experts_capture=False)
        topk.bind_runner_backend(MoeRunnerBackend.FLASHINFER_TRTLLM)
        with get_flags().moe.override(runner_backend=MoeRunnerBackend.TRITON):
            output = topk.forward_cuda(torch.ones((1, 4)), torch.ones((1, 4)))
        self.assertEqual(output.format, TopKOutputFormat.BYPASSED)

    def test_implicit_router_without_a_layer_contract_fails_closed(self):
        """A forgotten model association must fail before invoking a wrong kernel."""
        topk = TopK(top_k=2, layer_id=0)
        with self.assertRaisesRegex(RuntimeError, "owning experts"):
            topk.forward_cuda(torch.ones((1, 4)), torch.ones((1, 4)))

    def test_explicit_standard_router_preserves_its_format(self):
        """Routers consumed by model-side operations must stay materialized."""
        topk = TopK(top_k=2, layer_id=0, output_format=TopKOutputFormat.STANDARD)
        topk.bind_runner_backend(MoeRunnerBackend.FLASHINFER_TRTLLM)
        self.assertEqual(topk.topk_config.output_format, TopKOutputFormat.STANDARD)


if __name__ == "__main__":
    import pytest

    sys.exit(pytest.main([__file__, "-v"]))
