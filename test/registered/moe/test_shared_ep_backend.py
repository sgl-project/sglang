import subprocess
import sys
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch

from sglang.kernels.ops.quantization.fp8_kernel import (
    sglang_per_token_group_quant_fp8,
)
from sglang.srt.layers.moe.ep_moe.layer import DeepEPMoE, get_moe_impl_class
from sglang.srt.layers.moe.fused_moe_triton.layer import create_moe_dispatcher
from sglang.srt.layers.moe.moe_runner.base import FusedOpPool, MoeRunnerConfig
from sglang.srt.layers.moe.moe_runner.deep_gemm import DeepGemmMoeQuantInfo
from sglang.srt.layers.moe.shared_ep.backend import (
    SharedEpDispatcher,
    _run_deep_gemm_fallback,
    compact_intermediate_capacity,
    create_shared_ep_dispatcher,
    decode_intermediate_capacity,
)
from sglang.srt.layers.moe.shared_ep.kernels import (
    quantize_pack_input,
)
from sglang.srt.layers.moe.shared_ep.layout import (
    SharedEpLayout,
    align_output_layout,
)
from sglang.srt.layers.moe.shared_ep.profiles import select_profile
from sglang.srt.layers.moe.token_dispatcher.base import CombineInputFormat
from sglang.srt.layers.moe.token_dispatcher.standard import StandardCombineInput
from sglang.srt.layers.moe.topk import StandardTopKOutput
from sglang.srt.layers.moe.utils import (
    DeepEPMode,
    MoeA2ABackend,
    MoeRunnerBackend,
    is_deepep_class_backend,
)
from sglang.srt.server_args import MOE_A2A_BACKEND_CHOICES
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=20, stage="base-b", runner_config="1-gpu-large")


def _config() -> MoeRunnerConfig:
    return MoeRunnerConfig(
        hidden_size=4096,
        intermediate_size_per_partition=2048,
        top_k=6,
        num_experts=256,
        num_local_experts=32,
        params_dtype=torch.bfloat16,
    )


def _deepep_kwargs(mode) -> dict:
    return dict(
        group="ep_group",
        router_topk=6,
        permute_fusion=True,
        num_experts=256,
        num_local_experts=32,
        hidden_size=4096,
        params_dtype=torch.bfloat16,
        deepep_mode=mode,
        async_finish=True,
        return_recv_hook=True,
    )


class TestSharedEpBackend(unittest.TestCase):
    def test_runner_bootstraps_shared_fused_func_before_pool_lookup(self):
        code = """
import torch

import sglang.srt.layers.moe.moe_runner.runner as runner_module
from sglang.srt.layers.moe.moe_runner.base import FusedOpPool
from sglang.srt.layers.moe.moe_runner.base import MoeRunnerConfig
from sglang.srt.layers.moe.utils import MoeA2ABackend, MoeRunnerBackend

assert FusedOpPool.get_fused_func("shared_ep", "deep_gemm") is None
runner_module.get_moe_a2a_backend = lambda: MoeA2ABackend.SHARED_EP
runner = runner_module.MoeRunner(
    MoeRunnerBackend.DEEP_GEMM,
    MoeRunnerConfig(
        hidden_size=4096,
        intermediate_size_per_partition=2048,
        top_k=6,
        num_experts=256,
        num_local_experts=32,
        params_dtype=torch.bfloat16,
    ),
)
assert runner.fused_func is not None
"""
        subprocess.run(
            [sys.executable, "-c", code],
            check=True,
            text=True,
            capture_output=True,
        )

    def test_fused_path_cannot_be_disabled(self):
        code = """
import os

import torch

os.environ["SGLANG_CI_DISABLE_MOE_FUSED_FUNC"] = "1"

import sglang.srt.layers.moe.moe_runner.runner as runner_module
from sglang.srt.layers.moe.moe_runner.base import MoeRunnerConfig
from sglang.srt.layers.moe.utils import MoeA2ABackend, MoeRunnerBackend

runner_module.get_moe_a2a_backend = lambda: MoeA2ABackend.SHARED_EP
try:
    runner_module.MoeRunner(
        MoeRunnerBackend.DEEP_GEMM,
        MoeRunnerConfig(
            hidden_size=4096,
            intermediate_size_per_partition=2048,
            top_k=6,
            num_experts=256,
            num_local_experts=32,
            params_dtype=torch.bfloat16,
        ),
    )
except RuntimeError as exc:
    assert "SharedEP requires its registered fused execution path" in str(exc)
else:
    raise AssertionError("SharedEP accepted a disabled fused execution path")
"""
        subprocess.run(
            [sys.executable, "-c", code],
            check=True,
            text=True,
            capture_output=True,
        )

    @patch(
        "sglang.srt.layers.moe.shared_ep.backend.get_deepep_mode",
        return_value=DeepEPMode.AUTO,
    )
    @patch(
        "sglang.srt.layers.moe.shared_ep.backend.get_moe_runner_backend",
        return_value=MoeRunnerBackend.DEEP_GEMM,
    )
    @patch("sglang.srt.layers.moe.shared_ep.backend.DeepEPDispatcher")
    @patch("sglang.srt.layers.moe.shared_ep.backend.SharedEpDispatcher")
    def test_prefill_factory_builds_direct_deepep_fallback(
        self,
        shared_dispatcher,
        deep_ep_dispatcher,
        _get_runner_backend,
        _get_mode,
    ):
        fallback = object()
        result = object()
        deep_ep_dispatcher.return_value = fallback
        shared_dispatcher.return_value = result

        self.assertIs(
            create_shared_ep_dispatcher(_config(), group="ep_group"),
            result,
        )
        deep_ep_dispatcher.assert_called_once_with(**_deepep_kwargs(DeepEPMode.AUTO))
        shared_dispatcher.assert_called_once_with(
            _config(),
            fallback_dispatcher=fallback,
        )

    def test_shared_backend_identity(self):
        backend = MoeA2ABackend("shared_ep")

        self.assertEqual(backend, MoeA2ABackend.SHARED_EP)
        self.assertTrue(backend.is_shared_ep())
        self.assertIn("shared_ep", MOE_A2A_BACKEND_CHOICES)
        self.assertIsNotNone(FusedOpPool.get_fused_func("shared_ep", "deep_gemm"))
        self.assertIsNone(FusedOpPool.get_fused_func("shared_ep", "triton"))
        with self.assertRaises(ValueError):
            MoeA2ABackend("shared_moe")

    def test_output_rows_absorb_vmm_granularity_without_owner_gaps(self):
        dsv4 = SharedEpLayout.build(
            hidden_size=4096,
            top_k=6,
            max_tokens_per_rank=32,
        )
        glm = SharedEpLayout.build(
            hidden_size=6144,
            top_k=8,
            max_tokens_per_rank=32,
        )

        aligned_dsv4 = align_output_layout(dsv4, granularity=2 * 1024 * 1024)
        aligned_glm = align_output_layout(glm, granularity=2 * 1024 * 1024)

        self.assertEqual(aligned_dsv4.output_row_bytes, 32 * 1024)
        self.assertEqual(aligned_dsv4.output_rank_bytes, 6 * 1024 * 1024)
        self.assertEqual(aligned_glm.output_row_bytes, 16 * 1024)
        self.assertEqual(aligned_glm.output_rank_bytes, 4 * 1024 * 1024)

        storage = torch.zeros(
            2 * aligned_dsv4.output_rank_bytes,
            dtype=torch.uint8,
        )
        output = aligned_dsv4.output_view(
            storage,
            world_size=2,
            mapped_rank_bytes=aligned_dsv4.output_rank_bytes,
        )
        flat_output = output.view(-1, aligned_dsv4.hidden_size)
        flat_output[aligned_dsv4.output_rows_per_rank + 7, 3] = 9
        self.assertEqual(output[1].view(-1, aligned_dsv4.hidden_size)[7, 3], 9)

    def test_compact_capacity_covers_routes_and_per_expert_padding(self):
        self.assertEqual(
            compact_intermediate_capacity(
                num_tokens=4,
                world_size=8,
                top_k=6,
                num_local_experts=32,
                block_size=16,
            ),
            672,
        )
        self.assertEqual(
            compact_intermediate_capacity(
                num_tokens=8,
                world_size=8,
                top_k=6,
                num_local_experts=32,
                block_size=16,
            ),
            864,
        )

    def test_decode_capacity_is_static_under_uneven_dp_batches(self):
        dsv4 = select_profile(
            _config(),
            capability=(9, 0),
            ep_size=8,
            block_shape=(128, 128),
            max_tokens_per_rank=32,
        )
        self.assertEqual(decode_intermediate_capacity(dsv4), 2016)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is required")
    def test_quantize_pack_input_matches_reference_layout(self):
        for hidden_size, top_k in ((4096, 6), (6144, 8)):
            with self.subTest(hidden_size=hidden_size, top_k=top_k):
                layout = SharedEpLayout.build(
                    hidden_size=hidden_size,
                    top_k=top_k,
                    max_tokens_per_rank=32,
                )
                direct_storage = torch.zeros(
                    layout.input_rank_bytes,
                    dtype=torch.uint8,
                    device="cuda",
                )
                direct = layout.input_views(
                    direct_storage,
                    world_size=1,
                    mapped_rank_bytes=layout.input_rank_bytes,
                ).owner(0)
                source = torch.linspace(
                    -2,
                    2,
                    4 * hidden_size,
                    dtype=torch.bfloat16,
                    device="cuda",
                ).view(4, hidden_size)
                source_ids = (
                    torch.arange(
                        4 * top_k,
                        dtype=torch.int32,
                        device="cuda",
                    ).view(4, top_k)
                    * 13
                ) % 256
                source_weights = torch.linspace(
                    0,
                    1,
                    4 * top_k,
                    dtype=torch.float32,
                    device="cuda",
                ).view(4, top_k)
                source_q, source_scales = sglang_per_token_group_quant_fp8(
                    source,
                    128,
                )

                quantize_pack_input(
                    direct,
                    source=source,
                    source_ids=source_ids,
                    source_weights=source_weights,
                    group_size=128,
                )

                torch.testing.assert_close(
                    direct.scales[:4],
                    source_scales,
                    rtol=1e-6,
                    atol=1e-8,
                )
                self.assertTrue(torch.equal(direct.topk_ids[:4], source_ids))
                self.assertTrue(torch.equal(direct.topk_weights[:4], source_weights))
                self.assertTrue(torch.all(direct.topk_ids[4:] == -1))
                self.assertTrue(torch.all(direct.topk_weights[4:] == 0))
                direct_values = direct.activations[:4].float()
                direct_values *= direct.scales[:4].repeat_interleave(128, dim=1)
                reference_values = source_q.float()
                reference_values *= source_scales.repeat_interleave(128, dim=1)
                torch.testing.assert_close(
                    direct_values,
                    reference_values,
                    rtol=0.125,
                    atol=0.25,
                )

    @patch(
        "sglang.srt.layers.moe.fused_moe_triton.layer.MaybeTboDeepEPDispatcher",
        side_effect=AssertionError("SharedEP must not construct the TBO wrapper"),
    )
    @patch(
        "sglang.srt.layers.moe.fused_moe_triton.layer._get_deepep_comm_group",
        return_value="ep_group",
    )
    @patch(
        "sglang.srt.layers.moe.fused_moe_triton.layer.get_moe_a2a_backend",
        return_value=MoeA2ABackend.SHARED_EP,
    )
    @patch("sglang.srt.layers.moe.shared_ep.create_shared_ep_dispatcher")
    def test_shared_backend_delegates_without_tbo_wrapper(
        self,
        shared_factory,
        _get_a2a_backend,
        _get_group,
        _tbo_dispatcher,
    ):
        result = object()
        shared_factory.return_value = result

        self.assertIs(create_moe_dispatcher(_config()), result)
        shared_factory.assert_called_once_with(
            _config(),
            group="ep_group",
        )

    @patch("sglang.srt.layers.moe.fused_moe_triton.layer.StandardDispatcher")
    @patch(
        "sglang.srt.layers.moe.fused_moe_triton.layer.get_moe_a2a_backend",
        return_value=MoeA2ABackend.NONE,
    )
    def test_direct_selection_preserves_standard_dispatcher(
        self,
        _get_a2a_backend,
        standard_dispatcher,
    ):
        expected = object()
        standard_dispatcher.return_value = expected

        self.assertIs(create_moe_dispatcher(_config()), expected)
        standard_dispatcher.assert_called_once_with(_config())

    @patch("sglang.srt.layers.moe.fused_moe_triton.layer.get_deepep_mode")
    @patch("sglang.srt.layers.moe.fused_moe_triton.layer.MaybeTboDeepEPDispatcher")
    @patch(
        "sglang.srt.layers.moe.fused_moe_triton.layer._get_deepep_comm_group",
        return_value="ep_group",
    )
    @patch(
        "sglang.srt.layers.moe.fused_moe_triton.layer.get_moe_a2a_backend",
        return_value=MoeA2ABackend.DEEPEP,
    )
    def test_direct_selection_preserves_deepep_class_dispatchers(
        self,
        get_a2a_backend,
        _get_group,
        deepep_dispatcher,
        get_deepep_mode,
    ):
        expected = object()
        mode = object()
        deepep_dispatcher.return_value = expected
        get_deepep_mode.return_value = mode

        for backend in (MoeA2ABackend.DEEPEP, MoeA2ABackend.MOONCAKE):
            with self.subTest(backend=backend.value):
                get_a2a_backend.return_value = backend
                self.assertIs(create_moe_dispatcher(_config()), expected)
                deepep_dispatcher.assert_called_once_with(**_deepep_kwargs(mode))
                deepep_dispatcher.reset_mock()

    @patch(
        "sglang.srt.layers.moe.shared_ep.backend.get_parallel",
        return_value=SimpleNamespace(moe_ep_size=8, moe_ep_rank=0),
    )
    @patch("sglang.srt.layers.moe.shared_ep.backend._get_shared_state")
    def test_dispatcher_initializes_vmm_before_forward(
        self,
        get_shared_state,
        _get_parallel,
    ):
        state = object()
        get_shared_state.return_value = state

        dispatcher = SharedEpDispatcher(_config())

        self.assertIs(dispatcher.state, state)
        get_shared_state.assert_called_once_with(dispatcher.config, dispatcher.profile)

    @patch(
        "sglang.srt.layers.moe.shared_ep.backend.get_is_extend_in_batch",
        return_value=True,
    )
    def test_prefill_delegates_dispatch_and_combine(self, _is_extend):
        fallback = Mock()
        dispatched = object()
        combined = torch.ones((33, 4))
        fallback.dispatch.return_value = dispatched
        fallback.combine.return_value = combined
        dispatcher = SharedEpDispatcher.__new__(SharedEpDispatcher)
        dispatcher.fallback_dispatcher = fallback
        hidden_states = torch.zeros((33, 4))
        topk_output = StandardTopKOutput(
            topk_weights=torch.ones((33, 1)),
            topk_ids=torch.zeros((33, 1), dtype=torch.int32),
            router_logits=None,
        )

        self.assertIs(dispatcher.dispatch(hidden_states, topk_output), dispatched)
        fallback.dispatch.assert_called_once_with(
            hidden_states=hidden_states,
            topk_output=topk_output,
        )
        fallback_input = SimpleNamespace(format=CombineInputFormat.DEEPEP_LL)
        self.assertIs(dispatcher.combine(fallback_input), combined)

    @patch(
        "sglang.srt.layers.moe.shared_ep.backend.get_is_extend_in_batch",
        return_value=False,
    )
    @patch("sglang.srt.layers.moe.shared_ep.backend.quantize_pack_input")
    def test_decode_rows_use_direct_path(
        self,
        quantize_pack,
        _is_extend,
    ):
        fallback = Mock()
        profile = SimpleNamespace(
            max_tokens_per_rank=32,
            block_shape=(128, 128),
        )
        state = SimpleNamespace(
            local_input=object(),
            global_input=SimpleNamespace(
                activations=object(),
                scales=object(),
            ),
            input_epoch=Mock(),
        )
        dispatcher = SharedEpDispatcher.__new__(SharedEpDispatcher)
        dispatcher.profile = profile
        dispatcher.state = state
        dispatcher.local_expert_start = 0
        dispatcher.fallback_dispatcher = fallback
        hidden_states = torch.zeros((16, 4))
        topk_output = StandardTopKOutput(
            topk_weights=torch.ones((16, 1)),
            topk_ids=torch.zeros((16, 1), dtype=torch.int32),
            router_logits=None,
        )

        output = dispatcher.dispatch(hidden_states, topk_output)

        self.assertEqual(output.num_tokens, 16)
        self.assertIs(output.state, state)
        quantize_pack.assert_called_once_with(
            state.local_input,
            source=hidden_states,
            source_ids=topk_output.topk_ids,
            source_weights=topk_output.topk_weights,
            group_size=128,
        )
        state.input_epoch.publish.assert_called_once_with()
        state.input_epoch.wait_all.assert_not_called()
        fallback.dispatch.assert_not_called()

    def test_decode_combine_does_not_depend_on_mutable_dispatch_state(self):
        fallback = Mock()
        dispatcher = SharedEpDispatcher.__new__(SharedEpDispatcher)
        dispatcher.fallback_dispatcher = fallback
        hidden_states = torch.ones((32, 4))

        self.assertIs(
            dispatcher.combine(StandardCombineInput(hidden_states=hidden_states)),
            hidden_states,
        )
        fallback.combine.assert_not_called()

    def test_backend_reuses_ep_sharded_model_contract(self):
        with patch(
            "sglang.srt.layers.moe.utils.get_moe_a2a_backend",
            return_value=MoeA2ABackend.SHARED_EP,
        ):
            self.assertTrue(is_deepep_class_backend())
        with patch(
            "sglang.srt.layers.moe.ep_moe.layer.get_moe_a2a_backend",
            return_value=MoeA2ABackend.SHARED_EP,
        ):
            self.assertIs(get_moe_impl_class(quant_config=None), DeepEPMoE)

    @patch("sglang.srt.layers.moe.shared_ep.backend.DeepGemmRunnerCore")
    @patch("sglang.srt.layers.moe.shared_ep.backend.PermuteMethodPool.get_post_permute")
    @patch("sglang.srt.layers.moe.shared_ep.backend.PermuteMethodPool.get_pre_permute")
    def test_prefill_runs_existing_deep_gemm_pipeline(
        self,
        get_pre_permute,
        get_post_permute,
        deep_gemm_runner_core,
    ):
        pre_permute = Mock(return_value="runner_input")
        post_permute = Mock(return_value="combined")
        runner = deep_gemm_runner_core.return_value
        runner.run.return_value = "runner_output"
        get_pre_permute.return_value = pre_permute
        get_post_permute.return_value = post_permute
        dispatch_output = SimpleNamespace(format=SimpleNamespace(value="deepep_normal"))
        quant_info = DeepGemmMoeQuantInfo(
            w13_weight=torch.empty((1, 2, 3)),
            w2_weight=torch.empty((1, 3, 1)),
            use_fp8=True,
        )
        config = MoeRunnerConfig()

        result = _run_deep_gemm_fallback(
            dispatch_output,
            quant_info,
            config,
        )

        self.assertEqual(result, "combined")
        pre_permute.assert_called_once_with(
            dispatch_output,
            quant_info,
            config,
            {},
        )
        runner.run.assert_called_once_with("runner_input", quant_info, {})
        post_permute.assert_called_once_with(
            "runner_output",
            quant_info,
            config,
            {},
        )


if __name__ == "__main__":
    unittest.main()
