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
from sglang.srt.layers.moe.moe_runner.triton import TritonMoeQuantInfo
from sglang.srt.layers.moe.shared_ep.backend import (
    SharedEpDispatcher,
    SharedEpDispatchOutput,
    compact_intermediate_capacity,
    create_shared_ep_dispatcher,
    intermediate_capacity,
    run_shared_ep,
)
from sglang.srt.layers.moe.shared_ep.kernels import (
    quantize_pack_input,
)
from sglang.srt.layers.moe.shared_ep.layout import (
    SharedEpLayout,
    align_output_layout,
)
from sglang.srt.layers.moe.shared_ep.profiles import (
    GLM52,
    make_pull_cache_prefill_profile,
    select_profile,
)
from sglang.srt.layers.moe.token_dispatcher.standard import StandardCombineInput
from sglang.srt.layers.moe.topk import StandardTopKOutput
from sglang.srt.layers.moe.utils import (
    MoeA2ABackend,
    MoeRunnerBackend,
    is_deepep_class_backend,
)
from sglang.srt.layers.quantization.compressed_tensors.schemes.compressed_tensors_w8a8_fp8_moe import (
    CompressedTensorsW8A8Fp8MoE,
)
from sglang.srt.layers.quantization.fp8 import Fp8MoEMethod
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


def _glm_config() -> MoeRunnerConfig:
    return MoeRunnerConfig(
        hidden_size=6144,
        intermediate_size_per_partition=2048,
        top_k=8,
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

assert FusedOpPool.get_fused_func("shared_ep", "triton") is None
runner_module.get_moe_a2a_backend = lambda: MoeA2ABackend.SHARED_EP
runner = runner_module.MoeRunner(
    MoeRunnerBackend.TRITON,
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
        MoeRunnerBackend.TRITON,
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

    @patch("sglang.srt.layers.moe.shared_ep.backend.SharedEpDispatcher")
    def test_factory_builds_standalone_shared_dispatcher(
        self,
        shared_dispatcher,
    ):
        result = object()
        shared_dispatcher.return_value = result

        self.assertIs(
            create_shared_ep_dispatcher(_config()),
            result,
        )
        shared_dispatcher.assert_called_once_with(_config())

    def test_shared_backend_identity(self):
        backend = MoeA2ABackend("shared_ep")

        self.assertEqual(backend, MoeA2ABackend.SHARED_EP)
        self.assertTrue(backend.is_shared_ep())
        self.assertIn("shared_ep", MOE_A2A_BACKEND_CHOICES)
        self.assertIsNone(FusedOpPool.get_fused_func("shared_ep", "deep_gemm"))
        self.assertIsNotNone(FusedOpPool.get_fused_func("shared_ep", "triton"))
        with self.assertRaises(ValueError):
            MoeA2ABackend("shared_moe")

    @patch(
        "sglang.srt.layers.quantization.fp8.get_moe_a2a_backend",
        return_value=MoeA2ABackend.SHARED_EP,
    )
    @patch(
        "sglang.srt.layers.quantization.fp8.get_moe_runner_backend",
        return_value=MoeRunnerBackend.AUTO,
    )
    @patch("sglang.srt.layers.quantization.fp8.MoeRunner")
    def test_fp8_auto_runner_resolves_to_triton_for_shared_ep(
        self,
        runner,
        _runner_backend,
        _a2a_backend,
    ):
        method = object.__new__(Fp8MoEMethod)

        method.create_moe_runner(object(), _config())

        runner.assert_called_once_with(MoeRunnerBackend.TRITON, _config())

    @patch(
        "sglang.srt.layers.quantization.compressed_tensors.schemes.compressed_tensors_w8a8_fp8_moe.get_moe_runner_backend",
        return_value=MoeRunnerBackend.AUTO,
    )
    @patch(
        "sglang.srt.layers.quantization.compressed_tensors.schemes.compressed_tensors_w8a8_fp8_moe.MoeRunner"
    )
    def test_compressed_fp8_auto_runner_resolves_to_triton_for_shared_ep(
        self,
        runner,
        _runner_backend,
    ):
        method = object.__new__(CompressedTensorsW8A8Fp8MoE)

        method.create_moe_runner(object(), _config())

        runner.assert_called_once_with(MoeRunnerBackend.TRITON, _config())

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
        self.assertEqual(intermediate_capacity(dsv4), 2016)

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
        "sglang.srt.layers.moe.fused_moe_triton.layer.get_moe_a2a_backend",
        return_value=MoeA2ABackend.SHARED_EP,
    )
    @patch("sglang.srt.layers.moe.shared_ep.create_shared_ep_dispatcher")
    def test_shared_backend_delegates_without_tbo_wrapper(
        self,
        shared_factory,
        _get_a2a_backend,
        _tbo_dispatcher,
    ):
        result = object()
        shared_factory.return_value = result

        self.assertIs(create_moe_dispatcher(_config()), result)
        shared_factory.assert_called_once_with(_config())

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
    @patch("sglang.srt.layers.moe.shared_ep.backend._get_shared_pull_cache")
    @patch("sglang.srt.layers.moe.shared_ep.backend._get_shared_state")
    @patch(
        "sglang.srt.layers.moe.shared_ep.backend.get_schedule",
        return_value=SimpleNamespace(chunked_prefill_size=64),
    )
    def test_dsv4_dispatcher_initializes_decode_and_prefill_vmm(
        self,
        _get_schedule,
        get_shared_state,
        get_shared_pull_cache,
        _get_parallel,
    ):
        decode_state = object()
        prefill_state = object()
        pull_cache = object()
        get_shared_state.side_effect = (decode_state, prefill_state)
        get_shared_pull_cache.return_value = pull_cache

        dispatcher = SharedEpDispatcher(_config())

        self.assertIs(dispatcher.state, decode_state)
        self.assertIs(dispatcher.prefill_state, prefill_state)
        self.assertIs(dispatcher.prefill_cache, pull_cache)
        self.assertEqual(dispatcher.prefill_profile.max_tokens_per_rank, 64)
        self.assertEqual(dispatcher.prefill_profile.hidden_size, 4096)
        self.assertEqual(dispatcher.prefill_profile.top_k, 6)
        self.assertEqual(get_shared_state.call_count, 2)
        get_shared_pull_cache.assert_called_once_with(dispatcher.prefill_profile)

    @patch(
        "sglang.srt.layers.moe.shared_ep.backend.get_schedule",
        return_value=SimpleNamespace(chunked_prefill_size=64),
    )
    @patch(
        "sglang.srt.layers.moe.shared_ep.backend.get_parallel",
        return_value=SimpleNamespace(moe_ep_size=8, moe_ep_rank=0),
    )
    @patch("sglang.srt.layers.moe.shared_ep.backend._get_shared_pull_cache")
    @patch("sglang.srt.layers.moe.shared_ep.backend._get_shared_state")
    def test_glm_dispatcher_initializes_decode_and_prefill_vmm(
        self,
        get_shared_state,
        get_shared_pull_cache,
        _get_parallel,
        _get_schedule,
    ):
        decode_state = object()
        prefill_state = object()
        pull_cache = object()
        get_shared_state.side_effect = (decode_state, prefill_state)
        get_shared_pull_cache.return_value = pull_cache

        dispatcher = SharedEpDispatcher(_glm_config())

        self.assertIs(dispatcher.state, decode_state)
        self.assertIs(dispatcher.prefill_state, prefill_state)
        self.assertIs(dispatcher.prefill_cache, pull_cache)
        self.assertEqual(dispatcher.prefill_profile.max_tokens_per_rank, 64)
        self.assertEqual(dispatcher.prefill_profile.block_size_m, 64)
        self.assertEqual(
            dispatcher.prefill_profile.w13_kernel_config(64)["BLOCK_SIZE_M"],
            64,
        )
        self.assertEqual(
            dispatcher.prefill_profile.w2_kernel_config(64)["BLOCK_SIZE_M"],
            64,
        )
        self.assertEqual(get_shared_state.call_count, 2)
        get_shared_pull_cache.assert_called_once_with(dispatcher.prefill_profile)

    def test_prefill_route_w13_and_w2_share_one_m_block(self):
        profile = make_pull_cache_prefill_profile(GLM52, 1024)

        self.assertIsNotNone(profile)
        self.assertEqual(
            {
                profile.block_size_m,
                profile.w13_kernel_config(1024)["BLOCK_SIZE_M"],
                profile.w2_kernel_config(1024)["BLOCK_SIZE_M"],
            },
            {64},
        )

    @patch("sglang.srt.layers.moe.shared_ep.backend._validate_weights")
    @patch(
        "sglang.srt.layers.moe.shared_ep.backend.intermediate_capacity",
        return_value=64,
    )
    @patch("sglang.srt.layers.moe.shared_ep.backend.prepare_routes")
    @patch("sglang.srt.layers.moe.shared_ep.backend.silu_and_mul_contig_post_quant")
    @patch("sglang.srt.layers.moe.shared_ep.backend.pull_cache_rows")
    @patch("sglang.srt.layers.moe.shared_ep.backend.invoke_pull_cache_w13")
    @patch("sglang.srt.layers.moe.shared_ep.backend.invoke_fused_moe_kernel")
    def test_prefill_pulls_w13_rows_while_decode_consumes_directly(
        self,
        fused_moe,
        pull_w13,
        pull_rows,
        _silu_and_mul,
        prepare_routes,
        _capacity,
        _validate_weights,
    ):
        routes = SimpleNamespace(
            local_ids=torch.zeros((8, 8), dtype=torch.int32),
            local_weights=torch.ones((8, 8), dtype=torch.float32),
            sorted_token_ids=torch.zeros(64, dtype=torch.int32),
            expert_ids=torch.zeros(1, dtype=torch.int32),
            num_tokens_post_padded=torch.ones(1, dtype=torch.int32),
        )
        prepare_routes.return_value = routes
        input_epoch = Mock()
        input_epoch.allocation.local_storage = torch.zeros(1, dtype=torch.uint8)
        input_epoch.epoch = torch.ones(1, dtype=torch.int32)
        state = SimpleNamespace(
            global_input=SimpleNamespace(
                topk_ids=torch.zeros((8, 1, 8), dtype=torch.int32),
                topk_weights=torch.ones((8, 1, 8), dtype=torch.float32),
            ),
            input_epoch=input_epoch,
            global_output=torch.empty((64, 6144), dtype=torch.bfloat16),
            local_output=torch.zeros((8, 8, 6144), dtype=torch.bfloat16),
            output_epoch=Mock(),
        )
        quant_info = SimpleNamespace(
            w13_weight=torch.empty((1, 1, 1), dtype=torch.float8_e4m3fn),
            w13_scale=torch.ones((1, 1, 1), dtype=torch.float32),
            w2_weight=torch.empty((1, 1, 1), dtype=torch.float8_e4m3fn),
            w2_scale=torch.ones((1, 1, 1), dtype=torch.float32),
        )

        prefill_profile = make_pull_cache_prefill_profile(GLM52, 1024)
        self.assertIsNotNone(prefill_profile)
        pull_cache = SimpleNamespace(active_rows=64)
        prefill = SharedEpDispatchOutput(
            hidden_states=torch.empty((8, 6144), dtype=torch.float8_e4m3fn),
            hidden_states_scale=torch.empty((8, 48), dtype=torch.float32),
            topk_output=object(),
            state=state,
            profile=prefill_profile,
            num_tokens=8,
            local_expert_start=0,
            phase="prefill",
            pull_cache=pull_cache,
        )

        run_shared_ep(prefill, quant_info, SimpleNamespace(swiglu_limit=None))

        pull_rows.assert_called_once()
        pull_w13.assert_called_once()
        self.assertEqual(fused_moe.call_count, 1)

        pull_rows.reset_mock()
        pull_w13.reset_mock()
        fused_moe.reset_mock()
        decode = SharedEpDispatchOutput(
            hidden_states=torch.empty((8, 6144), dtype=torch.float8_e4m3fn),
            hidden_states_scale=torch.empty((8, 48), dtype=torch.float32),
            topk_output=object(),
            state=state,
            profile=GLM52,
            num_tokens=8,
            local_expert_start=0,
            phase="decode",
            pull_cache=None,
        )

        run_shared_ep(decode, quant_info, SimpleNamespace(swiglu_limit=None))

        pull_rows.assert_not_called()
        pull_w13.assert_not_called()
        self.assertEqual(fused_moe.call_count, 2)

    @patch(
        "sglang.srt.layers.moe.shared_ep.backend.get_is_extend_in_batch",
        return_value=True,
    )
    @patch(
        "sglang.srt.layers.moe.shared_ep.backend.get_dp_global_num_tokens",
        return_value=[16, 33, 8, 8, 8, 8, 8, 8],
    )
    @patch("sglang.srt.layers.moe.shared_ep.backend.quantize_pack_input")
    def test_prefill_rejects_peer_capacity_overflow_before_publish(
        self,
        quantize_pack,
        _global_num_tokens,
        _is_extend,
    ):
        dispatcher = SharedEpDispatcher.__new__(SharedEpDispatcher)
        profile = SimpleNamespace(
            max_tokens_per_rank=32,
            ep_size=8,
            supports_pull_cache_prefill=True,
        )
        dispatcher.profile = profile
        dispatcher.prefill_profile = profile
        state = SimpleNamespace(
            local_input=object(),
            input_epoch=Mock(),
        )
        dispatcher.prefill_state = state
        dispatcher.prefill_cache = object()
        hidden_states = torch.zeros((16, 4))
        topk_output = StandardTopKOutput(
            topk_weights=torch.ones((16, 1)),
            topk_ids=torch.zeros((16, 1), dtype=torch.int32),
            router_logits=None,
        )

        with self.assertRaisesRegex(
            ValueError,
            "SharedEP prefill capacity exceeded.*33 > 32",
        ):
            dispatcher.dispatch(hidden_states, topk_output)

        quantize_pack.assert_not_called()
        state.input_epoch.publish.assert_not_called()

    @patch(
        "sglang.srt.layers.moe.shared_ep.backend.get_is_extend_in_batch",
        return_value=False,
    )
    @patch(
        "sglang.srt.layers.moe.shared_ep.backend.get_dp_global_num_tokens",
        return_value=[16, 33, 8, 8, 8, 8, 8, 8],
    )
    @patch("sglang.srt.layers.moe.shared_ep.backend.quantize_pack_input")
    def test_decode_rejects_peer_capacity_overflow_before_publish(
        self,
        quantize_pack,
        _global_num_tokens,
        _is_extend,
    ):
        profile = SimpleNamespace(
            max_tokens_per_rank=32,
            ep_size=8,
            block_shape=(128, 128),
            supports_pull_cache_prefill=False,
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
        dispatcher.prefill_profile = profile
        dispatcher.state = state
        dispatcher.local_expert_start = 0
        hidden_states = torch.zeros((16, 4))
        topk_output = StandardTopKOutput(
            topk_weights=torch.ones((16, 1)),
            topk_ids=torch.zeros((16, 1), dtype=torch.int32),
            router_logits=None,
        )

        with self.assertRaisesRegex(
            ValueError,
            "rank 1 has 33 local tokens.*capacity is 32",
        ):
            dispatcher.dispatch(hidden_states, topk_output)

        quantize_pack.assert_not_called()
        state.input_epoch.publish.assert_not_called()

    @patch(
        "sglang.srt.layers.moe.shared_ep.backend.get_is_extend_in_batch",
        return_value=False,
    )
    @patch(
        "sglang.srt.layers.moe.shared_ep.backend.get_dp_global_num_tokens",
        return_value=[16] * 8,
    )
    @patch("sglang.srt.layers.moe.shared_ep.backend.quantize_pack_input")
    def test_decode_rows_use_direct_path(
        self,
        quantize_pack,
        _global_num_tokens,
        _is_extend,
    ):
        profile = SimpleNamespace(
            max_tokens_per_rank=32,
            ep_size=8,
            block_shape=(128, 128),
            supports_pull_cache_prefill=False,
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
        dispatcher.prefill_profile = profile
        dispatcher.state = state
        dispatcher.local_expert_start = 0
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

    def test_decode_combine_does_not_depend_on_mutable_dispatch_state(self):
        dispatcher = SharedEpDispatcher.__new__(SharedEpDispatcher)
        hidden_states = torch.ones((32, 4))

        self.assertIs(
            dispatcher.combine(StandardCombineInput(hidden_states=hidden_states)),
            hidden_states,
        )

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

    def test_run_rejects_non_shared_dispatch_output_without_fallback(self):
        quant_info = TritonMoeQuantInfo(
            w13_weight=torch.empty((1, 2, 3)),
            w2_weight=torch.empty((1, 3, 1)),
            use_fp8_w8a8=True,
        )
        with self.assertRaisesRegex(TypeError, "SharedEpDispatchOutput"):
            run_shared_ep(object(), quant_info, MoeRunnerConfig())


if __name__ == "__main__":
    unittest.main()
