import subprocess
import sys
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, call, patch

import torch

from sglang.kernels.ops.quantization.fp8_kernel import (
    per_token_group_quant_fp8,
)
from sglang.srt.layers.moe.ep_moe.layer import DeepEPMoE, get_moe_impl_class
from sglang.srt.layers.moe.fused_moe_triton.layer import (
    _get_deepep_comm_group,
    create_moe_dispatcher,
)
from sglang.srt.layers.moe.moe_runner.base import (
    FusedOpPool,
    MoeRunnerConfig,
    PermuteMethodPool,
)
from sglang.srt.layers.moe.moe_runner.runner import MoeRunner
from sglang.srt.layers.moe.moe_runner.shared_ep import (
    SharedEpQuantCapability,
    SharedEpQuantInfo,
    SharedEpQuantization,
    SharedEpWeightLayout,
)
from sglang.srt.layers.moe.shared_ep.backend import (
    SharedEpDispatcher,
    SharedEpDispatchOutput,
    SharedEpLaneDispatcher,
    _create_shared_ep_prefill_dispatcher,
    _get_device_profile_capability,
    _resolve_pull_prefill_profile,
    _synchronize_admission_stage,
    _validate_decode_capacity,
    compact_intermediate_capacity,
    create_shared_ep_dispatcher,
    decode_intermediate_capacity,
    run_shared_ep,
)
from sglang.srt.layers.moe.shared_ep.kernels import (
    quantize_pack_input,
)
from sglang.srt.layers.moe.shared_ep.lanes import SharedEpLaneProtocol
from sglang.srt.layers.moe.shared_ep.layout import (
    SharedEpLayout,
    align_output_layout,
)
from sglang.srt.layers.moe.shared_ep.profiles import (
    make_pull_cache_prefill_profile,
    select_profile,
)
from sglang.srt.layers.moe.shared_ep.runtime import (
    SharedEpRuntimeCapability,
    SharedEpRuntimeHooks,
)
from sglang.srt.layers.moe.token_dispatcher.base import CombineInputFormat
from sglang.srt.layers.moe.token_dispatcher.standard import StandardCombineInput
from sglang.srt.layers.moe.topk import StandardTopKOutput
from sglang.srt.layers.moe.utils import (
    DeepEPMode,
    MoeA2ABackend,
    MoeRunnerBackend,
    is_deepep_class_backend,
)
from sglang.srt.layers.quantization.fp8 import Fp8MoEMethod
from sglang.srt.model_executor.cuda_graph_config import Backend
from sglang.srt.server_args import MOE_A2A_BACKEND_CHOICES
from sglang.test.ci.ci_register import register_amd_ci, register_cuda_ci

register_cuda_ci(est_time=20, stage="base-b", runner_config="1-gpu-large")
register_amd_ci(est_time=20, stage="stage-b", runner_config="1-gpu-small-amd")


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
import sys

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
assert "sglang.srt.layers.moe.shared_ep.state" not in sys.modules
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

    def test_composite_factory_attaches_platform_fallback(self):
        fallbacks = [object(), object()]
        inners = [Mock(), Mock()]
        for lane_id, inner in enumerate(inners):
            inner.lane_id = lane_id
        admission = (object(), None, object(), object())
        parallel = object()
        with (
            patch(
                "sglang.srt.layers.moe.shared_ep.backend.get_shared_ep_prefill_backend",
                return_value=MoeRunnerBackend.DEEP_GEMM,
            ),
            patch(
                "sglang.srt.layers.moe.shared_ep.backend.get_moe_runner_backend",
                return_value=MoeRunnerBackend.DEEP_GEMM,
            ),
            patch(
                "sglang.srt.layers.moe.shared_ep.backend.compute_shared_ep_lane_protocol",
                return_value=SharedEpLaneProtocol(2, 1),
            ),
            patch(
                "sglang.srt.layers.moe.shared_ep.backend._admit_shared_ep_framework",
                return_value=admission,
            ),
            patch(
                "sglang.srt.layers.moe.shared_ep.backend.get_server_args",
                return_value=object(),
            ),
            patch(
                "sglang.srt.layers.moe.shared_ep.backend.get_parallel",
                return_value=parallel,
            ),
            patch(
                "sglang.srt.layers.moe.shared_ep.backend.SharedEpDispatcher",
                side_effect=inners,
            ) as shared_dispatcher,
            patch(
                "sglang.srt.layers.moe.shared_ep.backend._create_shared_ep_prefill_dispatcher",
                side_effect=fallbacks,
            ) as create_fallback,
        ):
            result = create_shared_ep_dispatcher(_config(), group="ep_group")

        self.assertIsInstance(result, SharedEpLaneDispatcher)
        self.assertEqual(result.inner_dispatchers, tuple(inners))
        self.assertEqual(
            shared_dispatcher.call_args_list,
            [
                call(
                    _config(),
                    model_namespace="target",
                    lane_id=lane_id,
                    admission=admission,
                )
                for lane_id in range(2)
            ],
        )
        self.assertEqual(
            create_fallback.call_args_list,
            [
                call(
                    _config(),
                    group="ep_group",
                    instance_id=lane_id,
                )
                for lane_id in range(2)
            ],
        )
        for inner, fallback in zip(inners, fallbacks):
            inner.set_fallback_dispatcher.assert_called_once_with(fallback)

    @patch("sglang.srt.layers.moe.token_dispatcher.moriep.MoriEPDispatcher")
    @patch(
        "sglang.srt.layers.moe.shared_ep.backend.get_shared_ep_prefill_backend",
        return_value=MoeRunnerBackend.AITER,
    )
    def test_rocm_prefill_uses_mori_dispatcher(
        self,
        _get_prefill_backend,
        mori_dispatcher,
    ):
        expected = object()
        mori_dispatcher.return_value = expected

        self.assertIs(
            _create_shared_ep_prefill_dispatcher(_config(), group="ep_group"),
            expected,
        )
        mori_dispatcher.assert_called_once_with(
            **_deepep_kwargs(DeepEPMode.NORMAL),
            instance_id=0,
        )

    @patch("sglang.srt.layers.moe.token_dispatcher.deepep.DeepEPDispatcher")
    @patch(
        "sglang.srt.layers.moe.shared_ep.backend.get_shared_ep_prefill_backend",
        return_value=MoeRunnerBackend.DEEP_GEMM,
    )
    def test_cuda_prefill_preserves_deepep_dispatcher(
        self,
        _get_prefill_backend,
        deepep_dispatcher,
    ):
        _create_shared_ep_prefill_dispatcher(_config(), group="ep_group")

        deepep_dispatcher.assert_called_once_with(**_deepep_kwargs(DeepEPMode.NORMAL))

    def test_shared_backend_identity(self):
        backend = MoeA2ABackend("shared_ep")

        self.assertEqual(backend, MoeA2ABackend.SHARED_EP)
        self.assertTrue(backend.is_shared_ep())
        self.assertIn("shared_ep", MOE_A2A_BACKEND_CHOICES)
        self.assertIsNotNone(FusedOpPool.get_fused_func("shared_ep", "deep_gemm"))
        self.assertIsNotNone(FusedOpPool.get_fused_func("shared_ep", "aiter"))
        self.assertIsNone(FusedOpPool.get_fused_func("shared_ep", "triton"))
        with self.assertRaises(ValueError):
            MoeA2ABackend("shared_moe")

    @patch("sglang.srt.layers.moe.shared_ep.backend.is_hip", return_value=True)
    @patch("sglang.srt.layers.moe.shared_ep.backend.torch.cuda.get_device_properties")
    @patch(
        "sglang.srt.layers.moe.shared_ep.backend.torch.cuda.current_device",
        return_value=0,
    )
    def test_rocm_device_admission_requires_exact_gfx950(
        self,
        _current_device,
        get_properties,
        _is_hip,
    ):
        get_properties.return_value = SimpleNamespace(
            gcnArchName="gfx950:sramecc+:xnack-"
        )
        self.assertEqual(
            _get_device_profile_capability(),
            ((9, 5), "rocm:gfx950"),
        )

        get_properties.return_value = SimpleNamespace(gcnArchName="gfx942")
        with self.assertRaisesRegex(ValueError, "requires gfx950"):
            _get_device_profile_capability()

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
            platform="cuda",
        )
        self.assertEqual(decode_intermediate_capacity(dsv4), 2016)

    @patch(
        "sglang.srt.layers.moe.shared_ep.backend.get_dp_global_num_tokens",
        return_value=None,
    )
    @patch(
        "sglang.srt.layers.moe.shared_ep.backend.get_forward",
        return_value=SimpleNamespace(
            shared_ep_global_num_tokens=(4, 3, 2, 1, 1, 1, 1, 1)
        ),
    )
    def test_decode_capacity_uses_forward_batch_counts(
        self,
        _get_forward,
        _get_dp_tokens,
    ):
        _validate_decode_capacity(SimpleNamespace(ep_size=8, max_tokens_per_rank=4))

    @patch(
        "sglang.srt.layers.moe.shared_ep.backend.get_dp_global_num_tokens",
        return_value=None,
    )
    @patch(
        "sglang.srt.layers.moe.shared_ep.backend.get_forward",
        return_value=SimpleNamespace(shared_ep_global_num_tokens=None),
    )
    def test_decode_capacity_fails_closed_without_global_counts(
        self,
        _get_forward,
        _get_dp_tokens,
    ):
        with self.assertRaisesRegex(RuntimeError, "complete DP token-count vector"):
            _validate_decode_capacity(
                SimpleNamespace(ep_size=8, max_tokens_per_rank=32)
            )

    def test_pull_prefill_profile_requires_single_lane_eager_rocm(self):
        decode = select_profile(
            _config(),
            capability=(9, 5),
            ep_size=8,
            block_shape=(128, 128),
            max_tokens_per_rank=32,
            platform="rocm",
        )
        eager_args = SimpleNamespace(
            cuda_graph_config=SimpleNamespace(
                prefill=SimpleNamespace(backend=Backend.DISABLED)
            )
        )
        with (
            patch(
                "sglang.srt.layers.moe.shared_ep.backend.get_server_args",
                return_value=eager_args,
            ),
            patch(
                "sglang.srt.layers.moe.shared_ep.backend.get_schedule",
                return_value=SimpleNamespace(chunked_prefill_size=1024),
            ),
        ):
            prefill = _resolve_pull_prefill_profile(
                decode,
                SharedEpLaneProtocol(1, 1),
            )
            self.assertIsNotNone(prefill)
            self.assertEqual(prefill.max_tokens_per_rank, 1024)
            self.assertEqual(prefill.block_size_m, 64)
            self.assertIsNone(
                _resolve_pull_prefill_profile(
                    decode,
                    SharedEpLaneProtocol(2, 1),
                )
            )

        graph_args = SimpleNamespace(
            cuda_graph_config=SimpleNamespace(
                prefill=SimpleNamespace(backend=Backend.FULL)
            )
        )
        with (
            patch(
                "sglang.srt.layers.moe.shared_ep.backend.get_server_args",
                return_value=graph_args,
            ),
            patch(
                "sglang.srt.layers.moe.shared_ep.backend.get_schedule",
                return_value=SimpleNamespace(chunked_prefill_size=1024),
            ),
        ):
            self.assertIsNone(
                _resolve_pull_prefill_profile(
                    decode,
                    SharedEpLaneProtocol(1, 1),
                )
            )

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
                source_q, source_scales = per_token_group_quant_fp8(
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

    @patch("sglang.srt.layers.moe.fused_moe_triton.layer.get_tp_group")
    def test_rocm_shared_ep_passes_mori_process_group(self, get_tp_group):
        group = SimpleNamespace(device_group=object())
        get_tp_group.return_value = group

        with patch(
            "sglang.srt.layers.moe.fused_moe_triton.layer._is_hip",
            True,
        ):
            self.assertIs(
                _get_deepep_comm_group(MoeA2ABackend.SHARED_EP),
                group,
            )
        with patch(
            "sglang.srt.layers.moe.fused_moe_triton.layer._is_hip",
            False,
        ):
            self.assertIs(
                _get_deepep_comm_group(MoeA2ABackend.SHARED_EP),
                group.device_group,
            )

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
    @patch("sglang.srt.layers.moe.shared_ep.backend._admit_shared_ep_framework")
    @patch("sglang.srt.layers.moe.shared_ep.backend._get_shared_state")
    def test_dispatcher_initializes_vmm_before_forward(
        self,
        get_shared_state,
        admit_framework,
        _get_parallel,
    ):
        state = object()
        profile = select_profile(
            _config(),
            capability=(9, 0),
            ep_size=8,
            block_shape=(128, 128),
            max_tokens_per_rank=32,
            platform="cuda",
        )
        runtime = SharedEpRuntimeHooks(
            name="test",
            platform="cuda",
            create_state=Mock(),
            capabilities=frozenset(SharedEpRuntimeCapability),
        )
        admit_framework.return_value = (profile, None, runtime, "cpu_group")
        get_shared_state.return_value = state

        dispatcher = SharedEpDispatcher(_config())

        self.assertIs(dispatcher.state, state)
        get_shared_state.assert_called_once_with(
            dispatcher.config,
            dispatcher.profile,
            runtime,
            model_namespace="target",
            lane_id=0,
        )

    @patch(
        "sglang.srt.layers.moe.shared_ep.backend.get_forward",
        return_value=SimpleNamespace(shared_ep_is_decode=False),
    )
    def test_non_decode_phase_delegates_dispatch_and_combine(self, _get_forward):
        fallback = Mock()
        dispatched = object()
        combined = torch.ones((33, 4))
        fallback.dispatch.return_value = dispatched
        fallback.combine.return_value = combined
        dispatcher = SharedEpDispatcher.__new__(SharedEpDispatcher)
        dispatcher.lane_id = 0
        dispatcher._stage = "initial"
        dispatcher._active_uses_shared_ep = None
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
        "sglang.srt.layers.moe.shared_ep.backend.get_forward",
        return_value=SimpleNamespace(shared_ep_is_decode=True),
    )
    @patch(
        "sglang.srt.layers.moe.shared_ep.backend.get_dp_global_num_tokens",
        return_value=[16, 33, 8, 8, 8, 8, 8, 8],
    )
    @patch("sglang.srt.layers.moe.shared_ep.kernels.quantize_pack_input")
    def test_decode_rejects_peer_capacity_overflow_before_publish(
        self,
        quantize_pack,
        _global_num_tokens,
        _get_forward,
    ):
        profile = SimpleNamespace(
            max_tokens_per_rank=32,
            ep_size=8,
            block_shape=(128, 128),
            quantization=SharedEpQuantization.BLOCK_FP8,
        )
        state = SimpleNamespace(
            local_input=object(),
            local_output=torch.ones((16, 1, 4)),
            global_input=SimpleNamespace(
                activations=object(),
                scales=object(),
            ),
            input_epoch=Mock(),
        )
        dispatcher = SharedEpDispatcher.__new__(SharedEpDispatcher)
        dispatcher.profile = profile
        dispatcher.state = state
        dispatcher.lane_id = 0
        dispatcher._stage = "initial"
        dispatcher._active_uses_shared_ep = None
        dispatcher.local_expert_start = 0
        dispatcher.fallback_dispatcher = Mock()
        dispatcher._decode_quant_admitted = True
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
        "sglang.srt.layers.moe.shared_ep.backend.get_forward",
        return_value=SimpleNamespace(shared_ep_is_decode=True),
    )
    @patch(
        "sglang.srt.layers.moe.shared_ep.backend.get_dp_global_num_tokens",
        return_value=[16] * 8,
    )
    @patch("sglang.srt.layers.moe.shared_ep.kernels.quantize_pack_input")
    def test_decode_rows_use_direct_path(
        self,
        quantize_pack,
        _global_num_tokens,
        _get_forward,
    ):
        fallback = Mock()
        profile = SimpleNamespace(
            max_tokens_per_rank=32,
            ep_size=8,
            block_shape=(128, 128),
            quantization=SharedEpQuantization.BLOCK_FP8,
        )
        state = SimpleNamespace(
            local_input=object(),
            local_output=torch.ones((16, 1, 4)),
            global_input=SimpleNamespace(
                activations=object(),
                scales=object(),
            ),
            input_epoch=Mock(),
        )
        dispatcher = SharedEpDispatcher.__new__(SharedEpDispatcher)
        dispatcher.profile = profile
        dispatcher.state = state
        dispatcher.lane_id = 0
        dispatcher._stage = "initial"
        dispatcher._active_uses_shared_ep = None
        dispatcher.local_expert_start = 0
        dispatcher.fallback_dispatcher = fallback
        dispatcher._decode_quant_admitted = True
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
        self.assertTrue(torch.all(state.local_output == 0))
        fallback.dispatch.assert_not_called()

    @patch(
        "sglang.srt.layers.moe.shared_ep.backend.get_forward",
        return_value=SimpleNamespace(
            shared_ep_is_decode=False,
            shared_ep_is_prefill=True,
        ),
    )
    @patch(
        "sglang.srt.layers.moe.shared_ep.backend.get_dp_global_num_tokens",
        return_value=[16] * 8,
    )
    @patch("sglang.srt.layers.moe.shared_ep.kernels.quantize_pack_input")
    def test_admitted_prefill_publishes_pull_cache_object(
        self,
        quantize_pack,
        _global_num_tokens,
        _get_forward,
    ):
        fallback = Mock()
        decode_profile = SimpleNamespace(
            max_tokens_per_rank=32,
            ep_size=8,
            block_shape=(128, 128),
            quantization=SharedEpQuantization.BLOCK_FP8,
        )
        prefill_profile = SimpleNamespace(
            max_tokens_per_rank=1024,
            ep_size=8,
            block_shape=(128, 128),
            quantization=SharedEpQuantization.BLOCK_FP8,
        )
        prefill_state = SimpleNamespace(
            local_input=object(),
            local_output=torch.ones((16, 1, 4)),
            global_input=SimpleNamespace(
                activations=object(),
                scales=object(),
            ),
            input_epoch=Mock(),
        )
        pull_cache = object()
        dispatcher = SharedEpDispatcher.__new__(SharedEpDispatcher)
        dispatcher.profile = decode_profile
        dispatcher.prefill_profile = prefill_profile
        dispatcher.prefill_state = prefill_state
        dispatcher.prefill_cache = pull_cache
        dispatcher.lane_id = 0
        dispatcher._stage = "initial"
        dispatcher._active_uses_shared_ep = None
        dispatcher.local_expert_start = 0
        dispatcher.fallback_dispatcher = fallback
        dispatcher._decode_quant_admitted = True
        hidden_states = torch.zeros((16, 4))
        topk_output = StandardTopKOutput(
            topk_weights=torch.ones((16, 1)),
            topk_ids=torch.zeros((16, 1), dtype=torch.int32),
            router_logits=None,
        )

        output = dispatcher.dispatch(hidden_states, topk_output)

        self.assertEqual(output.phase, "prefill")
        self.assertIs(output.profile, prefill_profile)
        self.assertIs(output.state, prefill_state)
        self.assertIs(output.pull_cache, pull_cache)
        quantize_pack.assert_called_once_with(
            prefill_state.local_input,
            source=hidden_states,
            source_ids=topk_output.topk_ids,
            source_weights=topk_output.topk_weights,
            group_size=128,
        )
        self.assertTrue(torch.all(prefill_state.local_output == 0))
        prefill_state.input_epoch.publish.assert_called_once_with()
        fallback.dispatch.assert_not_called()

    def test_decode_combine_uses_the_admitted_lane_stage(self):
        fallback = Mock()
        dispatcher = SharedEpDispatcher.__new__(SharedEpDispatcher)
        dispatcher.lane_id = 0
        dispatcher._stage = "after_dispatch_b"
        dispatcher._active_uses_shared_ep = True
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

    @patch.object(PermuteMethodPool, "get_post_permute")
    @patch.object(PermuteMethodPool, "get_pre_permute")
    def test_prefill_reuses_constructed_native_runner(
        self,
        get_pre_permute,
        get_post_permute,
    ):
        pre_permute = Mock(return_value="runner_input")
        post_permute = Mock(return_value="combined")
        get_pre_permute.return_value = pre_permute
        get_post_permute.return_value = post_permute
        dispatch_output = SimpleNamespace(format=SimpleNamespace(value="deepep_normal"))
        fallback_quant_info = Mock()
        quant_info = SharedEpQuantInfo(
            w13_weight=torch.empty((1, 2, 3)),
            w2_weight=torch.empty((1, 3, 1)),
            w13_scale=torch.empty((1, 1, 1)),
            w2_scale=torch.empty((1, 1, 1)),
            block_shape=(128, 128),
            fallback_quant_info=fallback_quant_info,
            fallback_backend=MoeRunnerBackend.DEEP_GEMM,
        )
        config = MoeRunnerConfig()
        runner_core = SimpleNamespace(
            runner_backend=MoeRunnerBackend.DEEP_GEMM,
            run=Mock(return_value="runner_output"),
        )
        runner = MoeRunner.__new__(MoeRunner)
        runner.runner_backend = MoeRunnerBackend.DEEP_GEMM
        runner.runner_core = runner_core
        runner.config = config
        runner.fused_func = Mock()
        runner.is_shared_ep = True
        runner.lora_enabled = False
        runner.down_gemm_overlap_args = None
        runner.meta_overlap_args = None

        result = runner.run(dispatch_output, quant_info)

        self.assertEqual(result, "combined")
        runner.fused_func.assert_not_called()
        pre_permute.assert_called_once_with(
            dispatch_output,
            fallback_quant_info,
            config,
            {},
        )
        runner_core.run.assert_called_once_with(
            "runner_input",
            fallback_quant_info,
            {},
            hooks=None,
        )
        post_permute.assert_called_once_with(
            "runner_output",
            fallback_quant_info,
            config,
            {},
        )

    @patch("sglang.srt.layers.moe.shared_ep.backend._validate_decode_weights")
    @patch(
        "sglang.srt.layers.moe.shared_ep.backend.intermediate_capacity",
        return_value=64,
    )
    @patch("sglang.srt.layers.moe.shared_ep.kernels.prepare_routes")
    @patch("sglang.kernels.ops.attention.dsv4.silu_and_mul_contig_post_quant")
    @patch("sglang.srt.layers.moe.shared_ep.backend.pull_cache_rows")
    @patch("sglang.srt.layers.moe.shared_ep.backend.invoke_pull_cache_w13")
    @patch("sglang.kernels.ops.moe.fused_moe_triton_kernels.invoke_fused_moe_kernel")
    def test_fused_prefill_pulls_w13_and_direct_returns_w2(
        self,
        fused_moe,
        pull_w13,
        pull_rows,
        _silu_and_mul,
        prepare_routes,
        _capacity,
        _validate_weights,
    ):
        decode_profile = select_profile(
            _config(),
            platform="rocm",
            capability=(9, 5),
            ep_size=8,
            block_shape=(128, 128),
            max_tokens_per_rank=32,
        )
        profile = make_pull_cache_prefill_profile(decode_profile, 1024)
        self.assertIsNotNone(profile)
        routes = SimpleNamespace(
            local_ids=torch.zeros((8, 1, profile.top_k), dtype=torch.int32),
            local_weights=torch.ones((8, 1, profile.top_k), dtype=torch.float32),
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
                topk_ids=torch.zeros((8, 1, profile.top_k), dtype=torch.int32),
                topk_weights=torch.ones((8, 1, profile.top_k), dtype=torch.float32),
            ),
            input_epoch=input_epoch,
            global_output=torch.empty((64, profile.hidden_size), dtype=torch.bfloat16),
            local_output=torch.zeros(
                (1, profile.top_k, profile.hidden_size), dtype=torch.bfloat16
            ),
            output_epoch=Mock(),
        )
        pull_cache = SimpleNamespace(active_rows=64)
        dispatch_output = SharedEpDispatchOutput(
            hidden_states=torch.empty(
                (8, profile.hidden_size), dtype=torch.float8_e4m3fn
            ),
            hidden_states_scale=torch.ones(
                (8, profile.hidden_size // 128), dtype=torch.float32
            ),
            topk_output=object(),
            state=state,
            profile=profile,
            num_tokens=1,
            local_expert_start=0,
            phase="prefill",
            pull_cache=pull_cache,
        )
        quant_info = SimpleNamespace(
            w13_weight=torch.empty((1, 1, 1), dtype=torch.float8_e4m3fn),
            w13_scale=torch.ones((1, 1, 1), dtype=torch.float32),
            w2_weight=torch.empty((1, 1, 1), dtype=torch.float8_e4m3fn),
            w2_scale=torch.ones((1, 1, 1), dtype=torch.float32),
        )

        output = run_shared_ep(
            dispatch_output,
            quant_info,
            SimpleNamespace(swiglu_limit=None),
        )

        self.assertEqual(output.hidden_states.shape, (1, profile.hidden_size))
        pull_rows.assert_called_once()
        pull_w13.assert_called_once()
        fused_moe.assert_called_once()
        state.output_epoch.publish.assert_called_once_with()
        state.output_epoch.wait_all.assert_called_once_with()

    def test_quant_metadata_rejects_aiter_shuffled_decode_weights(self):
        w13 = torch.empty((1, 2, 3))
        w2 = torch.empty((1, 3, 1))
        info = SharedEpQuantInfo(
            w13_weight=w13,
            w2_weight=w2,
            w13_scale=torch.empty((1, 1, 1)),
            w2_scale=torch.empty((1, 1, 1)),
            block_shape=(128, 128),
            fallback_quant_info=Mock(),
            fallback_backend=MoeRunnerBackend.AITER,
        )
        info.require_decode_capability(SharedEpQuantCapability.CANONICAL_BLOCK_FP8)

        w13.is_shuffled = True
        with self.assertRaisesRegex(ValueError, "AITER-pre-shuffled"):
            info.require_decode_capability(SharedEpQuantCapability.CANONICAL_BLOCK_FP8)

    def test_aiter_fallback_shuffle_preserves_canonical_decode_weights(self):
        layer = SimpleNamespace(
            w13_weight=torch.nn.Parameter(torch.arange(6.0).view(1, 2, 3)),
            w2_weight=torch.nn.Parameter(torch.arange(3.0).view(1, 3, 1)),
        )
        canonical_w13 = layer.w13_weight.detach().clone()
        canonical_w2 = layer.w2_weight.detach().clone()

        def fake_shuffle(weight, _shape):
            return weight.detach().clone() + 10

        with patch(
            "sglang.srt.layers.quantization.fp8.shuffle_weight",
            side_effect=fake_shuffle,
            create=True,
        ):
            Fp8MoEMethod._prepare_shared_ep_aiter_fallback_weights(layer)

        torch.testing.assert_close(layer.w13_weight, canonical_w13)
        torch.testing.assert_close(layer.w2_weight, canonical_w2)
        self.assertTrue(layer._shared_ep_aiter_w13_weight.is_shuffled)
        self.assertTrue(layer._shared_ep_aiter_w2_weight.is_shuffled)
        self.assertFalse(layer.w13_weight.is_shuffled)
        self.assertFalse(layer.w2_weight.is_shuffled)

    @patch("sglang.srt.layers.moe.shared_ep.backend._synchronize_admission_stage")
    def test_quant_admission_publishes_canonical_shapes(self, synchronize):
        dispatcher = SharedEpDispatcher.__new__(SharedEpDispatcher)
        dispatcher.profile = select_profile(
            _config(),
            capability=(9, 0),
            ep_size=8,
            block_shape=(128, 128),
            max_tokens_per_rank=32,
            platform="cuda",
        )
        dispatcher.cpu_group = "cpu_group"
        dispatcher.fallback_dispatcher = Mock()
        dispatcher._decode_quant_admitted = False
        quant_config = {
            "weight_dtype": torch.float8_e4m3fn,
            "shared_ep_quantization": "block_fp8",
            "shared_ep_weight_layout": SharedEpWeightLayout.CANONICAL.value,
            "block_shape": (128, 128),
            "w13_shape": (32, 4096, 4096),
            "w2_shape": (32, 4096, 2048),
            "w13_scale_shape": (32, 32, 32),
            "w2_scale_shape": (32, 32, 16),
        }

        dispatcher.set_quant_config(quant_config)

        self.assertTrue(dispatcher._decode_quant_admitted)
        dispatcher.fallback_dispatcher.set_quant_config.assert_called_once_with(
            quant_config
        )
        synchronize.assert_called_once()
        self.assertEqual(
            synchronize.call_args.kwargs["stage"],
            "quantization",
        )

    def test_remote_admission_failure_is_synchronized(self):
        def fake_all_gather(results, local_result, *, group):
            results[:] = [
                local_result,
                (None, "ValueError: unsupported gfx architecture"),
            ]

        with (
            patch(
                "sglang.srt.layers.moe.shared_ep.backend.dist.get_world_size",
                return_value=2,
            ),
            patch(
                "sglang.srt.layers.moe.shared_ep.backend.dist.get_rank",
                return_value=0,
            ),
            patch(
                "sglang.srt.layers.moe.shared_ep.backend.dist.all_gather_object",
                side_effect=fake_all_gather,
            ),
            self.assertRaisesRegex(
                RuntimeError,
                "framework admission failed on rank 1.*unsupported gfx",
            ),
        ):
            _synchronize_admission_stage(
                "cpu_group",
                stage="framework",
                descriptor=("rocm:gfx950", "runtime", ("profile",)),
                local_error=None,
            )

    def test_runtime_requires_vmm_and_epoch_capabilities(self):
        hooks = SharedEpRuntimeHooks(
            name="incomplete_rocm",
            platform="rocm",
            create_state=Mock(),
            capabilities=frozenset({SharedEpRuntimeCapability.RANK_MAJOR_VMM}),
        )

        with self.assertRaisesRegex(RuntimeError, "system_scope_gpu_epoch"):
            hooks.validate()


if __name__ == "__main__":
    unittest.main()
