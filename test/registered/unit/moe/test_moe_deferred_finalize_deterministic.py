"""Hermetic CPU unit tests: deterministic inference disables MoE deferred finalize.

MoE deferred finalize skips the in-op expert reduction below a token-count
threshold (default ``SGLANG_MOE_DEFERRED_FINALIZE_MAX_TOKENS = 192``) and fuses
the weighted expert sum with the shared-expert output in
``finalize_flashinfer_trtllm_deferred_output``, while larger batches finalize
inside the MoE kernel in BF16 first and add ``shared_output`` afterward. That
changes the BF16 rounding order across ``M = 192`` (decode vs. prefill, or small
vs. large batches), violating batch invariance under
``--enable-deterministic-inference``.
"""

import unittest
from contextlib import nullcontext
from types import SimpleNamespace
from unittest import mock

import torch

from sglang.srt.arg_groups import serving_hook
from sglang.srt.environ import envs
from sglang.srt.layers.moe import MoeA2ABackend, MoeRunnerBackend
from sglang.srt.layers.moe.fused_moe_triton import layer as fused_moe_layer
from sglang.srt.layers.moe.topk import TopKOutputFormat
from sglang.srt.layers.quantization.fp8 import Fp8MoEMethod
from sglang.srt.layers.quantization.modelopt_quant import ModelOptNvFp4FusedMoEMethod
from sglang.srt.layers.quantization.unquant import UnquantizedFusedMoEMethod
from sglang.srt.models import deepseek_v2
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


def _fake_exec(deterministic: bool):
    return SimpleNamespace(
        deterministic=SimpleNamespace(enable_deterministic_inference=deterministic),
        moe=SimpleNamespace(
            ep_join_mode="none",
            enable_w4a4_mxfp4_megamoe=False,
            enable_paged_experts=False,
            ep_num_redundant_experts=0,
            ep_dispatch_algorithm=None,
            expert_distribution_recorder_mode=None,
        ),
        features=SimpleNamespace(enable_return_routed_experts=False),
    )


class _DummyMethod(UnquantizedFusedMoEMethod):
    def __init__(self, backend: MoeRunnerBackend):
        self._backend = backend

    def create_weights(self, *args, **kwargs):
        return None

    def create_moe_runner(self, *args, **kwargs):
        return None

    def get_moe_runner_backend(self, layer):
        return self._backend


class _DummyNvFp4Method(ModelOptNvFp4FusedMoEMethod):
    def __init__(self, backend: MoeRunnerBackend):
        self._backend = backend

    def create_weights(self, *args, **kwargs):
        return None

    def create_moe_runner(self, *args, **kwargs):
        return None

    def get_moe_runner_backend(self, layer):
        return self._backend


class _DummyBlockFp8Method(Fp8MoEMethod):
    block_quant = True
    use_mxfp8 = False

    def __init__(self, backend: MoeRunnerBackend):
        self._backend = backend

    def create_weights(self, *args, **kwargs):
        return None

    def create_moe_runner(self, *args, **kwargs):
        return None

    def get_moe_runner_backend(self, layer):
        return self._backend


class TestMoeDeferredFinalizeDeterministic(CustomTestCase):
    def test_serving_hook_disables_deferred_finalize_env_in_deterministic_mode(self):
        cfg = SimpleNamespace(
            enable_torch_compile=False,
            mamba_ssm_dtype=None,
            disable_outlines_disk_cache=False,
            enable_deterministic_inference=True,
            debug_cuda_graph=False,
            enable_deepseek_v4_fp4_indexer=False,
        )
        with (
            envs.SGLANG_ENABLE_MOE_DEFERRED_FINALIZE.override(True),
            envs.SGLANG_FLASHINFER_MOE_FUSED_FINALIZE.override(True),
            envs.SGLANG_ENABLE_DETERMINISTIC_INFERENCE.override(False),
            mock.patch.object(serving_hook, "resolving_view", return_value=cfg),
            mock.patch.object(serving_hook, "handle_multimodal_feature_transport"),
        ):
            serving_hook.handle_environment_variables(SimpleNamespace())
            self.assertFalse(envs.SGLANG_ENABLE_MOE_DEFERRED_FINALIZE.get())
            self.assertFalse(envs.SGLANG_FLASHINFER_MOE_FUSED_FINALIZE.get())

    def _build_fused_moe(
        self,
        *,
        deterministic: bool,
        quant_method,
        hidden_size: int = 2560,
        num_experts: int = 512,
        top_k: int = 10,
        enable_qwen35_fp8_deferred_finalize: bool = False,
    ):
        quant_cfg = mock.MagicMock()
        quant_cfg.get_name.return_value = "custom"
        quant_cfg.get_quant_method.return_value = quant_method
        with (
            envs.SGLANG_ENABLE_MOE_DEFERRED_FINALIZE.override(True),
            mock.patch.object(
                fused_moe_layer, "get_exec", return_value=_fake_exec(deterministic)
            ),
            mock.patch.object(
                fused_moe_layer,
                "get_flags",
                return_value=SimpleNamespace(
                    moe=SimpleNamespace(in_speculative_scope=False)
                ),
            ),
            mock.patch.object(
                fused_moe_layer,
                "get_parallel",
                return_value=SimpleNamespace(
                    moe_ep_size=1, moe_ep_rank=0, moe_tp_size=1, moe_tp_rank=0
                ),
            ),
            mock.patch.object(
                fused_moe_layer,
                "get_server_args",
                return_value=SimpleNamespace(),
            ),
            mock.patch.object(
                fused_moe_layer,
                "create_kt_config_from_server_args",
                return_value=None,
            ),
            mock.patch.object(
                fused_moe_layer,
                "get_moe_runner_backend",
                return_value=MoeRunnerBackend.AUTO,
            ),
            mock.patch.object(
                fused_moe_layer,
                "get_moe_a2a_backend",
                return_value=MoeA2ABackend.NONE,
            ),
            mock.patch.object(
                fused_moe_layer,
                "_fuses_routed_scaling_factor_in_topk",
                return_value=True,
            ),
            mock.patch.object(
                fused_moe_layer,
                "_validate_hpc_ops_quant_method",
                return_value=None,
            ),
            mock.patch.object(
                fused_moe_layer, "_validate_deepep_v2_quant_method", return_value=None
            ),
            mock.patch.object(
                fused_moe_layer, "_validate_gluon_quant_method", return_value=None
            ),
            mock.patch.object(
                fused_moe_layer.FusedMoE, "_set_runner_flags", return_value=None
            ),
            mock.patch.object(
                fused_moe_layer,
                "create_moe_dispatcher",
                return_value=SimpleNamespace(expert_mask_gpu=None),
            ),
        ):
            return fused_moe_layer.FusedMoE(
                num_experts=num_experts,
                top_k=top_k,
                hidden_size=hidden_size,
                intermediate_size=1024,
                layer_id=0,
                params_dtype=torch.bfloat16,
                quant_config=quant_cfg,
                enable_qwen35_fp8_deferred_finalize=enable_qwen35_fp8_deferred_finalize,
            )

    def test_fused_moe_disables_all_deferred_finalize_variants_in_deterministic_mode(
        self,
    ):
        # 1. NVFP4 bypassed (supports_deferred_finalize)
        moe_def = self._build_fused_moe(
            deterministic=False,
            quant_method=_DummyNvFp4Method(MoeRunnerBackend.FLASHINFER_TRTLLM),
        )
        self.assertTrue(moe_def.supports_deferred_finalize)
        moe_det = self._build_fused_moe(
            deterministic=True,
            quant_method=_DummyNvFp4Method(MoeRunnerBackend.FLASHINFER_TRTLLM),
        )
        self.assertFalse(moe_det.supports_deferred_finalize)

        # 2. NVFP4 routed (supports_routed_deferred_finalize)
        routed_def = self._build_fused_moe(
            deterministic=False,
            quant_method=_DummyNvFp4Method(MoeRunnerBackend.FLASHINFER_TRTLLM_ROUTED),
        )
        self.assertTrue(routed_def.supports_routed_deferred_finalize)
        routed_det = self._build_fused_moe(
            deterministic=True,
            quant_method=_DummyNvFp4Method(MoeRunnerBackend.FLASHINFER_TRTLLM_ROUTED),
        )
        self.assertFalse(routed_det.supports_routed_deferred_finalize)

        # 3. Qwen3.5 block-FP8 & Qwen4 BF16 (do not read SGLANG_ENABLE_MOE_DEFERRED_FINALIZE)
        q35_def = self._build_fused_moe(
            deterministic=False,
            quant_method=_DummyBlockFp8Method(MoeRunnerBackend.FLASHINFER_TRTLLM),
            enable_qwen35_fp8_deferred_finalize=True,
        )
        self.assertTrue(q35_def.supports_deferred_finalize)
        q35_det = self._build_fused_moe(
            deterministic=True,
            quant_method=_DummyBlockFp8Method(MoeRunnerBackend.FLASHINFER_TRTLLM),
            enable_qwen35_fp8_deferred_finalize=True,
        )
        self.assertFalse(q35_det.supports_deferred_finalize)

        q4_def = self._build_fused_moe(
            deterministic=False,
            quant_method=_DummyMethod(MoeRunnerBackend.FLASHINFER_TRTLLM),
        )
        self.assertTrue(q4_def.supports_deferred_finalize)
        q4_det = self._build_fused_moe(
            deterministic=True,
            quant_method=_DummyMethod(MoeRunnerBackend.FLASHINFER_TRTLLM),
        )
        self.assertFalse(q4_det.supports_deferred_finalize)

    def _run_deepseek_dual_stream(self, *, deterministic: bool, num_tokens: int):
        deferred_calls = []
        eager_calls = []

        class _FakeExperts:
            supports_deferred_finalize = True
            supports_routed_deferred_finalize = True
            quant_method = object()

            def forward_deferred_finalize(self, h, topk, pre_quant_input=None):
                deferred_calls.append(h.shape[0])
                return torch.ones_like(h)

            def forward_impl(self, h, topk):
                eager_calls.append(h.shape[0])
                return torch.ones_like(h)

            def __call__(self, h, topk, pre_quant_input=None):
                eager_calls.append(h.shape[0])
                return torch.ones_like(h)

        fake_stream = SimpleNamespace(wait_stream=lambda other: None)
        fake_moe = SimpleNamespace(
            shared_experts=object(),
            num_fused_shared_experts=0,
            _shared_expert_tp1=False,
            _deferred_finalize_max_tokens=192,
            experts=_FakeExperts(),
            routed_scaling_factor=1.0,
            routed_scaling_factor_fused_in_topk=True,
            alt_stream=fake_stream,
            routed_quant_stream=fake_stream,
            reduce_results=False,
            gate=lambda h, gemm_output_zero_allocator=None: (
                torch.zeros(h.shape[0], 4),
                None,
            ),
            topk=lambda h, logits, **kw: SimpleNamespace(
                format=TopKOutputFormat.STANDARD
            ),
            _forward_shared_experts=lambda h, z, pre_quant_input=None: torch.zeros_like(
                h
            ),
            _maybe_quant_moe_input_once=lambda h: None,
            _should_quant_routed_input_mxfp8=lambda h: False,
            _build_topk_kwargs=lambda pre_quant_input: {},
            tp_size=1,
        )
        with (
            mock.patch.object(
                deepseek_v2, "get_exec", return_value=_fake_exec(deterministic)
            ),
            mock.patch.object(deepseek_v2, "_is_cuda", True),
            mock.patch.object(
                deepseek_v2,
                "get_forward",
                return_value=SimpleNamespace(
                    flashinfer_trtllm_bypass=False, defer_moe_finalize=False
                ),
            ),
            mock.patch.object(
                deepseek_v2,
                "maybe_fuse_routed_scale_and_shared_add",
                side_effect=lambda experts, routed, shared, scale: routed + shared,
            ),
            mock.patch.object(torch.cuda, "current_stream", return_value=fake_stream),
            mock.patch.object(
                torch.cuda, "stream", side_effect=lambda s: nullcontext()
            ),
            mock.patch(
                "sglang.srt.layers.moe.moe_runner.flashinfer_trtllm.finalize_flashinfer_trtllm_deferred_output",
                side_effect=lambda routed, shared: routed + shared,
            ),
        ):
            deepseek_v2.DeepseekV2MoE.forward_normal_dual_stream(
                fake_moe,
                torch.zeros(num_tokens, 8, dtype=torch.bfloat16),
            )
        return deferred_calls, eager_calls

    def test_deepseek_dual_stream_keeps_single_finalize_path_across_m192(self):
        # Default mode switches paths at M = 192.
        def_small, eag_small = self._run_deepseek_dual_stream(
            deterministic=False, num_tokens=16
        )
        def_large, eag_large = self._run_deepseek_dual_stream(
            deterministic=False, num_tokens=256
        )
        self.assertEqual((def_small, eag_small), ([16], []))
        self.assertEqual((def_large, eag_large), ([], [256]))

        # Deterministic mode stays on the in-kernel finalize path for all M.
        det_def_small, det_eag_small = self._run_deepseek_dual_stream(
            deterministic=True, num_tokens=16
        )
        det_def_large, det_eag_large = self._run_deepseek_dual_stream(
            deterministic=True, num_tokens=256
        )
        self.assertEqual((det_def_small, det_eag_small), ([], [16]))
        self.assertEqual((det_def_large, det_eag_large), ([], [256]))


if __name__ == "__main__":
    unittest.main()
