from __future__ import annotations

import unittest
from types import SimpleNamespace
from unittest import mock

import torch

from sglang.srt.layers.moe.utils import MoeRunnerBackend
from sglang.srt.lora.backend.base_backend import BaseLoRABackend
from sglang.srt.lora.moe.base_gemm_provider import select_provider_cls
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=10, stage="base-b-kernel-unit", runner_config="1-gpu-large")


class _FakeMoeLayer:
    def __init__(self, *, device=None):
        device = device or torch.device("cpu")
        # The buffer sizing reads logical dims from the layer (quant-info
        # tensor shapes are form-specific; Marlin payloads pack both dims).
        self.base_layer = SimpleNamespace(
            top_k=2,
            num_experts=4,
            num_local_experts=4,
            intermediate_size_per_partition=4,
            hidden_size=6,
            moe_runner_config=SimpleNamespace(is_gated=True),
            w13_weight=torch.zeros(4, 8, 6, device=device),
        )
        self._quant_info = SimpleNamespace(
            w13_weight=torch.zeros(4, 8, 6, device=device),
            w2_weight=torch.zeros(4, 6, 4, device=device),
        )


class TestGraphBatchMetadata(CustomTestCase):
    """Graph buffers of the legacy MoE LoRA kernels; V2 owns its own metadata."""

    METADATA_KEYS = ("adapter_enabled", "token_lora_mapping")

    def _backend(self) -> BaseLoRABackend:
        backend = object.__new__(BaseLoRABackend)
        backend._is_moe_lora = True
        backend.prefill_cuda_graph_batch_info = None
        return backend

    def test_legacy_path_still_gets_every_buffer(self):
        backend = self._backend()
        backend.init_cuda_graph_moe_buffers(
            max_bs=8,
            max_loras=4,
            compute_dtype=torch.bfloat16,
            moe_layer=_FakeMoeLayer(),
        )
        for key in (
            *self.METADATA_KEYS,
            "sorted_token_ids_lora",
            "expert_ids_lora",
            "num_tokens_post_padded_lora",
            "cumsum_buffer",
            "token_mask",
            "lora_ids",
        ):
            self.assertIn(key, backend.moe_cg_buffers)

    def test_two_metadata_updates_keep_addresses_and_change_contents(self):
        """``prepare_lora_batch`` runs OUTSIDE the captured graph while the
        runner reads these tensors INSIDE it, so an update must write in place."""
        if not torch.cuda.is_available():
            raise unittest.SkipTest("CUDA required: the metadata update is Triton")
        device = torch.device("cuda")
        backend = self._backend()
        backend.init_cuda_graph_moe_buffers(
            max_bs=8,
            max_loras=4,
            compute_dtype=torch.bfloat16,
            moe_layer=_FakeMoeLayer(device=device),
        )
        enabled_ptr = backend.moe_cg_buffers["adapter_enabled"].data_ptr()
        mapping_ptr = backend.moe_cg_buffers["token_lora_mapping"].data_ptr()

        def update(*, weight_indices, lora_ranks, seg_lens):
            seg_indptr = torch.tensor(
                [0, *torch.tensor(seg_lens).cumsum(0).tolist()],
                dtype=torch.int32,
                device=device,
            )
            batch_info = SimpleNamespace(
                use_cuda_graph=True,
                bs=len(seg_lens),
                num_segments=len(seg_lens),
                seg_indptr=seg_indptr,
                weight_indices=torch.tensor(
                    weight_indices, dtype=torch.int32, device=device
                ),
                lora_ranks=torch.tensor(lora_ranks, dtype=torch.int32, device=device),
                req_seg_indptr=None,
                req_weight_indices=None,
                moe_lora_info=None,
            )
            forward_batch = SimpleNamespace(
                batch_size=sum(seg_lens),
                extend_seq_lens_cpu=seg_lens,
                extend_num_tokens=sum(seg_lens),
                forward_mode=SimpleNamespace(
                    is_idle=lambda: False,
                    is_extend=lambda: True,
                    is_decode=lambda: False,
                    is_target_verify=lambda: False,
                ),
            )
            backend._add_moe_lora_info(forward_batch, batch_info)
            return batch_info.moe_lora_info

        # Slot 1 active (rank 16), slot 0 disabled (rank 0 = the base slot).
        first = update(weight_indices=[1, 0], lora_ranks=[0, 16], seg_lens=[2, 2])
        first_mapping = first.token_lora_mapping[:4].clone()
        first_enabled = first.adapter_enabled.clone()

        second = update(weight_indices=[2, 0], lora_ranks=[0, 16, 32], seg_lens=[3, 1])
        torch.cuda.synchronize()

        self.assertEqual(first.token_lora_mapping.data_ptr(), mapping_ptr)
        self.assertEqual(second.token_lora_mapping.data_ptr(), mapping_ptr)
        self.assertEqual(second.adapter_enabled.data_ptr(), enabled_ptr)
        self.assertFalse(
            torch.equal(second.token_lora_mapping[:4], first_mapping),
            "the second assignment must actually change the mapping in place",
        )
        self.assertFalse(
            torch.equal(second.adapter_enabled, first_enabled),
            "enabling a different slot must change adapter_enabled in place",
        )
        self.assertEqual(int(second.adapter_enabled[2]), 1)
        self.assertEqual(int(second.adapter_enabled[0]), 0)


class TestEngineAdmission(CustomTestCase):
    def _base_layer(self, **overrides):
        from sglang.srt.layers.moe.token_dispatcher.standard import StandardDispatcher
        from sglang.srt.layers.quantization.unquant import UnquantizedFusedMoEMethod

        quant_method = object.__new__(UnquantizedFusedMoEMethod)
        dispatcher = object.__new__(StandardDispatcher)
        dispatcher.skip_local_expert_mapping = overrides.get("skip_local", False)
        return SimpleNamespace(
            quant_method=quant_method,
            dispatcher=dispatcher,
            w13_weight=torch.zeros(2, 8, 4, dtype=torch.bfloat16),
            w2_weight=torch.zeros(2, 4, 4, dtype=torch.bfloat16),
            num_local_experts=2,
            should_fuse_routed_scaling_factor_in_topk=overrides.get(
                "fused_scaling", False
            ),
            moe_runner_config=SimpleNamespace(
                activation=overrides.get("activation", "silu"),
                is_gated=True,
                gemm1_alpha=None,
                gemm1_clamp_limit=None,
                swiglu_limit=None,
                apply_router_weight_on_input=False,
                no_combine=False,
                num_fused_shared_experts=0,
                top_k=2,
                routed_scaling_factor=1.0,
            ),
        )

    def _admit(self, **overrides):
        from sglang.srt.lora.moe.runner import MoeLoraRunner

        MoeLoraRunner._admit(self._base_layer(**overrides))

    def test_admits_a_supported_layer(self):
        self._admit()

    def test_weight_family_of_a_bf16_layer(self):
        from sglang.srt.lora.moe.runner import MoeLoraRunner

        assert MoeLoraRunner._weight_family(self._base_layer()) == "bf16"

    def _fp8_layer(self, *, block_quant=True, weight_dtype=torch.float8_e4m3fn):
        from sglang.srt.layers.quantization.fp8 import Fp8MoEMethod

        layer = self._base_layer()
        quant_method = object.__new__(Fp8MoEMethod)
        quant_method.block_quant = block_quant
        quant_method.use_mxfp8 = False
        quant_method.weight_block_size = [128, 128] if block_quant else None
        layer.quant_method = quant_method
        layer.w13_weight = layer.w13_weight.to(weight_dtype)
        layer.w2_weight = layer.w2_weight.to(weight_dtype)
        return layer

    def test_weight_family_of_a_block_fp8_layer(self):
        from sglang.srt.lora.moe.runner import MoeLoraRunner

        assert MoeLoraRunner._weight_family(self._fp8_layer()) == "fp8"

    def test_rejects_per_tensor_fp8(self):
        from sglang.srt.lora.moe.runner import MoeLoraRunner

        with self.assertRaisesRegex(NotImplementedError, "128-block"):
            MoeLoraRunner._weight_family(self._fp8_layer(block_quant=False))

    def test_rejects_marlin_repacked_fp8(self):
        from sglang.srt.lora.moe.runner import MoeLoraRunner

        with self.assertRaisesRegex(NotImplementedError, "float8_e4m3fn"):
            MoeLoraRunner._weight_family(self._fp8_layer(weight_dtype=torch.bfloat16))

    def test_every_listed_vendor_resolves_and_others_fall_back_to_the_first(self):
        """Mixed-quant layers resolve independently, falling back to their family's vendor."""
        from sglang.srt.lora.moe.base_gemm_provider import VENDORS

        every_vendor = {vendor for vendors in VENDORS.values() for vendor in vendors}
        for family, vendors in VENDORS.items():
            default = select_provider_cls("route_major", family)
            assert default is select_provider_cls("route_major", family, vendors[0])
            for vendor in vendors:
                assert select_provider_cls("expert_major", family, vendor)
            for other in every_vendor - set(vendors):
                assert select_provider_cls("route_major", family, other) is default

    def test_rejects_architectures_with_neither_mma_family(self):
        """SM90 and SM100 are a closed set, not a floor: SM120 reports major 12
        yet has neither WGMMA nor tcgen05, so a ">= SM90" check mis-admits it."""
        from sglang.srt.lora.moe.runner import MoeLoraRunner

        for capability in ((8, 0), (12, 0), (11, 0)):
            with mock.patch(
                "torch.cuda.get_device_capability", lambda device=None: capability
            ):
                with self.assertRaisesRegex(NotImplementedError, "SM90 and SM100"):
                    MoeLoraRunner._admit(self._base_layer())

    def test_rejects_global_expert_id_dispatch(self):
        with self.assertRaisesRegex(NotImplementedError, "EP-local"):
            self._admit(skip_local=True)

    def test_rejects_prefolded_routed_scaling(self):
        """Scaling folded into top-k would be applied twice."""
        with self.assertRaisesRegex(NotImplementedError, "routed scaling"):
            self._admit(fused_scaling=True)

    def test_rejects_unsupported_activation(self):
        with self.assertRaisesRegex(NotImplementedError, "SiLU or ReLU2"):
            self._admit(activation="gelu")


class TestLayerRunnerBackendResolution(CustomTestCase):
    """Bind the selected LoRA runner, otherwise the quant method's runner."""

    def _construct(self, *, global_backend, base_runner_backend, lora_backend="triton"):
        from sglang.srt.layers.moe.moe_runner.base import MoeRunnerConfig
        from sglang.srt.lora.layers import FusedMoEWithLoRA

        quant_method = mock.MagicMock()
        base_layer = SimpleNamespace(
            quant_method=quant_method,
            moe_runner_config=MoeRunnerConfig(
                num_experts=8,
                num_local_experts=8,
                hidden_size=64,
                intermediate_size_per_partition=32,
                layer_id=0,
                top_k=2,
                num_fused_shared_experts=0,
                params_dtype=torch.bfloat16,
                activation="silu",
                inplace=True,
            ),
            dispatcher=None,
            num_local_experts=8,
            should_fuse_routed_scaling_factor_in_topk=False,
            moe_tp_size=1,
            moe_tp_rank=0,
            intermediate_size_per_partition=32,
            runner=SimpleNamespace(runner_backend=base_runner_backend),
        )
        with (
            mock.patch(
                "sglang.srt.layers.moe.utils.get_moe_runner_backend",
                return_value=global_backend,
            ),
            mock.patch.object(FusedMoEWithLoRA, "_initialize_moe_lora_execution"),
        ):
            return FusedMoEWithLoRA(base_layer, SimpleNamespace(name=lora_backend))

    def test_v2_requires_new_moe_runner(self):
        for backend in (
            MoeRunnerBackend.TRITON,
            MoeRunnerBackend.DEEP_GEMM,
            MoeRunnerBackend.EXPERIMENTAL_SGL_TRTLLM,
            MoeRunnerBackend.EXPERIMENTAL_SGL_MARLIN,
        ):
            with (
                self.subTest(backend=backend),
                self.assertRaisesRegex(ValueError, "mixing new and legacy"),
            ):
                self._construct(
                    global_backend=backend,
                    base_runner_backend=backend,
                    lora_backend="triton_v2",
                )

    def test_v2_accepts_each_new_moe_runner(self):
        for backend in (
            MoeRunnerBackend.LORA_CUTEDSL,
            MoeRunnerBackend.LORA_TRITON,
            MoeRunnerBackend.LORA_MARLIN,
        ):
            with self.subTest(backend=backend):
                layer = self._construct(
                    global_backend=backend,
                    base_runner_backend=MoeRunnerBackend.DEEP_GEMM,
                    lora_backend="triton_v2",
                )
                self.assertIs(layer._lora_runner_backend, backend)

    def test_new_moe_runner_rejects_legacy_dense_backend(self):
        for backend in (
            MoeRunnerBackend.LORA_CUTEDSL,
            MoeRunnerBackend.LORA_TRITON,
            MoeRunnerBackend.LORA_MARLIN,
        ):
            with (
                self.subTest(backend=backend),
                self.assertRaisesRegex(ValueError, "mixing new and legacy"),
            ):
                self._construct(
                    global_backend=backend,
                    base_runner_backend=MoeRunnerBackend.DEEP_GEMM,
                )

    def test_lora_runner_wins_over_the_base_layers_vendor_runner(self):
        """bf16 base layers run DeepGEMM's layout under the LoRA runner; the
        legacy path must not be picked off that runner (it has no quant info)."""
        layer = self._construct(
            global_backend=MoeRunnerBackend.LORA_CUTEDSL,
            base_runner_backend=MoeRunnerBackend.DEEP_GEMM,
            lora_backend="triton_v2",
        )
        self.assertIs(layer._lora_runner_backend, MoeRunnerBackend.LORA_CUTEDSL)

    def test_experimental_backends_win_over_their_aliased_base_runner(self):
        """is_marlin() / is_flashinfer_trtllm() alias the experimental backends,
        so the base layer's runner says MARLIN / FLASHINFER_TRTLLM; the layer
        must still bind the experimental engine the server selected."""
        cases = (
            (
                MoeRunnerBackend.EXPERIMENTAL_SGL_MARLIN,
                MoeRunnerBackend.MARLIN,
                "sglang.srt.lora.marlin_lora_temp.lora_layer.init_experimental_sgl_marlin_lora",
            ),
            (
                MoeRunnerBackend.EXPERIMENTAL_SGL_TRTLLM,
                MoeRunnerBackend.FLASHINFER_TRTLLM,
                "sglang.srt.lora.trtllm_lora_temp.lora_layer.init_experimental_sgl_trtllm_lora",
            ),
        )
        for global_backend, base_runner_backend, init_path in cases:
            with mock.patch(init_path):
                layer = self._construct(
                    global_backend=global_backend,
                    base_runner_backend=base_runner_backend,
                )
            self.assertIs(layer._lora_runner_backend, global_backend)

    def test_legacy_path_follows_the_quant_methods_runner(self):
        layer = self._construct(
            global_backend=MoeRunnerBackend.AUTO,
            base_runner_backend=MoeRunnerBackend.TRITON,
        )
        self.assertIs(layer._lora_runner_backend, MoeRunnerBackend.TRITON)


if __name__ == "__main__":
    unittest.main()
