"""Behavioral tests for SGL LoRA attach, admission, and graph batch metadata.

Coverage, stated precisely:

* ``BaseLoRABackend.init_cuda_graph_moe_buffers`` is driven directly for both
  engines, including with a layer that has no ``_quant_info`` at all (the new
  engine never defines one), so the metadata path must not read layer geometry.
* ``_add_moe_lora_info`` is then driven twice with DIFFERENT assignments under
  ``use_cuda_graph=True`` and the metadata tensors are checked to keep their
  addresses while their contents change — the property CUDA-graph replay
  depends on, and the one a graph-enabled server previously crashed without.
  This requires CUDA because the underlying update is a Triton kernel.
* Admission is exercised through ``SglMoeLoraRunner._admit`` against
  synthesized resident states; it does not build a full ``FusedMoE``.
"""

from __future__ import annotations

import unittest
from types import SimpleNamespace
from unittest import mock

import torch

from sglang.srt.lora.backend.base_backend import BaseLoRABackend
from sglang.test.ci.ci_register import register_amd_ci, register_cuda_ci

register_cuda_ci(est_time=10, stage="base-b", runner_config="1-gpu-small")
register_amd_ci(est_time=10, suite="stage-b-test-1-gpu-small-amd")


class _FakeMoeLayer:
    """Minimal stand-in exposing only what buffer init reads."""

    def __init__(self, *, engine: str, with_quant_info: bool, device=None):
        device = device or torch.device("cpu")
        self.lora_execution_engine = engine
        self.base_layer = SimpleNamespace(
            top_k=2,
            num_experts=4,
            w13_weight=torch.zeros(4, 8, 6, device=device),
        )
        if with_quant_info:
            self._quant_info = SimpleNamespace(
                w13_weight=torch.zeros(4, 8, 6, device=device),
                w2_weight=torch.zeros(4, 6, 4, device=device),
            )


class TestGraphBatchMetadata(unittest.TestCase):
    """The two engine-independent buffers must exist for every engine."""

    METADATA_KEYS = ("adapter_enabled", "token_lora_mapping")

    def _backend(self) -> BaseLoRABackend:
        backend = object.__new__(BaseLoRABackend)
        backend._is_moe_lora = True
        return backend

    def test_metadata_present_without_legacy_kernel_buffers(self):
        """Regression: skipping the whole initializer broke graph capture.

        `_add_moe_lora_info` writes into these exact tensors under CUDA graphs,
        so their addresses must stay stable across replays. Skipping their
        allocation raised AttributeError at capture; allocating them fresh per
        step would be worse — the captured graph would keep capture-time
        pointers while replays wrote elsewhere, i.e. silent stale routing.
        """
        backend = self._backend()
        backend.init_cuda_graph_moe_buffers(
            max_bs=8,
            max_loras=4,
            compute_dtype=torch.bfloat16,
            # No _quant_info at all: the new engine never defines it, so the
            # metadata path must not read layer geometry.
            moe_layer=_FakeMoeLayer(engine="sgl_lora", with_quant_info=False),
            include_legacy_kernel_buffers=False,
        )
        for key in self.METADATA_KEYS:
            self.assertIn(key, backend.moe_cg_buffers)
        self.assertEqual(backend.moe_cg_buffers["adapter_enabled"].shape, (4,))
        self.assertEqual(backend.moe_cg_buffers["token_lora_mapping"].shape, (8,))
        # Legacy kernel scratch must be absent so it does not consume budget
        # that the KV cache would otherwise get.
        for key in ("sorted_token_ids_lora", "expert_ids_lora", "cumsum_buffer"):
            self.assertNotIn(key, backend.moe_cg_buffers)

    def test_legacy_path_still_gets_every_buffer(self):
        backend = self._backend()
        backend.init_cuda_graph_moe_buffers(
            max_bs=8,
            max_loras=4,
            compute_dtype=torch.bfloat16,
            moe_layer=_FakeMoeLayer(engine="legacy", with_quant_info=True),
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
        """The actual graph-replay contract, exercised end to end.

        ``prepare_lora_batch`` runs OUTSIDE the captured graph while the runner
        reads these tensors INSIDE it. So an update must write in place: the
        addresses have to survive (or a replay reads capture-time storage while
        the writer fills a different tensor — silent stale routing), and the
        contents have to change (or the update did nothing).
        """
        if not torch.cuda.is_available():
            raise unittest.SkipTest("CUDA required: the metadata update is Triton")
        device = torch.device("cuda")
        backend = self._backend()
        backend.init_cuda_graph_moe_buffers(
            max_bs=8,
            max_loras=4,
            compute_dtype=torch.bfloat16,
            moe_layer=_FakeMoeLayer(
                engine="sgl_lora", with_quant_info=False, device=device
            ),
            include_legacy_kernel_buffers=False,
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
                forward_mode=SimpleNamespace(is_extend=lambda: True),
            )
            backend._add_moe_lora_info(forward_batch, batch_info)
            return batch_info.moe_lora_info

        # Slot 1 active (rank 16), slot 0 disabled (rank 0 = the base slot).
        first = update(weight_indices=[1, 0], lora_ranks=[0, 16], seg_lens=[2, 2])
        first_mapping = first.token_lora_mapping[:4].clone()
        first_enabled = first.adapter_enabled.clone()

        # A different assignment: slot 2 becomes the active one.
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
        # And the semantics the runner depends on: slot 2 active, slot 0 not.
        self.assertEqual(int(second.adapter_enabled[2]), 1)
        self.assertEqual(int(second.adapter_enabled[0]), 0)


class TestEngineAdmission(unittest.TestCase):
    """Admission must reject resident states the engine cannot consume."""

    def _base_layer(self, **overrides):
        from sglang.srt.layers.moe.token_dispatcher.standard import StandardDispatcher
        from sglang.srt.layers.quantization.unquant import UnquantizedFusedMoEMethod

        runner_backend = SimpleNamespace(
            is_deep_gemm=lambda: overrides.get("deep_gemm", True),
        )
        quant_method = object.__new__(UnquantizedFusedMoEMethod)
        quant_method.runner = SimpleNamespace(runner_backend=runner_backend)
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
        from sglang.srt.lora.sgl_lora.moe_lora_runner import SglMoeLoraRunner

        with mock.patch(
            "sglang.srt.layers.deep_gemm_wrapper.ENABLE_JIT_DEEPGEMM",
            overrides.pop("jit_deep_gemm", True),
        ):
            SglMoeLoraRunner._admit(self._base_layer(**overrides))

    def test_admits_a_supported_layer(self):
        self._admit()

    def test_rejects_non_deep_gemm_resident_backend(self):
        """Triton-resident layers were admitted then bound a DeepGEMM provider.

        Regression: the first forward reached an unbound `deep_gemm` symbol.
        Triton is the DEFAULT for unquantized MoE, so this was the common case.
        """
        with self.assertRaisesRegex(NotImplementedError, "deep_gemm"):
            self._admit(deep_gemm=False)

    def test_rejects_unusable_deep_gemm_build(self):
        with self.assertRaisesRegex(NotImplementedError, "JIT DeepGEMM"):
            self._admit(jit_deep_gemm=False)

    def test_rejects_global_expert_id_dispatch(self):
        with self.assertRaisesRegex(NotImplementedError, "EP-local"):
            self._admit(skip_local=True)

    def test_rejects_prefolded_routed_scaling(self):
        """Scaling folded into top-k would be applied twice."""
        with self.assertRaisesRegex(NotImplementedError, "routed scaling"):
            self._admit(fused_scaling=True)

    def test_rejects_unsupported_activation(self):
        with self.assertRaisesRegex(NotImplementedError, "gated SiLU"):
            self._admit(activation="gelu")


if __name__ == "__main__":
    unittest.main()
