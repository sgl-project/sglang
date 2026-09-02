"""Target-layer validation for UNO's specialized LoRA backend."""

import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock

from sglang.srt.layers.linear import (
    ColumnParallelLinear,
    ReplicatedLinear,
    RowParallelLinear,
)
from sglang.srt.layers.moe.fused_moe_triton.layer import FusedMoE
from sglang.srt.layers.vocab_parallel_embedding import (
    ParallelLMHead,
    VocabParallelEmbedding,
)
from sglang.srt.lora.backend.uno_cublas_backend import UnoCublasLoRABackend
from sglang.srt.lora.lora_manager import LoRAManager
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


class TestUnoLoRATargets(CustomTestCase):
    def setUp(self):
        self.backend = UnoCublasLoRABackend.__new__(UnoCublasLoRABackend)

    @staticmethod
    def _model(modules, **attributes):
        return SimpleNamespace(
            named_modules=lambda: modules,
            **attributes,
        )

    def test_supported_parallel_linear_targets_are_accepted(self):
        modules = [
            (
                "model.layers.0.qkv_proj",
                ColumnParallelLinear.__new__(ColumnParallelLinear),
            ),
            (
                "model.layers.0.o_proj",
                RowParallelLinear.__new__(RowParallelLinear),
            ),
        ]
        self.backend.validate_lora_targets(
            base_model=self._model(modules),
            target_modules={"qkv_proj", "o_proj"},
        )

    def test_unsupported_targets_are_rejected(self):
        cases = {
            "lm_head": (
                self._model([], lm_head=ParallelLMHead.__new__(ParallelLMHead)),
                {"lm_head"},
                "ParallelLMHead",
            ),
            "embedding": (
                self._model(
                    [
                        (
                            "model.embed_tokens",
                            VocabParallelEmbedding.__new__(VocabParallelEmbedding),
                        )
                    ]
                ),
                {"embed_tokens"},
                "embed_tokens",
            ),
            "replicated linear": (
                self._model(
                    [
                        (
                            "model.layers.0.fused_qkv_a_proj_with_mqa",
                            ReplicatedLinear.__new__(ReplicatedLinear),
                        )
                    ]
                ),
                {"fused_qkv_a_proj_with_mqa"},
                "ReplicatedLinear",
            ),
            "fused MoE": (
                self._model(
                    [
                        (
                            "model.layers.0.mlp",
                            FusedMoE.__new__(FusedMoE),
                        )
                    ]
                ),
                {"gate_up_proj", "down_proj"},
                "FusedMoE",
            ),
        }

        for name, (model, targets, expected) in cases.items():
            with self.subTest(name=name), self.assertRaisesRegex(ValueError, expected):
                self.backend.validate_lora_targets(
                    base_model=model,
                    target_modules=targets,
                )

    def test_manager_preflights_targets_before_wrapping(self):
        manager = LoRAManager.__new__(LoRAManager)
        manager.base_model = object()
        manager.lora_backend = MagicMock()
        manager._experts_shared_outer_override = None
        manager.init_lora_adapters = MagicMock()
        manager.init_lora_shapes = MagicMock(
            side_effect=lambda **_: setattr(manager, "target_modules", {"qkv_proj"})
        )
        manager._detect_shared_outer_loras = MagicMock(return_value=False)
        manager.init_lora_modules = MagicMock()
        manager.init_memory_pool = MagicMock()
        manager.update_lora_info = MagicMock()
        manager.lora_backend.validate_lora_targets.side_effect = ValueError(
            "unsupported target"
        )

        with self.assertRaisesRegex(ValueError, "unsupported target"):
            manager.init_state(max_lora_rank=1, target_modules={"q_proj"})

        manager.lora_backend.validate_lora_targets.assert_called_once_with(
            base_model=manager.base_model,
            target_modules={"qkv_proj"},
        )
        manager.init_lora_modules.assert_not_called()


if __name__ == "__main__":
    unittest.main()
