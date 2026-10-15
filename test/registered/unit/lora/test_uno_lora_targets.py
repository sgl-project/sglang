"""Target-layer validation for UNO's specialized LoRA backend."""

import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock

import torch

from sglang.srt.layers.linear import (
    ColumnParallelLinear,
    ReplicatedLinear,
    RowParallelLinear,
)
from sglang.srt.layers.moe.fused_moe_triton.layer import FusedMoE
from sglang.srt.lora.backend.uno_cublas_backend import UnoCublasLoRABackend
from sglang.srt.lora.lora_manager import LoRAManager
from sglang.srt.runtime_context import get_context, get_parallel
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=8, suite="base-a-test-cpu")


class TestUnoLoRATargets(CustomTestCase):
    def setUp(self):
        self.backend = UnoCublasLoRABackend.__new__(UnoCublasLoRABackend)
        self.backend._pending_lora_a = None
        self.backend._use_cublas_lora_b = False

    @staticmethod
    def _model(modules, **attributes):
        return SimpleNamespace(
            named_modules=lambda: modules,
            **attributes,
        )

    def test_supported_decoder_targets_are_accepted(self):
        modules = [
            (
                "model.layers.0.qkv_proj",
                ColumnParallelLinear.__new__(ColumnParallelLinear),
            ),
            (
                "model.layers.0.o_proj",
                RowParallelLinear.__new__(RowParallelLinear),
            ),
            (
                "model.layers.0.fused_qkv_a_proj_with_mqa",
                ReplicatedLinear.__new__(ReplicatedLinear),
            ),
        ]
        self.backend.validate_lora_targets(
            base_model=self._model(modules),
            target_modules={
                "qkv_proj",
                "o_proj",
                "fused_qkv_a_proj_with_mqa",
            },
        )

    def test_unsupported_targets_are_rejected(self):
        cases = {
            "unknown decoder layer": (
                self._model(
                    [
                        (
                            "model.layers.0.custom_proj",
                            torch.nn.Linear(2, 2),
                        )
                    ]
                ),
                {"custom_proj"},
                "Linear",
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

    def test_manager_rejects_uno_with_dp_attention(self):
        with (
            get_context().override_server_args(
                tp_size=2, attn_dp_size=2, enable_lora_overlap_loading=False
            ) as args,
            get_parallel().override(attn_tp_size=1),
            self.assertRaisesRegex(
                ValueError, "uno_cublas.*does not support DP attention"
            ),
        ):
            LoRAManager(
                base_model=torch.nn.Linear(2, 2),
                base_hf_config=SimpleNamespace(),
                max_loras_per_batch=2,
                load_config=None,
                dtype=torch.float32,
                server_args=args,
                lora_backend="uno_cublas",
            )


if __name__ == "__main__":
    unittest.main()
