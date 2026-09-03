"""Target-layer validation for UNO's specialized LoRA backend."""

import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import torch

from sglang.srt.layers.linear import (
    ColumnParallelLinear,
    ReplicatedLinear,
    RowParallelLinear,
)
from sglang.srt.layers.moe.fused_moe_triton.layer import FusedMoE
from sglang.srt.lora.backend.triton_backend import TritonLoRABackend
from sglang.srt.lora.backend.uno_cublas_backend import UnoCublasLoRABackend
from sglang.srt.lora.lora_manager import LoRAManager
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


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

    def test_nonoverlap_dense_calls_fall_back_to_triton(self):
        x = object()
        weights = object()
        hidden = object()
        base_output = object()
        pruned_batch_info = object()
        expected = object()

        with (
            patch.object(
                TritonLoRABackend,
                "run_lora_a_sgemm",
                return_value=hidden,
            ) as run_lora_a,
            patch.object(
                TritonLoRABackend,
                "run_lora_b_sgemm",
                return_value=expected,
            ) as run_lora_b,
        ):
            actual_hidden = self.backend.run_lora_a_sgemm(
                x,
                weights,
                pruned_batch_info=pruned_batch_info,
            )
            actual = self.backend.run_lora_b_sgemm(
                actual_hidden,
                weights,
                base_output=base_output,
                pruned_batch_info=pruned_batch_info,
            )

        self.assertIs(actual_hidden, hidden)
        self.assertIs(actual, expected)
        run_lora_a.assert_called_once_with(
            x,
            weights,
            pruned_batch_info,
            1,
        )
        run_lora_b.assert_called_once_with(
            hidden,
            weights,
            base_output,
            pruned_batch_info,
        )

    def test_overlap_launch_selects_cublas(self):
        pending = object()
        x = object()
        weights = object()
        hidden = object()
        base_output = object()
        expected = object()
        self.backend._pending_lora_a = pending
        self.backend._consume_lora_a_overlap = MagicMock(return_value=hidden)
        self.backend._run_lora_b = MagicMock(return_value=expected)

        with (
            patch.object(TritonLoRABackend, "run_lora_a_sgemm") as run_lora_a,
            patch.object(TritonLoRABackend, "run_lora_b_sgemm") as run_lora_b,
        ):
            actual_hidden = self.backend.run_lora_a_sgemm(x, weights)
            actual = self.backend.run_lora_b_sgemm(
                actual_hidden,
                weights,
                base_output=base_output,
            )

        self.assertIs(actual_hidden, hidden)
        self.assertIs(actual, expected)
        self.backend._consume_lora_a_overlap.assert_called_once_with(pending)
        self.backend._run_lora_b.assert_called_once_with(
            hidden,
            weights,
            base_output,
        )
        self.assertFalse(self.backend._use_cublas_lora_b)
        run_lora_a.assert_not_called()
        run_lora_b.assert_not_called()

    def test_nonoverlap_qkv_call_falls_back_to_triton(self):
        expected = object()
        args = {
            "x": object(),
            "qkv_lora_a": object(),
            "qkv_lora_b": object(),
            "output_offset": object(),
            "output_offset_cpu": object(),
            "max_qkv_out_dim": 128,
            "base_output": object(),
            "n_slices": 2,
        }

        with patch.object(
            TritonLoRABackend,
            "run_qkv_lora",
            return_value=expected,
        ) as run_qkv_lora:
            actual = self.backend.run_qkv_lora(**args)

        self.assertIs(actual, expected)
        run_qkv_lora.assert_called_once_with(
            args["x"],
            args["qkv_lora_a"],
            args["qkv_lora_b"],
            args["output_offset"],
            128,
            args["base_output"],
            2,
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
