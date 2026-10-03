import unittest
from types import SimpleNamespace
from unittest import mock

import torch

from sglang.kernels.ops.attention import kimi_attn_residual_gluon_hip as adapter
from sglang.srt.layers import attn_residual as native


def make_inputs(rows=1, valid_rows=4):
    device = torch.device("meta")
    return dict(
        prefix=torch.empty((rows, 7168), dtype=torch.bfloat16, device=device),
        addend=torch.empty((rows, 7168), dtype=torch.bfloat16, device=device),
        bank=torch.empty((rows, 8, 7168), dtype=torch.bfloat16, device=device),
        valid_rows=valid_rows,
        score_proj=SimpleNamespace(
            weight=torch.empty((1, 7168), dtype=torch.bfloat16, device=device)
        ),
        score_norm=SimpleNamespace(
            weight=torch.empty((7168,), dtype=torch.bfloat16, device=device),
            variance_epsilon=1e-5,
        ),
        out_norm=SimpleNamespace(
            weight=torch.empty((7168,), dtype=torch.bfloat16, device=device),
            variance_epsilon=1e-5,
        ),
    )


class TestKimiAttnResidual(unittest.TestCase):
    def test_compact_schema_covers_exactly_177_profiles(self):
        count = sum(
            adapter.entrypoint_name(rows, banks, mode) is not None
            for rows in (1, 2, 4, 8, 16, 32, 64, 128, 256)
            for banks in range(1, 9)
            for mode in range(8)
        )
        self.assertEqual(count, 177)
        self.assertEqual(
            adapter.entrypoint_name(1, 4, 5),
            "attention_residual_norm_m1_2_banks4_modes5",
        )
        self.assertEqual(
            adapter.entrypoint_name(64, 8, 5),
            "attention_residual_norm_m64_banks8_modes5",
        )
        self.assertEqual(
            adapter.entrypoint_name(256, 3, 6),
            "attention_residual_norm_m32_64_128_256_banks1_8_modes4_6",
        )
        self.assertIsNone(adapter.entrypoint_name(1, 2, 5))
        self.assertIsNone(adapter.entrypoint_name(256, 8, 6))

    def test_runtime_contract_is_fail_closed(self):
        inputs = make_inputs()
        self.assertEqual(
            adapter.covered(**inputs, write_bank=False),
            "attention_residual_norm_m1_2_banks4_modes5",
        )
        bad_norm = SimpleNamespace(
            weight=inputs["out_norm"].weight, variance_epsilon=1e-6
        )
        self.assertIsNone(
            adapter.covered(**{**inputs, "out_norm": bad_norm}, write_bank=False)
        )
        self.assertEqual(
            adapter.covered(**{**inputs, "addend": None}, write_bank=True),
            "attention_residual_norm_m1_banks1_4_8_modes4_6",
        )
        self.assertIsNone(adapter.covered(**inputs, write_bank=True))

    def test_install_preserves_fallback_and_propagates_failure(self):
        inputs = make_inputs()
        original = mock.Mock(return_value=("native-out", "native-current"))
        adapter._installed = False
        with (
            mock.patch.object(adapter, "qualified_model", return_value=True),
            mock.patch.object(adapter, "rank0_log"),
            mock.patch.object(native, "_aggregate_hip", original),
        ):
            self.assertTrue(adapter.install(object()))
            installed = native._aggregate_hip

            unsupported = make_inputs(3)
            self.assertEqual(
                installed(**unsupported, write_bank_row=False),
                ("native-out", "native-current"),
            )
            original.assert_called_once()

            with mock.patch.object(adapter, "run", side_effect=RuntimeError("launch")):
                with self.assertRaisesRegex(RuntimeError, "launch"):
                    installed(**inputs, write_bank_row=False)
            self.assertEqual(original.call_count, 1)
        adapter._installed = False

    def test_installed_kernel_validates_output_abi(self):
        inputs = make_inputs()
        original = mock.Mock()
        adapter._installed = False
        with (
            mock.patch.object(adapter, "qualified_model", return_value=True),
            mock.patch.object(adapter, "rank0_log"),
            mock.patch.object(native, "_aggregate_hip", original),
        ):
            adapter.install(object())
            installed = native._aggregate_hip
            output = torch.empty_like(inputs["prefix"])
            current = torch.empty_like(inputs["prefix"])
            with mock.patch.object(
                adapter, "run", return_value=(output, current, inputs["bank"])
            ):
                self.assertEqual(
                    installed(**inputs, write_bank_row=False), (output, current)
                )
            original.assert_not_called()

            wrong = torch.empty((1, 7168), dtype=torch.float32, device="meta")
            with (
                mock.patch.object(
                    adapter, "run", return_value=(wrong, current, inputs["bank"])
                ),
                self.assertRaisesRegex(RuntimeError, "output ABI"),
            ):
                installed(**inputs, write_bank_row=False)
        adapter._installed = False


if __name__ == "__main__":
    unittest.main()
