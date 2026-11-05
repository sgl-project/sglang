"""CPU fused-checkpoint row preparation, TP slicing and exact rollback."""

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch
from safetensors.torch import save_file

from sglang.srt.eplb.expert_swap import plan_expert_swap
from sglang.srt.eplb.expert_swap_weights import prepare_expert_rows
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class TestExpertSwapWeights(CustomTestCase):
    def test_fused_primary_tensors_are_not_confused_with_mtp(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            w13 = torch.full((8, 16, 4), 2.0, dtype=torch.bfloat16)
            w2 = torch.full((8, 4, 8), 3.0, dtype=torch.bfloat16)
            save_file(
                {
                    "model.language_model.layers.0.mlp.experts.gate_up_proj": w13,
                    "model.language_model.layers.0.mlp.experts.down_proj": w2,
                    "model.language_model.layers.1.mlp.experts.gate_up_proj": w13 + 2,
                    "model.language_model.layers.1.mlp.experts.down_proj": w2 + 2,
                    "mtp.layers.0.mlp.experts.gate_up_proj": w13 + 100,
                    "mtp.layers.0.mlp.experts.down_proj": w2 + 100,
                },
                str(path / "model.safetensors"),
            )
            modules = {
                layer: SimpleNamespace(
                    w13_weight=torch.zeros(4, 4, 4, dtype=torch.bfloat16),
                    w2_weight=torch.zeros(4, 4, 2, dtype=torch.bfloat16),
                    quant_method=SimpleNamespace(w13_swiglu_interleaved=False),
                )
                for layer in (0, 1)
            }
            plan = plan_expert_swap([[0, 1, 2, 3]] * 2, [[0, 4, 2, 3]] * 2, 8)
            with (
                patch(
                    "sglang.srt.eplb.expert_swap_weights.get_free_cpu_memory",
                    return_value=2**30,
                ),
                patch(
                    "sglang.srt.eplb.expert_swap_weights._drop_file_cache_after_load"
                ) as drop,
            ):
                rows, _ = prepare_expert_rows(plan, modules, path, 8, 0, 4, 2**20, 0)
            self.assertTrue(torch.all(rows[0].new_w13 == 2))
            self.assertTrue(torch.all(rows[0].new_w2 == 3))
            self.assertTrue(torch.all(rows[1].new_w13 == 4))
            self.assertTrue(torch.all(rows[1].new_w2 == 5))
            self.assertEqual(drop.call_count, 2)

    def test_prepare_tp_slice_interleaving_and_restore_preserve_pointers(self):
        for interleaved in (False, True):
            with (
                self.subTest(interleaved=interleaved),
                tempfile.TemporaryDirectory() as directory,
            ):
                path = Path(directory)
                w13 = torch.arange(8 * 16 * 4).reshape(8, 16, 4).to(torch.bfloat16)
                w2 = torch.arange(8 * 4 * 8).reshape(8, 4, 8).to(torch.bfloat16)
                save_file(
                    {
                        "model.layers.0.mlp.experts.gate_up_proj": w13,
                        "model.layers.0.mlp.experts.down_proj": w2,
                    },
                    str(path / "model.safetensors"),
                )
                module = SimpleNamespace(
                    w13_weight=torch.zeros(4, 4, 4, dtype=torch.bfloat16),
                    w2_weight=torch.zeros(4, 4, 2, dtype=torch.bfloat16),
                    quant_method=SimpleNamespace(w13_swiglu_interleaved=interleaved),
                )
                module.w13_weight[1].fill_(9)
                module.w2_weight[1].fill_(7)
                before = (module.w13_weight.clone(), module.w2_weight.clone())
                pointers = (module.w13_weight.data_ptr(), module.w2_weight.data_ptr())
                plan = plan_expert_swap([[0, 1, 2, 3]], [[0, 4, 2, 3]], 8)
                with patch(
                    "sglang.srt.eplb.expert_swap_weights.get_free_cpu_memory",
                    return_value=2**30,
                ):
                    rows, node_bytes = prepare_expert_rows(
                        plan, {0: module}, path, 8, 2, 4, 2**20, 0
                    )
                self.assertEqual(node_bytes, 384)
                self.assertTrue(torch.equal(module.w13_weight, before[0]))
                self.assertEqual(len(rows), 1)
                rows[0].copy_new()
                gate, up = w13[4, 4:6], w13[4, 12:14]
                expected = (
                    torch.stack((gate, up), dim=1).flatten(0, 1)
                    if interleaved
                    else torch.cat((gate, up))
                )
                torch.testing.assert_close(
                    module.w13_weight[1], expected, rtol=0, atol=0
                )
                torch.testing.assert_close(
                    module.w2_weight[1], w2[4, :, 4:6], rtol=0, atol=0
                )
                torch.testing.assert_close(
                    module.w13_weight[0], before[0][0], rtol=0, atol=0
                )
                rows[0].restore()
                torch.testing.assert_close(module.w13_weight, before[0], rtol=0, atol=0)
                torch.testing.assert_close(module.w2_weight, before[1], rtol=0, atol=0)
                self.assertEqual(
                    (module.w13_weight.data_ptr(), module.w2_weight.data_ptr()),
                    pointers,
                )

    def test_budget_counts_all_ranks_before_reading_checkpoint(self):
        module = SimpleNamespace(
            w13_weight=torch.zeros(4, 4, 4, dtype=torch.bfloat16),
            w2_weight=torch.zeros(4, 4, 2, dtype=torch.bfloat16),
        )
        plan = plan_expert_swap([[0, 1, 2, 3]], [[0, 4, 2, 3]], 8)
        with patch(
            "sglang.srt.eplb.expert_swap_weights.get_free_cpu_memory",
            return_value=2**30,
        ):
            with self.assertRaises(MemoryError):
                prepare_expert_rows(
                    plan, {0: module}, "must-not-be-opened", 8, 0, 4, 383, 0
                )
        with patch(
            "sglang.srt.eplb.expert_swap_weights.get_free_cpu_memory", return_value=400
        ):
            with self.assertRaises(MemoryError):
                prepare_expert_rows(
                    plan, {0: module}, "must-not-be-opened", 8, 0, 4, 4096, 17
                )


if __name__ == "__main__":
    unittest.main()
