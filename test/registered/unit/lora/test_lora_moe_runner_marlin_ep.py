"""CPU contracts for MXFP4 Marlin LoRA quant info and lazy EP alignment."""

import unittest
from types import SimpleNamespace
from unittest import mock

import torch

from sglang.srt.layers.quantization.mxfp4_marlin_moe import Mxfp4MarlinMoEMethod
from sglang.srt.lora import lora_moe_runner_marlin as runner_module
from sglang.srt.lora.lora_moe_runner_marlin import MarlinLoraRunnerCore
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class TestMxfp4MarlinLoRAQuantInfo(CustomTestCase):
    def test_quant_info_reads_the_prepared_marlin_experts(self):
        expert_map = torch.tensor([0, 1, -1, -1], dtype=torch.int32)
        layer = SimpleNamespace(
            w13_weight=torch.zeros(2, 4, 8, dtype=torch.int32),
            w2_weight=torch.zeros(2, 2, 8, dtype=torch.int32),
            w13_weight_scale=torch.zeros(2, 2, 16, dtype=torch.float8_e8m0fnu),
            w2_weight_scale=torch.zeros(2, 1, 16, dtype=torch.float8_e8m0fnu),
            dispatcher=SimpleNamespace(local_expert_mapping=expert_map, num_experts=4),
        )

        info = Mxfp4MarlinMoEMethod(None, "experts").get_marlin_quant_info(layer)

        self.assertIs(info.w13_qweight, layer.w13_weight)
        self.assertIs(info.w2_qweight, layer.w2_weight)
        self.assertIs(info.w13_scales, layer.w13_weight_scale)
        self.assertIs(info.w2_scales, layer.w2_weight_scale)
        self.assertEqual(info.weight_bits, 4)
        self.assertIs(info.expert_map, expert_map)
        self.assertEqual(info.global_num_experts, 4)


class TestMarlinLoRARunnerEpAlignment(CustomTestCase):
    def _aligned_num_experts(self, expert_map):
        config = SimpleNamespace(
            is_gated=True, routed_scaling_factor=None, num_experts=8
        )
        # Quant info is cached before the dispatcher creates the EP map.
        quant_info = SimpleNamespace(
            w13_qweight=torch.zeros(2, 1, 1),
            w2_qweight=torch.zeros(2, 1, 1),
            weight_bits=4,
            w13_scales=None,
            w2_scales=None,
            w13_qzeros=None,
            w2_qzeros=None,
            expert_map=expert_map,
            global_num_experts=-1,
        )
        dispatch_output = SimpleNamespace(
            hidden_states=torch.zeros(3, 16),
            topk_output=SimpleNamespace(
                topk_weights=torch.ones(3, 2),
                topk_ids=torch.zeros(3, 2, dtype=torch.int32),
            ),
        )
        with (
            mock.patch("torch.cuda.get_device_capability", return_value=(9, 0)),
            mock.patch.multiple(
                runner_module,
                create=True,
                marlin_moe_use_atomic_add=lambda *_: False,
                is_mxfp4_marlin_weights=lambda *_: True,
            ),
            mock.patch.object(
                runner_module,
                "moe_align_block_size",
                create=True,
                side_effect=RuntimeError("alignment reached"),
            ) as align,
        ):
            with self.assertRaisesRegex(RuntimeError, "alignment reached"):
                MarlinLoraRunnerCore(config).run_from_dispatch(
                    dispatch_output, quant_info, config, hooks=SimpleNamespace()
                )
        align.assert_called_once()
        return align.call_args.args[2]

    def test_ep_aligns_over_the_global_expert_count(self):
        expert_map = torch.tensor([0, 1] + [-1] * 6, dtype=torch.int32)
        self.assertEqual(self._aligned_num_experts(expert_map), 8)

    def test_without_ep_aligns_over_the_local_experts(self):
        self.assertEqual(self._aligned_num_experts(None), 2)


if __name__ == "__main__":
    unittest.main()
