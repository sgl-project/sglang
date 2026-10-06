"""Model MLP weight layouts stay fixed across construction and reload scopes."""

import importlib
import inspect
import unittest
from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch
import torch.nn.functional as F

from sglang.srt.runtime_context import SpawnRanks, reset_context
from sglang.srt.server_args import ServerArgs
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.parallel_groups import parallel_scope, publish, rank_size
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=15, suite="base-a-test-cpu")

MODELS = {
    "bailing_moe": "BailingMoEMLP",
    "bailing_moe_v3": "BailingMLP",
    "deepseek_v2": "DeepseekV2MLP",
    "dots3_common.modeling": "Dots3MLP",
    "glm4_moe": "Glm4MoeMLP",
    "glm4_moe_lite": "Glm4MoeLiteMLP",
    "llada2": "LLaDA2MoeMLP",
    "mimo_v2": "MiMoV2MLP",
    "minimax_m3": "MiniMaxM3MLP",
    "sarvam_moe": "SarvamMoEMLP",
    "xllm": "XllmMLP",
}


class TestMLPParallelGroups(CustomTestCase):
    def setUp(self):
        reset_context()
        self.addCleanup(reset_context)
        publish(
            ServerArgs(model_path="dummy", device="cpu", tp_size=4, attn_dp_size=2),
            role="test",
            ranks=SpawnRanks(world_rank=3),
        )
        self.x = torch.arange(16, dtype=torch.float32).reshape(2, 8) / 16

    def test_model_layout_math_reload_and_row_policy(self):
        tp = Mock()
        tp.all_reduce.side_effect = lambda x: x * 2
        for module_name, class_name in MODELS.items():
            cls = getattr(
                importlib.import_module("sglang.srt.models." + module_name), class_name
            )
            for group, rank, size in ((None, 3, 4), ("tp", 3, 4), ("replicated", 0, 1)):
                for reduce in (False, True):
                    for padded in (
                        (False, True) if class_name == "BailingMLP" else (False,)
                    ):
                        with self.subTest(
                            model=module_name, group=group, reduce=reduce, padded=padded
                        ):
                            config = SimpleNamespace(
                                hidden_size=8, hidden_act="silu", use_bias=True
                            )
                            options = dict(
                                hidden_size=8,
                                intermediate_size=8,
                                hidden_act="silu",
                                config=config,
                                parallel_group=group,
                                reduce_results=reduce,
                                padded_intermediate_size=16 if padded else None,
                            )
                            if group is None:
                                options.pop("parallel_group")
                            signature = inspect.signature(cls).parameters
                            mlp = cls(
                                **{k: v for k, v in options.items() if k in signature}
                            )
                            mlp.act_fn._forward_method = mlp.act_fn.forward_native
                            up, down = mlp.gate_up_proj, mlp.down_proj
                            width = 16 if padded else 8
                            gate = (
                                torch.arange(width * 8, dtype=torch.float32).reshape(
                                    width, 8
                                )
                                / 128
                            )
                            other = gate + 0.25
                            weight = gate.T.contiguous() + 0.5
                            with parallel_scope(
                                tp_rank=0, attn_dp_rank=0, attn_tp_rank=0
                            ):
                                up.weight.weight_loader(up.weight, gate, 0)
                                up.weight.weight_loader(up.weight, other, 1)
                                down.weight.weight_loader(down.weight, weight)
                                if up.bias is not None:
                                    up.bias.data.zero_()
                                    down.bias.data.zero_()
                            local_gate = gate.chunk(size)[rank]
                            local_up = other.chunk(size)[rank]
                            projected = F.silu(F.linear(self.x, local_gate)) * F.linear(
                                self.x, local_up
                            )
                            if padded:
                                projected = F.pad(
                                    projected[:, : 8 // size], (0, 8 // size)
                                )
                            expected = F.linear(
                                projected, weight.chunk(size, dim=1)[rank]
                            )
                            if reduce and size > 1:
                                expected = expected * 2
                            tp.all_reduce.reset_mock()
                            with (
                                parallel_scope(tp_group=tp),
                                patch(
                                    "sglang.srt.layers.linear.is_allocation_symmetric",
                                    return_value=True,
                                ),
                                patch(
                                    "sglang.srt.layers.linear.use_symmetric_memory",
                                    return_value=nullcontext(),
                                ) as allocator,
                            ):
                                torch.testing.assert_close(mlp(self.x), expected)
                                allocator.assert_called_once_with(tp, disabled=False)
                                self.assertEqual(
                                    tp.all_reduce.call_count,
                                    1 if reduce and size > 1 else 0,
                                )
                                torch.testing.assert_close(mlp(self.x[:0]), self.x[:0])
                            self.assertEqual(rank_size(up), (rank, size))
                            self.assertEqual(rank_size(down), (rank, size))
                            self.assertEqual(down.reduce_results, reduce)
                            self.assertFalse(down.use_dp_attention_reduce)


if __name__ == "__main__":
    unittest.main()
