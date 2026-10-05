"""Dense MLP placement stays frozen while reduction uses its own policy."""

import unittest
from contextlib import nullcontext
from unittest.mock import Mock, patch

import torch
import torch.nn.functional as F

from sglang.srt.models.inkling_common import dense_mlp
from sglang.srt.models.llama import LlamaMLP
from sglang.srt.runtime_context import SpawnRanks, reset_context
from sglang.srt.server_args import ServerArgs
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.parallel_groups import parallel_scope, publish
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=10, suite="base-a-test-cpu")


def build_mlp(
    model,
    group=None,
    *,
    width=8,
    reduce=True,
    dp_reduce=False,
    quant_config=None,
    fused=False,
    lora=False,
    execution_group=None,
):
    cls = LlamaMLP if model == "llama" else dense_mlp.InklingDenseMLP
    kwargs = dict(
        hidden_size=width,
        intermediate_size=width,
        quant_config=quant_config,
        use_dp_attention_reduce=dp_reduce,
    )
    if model == "llama":
        kwargs.update(hidden_act="silu", reduce_results=reduce)
    else:
        kwargs.update(
            use_global_scale=True, layer_id=0, fused=fused, tp_group=execution_group
        )
    if group is not None:
        kwargs["parallel_group"] = group
    with patch.object(dense_mlp, "lora_compatible_layout_enabled", return_value=lora):
        module = cls(**kwargs)
    return module


def load_projection(layer):
    row = hasattr(layer, "reduce_results")
    shape = (layer.output_size, layer.input_size)
    weight = (
        torch.arange(shape[0] * shape[1], device=layer.weight.device).reshape(shape)
        % 23
        - 11
    ).to(layer.weight.dtype) / 64
    layer.weight.weight_loader(layer.weight, weight)
    if row:
        expected = weight.chunk(layer.tp_size, dim=1)[layer.tp_rank]
    else:
        expected = torch.cat(
            [
                part.chunk(layer.tp_size)[layer.tp_rank]
                for part in weight.split(layer.output_sizes)
            ]
        )
    torch.testing.assert_close(layer.weight, expected)
    return expected


class TestLlamaInklingParallelGroups(CustomTestCase):
    def setUp(self):
        reset_context()
        self.addCleanup(reset_context)
        publish(
            ServerArgs(model_path="dummy", device="cpu", tp_size=4, attn_dp_size=2),
            role="test",
            ranks=SpawnRanks(world_rank=3),
        )

    def test_frozen_loaders_forward_and_execution_policy(self):
        tp, attn, execution = Mock(), Mock(), Mock()
        tp.all_reduce.side_effect = lambda x: x * 2
        attn.all_reduce.side_effect = lambda x: x * 3
        x = torch.arange(16, dtype=torch.float32).reshape(2, 8) / 32
        for model in ("llama", "inkling"):
            for group in (None, "tp", "attn_tp", "replicated"):
                for reduce in (False, True):
                    for dp_reduce in (False, True):
                        with self.subTest(
                            model=model, group=group, reduce=reduce, dp_reduce=dp_reduce
                        ):
                            module = build_mlp(
                                model,
                                group,
                                reduce=reduce,
                                dp_reduce=dp_reduce,
                                execution_group=execution,
                            )
                            up, down = module.gate_up_proj, module.down_proj
                            expected_group = group or (
                                "tp" if model == "llama" else "replicated"
                            )
                            rank, size = {
                                "tp": (3, 4),
                                "attn_tp": (1, 2),
                                "replicated": (0, 1),
                            }[expected_group]
                            with parallel_scope(
                                tp_rank=0, attn_dp_rank=0, attn_tp_rank=0
                            ):
                                a, b = load_projection(up), load_projection(down)
                            gate, other = F.linear(x, a).chunk(2, dim=-1)
                            expected = F.linear(F.silu(gate) * other, b)
                            row_reduce = reduce and model == "llama"
                            if row_reduce and size > 1:
                                expected *= 3 if dp_reduce else 2
                            if model == "llama":
                                module.act_fn._forward_method = (
                                    module.act_fn.forward_native
                                )
                            else:
                                module.global_scale.data.fill_(2)
                                expected *= 2 * 5
                                module.act_fn.forward = dense_mlp.swiglu_contiguous._torchdynamo_orig_callable
                            tp.all_reduce.reset_mock()
                            attn.all_reduce.reset_mock()
                            with (
                                parallel_scope(tp_group=tp, attn_tp_group=attn),
                                patch(
                                    "sglang.srt.layers.linear.is_allocation_symmetric",
                                    return_value=False,
                                ),
                                patch(
                                    "sglang.srt.layers.linear.use_symmetric_memory",
                                    return_value=nullcontext(),
                                ) as allocator,
                                patch.object(
                                    dense_mlp,
                                    "symm_mem_all_reduce",
                                    side_effect=lambda y, g: y * 5,
                                ) as comm,
                            ):
                                actual = module(x)
                                torch.testing.assert_close(actual, expected)
                                allocator.assert_called_once_with(
                                    attn
                                ) if dp_reduce else allocator.assert_called_once_with(
                                    tp, disabled=True
                                )
                                self.assertEqual(
                                    tp.all_reduce.call_count,
                                    int(row_reduce and size > 1 and not dp_reduce),
                                )
                                self.assertEqual(
                                    attn.all_reduce.call_count,
                                    int(row_reduce and size > 1 and dp_reduce),
                                )
                                if model == "inkling":
                                    comm.assert_called_once_with(
                                        unittest.mock.ANY, execution
                                    )
                                else:
                                    comm.assert_not_called()
                            self.assertEqual((up.tp_rank, up.tp_size), (rank, size))
                            self.assertEqual((down.tp_rank, down.tp_size), (rank, size))
                            self.assertEqual(down.reduce_results, row_reduce)
                            self.assertEqual(down.use_dp_attention_reduce, dp_reduce)

    def test_inkling_lora_layout_and_scattered_execution(self):
        x = torch.arange(16, dtype=torch.float32).reshape(2, 8) / 32
        execution = Mock()
        for lora in (False, True):
            module = build_mlp(
                "inkling", "attn_tp", fused=True, lora=lora, execution_group=execution
            )
            self.assertEqual(module.act_fn.interleaved, not lora)
            up, down = module.gate_up_proj, module.down_proj
            a, b = load_projection(up), load_projection(down)
            z = F.linear(x, a)
            gate, other = z.chunk(2, dim=-1) if lora else (z[..., ::2], z[..., 1::2])
            expected = F.linear(F.silu(gate) * other, b)
            module.global_scale.data.fill_(1)
            function = dense_mlp.swiglu_contiguous if lora else dense_mlp.swiglu
            module.act_fn.forward = function._torchdynamo_orig_callable
            module.scattered_sconv = True
            with (
                parallel_scope(tp_group=Mock()),
                patch(
                    "sglang.srt.layers.linear.use_symmetric_memory",
                    return_value=nullcontext(),
                ),
                patch.object(
                    dense_mlp, "reduce_scatter_hidden", side_effect=lambda y, g: y
                ) as scatter,
            ):
                torch.testing.assert_close(module(x), expected)
                scatter.assert_called_once_with(unittest.mock.ANY, execution)
                scatter.reset_mock()
                torch.testing.assert_close(module(x, use_reduce_scatter=True), expected)
                scatter.assert_not_called()


if __name__ == "__main__":
    unittest.main()
