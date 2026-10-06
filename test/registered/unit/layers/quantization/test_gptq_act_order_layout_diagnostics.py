"""XPU GPTQ act-order diagnostics describe the constructed weight owner."""

import unittest
from contextlib import nullcontext

import torch

from sglang.srt.layers.linear import (
    ColumnParallelLinear,
    ReplicatedLinear,
    RowParallelLinear,
)
from sglang.srt.layers.quantization.gptq.gptq import GPTQConfig
from sglang.srt.layers.quantization.gptq.schemes.gptq_linear import GPTQXPULinearScheme
from sglang.srt.runtime_context import SpawnRanks, reset_context
from sglang.srt.server_args import ServerArgs
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.parallel_groups import parallel_scope, publish, rank_size
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=8, stage="base-b", runner_config="1-gpu-small")


def loading_scope(changed):
    if not changed:
        return nullcontext()
    return parallel_scope(
        tp_size=1,
        tp_rank=0,
        tp_group=None,
        attn_tp_size=1,
        attn_tp_rank=0,
        attn_tp_group=None,
        attn_dp_size=1,
        attn_dp_rank=0,
        attn_cp_size=1,
        attn_cp_rank=0,
        moe_tp_size=1,
        moe_tp_rank=0,
        moe_ep_size=1,
        moe_ep_rank=0,
        moe_ep_group=None,
        moe_dp_size=1,
        moe_dp_rank=0,
    )


def build_owner(layout, group, group_size=32, checkpoint_format="", device="cpu"):
    with torch.device(device):
        kwargs = dict(bias=False, params_dtype=torch.bfloat16, parallel_group=group)
        if layout == "row":
            layer = RowParallelLinear(1024, 256, **kwargs)
        elif layout == "column":
            layer = ColumnParallelLinear(1024, 256, **kwargs)
        else:
            kwargs.pop("parallel_group")
            layer = ReplicatedLinear(1024, 256, **kwargs)
        n, k = layer.weight.shape
        del layer.weight
        config = GPTQConfig(
            weight_bits=4,
            group_size=group_size,
            desc_act=True,
            lm_head_quantized=False,
            dynamic={},
            checkpoint_format=checkpoint_format,
        )
        scheme = GPTQXPULinearScheme(config)
        scheme.create_weights(
            layer,
            input_size_per_partition=k,
            output_partition_sizes=[n],
            input_size=1024,
            params_dtype=torch.bfloat16,
            weight_loader=layer.weight_loader,
        )
        full = {
            "qweight": torch.zeros(1024 // 8, 256, dtype=torch.int32),
            "qzeros": torch.zeros(1024 // group_size, 256 // 8, dtype=torch.int32),
            "scales": torch.ones(1024 // group_size, 256, dtype=torch.bfloat16),
            "g_idx": torch.arange(1024, dtype=torch.int32) // group_size,
        }
        full["g_idx"][::group_size] += 1
        for name, value in full.items():
            layer.weight_loader(getattr(layer, name), value)
    return layer, scheme


def diagnostic(layer, scheme, changed=False):
    before = {n: t.detach().clone() for n, t in layer.named_parameters()}
    with loading_scope(changed):
        try:
            scheme.process_weights_after_loading(layer)
        except NotImplementedError as error:
            message = str(error)
        else:
            raise AssertionError("Expected the existing split act-order group guard")
    assert "splits a group across the K boundary" in message
    for name, value in before.items():
        torch.testing.assert_close(getattr(layer, name), value, rtol=0, atol=0)
    return message


@unittest.skipUnless(torch.cuda.is_available(), "needs the native SGLang GPU runtime")
class TestGptqActOrderLayoutDiagnostics(CustomTestCase):
    def setUp(self):
        reset_context()
        self.addCleanup(reset_context)
        publish(
            ServerArgs(model_path="dummy", device="cuda", tp_size=4, attn_dp_size=2),
            role="test",
            ranks=SpawnRanks(world_rank=3),
        )

    def test_diagnostic_survives_scope_exit(self):
        for layout in ("row", "column"):
            for group in ("tp", "attn_tp", "replicated"):
                with self.subTest(layout=layout, group=group):
                    layer, scheme = build_owner(layout, group)
                    _, expected = rank_size(layer)
                    for changed in (False, True):
                        message = diagnostic(layer, scheme, changed)
                        if expected > 1:
                            self.assertIn(f"Got tp_size={expected}", message)
                        else:
                            self.assertNotIn("--tp-size", message)

    def test_replicated_owner_has_no_tp_hint(self):
        layer, scheme = build_owner("replicated", "replicated")
        for changed in (False, True):
            self.assertNotIn("--tp-size", diagnostic(layer, scheme, changed))

    def test_all_group_sizes_and_checkpoint_zero_formats_keep_guard(self):
        for size in (32, 64, 128, 256):
            for checkpoint_format in ("", "gptq_v2"):
                layer, scheme = build_owner(
                    "column", "attn_tp", size, checkpoint_format
                )
                self.assertIn("Got tp_size=2", diagnostic(layer, scheme, True))


if __name__ == "__main__":
    unittest.main()
