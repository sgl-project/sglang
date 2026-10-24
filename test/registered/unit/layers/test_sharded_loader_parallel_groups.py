"""Auxiliary checkpoint loaders retain their constructed owner layout."""

import unittest
from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch

from sglang.srt.layers.linear import ReplicatedParallelGroup
from sglang.srt.model_loader import weight_utils
from sglang.srt.runtime_context import SpawnRanks, get_parallel, reset_context
from sglang.srt.server_args import ServerArgs
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.parallel_groups import parallel_scope, publish
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=15, stage="nightly", runner_config="1-gpu-large")


def values(shape, *, device="cpu", dtype=torch.float32, offset=0):
    count = 1
    for size in shape:
        count *= size
    return (
        (((torch.arange(count, device=device) + offset) % 23 - 11) / 128)
        .reshape(shape)
        .to(dtype)
    )


def loading_scope(changed):
    return (
        parallel_scope(tp_rank=0, attn_tp_rank=0, attn_dp_rank=0, moe_tp_rank=0)
        if changed
        else nullcontext()
    )


def full_tp_parameters(kind, width=32, heads=8):
    if kind == "mamba1":
        from sglang.srt.layers.attention.mamba.mamba1 import MambaMixer1

        module = MambaMixer1(
            hidden_size=width,
            intermediate_size=2 * width,
            state_size=4,
            conv_kernel=3,
            time_step_rank=4,
            use_conv_bias=True,
            use_bias=False,
        )
        return module, [(module.A_log, 0), (module.D, 0)]
    if kind in ("lfm", "lfm_moe"):
        from sglang.srt.models.lfm2 import Lfm2ShortConv
        from sglang.srt.models.lfm2_moe import Lfm2MoeShortConv

        cls = Lfm2ShortConv if kind == "lfm" else Lfm2MoeShortConv
        module = cls(
            SimpleNamespace(hidden_size=width, conv_L_cache=3, conv_bias=True), 0
        )
        return module, [(module.conv_weight, 0), (module.conv_bias, 0)]
    if kind == "granite":
        from sglang.srt.models.granite import build_attention_sinks

        parameter = build_attention_sinks(heads // get_parallel().tp_size)
        return None, [(parameter, 0)]
    if kind == "minicpm":
        from sglang.srt.models.minicpm import MiniCPMLightningMixer

        module = MiniCPMLightningMixer(
            width, heads, heads, width // heads, use_rope=False, use_output_norm=True
        )
        return module, [(module.o_norm.weight, 0)]
    if kind == "dflash":
        from transformers import LlamaConfig

        from sglang.srt.models.dflash import DFlashAttention

        config = LlamaConfig(
            hidden_size=width,
            num_attention_heads=heads,
            num_key_value_heads=heads,
            head_dim=width // heads,
            num_hidden_layers=1,
            intermediate_size=2 * width,
        )
        config.dflash_config = {"attention_sink_bias": True}
        module = DFlashAttention(config, 0)
        return module, [(module.attention_sink_bias, 0)]
    raise ValueError(kind)


def load_parameters(parameters, rank, size, *, changed=False, transform=None):
    for offset in (0, 7):
        for param, axis in parameters:
            shape = list(param.shape)
            shape[axis] *= size
            full = values(shape, device=param.device, dtype=param.dtype, offset=offset)
            with loading_scope(changed):
                param.weight_loader(param, full)
            expected = full.chunk(size, dim=axis)[rank]
            if transform is not None:
                expected = transform(param, expected)
            torch.testing.assert_close(param, expected, rtol=0, atol=0)


class TestShardedLoaderParallelGroups(CustomTestCase):
    def setUp(self):
        reset_context()
        self.addCleanup(reset_context)

    def exercise_full_tp(self, changed=False, dps=(1,)):
        for dp in dps:
            for rank in range(4):
                reset_context()
                publish(
                    ServerArgs(
                        model_path="dummy", device="cpu", tp_size=4, attn_dp_size=dp
                    ),
                    role="test",
                    ranks=SpawnRanks(world_rank=rank),
                )
                for kind in (
                    "mamba1",
                    "lfm",
                    "lfm_moe",
                    "granite",
                    "minicpm",
                    "dflash",
                ):
                    _, parameters = full_tp_parameters(kind)
                    load_parameters(parameters, rank, 4, changed=changed)

    def test_full_tp_owners_after_scope_exit_and_with_attention_dp(self):
        self.exercise_full_tp(True, (1, 2))

    def test_generic_groups_axes_and_frozen_reload(self):
        publish(
            ServerArgs(model_path="dummy", device="cpu", tp_size=4, attn_dp_size=2),
            role="test",
            ranks=SpawnRanks(world_rank=3),
        )
        for group, rank, size in (
            (None, 1, 2),
            ("tp", 3, 4),
            ("attn_tp", 1, 2),
            ("replicated", 0, 1),
            (ReplicatedParallelGroup("tp", 2), 1, 2),
        ):
            for axis in (0, 1, 2):
                shape = [2, 2, 2]
                param = torch.nn.Parameter(torch.empty(shape), requires_grad=False)
                loader = weight_utils.sharded_weight_loader(
                    axis, **({} if group is None else {"parallel_group": group})
                )
                shape[axis] *= size
                full = values(shape)
                with loading_scope(True):
                    loader(param, full)
                torch.testing.assert_close(
                    param, full.chunk(size, dim=axis)[rank], rtol=0, atol=0
                )
        callback = Mock(side_effect=lambda: get_parallel().tp_rank)
        loader = weight_utils.sharded_weight_loader(0, callback)
        param = torch.nn.Parameter(torch.empty(2), requires_grad=False)
        with loading_scope(True):
            loader(param, torch.arange(8, dtype=torch.float32))
        torch.testing.assert_close(param, torch.tensor([6.0, 7.0]))
        callback.assert_called_once()
        with self.assertRaisesRegex(ValueError, "cannot be combined"):
            weight_utils.sharded_weight_loader(0, callback, parallel_group="tp")

    def test_cpu_uneven_padding_uses_frozen_width(self):
        publish(
            ServerArgs(model_path="dummy", device="cpu", tp_size=3),
            role="test",
            ranks=SpawnRanks(world_rank=2),
        )
        loader = weight_utils.sharded_weight_loader(0, parallel_group="tp")
        param = torch.nn.Parameter(torch.full((3,), -1.0), requires_grad=False)
        with (
            parallel_scope(
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
            ),
            patch.object(weight_utils, "is_cpu", return_value=True),
        ):
            loader(param, torch.arange(7, dtype=torch.float32))
        torch.testing.assert_close(param, torch.tensor([6.0, 0.0, 0.0]))


if __name__ == "__main__":
    unittest.main()
