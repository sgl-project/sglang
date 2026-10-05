"""Auxiliary checkpoint shards stay with the constructed activation/norm layer."""

import unittest
from contextlib import nullcontext
from types import SimpleNamespace

import torch
from torch import nn

from sglang.srt.layers.activation import ScaledActivation
from sglang.srt.models.bailing_moe_linear import BailingGroupRMSNormGate
from sglang.srt.models.commandr import LayerNorm
from sglang.srt.models.kimi_k3 import KimiK3DeltaAttention
from sglang.srt.models.minimax_m3 import MultiHeadRMSNorm
from sglang.srt.runtime_context import SpawnRanks, get_parallel, reset_context
from sglang.srt.server_args import ServerArgs
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.parallel_groups import parallel_scope, publish
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=20, suite="base-a-test-cpu")

KINDS = (
    "scaled",
    "scaled_replicated",
    "command",
    "command_replicated",
    "bailing",
    "minimax",
    "minimax_1p",
    "kimi",
)


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


def build_layer(kind):
    parallel = get_parallel()
    if kind.startswith("scaled"):
        replicated = kind.endswith("replicated")
        module = ScaledActivation(nn.GELU(), 32, input_is_parallel=not replicated)
        return (
            module,
            module.scales,
            (32,),
            0 if replicated else parallel.tp_rank,
            1 if replicated else parallel.tp_size,
        )
    if kind.startswith("command"):
        replicated = kind.endswith("replicated")
        module = LayerNorm(16 if replicated else (8 // parallel.tp_size, 16))
        return (
            module,
            module.weight,
            (16,) if replicated else (8, 16),
            0 if replicated else parallel.tp_rank,
            1 if replicated else parallel.tp_size,
        )
    if kind == "bailing":
        module = BailingGroupRMSNormGate(64 // parallel.attn_tp_size, group_size=8)
        return (
            module,
            module.weight,
            (64,),
            parallel.attn_tp_rank,
            parallel.attn_tp_size,
        )
    if kind.startswith("minimax"):
        module = MultiHeadRMSNorm(8, 16, apply_layernorm_1p=kind.endswith("1p"))
        return (
            module,
            module.weight,
            (8, 16),
            parallel.attn_tp_rank,
            parallel.attn_tp_size,
        )
    config = SimpleNamespace(
        dtype=torch.get_default_dtype(),
        v_head_dim=8,
        rms_norm_eps=1e-6,
        linear_attn_config=dict(head_dim=8, num_heads=8, short_conv_kernel_size=3),
    )
    module = KimiK3DeltaAttention(0, 32, config)
    return module, module.A_log, (8,), parallel.attn_tp_rank, parallel.attn_tp_size


def values(shape, param, offset=0):
    count = 1
    for size in shape:
        count *= size
    return ((torch.arange(count, device=param.device) + offset) % 17 + 1).reshape(
        shape
    ).to(param.dtype) / 32


def load_layer(kind, *, changed=False, flat=False, old_kimi=False, offset=0):
    module, param, shape, rank, size = build_layer(kind)
    full = values(shape, param, offset)
    checkpoint = full.flatten() if flat else full
    if old_kimi:
        checkpoint = full.view(1, 1, 8, 1)
    expected = full.chunk(size, dim=0)[rank].reshape_as(param)
    with loading_scope(changed):
        param.weight_loader(param, checkpoint)
    torch.testing.assert_close(param, expected, rtol=0, atol=0)
    return module, param, expected


class TestAuxiliaryLoaderLayouts(CustomTestCase):
    def setUp(self):
        reset_context()
        self.addCleanup(reset_context)

    def check_loads(self, changed):
        for dp in (1, 2):
            for rank in range(4):
                reset_context()
                publish(
                    ServerArgs(
                        model_path="dummy", device="cpu", tp_size=4, attn_dp_size=dp
                    ),
                    role="test",
                    ranks=SpawnRanks(world_rank=rank),
                )
                for kind in KINDS:
                    for offset in (0, 7):
                        load_layer(kind, changed=changed, offset=offset)
                        if kind.startswith("minimax"):
                            load_layer(kind, changed=changed, flat=True, offset=offset)
                        if kind == "kimi":
                            load_layer(
                                kind, changed=changed, old_kimi=True, offset=offset
                            )

    def test_loaders_in_the_construction_scope(self):
        self.check_loads(False)

    def test_loaders_after_scope_exit(self):
        self.check_loads(True)


if __name__ == "__main__":
    unittest.main()
