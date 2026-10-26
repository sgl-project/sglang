"""When the flashinfer fused all-reduce + residual add + RMSNorm kernel declines a
batch, forward_with_allreduce_fusion still completes the sum, over the group the
kernel reduces over."""

import contextlib
import types
import unittest
from unittest.mock import patch

import torch

from sglang.srt.distributed import communication_op
from sglang.srt.layers import flashinfer_comm_fusion, layernorm
from sglang.srt.layers.layernorm import RMSNorm
from sglang.srt.layers.moe import utils as moe_utils
from sglang.srt.runtime_context import get_parallel
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

HIDDEN = 8
PEERS = 4

# (use_attn_tp_group, MoE topology over tp_size=4, group the kernel reduces over)
FUSION_GROUPS = (
    (True, dict(moe_ep_size=1, moe_tp_size=4), "attn_tp"),
    (False, dict(moe_ep_size=1, moe_tp_size=4), "moe_tp"),
    (False, dict(moe_ep_size=4, moe_tp_size=1), "moe_ep"),
    (False, dict(moe_ep_size=2, moe_tp_size=2), "tp"),
)


class Group:
    def __init__(self, name, reduced):
        self.name = name
        self.reduced = reduced

    def all_reduce(self, x):
        self.reduced.append(self.name)
        return x * PEERS


@contextlib.contextmanager
def declined_fused_kernel(reduced, *, moe_ep_size, moe_tp_size):
    """The fused kernel declines every batch; each all-reduce logs its group."""
    getters = {
        "get_attn_tp_group": "attn_tp",
        "get_tp_group": "tp",
        "get_moe_ep_group": "moe_ep",
        "get_moe_tp_group": "moe_tp",
    }
    with contextlib.ExitStack() as stack:
        for getter, name in getters.items():
            stack.enter_context(
                patch.object(
                    communication_op, getter, return_value=Group(name, reduced)
                )
            )
        # The MoE output's group is read from the parallel state directly.
        stack.enter_context(
            patch.object(
                moe_utils,
                "get_parallel",
                return_value=types.SimpleNamespace(
                    moe_ep_size=moe_ep_size,
                    moe_tp_size=moe_tp_size,
                    moe_dp_size=1,
                    tp_group=Group("tp", reduced),
                    moe_ep_group=Group("moe_ep", reduced),
                    moe_tp_group=Group("moe_tp", reduced),
                ),
            )
        )
        stack.enter_context(patch.object(layernorm, "_use_aiter", False))
        stack.enter_context(
            patch.object(
                flashinfer_comm_fusion,
                "flashinfer_allreduce_residual_rmsnorm",
                return_value=(None, None),
            )
        )
        stack.enter_context(
            get_parallel().override(
                tp_size=PEERS,
                attn_tp_size=PEERS,
                attn_dp_size=1,
                attn_cp_size=1,
                moe_ep_size=moe_ep_size,
                moe_tp_size=moe_tp_size,
                moe_dp_size=1,
            )
        )
        yield


class FoldingNorm:
    """Folds x into the residual in place, as the CUDA RMSNorm kernel does."""

    variance_epsilon = 1e-6

    def forward(self, x, residual, post_residual_addition=None):
        if post_residual_addition is not None:
            residual = residual + post_residual_addition
        residual += x
        return residual * 2, residual


def rmsnorm():
    norm = RMSNorm(HIDDEN, force_native=True)
    norm.weight.data.copy_(torch.randn(HIDDEN))
    return norm


class TestDeclinedFusedAllReduceNorm(CustomTestCase):
    def setUp(self):
        torch.manual_seed(0)

    def test_declined_kernel_all_reduces_once_over_the_kernel_group(self):
        for use_attn_tp_group, topology, group in FUSION_GROUPS:
            reduced = []
            with (
                self.subTest(use_attn_tp_group=use_attn_tp_group, **topology),
                declined_fused_kernel(reduced, **topology),
            ):
                norm = rmsnorm()
                x, residual = torch.randn(3, HIDDEN), torch.randn(3, HIDDEN)
                expected = norm.forward_native(x * PEERS, residual)

                out, residual_out = norm.forward_with_allreduce_fusion(
                    x, residual, use_attn_tp_group=use_attn_tp_group
                )

                self.assertEqual(reduced, [group])
                torch.testing.assert_close(out, expected[0])
                torch.testing.assert_close(residual_out, expected[1])

    def test_declined_kernel_adds_post_residual_addition_once(self):
        for use_attn_tp_group, topology, _ in FUSION_GROUPS:
            with (
                self.subTest(use_attn_tp_group=use_attn_tp_group, **topology),
                declined_fused_kernel([], **topology),
            ):
                norm = rmsnorm()
                x, residual, addition = (torch.randn(3, HIDDEN) for _ in range(3))
                expected = norm.forward_native(x * PEERS, residual + addition)

                out, residual_out = norm.forward_with_allreduce_fusion(
                    x, residual, addition, use_attn_tp_group=use_attn_tp_group
                )

                torch.testing.assert_close(out, expected[0])
                torch.testing.assert_close(residual_out, expected[1])

    def test_declined_kernel_leaves_the_input_residual_unchanged(self):
        with declined_fused_kernel([], moe_ep_size=1, moe_tp_size=4):
            x, residual = torch.randn(3, HIDDEN), torch.randn(3, HIDDEN)
            residual_before = residual.clone()

            _, residual_out = layernorm._forward_with_allreduce_fusion(
                FoldingNorm(), x, residual, None, torch.ones(HIDDEN)
            )

            torch.testing.assert_close(residual, residual_before)
            torch.testing.assert_close(residual_out, residual_before + x * PEERS)


if __name__ == "__main__":
    unittest.main()
