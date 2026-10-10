"""
Unit tests for the NemotronHMoE routed-expert output buffer.

The shared experts read ``hidden_states`` on the main stream while the routed
experts run on the alt stream, so the routed experts may only write their
output in place when nothing else reads their input.
"""

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=10, suite="base-a-test-cpu")

import unittest
from types import SimpleNamespace
from unittest import mock

from torch import nn

from sglang.srt.models import nemotron_h
from sglang.test.test_utils import CustomTestCase


class _RecordingExperts(nn.Module):
    should_fuse_routed_scaling_factor_in_topk = False

    def __init__(self, **kwargs):
        super().__init__()
        self.kwargs = kwargs


def _config(*, n_shared_experts: int, moe_latent_size: int | None):
    return SimpleNamespace(
        routed_scaling_factor=1.0,
        n_routed_experts=4,
        n_shared_experts=n_shared_experts,
        moe_latent_size=moe_latent_size,
        hidden_size=8,
        num_experts_per_tok=2,
        moe_intermediate_size=16,
        moe_shared_expert_intermediate_size=16,
        mlp_hidden_act="relu2",
        mlp_bias=False,
        topk_group=1,
        n_group=1,
        norm_topk_prob=True,
    )


def _routed_inplace(**config_kwargs) -> bool:
    group = SimpleNamespace(rank=lambda: 0, size=lambda: 1)
    parallel = SimpleNamespace(
        tp_size=1, moe_ep_group=SimpleNamespace(device_group=group)
    )
    exec_config = SimpleNamespace(moe=SimpleNamespace(ep_num_redundant_experts=0))
    with (
        mock.patch.object(nemotron_h, "get_parallel", return_value=parallel),
        mock.patch.object(nemotron_h, "get_exec", return_value=exec_config),
        mock.patch.object(
            nemotron_h, "get_moe_impl_class", return_value=_RecordingExperts
        ),
        mock.patch.object(nemotron_h, "TopK", return_value=nn.Identity()),
        mock.patch.object(nemotron_h, "NemotronHMLP", return_value=nn.Identity()),
    ):
        moe = nemotron_h.NemotronHMoE(_config(**config_kwargs), layer_idx=0)
    return moe.experts.kwargs["inplace"]


class TestNemotronHMoEInplace(CustomTestCase):
    def test_shared_experts_keep_their_input_intact(self):
        self.assertFalse(_routed_inplace(n_shared_experts=1, moe_latent_size=None))

    def test_latent_moe_writes_its_private_projection_in_place(self):
        self.assertTrue(_routed_inplace(n_shared_experts=1, moe_latent_size=4))

    def test_without_shared_experts_the_input_has_no_other_reader(self):
        self.assertTrue(_routed_inplace(n_shared_experts=0, moe_latent_size=None))


if __name__ == "__main__":
    unittest.main()
