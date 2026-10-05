"""Kimi K3's own kernels for a layer's sums and gathers serve its
attention-residual bank only: the standard residual path keeps the
boundary's collectives, as the decoder did, even where those kernels run."""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import test_declared_decoder_boundary as fixture
from torch import nn

from sglang.srt.layers.communication import k3_ar_fusion, k3_sp_collective
from sglang.srt.layers.layer_boundary import BatchVariant, layer_stack
from sglang.srt.layers.layer_boundary import prepare as comm_ops
from sglang.srt.layers.layer_boundary.residual.attn_bank import AttnBank
from sglang.srt.models import kimi_k3
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

CONFIG = SimpleNamespace(
    is_moe=True, num_experts=8, first_k_dense_replace=1, moe_layer_freq=1
)


def moe_layer(*, bank, sp_moe, all_reduce_fusion):
    """A K3 latent MoE layer's attributes that its stage declarations read."""
    layer = kimi_k3.KimiK3DecoderLayer.__new__(kimi_k3.KimiK3DecoderLayer)
    nn.Module.__init__(layer)
    layer.use_attn_residuals = bank
    layer.is_block_write_layer = False
    layer._is_moe_layer = True
    layer._sp_moe = sp_moe
    layer.all_reduce_fusion = all_reduce_fusion
    layer._ffn_writes_stream = bank
    layer.mlp = SimpleNamespace(use_latent_moe=True, _ep_a2a=sp_moe)
    layer.input_layernorm = fixture.Norm()
    layer.post_attention_layernorm = fixture.Norm()
    for name in (
        "self_attention_res_proj",
        "self_attention_res_norm",
        "mlp_res_proj",
        "mlp_res_norm",
    ):
        setattr(layer, name, None)
    return layer


class TestKimiK3StageKernels(CustomTestCase):
    def ffn_path(self, *, bank, sp_moe, all_reduce_fusion):
        layer = moe_layer(bank=bank, sp_moe=sp_moe, all_reduce_fusion=all_reduce_fusion)
        with (
            patch.object(k3_sp_collective, "enabled", return_value=True),
            patch.object(k3_ar_fusion, "enabled", return_value=True),
            fixture.planning(
                fixture.parallel_of(attn_dp=1, attn_tp=2),
                a2a=sp_moe,
                boundary_reduction="ar",
            ),
            layer_stack(),
        ):
            layer._declare_stages(CONFIG, 1, AttnBank())
        return layer.ffn_boundary.plan.paths[BatchVariant.ORDINARY]

    @staticmethod
    def kernels(path):
        step = path.entry.prepare.keywords["step"]
        gather = getattr(path.output_move, "keywords", {}).get("gather")
        return step.func, step.keywords["read_fusions"], gather

    def test_the_standard_path_keeps_the_boundary_collectives(self):
        for sp_moe in (True, False):
            with self.subTest(sp_moe=sp_moe):
                entry, fusions, gather = self.kernels(
                    self.ffn_path(bank=False, sp_moe=sp_moe, all_reduce_fusion=True)
                )
                self.assertIs(
                    entry,
                    comm_ops._attn_tp_reduce_scatter_update_read
                    if sp_moe
                    else comm_ops._reduce_update_read,
                )
                self.assertEqual(fusions, ())
                self.assertIsNone(gather)

    def test_the_bank_takes_the_tuned_reduce_scatter_and_gather(self):
        entry, fusions, gather = self.kernels(
            self.ffn_path(bank=True, sp_moe=True, all_reduce_fusion=False)
        )
        self.assertIs(entry, comm_ops._attn_tp_reduce_scatter_update_read)
        self.assertEqual(fusions, (kimi_k3._k3_reduce_scatter_add,))
        self.assertIs(gather, k3_sp_collective.all_gather)

    def test_the_bank_takes_the_fused_all_reduce(self):
        entry, fusions, gather = self.kernels(
            self.ffn_path(bank=True, sp_moe=False, all_reduce_fusion=True)
        )
        self.assertIs(entry, comm_ops._reduce_update_read)
        self.assertEqual(fusions, (kimi_k3._k3_all_reduce_add,))
        self.assertIsNone(gather)


if __name__ == "__main__":
    unittest.main()
