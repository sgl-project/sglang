"""Muse Spark weight reloads must refill the derived tensors CUDA graphs read."""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch
from torch import nn

from sglang.srt.layers.muse_spark.muse_spark_layers import (
    MuseSpark1xNormPreservingResidualConnection,
    MuseSpark1xRMSNorm,
)
from sglang.srt.models import muse_spark
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=2, stage="base-a", runner_config="cpu")


class _TwoRankGroup:
    world_size = 2

    def all_gather_into_tensor(self, out, shard):
        out[0].copy_(shard)
        out[1].copy_(shard + 1)


class TestMuseSparkWeightReload(CustomTestCase):
    def test_reload_refills_derived_weights_in_place(self):
        """A captured graph keeps reading the tensors post_load_weights first built
        (SP o_proj replica, split router, norm and residual coefficients); a later
        default-format weight update must write its values there."""
        torch.manual_seed(0)
        o_proj = SimpleNamespace(weight=nn.Parameter(torch.randn(4, 3)))
        layers = [SimpleNamespace(self_attn=SimpleNamespace(o_proj=o_proj))]
        router = muse_spark.MuseSpark1xRouter(16, 8, dtype=torch.float32)
        norm = MuseSpark1xRMSNorm(16, eps=1e-6, zero_centered_gamma=True)
        residual = MuseSpark1xNormPreservingResidualConnection(16)
        weights = (o_proj.weight, router.weight, norm.weight, residual.gate)
        with torch.no_grad():
            for w in weights:
                w.normal_()

        def post_load():
            group = SimpleNamespace(tp_group=_TwoRankGroup())
            with patch.object(muse_spark, "get_parallel", return_value=group):
                muse_spark._replicate_sp_o_proj(layers)
            router._build_split2()
            norm.precompute_effective_weight()
            residual.precompute_coefficients()
            return (
                o_proj._sp_full_weight,
                router._split2_hilo,
                norm._frozen_weight,
                residual._alpha,
                residual._beta,
            )

        captured = post_load()
        with torch.no_grad():
            for w in weights:
                w.add_(1.0)
        for old, new in zip(captured, post_load()):
            self.assertIs(new, old)
        w = o_proj.weight.detach()
        torch.testing.assert_close(o_proj._sp_full_weight, torch.cat([w, w + 1], 1))
        torch.testing.assert_close(
            norm._frozen_weight, norm.weight.detach() + norm.gain_center
        )
        hilo = router._split2_hilo.t().float().view(1, 2, 8, 16)
        torch.testing.assert_close(
            hilo.sum(1).view(8, 16), router.weight.detach(), rtol=1e-3, atol=1e-3
        )


if __name__ == "__main__":
    unittest.main()
