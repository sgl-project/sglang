"""Kimi-K3 post_load_weights rebuilds the replicated-Q DCP weights of the MLA
layers whose attention weights were loaded, in place, and gathers nothing before
the model runner has built them."""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch
from torch import nn

from sglang.srt.models import kimi_k3
from sglang.srt.models.kimi_k3 import KimiK3LinearForCausalLM
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=10, suite="base-a-test-cpu")


class _TwoRankGroup:
    """A DCP group of two ranks whose peer holds this rank's shard plus 100."""

    def all_gather(self, tensor, dim):
        return torch.cat([tensor, tensor + 100], dim=dim)


class _NoCollectives:
    def all_gather(self, tensor, dim):
        raise AssertionError("gathered before the replicated-Q buffers exist")


def _use_dcp_group(test: CustomTestCase, group) -> None:
    patcher = patch.object(
        kimi_k3, "get_parallel", return_value=SimpleNamespace(dcp_group=group)
    )
    patcher.start()
    test.addCleanup(patcher.stop)


def _attention():
    # 3 local heads, nope = v = 2, kv_lora = 4; gathered at start-up from the
    # dummy weights.
    return SimpleNamespace(
        kv_b_proj=nn.Linear(4, 3 * (2 + 2), bias=False),
        qk_nope_head_dim=2,
        v_head_dim=2,
        w_kc=None,
        w_vc=None,
        has_q_b_proj=True,
        q_b_proj=nn.Linear(2, 3, bias=False),
        w_kc_qrep=torch.zeros(6, 2, 4),
        q_b_proj_qrep_weight=torch.zeros(6, 2),
        q_b_proj_qrep=None,
    )


def _replicated_q_model():
    layers = [
        SimpleNamespace(
            self_attn=_attention(), mlp=nn.Identity(), use_attn_residuals=False
        )
        for _ in range(3)
    ]
    return SimpleNamespace(
        # Layer 1 stands in for a KDA layer.
        config=SimpleNamespace(full_attention_layer_ids=[0, 2]),
        model=SimpleNamespace(layers=layers),
        _kda_prefill_precompiled=True,
    )


def _gathered_w_kc(attn) -> torch.Tensor:
    w_kc = attn.kv_b_proj.weight.data.unflatten(0, (3, 4))[:, :2]
    return torch.cat([w_kc, w_kc + 100])


class TestKimiK3ReplicatedQRefresh(CustomTestCase):
    def setUp(self):
        _use_dcp_group(self, _TwoRankGroup())

    def _assert_refreshed(self, attn):
        torch.testing.assert_close(attn.w_kc_qrep, _gathered_w_kc(attn))
        weight = attn.q_b_proj.weight.data
        torch.testing.assert_close(
            attn.q_b_proj_qrep_weight, torch.cat([weight, weight + 100])
        )

    def _assert_untouched(self, attn):
        self.assertFalse(attn.w_kc_qrep.any())
        self.assertFalse(attn.q_b_proj_qrep_weight.any())

    def test_weight_update_rebuilds_replicated_q_in_place(self):
        model = _replicated_q_model()
        attn = model.model.layers[0].self_attn
        buffers = (attn.w_kc_qrep.data_ptr(), attn.q_b_proj_qrep_weight.data_ptr())
        KimiK3LinearForCausalLM.post_load_weights(
            model, {"model.layers.0.self_attn.kv_b_proj.weight"}
        )
        # Captured CUDA graphs read these buffers: rewritten, not replaced.
        self.assertEqual(
            (attn.w_kc_qrep.data_ptr(), attn.q_b_proj_qrep_weight.data_ptr()),
            buffers,
        )
        self._assert_refreshed(attn)
        self._assert_untouched(model.model.layers[2].self_attn)

    def test_refresh_skips_layers_without_attention_weights(self):
        model = _replicated_q_model()
        # Layer 2 got only experts (which a rank may skip under EP); layer 1 is
        # KDA.
        KimiK3LinearForCausalLM.post_load_weights(
            model,
            {
                "model.layers.2.mlp.experts.w13_weight",
                "model.layers.1.self_attn.q_proj.weight",
            },
        )
        for layer in model.model.layers:
            self._assert_untouched(layer.self_attn)

    def test_post_load_without_names_refreshes_every_mla_layer(self):
        # P2P/RDMA weight updates bypass load_weights and call post_load_weights().
        model = _replicated_q_model()
        KimiK3LinearForCausalLM.post_load_weights(model)
        self._assert_refreshed(model.model.layers[0].self_attn)
        self._assert_refreshed(model.model.layers[2].self_attn)
        self._assert_untouched(model.model.layers[1].self_attn)

    def test_refresh_gathers_nothing_before_the_buffers_exist(self):
        # The first load runs before the model runner builds the buffers.
        model = _replicated_q_model()
        for layer in model.model.layers:
            layer.self_attn.w_kc_qrep = layer.self_attn.q_b_proj_qrep_weight = None
        _use_dcp_group(self, _NoCollectives())
        KimiK3LinearForCausalLM.post_load_weights(model)
        self.assertIsNone(model.model.layers[0].self_attn.w_kc_qrep)

    def test_block_fp8_replica_is_refreshed_in_place(self):
        def gather_block_fp8_linear(layer, group):
            replica = nn.Module()
            for name in ("weight", "weight_scale_inv"):
                shard = getattr(layer, name).data
                setattr(replica, name, nn.Parameter(group.all_gather(shard, dim=0)))
            return replica

        model = _replicated_q_model()
        attn = model.model.layers[0].self_attn
        attn.q_b_proj.weight_scale_inv = nn.Parameter(torch.rand(3, 1))
        attn.q_b_proj_qrep_weight = None
        attn.q_b_proj_qrep = nn.Module()
        attn.q_b_proj_qrep.weight = nn.Parameter(torch.zeros(6, 2))
        attn.q_b_proj_qrep.weight_scale_inv = nn.Parameter(torch.zeros(6, 1))
        replica = attn.q_b_proj_qrep
        buffers = [param.data_ptr() for param in replica.parameters()]
        with patch(
            "sglang.srt.model_executor.model_runner._gather_block_fp8_linear",
            gather_block_fp8_linear,
        ):
            KimiK3LinearForCausalLM.post_load_weights(
                model, {"model.layers.0.self_attn.q_b_proj.weight"}
            )
        self.assertIs(attn.q_b_proj_qrep, replica)
        self.assertEqual([param.data_ptr() for param in replica.parameters()], buffers)
        for name in ("weight", "weight_scale_inv"):
            shard = getattr(attn.q_b_proj, name).data
            torch.testing.assert_close(
                getattr(replica, name).data, torch.cat([shard, shard + 100])
            )
        torch.testing.assert_close(attn.w_kc_qrep, _gathered_w_kc(attn))


if __name__ == "__main__":
    unittest.main()
