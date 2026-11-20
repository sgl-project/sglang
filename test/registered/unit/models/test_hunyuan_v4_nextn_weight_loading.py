import unittest
from types import SimpleNamespace

import torch
from torch import nn

from sglang.srt.models.deepseek_common.deepseek_weight_loader import (
    DeepseekV2WeightLoaderMixin,
)
from sglang.srt.models.hunyuan_v4_nextn import HYV4ForCausalLMNextN
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=10, suite="base-a-test-cpu")

_INDEX_HEAD_DIM = 128
_MTP = "model.mtp_layers.0"


class _OtherFamilyModel(nn.Module, DeepseekV2WeightLoaderMixin):
    pass


def _loader(cls, params):
    model = cls.__new__(cls)
    nn.Module.__init__(model)
    model.config = SimpleNamespace(
        n_routed_experts=1,
        num_nextn_predict_layers=1,
        num_hidden_layers=2,
        q_lora_rank=None,
        index_head_dim=_INDEX_HEAD_DIM,
        qk_rope_head_dim=64,
    )
    model.quant_config = None
    model.num_fused_shared_experts = 0
    model.pp_group = SimpleNamespace(is_last_rank=True)
    model.model = nn.Module()
    model.named_parameters = lambda: iter(params.items())
    model.post_load_weights = lambda **kwargs: None
    return model


def _fused_wk_param():
    return nn.Parameter(
        torch.zeros(_INDEX_HEAD_DIM + 4, 128, dtype=torch.bfloat16),
        requires_grad=False,
    )


def _fp8_wk():
    return torch.ones(_INDEX_HEAD_DIM, 128).to(torch.float8_e4m3fn)


class TestHunyuanV4NextNWeightLoading(unittest.TestCase):
    def test_indexer_checkpoint_layout_is_permuted(self):
        model = object.__new__(HYV4ForCausalLMNextN)
        model.config = SimpleNamespace(
            num_hidden_layers=80,
            index_n_heads=2,
            index_head_dim=4,
            qk_rope_head_dim=2,
        )
        model.quant_config = None
        captured = []
        object.__setattr__(
            model,
            "do_load_weights",
            lambda weights, **kwargs: captured.extend(weights),
        )
        loaded_weight = torch.arange(8).reshape(8, 1)

        model.load_weights(
            [("model.mtp_layers.0.self_attn.indexer.wq_b.weight", loaded_weight)]
        )

        self.assertEqual(
            captured[0][0], "model.layers.80.self_attn.indexer.wq_b.weight"
        )
        torch.testing.assert_close(
            captured[0][1].flatten(), torch.tensor([2, 3, 0, 1, 6, 7, 4, 5])
        )

    def test_unresolved_fp8_indexer_pair_fails_the_load(self):
        """An FP8 indexer wk without its scale must fail the MTP load instead of
        leaving the fused indexer projection unloaded."""
        fused = _fused_wk_param()
        model = _loader(
            HYV4ForCausalLMNextN,
            {"model.decoder.self_attn.indexer.wk_weights_proj.weight": fused},
        )
        wk = (f"{_MTP}.self_attn.indexer.wk.weight", _fp8_wk())
        with self.assertRaisesRegex(ValueError, "Incomplete FP8 indexer"):
            model.load_weights([wk])

        scale = (f"{_MTP}.self_attn.indexer.wk.scale", torch.full((1, 1), 2.0))
        model.load_weights([wk, scale])
        torch.testing.assert_close(
            fused[:_INDEX_HEAD_DIM], torch.full_like(fused[:_INDEX_HEAD_DIM], 2.0)
        )

    def test_other_families_only_warn_on_unresolved_fp8_indexer_pair(self):
        model = _loader(
            _OtherFamilyModel,
            {
                "model.layers.0.self_attn.indexer.wk_weights_proj.weight": (
                    _fused_wk_param()
                )
            },
        )
        with self.assertLogs(level="WARNING") as logs:
            model.do_load_weights(
                [("model.layers.0.self_attn.indexer.wk.weight", _fp8_wk())]
            )
        self.assertIn("FP8 indexer weight/scale pairs never completed", logs.output[0])

    def test_missing_weight_scale_fails_the_load(self):
        """An FP8 MTP weight whose scale is absent from the checkpoint must fail
        the load; KV-cache scales keep their runtime default."""
        prefix = "model.decoder.mlp.shared_experts.down_proj"
        scale_param = nn.Parameter(torch.zeros(1, 1), requires_grad=False)
        model = _loader(
            HYV4ForCausalLMNextN,
            {
                f"{prefix}.weight": nn.Parameter(
                    torch.zeros(2, 2), requires_grad=False
                ),
                f"{prefix}.weight_scale_inv": scale_param,
                "model.decoder.self_attn.attn_mqa.k_scale": nn.Parameter(
                    torch.ones(()), requires_grad=False
                ),
            },
        )
        weight = (f"{_MTP}.mlp.shared_experts.down_proj.weight", torch.ones(2, 2))
        with self.assertRaisesRegex(ValueError, f"missing weight scales: {prefix}"):
            model.load_weights([weight])

        scale = (f"{_MTP}.mlp.shared_experts.down_proj.scale", torch.full((1, 1), 3.0))
        model.load_weights([weight, scale])
        self.assertEqual(scale_param.item(), 3.0)

    def test_mtp_weights_outside_the_single_head_fail_the_load(self):
        """MTP weights under another head, or under a model.layers index past
        the backbone, must fail the load instead of leaving the draft unloaded."""
        param = nn.Parameter(torch.zeros(2, 2), requires_grad=False)
        model = _loader(
            HYV4ForCausalLMNextN,
            {"model.decoder.mlp.shared_experts.down_proj.weight": param},
        )
        suffix = "mlp.shared_experts.down_proj.weight"
        for prefix in ("model.mtp_layers.1", "model.layers.2"):
            with self.assertRaisesRegex(ValueError, f"single draft head.*{prefix}"):
                model.load_weights([(f"{prefix}.{suffix}", torch.ones(2, 2))])

        backbone = (f"model.layers.1.{suffix}", torch.full((2, 2), 5.0))
        model.load_weights([backbone, (f"{_MTP}.{suffix}", torch.ones(2, 2))])
        torch.testing.assert_close(param, torch.ones(2, 2))


if __name__ == "__main__":
    unittest.main()
