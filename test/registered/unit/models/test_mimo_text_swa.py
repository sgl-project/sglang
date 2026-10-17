"""MiMo text windows include the current token, as in the release HF mask."""

import unittest
from contextlib import ExitStack
from types import SimpleNamespace
from unittest.mock import patch

import torch
from torch import nn
from transformers.masking_utils import sliding_window_causal_mask_function

from sglang.srt.model_executor.model_runner_components.load_model_utils import (
    resolve_sliding_window_size,
)
from sglang.srt.models import mimo_v2, mimo_v2_nextn
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=10, suite="base-a-test-cpu")


def config():
    return SimpleNamespace(
        hidden_size=128,
        num_hidden_layers=2,
        hybrid_layer_pattern=[1, 0],
        num_attention_heads=16,
        num_key_value_heads=2,
        swa_num_attention_heads=16,
        swa_num_key_value_heads=2,
        head_dim=192,
        swa_head_dim=192,
        v_head_dim=128,
        swa_v_head_dim=128,
        sliding_window_size=128,
        attention_bias=False,
        add_swa_attention_sink_bias=True,
        add_full_attention_sink_bias=False,
        intermediate_size=256,
        hidden_act="silu",
        layernorm_epsilon=1e-6,
    )


def make_attention(cfg, layer_id=0, mtp=False):
    """Keep the real decoder/attention constructors; stub unrelated weights/TP."""
    module = mimo_v2_nextn if mtp else mimo_v2
    cls = module.MiMoV2MTPLayer if mtp else module.MiMoV2DecoderLayer
    with ExitStack() as stack:
        stack.enter_context(
            patch.object(
                mimo_v2,
                "get_parallel",
                return_value=SimpleNamespace(attn_tp_rank=0, attn_tp_size=1),
            )
        )
        for name in ("QKVParallelLinear", "RowParallelLinear", "get_rope"):
            stack.enter_context(patch.object(mimo_v2, name, return_value=nn.Identity()))
        stack.enter_context(
            patch.object(module, "MiMoV2MLP", return_value=nn.Identity())
        )
        stack.enter_context(
            patch.object(module, "is_dense_ffn_fully_dp", return_value=False)
        )
        stack.enter_context(patch.object(module, "declare_attn"))
        stack.enter_context(patch.object(module, "declare_ffn"))
        stack.enter_context(
            patch.object(
                module, "append_stages", return_value=(nn.Identity(), nn.Identity())
            )
        )
        return cls(cfg, layer_id=layer_id).self_attn


def runner_window(cfg, mtp=False):
    cls = mimo_v2_nextn.MiMoV2MTP if mtp else mimo_v2.MiMoV2ForCausalLM
    model = cls.__new__(cls)
    nn.Module.__init__(model)
    model.config = cfg
    return resolve_sliding_window_size(
        model,
        SimpleNamespace(
            is_hybrid_swa=True, sliding_window_size=cfg.sliding_window_size
        ),
    )


class TestTextSWAConfig(CustomTestCase):
    def test_target_layers_and_cache_agree(self):
        cfg = config()
        swa = make_attention(cfg)
        full = make_attention(cfg, layer_id=1)
        self.assertEqual(swa.attn.sliding_window_size, 127)
        self.assertEqual(runner_window(cfg), 127)
        self.assertEqual(full.attn.sliding_window_size, -1)
        self.assertIsNotNone(swa.attention_sink_bias)
        self.assertIsNone(full.attention_sink_bias)
        self.assertEqual(cfg.sliding_window_size, 128)

    def test_mtp_layer_and_cache_agree(self):
        cfg = config()
        self.assertEqual(make_attention(cfg, mtp=True).attn.sliding_window_size, 127)
        self.assertEqual(runner_window(cfg, mtp=True), 127)

    def test_unbounded_window_is_preserved(self):
        cfg = config()
        for window in (None, -1):
            cfg.sliding_window_size = window
            self.assertEqual(runner_window(cfg), window)

    def test_visibility_matches_release_mask(self):
        cfg = config()
        for count in (64, 128, 256):
            cfg.sliding_window_size = count
            left = make_attention(cfg).attn.sliding_window_size
            self.assertEqual(runner_window(cfg), left)
            qi = torch.arange(2 * count + 1)[:, None]
            ki = torch.arange(2 * count + 1)[None, :]
            expected = sliding_window_causal_mask_function(count)(0, 0, qi, ki)
            actual = (ki <= qi) & (ki >= qi - left)
            torch.testing.assert_close(actual, expected)
            self.assertEqual(int(actual[-1].sum()), count)


if __name__ == "__main__":
    unittest.main()
