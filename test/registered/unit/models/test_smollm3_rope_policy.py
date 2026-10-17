"""Unit tests for SmolLM3 per-layer RoPE selection and unimplemented-config guards."""

from types import SimpleNamespace
from unittest.mock import patch

import torch

import sglang.srt.models.llama as llama
from sglang.srt.models.smollm3 import SmolLM3Attention
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=8, suite="base-a-test-cpu")


class _ModuleStub(torch.nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()


def _config(**overrides):
    values = dict(
        use_sliding_window=False,
        num_hidden_layers=8,
        no_rope_layers=[1, 1, 1, 0, 1, 1, 1, 0],  # every 4th layer is NoPE
        head_dim=8,
        partial_rotary_factor=1,
    )
    values.update(overrides)
    return SimpleNamespace(**values)


def _make_attention(config, layer_id, start_layer=0):
    parallel = SimpleNamespace(tp_size=1)
    with (
        patch.object(llama, "get_parallel", return_value=parallel),
        patch.object(llama, "QKVParallelLinear", _ModuleStub),
        patch.object(llama, "RowParallelLinear", _ModuleStub),
        patch.object(llama, "RadixAttention", _ModuleStub),
        patch.object(llama, "get_rope", side_effect=lambda *a, **k: _ModuleStub()),
    ):
        return SmolLM3Attention(
            config=config,
            hidden_size=16,
            num_heads=2,
            num_kv_heads=2,
            layer_id=layer_id,
            start_layer=start_layer,
        )


class TestSmolLM3RopePolicy(CustomTestCase):
    def test_rope_layer_gets_rotary_embedding(self):
        attention = _make_attention(_config(), layer_id=0)
        self.assertIsNotNone(attention.rotary_emb)

    def test_nope_layer_has_no_rotary_embedding(self):
        attention = _make_attention(_config(), layer_id=3)
        self.assertIsNone(attention.rotary_emb)

    def test_npu_cos_sin_refresh_layer(self):
        # forward_prepare_npu refreshes cos/sin on the layer matching start_layer;
        # a PP stage that begins on a NoPE layer must defer it to the next RoPE layer.
        config = _config()
        for layer_id in range(8):
            self.assertEqual(_make_attention(config, layer_id, 0).start_layer, 0)
            self.assertEqual(_make_attention(config, layer_id, 2).start_layer, 2)
            self.assertEqual(_make_attention(config, layer_id, 3).start_layer, 4)
        # A stage with no RoPE layers left keeps its own start_layer.
        all_nope_tail = _config(no_rope_layers=[1, 1, 1, 1, 1, 1, 0, 0])
        self.assertEqual(_make_attention(all_nope_tail, 7, 6).start_layer, 6)

    def test_rejects_sliding_window(self):
        with self.assertRaises(NotImplementedError):
            _make_attention(_config(use_sliding_window=True), layer_id=0)

    def _bare_attention(self):
        # forward_prepare_native only touches qkv_proj/q_size/kv_size/rotary_emb,
        # so a bare instance (skipping __init__) avoids the full mocked construction
        # nn.Module.__setattr__ would otherwise refuse to overwrite qkv_proj with a
        # plain lambda once it's already registered as a submodule.
        attention = object.__new__(SmolLM3Attention)
        torch.nn.Module.__init__(attention)
        attention.qkv_proj = lambda x: (torch.cat([x, x, x], dim=-1), None)
        attention.q_size = attention.kv_size = 4
        return attention

    def test_forward_prepare_native_skips_rope_when_disabled(self):
        attention = self._bare_attention()
        attention.rotary_emb = None

        hidden_states = torch.randn(2, 4)
        q, k, v = attention.forward_prepare_native(torch.arange(2), hidden_states)
        torch.testing.assert_close(q, hidden_states)

    def test_forward_prepare_native_applies_rope_when_enabled(self):
        attention = self._bare_attention()
        calls = []

        def rotary_emb(positions, q, k):
            calls.append((positions, q, k))
            return q, k

        attention.rotary_emb = rotary_emb

        attention.forward_prepare_native(torch.arange(2), torch.randn(2, 4))
        self.assertEqual(len(calls), 1)


if __name__ == "__main__":
    import unittest

    unittest.main()
