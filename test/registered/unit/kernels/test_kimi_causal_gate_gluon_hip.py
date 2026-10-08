import unittest
from types import SimpleNamespace
from unittest import mock

import torch

from sglang.kernels.ops.attention import kimi_causal_gate_gluon_hip as adapter


def make_attn(rows=1024):
    gate_hidden = torch.empty((rows, 7168), dtype=torch.bfloat16, device="meta")
    gate = torch.empty((rows, 1536), dtype=torch.bfloat16, device="meta")
    projected = torch.empty((rows, 7168), dtype=torch.bfloat16, device="meta")
    return SimpleNamespace(
        forward_normal_core=mock.Mock(return_value="native"),
        g_proj=mock.Mock(return_value=(gate, None)),
        o_proj=mock.Mock(return_value=(projected, None)),
        attn_mha=SimpleNamespace(scaling=192**-0.5),
        _gate_hidden_states=gate_hidden,
        _gate_precomputed=None,
    )


class TestKimiCausalGate(unittest.TestCase):
    def test_native_runtime_uses_current_mla_cp_predicate(self):
        from sglang.srt.layers.cp.utils import is_mla_cp_active

        self.assertIs(adapter.native_runtime().mla_cp, is_mla_cp_active)

    def test_shape_table_keeps_losses_native(self):
        for requests in (1, 2, 4, 8, 16, 32):
            self.assertTrue(adapter.supported_shape(1024, requests))
        for shape in ((2048, 1), (2048, 2), (4096, 1), (8192, 8)):
            self.assertFalse(adapter.supported_shape(*shape))
        for shape in ((2048, 16), (4096, 8), (8192, 32)):
            self.assertTrue(adapter.supported_shape(*shape))

    def test_layout_requires_fresh_complete_sequences(self):
        self.assertEqual(adapter.layout([512, 512], [0, 0], 1024), (512, 512))
        self.assertIsNone(adapter.layout([512, 512], [0, 1], 1024))
        self.assertIsNone(adapter.layout([512, 511], [0, 0], 1024))

    def test_binding_preserves_fallback_and_propagates_launch_failure(self):
        attn = make_attn()
        original = attn.forward_normal_core
        adapter.bind(attn, _runtime=SimpleNamespace(is_tensor=torch.is_tensor))
        q = torch.empty((1024, 12, 192), dtype=torch.bfloat16, device="meta")
        k = torch.empty_like(q)
        v = torch.empty((1024, 12, 128), dtype=torch.bfloat16, device="meta")
        fb = object()

        with mock.patch.object(adapter, "eligible", return_value=None):
            self.assertEqual(attn.forward_normal_core(q, k, v, fb), "native")
        original.assert_called_once()

        indptr = torch.empty((2,), dtype=torch.int64, device="meta")
        with (
            mock.patch.object(adapter, "eligible", return_value=((1024,), indptr)),
            mock.patch.object(adapter, "run", side_effect=RuntimeError("launch")),
            self.assertRaisesRegex(RuntimeError, "launch"),
        ):
            attn.forward_normal_core(q, k, v, fb)
        self.assertEqual(original.call_count, 1)

    def test_success_consumes_gate_once_and_calls_output_projection(self):
        attn = make_attn()
        adapter.bind(attn, _runtime=SimpleNamespace(is_tensor=torch.is_tensor))
        q = torch.empty((1024, 12, 192), dtype=torch.bfloat16, device="meta")
        k = torch.empty_like(q)
        v = torch.empty((1024, 12, 128), dtype=torch.bfloat16, device="meta")
        indptr = torch.empty((2,), dtype=torch.int64, device="meta")
        output = torch.empty((1024, 1536), dtype=torch.bfloat16, device="meta")
        with (
            mock.patch.object(adapter, "eligible", return_value=((1024,), indptr)),
            mock.patch.object(adapter, "run", return_value=output),
        ):
            result = attn.forward_normal_core(q, k, v, object())
        self.assertIs(result, attn.o_proj.return_value[0])
        self.assertIsNone(attn._gate_hidden_states)
        self.assertIsNone(attn._gate_precomputed)
        attn.g_proj.assert_called_once()
        attn.o_proj.assert_called_once_with(output)


if __name__ == "__main__":
    unittest.main()
