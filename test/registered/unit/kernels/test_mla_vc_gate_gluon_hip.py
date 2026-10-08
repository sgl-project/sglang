import unittest
from types import SimpleNamespace
from unittest import mock

import torch

from sglang.kernels.ops.attention import mla_vc_gate_gluon_hip as adapter


def make_attn(rows=1):
    return SimpleNamespace(
        w_vc=torch.empty((12, 128, 512), dtype=torch.bfloat16).transpose(1, 2),
        _gate_hidden_states=torch.empty((rows, 7168), dtype=torch.bfloat16),
    )


class TestMlaValueGate(unittest.TestCase):
    def test_unqualified_rows_and_layouts_are_rejected(self):
        attn = make_attn()
        latent = torch.empty((1, 12, 512), dtype=torch.bfloat16)
        self.assertTrue(adapter.covered(attn, latent))
        for rows in (0, 3, 12, 24, 257, 1023, 8193):
            self.assertIsNone(adapter.entrypoint_name(rows))
        attn.w_vc = attn.w_vc.contiguous()
        self.assertFalse(adapter.covered(attn, latent))
        attn = make_attn()
        self.assertFalse(adapter.covered(attn, latent.float()))
        attn._gate_hidden_states = None
        self.assertFalse(adapter.covered(attn, latent))

    def test_gate_is_computed_and_consumed_once(self):
        attn = make_attn()
        hidden = attn._gate_hidden_states
        latent = torch.empty((1, 12, 512), dtype=torch.bfloat16)
        gate = torch.empty((1, 1536), dtype=torch.bfloat16)
        expected = torch.ones_like(gate)
        attn._compute_output_gate = mock.Mock(return_value=gate)
        with mock.patch.object(adapter, "run", return_value=expected) as kernel:
            self.assertIs(adapter.apply(attn, latent), expected)
            kernel.assert_called_once_with(latent, attn.w_vc, gate)
        attn._compute_output_gate.assert_called_once_with(hidden)
        self.assertIsNone(attn._gate_hidden_states)
        # The native o_proj wrapper uses this same sentinel to skip its gate.
        self.assertFalse(adapter.covered(attn, latent))

    def test_failure_does_not_consume_gate_or_hide_exception(self):
        attn = make_attn()
        hidden = attn._gate_hidden_states
        latent = torch.empty((1, 12, 512), dtype=torch.bfloat16)
        gate = torch.empty((1, 1536), dtype=torch.bfloat16)
        attn._compute_output_gate = mock.Mock(return_value=gate)
        with mock.patch.object(adapter, "run", side_effect=RuntimeError("launch")):
            with self.assertRaisesRegex(RuntimeError, "launch"):
                adapter.apply(attn, latent)
        self.assertIs(attn._gate_hidden_states, hidden)
        with mock.patch.object(adapter, "run", return_value=gate):
            with self.assertRaisesRegex(RuntimeError, "ABI"):
                adapter.apply(attn, latent)
        self.assertIs(attn._gate_hidden_states, hidden)

    def test_forward_scope_rejects_speculation_and_resets_on_failure(self):
        from sglang.srt.model_executor.forward_batch_info import ForwardMode
        from sglang.srt.models import kimi_k3

        attn = kimi_k3.KimiK3MLAAttention.__new__(kimi_k3.KimiK3MLAAttention)
        torch.nn.Module.__init__(attn)
        attn.use_output_gate = False
        attn._mla_gluon_vc_ready = True
        attn._mla_gluon_vc_active = False
        latent = torch.empty((1, 7168), dtype=torch.bfloat16)
        with (
            mock.patch.object(
                kimi_k3, "get_forward", return_value=SimpleNamespace(sp_active=False)
            ),
            mock.patch.object(
                kimi_k3.DeepseekV2AttentionMLA,
                "forward",
                side_effect=lambda *args, **kwargs: attn._mla_gluon_vc_active,
            ) as native,
        ):
            for mode, spec, expected in (
                (ForwardMode.DECODE, None, True),
                (ForwardMode.EXTEND, None, True),
                (ForwardMode.TARGET_VERIFY, None, False),
                (ForwardMode.DECODE, object(), False),
            ):
                batch = SimpleNamespace(forward_mode=mode, spec_info=spec)
                self.assertIs(attn.forward(None, latent, batch, None), expected)
                self.assertFalse(attn._mla_gluon_vc_active)
            native.side_effect = RuntimeError("forward failure")
            batch = SimpleNamespace(forward_mode=ForwardMode.DECODE, spec_info=None)
            with self.assertRaisesRegex(RuntimeError, "forward failure"):
                attn.forward(None, latent, batch, None)
            self.assertFalse(attn._mla_gluon_vc_active)

    def test_model_hook_preserves_native_dispatch(self):
        from sglang.srt.models.deepseek_common.attention_forward_methods.forward_mla_rocm import (
            DeepseekMLARocmForwardMixin,
        )
        from sglang.srt.models.kimi_k3 import KimiK3MLAAttention

        attn = KimiK3MLAAttention.__new__(KimiK3MLAAttention)
        torch.nn.Module.__init__(attn)
        attn.__dict__.update(vars(make_attn()))
        latent = torch.empty((1, 12, 512), dtype=torch.bfloat16)
        native_out, fused_out = object(), object()
        with (
            mock.patch.object(
                DeepseekMLARocmForwardMixin, "_absorb_v_bmm", return_value=native_out
            ) as native,
            mock.patch.object(adapter, "apply", return_value=fused_out) as fused,
        ):
            attn._mla_gluon_vc_active = False
            self.assertIs(attn._absorb_v_bmm(latent), native_out)
            fused.assert_not_called()
            attn._mla_gluon_vc_active = True
            self.assertIs(attn._absorb_v_bmm(latent), fused_out)
            self.assertIs(attn._absorb_v_bmm(latent.float()), native_out)
            self.assertEqual(native.call_count, 2)
            self.assertEqual(fused.call_count, 1)


if __name__ == "__main__":
    unittest.main()
