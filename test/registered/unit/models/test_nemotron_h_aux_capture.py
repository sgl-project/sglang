import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch
from torch import nn

from sglang.srt.layers import communicator as comm
from sglang.srt.layers.moe.utils import should_skip_mlp_all_reduce
from sglang.srt.model_executor.forward_batch_info import ForwardMode
from sglang.srt.models import nemotron_h as model
from sglang.srt.runtime_context import get_context, get_flags, get_parallel
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


def _rms(x):
    return x * torch.rsqrt(x.square().mean(-1, keepdim=True) + 1e-5)


class _Norm(nn.Module):
    def forward(self, x, residual=None, post_residual_addition=None):
        if residual is None:
            return _rms(x)
        residual.add_(x)
        return _rms(residual), residual


class _Mixer(nn.Module):
    """Row-parallel stand-in: a TP partial, reduced unless the layer skips it."""

    def __init__(self, scale, tp=1):
        super().__init__()
        self.scale = scale
        self.tp = tp

    def forward(self, hidden_states, **kwargs):
        partial = hidden_states * (self.scale / self.tp)
        return partial if should_skip_mlp_all_reduce() else partial * self.tp


class _Stack:
    """Enter several context managers as one."""

    def __init__(self, *managers):
        self.managers = managers

    def __enter__(self):
        for manager in self.managers:
            manager.__enter__()
        return self

    def __exit__(self, *exc_info):
        for manager in reversed(self.managers):
            manager.__exit__(*exc_info)


def _comms(layers):
    return [layer.layer_communicator for layer in layers]


def _build(pattern, tp, capture):
    config = SimpleNamespace(hybrid_override_pattern=pattern)
    instance = model.NemotronHModel.__new__(model.NemotronHModel)
    nn.Module.__init__(instance)
    instance.config = config
    instance.pp_group = SimpleNamespace(is_first_rank=True, is_last_rank=True)
    instance.start_layer, instance.end_layer = 0, len(pattern)
    instance.norm_f = _Norm()
    instance.layers_to_capture = set(range(len(pattern) + 1)) if capture else set()
    layers = []
    for i, kind in enumerate(pattern):
        cls = model.ALL_DECODER_LAYER_TYPES[kind]
        layer = cls.__new__(cls)
        nn.Module.__init__(layer)
        layer.norm = _Norm()
        if kind in "M*":
            layer._init_layer_communicator(config, i)
            layer.mixer = _Mixer(0.5, tp)
        else:
            layer._init_layer_communicator(config, i, is_sparse=False)
            layer.mixer = _Mixer(0.25, tp)
        if kind == "M":
            layer._forward_mamba = lambda h, batch, mixer=layer.mixer: mixer(h)
        layers.append(layer)
    instance.layers = nn.ModuleList(layers)
    return instance


class TestNemotronAuxCapture(CustomTestCase):
    def test_capture_reduces_only_its_snapshot(self):
        """Each auxiliary snapshot equals the full hidden state at its boundary,
        and the output equals the reference, so every stage's sum is completed
        exactly once, whichever stage completes it; also under DP attention and
        after later norms update the residual in place."""
        patterns = ("*-", "M-", "**-", "*", "*--", "-*", "M*-", "-", "M", "--")
        for pattern in patterns + ("*M-*", "-M-", "E*E"):
            for dp_enabled, tp in ((True, 2), (True, 1), (False, 2)):
                with self.subTest(pattern=pattern, dp_enabled=dp_enabled, tp=tp):
                    self._check(pattern, dp_enabled, tp)

    def test_each_stage_declares_its_side_from_the_pattern(self):
        with self._context(dp_enabled=False, tp=2):
            layers = _build("M-*E-", 2, False).layers
        mixer_before_ffn = [c.next_takes_attention_partial for c in _comms(layers)]
        self.assertEqual(mixer_before_ffn, [True, False, True, False, False])
        self.assertEqual(
            [c.standalone_ffn for c in _comms(layers)],
            [False, True, False, True, True],
        )
        self.assertEqual(
            [c.previous_leaves_attention_partial for c in _comms(layers)],
            [False, True, False, True, False],
        )
        self.assertEqual(
            [c.allow_deferred_ffn_reduction for c in _comms(layers)],
            [True, True, True, False, True],
        )
        self.assertEqual(
            [c.is_last_layer for c in _comms(layers)], [False] * 4 + [True]
        )
        for layer, c in zip(layers, _comms(layers)):
            if c.standalone_ffn:
                self.assertIsNone(c.input_layernorm)
                self.assertIs(c.post_attention_layernorm, layer.norm)
            else:
                self.assertIs(c.input_layernorm, layer.norm)
                self.assertIsNone(c.post_attention_layernorm)

    def _context(self, *, dp_enabled, tp, group=None):
        group = group or SimpleNamespace(all_reduce=lambda x: x.mul_(tp))
        return _Stack(
            get_context().override_server_args(
                tp_size=tp, enable_dp_attention=dp_enabled
            ),
            get_flags().dp.override(enabled=dp_enabled),
            get_parallel().override(
                # Attention TP and MoE TP are TP here, so the same group object.
                attn_tp_group=group,
                tp_group=group,
                moe_tp_group=group,
                launch_world_rank=0,
                tp_rank=0,
                tp_size=tp,
                attn_tp_rank=0,
                attn_tp_size=tp,
                attn_dp_rank=0,
                attn_dp_size=1,
                attn_cp_rank=0,
                attn_cp_size=1,
                moe_tp_rank=0,
                moe_tp_size=tp,
                moe_ep_rank=0,
                moe_ep_size=1,
                moe_dp_rank=0,
                moe_dp_size=1,
            ),
            patch.object(comm, "get_moe_cp_size", return_value=1),
            patch.object(comm, "apply_flashinfer_allreduce_fusion", return_value=False),
            patch.object(comm, "apply_aiter_all_reduce_fusion", return_value=False),
        )

    def _check(self, pattern, dp_enabled, tp):
        batch = SimpleNamespace(
            input_ids=torch.zeros(2, dtype=torch.long),
            forward_mode=ForwardMode.DECODE,
            global_num_token_non_padded_cpu=2,
        )
        inputs = torch.tensor([[0.3, -0.5, 0.7, 1.1], [-0.4, 0.9, 0.2, -0.6]])
        with self._context(dp_enabled=dp_enabled, tp=tp):
            baseline = _build(pattern, tp, False)(
                batch.input_ids, torch.arange(2), batch, inputs_embeds=inputs.clone()
            )
            output, snapshots = _build(pattern, tp, True)(
                batch.input_ids, torch.arange(2), batch, inputs_embeds=inputs.clone()
            )
        torch.testing.assert_close(output, baseline)
        expected = [inputs.clone()]
        for kind in pattern:
            value = expected[-1]
            expected.append(value + _rms(value) * (0.5 if kind in "M*" else 0.25))
        torch.testing.assert_close(output, _rms(expected[-1]))
        for i in (*range(1, len(expected)), 0):
            torch.testing.assert_close(snapshots[i], expected[i], msg=f"Boundary {i}")


if __name__ == "__main__":
    unittest.main()
