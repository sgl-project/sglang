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
            layer.mixer = _Mixer(0.25)
        if kind == "M":
            layer._forward_mamba = lambda h, batch, mixer=layer.mixer: mixer(h)
        layers.append(layer)
    instance.layers = nn.ModuleList(layers)
    return instance


class TestNemotronAuxCapture(CustomTestCase):
    def test_capture_reduces_only_its_snapshot(self):
        """Each auxiliary snapshot equals the full hidden state at its boundary,
        also under DP attention and after later norms update the residual in place."""
        for pattern in ("*-", "M-", "**-", "*", "*--", "-*"):
            for dp_enabled, tp in ((True, 2), (True, 1), (False, 2)):
                with self.subTest(pattern=pattern, dp_enabled=dp_enabled, tp=tp):
                    self._check(pattern, dp_enabled, tp)

    def _check(self, pattern, dp_enabled, tp):
        def reduce_in_place(x):
            return x.mul_(tp)

        group = SimpleNamespace(all_reduce=reduce_in_place)
        batch = SimpleNamespace(
            input_ids=torch.zeros(2, dtype=torch.long),
            forward_mode=ForwardMode.DECODE,
            global_num_token_non_padded_cpu=2,
        )
        inputs = torch.tensor([[0.3, -0.5, 0.7, 1.1], [-0.4, 0.9, 0.2, -0.6]])
        with (
            get_context().override_server_args(
                tp_size=tp, enable_dp_attention=dp_enabled
            ),
            get_flags().dp.override(enabled=dp_enabled),
            get_parallel().override(
                attn_tp_group=group,
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
        ):
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
        for i in (*range(1, len(expected)), 0):
            torch.testing.assert_close(snapshots[i], expected[i], msg=f"Boundary {i}")


if __name__ == "__main__":
    unittest.main()
