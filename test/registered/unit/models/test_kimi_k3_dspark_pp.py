import unittest
from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import patch

import torch
from torch import nn

from sglang.srt.layers.attn_residual import AttnResidual, aggregate_stream_torch
from sglang.srt.layers.aux_hidden_states import pack_aux_hidden_states
from sglang.srt.layers.utils import PPMissingLayer
from sglang.srt.models.kimi_k3 import KimiK3LinearForCausalLM, KimiK3LinearModel
from sglang.srt.models.kimi_linear import KimiLinearForCausalLM, KimiLinearModel
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=10, suite="base-a-test-cpu")


class _Score(nn.Module):
    def __init__(self, index):
        super().__init__()
        self.index = index

    def forward(self, value):
        return value[..., :1] * (self.index + 1) / 10, None


class _Norm(nn.Module):
    def forward(self, value, residual=None):
        return value if residual is None else (value + residual, None)


class _CpuAttnResidual(AttnResidual):
    def forward(self, hidden, residual, proj, norm, out_norm):
        head = hidden if residual is None else hidden + residual
        return out_norm(
            aggregate_stream_torch(
                head, self.block_residual, self.num_valid_blocks, proj, norm
            )
        ), head


class _Layer(nn.Module):
    _sp_moe = False

    def __init__(self, index):
        super().__init__()
        self.index = index
        self.prev_valid_blocks = (index + 1) // 2
        self.self_attention_res_proj = _Score(index)
        self.self_attention_res_norm = nn.Identity()

    def forward(self, *, hidden_states, residual, attn_res, **kwargs):
        head = hidden_states if residual is None else hidden_states + residual
        if attn_res is not None and self.index % 2 == 0:
            attn_res.write(head)
        return head * 0.1 + self.index, head, False


def _model(start, end, captures, attn_res):
    model = KimiK3LinearModel.__new__(KimiK3LinearModel)
    nn.Module.__init__(model)
    model.config = SimpleNamespace(attn_res_block_size=2 if attn_res else None)
    model.pp_group = SimpleNamespace(
        is_first_rank=start == 0,
        is_last_rank=end == 6,
        world_size=1 if (start, end) == (0, 6) else 3,
    )
    model._trim_padded_attn = False
    model.start_layer, model.end_layer = start, end
    model.layers = nn.ModuleList(
        _Layer(i) if start <= i < end else PPMissingLayer() for i in range(6)
    )
    model.norm = _Norm() if end == 6 else PPMissingLayer()
    if end == 6:
        model.output_attn_res_proj = _Score(6)
        model.output_attn_res_norm = nn.Identity()
    owner = SimpleNamespace(pp_group=model.pp_group, model=model)
    KimiK3LinearForCausalLM.set_dspark_layers_to_capture(owner, captures)
    return model


def _forward(model, hidden, proxy=None):
    recorder = SimpleNamespace(with_current_layer=lambda _: nullcontext())
    with (
        patch("sglang.srt.models.kimi_k3.get_pp_group", return_value=model.pp_group),
        patch(
            "sglang.srt.models.kimi_k3.get_global_expert_distribution_recorder",
            return_value=recorder,
        ),
        patch("sglang.srt.models.kimi_k3.AttnResidual", _CpuAttnResidual),
        patch("sglang.srt.models.kimi_k3.aggregate_stream", aggregate_stream_torch),
    ):
        return model(
            None,
            torch.arange(hidden.shape[0]),
            SimpleNamespace(),
            inputs_embeds=hidden,
            pp_proxy_tensors=proxy,
        )


class TestKimiK3DSparkPP(CustomTestCase):
    def test_kimi_linear_partitioning_preserves_all_capture_streams(self):
        class Layer(nn.Module):
            def __init__(self, index):
                super().__init__()
                self.index = index

            def forward(self, *, hidden_states, residual, **kwargs):
                head = hidden_states if residual is None else hidden_states + residual
                return head * 0.1 + self.index, head

        def make_model(start, end, captures):
            model = KimiLinearModel.__new__(KimiLinearModel)
            nn.Module.__init__(model)
            model.start_layer, model.end_layer = start, end
            model.pp_group = SimpleNamespace(
                is_first_rank=start == 0,
                is_last_rank=end == 6,
                world_size=1 if (start, end) == (0, 6) else 3,
            )
            model.layers = nn.ModuleList(
                Layer(i) if start <= i < end else PPMissingLayer() for i in range(6)
            )
            model.norm = _Norm() if end == 6 else PPMissingLayer()
            KimiLinearForCausalLM.set_dspark_layers_to_capture(
                SimpleNamespace(model=model, pp_group=model.pp_group), captures
            )
            return model

        def forward(model, hidden, proxy=None):
            with (
                patch(
                    "sglang.srt.models.kimi_linear.get_pp_group",
                    return_value=model.pp_group,
                ),
                patch(
                    "sglang.srt.models.kimi_linear.get_global_expert_distribution_recorder",
                    return_value=SimpleNamespace(
                        with_current_layer=lambda _: nullcontext()
                    ),
                ),
            ):
                return model(
                    None,
                    torch.arange(hidden.shape[0]),
                    SimpleNamespace(),
                    inputs_embeds=hidden,
                    pp_proxy_tensors=proxy,
                )

        for captures in ([0, 1, 2, 3, 4, 5], [1, 3], [4, 5]):
            with self.subTest(captures=captures):
                hidden = torch.arange(12, dtype=torch.float32).reshape(3, 4) / 10
                expected, expected_aux = forward(make_model(0, 6, captures), hidden)
                proxy = forward(make_model(0, 2, captures), hidden)
                proxy = forward(make_model(2, 4, captures), hidden, proxy)
                actual, actual_aux = forward(make_model(4, 6, captures), hidden, proxy)
                torch.testing.assert_close(actual, expected)
                torch.testing.assert_close(
                    pack_aux_hidden_states(actual_aux),
                    pack_aux_hidden_states(expected_aux),
                )

    def test_partitioning_preserves_capture_order_and_boundary_mixtures(self):
        for attn_res in (False, True):
            for captures in ([0, 1, 2, 3, 4, 5], [1, 3], [4, 5]):
                with self.subTest(attn_res=attn_res, captures=captures):
                    hidden = torch.arange(12, dtype=torch.float32).reshape(3, 4) / 10
                    expected_output, expected_aux = _forward(
                        _model(0, 6, captures, attn_res), hidden
                    )
                    proxy = _forward(_model(0, 2, captures, attn_res), hidden)
                    proxy = _forward(_model(2, 4, captures, attn_res), hidden, proxy)
                    output, aux = _forward(
                        _model(4, 6, captures, attn_res), hidden, proxy
                    )
                    torch.testing.assert_close(output, expected_output)
                    torch.testing.assert_close(
                        pack_aux_hidden_states(aux),
                        pack_aux_hidden_states(expected_aux),
                    )


if __name__ == "__main__":
    unittest.main()
