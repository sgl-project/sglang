import sys
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch
from torch import nn

from sglang.srt.environ import envs
from sglang.srt.layers.communicator import CommunicateSimpleFn, ScatterMode
from sglang.srt.layers.communicator_mhc import (
    MHCCommunicateSummableTensorPairFn,
    MHCCommunicateWithAllReduceAndLayerNormFn,
    MHCLayerCommunicator,
    MHCPostPreResult,
    MHCState,
)
from sglang.srt.models import glm5_next
from sglang.srt.models.glm5_next import Glm5NextDecoderLayer, Glm5NextModel, Glm5NextMoE
from sglang.srt.runtime_context import get_context, get_forward, get_parallel
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


@pytest.mark.parametrize(
    "case",
    [
        "eligible",
        "zero",
        "seven",
        "eight",
        "nine",
        "prefill",
        "cpu",
        "dtype",
        "sm",
        "plain",
        "hc",
        "hidden",
        "pp",
        "cp",
        "dp",
        "tbo",
        "aux",
        "dflash",
        "compile",
        "fusion",
        "pre",
        "post",
    ],
)
def test_cross_layer_mhc_guard_preserves_forward_contracts(case):
    model = Glm5NextModel.__new__(Glm5NextModel)
    nn.Module.__init__(model)
    model.config = SimpleNamespace(
        mhc=case != "plain",
        hc_mult=2 if case == "hc" else 4,
        hidden_size=7168 if case == "hidden" else 4096,
    )
    model.pp_group = SimpleNamespace(world_size=2 if case == "pp" else 1)
    model.layers_to_capture = [1] if case == "aux" else []
    model.dflash_capture = case == "dflash"
    batch = SimpleNamespace(can_run_tbo=case == "tbo")
    hidden = SimpleNamespace(
        shape=(
            {"zero": 0, "seven": 7, "eight": 8, "nine": 9, "prefill": 8000}.get(
                case, 6
            ),
            4096,
        ),
        dtype=torch.float32 if case == "dtype" else torch.bfloat16,
        is_cuda=case != "cpu",
    )
    with (
        get_context().override_server_args(),
        get_parallel().override(
            attn_cp_size=2 if case == "cp" else 1, attn_dp_size=2 if case == "dp" else 1
        ),
        envs.SGLANG_OPT_FUSE_MHC_POST_PRE.override(case != "fusion"),
        envs.SGLANG_OPT_USE_TILELANG_MHC_PRE.override(case != "pre"),
        envs.SGLANG_OPT_USE_TILELANG_MHC_POST.override(case != "post"),
        patch.object(glm5_next, "_is_cuda", True),
        patch.object(glm5_next, "_device_sm", 100 if case == "sm" else 103),
        patch.object(torch.compiler, "is_compiling", return_value=case == "compile"),
    ):
        assert model._can_fuse_mhc_layers(hidden, batch) == (
            case in {"eligible", "seven", "eight"}
        )


class _Attention(nn.Module):
    def forward(self, hidden_states, **kwargs):
        return hidden_states * 3


class _BufferAwareMoE(Glm5NextMoE):
    def __init__(self):
        nn.Module.__init__(self)
        self.experts = SimpleNamespace(moe_runner_config=SimpleNamespace(inplace=False))
        self.observed_buffer = None

    def forward(self, hidden_states, *args):
        self.observed_buffer = get_forward().moe_output_buffer
        result = hidden_states * 2
        if self.observed_buffer.shape == result.shape:
            self.observed_buffer.copy_(result)
            return self.observed_buffer
        return result


def test_cross_layer_decoder_preserves_moe_buffer_and_final_contract():
    def post(x, residual, comb, post_mix):
        return (
            torch.bmm(comb.unflatten(1, (2, 2)), residual.unflatten(1, (2, 4)))
            + x[:, None] * post_mix[:, :, None]
        ).flatten(1)

    def pre(residual, weight, eps):
        n = residual.shape[0]
        return (
            residual.unflatten(1, (2, 4)).mean(1),
            torch.eye(2).expand(n, 2, 2).flatten(1),
            torch.ones(n, 2),
            False,
        )

    def fused(x, residual, comb, post_mix, weight, eps):
        residual = post(x, residual, comb, post_mix)
        x, comb, post_mix, norm_fused = pre(residual, weight, eps)
        return MHCPostPreResult(x, residual, comb, post_mix, norm_fused)

    layer = Glm5NextDecoderLayer.__new__(Glm5NextDecoderLayer)
    nn.Module.__init__(layer)
    layer.self_attn = _Attention()
    layer.mlp = _BufferAwareMoE()
    layer.layer_scatter_modes = SimpleNamespace(
        layer_input_mode=ScatterMode.TP_ATTN_FULL
    )
    communicator = MHCLayerCommunicator.__new__(MHCLayerCommunicator)
    communicator.is_first_layer = False
    communicator.is_last_layer = True
    communicator.allow_reduce_scatter = False
    communicator.input_layernorm = communicator.post_attention_layernorm = None
    communicator.qkv_latent_func = None
    communicator._context = None
    communicator._communicate_simple_fn = CommunicateSimpleFn._trivial
    communicator._communicate_with_all_reduce_and_layer_norm_fn = (
        MHCCommunicateWithAllReduceAndLayerNormFn._simple
    )
    communicator._communicate_summable_tensor_pair_fn = (
        MHCCommunicateSummableTensorPairFn._trivial
    )
    communicator.mhc = MHCState(2, pre, pre, post, hc_post_attn_pre=fused)
    layer.layer_communicator = communicator
    hidden = torch.arange(8, dtype=torch.float32).reshape(2, 4)
    residual = torch.arange(16, dtype=torch.float32).reshape(2, 8)
    comb = torch.tensor([[0.7, 0.3], [0.2, 0.8]]).expand(2, 2, 2).flatten(1)
    post_mix = torch.tensor([0.4, 0.9]).expand(2, 2)
    materialized = post(hidden, residual, comb, post_mix)
    expected, expected_residual, _ = layer(torch.arange(2), materialized, None, None)
    previous = MHCState(2, pre, pre, post, h_res=comb, h_post=post_mix)
    actual, actual_residual, topk = layer(
        torch.arange(2), hidden, None, residual, previous_mhc=previous
    )
    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(layer.mlp.observed_buffer, materialized)
    assert layer.mlp.observed_buffer.shape == (2, 8)
    assert actual.shape == hidden.shape
    assert actual_residual is expected_residual is topk is None
    assert previous.h_res is previous.h_post is None
    assert communicator.mhc.h_res is communicator.mhc.h_post is None
    assert get_forward().moe_output_buffer is None


if __name__ == "__main__":
    sys.exit(pytest.main([__file__]))
