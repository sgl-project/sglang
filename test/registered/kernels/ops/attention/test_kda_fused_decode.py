"""Kimi-K3 fused KDA decode must match the existing unfused decode chain.

The fused kernel replaces:

    causal_conv1d_update -> kda_packed_decode -> sigmoid-gated RMSNorm

This file covers the local head layouts used by Kimi-K3 TP8/TP16/TP32:
H = 12/6/3. The H=6 and H=3 cases are the branches added by the fixed-head
dispatch in ``kda_fused_decode.cuh``.
GLM-5.3-Flash TP8/TP4 add H = 8/16 with its gate lower bound.
"""

from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch

from sglang.kernels.ops.attention import kda_fused_decode
from sglang.kernels.ops.attention.fla.fused_norm_gate import rms_norm_gated
from sglang.kernels.ops.attention.fla.fused_recurrent import (
    fused_recurrent_kda_packed_decode,
)
from sglang.kernels.ops.mamba.causal_conv1d_triton import causal_conv1d_update
from sglang.srt.layers.utils.common import PPMissingLayer
from sglang.srt.models import glm5_next
from sglang.srt.runtime_context import get_context
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=8, stage="base-b-kernel-unit", runner_config="1-gpu-small")

_HEAD_DIM = 128
_CONV_STATE_W = 3
_SLOTS = 80
_BATCH = 72


def _randn(shape, dtype, generator, scale=1.0):
    return (torch.randn(shape, device="cuda", generator=generator) * scale).to(dtype)


def _make_case(heads: int, seed: int):
    generator = torch.Generator(device="cuda").manual_seed(seed)
    seg = heads * _HEAD_DIM
    conv_dim = 3 * seg

    # Keep magnitudes moderate so fp32 state updates stay in a stable range.
    mixed_qkv = _randn((_BATCH, conv_dim), torch.bfloat16, generator, scale=0.2)
    a = _randn((_BATCH, seg), torch.bfloat16, generator, scale=0.2)
    b = _randn((_BATCH, heads), torch.bfloat16, generator, scale=0.2)
    onorm_g = _randn((_BATCH, seg), torch.bfloat16, generator, scale=0.2)

    conv_states = _randn(
        (_SLOTS, _CONV_STATE_W, conv_dim), torch.bfloat16, generator, scale=0.2
    )
    ssm_states = _randn(
        (_SLOTS, heads, _HEAD_DIM, _HEAD_DIM), torch.float32, generator, scale=0.02
    )
    cache_indices = torch.arange(_BATCH, device="cuda", dtype=torch.int32)

    conv_weights = _randn((conv_dim, 4), torch.float32, generator, scale=0.1)
    conv_bias = _randn((conv_dim,), torch.float32, generator, scale=0.05)
    a_log = _randn((heads,), torch.float32, generator, scale=0.1)
    dt_bias = _randn((seg,), torch.float32, generator, scale=0.1)
    onorm_weight = _randn((_HEAD_DIM,), torch.float32, generator, scale=0.1) + 1.0

    return (
        mixed_qkv,
        a,
        b,
        onorm_g,
        conv_states,
        ssm_states,
        cache_indices,
        conv_weights,
        conv_bias,
        a_log,
        dt_bias,
        onorm_weight,
    )


def _run_unfused_reference(
    mixed_qkv,
    a,
    b,
    onorm_g,
    conv_states,
    ssm_states,
    cache_indices,
    conv_weights,
    conv_bias,
    a_log,
    dt_bias,
    onorm_weight,
    lower_bound,
):
    heads = ssm_states.shape[-3]
    qkv = causal_conv1d_update(
        mixed_qkv,
        conv_states.transpose(-1, -2),
        conv_weights,
        conv_bias,
        activation="silu",
        conv_state_indices=cache_indices,
    )
    out = torch.empty(
        (mixed_qkv.shape[0], 1, heads, _HEAD_DIM), dtype=torch.bfloat16, device="cuda"
    )
    out, _ = fused_recurrent_kda_packed_decode(
        qkv,
        a,
        b,
        a_log,
        dt_bias,
        _HEAD_DIM**-0.5,
        ssm_states,
        out,
        cache_indices,
        use_qk_l2norm_in_kernel=True,
        lower_bound=lower_bound,
    )
    ref = rms_norm_gated(
        out,
        onorm_g.view(1, -1, heads, _HEAD_DIM),
        onorm_weight,
        None,
        activation="sigmoid",
        eps=1e-6,
    )
    return ref.transpose(0, 1).contiguous()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize(
    "heads,tp_size",
    [
        pytest.param(3, 32, id="tp32_h3"),
        pytest.param(6, 16, id="tp16_h6"),
        pytest.param(12, 8, id="tp8_h12"),
        pytest.param(8, 8, id="glm53_flash_tp8_h8"),
        pytest.param(16, 4, id="glm53_flash_tp4_h16"),
    ],
)
# Batch 72 takes the 3-stage state staging at H >= 8, on SM90 with streaming at H = 8.
@pytest.mark.parametrize("batch", [4, _BATCH])
def test_kda_fused_decode_matches_unfused_chain(heads: int, tp_size: int, batch: int):
    lower_bound = -5.0 if heads in (8, 16) else None
    (
        mixed_qkv,
        a,
        b,
        onorm_g,
        conv_states,
        ssm_states,
        cache_indices,
        conv_weights,
        conv_bias,
        a_log,
        dt_bias,
        onorm_weight,
    ) = _make_case(heads=heads, seed=20260731 + tp_size)
    mixed_qkv, a, b, onorm_g, cache_indices = (
        t[:batch] for t in (mixed_qkv, a, b, onorm_g, cache_indices)
    )

    conv_ref = conv_states.clone()
    conv_fused = conv_states.clone()
    state_ref = ssm_states.clone()
    state_fused = ssm_states.clone()

    w_q_t, w_k_t, w_v_t = [
        weight.t().contiguous()
        for weight in conv_weights.split(heads * _HEAD_DIM, dim=0)
    ]

    assert kda_fused_decode.covered(
        mixed_qkv,
        a,
        b,
        conv_fused,
        state_fused,
        cache_indices,
        onorm_g,
    )

    ref = _run_unfused_reference(
        mixed_qkv.clone(),
        a,
        b,
        onorm_g,
        conv_ref,
        state_ref,
        cache_indices,
        conv_weights,
        conv_bias,
        a_log,
        dt_bias,
        onorm_weight,
        lower_bound,
    )
    fused = kda_fused_decode.kda_fused_decode(
        mixed_qkv.clone(),
        a,
        b,
        conv_fused,
        w_q_t,
        w_k_t,
        w_v_t,
        conv_bias,
        a_log,
        dt_bias,
        onorm_g,
        onorm_weight,
        state_fused,
        cache_indices,
        scale=_HEAD_DIM**-0.5,
        onorm_eps=1e-6,
        lower_bound=lower_bound,
    )
    torch.cuda.synchronize()

    # JIT log breadcrumb for PR/CI evidence that the fused fixed-head branch ran.
    print(f"K3 fused KDA decode test used fused path: TP{tp_size}, H={heads}")

    torch.testing.assert_close(fused, ref, rtol=2e-2, atol=2e-2)
    torch.testing.assert_close(state_fused, state_ref, rtol=2e-2, atol=2e-2)
    torch.testing.assert_close(conv_fused, conv_ref, rtol=0, atol=0)


# A decoder stack of a pipeline-missing layer, one KDA layer and a non-KDA layer.
_KDA_PREFIX = "model.layers.1.self_attn"
_GLM_KDA = {"head_dim": 128, "short_conv_kernel_size": 4, "gate_lower_bound": -5.0}


def _glm5_next_model(heads: int):
    """A real GLM model holding only what its weight loading reads."""
    cls = glm5_next.Glm5NextForConditionalGeneration
    model = cls.__new__(cls)
    torch.nn.Module.__init__(model)
    model.config = SimpleNamespace(n_routed_experts=0)
    model.num_fused_shared_experts = 0
    model.quant_config = None
    model.encoder_only = False
    with torch.device("cuda"), get_context().override_server_args(enable_lora=False):
        attn = glm5_next.Glm5NextLinearAttention(
            layer_idx=1,
            hidden_size=64,
            config=SimpleNamespace(linear_attn_config={**_GLM_KDA, "num_heads": heads}),
            prefix=_KDA_PREFIX,
        )
    attn.o_norm.bfloat16()
    model.model = torch.nn.Module()
    model.model.layers = torch.nn.ModuleList(
        [PPMissingLayer()]
        + [torch.nn.ModuleDict({"self_attn": m}) for m in (attn, torch.nn.Identity())]
    )
    return model, attn


def _load_weights(model, attn, hook: str, seed: int):
    generator = torch.Generator(device="cuda").manual_seed(seed)
    conv = _randn((3 * attn.num_heads * _HEAD_DIM, 1, 4), torch.float32, generator)
    a_log = _randn((attn.num_heads,), torch.float32, generator)
    onorm = _randn((_HEAD_DIM,), torch.bfloat16, generator)
    # MLA weight absorption has no layer here to work on.
    with patch.object(glm5_next.DeepseekV2WeightLoaderMixin, "post_load_weights"):
        if hook == "load_weights":
            weights = [(f"{p}_conv1d.weight", w) for p, w in zip("qkv", conv.chunk(3))]
            weights += [("A_log", a_log), ("o_norm.weight", onorm)]
            model.load_weights([(f"{_KDA_PREFIX}.{n}", w) for n, w in weights])
        else:
            # Loaders that skip load_weights fill the parameters, then call this.
            attn.qkv_conv1d.weight.data.copy_(conv)
            attn.A_log.data.copy_(a_log.view(1, 1, -1, 1))
            attn.o_norm.weight.data.copy_(onorm)
            model.post_load_weights()
    return conv, a_log, onorm


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("heads", [8, 16])
@pytest.mark.parametrize("hook", ["load_weights", "post_load_weights"])
@pytest.mark.parametrize("device_sm", [80, 90, 100, 120])
def test_glm5_next_weight_loading_publishes_fused_decode_inputs(
    heads: int, hook: str, device_sm: int
):
    """Both hooks publish the fused decode inputs and later loads rewrite them in
    place, since captured graphs hold them; off SM9x/SM10x nothing is prepared."""
    model, attn = _glm5_next_model(heads)
    with patch.object(glm5_next, "_device_sm", device_sm):
        _load_weights(model, attn, hook, seed=1)
        if device_sm not in (90, 100):
            assert attn.attn._k3_fused_decode_args is None
            assert attn._kda_fused_decode_max_tokens == 0
            assert attn._direct_weight_update_error is None
            return
        pointers = [t.data_ptr() for t in attn.attn._k3_fused_decode_args[:6]]
        conv, a_log, onorm = _load_weights(model, attn, hook, seed=2)
    *tensors, onorm_eps = attn.attn._k3_fused_decode_args
    assert [t.data_ptr() for t in tensors] == pointers
    assert all(t.is_contiguous() for t in tensors)
    assert torch.equal(torch.cat(tensors[:3], dim=1), conv.squeeze(1).t())
    assert not tensors[3].any() and tensors[3].numel() == conv.shape[0]
    assert torch.equal(tensors[4], a_log)
    assert tensors[5].dtype == torch.float32 and torch.equal(tensors[5], onorm.float())
    assert onorm_eps == attn.o_norm.eps
    cap = 2048 // heads if device_sm == 90 else float("inf")
    assert attn._kda_fused_decode_max_tokens == cap
    assert attn._direct_weight_update_error is not None


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__]))
