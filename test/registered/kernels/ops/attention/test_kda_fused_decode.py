"""Kimi-K3 fused KDA decode must match the existing unfused decode chain.

The fused kernel replaces:

    causal_conv1d_update -> kda_packed_decode -> sigmoid-gated RMSNorm

This file covers the local head layouts used by Kimi-K3 TP8/TP16/TP32
(H = 12/6/3) and by GLM-5.3-Flash TP8/TP4 with its gate lower bound
(H = 8/16), one per fixed-head branch of ``kda_fused_decode.cuh``.
"""

import pytest
import torch

from sglang.kernels.ops.attention import kda_fused_decode
from sglang.kernels.ops.attention.fla.fused_norm_gate import rms_norm_gated
from sglang.kernels.ops.attention.fla.fused_recurrent import (
    fused_recurrent_kda_packed_decode,
)
from sglang.kernels.ops.mamba.causal_conv1d_triton import causal_conv1d_update
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=8, stage="base-b-kernel-unit", runner_config="1-gpu-large")

_HEAD_DIM = 128
_CONV_STATE_W = 3
_FREE_SLOTS = 4
_BATCH = 4


def _randn(shape, dtype, generator, scale=1.0):
    return (torch.randn(shape, device="cuda", generator=generator) * scale).to(dtype)


def _make_case(heads: int, seed: int, batch: int):
    generator = torch.Generator(device="cuda").manual_seed(seed)
    seg = heads * _HEAD_DIM
    conv_dim = 3 * seg

    # Keep magnitudes moderate so fp32 state updates stay in a stable range.
    mixed_qkv = _randn((batch, conv_dim), torch.bfloat16, generator, scale=0.2)
    a = _randn((batch, seg), torch.bfloat16, generator, scale=0.2)
    b = _randn((batch, heads), torch.bfloat16, generator, scale=0.2)
    onorm_g = _randn((batch, seg), torch.bfloat16, generator, scale=0.2)

    conv_states = _randn(
        (batch + _FREE_SLOTS, _CONV_STATE_W, conv_dim),
        torch.bfloat16,
        generator,
        scale=0.2,
    )
    ssm_states = _randn(
        (batch + _FREE_SLOTS, heads, _HEAD_DIM, _HEAD_DIM),
        torch.float32,
        generator,
        scale=0.02,
    )
    cache_indices = torch.arange(batch, device="cuda", dtype=torch.int32)

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
        (mixed_qkv.shape[0], 1, heads, _HEAD_DIM),
        dtype=torch.bfloat16,
        device="cuda",
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
    "heads,tp_size,lower_bound,batch",
    [
        pytest.param(3, 32, None, _BATCH, id="tp32_h3"),
        pytest.param(6, 16, None, _BATCH, id="tp16_h6"),
        pytest.param(12, 8, None, _BATCH, id="tp8_h12"),
        pytest.param(8, 8, -5.0, _BATCH, id="glm53_flash_tp8_h8"),
        pytest.param(16, 4, -5.0, _BATCH, id="glm53_flash_tp4_h16"),
        # 576 and 1152 (token, head) blocks take the 3-stage state staging with
        # stage reuse; on SM90 the first takes the streaming state variant.
        pytest.param(8, 8, -5.0, 72, id="glm53_flash_tp8_h8_b72"),
        pytest.param(16, 4, -5.0, 72, id="glm53_flash_tp4_h16_b72"),
    ],
)
def test_kda_fused_decode_matches_unfused_chain(
    heads: int, tp_size: int, lower_bound, batch: int
):
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
    ) = _make_case(heads=heads, seed=20260731 + tp_size, batch=batch)

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
    print(f"Fused KDA decode test used fused path: TP{tp_size}, H={heads}")

    torch.testing.assert_close(fused, ref, rtol=2e-2, atol=2e-2)
    torch.testing.assert_close(state_fused, state_ref, rtol=2e-2, atol=2e-2)
    torch.testing.assert_close(conv_fused, conv_ref, rtol=0, atol=0)


# GLM-5.3-Flash publishes the fused decode inputs from its weight-loading hooks;
# a decoder stack of a pipeline-missing layer, one KDA layer and a non-KDA layer.
_KDA_LAYER = 1
_KDA_PREFIX = f"model.layers.{_KDA_LAYER}.self_attn"


@pytest.fixture
def glm5_next_runtime():
    from sglang.srt.runtime_context import get_context, get_parallel

    override = get_context().override_server_args(
        device="cuda", enable_lora=False, lora_paths=None
    )
    override.install()
    try:
        with get_parallel().override(
            tp_size=1,
            tp_rank=0,
            attn_tp_size=1,
            attn_tp_rank=0,
            attn_dp_size=1,
            attn_dp_rank=0,
            attn_cp_size=1,
            attn_cp_rank=0,
            moe_tp_size=1,
        ):
            yield
    finally:
        override.restore()


def _glm5_next_with_kda_layer(heads: int):
    """A real Glm5NextForConditionalGeneration holding only what its weight
    loading reads; the full constructor needs the vision tower and MoE stack."""
    from types import SimpleNamespace

    from torch import nn

    from sglang.srt.layers.utils.common import PPMissingLayer
    from sglang.srt.models import glm5_next

    cls = glm5_next.Glm5NextForConditionalGeneration
    model = cls.__new__(cls)
    nn.Module.__init__(model)
    model.config = SimpleNamespace(n_routed_experts=0, num_hidden_layers=3)
    model.num_fused_shared_experts = 0
    model.quant_config = None
    model.encoder_only = False
    default_dtype = torch.get_default_dtype()
    torch.set_default_dtype(torch.bfloat16)
    try:
        with torch.device("cuda"):
            kda_layer = nn.Module()
            kda_layer.self_attn = glm5_next.Glm5NextLinearAttention(
                layer_idx=_KDA_LAYER,
                hidden_size=64,
                config=SimpleNamespace(
                    linear_attn_config={
                        "head_dim": _HEAD_DIM,
                        "num_heads": heads,
                        "short_conv_kernel_size": 4,
                        "gate_lower_bound": -5.0,
                    }
                ),
                quant_config=None,
                rms_norm_eps=1e-6,
                prefix=_KDA_PREFIX,
            )
            other_layer = nn.Module()
            other_layer.self_attn = nn.Identity()
    finally:
        torch.set_default_dtype(default_dtype)
    model.model = nn.Module()
    model.model.layers = nn.ModuleList([PPMissingLayer(), kda_layer, other_layer])
    return model, kda_layer.self_attn


def _kda_checkpoint(heads: int, seed: int):
    generator = torch.Generator(device="cuda").manual_seed(seed)
    seg = heads * _HEAD_DIM
    return {
        "q_conv1d.weight": _randn((seg, 1, 4), torch.float32, generator, scale=0.1),
        "k_conv1d.weight": _randn((seg, 1, 4), torch.float32, generator, scale=0.1),
        "v_conv1d.weight": _randn((seg, 1, 4), torch.float32, generator, scale=0.1),
        "A_log": _randn((heads,), torch.float32, generator, scale=0.1),
        "dt_bias": _randn((seg,), torch.float32, generator, scale=0.1),
        "o_norm.weight": _randn((_HEAD_DIM,), torch.bfloat16, generator, scale=0.1)
        + 1.0,
    }


def _load(model, ckpt, *, through_load_weights: bool):
    from unittest.mock import patch

    from sglang.srt.models import glm5_next

    # MLA weight absorption has no layer here to work on.
    with patch.object(glm5_next.DeepseekV2WeightLoaderMixin, "post_load_weights"):
        if through_load_weights:
            model.load_weights(
                [(f"{_KDA_PREFIX}.{name}", w) for name, w in ckpt.items()]
            )
        else:
            # Loaders that bypass load_weights (dummy, sharded state) fill the
            # parameters and then call only post_load_weights.
            attn = model.model.layers[_KDA_LAYER].self_attn
            conv = torch.cat([ckpt[f"{p}_conv1d.weight"] for p in "qkv"])
            attn.qkv_conv1d.weight.data.copy_(conv)
            attn.A_log.data.copy_(ckpt["A_log"].view(1, 1, -1, 1))
            attn.dt_bias.data.copy_(ckpt["dt_bias"])
            attn.o_norm.weight.data.copy_(ckpt["o_norm.weight"])
            model.post_load_weights()


def _assert_fused_args_match(attn, ckpt, heads: int):
    args = attn.attn._k3_fused_decode_args
    assert args is not None and len(args) == 7
    w_q_t, w_k_t, w_v_t, conv_bias, a_log, onorm_w, onorm_eps = args
    for got, proj in ((w_q_t, "q"), (w_k_t, "k"), (w_v_t, "v")):
        want = ckpt[f"{proj}_conv1d.weight"].squeeze(1).t()
        torch.testing.assert_close(got, want, rtol=0, atol=0)
        assert got.is_contiguous()
    torch.testing.assert_close(
        conv_bias, torch.zeros(3 * heads * _HEAD_DIM, device="cuda"), rtol=0, atol=0
    )
    torch.testing.assert_close(a_log, ckpt["A_log"], rtol=0, atol=0)
    assert onorm_w.dtype == torch.float32
    torch.testing.assert_close(onorm_w, ckpt["o_norm.weight"].float(), rtol=0, atol=0)
    assert onorm_eps == pytest.approx(1e-6)
    hopper = torch.cuda.get_device_capability()[0] == 9
    assert attn._kda_fused_decode_max_tokens == (
        2048 // heads if hopper else float("inf")
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("heads", [8, 16], ids=["glm53_flash_h8", "glm53_flash_h16"])
@pytest.mark.parametrize("through_load_weights", [True, False], ids=["load", "post"])
def test_glm5_next_weight_loading_prepares_fused_decode(
    glm5_next_runtime, heads: int, through_load_weights: bool
):
    """Both weight-loading entry points must publish the fused decode inputs;
    without them every decode batch silently stays on the unfused chain."""
    model, attn = _glm5_next_with_kda_layer(heads)
    assert attn.attn._k3_fused_decode_args is None
    ckpt = _kda_checkpoint(heads, seed=1)
    _load(model, ckpt, through_load_weights=through_load_weights)
    _assert_fused_args_match(attn, ckpt, heads)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("heads", [8, 16], ids=["glm53_flash_h8", "glm53_flash_h16"])
def test_glm5_next_reload_refreshes_graph_held_fused_buffers(
    glm5_next_runtime, heads: int
):
    """Captured decode graphs hold the fused-decode buffers, so a weight reload
    must rewrite them in place: a graph captured before the reload replays
    with the new weights."""
    model, attn = _glm5_next_with_kda_layer(heads)
    _load(model, _kda_checkpoint(heads, seed=1), through_load_weights=True)
    args = attn.attn._k3_fused_decode_args
    pointers = [t.data_ptr() for t in args[:6]]

    (mixed_qkv, a, b, onorm_g, conv_states, ssm_states, cache_indices) = _make_case(
        heads=heads, seed=7, batch=_BATCH
    )[:7]
    conv_pool, state_pool = conv_states.clone(), ssm_states.clone()

    def fused_step():
        w_q_t, w_k_t, w_v_t, conv_bias, a_log, onorm_w, onorm_eps = args
        return kda_fused_decode.kda_fused_decode(
            mixed_qkv,
            a,
            b,
            conv_pool,
            w_q_t,
            w_k_t,
            w_v_t,
            conv_bias,
            a_log,
            attn.dt_bias,
            onorm_g,
            onorm_w,
            state_pool,
            cache_indices,
            scale=_HEAD_DIM**-0.5,
            onorm_eps=onorm_eps,
            lower_bound=-5.0,
        )

    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        fused_step()  # JIT build outside the capture
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        out = fused_step()

    new_ckpt = _kda_checkpoint(heads, seed=2)
    _load(model, new_ckpt, through_load_weights=True)
    assert [t.data_ptr() for t in attn.attn._k3_fused_decode_args[:6]] == pointers
    _assert_fused_args_match(attn, new_ckpt, heads)

    conv_pool.copy_(conv_states)
    state_pool.copy_(ssm_states)
    graph.replay()
    torch.cuda.synchronize()

    conv_ref, state_ref = conv_states.clone(), ssm_states.clone()
    conv_weights = torch.cat([new_ckpt[f"{p}_conv1d.weight"].squeeze(1) for p in "qkv"])
    ref = _run_unfused_reference(
        mixed_qkv.clone(),
        a,
        b,
        onorm_g,
        conv_ref,
        state_ref,
        cache_indices,
        conv_weights,
        torch.zeros(conv_weights.shape[0], device="cuda"),
        new_ckpt["A_log"],
        new_ckpt["dt_bias"],
        new_ckpt["o_norm.weight"].float(),
        -5.0,
    )
    torch.testing.assert_close(out, ref, rtol=2e-2, atol=2e-2)
    torch.testing.assert_close(state_pool, state_ref, rtol=2e-2, atol=2e-2)
    torch.testing.assert_close(conv_pool, conv_ref, rtol=0, atol=0)


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__]))
