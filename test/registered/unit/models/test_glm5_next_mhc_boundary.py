import sys
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock, patch

import torch

from sglang.srt.layers.layer_boundary.residual import batch as residual_batch
from sglang.srt.layers.layer_boundary.residual.mhc import MHCState
from sglang.srt.layers.layer_boundary.residual.stream import ResidualStream
from sglang.srt.models.glm5_next import Glm5NextDecoderLayer
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


def _layer(mock_packer=True):
    layer = Glm5NextDecoderLayer.__new__(Glm5NextDecoderLayer)
    torch.nn.Module.__init__(layer)
    layer.config = SimpleNamespace(
        mhc=True,
        hc_mult=4,
        rms_norm_eps=1e-6,
        hc_eps=1e-6,
        hc_sinkhorn_iters=20,
    )
    for stage in ("attn", "ffn"):
        setattr(
            layer,
            f"hc_{stage}_fn",
            torch.nn.Parameter(
                torch.empty(24, 4 * 4096, device="meta", dtype=torch.float32)
            ),
        )
        setattr(
            layer,
            f"hc_{stage}_scale",
            torch.nn.Parameter(torch.empty(3, device="meta", dtype=torch.float32)),
        )
        setattr(
            layer,
            f"hc_{stage}_base",
            torch.nn.Parameter(torch.empty(24, device="meta", dtype=torch.float32)),
        )
        layer.register_buffer(f"_hc_{stage}_fn_packed", None, persistent=False)
        setattr(layer, f"_hc_{stage}_fn_packed_source", None)
    if mock_packer:
        layer._get_hc_fn_packed = Mock(side_effect=lambda stage: f"packed_{stage}")
    return layer


def _inputs(m):
    return (
        torch.empty(m, 4096, device="meta", dtype=torch.bfloat16),
        torch.empty(m, 4 * 4096, device="meta", dtype=torch.bfloat16),
        torch.empty(m, 4 * 4, device="meta", dtype=torch.float32),
        torch.empty(m, 4, device="meta", dtype=torch.float32),
        torch.empty(4096, device="meta", dtype=torch.bfloat16),
    )


@patch("sglang.srt.models.glm5_next._use_aiter_gfx95", True)
@patch("sglang.srt.models.glm5_next.apply_mhc_post_pre_boundary")
def test_glm53_large_m_uses_packed_forced_fusion(mock_apply):
    layer = _layer()
    for stage in ("attn", "ffn"):
        callback = getattr(layer, f"hc_{stage}_post_pre")
        for m in (4096, 8192, 16384, 131072):
            hidden, residual, h_res, h_post, norm_weight = _inputs(m)
            mock_apply.return_value = (
                residual.view(m, 4, 4096),
                hidden,
                h_post.view(m, 4),
                h_res.view(m, 4, 4),
                True,
            )
            result = callback(
                hidden,
                residual,
                h_res,
                h_post,
                norm_weight,
                1e-6,
                True,
            )
            assert result is not None
            kwargs = mock_apply.call_args.kwargs
            assert kwargs["hc_fn"] == f"packed_{stage}"
            assert kwargs["force_fused"] is True
            assert kwargs["w_preshuffle_bf16"] is True


@patch("sglang.srt.models.glm5_next._use_aiter_gfx95", True)
@patch("sglang.srt.models.glm5_next.apply_mhc_post_pre_boundary")
def test_glm53_large_m_requires_retained_prefill_cell(mock_apply):
    layer = _layer()
    assert layer.hc_ffn_post_pre(*_inputs(2048), 1e-6, True) is None
    assert layer.hc_ffn_post_pre(*_inputs(4096), 1e-6, False) is None
    mock_apply.assert_not_called()
    layer._get_hc_fn_packed.assert_not_called()


@patch("sglang.srt.models.glm5_next._use_aiter_gfx95", True)
@patch("sglang.srt.models.glm5_next.apply_mhc_post_pre_boundary")
def test_glm53_small_decode_keeps_fp32_fusion(mock_apply):
    layer = _layer()
    hidden, residual, h_res, h_post, norm_weight = _inputs(16)
    mock_apply.return_value = (
        residual.view(16, 4, 4096),
        hidden,
        h_post.view(16, 4),
        h_res.view(16, 4, 4),
        True,
    )
    assert (
        layer.hc_ffn_post_pre(hidden, residual, h_res, h_post, norm_weight, 1e-6, False)
        is not None
    )
    kwargs = mock_apply.call_args.kwargs
    assert kwargs["hc_fn"] is layer.hc_ffn_fn
    assert kwargs["force_fused"] is False
    assert kwargs["w_preshuffle_bf16"] is False


def test_glm53_packed_weight_caches_track_parameter_versions():
    layer = _layer(mock_packer=False)
    pack = Mock(
        side_effect=lambda weight: torch.empty(
            weight.shape, device="meta", dtype=torch.int32
        )
    )
    modules = {
        "aiter": ModuleType("aiter"),
        "aiter.ops": ModuleType("aiter.ops"),
        "aiter.ops.mhc": ModuleType("aiter.ops.mhc"),
    }
    modules["aiter.ops.mhc"].mhc_shuffle_fn = pack
    with patch.dict(sys.modules, modules):
        for index, stage in enumerate(("attn", "ffn")):
            first = layer._get_hc_fn_packed(stage)
            second = layer._get_hc_fn_packed(stage)
            assert first is second
            assert pack.call_count == index * 2 + 1

            with torch.no_grad():
                getattr(layer, f"hc_{stage}_fn").add_(1)
            third = layer._get_hc_fn_packed(stage)
            assert third is not first
            assert pack.call_count == index * 2 + 2


def _forward_batch(prefill=True):
    return SimpleNamespace(
        forward_mode=SimpleNamespace(is_extend_without_speculative=lambda: prefill)
    )


def test_cross_layer_mhc_consumes_producer_coefficients_once():
    producer_post = Mock()
    consumer_fused = Mock()
    producer = MHCState(4, Mock(), Mock(), producer_post)
    consumer = MHCState(
        4,
        Mock(),
        Mock(),
        Mock(),
        hc_attn_post_pre=consumer_fused,
    )
    producer.h_res, producer.h_post = "producer_comb", "producer_post"
    hidden = torch.empty(4, 4096, device="meta")
    residual = torch.empty(4, 4 * 4096, device="meta")
    consumer_fused.return_value = (
        "attn_input",
        "next_residual",
        "next_comb",
        "next_post",
        True,
    )

    result = consumer.update_and_read_attn_input(
        producer,
        hidden,
        residual,
        forward_batch=_forward_batch(),
    )
    assert result == ("attn_input", "next_residual")
    assert producer.h_res is None and producer.h_post is None
    assert consumer.h_res == "next_comb" and consumer.h_post == "next_post"
    producer_post.assert_not_called()
    assert consumer_fused.call_args.kwargs["is_prefill"] is True


def test_cross_layer_mhc_fallback_preserves_post_then_pre_order():
    calls = []
    producer = MHCState(
        4,
        Mock(),
        Mock(),
        Mock(side_effect=lambda *args: calls.append("post") or "written"),
    )
    consumer = MHCState(
        4,
        Mock(
            side_effect=lambda *args: (
                calls.append("pre") or ("attn_input", "comb", "post", True)
            )
        ),
        Mock(),
        Mock(),
        hc_attn_post_pre=Mock(return_value=None),
    )
    producer.h_res, producer.h_post = "producer_comb", "producer_post"
    result = consumer.update_and_read_attn_input(
        producer,
        torch.empty(4, 4096, device="meta"),
        torch.empty(4, 4 * 4096, device="meta"),
        forward_batch=_forward_batch(),
    )
    assert result == ("attn_input", "written")
    assert calls == ["post", "pre"]
    assert producer.h_res is None and producer.h_post is None


def test_terminal_and_nonterminal_ffn_update_contracts():
    nonterminal = MHCState(
        4, Mock(), Mock(), Mock(), defer_ffn_update=True, is_last_layer=False
    )
    terminal = MHCState(4, Mock(), Mock(), Mock(), is_last_layer=True)
    nonterminal_update = nonterminal.residual_ops().ffn_update
    terminal_update = terminal.residual_ops().ffn_update
    assert nonterminal_update.applied_at_exit is False
    assert nonterminal_update.outlives_layer is True
    assert terminal_update.applied_at_exit is True
    assert terminal_update.outlives_layer is False


def test_pp_export_flushes_nonlinear_pending_update():
    update = SimpleNamespace(
        is_plain_add=False,
        update=Mock(return_value=torch.empty(2, 8, device="meta")),
    )
    stream = ResidualStream(torch.empty(2, 8, device="meta"))
    hidden = stream.record(torch.empty(2, 8, device="meta"), update)
    forward_batch = SimpleNamespace(residual_stream=stream)
    proxy = residual_batch.to_pp(hidden, forward_batch)
    update.update.assert_called_once()
    assert set(proxy.tensors) == {"hidden_states"}
