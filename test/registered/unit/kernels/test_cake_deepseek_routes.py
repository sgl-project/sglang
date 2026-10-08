"""Unit tests for the DeepSeek P1 Cake routes (``SGLANG_CAKE_ROUTES``):

* ``dsv3_grouped_routing`` in ``sglang.srt.layers.moe.topk.biased_grouped_topk_gpu``
* ``dsv4_sparse_mla_decode`` helpers in
  ``sglang.srt.layers.attention.dsv4.cake_routes`` (SM100/103 trtllm site and
  the SM120/121 DeepSeek-V4.1 mixed-cache decode site)

Everything is mocked: the route switch, the adapter admission, the Cake
forwarders and the stock kernels.  The tests only check *which* callable
receives the engine's tensors, that the Cake branch keeps the stock return
contract, and the fallback rules (admission False, FlashInfer host rejection,
CUDA-graph capture without eager warm-up).  CPU tensors; no FlashInfer, Triton
kernels or CUDA involved.
"""

import contextlib
import importlib
import logging
import sys
from unittest import mock

import pytest
import torch

from sglang.srt.layers.attention.dsv4 import cake_routes as dsv4_routes
from sglang.srt.layers.moe import topk as topk_mod
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=10, stage="base-a-test-cpu")


def _stack(*managers):
    stack = contextlib.ExitStack()
    for manager in managers:
        stack.enter_context(manager)
    return stack


@pytest.fixture(autouse=True)
def _reset_route_state():
    topk_mod.reset_cake_route_state_for_tests()
    dsv4_routes.reset_cake_route_state_for_tests()
    yield
    topk_mod.reset_cake_route_state_for_tests()
    dsv4_routes.reset_cake_route_state_for_tests()


# ---------------------------------------------------------------------------
# dsv3_grouped_routing (topk.biased_grouped_topk_gpu)
# ---------------------------------------------------------------------------

T, E, G, TG, K = 4, 256, 8, 4, 8  # DeepSeek-V3 routing shape
RSF = 2.5


def _routing_inputs():
    hidden = torch.randn(T, 16).bfloat16()
    gating = torch.randn(T, E).bfloat16()
    bias = torch.randn(E, dtype=torch.float32)
    return hidden, gating, bias


def _fill(value):
    def side_effect(scores, bias, n_group, topk_group, topk, scaling, w, ids, pdl):
        w.fill_(value)
        ids.fill_(int(value))

    return side_effect


def _routing_env(*routes, stock, supports, cake):
    return _stack(
        mock.patch.object(topk_mod, "_is_cuda", True),
        mock.patch.object(topk_mod, "_use_aiter", False),
        mock.patch.object(topk_mod, "fused_topk_deepseek", stock),
        mock.patch.object(topk_mod, "cake_route_enabled", lambda n: n in routes),
        mock.patch.object(
            topk_mod, "_cake_fused_topk_deepseek_kernels", lambda: (supports, cake)
        ),
        mock.patch.object(
            topk_mod.envs.SGLANG_OPT_USE_JIT_KERNEL_GROUPED_TOPK,
            "get",
            return_value=False,
        ),
    )


def _route(hidden, gating, bias, **overrides):
    kwargs = dict(
        topk=K,
        renormalize=True,
        num_expert_group=G,
        topk_group=TG,
        num_fused_shared_experts=0,
        routed_scaling_factor=RSF,
        apply_routed_scaling_factor_on_output=False,
    )
    kwargs.update(overrides)
    return topk_mod.biased_grouped_topk_gpu(hidden, gating, bias, **kwargs)


def test_routing_route_off_uses_flashinfer_default():
    stock = mock.Mock(side_effect=_fill(1.0))
    supports = mock.Mock(return_value=True)
    cake = mock.Mock(side_effect=_fill(2.0))
    hidden, gating, bias = _routing_inputs()
    with _routing_env(stock=stock, supports=supports, cake=cake):
        weights, ids = _route(hidden, gating, bias)
    stock.assert_called_once()
    supports.assert_not_called()
    cake.assert_not_called()
    assert tuple(weights.shape) == (T, K) and torch.all(weights == 1.0)
    assert ids.dtype == torch.int32 and weights.dtype == torch.float32


def test_routing_route_on_admitted_uses_cake_with_engine_tensors(caplog):
    caplog.set_level(logging.INFO, logger=topk_mod.logger.name)
    stock = mock.Mock(side_effect=_fill(1.0))
    supports = mock.Mock(return_value=True)
    cake = mock.Mock(side_effect=_fill(2.0))
    hidden, gating, bias = _routing_inputs()
    with _routing_env(
        "dsv3_grouped_routing", stock=stock, supports=supports, cake=cake
    ):
        weights, ids = _route(hidden, gating, bias)
    stock.assert_not_called()
    cake.assert_called_once()
    args, kw = cake.call_args
    assert not kw  # positional FlashInfer contract, exactly as the stock branch
    scores, c_bias, n_group, topk_group, topk, scaling, w, i, pdl = args
    assert scores.dtype == torch.float32 and torch.equal(scores, gating.float())
    assert c_bias is bias
    assert (n_group, topk_group, topk) == (G, TG, K)
    assert scaling == 1.0  # flashinfer applies the factor internally
    assert w.dtype == torch.float32 and tuple(w.shape) == (T, K)
    assert i.dtype == torch.int32 and tuple(i.shape) == (T, K)
    assert pdl is True
    # Admission saw the engine's router logits and bias.
    s_args, s_kw = supports.call_args
    assert s_args[0] is gating and s_args[1] is bias
    assert s_kw == {"n_group": G, "topk_group": TG, "topk": K}
    # Same return contract as the stock branch.
    assert weights is w and ids is i and torch.all(weights == 2.0)
    assert "[cake-route] dsv3_grouped_routing: Cake kernel selected" in caplog.text


def test_routing_scaling_factor_follows_stock_convention():
    stock = mock.Mock(side_effect=_fill(1.0))
    supports = mock.Mock(return_value=True)
    cake = mock.Mock(side_effect=_fill(2.0))
    hidden, gating, bias = _routing_inputs()
    with _routing_env(
        "dsv3_grouped_routing", stock=stock, supports=supports, cake=cake
    ):
        _route(hidden, gating, bias, apply_routed_scaling_factor_on_output=True)
    assert cake.call_args.args[5] == RSF


def test_routing_fused_shared_experts_appended_like_stock():
    stock = mock.Mock(side_effect=_fill(1.0))
    supports = mock.Mock(return_value=True)
    cake = mock.Mock(side_effect=_fill(2.0))
    hidden, gating, bias = _routing_inputs()
    with _routing_env(
        "dsv3_grouped_routing", stock=stock, supports=supports, cake=cake
    ):
        weights, ids = _route(
            hidden, gating, bias, topk=K + 1, num_fused_shared_experts=1
        )
    assert cake.call_args.args[4] == K  # routed experts only
    assert tuple(weights.shape) == (T, K + 1) and tuple(ids.shape) == (T, K + 1)
    assert torch.all(ids[:, K] == E)
    assert torch.allclose(weights[:, K], torch.full((T,), K * 2.0 / RSF))


def test_routing_route_on_rejected_falls_back(caplog):
    caplog.set_level(logging.INFO, logger=topk_mod.logger.name)
    stock = mock.Mock(side_effect=_fill(1.0))
    supports = mock.Mock(return_value=False)
    cake = mock.Mock(side_effect=_fill(2.0))
    hidden, gating, bias = _routing_inputs()
    with _routing_env(
        "dsv3_grouped_routing", stock=stock, supports=supports, cake=cake
    ):
        weights, _ = _route(hidden, gating, bias)
        _route(hidden, gating, bias)
    assert stock.call_count == 2 and supports.call_count == 2
    cake.assert_not_called()
    assert torch.all(weights == 1.0)
    assert caplog.text.count("[cake-route] dsv3_grouped_routing: fallback") == 1
    assert "adapter admission rejected" in caplog.text


def test_routing_flashinfer_host_rejection_is_cached(caplog):
    caplog.set_level(logging.INFO, logger=topk_mod.logger.name)
    stock = mock.Mock(side_effect=_fill(1.0))
    supports = mock.Mock(return_value=True)
    cake = mock.Mock(side_effect=ValueError("no cake routing kernel"))
    hidden, gating, bias = _routing_inputs()
    with _routing_env(
        "dsv3_grouped_routing", stock=stock, supports=supports, cake=cake
    ):
        weights, _ = _route(hidden, gating, bias)
        _route(hidden, gating, bias)
    assert stock.call_count == 2
    assert cake.call_count == 1 and supports.call_count == 1  # key not retried
    assert torch.all(weights == 1.0)
    assert "no cake routing kernel" in caplog.text


# ---------------------------------------------------------------------------
# dsv4_sparse_mla_decode: SM100/103 trtllm site
# ---------------------------------------------------------------------------

BS, HEADS, SWA_TOPK = 2, 64, 128


def _trtllm_inputs(bs=BS, topk=SWA_TOPK):
    q = torch.randn(bs, 1, HEADS, 512).to(torch.float8_e4m3fn)
    swa = torch.zeros(4, 1, 256, 512, dtype=torch.float8_e4m3fn)
    return dict(
        query=q,
        swa_kv_cache=swa,
        workspace_buffer=torch.zeros(1 << 20, dtype=torch.int8),
        sparse_indices=torch.zeros(bs, topk, dtype=torch.int32),
        compressed_kv_cache=swa,
        sparse_topk_lens=torch.full((bs,), topk, dtype=torch.int32),
        seq_lens=torch.full((bs,), 10, dtype=torch.int32),
        bmm1_scale=512**-0.5,
        bmm2_scale=1.0,
        sinks=torch.zeros(HEADS, dtype=torch.float32),
        kv_layout="HND",
    )


def _trtllm_env(*routes, supports, forward, get_bytes, reset, capturing=False):
    return _stack(
        mock.patch.object(dsv4_routes, "cake_route_enabled", lambda n: n in routes),
        mock.patch.object(
            dsv4_routes,
            "_cake_sm100_kernels",
            lambda: (supports, forward, get_bytes, reset),
        ),
        mock.patch.object(dsv4_routes, "_is_capturing", lambda: capturing),
    )


def test_trtllm_route_off_touches_nothing():
    kernels = mock.Mock()
    route = dsv4_routes.CakeDsv4TrtllmRoute()
    with (
        mock.patch.object(dsv4_routes, "cake_route_enabled", lambda n: False),
        mock.patch.object(dsv4_routes, "_cake_sm100_kernels", kernels),
    ):
        assert route.run(**_trtllm_inputs()) is None
    kernels.assert_not_called()


def test_trtllm_route_on_admitted_forwards_engine_tensors_and_primes_once(caplog):
    caplog.set_level(logging.INFO, logger=dsv4_routes.logger.name)
    supports = mock.Mock(return_value=True)
    forward = mock.Mock(return_value=torch.zeros(BS, 1, HEADS, 512).bfloat16())
    get_bytes = mock.Mock(return_value=4096)
    reset = mock.Mock()
    route = dsv4_routes.CakeDsv4TrtllmRoute()
    inputs = _trtllm_inputs()
    with _trtllm_env(
        "dsv4_sparse_mla_decode",
        supports=supports,
        forward=forward,
        get_bytes=get_bytes,
        reset=reset,
    ):
        out = route.run(**inputs)
        route.run(**inputs)
    assert out is forward.return_value
    assert forward.call_count == 2
    args, kw = forward.call_args
    assert args[0] is inputs["query"] and args[1] is inputs["swa_kv_cache"]
    assert args[2] is inputs["workspace_buffer"]
    for name in (
        "sparse_indices",
        "compressed_kv_cache",
        "sparse_topk_lens",
        "seq_lens",
        "sinks",
    ):
        assert kw[name] is inputs[name]
    assert kw["bmm1_scale"] == inputs["bmm1_scale"] and kw["bmm2_scale"] == 1.0
    assert kw["kv_layout"] == "HND" and kw["out"] is None
    assert kw["cum_seq_lens_q"] is None and kw["max_q_len"] is None
    # Admission saw the same query / pools with the SM100 format.
    s_args, s_kw = supports.call_args
    assert s_args[0] is inputs["query"] and s_args[1] is inputs["swa_kv_cache"]
    assert s_kw["kv_cache_format"] == "fp8"
    assert s_kw["compressed_kv_cache"] is inputs["compressed_kv_cache"]
    # Workspace sized from the metadata rows / heads / table width, primed once.
    get_bytes.assert_called_once_with(BS, HEADS, SWA_TOPK, torch.float8_e4m3fn)
    reset.assert_called_once_with(inputs["workspace_buffer"])
    assert "dsv4_sparse_mla_decode/trtllm: Cake kernel selected" in caplog.text


def test_trtllm_prefill_kwargs_are_forwarded():
    supports = mock.Mock(return_value=True)
    forward = mock.Mock(return_value=torch.zeros(5, HEADS, 512).bfloat16())
    route = dsv4_routes.CakeDsv4TrtllmRoute()
    inputs = _trtllm_inputs(bs=5)
    inputs["query"] = inputs["query"].view(5, HEADS, 512)
    out_arg = torch.zeros(5, HEADS, 512).bfloat16()
    cum = torch.tensor([0, 2, 5], dtype=torch.int32)
    with _trtllm_env(
        "dsv4_sparse_mla_decode",
        supports=supports,
        forward=forward,
        get_bytes=mock.Mock(return_value=1),
        reset=mock.Mock(),
    ):
        route.run(**inputs, out=out_arg, cum_seq_lens_q=cum, max_q_len=3)
    kw = forward.call_args.kwargs
    assert kw["out"] is out_arg and kw["cum_seq_lens_q"] is cum
    assert kw["max_q_len"] == 3


def test_trtllm_route_on_rejected_falls_back(caplog):
    caplog.set_level(logging.INFO, logger=dsv4_routes.logger.name)
    forward = mock.Mock()
    reset = mock.Mock()
    route = dsv4_routes.CakeDsv4TrtllmRoute()
    inputs = _trtllm_inputs()
    with _trtllm_env(
        "dsv4_sparse_mla_decode",
        supports=mock.Mock(return_value=False),
        forward=forward,
        get_bytes=mock.Mock(return_value=1),
        reset=reset,
    ):
        assert route.run(**inputs) is None
        assert route.run(**inputs) is None
    forward.assert_not_called()
    reset.assert_not_called()
    assert caplog.text.count("dsv4_sparse_mla_decode/trtllm: fallback") == 1
    assert "adapter admission rejected" in caplog.text


def test_trtllm_workspace_too_small_falls_back(caplog):
    caplog.set_level(logging.INFO, logger=dsv4_routes.logger.name)
    forward = mock.Mock()
    route = dsv4_routes.CakeDsv4TrtllmRoute()
    inputs = _trtllm_inputs()
    with _trtllm_env(
        "dsv4_sparse_mla_decode",
        supports=mock.Mock(return_value=True),
        forward=forward,
        get_bytes=mock.Mock(return_value=(1 << 20) + 1),
        reset=mock.Mock(),
    ):
        assert route.run(**inputs) is None
    forward.assert_not_called()
    assert "workspace needs" in caplog.text


def test_trtllm_unprimed_workspace_inside_capture_falls_back(caplog):
    caplog.set_level(logging.INFO, logger=dsv4_routes.logger.name)
    forward = mock.Mock(return_value=torch.zeros(BS, 1, HEADS, 512).bfloat16())
    reset = mock.Mock()
    route = dsv4_routes.CakeDsv4TrtllmRoute()
    inputs = _trtllm_inputs()
    common = dict(
        supports=mock.Mock(return_value=True),
        forward=forward,
        get_bytes=mock.Mock(return_value=1),
        reset=reset,
    )
    with _trtllm_env("dsv4_sparse_mla_decode", capturing=True, **common):
        assert route.run(**inputs) is None  # no eager warm-up reached this site
    forward.assert_not_called()
    reset.assert_not_called()
    assert "not primed before CUDA-graph capture" in caplog.text
    with _trtllm_env("dsv4_sparse_mla_decode", capturing=False, **common):
        assert route.run(**inputs) is not None  # eager warm-up primes
    with _trtllm_env("dsv4_sparse_mla_decode", capturing=True, **common):
        assert route.run(**inputs) is not None  # capture reuses the primed one
    reset.assert_called_once()
    assert forward.call_count == 2


# ---------------------------------------------------------------------------
# dsv4_sparse_mla_decode: SM120/121 DeepSeek-V4.1 mixed-cache decode site
# ---------------------------------------------------------------------------

TOPK, EXTRA_TOPK, CHUNKS = 128, 64, 3


def _sm120_inputs(num_tokens=2, with_extra=True, head_dim_v=512):
    return dict(
        q=torch.randn(num_tokens, 1, HEADS, 512).bfloat16(),
        k_cache=torch.zeros(8, 256, 1, 528, dtype=torch.uint8),
        indices=torch.zeros(num_tokens, 1, TOPK, dtype=torch.int32),
        topk_length=torch.full((num_tokens,), TOPK, dtype=torch.int32),
        attn_sink=torch.zeros(HEADS, dtype=torch.float32),
        extra_k_cache=(
            torch.zeros(8, 64, 1, 288, dtype=torch.uint8) if with_extra else None
        ),
        extra_indices=(
            torch.zeros(num_tokens, 1, EXTRA_TOPK, dtype=torch.int32)
            if with_extra
            else None
        ),
        extra_topk_length=(
            torch.full((num_tokens,), EXTRA_TOPK, dtype=torch.int32)
            if with_extra
            else None
        ),
        sm_scale=512**-0.5,
        head_dim_v=head_dim_v,
    )


def _decode_side_effect(q, k_cache, indices, output, out_lse, sm_scale, **kw):
    output.fill_(2.0)
    out_lse.fill_(0.0)
    return {"head_tiles": 1, "num_splits": 2, "chunks_per_block": 2, "precision": 0}


def _sm120_env(*routes, supports, decode, num_chunks=None, capturing=False):
    num_chunks = num_chunks or mock.Mock(return_value=CHUNKS)
    return _stack(
        mock.patch.object(dsv4_routes, "cake_route_enabled", lambda n: n in routes),
        mock.patch.object(
            dsv4_routes,
            "_cake_sm120_dsv41_kernels",
            lambda: (supports, decode, num_chunks),
        ),
        mock.patch.object(dsv4_routes, "_is_capturing", lambda: capturing),
    )


def test_sm120_route_off_touches_nothing():
    kernels = mock.Mock()
    route = dsv4_routes.CakeDsv41MixedDecodeRoute()
    with (
        mock.patch.object(dsv4_routes, "cake_route_enabled", lambda n: False),
        mock.patch.object(dsv4_routes, "_cake_sm120_dsv41_kernels", kernels),
    ):
        assert route.run(**_sm120_inputs()) is None
    kernels.assert_not_called()


def test_sm120_route_on_admitted_calls_cake_with_stock_return_contract(caplog):
    caplog.set_level(logging.INFO, logger=dsv4_routes.logger.name)
    supports = mock.Mock(return_value=True)
    decode = mock.Mock(side_effect=_decode_side_effect)
    num_chunks = mock.Mock(return_value=CHUNKS)
    route = dsv4_routes.CakeDsv41MixedDecodeRoute()
    inputs = _sm120_inputs()
    with _sm120_env(
        "dsv4_sparse_mla_decode",
        supports=supports,
        decode=decode,
        num_chunks=num_chunks,
    ):
        o = route.run(**inputs)
    decode.assert_called_once()
    args, kw = decode.call_args
    q3, k_cache, indices, output, out_lse, sm_scale = args
    assert tuple(q3.shape) == (2, HEADS, 512) and q3.is_contiguous()
    assert q3.data_ptr() == inputs["q"].data_ptr()  # a view, no copy
    assert k_cache is inputs["k_cache"] and indices is inputs["indices"]
    assert output.dtype == torch.bfloat16 and tuple(output.shape) == (2, HEADS, 512)
    assert out_lse.dtype == torch.float32 and tuple(out_lse.shape) == (2, HEADS)
    assert sm_scale == inputs["sm_scale"]
    assert kw["topk_length"] is inputs["topk_length"]
    assert kw["attn_sink"] is inputs["attn_sink"]
    assert kw["extra_kv_cache"] is inputs["extra_k_cache"]
    assert kw["extra_indices"] is inputs["extra_indices"]
    assert kw["extra_topk_length"] is inputs["extra_topk_length"]
    assert kw["compute_precision"] == "bf16"
    # Split-merge scratch covers every plan: num_chunks(topk, extra_topk) splits.
    num_chunks.assert_called_once_with(TOPK, EXTRA_TOPK)
    assert tuple(kw["mid_out"].shape) == (2, HEADS, CHUNKS, 512)
    assert kw["mid_out"].dtype == torch.bfloat16 and kw["mid_out"].is_contiguous()
    assert tuple(kw["mid_lse"].shape) == (2, HEADS, CHUNKS)
    assert kw["mid_lse"].dtype == torch.float32
    # Admission saw the [T, H, 512] query, the pools and the index tables.
    s_args, s_kw = supports.call_args
    assert s_args[0].data_ptr() == inputs["q"].data_ptr()
    assert s_args[1] is inputs["k_cache"] and s_args[2] is inputs["indices"]
    assert s_kw["extra_kv_cache"] is inputs["extra_k_cache"]
    assert s_kw["extra_indices"] is inputs["extra_indices"]
    assert s_kw["compute_precision"] == "bf16"
    # Stock flash_mla_with_kvcache_sm120 contract: [T, 1, H, 512].
    assert tuple(o.shape) == (2, 1, HEADS, 512) and torch.all(o == 2.0)
    assert "dsv4_sparse_mla_decode/sm120_dsv41: Cake kernel selected" in caplog.text


def test_sm120_swa_only_layer_passes_no_extra_cache():
    decode = mock.Mock(side_effect=_decode_side_effect)
    num_chunks = mock.Mock(return_value=2)
    route = dsv4_routes.CakeDsv41MixedDecodeRoute()
    with _sm120_env(
        "dsv4_sparse_mla_decode",
        supports=mock.Mock(return_value=True),
        decode=decode,
        num_chunks=num_chunks,
    ):
        route.run(**_sm120_inputs(with_extra=False))
    kw = decode.call_args.kwargs
    assert kw["extra_kv_cache"] is None and kw["extra_indices"] is None
    num_chunks.assert_called_once_with(TOPK, 0)


def test_sm120_route_on_rejected_falls_back(caplog):
    caplog.set_level(logging.INFO, logger=dsv4_routes.logger.name)
    decode = mock.Mock(side_effect=_decode_side_effect)
    route = dsv4_routes.CakeDsv41MixedDecodeRoute()
    inputs = _sm120_inputs()
    with _sm120_env(
        "dsv4_sparse_mla_decode",
        supports=mock.Mock(return_value=False),
        decode=decode,
    ):
        assert route.run(**inputs) is None
        assert route.run(**inputs) is None
    decode.assert_not_called()
    assert caplog.text.count("dsv4_sparse_mla_decode/sm120_dsv41: fallback") == 1
    assert "adapter admission rejected" in caplog.text


def test_sm120_non_512_value_dim_skips_admission():
    supports = mock.Mock(return_value=True)
    decode = mock.Mock(side_effect=_decode_side_effect)
    route = dsv4_routes.CakeDsv41MixedDecodeRoute()
    with _sm120_env("dsv4_sparse_mla_decode", supports=supports, decode=decode):
        assert route.run(**_sm120_inputs(head_dim_v=256)) is None
    supports.assert_not_called()
    decode.assert_not_called()


def test_sm120_scratch_grows_eagerly_and_retains_old_storage():
    decode = mock.Mock(side_effect=_decode_side_effect)
    route = dsv4_routes.CakeDsv41MixedDecodeRoute()
    with _sm120_env(
        "dsv4_sparse_mla_decode",
        supports=mock.Mock(return_value=True),
        decode=decode,
    ):
        route.run(**_sm120_inputs(num_tokens=2))
        first = route._mid_out
        route.run(**_sm120_inputs(num_tokens=4))
        second = route._mid_out
        route.run(**_sm120_inputs(num_tokens=1))  # smaller shape reuses the arena
    assert first.numel() == 2 * HEADS * CHUNKS * 512
    assert second.numel() == 4 * HEADS * CHUNKS * 512 and second is route._mid_out
    assert any(t is first for t in route._retired)  # graphs may still point at it
    mids = [c.kwargs["mid_out"] for c in decode.call_args_list]
    assert mids[1].data_ptr() == second.data_ptr()
    assert mids[2].data_ptr() == second.data_ptr()
    assert tuple(mids[2].shape) == (1, HEADS, CHUNKS, 512)


def test_sm120_scratch_first_needed_inside_capture_falls_back(caplog):
    caplog.set_level(logging.INFO, logger=dsv4_routes.logger.name)
    decode = mock.Mock(side_effect=_decode_side_effect)
    supports = mock.Mock(return_value=True)
    route = dsv4_routes.CakeDsv41MixedDecodeRoute()
    with _sm120_env(
        "dsv4_sparse_mla_decode", supports=supports, decode=decode, capturing=True
    ):
        assert route.run(**_sm120_inputs(num_tokens=2)) is None
    decode.assert_not_called()
    assert "first needed inside CUDA-graph capture" in caplog.text
    with _sm120_env("dsv4_sparse_mla_decode", supports=supports, decode=decode):
        assert route.run(**_sm120_inputs(num_tokens=2)) is not None  # warm-up
    with _sm120_env(
        "dsv4_sparse_mla_decode", supports=supports, decode=decode, capturing=True
    ):
        assert route.run(**_sm120_inputs(num_tokens=2)) is not None  # capture
        assert route.run(**_sm120_inputs(num_tokens=1)) is not None  # smaller
        assert route.run(**_sm120_inputs(num_tokens=3)) is None  # would grow
    assert decode.call_count == 3


# ---------------------------------------------------------------------------
# module hygiene
# ---------------------------------------------------------------------------


def test_route_names_exist_in_route_table():
    from sglang.kernels.cake_kernels._routes import ROUTES

    assert topk_mod.CAKE_ROUTE_DSV3_GROUPED_ROUTING in ROUTES
    assert dsv4_routes.CAKE_ROUTE_DSV4_SPARSE_MLA_DECODE in ROUTES


def test_dsv4_route_module_imports_no_flashinfer_at_import_time():
    spec = importlib.util.find_spec("sglang.srt.layers.attention.dsv4.cake_routes")
    with open(spec.origin) as f:
        top_level_imports = [
            line for line in f if line.startswith(("import ", "from "))
        ]
    assert top_level_imports
    assert all("flashinfer" not in line for line in top_level_imports)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
