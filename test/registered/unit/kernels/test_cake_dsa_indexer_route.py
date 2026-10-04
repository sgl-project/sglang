"""Unit tests for the ``dsa_indexer`` Cake route helpers
(``sglang.srt.layers.attention.dsa.cake_indexer_routes``).

Everything is mocked: the route switch, the adapter admission and the Cake
forwarders. The tests only check *which* callable receives the engine's
tensors, that the Cake branch keeps the DeepGEMM call contract (positional
tensors, ``clean_logits=False``, the engine's SM count as the CTA budget), and
the fallback rules (route off, admission False, FlashInfer host rejection
cached per reason, a shape first seen inside CUDA-graph capture). CPU tensors;
no FlashInfer, DeepGEMM or CUDA involved.
"""

import contextlib
import logging
from unittest import mock

import pytest
import torch

from sglang.srt.layers.attention.dsa import cake_indexer_routes as routes
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, stage="base-a-test-cpu")

Q, K, H, D = 6, 512, 64, 128
B, NEXT_N, PAGES, MAX_LEN, SMS = 3, 2, 8, 256, 148


def _stack(*managers):
    stack = contextlib.ExitStack()
    for manager in managers:
        stack.enter_context(manager)
    return stack


@pytest.fixture(autouse=True)
def _reset_route_state():
    routes.reset_cake_route_state_for_tests()
    yield
    routes.reset_cake_route_state_for_tests()


def _ragged_inputs():
    q = torch.randn(Q, H, D).to(torch.float8_e4m3fn)
    kv = torch.randn(K, D).to(torch.float8_e4m3fn)
    kv_scales = torch.rand(K) + 0.5
    weights = torch.randn(Q, H)
    ks = torch.zeros(Q, dtype=torch.int32)
    ke = torch.full((Q,), K, dtype=torch.int32)
    return q, kv, kv_scales, weights, ks, ke


def _paged_inputs():
    q = torch.randn(B, NEXT_N, H, D).to(torch.float8_e4m3fn)
    kv_cache = torch.zeros(PAGES, 64, 1, 132, dtype=torch.uint8)
    weights = torch.randn(B * NEXT_N, H)
    ctx = torch.full((B, NEXT_N), 100, dtype=torch.int32)
    block_table = torch.zeros(B * NEXT_N, MAX_LEN // 64, dtype=torch.int32)[::NEXT_N]
    return q, kv_cache, weights, ctx, block_table


def _ragged_env(*on, supports, forward, capturing=False):
    return _stack(
        mock.patch.object(routes, "cake_route_enabled", lambda n: n in on),
        mock.patch.object(routes, "_cake_ragged_kernels", lambda: (supports, forward)),
        mock.patch.object(routes, "_is_capturing", lambda: capturing),
    )


def _paged_env(*on, supports, metadata, forward, capturing=False):
    return _stack(
        mock.patch.object(routes, "cake_route_enabled", lambda n: n in on),
        mock.patch.object(
            routes, "_cake_paged_kernels", lambda: (supports, metadata, forward)
        ),
        mock.patch.object(routes, "_is_capturing", lambda: capturing),
    )


# ---------------------------------------------------------------------------
# ragged prefill site (fp8_mqa_logits)
# ---------------------------------------------------------------------------


def test_ragged_route_off_touches_nothing():
    kernels = mock.Mock()
    q, kv, kv_scales, weights, ks, ke = _ragged_inputs()
    with (
        mock.patch.object(routes, "cake_route_enabled", lambda n: False),
        mock.patch.object(routes, "_cake_ragged_kernels", kernels),
    ):
        assert (
            routes.cake_fp8_mqa_logits(q, (kv, kv_scales), weights, ks, ke, num_sms=SMS)
            is None
        )
    kernels.assert_not_called()


def test_ragged_route_on_admitted_forwards_engine_tensors(caplog):
    caplog.set_level(logging.INFO, logger=routes.logger.name)
    q, kv, kv_scales, weights, ks, ke = _ragged_inputs()
    out = torch.full((Q, K), 2.0)
    supports = mock.Mock(return_value=True)
    forward = mock.Mock(return_value=out)
    with _ragged_env("dsa_indexer", supports=supports, forward=forward):
        result = routes.cake_fp8_mqa_logits(
            q, (kv, kv_scales), weights, ks, ke, num_sms=SMS - 1
        )
    assert result is out
    s_args, s_kw = supports.call_args
    assert not s_kw and len(s_args) == 6
    assert s_args[0] is q and s_args[1] is kv and s_args[2] is kv_scales
    assert s_args[3] is weights and s_args[4] is ks and s_args[5] is ke
    args, kw = forward.call_args
    assert args[0] is q and args[1][0] is kv and args[1][1] is kv_scales
    assert args[2] is weights and args[3] is ks and args[4] is ke
    # DeepGEMM's call contract: clean_logits=False, no max_seqlen_k narrowing,
    # the engine's (pipeline-parallel adjusted) SM count as the CTA budget.
    assert kw == {"clean_logits": False, "max_seqlen_k": 0, "sm_count": SMS - 1}
    assert (
        "[cake-route] dsa_indexer/fp8_mqa_logits: Cake kernel selected" in caplog.text
    )


def test_ragged_route_on_rejected_falls_back_and_logs_once(caplog):
    caplog.set_level(logging.INFO, logger=routes.logger.name)
    q, kv, kv_scales, weights, ks, ke = _ragged_inputs()
    supports = mock.Mock(return_value=False)
    forward = mock.Mock()
    with _ragged_env("dsa_indexer", supports=supports, forward=forward):
        for _ in range(3):
            assert (
                routes.cake_fp8_mqa_logits(
                    q, (kv, kv_scales), weights, ks, ke, num_sms=SMS
                )
                is None
            )
    assert supports.call_count == 3  # admission is re-evaluated per call
    forward.assert_not_called()
    assert caplog.text.count("fallback to the default kernel") == 1
    assert "adapter admission rejected" in caplog.text


def test_ragged_host_rejection_is_cached_per_reason(caplog):
    caplog.set_level(logging.INFO, logger=routes.logger.name)
    q, kv, kv_scales, weights, ks, ke = _ragged_inputs()
    supports = mock.Mock(return_value=True)
    forward = mock.Mock(
        side_effect=ValueError("positive K divisible by 256 is required")
    )
    with _ragged_env("dsa_indexer", supports=supports, forward=forward):
        for _ in range(2):
            assert (
                routes.cake_fp8_mqa_logits(
                    q, (kv, kv_scales), weights, ks, ke, num_sms=SMS
                )
                is None
            )
    assert forward.call_count == 2  # the host call is cheap; the log is once
    assert caplog.text.count("FlashInfer host rejection") == 1


def test_ragged_shape_first_seen_in_capture_falls_back_then_primes():
    q, kv, kv_scales, weights, ks, ke = _ragged_inputs()
    out = torch.zeros(Q, K)
    supports = mock.Mock(return_value=True)
    forward = mock.Mock(return_value=out)
    with _ragged_env("dsa_indexer", supports=supports, forward=forward, capturing=True):
        assert (
            routes.cake_fp8_mqa_logits(q, (kv, kv_scales), weights, ks, ke, num_sms=SMS)
            is None
        )
    forward.assert_not_called()
    with _ragged_env("dsa_indexer", supports=supports, forward=forward):
        assert (
            routes.cake_fp8_mqa_logits(q, (kv, kv_scales), weights, ks, ke, num_sms=SMS)
            is out
        )
    with _ragged_env("dsa_indexer", supports=supports, forward=forward, capturing=True):
        assert (
            routes.cake_fp8_mqa_logits(q, (kv, kv_scales), weights, ks, ke, num_sms=SMS)
            is out
        )
    assert forward.call_count == 2


# ---------------------------------------------------------------------------
# paged decode / verify site (metadata + fp8_paged_mqa_logits)
# ---------------------------------------------------------------------------


def test_paged_route_off_touches_nothing():
    kernels = mock.Mock()
    q, kv_cache, weights, ctx, block_table = _paged_inputs()
    with (
        mock.patch.object(routes, "cake_route_enabled", lambda n: False),
        mock.patch.object(routes, "_cake_paged_kernels", kernels),
    ):
        assert (
            routes.cake_fp8_paged_mqa_logits(
                q,
                kv_cache,
                weights,
                ctx,
                block_table,
                MAX_LEN,
                block_kv=64,
                num_sms=SMS,
            )
            is None
        )
    kernels.assert_not_called()


def test_paged_route_on_admitted_builds_cake_metadata_and_forwards(caplog):
    caplog.set_level(logging.INFO, logger=routes.logger.name)
    q, kv_cache, weights, ctx, block_table = _paged_inputs()
    meta = torch.zeros(SMS + 1, 2, dtype=torch.int32)
    out = torch.full((B * NEXT_N, MAX_LEN), 3.0)
    supports = mock.Mock(return_value=True)
    metadata = mock.Mock(return_value=meta)
    forward = mock.Mock(return_value=out)
    with _paged_env(
        "dsa_indexer", supports=supports, metadata=metadata, forward=forward
    ):
        result = routes.cake_fp8_paged_mqa_logits(
            q, kv_cache, weights, ctx, block_table, MAX_LEN, block_kv=64, num_sms=SMS
        )
    assert result is out
    s_args, _ = supports.call_args
    assert s_args[0] is q and s_args[1] is kv_cache and s_args[2] is weights
    assert s_args[3] is ctx and s_args[4] is block_table
    m_args, m_kw = metadata.call_args
    assert m_args[0] is ctx and m_args[1:] == (64, SMS) and not m_kw
    args, kw = forward.call_args
    assert args[0] is q and args[1] is kv_cache and args[2] is weights
    assert args[3] is ctx and args[4] is block_table and args[5] is meta
    assert args[6] == MAX_LEN and kw == {"clean_logits": False}
    # The engine's strided [::next_n] block-table view is passed through as is.
    assert block_table.stride(0) == NEXT_N * (MAX_LEN // 64)
    assert (
        "[cake-route] dsa_indexer/fp8_paged_mqa_logits: Cake kernel selected"
        in caplog.text
    )


def test_paged_route_requires_page_64(caplog):
    caplog.set_level(logging.INFO, logger=routes.logger.name)
    q, kv_cache, weights, ctx, block_table = _paged_inputs()
    supports = mock.Mock(return_value=True)
    metadata = mock.Mock()
    forward = mock.Mock()
    with _paged_env(
        "dsa_indexer", supports=supports, metadata=metadata, forward=forward
    ):
        assert (
            routes.cake_fp8_paged_mqa_logits(
                q,
                kv_cache,
                weights,
                ctx,
                block_table,
                MAX_LEN,
                block_kv=32,
                num_sms=SMS,
            )
            is None
        )
    supports.assert_not_called()
    metadata.assert_not_called()
    forward.assert_not_called()
    assert "adapter admission rejected: page 32" in caplog.text


def test_paged_host_rejection_in_metadata_falls_back(caplog):
    caplog.set_level(logging.INFO, logger=routes.logger.name)
    q, kv_cache, weights, ctx, block_table = _paged_inputs()
    supports = mock.Mock(return_value=True)
    metadata = mock.Mock(
        side_effect=ValueError("context_lens must be int32 [B, next_n]")
    )
    forward = mock.Mock()
    with _paged_env(
        "dsa_indexer", supports=supports, metadata=metadata, forward=forward
    ):
        for _ in range(2):
            assert (
                routes.cake_fp8_paged_mqa_logits(
                    q,
                    kv_cache,
                    weights,
                    ctx,
                    block_table,
                    MAX_LEN,
                    block_kv=64,
                    num_sms=SMS,
                )
                is None
            )
    forward.assert_not_called()
    assert caplog.text.count("FlashInfer host rejection") == 1
