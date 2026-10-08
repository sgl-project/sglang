"""CPU unit tests for the opt-in Cake Kimi-K3 FP8 MLA decode route (``SGLANG_CAKE_ROUTES=kimi_k3_mla``).

The FlashInfer entry and the adapter's admission predicate are mocked, so the
tests pin the engine-side wiring in ``TRTLLMMLABackend._run_decode_kernel``
only: route off -> stock trtllm-gen call (adapter never consulted); route on and
admitted -> the public entry is called with ``backend="cake"`` and the reduced
Cake contract (no PDL, no skip-softmax, no counter buffer, host float scale);
route on and rejected / LSE requested / explicit cute-dsl backend -> the stock
call is kept. Verdicts are cached per query shape.
"""

import os
from types import SimpleNamespace
from unittest import mock

import pytest
import torch

from sglang.kernels.cake_kernels import _routes
from sglang.srt.layers.attention import trtllm_mla_backend as tmb
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=10, stage="base-a-test-cpu")

BS, H, LATENT, ROPE, PAGE = 2, 12, 512, 64, 64

_env_patchers = []


@pytest.fixture(autouse=True)
def _reset_routes():
    _routes.reset_cache_for_tests()
    tmb._cake_mla_logged.clear()
    yield
    while _env_patchers:
        _env_patchers.pop().stop()
    _routes.reset_cache_for_tests()
    tmb._cake_mla_logged.clear()


def _route(on: bool) -> None:
    patcher = mock.patch.dict(os.environ, {}, clear=False)
    patcher.start()
    _env_patchers.append(patcher)
    if on:
        os.environ[_routes.ENV_VAR] = "kimi_k3_mla"
    else:
        os.environ.pop(_routes.ENV_VAR, None)
    _routes.reset_cache_for_tests()


def _backend(kernel_backend: str = "trtllm-gen"):
    """A ``TRTLLMMLABackend`` shell with only the fields ``_run_decode_kernel`` touches."""
    be = tmb.TRTLLMMLABackend.__new__(tmb.TRTLLMMLABackend)
    be.backend = kernel_backend
    be._cake_mla_verdicts = {}
    be.page_size = PAGE
    be.qk_nope_head_dim = 128
    be.kv_lora_rank = LATENT
    be.qk_rope_head_dim = ROPE
    be.workspace_buffer = torch.zeros(16, dtype=torch.uint8)
    be._multi_ctas_kv_counter_buffer = None
    be._compute_decode_bmm1_scale = lambda layer: 0.125
    return be


def _inputs():
    query = torch.zeros((BS, 1, H, LATENT + ROPE), dtype=torch.float8_e4m3fn)
    kv_cache = torch.zeros((4, 1, PAGE, LATENT + ROPE), dtype=torch.float8_e4m3fn)
    block_tables = torch.zeros((BS, 2), dtype=torch.int32)
    seq_lens = torch.tensor([64, 100], dtype=torch.int64)
    return query, kv_cache, block_tables, seq_lens


def _run(be, admitted, return_lse=False):
    query, kv_cache, block_tables, seq_lens = _inputs()
    fi_call = mock.Mock(return_value="out")
    supports = mock.Mock(return_value=admitted)
    fake_fi = SimpleNamespace(
        decode=SimpleNamespace(trtllm_batch_decode_with_kv_cache_mla=fi_call)
    )
    with (
        mock.patch.object(tmb, "flashinfer", fake_fi, create=True),
        mock.patch.object(
            tmb, "get_parallel", lambda: SimpleNamespace(dcp_enabled=False)
        ),
        mock.patch(
            "sglang.kernels.cake_kernels.attention_mla.supports_trtllm_batch_decode_with_kv_cache_mla",
            supports,
        ),
    ):
        out = be._run_decode_kernel(
            query,
            kv_cache,
            block_tables,
            seq_lens,
            128,
            layer=None,
            return_lse=return_lse,
        )
    return out, fi_call, supports


def test_route_off_keeps_stock_call():
    _route(False)
    out, fi_call, supports = _run(_backend(), admitted=True)
    assert out == "out"
    supports.assert_not_called()
    kwargs = fi_call.call_args.kwargs
    assert "backend" not in kwargs
    assert kwargs["enable_pdl"] == tmb._ENABLE_PDL
    assert "multi_ctas_kv_counter_buffer" in kwargs
    assert kwargs["return_lse"] is False


def test_route_on_admitted_uses_cake_backend_with_reduced_contract():
    _route(True)
    be = _backend()
    out, fi_call, supports = _run(be, admitted=True)
    assert out == "out"
    supports.assert_called_once()
    s_kwargs = supports.call_args.kwargs
    assert (
        s_kwargs["qk_nope_head_dim"],
        s_kwargs["kv_lora_rank"],
        s_kwargs["qk_rope_head_dim"],
    ) == (128, LATENT, ROPE)
    kwargs = fi_call.call_args.kwargs
    assert kwargs["backend"] == "cake"
    assert kwargs["enable_pdl"] is False
    assert "skip_softmax_threshold_scale_factor" not in kwargs
    assert "multi_ctas_kv_counter_buffer" not in kwargs
    assert "return_lse" not in kwargs
    assert isinstance(kwargs["bmm1_scale"], float)
    assert kwargs["seq_lens"].dtype == torch.int32
    assert kwargs["max_seq_len"] == 128
    # second call with the same shape: cached verdict, adapter not consulted again
    _, fi_call2, supports2 = _run(be, admitted=True)
    supports2.assert_not_called()
    assert fi_call2.call_args.kwargs["backend"] == "cake"


def test_route_on_rejected_keeps_stock_call():
    _route(True)
    out, fi_call, supports = _run(_backend(), admitted=False)
    assert out == "out"
    supports.assert_called_once()
    assert "backend" not in fi_call.call_args.kwargs


def test_route_on_lse_or_explicit_kernel_backend_keeps_stock_call():
    _route(True)
    _, fi_call, supports = _run(_backend(), admitted=True, return_lse=True)
    supports.assert_not_called()
    assert "backend" not in fi_call.call_args.kwargs
    assert fi_call.call_args.kwargs["return_lse"] is True
    _, fi_call, supports = _run(_backend("cute-dsl"), admitted=True)
    supports.assert_not_called()
    assert fi_call.call_args.kwargs["backend"] == "cute-dsl"
