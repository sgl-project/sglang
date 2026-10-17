"""Unit tests for the ``gdn_prefill`` / ``gdn_decode`` Cake routes in the
FlashInfer GDN kernel (``SGLANG_CAKE_ROUTES``).

Everything is mocked: the route switch, the adapter admission, the Cake
forwarders and the stock FlashInfer functions.  The tests only check *which*
callable receives the engine's tensors and that the return contract of the
Cake branch matches the stock branch (output view, in-place pool update,
checkpoint layout).  CPU tensors; no FlashInfer, Triton or CUDA involved.
"""

import logging
import sys
import types
from unittest import mock

import pytest
import torch

from sglang.srt.layers.attention.linear.kernels import gdn_flashinfer as mod
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=10, stage="base-a-test-cpu")

HEAD_DIM = 128
B, T, H, HV, POOL = 2, 8, 2, 4, 3
EVERY = 64


def _kernel(fi_prefill, fi_decode):
    """A ``FlashInferGDNKernel`` on the SM100 state-pool path without running
    ``__init__`` (which would import FlashInfer and Triton)."""
    kernel = mod.FlashInferGDNKernel.__new__(mod.FlashInferGDNKernel)
    kernel._prefill_fn = fi_prefill
    kernel._decode_fn = fi_decode
    kernel._mtp_fn = None
    kernel.use_state_pool = True
    kernel._prefill_needs_fp32_state = False
    kernel.supports_target_verify = True
    kernel._aligned_input_buffers = {}
    kernel._aligned_parameter_cache = {}
    kernel._verify_intermediate_buffers = {}
    kernel._alignment_fallback_warned = False
    kernel._alignment_fallback_kernel = mock.Mock(name="triton_fallback")
    # CPU allocations are not guaranteed 32-byte aligned; alignment is not
    # what these tests exercise.
    kernel._mutable_inputs_are_aligned = lambda *named: True
    return kernel


def _prefill_inputs():
    q = torch.randn(1, T, H, HEAD_DIM).bfloat16()
    k = torch.randn(1, T, H, HEAD_DIM).bfloat16()
    v = torch.randn(1, T, HV, HEAD_DIM).bfloat16()
    g = -torch.rand(1, T, HV)
    beta = torch.rand(1, T, HV)
    ssm_states = torch.zeros(POOL, HV, HEAD_DIM, HEAD_DIM, dtype=torch.bfloat16)
    cache_indices = torch.tensor([1, 2], dtype=torch.int32)
    query_start_loc = torch.tensor([0, T // 2, T], dtype=torch.int32)
    checkpoint_cu_starts = torch.tensor([0, 1, 2], dtype=torch.int64)
    return dict(
        q=q,
        k=k,
        v=v,
        g=g,
        beta=beta,
        ssm_states=ssm_states,
        cache_indices=cache_indices,
        query_start_loc=query_start_loc,
        state_checkpoint_cu_starts=checkpoint_cu_starts,
        num_state_checkpoints=2,
        state_checkpoint_every_n_tokens=EVERY,
    )


def _decode_inputs():
    q = torch.randn(1, B, H, HEAD_DIM).bfloat16()
    k = torch.randn(1, B, H, HEAD_DIM).bfloat16()
    v = torch.randn(1, B, HV, HEAD_DIM).bfloat16()
    a = torch.randn(1, B, HV).bfloat16()
    b = torch.randn(1, B, HV).bfloat16()
    A_log = torch.randn(HV, dtype=torch.float32)
    dt_bias = torch.ones(HV, dtype=torch.bfloat16)  # Qwen3.5 keeps it in model dtype
    ssm_states = torch.zeros(POOL, HV, HEAD_DIM, HEAD_DIM, dtype=torch.bfloat16)
    cache_indices = torch.tensor([1, 2], dtype=torch.int32)
    query_start_loc = torch.arange(B + 1, dtype=torch.int32)
    return dict(
        q=q,
        k=k,
        v=v,
        a=a,
        b=b,
        A_log=A_log,
        dt_bias=dt_bias,
        ssm_states=ssm_states,
        cache_indices=cache_indices,
        query_start_loc=query_start_loc,
    )


def _prefill_output(kw, value):
    """Mimic FlashInfer: write ``output`` (allocating when None) and the state."""
    out = kw["output"]
    if out is None:
        out = torch.empty(kw["v"].shape[0], HV, HEAD_DIM, dtype=kw["v"].dtype)
    out.fill_(value)
    kw["output_state"].fill_(value)
    if kw["state_checkpoints"] is not None:
        kw["state_checkpoints"].fill_(value)
    return out, kw["output_state"]


def _fi_prefill_side_effect(**kw):
    return _prefill_output(kw, 1.0)


def _cake_prefill_side_effect(**kw):
    return _prefill_output(kw, 2.0)


def _fi_decode_side_effect(**kw):
    return torch.full((B, 1, HV, HEAD_DIM), 1.0, dtype=torch.bfloat16), None


def _cake_decode_side_effect(*args, **kw):
    kw["initial_state"][kw["initial_state_indices"].long()] = 2.0
    return torch.full((B, 1, HV, HEAD_DIM), 2.0, dtype=torch.bfloat16), kw[
        "initial_state"
    ]


@pytest.fixture(autouse=True)
def _reset_route_state():
    mod.reset_cake_route_state_for_tests()
    # ``extend`` imports the Triton l2norm helper lazily; replace it with a
    # module that leaves q/k/v untouched so no Triton is needed on CPU.
    fake = types.ModuleType("sglang.kernels.ops.attention.fla.l2norm")
    fake.gdn_prefill_qkv_prepare_fwd = lambda q, k, v: (
        q.contiguous(),
        k.contiguous(),
        v.contiguous(),
    )
    with mock.patch.dict(sys.modules, {fake.__name__: fake}):
        yield
    mod.reset_cake_route_state_for_tests()


def _routes(*enabled):
    return mock.patch.object(mod, "cake_route_enabled", lambda name: name in enabled)


# ---------------------------------------------------------------------------
# prefill
# ---------------------------------------------------------------------------


def test_prefill_route_off_uses_flashinfer():
    fi_prefill = mock.Mock(side_effect=_fi_prefill_side_effect)
    supports = mock.Mock(return_value=True)
    cake = mock.Mock(side_effect=_cake_prefill_side_effect)
    kernel = _kernel(fi_prefill, mock.Mock())
    inputs = _prefill_inputs()
    with (
        _routes(),
        mock.patch.object(mod, "_cake_gdn_prefill_kernels", lambda: (supports, cake)),
    ):
        out, _, h = kernel.extend(**inputs)
    fi_prefill.assert_called_once()
    supports.assert_not_called()
    cake.assert_not_called()
    assert tuple(out.shape) == (1, T, HV, HEAD_DIM)
    assert tuple(h.shape) == (1, 2, HV, HEAD_DIM, HEAD_DIM)
    assert torch.all(inputs["ssm_states"][1:] == 1.0)


def test_prefill_route_on_admitted_uses_cake_with_engine_tensors(caplog):
    caplog.set_level(logging.INFO, logger=mod.logger.name)
    fi_prefill = mock.Mock(side_effect=_fi_prefill_side_effect)
    supports = mock.Mock(return_value=True)
    cake = mock.Mock(side_effect=_cake_prefill_side_effect)
    kernel = _kernel(fi_prefill, mock.Mock())
    inputs = _prefill_inputs()
    with (
        _routes("gdn_prefill"),
        mock.patch.object(mod, "_cake_gdn_prefill_kernels", lambda: (supports, cake)),
    ):
        out, _, h = kernel.extend(**inputs)
    fi_prefill.assert_not_called()
    cake.assert_called_once()
    kw = cake.call_args.kwargs
    # Exactly the stock call's kwargs, no extra arguments.
    assert set(kw) == {
        "q",
        "k",
        "v",
        "g",
        "beta",
        "scale",
        "initial_state",
        "output_final_state",
        "cu_seqlens",
        "use_qk_l2norm_in_kernel",
        "output",
        "output_state",
        "state_checkpoints",
        "checkpoint_cu_starts",
        "checkpoint_every_n_tokens",
    }
    assert kw["cu_seqlens"].dtype == torch.int64
    assert kw["g"].dtype == torch.float32 and kw["beta"].dtype == torch.float32
    assert tuple(kw["initial_state"].shape) == (B, HV, HEAD_DIM, HEAD_DIM)
    assert kw["checkpoint_every_n_tokens"] == EVERY
    assert kw["checkpoint_cu_starts"] is inputs["state_checkpoint_cu_starts"]
    assert kw["use_qk_l2norm_in_kernel"] is False and kw["output_final_state"]
    # The admission saw the same tensors the forwarder received.
    s_args, s_kw = supports.call_args
    assert s_args[0] is kw["q"] and s_args[1] is kw["k"] and s_args[2] is kw["v"]
    assert s_args[3] is kw["g"] and s_args[4] is kw["beta"]
    assert s_args[5] is kw["cu_seqlens"]
    assert s_kw["initial_state"] is kw["initial_state"]
    assert s_kw["state_checkpoints"] is kw["state_checkpoints"]
    assert s_kw["checkpoint_cu_starts"] is kw["checkpoint_cu_starts"]
    assert s_kw["checkpoint_every_n_tokens"] == EVERY
    assert s_kw["use_cp"] == "auto"
    # Same return contract as the stock path.
    assert tuple(out.shape) == (1, T, HV, HEAD_DIM) and torch.all(out == 2.0)
    assert tuple(h.shape) == (1, 2, HV, HEAD_DIM, HEAD_DIM) and torch.all(h == 2.0)
    assert torch.all(inputs["ssm_states"][1:] == 2.0)
    assert torch.all(inputs["ssm_states"][0] == 0.0)
    assert "[cake-route] gdn_prefill: Cake kernel selected" in caplog.text


def test_prefill_route_on_rejected_falls_back(caplog):
    caplog.set_level(logging.INFO, logger=mod.logger.name)
    fi_prefill = mock.Mock(side_effect=_fi_prefill_side_effect)
    supports = mock.Mock(return_value=False)
    cake = mock.Mock(side_effect=_cake_prefill_side_effect)
    kernel = _kernel(fi_prefill, mock.Mock())
    inputs = _prefill_inputs()
    with (
        _routes("gdn_prefill"),
        mock.patch.object(mod, "_cake_gdn_prefill_kernels", lambda: (supports, cake)),
    ):
        kernel.extend(**inputs)
        kernel.extend(**inputs)
    assert fi_prefill.call_count == 2
    cake.assert_not_called()
    assert supports.call_count == 2  # admission is per call (shape dependent)
    assert caplog.text.count("[cake-route] gdn_prefill: fallback") == 1  # logged once
    assert "adapter admission rejected" in caplog.text


def test_prefill_manifest_miss_falls_back_and_is_cached(caplog):
    caplog.set_level(logging.INFO, logger=mod.logger.name)
    fi_prefill = mock.Mock(side_effect=_fi_prefill_side_effect)
    supports = mock.Mock(return_value=True)
    cake = mock.Mock(side_effect=NotImplementedError("no manifest row"))
    kernel = _kernel(fi_prefill, mock.Mock())
    inputs = _prefill_inputs()
    with (
        _routes("gdn_prefill"),
        mock.patch.object(mod, "_cake_gdn_prefill_kernels", lambda: (supports, cake)),
    ):
        out, _, _ = kernel.extend(**inputs)
        kernel.extend(**inputs)
    assert fi_prefill.call_count == 2
    assert cake.call_count == 1  # the rejected shape key is not retried
    assert torch.all(out == 1.0)
    assert "no manifest row" in caplog.text


def test_prefill_route_skipped_off_state_pool():
    fi_prefill = mock.Mock(side_effect=_fi_prefill_side_effect)
    supports = mock.Mock(return_value=True)
    cake = mock.Mock(side_effect=_cake_prefill_side_effect)
    kernel = _kernel(fi_prefill, mock.Mock())
    kernel.use_state_pool = False  # SM90 gather/scatter branch stays stock
    inputs = _prefill_inputs()
    with (
        _routes("gdn_prefill"),
        mock.patch.object(mod, "_cake_gdn_prefill_kernels", lambda: (supports, cake)),
    ):
        kernel.extend(**inputs)
    fi_prefill.assert_called_once()
    cake.assert_not_called()


# ---------------------------------------------------------------------------
# decode
# ---------------------------------------------------------------------------


def test_decode_route_off_uses_flashinfer():
    fi_decode = mock.Mock(side_effect=_fi_decode_side_effect)
    supports = mock.Mock(return_value=True)
    cake = mock.Mock(side_effect=_cake_decode_side_effect)
    kernel = _kernel(mock.Mock(), fi_decode)
    inputs = _decode_inputs()
    with (
        _routes(),
        mock.patch.object(mod, "_cake_gdn_decode_kernels", lambda: (supports, cake)),
    ):
        out = kernel.decode(**inputs)
    fi_decode.assert_called_once()
    assert fi_decode.call_args.kwargs["initial_state"] is inputs["ssm_states"]
    supports.assert_not_called()
    cake.assert_not_called()
    assert tuple(out.shape) == (1, B, HV, HEAD_DIM) and torch.all(out == 1.0)


def test_decode_route_on_admitted_uses_cake_with_engine_tensors(caplog):
    caplog.set_level(logging.INFO, logger=mod.logger.name)
    fi_decode = mock.Mock(side_effect=_fi_decode_side_effect)
    supports = mock.Mock(return_value=True)
    cake = mock.Mock(side_effect=_cake_decode_side_effect)
    kernel = _kernel(mock.Mock(), fi_decode)
    inputs = _decode_inputs()
    with (
        _routes("gdn_decode"),
        mock.patch.object(mod, "_cake_gdn_decode_kernels", lambda: (supports, cake)),
    ):
        out = kernel.decode(**inputs)
    fi_decode.assert_not_called()
    cake.assert_called_once()
    args, kw = cake.call_args
    q, k, v, state, A_log, a, dt_bias, b = args
    assert tuple(q.shape) == (B, 1, H, HEAD_DIM) and tuple(k.shape) == q.shape
    assert tuple(v.shape) == (B, 1, HV, HEAD_DIM)
    assert tuple(a.shape) == (B, 1, HV) and tuple(b.shape) == (B, 1, HV)
    assert state is None
    assert A_log.dtype == torch.float32 and dt_bias.dtype == torch.float32
    assert kw["initial_state"] is inputs["ssm_states"]
    assert kw["initial_state_indices"].dtype == torch.int32
    assert kw["use_qk_l2norm"] is True and kw["backend"] == "cake_gdn"
    # Admission saw the same tensors.
    s_args, s_kw = supports.call_args
    assert s_args[0] is q and s_args[1] is k and s_args[2] is v
    assert s_args[3] is inputs["ssm_states"]
    assert s_args[4] is kw["initial_state_indices"]
    assert s_kw["A_log"] is A_log and s_kw["dt_bias"] is dt_bias
    assert s_kw["a"] is a and s_kw["b"] is b
    # Same contract: pool rows updated in place, [1, B, HV, 128] output.
    assert tuple(out.shape) == (1, B, HV, HEAD_DIM) and torch.all(out == 2.0)
    assert torch.all(inputs["ssm_states"][1:] == 2.0)
    assert torch.all(inputs["ssm_states"][0] == 0.0)
    assert "[cake-route] gdn_decode: Cake kernel selected" in caplog.text


def test_decode_route_on_rejected_falls_back(caplog):
    caplog.set_level(logging.INFO, logger=mod.logger.name)
    fi_decode = mock.Mock(side_effect=_fi_decode_side_effect)
    supports = mock.Mock(return_value=False)
    cake = mock.Mock(side_effect=_cake_decode_side_effect)
    kernel = _kernel(mock.Mock(), fi_decode)
    inputs = _decode_inputs()
    with (
        _routes("gdn_decode"),
        mock.patch.object(mod, "_cake_gdn_decode_kernels", lambda: (supports, cake)),
    ):
        out = kernel.decode(**inputs)
        kernel.decode(**inputs)
    assert fi_decode.call_count == 2
    cake.assert_not_called()
    assert torch.all(out == 1.0)
    assert caplog.text.count("[cake-route] gdn_decode: fallback") == 1
    assert "adapter admission rejected" in caplog.text


def test_decode_manifest_miss_falls_back_and_is_cached():
    fi_decode = mock.Mock(side_effect=_fi_decode_side_effect)
    supports = mock.Mock(return_value=True)
    cake = mock.Mock(side_effect=NotImplementedError("no manifest row"))
    kernel = _kernel(mock.Mock(), fi_decode)
    inputs = _decode_inputs()
    with (
        _routes("gdn_decode"),
        mock.patch.object(mod, "_cake_gdn_decode_kernels", lambda: (supports, cake)),
    ):
        kernel.decode(**inputs)
        kernel.decode(**inputs)
    assert fi_decode.call_count == 2
    assert cake.call_count == 1
    assert supports.call_count == 1


def test_route_names_exist_in_route_table():
    from sglang.kernels.cake_kernels._routes import ROUTES

    assert mod.CAKE_ROUTE_PREFILL in ROUTES and mod.CAKE_ROUTE_DECODE in ROUTES


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
