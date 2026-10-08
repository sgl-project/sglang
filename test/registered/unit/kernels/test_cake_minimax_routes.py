"""CPU unit tests for the MiniMax Cake routes (``SGLANG_CAKE_ROUTES``).

``minimax_h3_diffusion`` -- the four fused stages of
``sglang.multimodal_gen.runtime.models.dits.minimax_h3_cake_routes``.  The
route switch, the adapter admissions and the Cake forwarders are mocked; the
tests pin *which* callable receives the engine's tensors, the FlashInfer
contract adaptations (contiguous AdaLN tables, int32 indices, per-row RoPE
table, contiguous THD attention operands), the "None -> stock path" fallback
contract, the CUDA-graph capture guard and the cached FlashInfer rejection.
CPU tensors; no FlashInfer, Triton or CUDA involved.

``msa_nvfp4_sparse_decode`` has no engine call site in this tree (the MiniMax-M3
sparse backend has no NVFP4 main KV pool); only its route registration is
checked here.
"""

import logging
import subprocess
import sys
from unittest import mock

import pytest
import torch

from sglang.multimodal_gen.runtime.models.dits import minimax_h3_cake_routes as mod
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=10, stage="base-a-test-cpu")

T, HIDDEN, HEADS, HEAD_DIM, FFN, ROWS = 6, 16, 2, 4, 8, 9
BF16 = torch.bfloat16


@pytest.fixture(autouse=True)
def _reset_state():
    mod.reset_state_for_tests()
    yield
    mod.reset_state_for_tests()


def _routes(*enabled):
    return mock.patch.object(mod, "cake_route_enabled", lambda name: name in enabled)


def _capturing(flag: bool):
    return mock.patch.object(mod, "_capturing", lambda: flag)


def _adaln_tables():
    """Six ``[ROWS, HIDDEN]`` column chunks of a ``[ROWS, 6 * HIDDEN]`` projection,
    exactly as ``MiniMaxH3AdalnProj.split_output`` produces them (non-contiguous)."""
    proj = torch.randn(ROWS, 6 * HIDDEN).to(BF16)
    chunks = proj.chunk(6, dim=-1)
    assert not chunks[0].is_contiguous()
    return chunks


def _pre_attention_inputs():
    shift, scale, *_ = _adaln_tables()
    cos_sin = torch.randn(T, 6).to(BF16)
    positions = torch.arange(T, dtype=torch.long).flip(0)
    return dict(
        x=torch.randn(T, HIDDEN).to(BF16),
        x_norm_weight=torch.ones(HIDDEN, dtype=BF16),
        adaln_shift=shift,
        adaln_scale=scale,
        adaln_index=torch.arange(T, dtype=torch.long) % ROWS,
        qkv_weight=torch.randn(3 * HEADS * HEAD_DIM, HIDDEN).to(BF16),
        q_norm_weight=torch.ones(HEAD_DIM, dtype=BF16),
        k_norm_weight=torch.ones(HEAD_DIM, dtype=BF16),
        rope_cache=(cos_sin, positions),
        eps=1e-5,
    )


def _pre_attention_side_effect(*args, **kw):
    out = kw["out"]
    for kind in range(3):
        out[:, :, :, kind, :] = float(kind + 1)
    return out


# ---------------------------------------------------------------------------
# registration / import hygiene
# ---------------------------------------------------------------------------


def test_route_names_exist_in_route_table():
    from sglang.kernels.cake_kernels._routes import ROUTES

    assert mod.ROUTE in ROUTES
    assert "msa_nvfp4_sparse_decode" in ROUTES


def test_module_import_loads_no_flashinfer():
    # The stock DiT already imports ``sglang.kernels.ops.diffusion`` (whose
    # package registers the Cake specs lazily); what must not happen is a
    # FlashInfer import at module import time.  Fresh interpreter: other tests
    # in the session legitimately import FlashInfer-backed sglang modules.
    code = (
        "import sys; "
        "import sglang.multimodal_gen.runtime.models.dits.minimax_h3_cake_routes; "
        "bad = sorted(n for n in sys.modules if n.split('.')[0] == 'flashinfer'); "
        "assert not bad, bad"
    )
    subprocess.run([sys.executable, "-c", code], check=True, timeout=600)


# ---------------------------------------------------------------------------
# stage 1: fused pre-attention
# ---------------------------------------------------------------------------


def test_pre_attention_route_off_returns_none_without_admission():
    supports = mock.Mock(return_value=True)
    cake = mock.Mock(side_effect=_pre_attention_side_effect)
    with (
        _routes(),
        mock.patch.object(mod, "_pre_attention_kernels", lambda: (supports, cake)),
    ):
        assert mod.pre_attention(**_pre_attention_inputs()) is None
    supports.assert_not_called()
    cake.assert_not_called()


def test_pre_attention_admitted_uses_cake_with_contract_tensors(caplog):
    caplog.set_level(logging.INFO, logger=mod.logger.name)
    supports = mock.Mock(return_value=True)
    cake = mock.Mock(side_effect=_pre_attention_side_effect)
    inputs = _pre_attention_inputs()
    with (
        _routes(mod.ROUTE),
        _capturing(False),
        mock.patch.object(mod, "_pre_attention_kernels", lambda: (supports, cake)),
    ):
        result = mod.pre_attention(**inputs)
    assert result is not None
    q, k, v = result
    cake.assert_called_once()
    args, kw = cake.call_args
    x, x_norm, scale, shift, index, qkv_w, q_norm, k_norm, rope = args
    assert x is inputs["x"] and x_norm is inputs["x_norm_weight"]
    assert qkv_w is inputs["qkv_weight"]
    assert q_norm is inputs["q_norm_weight"] and k_norm is inputs["k_norm_weight"]
    # FlashInfer contract: contiguous [9, H] tables, int32 index, per-row RoPE.
    assert scale.is_contiguous() and torch.equal(scale, inputs["adaln_scale"])
    assert shift.is_contiguous() and torch.equal(shift, inputs["adaln_shift"])
    assert index.dtype == torch.int32
    assert torch.equal(index.long(), inputs["adaln_index"])
    cos_sin, positions = inputs["rope_cache"]
    assert torch.equal(rope, cos_sin[positions])
    assert kw["ulysses_degree"] == 1 and kw["eps"] == 1e-5
    assert tuple(kw["out"].shape) == (1, T, HEADS, 3, HEAD_DIM)
    assert kw["out"].dtype == BF16
    # Admission saw the same tensors the forwarder received.
    s_args, s_kw = supports.call_args
    assert all(a is b for a, b in zip(s_args, args))
    assert s_kw["out"] is kw["out"] and s_kw["ulysses_degree"] == 1
    # Return contract: Q/K/V views of the packed output.
    for kind, t in enumerate((q, k, v)):
        assert tuple(t.shape) == (T, HEADS, HEAD_DIM)
        assert torch.all(t == float(kind + 1))
    taken = f"[cake-route] {mod.ROUTE}/{mod.STAGE_PRE_ATTENTION}: Cake kernel selected"
    assert taken in caplog.text


def test_pre_attention_rejected_falls_back_and_caches_verdict(caplog):
    caplog.set_level(logging.INFO, logger=mod.logger.name)
    supports = mock.Mock(return_value=False)
    cake = mock.Mock(side_effect=_pre_attention_side_effect)
    inputs = _pre_attention_inputs()
    with (
        _routes(mod.ROUTE),
        _capturing(False),
        mock.patch.object(mod, "_pre_attention_kernels", lambda: (supports, cake)),
    ):
        assert mod.pre_attention(**inputs) is None
        assert mod.pre_attention(**inputs) is None
    cake.assert_not_called()
    assert supports.call_count == 1  # the rejected shape key is not retried
    assert caplog.text.count(f"{mod.STAGE_PRE_ATTENTION}: fallback") == 1
    assert "adapter admission rejected" in caplog.text


def test_pre_attention_without_rope_or_weight_stays_stock():
    supports = mock.Mock(return_value=True)
    cake = mock.Mock(side_effect=_pre_attention_side_effect)
    with (
        _routes(mod.ROUTE),
        _capturing(False),
        mock.patch.object(mod, "_pre_attention_kernels", lambda: (supports, cake)),
    ):
        inputs = _pre_attention_inputs()
        inputs["rope_cache"] = None  # token refiner: no RoPE
        assert mod.pre_attention(**inputs) is None
        inputs = _pre_attention_inputs()
        inputs["qkv_weight"] = None  # quantized linear without a plain weight
        assert mod.pre_attention(**inputs) is None
    supports.assert_not_called()
    cake.assert_not_called()


def test_pre_attention_capture_falls_back_until_warmed(caplog):
    caplog.set_level(logging.INFO, logger=mod.logger.name)
    supports = mock.Mock(return_value=True)
    cake = mock.Mock(side_effect=_pre_attention_side_effect)
    inputs = _pre_attention_inputs()
    with (
        _routes(mod.ROUTE),
        mock.patch.object(mod, "_pre_attention_kernels", lambda: (supports, cake)),
    ):
        with _capturing(True):
            assert mod.pre_attention(**inputs) is None
        cake.assert_not_called()
        assert "inside CUDA-graph capture" in caplog.text
        with _capturing(False):
            assert mod.pre_attention(**inputs) is not None
        with _capturing(True):
            assert mod.pre_attention(**inputs) is not None
    assert cake.call_count == 2


def test_pre_attention_flashinfer_rejection_falls_back_for_good():
    supports = mock.Mock(return_value=True)
    cake = mock.Mock(side_effect=ValueError("host validation failed"))
    inputs = _pre_attention_inputs()
    with (
        _routes(mod.ROUTE),
        _capturing(False),
        mock.patch.object(mod, "_pre_attention_kernels", lambda: (supports, cake)),
    ):
        assert mod.pre_attention(**inputs) is None
        assert mod.pre_attention(**inputs) is None
    assert cake.call_count == 1 and supports.call_count == 1


# ---------------------------------------------------------------------------
# stage 2: packed-varlen attention
# ---------------------------------------------------------------------------


def _strided_qkv():
    """q/k/v as the engine builds them: views of the fused [T, 3*H*D] projection."""
    qkv = torch.randn(T, 3 * HEADS * HEAD_DIM).to(BF16)
    q, k, v = qkv.split(HEADS * HEAD_DIM, dim=-1)
    q, k, v = (t.view(T, HEADS, HEAD_DIM) for t in (q, k, v))
    assert not q.is_contiguous()
    return q, k, v


def test_attention_route_off_returns_none():
    q, k, v = _strided_qkv()
    cu = torch.tensor([0, T], dtype=torch.int32)
    supports = mock.Mock(return_value=True)
    cake = mock.Mock()
    with (
        _routes(),
        mock.patch.object(mod, "_attention_kernels", lambda: (supports, cake)),
    ):
        out = mod.varlen_attention(
            q, k, v, cu, cu_seqlens_host=(0, T), softmax_scale=0.5
        )
    assert out is None
    supports.assert_not_called()
    cake.assert_not_called()


def test_attention_admitted_passes_contiguous_copies_and_scale(caplog):
    caplog.set_level(logging.INFO, logger=mod.logger.name)
    q, k, v = _strided_qkv()
    cu = torch.tensor([0, 4, T], dtype=torch.int32)
    expected = torch.full((T, HEADS, HEAD_DIM), 3.0, dtype=BF16)
    supports = mock.Mock(return_value=True)
    cake = mock.Mock(return_value=expected)
    with (
        _routes(mod.ROUTE),
        _capturing(False),
        mock.patch.object(mod, "_attention_kernels", lambda: (supports, cake)),
    ):
        out = mod.varlen_attention(
            q, k, v, cu, cu_seqlens_host=[0, 4, T], softmax_scale=0.25
        )
    assert out is expected
    args, kw = cake.call_args
    cq, ck, cv, ccu = args
    # FlashInfer contract: contiguous THD operands (one copy each), same values.
    assert cq.is_contiguous() and ck.is_contiguous() and cv.is_contiguous()
    assert torch.equal(cq, q) and torch.equal(ck, k) and torch.equal(cv, v)
    assert ccu is cu
    assert kw == {"softmax_scale": 0.25, "cu_seqlens_host": (0, 4, T)}
    s_args = supports.call_args.args
    assert s_args[0] is cq and s_args[1] is ck and s_args[2] is cv and s_args[3] is cu
    assert f"{mod.STAGE_ATTENTION}: Cake kernel selected" in caplog.text


def test_attention_inside_capture_always_falls_back(caplog):
    caplog.set_level(logging.INFO, logger=mod.logger.name)
    q, k, v = (t.contiguous() for t in _strided_qkv())
    cu = torch.tensor([0, T], dtype=torch.int32)
    supports = mock.Mock(return_value=True)
    cake = mock.Mock(return_value=torch.zeros(T, HEADS, HEAD_DIM, dtype=BF16))
    with (
        _routes(mod.ROUTE),
        mock.patch.object(mod, "_attention_kernels", lambda: (supports, cake)),
    ):
        with _capturing(False):
            assert (
                mod.varlen_attention(
                    q, k, v, cu, cu_seqlens_host=None, softmax_scale=0.5
                )
                is not None
            )
        with _capturing(True):
            assert (
                mod.varlen_attention(
                    q, k, v, cu, cu_seqlens_host=None, softmax_scale=0.5
                )
                is None
            )
    assert cake.call_count == 1
    assert "inside a CUDA-graph capture" in caplog.text


def test_attention_flashinfer_rejection_is_cached():
    q, k, v = (t.contiguous() for t in _strided_qkv())
    cu = torch.tensor([0, T], dtype=torch.int32)
    supports = mock.Mock(return_value=True)
    cake = mock.Mock(side_effect=NotImplementedError("no route for sm_90"))
    with (
        _routes(mod.ROUTE),
        _capturing(False),
        mock.patch.object(mod, "_attention_kernels", lambda: (supports, cake)),
    ):
        assert (
            mod.varlen_attention(q, k, v, cu, cu_seqlens_host=None, softmax_scale=0.5)
            is None
        )
        assert (
            mod.varlen_attention(q, k, v, cu, cu_seqlens_host=None, softmax_scale=0.5)
            is None
        )
    assert cake.call_count == 1 and supports.call_count == 1


# ---------------------------------------------------------------------------
# stage 3: out-projection + gated residual
# ---------------------------------------------------------------------------


def _out_proj_inputs():
    _, _, gate, *_ = _adaln_tables()
    return dict(
        attn_out=torch.randn(T, HEADS, HEAD_DIM).to(BF16),
        o_weight=torch.randn(HIDDEN, HEADS * HEAD_DIM).to(BF16),
        gate=gate,
        gate_index=torch.arange(T, dtype=torch.long) % ROWS,
        residual=torch.randn(T, HIDDEN).to(BF16),
    )


def _out_proj_side_effect(*args, **kw):
    kw["out"].fill_(7.0)
    return kw["out"]


def test_out_proj_route_off_or_strided_input_returns_none():
    supports = mock.Mock(return_value=True)
    cake = mock.Mock(side_effect=_out_proj_side_effect)
    inputs = _out_proj_inputs()
    with mock.patch.object(mod, "_out_proj_kernels", lambda: (supports, cake)):
        with _routes():
            assert mod.out_proj_gated_residual(**inputs) is None
        with _routes(mod.ROUTE), _capturing(False):
            strided = dict(inputs, attn_out=inputs["attn_out"].transpose(0, 1))
            assert mod.out_proj_gated_residual(**strided) is None
    supports.assert_not_called()
    cake.assert_not_called()


def test_out_proj_admitted_uses_cake_with_contract_tensors(caplog):
    caplog.set_level(logging.INFO, logger=mod.logger.name)
    supports = mock.Mock(return_value=True)
    cake = mock.Mock(side_effect=_out_proj_side_effect)
    inputs = _out_proj_inputs()
    with (
        _routes(mod.ROUTE),
        _capturing(False),
        mock.patch.object(mod, "_out_proj_kernels", lambda: (supports, cake)),
    ):
        out = mod.out_proj_gated_residual(**inputs)
    args, kw = cake.call_args
    packed, o_weight, gate, index, residual = args
    # Ulysses receive layout at P=1 is the free [1, T, H, D] view.
    assert tuple(packed.shape) == (1, T, HEADS, HEAD_DIM)
    assert packed.data_ptr() == inputs["attn_out"].data_ptr()
    assert o_weight is inputs["o_weight"] and residual is inputs["residual"]
    assert gate.is_contiguous() and torch.equal(gate, inputs["gate"])
    assert index.dtype == torch.int32
    assert torch.equal(index.long(), inputs["gate_index"])
    assert tuple(kw["out"].shape) == (T, HIDDEN) and kw["out"].dtype == BF16
    assert out is kw["out"] and torch.all(out == 7.0)
    assert torch.equal(inputs["residual"], inputs["residual"])  # never written
    s_args = supports.call_args.args
    assert all(a is b for a, b in zip(s_args, args))
    assert f"{mod.STAGE_OUT_PROJ}: Cake kernel selected" in caplog.text


def test_out_proj_rejected_falls_back_once_logged(caplog):
    caplog.set_level(logging.INFO, logger=mod.logger.name)
    supports = mock.Mock(return_value=False)
    cake = mock.Mock(side_effect=_out_proj_side_effect)
    inputs = _out_proj_inputs()
    with (
        _routes(mod.ROUTE),
        _capturing(False),
        mock.patch.object(mod, "_out_proj_kernels", lambda: (supports, cake)),
    ):
        assert mod.out_proj_gated_residual(**inputs) is None
        assert mod.out_proj_gated_residual(**inputs) is None
    cake.assert_not_called()
    assert supports.call_count == 1
    assert caplog.text.count(f"{mod.STAGE_OUT_PROJ}: fallback") == 1


# ---------------------------------------------------------------------------
# stage 4: fc1 + SwiGLU
# ---------------------------------------------------------------------------


def _fc1_inputs():
    _, _, _, shift, scale, _ = _adaln_tables()
    return dict(
        x=torch.randn(T, HIDDEN).to(BF16),
        x_norm_weight=torch.ones(HIDDEN, dtype=BF16),
        adaln_shift=shift,
        adaln_scale=scale,
        adaln_index=torch.arange(T, dtype=torch.long) % ROWS,
        fc1_weight=torch.randn(2 * FFN, HIDDEN).to(BF16),
        eps=1e-5,
    )


def _fc1_side_effect(*args, **kw):
    kw["out"].fill_(5.0)
    kw["workspace"].fill_(1.0)
    return kw["out"]


def test_fc1_route_off_returns_none():
    supports = mock.Mock(return_value=True)
    cake = mock.Mock(side_effect=_fc1_side_effect)
    with _routes(), mock.patch.object(mod, "_fc1_kernels", lambda: (supports, cake)):
        assert mod.fc1_swiglu(**_fc1_inputs()) is None
    supports.assert_not_called()
    cake.assert_not_called()


def test_fc1_admitted_uses_cake_with_owned_buffers(caplog):
    caplog.set_level(logging.INFO, logger=mod.logger.name)
    supports = mock.Mock(return_value=True)
    cake = mock.Mock(side_effect=_fc1_side_effect)
    inputs = _fc1_inputs()
    with (
        _routes(mod.ROUTE),
        _capturing(False),
        mock.patch.object(mod, "_fc1_kernels", lambda: (supports, cake)),
    ):
        hidden = mod.fc1_swiglu(**inputs)
    args, kw = cake.call_args
    x, x_norm, scale, shift, index, fc1_w = args
    assert x is inputs["x"] and x_norm is inputs["x_norm_weight"]
    assert fc1_w is inputs["fc1_weight"]
    assert scale.is_contiguous() and torch.equal(scale, inputs["adaln_scale"])
    assert shift.is_contiguous() and torch.equal(shift, inputs["adaln_shift"])
    assert index.dtype == torch.int32
    # Caller-owned out [T, FFN] and workspace [T, HIDDEN] keep the launch
    # allocation-free for graph capture.
    assert tuple(kw["out"].shape) == (T, FFN) and kw["out"].dtype == BF16
    assert tuple(kw["workspace"].shape) == (T, HIDDEN)
    assert kw["workspace"].dtype == BF16 and kw["eps"] == 1e-5
    assert hidden is kw["out"] and torch.all(hidden == 5.0)
    s_args, s_kw = supports.call_args
    assert all(a is b for a, b in zip(s_args, args)) and s_kw == {"eps": 1e-5}
    assert f"{mod.STAGE_FC1}: Cake kernel selected" in caplog.text


def test_fc1_capture_guard_and_rejection_cache(caplog):
    caplog.set_level(logging.INFO, logger=mod.logger.name)
    supports = mock.Mock(return_value=True)
    cake = mock.Mock(side_effect=_fc1_side_effect)
    inputs = _fc1_inputs()
    with mock.patch.object(mod, "_fc1_kernels", lambda: (supports, cake)):
        with _routes(mod.ROUTE), _capturing(True):
            assert mod.fc1_swiglu(**inputs) is None  # not warmed yet
        with _routes(mod.ROUTE), _capturing(False):
            assert mod.fc1_swiglu(**inputs) is not None
        with _routes(mod.ROUTE), _capturing(True):
            assert mod.fc1_swiglu(**inputs) is not None  # warmed: captured
        assert cake.call_count == 2
        cake.side_effect = RuntimeError("FlashInfer host check")
        with _routes(mod.ROUTE), _capturing(False):
            assert mod.fc1_swiglu(**inputs) is None
            assert mod.fc1_swiglu(**inputs) is None
        assert cake.call_count == 3  # rejected shape key is not retried
    assert "FlashInfer rejected the call" in caplog.text


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
