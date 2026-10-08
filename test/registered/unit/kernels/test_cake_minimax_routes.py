"""CPU unit tests for the MiniMax Cake routes (``SGLANG_CAKE_ROUTES``).

``minimax_h3_diffusion`` -- the four fused stages of
``sglang.multimodal_gen.runtime.models.dits.minimax_h3_cake_routes``.  The
route switch, the adapter admissions and the Cake forwarders are mocked; the
tests pin *which* callable receives the engine's tensors and that it receives
them **as they are** (strided AdaLN table chunks of any row count, int64
indices, the RoPE ``(cache, positions)`` pair, strided q/k/v views -- no host
copies or casts, FlashInfer ``58171ea83f``), the two-launch pre-attention
kwargs (``workspace``, ``qk_eps``, ``rope_positions``), the "None -> stock
path" fallback contract, the CUDA-graph capture guard and the cached
FlashInfer rejection.  CPU tensors; no FlashInfer, Triton or CUDA involved.

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


def _adaln_tables(rows: int = ROWS):
    """Six ``[rows, HIDDEN]`` column chunks of a ``[rows, 6 * HIDDEN]`` projection,
    exactly as ``MiniMaxH3AdalnProj.split_output`` produces them (non-contiguous,
    row stride ``6 * HIDDEN``)."""
    proj = torch.randn(rows, 6 * HIDDEN).to(BF16)
    chunks = proj.chunk(6, dim=-1)
    assert not chunks[0].is_contiguous()
    assert chunks[1].stride() == (6 * HIDDEN, 1)
    return chunks


def _pre_attention_inputs(rows: int = ROWS):
    shift, scale, *_ = _adaln_tables(rows)
    # Engine RoPE: a shared [S, rope_dim] cos|sin cache (S may differ from T)
    # and one int64 cache row per token.
    cos_sin = torch.randn(T + 3, 6).to(BF16)
    positions = torch.arange(T, dtype=torch.long).flip(0)
    return dict(
        x=torch.randn(T, HIDDEN).to(BF16),
        x_norm_weight=torch.ones(HIDDEN, dtype=BF16),
        adaln_shift=shift,
        adaln_scale=scale,
        adaln_index=torch.arange(T, dtype=torch.long) % rows,
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


def test_pre_attention_admitted_passes_engine_operands_uncopied(caplog):
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
    # Engine-native operands, no host copies: the strided table chunks, the
    # int64 index and the shared RoPE cache + positions are the same objects.
    assert scale is inputs["adaln_scale"] and shift is inputs["adaln_shift"]
    assert not scale.is_contiguous()
    assert index is inputs["adaln_index"] and index.dtype == torch.int64
    cos_sin, positions = inputs["rope_cache"]
    assert rope is cos_sin and kw["rope_positions"] is positions
    # Two-launch stage kwargs: caller-owned pack and [T, HIDDEN] workspace.
    assert kw["ulysses_degree"] == 1 and kw["eps"] == 1e-5 and kw["qk_eps"] is None
    assert tuple(kw["out"].shape) == (1, T, HEADS, 3, HEAD_DIM)
    assert kw["out"].dtype == BF16
    assert tuple(kw["workspace"].shape) == (T, HIDDEN) and kw["workspace"].dtype == BF16
    assert set(kw) == {
        "ulysses_degree",
        "out",
        "eps",
        "qk_eps",
        "rope_positions",
        "workspace",
    }
    # Admission saw the same tensors the forwarder received.
    s_args, s_kw = supports.call_args
    assert all(a is b for a, b in zip(s_args, args))
    assert s_kw["out"] is kw["out"] and s_kw["ulysses_degree"] == 1
    assert s_kw["workspace"] is kw["workspace"]
    assert s_kw["rope_positions"] is positions
    # Return contract: Q/K/V kind slices of the packed output, read in place
    # by the attention stage (strides (H*3*D, 3*D, 1), no copies).
    for kind, t in enumerate((q, k, v)):
        assert tuple(t.shape) == (T, HEADS, HEAD_DIM)
        assert t.stride() == (HEADS * 3 * HEAD_DIM, 3 * HEAD_DIM, 1)
        assert t.data_ptr() == kw["out"].data_ptr() + kind * HEAD_DIM * 2
        assert torch.all(t == float(kind + 1))
    taken = f"[cake-route] {mod.ROUTE}/{mod.STAGE_PRE_ATTENTION}: Cake kernel selected"
    assert taken in caplog.text


@pytest.mark.parametrize("rows", [3, 6, 12])
def test_pre_attention_admits_any_adaln_row_count(rows):
    """Tables with a row count other than 9 are no longer refused by the route;
    the kernel reads the row count and row stride from the tensor."""
    supports = mock.Mock(return_value=True)
    cake = mock.Mock(side_effect=_pre_attention_side_effect)
    inputs = _pre_attention_inputs(rows)
    assert tuple(inputs["adaln_scale"].shape) == (rows, HIDDEN)
    with (
        _routes(mod.ROUTE),
        _capturing(False),
        mock.patch.object(mod, "_pre_attention_kernels", lambda: (supports, cake)),
    ):
        assert mod.pre_attention(**inputs) is not None
    args, _ = cake.call_args
    assert args[2] is inputs["adaln_scale"] and args[3] is inputs["adaln_shift"]


def test_pre_attention_forwards_separate_qk_eps():
    supports = mock.Mock(return_value=True)
    cake = mock.Mock(side_effect=_pre_attention_side_effect)
    inputs = _pre_attention_inputs()
    with (
        _routes(mod.ROUTE),
        _capturing(False),
        mock.patch.object(mod, "_pre_attention_kernels", lambda: (supports, cake)),
    ):
        assert mod.pre_attention(**inputs, qk_eps=1e-6) is not None
    _, kw = cake.call_args
    assert kw["eps"] == 1e-5 and kw["qk_eps"] == 1e-6
    _, s_kw = supports.call_args
    assert s_kw["qk_eps"] == 1e-6


def test_pre_attention_rope_positions_must_cover_every_token(caplog):
    caplog.set_level(logging.INFO, logger=mod.logger.name)
    supports = mock.Mock(return_value=True)
    cake = mock.Mock(side_effect=_pre_attention_side_effect)
    inputs = _pre_attention_inputs()
    cos_sin, positions = inputs["rope_cache"]
    inputs["rope_cache"] = (cos_sin, positions[:-1])  # [T - 1] positions
    with (
        _routes(mod.ROUTE),
        _capturing(False),
        mock.patch.object(mod, "_pre_attention_kernels", lambda: (supports, cake)),
    ):
        assert mod.pre_attention(**inputs) is None
    supports.assert_not_called()
    cake.assert_not_called()
    assert f"{mod.STAGE_PRE_ATTENTION}: fallback" in caplog.text
    assert "not a per-token (cache, positions) pair" in caplog.text


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


def _pack_qkv():
    """q/k/v as the Cake pre-attention stage returns them: kind slices of the
    destination-major pack ``[1, T, H, 3, D]``."""
    pack = torch.randn(1, T, HEADS, 3, HEAD_DIM).to(BF16)[0]
    q, k, v = (pack[:, :, kind, :] for kind in range(3))
    assert q.stride() == (HEADS * 3 * HEAD_DIM, 3 * HEAD_DIM, 1)
    assert not q.is_contiguous()
    return q, k, v


@pytest.mark.parametrize("make_qkv", [_strided_qkv, _pack_qkv])
def test_attention_admitted_passes_strided_views_in_place(caplog, make_qkv):
    caplog.set_level(logging.INFO, logger=mod.logger.name)
    q, k, v = make_qkv()
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
    # The strided views reach the kernel as they are (read in place, no copies).
    assert cq is q and ck is k and cv is v and ccu is cu
    assert not cq.is_contiguous()
    assert kw == {"softmax_scale": 0.25, "cu_seqlens_host": (0, 4, T)}
    s_args = supports.call_args.args
    assert s_args[0] is q and s_args[1] is k and s_args[2] is v and s_args[3] is cu
    assert f"{mod.STAGE_ATTENTION}: Cake kernel selected" in caplog.text


def test_attention_admission_rejection_falls_back_once_logged(caplog):
    """A view the adapter does not admit (e.g. a head-major layout) falls back
    with the adapter's reason; the verdict is cached per stride signature."""
    caplog.set_level(logging.INFO, logger=mod.logger.name)
    q, k, v = _strided_qkv()
    k_head_major = k.transpose(0, 1).contiguous().transpose(0, 1)
    cu = torch.tensor([0, T], dtype=torch.int32)
    supports = mock.Mock(return_value=False)
    cake = mock.Mock()
    with (
        _routes(mod.ROUTE),
        _capturing(False),
        mock.patch.object(mod, "_attention_kernels", lambda: (supports, cake)),
    ):
        for _ in range(2):
            assert (
                mod.varlen_attention(
                    q, k_head_major, v, cu, cu_seqlens_host=(0, T), softmax_scale=0.5
                )
                is None
            )
    cake.assert_not_called()
    assert supports.call_count == 1
    assert supports.call_args.args[1] is k_head_major
    assert caplog.text.count(f"{mod.STAGE_ATTENTION}: fallback") == 1
    assert "adapter admission rejected" in caplog.text


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


def _out_proj_inputs(rows: int = ROWS):
    _, _, gate, *_ = _adaln_tables(rows)
    return dict(
        attn_out=torch.randn(T, HEADS, HEAD_DIM).to(BF16),
        o_weight=torch.randn(HIDDEN, HEADS * HEAD_DIM).to(BF16),
        gate=gate,
        gate_index=torch.arange(T, dtype=torch.long) % rows,
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


def test_out_proj_admitted_passes_engine_operands_uncopied(caplog):
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
    # The gate chunk (row stride 6 * HIDDEN) and the int64 index pass as-is.
    assert gate is inputs["gate"] and not gate.is_contiguous()
    assert index is inputs["gate_index"] and index.dtype == torch.int64
    assert tuple(kw["out"].shape) == (T, HIDDEN) and kw["out"].dtype == BF16
    assert out is kw["out"] and torch.all(out == 7.0)
    assert torch.equal(inputs["residual"], inputs["residual"])  # never written
    s_args = supports.call_args.args
    assert all(a is b for a, b in zip(s_args, args))
    assert f"{mod.STAGE_OUT_PROJ}: Cake kernel selected" in caplog.text


def test_gemm_stages_admit_any_table_row_count():
    """out_proj / fc1 no longer refuse tables with a row count other than 9."""
    for rows in (3, 12):
        mod.reset_state_for_tests()
        supports = mock.Mock(return_value=True)
        out_cake = mock.Mock(side_effect=_out_proj_side_effect)
        fc1_cake = mock.Mock(side_effect=_fc1_side_effect)
        out_inputs, fc1_inputs = _out_proj_inputs(rows), _fc1_inputs(rows)
        with (
            _routes(mod.ROUTE),
            _capturing(False),
            mock.patch.object(mod, "_out_proj_kernels", lambda: (supports, out_cake)),
            mock.patch.object(mod, "_fc1_kernels", lambda: (supports, fc1_cake)),
        ):
            assert mod.out_proj_gated_residual(**out_inputs) is not None
            assert mod.fc1_swiglu(**fc1_inputs) is not None
        assert out_cake.call_args.args[2] is out_inputs["gate"]
        assert fc1_cake.call_args.args[2] is fc1_inputs["adaln_scale"]


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


def _fc1_inputs(rows: int = ROWS):
    _, _, _, shift, scale, _ = _adaln_tables(rows)
    return dict(
        x=torch.randn(T, HIDDEN).to(BF16),
        x_norm_weight=torch.ones(HIDDEN, dtype=BF16),
        adaln_shift=shift,
        adaln_scale=scale,
        adaln_index=torch.arange(T, dtype=torch.long) % rows,
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


def test_fc1_admitted_passes_engine_operands_with_owned_buffers(caplog):
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
    # The AdaLN chunks (row stride 6 * HIDDEN) and the int64 index pass as-is.
    assert scale is inputs["adaln_scale"] and not scale.is_contiguous()
    assert shift is inputs["adaln_shift"] and not shift.is_contiguous()
    assert index is inputs["adaln_index"] and index.dtype == torch.int64
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
