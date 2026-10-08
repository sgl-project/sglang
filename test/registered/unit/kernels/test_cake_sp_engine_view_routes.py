"""CPU unit tests for the ``sp_all_gather_matmul`` Cake route of the
LayerNorm-SP column-parallel participant on FlashInfer's engine-view contract
(``prepare_all_gather_matmul(inp, weight.t(), group, backend="cake",
max_rows=...)``: any local row count, any ``N % 256 == 0``, the ``[N, K]``
parameter read in place through its transposed view).

Everything is mocked: the route switch, the adapter admission, the Cake
forwarder and the stock linear method. The tests check *which* callable
receives the engine's tensors, that the prepared launcher is bound to the
engine's weight view (no copy) with a 128-row-tile capacity, when it is
re-prepared (capacity growth, re-bound parameter storage) and when it is
not (smaller row counts, in-place weight reload), that every refusal falls
back to the stock path and is logged once, and that the NVSHMEM
symmetric-memory selection / refusal path is unchanged. CPU tensors; no
FlashInfer, CUDA or process group involved.
"""

import contextlib
import logging
import sys
import types
from types import SimpleNamespace
from unittest import mock

import pytest
import torch

from sglang.kernels.cake_kernels import communication as comm_mod
from sglang.kernels.ops.communication import cake as comm_ops
from sglang.srt.layers import layernorm_sp as sp_mod
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=10, stage="base-a-test-cpu")

TP = 4
K, N = 32, 16
ROWS = 4  # local shard rows (M_pad / tp); deliberately not a multiple of 128
NUM_TOKENS = TP * ROWS - 3  # real tokens: the exit narrow drops the padding
TILE = sp_mod._CAKE_SP_ROW_TILE


@pytest.fixture(autouse=True)
def _reset_route_state():
    sp_mod.reset_cake_sp_state_for_tests()
    yield
    sp_mod.reset_cake_sp_state_for_tests()


def _routes(*enabled):
    return mock.patch.object(sp_mod, "cake_route_enabled", lambda name: name in enabled)


def _not_capturing():
    return mock.patch.object(
        torch.cuda, "is_current_stream_capturing", lambda: False, create=True
    )


def _capturing():
    return mock.patch.object(
        torch.cuda, "is_current_stream_capturing", lambda: True, create=True
    )


@pytest.fixture
def sp_env():
    """Fake TP group, SP token count and a fake ``UnquantizedLinearMethod``."""
    fake_unquant = types.ModuleType("sglang.srt.layers.quantization.unquant")

    class UnquantizedLinearMethod:
        def __init__(self):
            self.apply = mock.Mock(
                side_effect=lambda linear, x, bias: torch.full(
                    (x.shape[0], linear.weight.shape[0]), 1.0, dtype=x.dtype
                )
            )

    fake_unquant.UnquantizedLinearMethod = UnquantizedLinearMethod
    tp_group = SimpleNamespace(
        world_size=TP,
        rank_in_group=0,
        device_group=SimpleNamespace(group_name="tp"),
    )
    with (
        mock.patch.dict(sys.modules, {fake_unquant.__name__: fake_unquant}),
        mock.patch.object(
            sp_mod, "get_parallel", lambda: SimpleNamespace(tp_group=tp_group)
        ),
        mock.patch.object(sp_mod, "_HAS_TORCH_SYMM_MEM_FUSED", False),
        mock.patch.object(
            sp_mod,
            "sp_exit_gather",
            lambda h, num_tokens: h.repeat(TP, 1)[:num_tokens],
        ),
        mock.patch.object(sp_mod._sp_state, "num_tokens", NUM_TOKENS),
        _not_capturing(),
    ):
        yield SimpleNamespace(
            group=tp_group.device_group, method_cls=UnquantizedLinearMethod
        )


def _linear(sp_env, bias=None, quantized=False):
    linear = torch.nn.Module()
    linear.weight = torch.nn.Parameter(
        torch.randn(N, K, dtype=torch.bfloat16), requires_grad=False
    )
    linear.bias = bias
    linear.quant_method = (
        SimpleNamespace(apply=mock.Mock()) if quantized else sp_env.method_cls()
    )
    return linear


def _sp_kernels(*, admitted=True, prepare_raises=None):
    """Fake ``(supports_prepare, prepare)`` of ``layernorm_sp._cake_sp_kernels``.

    The fake launcher mirrors FlashInfer's ``launcher(inp, *, out=None)`` and
    returns ``[rows * TP, N]`` filled with 2.0; the stock method returns 1.0.
    """
    supports_prepare = mock.Mock(return_value=admitted)
    launchers = []

    def _prepare(inp, w, group, *, max_rows=None):
        if prepare_raises is not None:
            raise prepare_raises("the Cake backend requires exact K=8192")
        launcher = mock.Mock(
            side_effect=lambda x, out=None: torch.full(
                (x.shape[0] * TP, w.shape[1]), 2.0, dtype=x.dtype
            )
        )
        launcher.max_rows = max_rows
        launchers.append(launcher)
        return launcher

    prepare = mock.Mock(side_effect=_prepare)
    return (supports_prepare, prepare), launchers


def _patch_sp_kernels(kernels):
    return mock.patch.object(sp_mod, "_cake_sp_kernels", lambda: kernels)


def _is_engine_weight_view(w, linear):
    """``w`` is ``linear.weight.t()``: the K-major view sharing the parameter storage."""
    return (
        tuple(w.shape) == (K, N)
        and tuple(w.stride()) == (1, K)
        and w.data_ptr() == linear.weight.data_ptr()
        and torch.equal(w, linear.weight.detach().t())
    )


# ---------------------------------------------------------------------------
# route selection
# ---------------------------------------------------------------------------


def test_sp_route_off_uses_stock_gather_and_matmul(sp_env):
    kernels, _ = _sp_kernels()
    linear = _linear(sp_env)
    inp = torch.randn(ROWS, K).bfloat16()
    with _routes(), _patch_sp_kernels(kernels):
        out = sp_mod.column_parallel_g_matmul(linear, inp, None)
    linear.quant_method.apply.assert_called_once()
    gathered = linear.quant_method.apply.call_args.args[1]
    assert tuple(gathered.shape) == (NUM_TOKENS, K)
    for fn in kernels:
        fn.assert_not_called()
    assert tuple(out.shape) == (NUM_TOKENS, N) and torch.all(out == 1.0)


def test_sp_route_on_prepares_engine_weight_view_once_with_tile_capacity(
    sp_env, caplog
):
    caplog.set_level(logging.INFO, logger=sp_mod.logger.name)
    kernels, launchers = _sp_kernels()
    supports_prepare, prepare = kernels
    linear = _linear(sp_env)
    inp = torch.randn(ROWS, K).bfloat16()
    with _routes("sp_all_gather_matmul"), _patch_sp_kernels(kernels):
        out = sp_mod.column_parallel_g_matmul(linear, inp, None)
        out2 = sp_mod.column_parallel_g_matmul(linear, inp.clone(), None)
    linear.quant_method.apply.assert_not_called()
    # One collective preparation on the engine's own tensors: the input shard,
    # the [N, K] parameter through its transposed view (no copy), the TP
    # device group, and a capacity rounded up to FlashInfer's 128-row tile.
    prepare.assert_called_once()
    p_inp, p_w, p_group = prepare.call_args.args
    assert p_inp is inp and p_group is sp_env.group
    assert _is_engine_weight_view(p_w, linear)
    assert prepare.call_args.kwargs == {"max_rows": TILE}
    # Admission runs on the real tensors of every call, with the same view.
    assert supports_prepare.call_count == 2
    s_inp, s_w = supports_prepare.call_args_list[0].args
    assert s_inp is inp and _is_engine_weight_view(s_w, linear)
    assert supports_prepare.call_args.kwargs == {"world_size": TP}
    # The launcher serves both calls (rows <= capacity) and is called with the
    # bare input: the output is allocated by FlashInfer, not pre-staged here.
    assert len(launchers) == 1 and launchers[0].call_count == 2
    assert launchers[0].call_args_list[0].args[0] is inp
    assert launchers[0].call_args.kwargs == {}
    assert tuple(out.shape) == (NUM_TOKENS, N) and torch.all(out == 2.0)
    assert torch.all(out2 == 2.0)
    assert "[cake-route] sp_all_gather_matmul: Cake kernel selected" in caplog.text
    assert f"prepared launcher for {TILE} rows" in caplog.text


@pytest.mark.parametrize("rows", [1, 125, 130, 1025])
def test_sp_route_admits_any_local_row_count(sp_env, rows):
    kernels, launchers = _sp_kernels()
    linear = _linear(sp_env)
    inp = torch.randn(rows, K).bfloat16()
    with (
        _routes("sp_all_gather_matmul"),
        _patch_sp_kernels(kernels),
        mock.patch.object(sp_mod._sp_state, "num_tokens", rows * TP),
    ):
        out = sp_mod.column_parallel_g_matmul(linear, inp, None)
    linear.quant_method.apply.assert_not_called()
    assert launchers[0].max_rows == -(-rows // TILE) * TILE >= rows
    assert tuple(out.shape) == (rows * TP, N) and torch.all(out == 2.0)


def test_sp_route_grows_capacity_once_and_reuses_it_for_smaller_shards(sp_env):
    kernels, launchers = _sp_kernels()
    _, prepare = kernels
    linear = _linear(sp_env)
    small = torch.randn(ROWS, K).bfloat16()
    large = torch.randn(TILE + 72, K).bfloat16()
    medium = torch.randn(TILE - 1, K).bfloat16()
    with (
        _routes("sp_all_gather_matmul"),
        _patch_sp_kernels(kernels),
        mock.patch.object(sp_mod._sp_state, "num_tokens", TP * ROWS),
    ):
        sp_mod.column_parallel_g_matmul(linear, small, None)
        with mock.patch.object(sp_mod._sp_state, "num_tokens", TP * large.shape[0]):
            sp_mod.column_parallel_g_matmul(linear, large, None)
        with mock.patch.object(sp_mod._sp_state, "num_tokens", TP * medium.shape[0]):
            sp_mod.column_parallel_g_matmul(linear, medium, None)
        sp_mod.column_parallel_g_matmul(linear, small, None)
    # Prepared twice: the first tile, then the grown capacity; the medium and
    # the repeated small shard reuse the grown launcher without a collective.
    assert prepare.call_count == 2
    assert [launcher.max_rows for launcher in launchers] == [TILE, 2 * TILE]
    assert launchers[0].call_count == 1 and launchers[1].call_count == 3
    assert prepare.call_args.args[0] is large
    assert len(sp_mod._cake_sp_launchers) == 1


def test_sp_route_hands_a_contiguous_shard_to_the_kernel(sp_env):
    kernels, launchers = _sp_kernels()
    _, prepare = kernels
    linear = _linear(sp_env)
    strided = torch.randn(ROWS, 2 * K).bfloat16()[:, ::2]
    assert not strided.is_contiguous()
    with _routes("sp_all_gather_matmul"), _patch_sp_kernels(kernels):
        out = sp_mod.column_parallel_g_matmul(linear, strided, None)
    p_inp = prepare.call_args.args[0]
    assert p_inp.is_contiguous() and torch.equal(p_inp, strided)
    assert launchers[0].call_args.args[0].is_contiguous()
    assert torch.all(out == 2.0)


# ---------------------------------------------------------------------------
# refusals
# ---------------------------------------------------------------------------


def test_sp_route_adapter_rejection_falls_back_logs_once_and_is_cached(sp_env, caplog):
    caplog.set_level(logging.INFO, logger=sp_mod.logger.name)
    kernels, _ = _sp_kernels(admitted=False)
    supports_prepare, prepare = kernels
    linear = _linear(sp_env)
    inp = torch.randn(ROWS, K).bfloat16()
    with _routes("sp_all_gather_matmul"), _patch_sp_kernels(kernels):
        out = sp_mod.column_parallel_g_matmul(linear, inp, None)
        sp_mod.column_parallel_g_matmul(linear, inp, None)
    assert linear.quant_method.apply.call_count == 2
    assert torch.all(out == 1.0)
    # Admission depends on static facts (device, dtype, layout, backend), so
    # the refusal is cached per participant and the adapter is asked once.
    assert supports_prepare.call_count == 1
    prepare.assert_not_called()
    assert caplog.text.count("[cake-route] sp_all_gather_matmul: fallback") == 1
    assert "adapter admission rejected" in caplog.text


@pytest.mark.parametrize("raise_type", [ValueError, NotImplementedError])
def test_sp_flashinfer_prepare_refusal_falls_back_and_is_cached(
    sp_env, caplog, raise_type
):
    # FlashInfer validates its host contract (NVSHMEM backend, K, N % 256,
    # weight strides, world size) with ValueError: a refusal, not a crash.
    caplog.set_level(logging.INFO, logger=sp_mod.logger.name)
    kernels, _ = _sp_kernels(prepare_raises=raise_type)
    _, prepare = kernels
    linear = _linear(sp_env)
    inp = torch.randn(ROWS, K).bfloat16()
    with _routes("sp_all_gather_matmul"), _patch_sp_kernels(kernels):
        out = sp_mod.column_parallel_g_matmul(linear, inp, None)
        sp_mod.column_parallel_g_matmul(linear, inp, None)
    assert prepare.call_count == 1
    assert linear.quant_method.apply.call_count == 2
    assert torch.all(out == 1.0)
    assert caplog.text.count("[cake-route] sp_all_gather_matmul: fallback") == 1
    assert "FlashInfer refused to prepare" in caplog.text
    assert "K=8192" in caplog.text


def test_sp_flashinfer_runtime_errors_propagate(sp_env):
    # A poisoned collective / topology change is a RuntimeError in FlashInfer
    # and is not a refusal: it must surface, not be hidden by the stock path.
    kernels, _ = _sp_kernels(prepare_raises=RuntimeError)
    linear = _linear(sp_env)
    inp = torch.randn(ROWS, K).bfloat16()
    with (
        _routes("sp_all_gather_matmul"),
        _patch_sp_kernels(kernels),
        pytest.raises(RuntimeError),
    ):
        sp_mod.column_parallel_g_matmul(linear, inp, None)


@pytest.mark.parametrize("case", ["bias", "quantized", "capture"])
def test_sp_route_static_fallbacks_skip_adapter(sp_env, case):
    kernels, _ = _sp_kernels()
    bias = torch.zeros(N, dtype=torch.bfloat16) if case == "bias" else None
    linear = _linear(sp_env, bias=bias, quantized=(case == "quantized"))
    inp = torch.randn(ROWS, K).bfloat16()
    capture = _capturing() if case == "capture" else contextlib.nullcontext()
    with _routes("sp_all_gather_matmul"), _patch_sp_kernels(kernels), capture:
        sp_mod.column_parallel_g_matmul(linear, inp, bias)
    linear.quant_method.apply.assert_called_once()
    for fn in kernels:
        fn.assert_not_called()


# ---------------------------------------------------------------------------
# weight lifetime
# ---------------------------------------------------------------------------


def test_sp_in_place_weight_reload_keeps_the_launcher(sp_env):
    # The launcher reads the parameter through its view: an in-place reload
    # (copy_) changes no data pointer / stride and needs no re-preparation.
    kernels, launchers = _sp_kernels()
    _, prepare = kernels
    linear = _linear(sp_env)
    inp = torch.randn(ROWS, K).bfloat16()
    with _routes("sp_all_gather_matmul"), _patch_sp_kernels(kernels):
        sp_mod.column_parallel_g_matmul(linear, inp, None)
        with torch.no_grad():
            linear.weight.copy_(torch.randn(N, K, dtype=torch.bfloat16))
        sp_mod.column_parallel_g_matmul(linear, inp, None)
    assert prepare.call_count == 1 and launchers[0].call_count == 2
    assert torch.equal(prepare.call_args.args[1], linear.weight.detach().t())


def test_sp_rebound_weight_storage_reprepares_the_launcher(sp_env):
    # Re-binding the parameter to new storage invalidates FlashInfer's bound
    # weight fingerprint (data pointer): the route prepares a new launcher on
    # the new view instead of letting the old one raise.
    kernels, launchers = _sp_kernels()
    _, prepare = kernels
    linear = _linear(sp_env)
    inp = torch.randn(ROWS, K).bfloat16()
    with _routes("sp_all_gather_matmul"), _patch_sp_kernels(kernels):
        sp_mod.column_parallel_g_matmul(linear, inp, None)
        linear.weight.data = torch.randn(N, K, dtype=torch.bfloat16)
        sp_mod.column_parallel_g_matmul(linear, inp, None)
    assert prepare.call_count == 2 and len(launchers) == 2
    assert launchers[0].call_count == 1 and launchers[1].call_count == 1
    assert _is_engine_weight_view(prepare.call_args.args[1], linear)
    assert len(sp_mod._cake_sp_launchers) == 1


# ---------------------------------------------------------------------------
# per-call check (SGLANG_CAKE_SP_CHECK=1)
# ---------------------------------------------------------------------------


def test_sp_check_compares_the_narrowed_cake_output_against_stock(sp_env, caplog):
    caplog.set_level(logging.INFO, logger=sp_mod.logger.name)
    kernels, _ = _sp_kernels()
    linear = _linear(sp_env)
    inp = torch.randn(ROWS, K).bfloat16()
    stats = dict.fromkeys(sp_mod._cake_sp_check_stats, 0)
    stats["max_abs"] = 0.0
    with (
        _routes("sp_all_gather_matmul"),
        _patch_sp_kernels(kernels),
        mock.patch.object(sp_mod, "_CAKE_SP_CHECK", True),
        mock.patch.object(sp_mod, "_cake_sp_check_stats", stats),
    ):
        out = sp_mod.column_parallel_g_matmul(linear, inp, None)
    # The Cake result is served; the stock all-gather + matmul ran once more
    # as the reference (fake Cake 2.0 vs fake stock 1.0 is outside BF16 tol).
    assert torch.all(out == 2.0) and tuple(out.shape) == (NUM_TOKENS, N)
    linear.quant_method.apply.assert_called_once()
    ref_inp = linear.quant_method.apply.call_args.args[1]
    assert tuple(ref_inp.shape) == (NUM_TOKENS, K)
    assert stats["calls"] == 1 and stats["bad_calls"] == 1
    assert stats["elems"] == NUM_TOKENS * N == stats["bad_elems"]
    assert stats["max_abs"] == 1.0
    assert "sp check: " in caplog.text and "elements outside tol" in caplog.text
    assert (sp_mod._CAKE_SP_CHECK_ATOL, sp_mod._CAKE_SP_CHECK_RTOL) == (1e-2, 1e-2)


# ---------------------------------------------------------------------------
# NVSHMEM symmetric-memory backend (unchanged FlashInfer requirement)
# ---------------------------------------------------------------------------


def _fake_symm_mem(*, backend="CUDA", available=True, set_raises=None):
    state = {"backend": backend}

    def _set(name):
        if set_raises is not None:
            raise set_raises
        state["backend"] = name

    return SimpleNamespace(
        get_backend=lambda device: state["backend"],
        is_nvshmem_available=lambda: available,
        set_backend=mock.Mock(side_effect=_set),
        state=state,
    )


@pytest.mark.parametrize(
    "fake, expected, text",
    [
        (_fake_symm_mem(backend="CUDA"), True, "set to NVSHMEM"),
        (_fake_symm_mem(backend="NVSHMEM"), True, ""),
        (_fake_symm_mem(available=False), False, "unavailable"),
        (
            _fake_symm_mem(set_raises=RuntimeError("already allocated")),
            False,
            "cannot select the NVSHMEM",
        ),
    ],
)
def test_sp_symm_mem_backend_selection(sp_env, caplog, fake, expected, text):
    caplog.set_level(logging.INFO, logger=sp_mod.logger.name)
    sp_mod.reset_cake_sp_state_for_tests()
    with (
        _routes("sp_all_gather_matmul"),
        mock.patch.object(sp_mod, "_symm_mem_module", lambda: fake),
        mock.patch.object(torch.cuda, "current_device", lambda: 0),
    ):
        assert sp_mod.select_cake_sp_symm_mem_backend() is expected
    assert (fake.state["backend"] == "NVSHMEM") is expected
    assert sp_mod._cake_sp_nvshmem_backend is expected
    if expected:
        # torch's fused symm-mem ops cannot allocate under NVSHMEM: stock fused
        # path off for the process, plain gather/scatter + matmul instead.
        assert sp_mod.sp_fused_matmul_eligible(_linear(sp_env)) is False
    sp_mod.reset_cake_sp_state_for_tests()
    assert text in caplog.text


def _symm_mem_module_patch(fake):
    # ``import torch.distributed._symmetric_memory as m`` resolves the leaf
    # through ``getattr(torch.distributed, ...)`` before ``sys.modules``, so
    # both are patched.
    stack = contextlib.ExitStack()
    stack.enter_context(mock.patch.dict(sys.modules, {fake.__name__: fake}))
    stack.enter_context(
        mock.patch.object(torch.distributed, "_symmetric_memory", fake, create=True)
    )
    return stack


def test_adapter_reads_the_nvshmem_backend_and_refuses_others():
    device = torch.device("cpu")
    for backend, expected in (("NVSHMEM", True), ("nvshmem", True), ("CUDA", False)):
        fake = types.ModuleType("torch.distributed._symmetric_memory")
        fake.get_backend = lambda d, backend=backend: backend
        with _symm_mem_module_patch(fake):
            assert comm_mod.symm_mem_backend_is_nvshmem(device) is expected
    broken = types.ModuleType("torch.distributed._symmetric_memory")
    with _symm_mem_module_patch(broken):
        assert comm_mod.symm_mem_backend_is_nvshmem(device) is False


# ---------------------------------------------------------------------------
# adapter admission surface (stride logic is device-independent)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("n", [256, 1280, 2048, 7168, 14336])
def test_adapter_classifies_both_weight_layouts_from_strides(n):
    k_major = torch.empty(n, comm_mod.AG_K, dtype=torch.bfloat16, device="meta").t()
    n_major = torch.empty(comm_mod.AG_K, n, dtype=torch.bfloat16, device="meta")
    assert comm_mod.all_gather_matmul_weight_layout(k_major) == "k_major"
    assert comm_mod.all_gather_matmul_weight_layout(n_major) == "n_major"


def test_adapter_rejects_other_weight_views_without_copying():
    k = comm_mod.AG_K
    cases = [
        torch.empty(k, 512 + 8, device="meta")[:, :512],  # padded N: n-major stride
        torch.empty(512, k + 8, device="meta")[:, :k].t(),  # padded K: k-major stride
        torch.empty(512, k, device="meta"),  # [N, K] not transposed
        torch.empty(k, 512, device="meta").t(),  # [N, K] view of a [K, N] param
        torch.empty(k, device="meta"),  # rank 1
    ]
    for w in cases:
        assert comm_mod.all_gather_matmul_weight_layout(w) is None


def test_adapter_contract_constants_match_flashinfer_main():
    assert comm_mod.AG_K == 8192
    assert comm_mod.AG_BLOCK_N == 256
    assert comm_mod.AG_WORLD_SIZES == (2, 4, 8)
    # The old pin's profile tables are gone: no fixed N, no packed-QKV table.
    assert not hasattr(comm_mod, "AG_N")
    assert not hasattr(comm_mod, "AG_PACKED_QKV_ROUTES")
    assert not hasattr(comm_mod, "AG_BLOCK_M")


def test_adapter_admission_never_raises_on_cpu_tensors():
    inp = torch.empty(5, comm_mod.AG_K, dtype=torch.bfloat16)
    w = torch.empty(1280, comm_mod.AG_K, dtype=torch.bfloat16).t()
    assert comm_mod.supports_all_gather_matmul(inp, w, world_size=TP) is False
    assert (
        comm_mod.supports_prepare_all_gather_matmul(inp, w, world_size=TP, max_rows=8)
        is False
    )


def test_adapter_prepare_capacity_must_cover_the_rows():
    inp = torch.empty(5, comm_mod.AG_K, dtype=torch.bfloat16)
    w = torch.empty(1280, comm_mod.AG_K, dtype=torch.bfloat16).t()
    with mock.patch.object(comm_mod, "supports_all_gather_matmul", return_value=True):
        assert comm_mod.supports_prepare_all_gather_matmul(inp, w, world_size=TP)
        assert comm_mod.supports_prepare_all_gather_matmul(
            inp, w, world_size=TP, max_rows=5
        )
        assert not comm_mod.supports_prepare_all_gather_matmul(
            inp, w, world_size=TP, max_rows=4
        )


def test_adapter_forwards_the_engine_view_and_capacity_to_flashinfer():
    fi = types.ModuleType("flashinfer.comm.all_gather_matmul.all_gather_matmul")
    fi.all_gather_matmul = mock.Mock(return_value="out")
    fi.prepare_all_gather_matmul = mock.Mock(return_value="launcher")
    inp, w, group = object(), object(), object()
    with mock.patch.dict(sys.modules, {fi.__name__: fi}):
        assert comm_mod.all_gather_matmul(inp, w, group) == "out"
        assert comm_mod.prepare_all_gather_matmul(inp, w, group, max_rows=2048) == (
            "launcher"
        )
    fi.all_gather_matmul.assert_called_once_with(
        inp, w, group, backend="cake", verbose=False
    )
    fi.prepare_all_gather_matmul.assert_called_once_with(
        inp, w, group, backend="cake", max_rows=2048, verbose=False
    )


def test_kernel_spec_entry_points_forward_max_rows():
    target = mock.Mock(return_value="launcher")
    inp, w, group = object(), object(), object()
    with mock.patch.object(comm_ops, "_k", lambda name: target):
        assert (
            comm_ops.cake_prepare_all_gather_matmul(inp, w, group, max_rows=512)
            == "launcher"
        )
        assert comm_ops.cake_all_gather_matmul(inp, w, group) == "launcher"
    assert target.call_args_list[0] == mock.call(
        inp, w, group, max_rows=512, verbose=False
    )
    assert target.call_args_list[1] == mock.call(inp, w, group, verbose=False)


def test_route_module_imports_no_flashinfer():
    import importlib

    source = open(importlib.util.find_spec(sp_mod.__name__).origin).read()
    assert "import flashinfer" not in source and "from flashinfer" not in source


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
