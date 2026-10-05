"""CPU unit tests for the ``mamba_ssd_prefill`` / ``mamba_ssu`` Cake routes of
the Mamba2 mixer, plus the route-table / import-hygiene checks shared with the
``sp_all_gather_matmul`` route of the LayerNorm-SP column-parallel participant
(``SGLANG_CAKE_ROUTES``). The SP route's own behaviour (engine weight view,
capacity-bound launcher, refusals, NVSHMEM selection) is covered by
``test_cake_sp_engine_view_routes.py``.

Everything is mocked: the route switch, the adapter admission, the Cake
forwarders and the stock kernels. The tests only check *which* callable
receives the engine's tensors, that the Cake branch is handed the contract
arguments (chunk-128 metadata, the explicit pool state dtype, the plain-int
sequence count, the engine's token-major buffer; the engine's raw BF16 /
int32 storage on the rows whose programs read it and FP32 broadcasts / int64
indices on the others) and that its return contract matches the stock branch.
CPU tensors; no FlashInfer, Triton, CUDA or process group involved.
"""

import contextlib
import importlib
import logging
import subprocess
import sys
import types
from types import SimpleNamespace
from unittest import mock

import pytest
import torch

from sglang.srt.layers import layernorm_sp as sp_mod
from sglang.srt.layers.attention.mamba import cake_routes as mamba_mod
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=10, stage="base-a-test-cpu")

H, G, HEADDIM, DSTATE, POOL = 4, 2, 64, 128, 3
LENGTHS = (96, 160)
S = sum(LENGTHS)


@pytest.fixture(autouse=True)
def _reset_route_state():
    mamba_mod.reset_cake_route_state_for_tests()
    sp_mod.reset_cake_sp_state_for_tests()
    yield
    mamba_mod.reset_cake_route_state_for_tests()
    sp_mod.reset_cake_sp_state_for_tests()


def _routes(module, *enabled):
    return mock.patch.object(module, "cake_route_enabled", lambda name: name in enabled)


def _not_capturing():
    return mock.patch.object(
        torch.cuda, "is_current_stream_capturing", lambda: False, create=True
    )


def _capturing():
    return mock.patch.object(
        torch.cuda, "is_current_stream_capturing", lambda: True, create=True
    )


# ---------------------------------------------------------------------------
# chunk-128 metadata
# ---------------------------------------------------------------------------


def test_cake_chunk_metadata_matches_flashinfer_packed_shape():
    ci, co = mamba_mod.cake_ssd_chunk_metadata(LENGTHS, torch.device("cpu"))
    assert ci.dtype == torch.int32 and co.dtype == torch.int32
    assert ci.tolist() == [0, 0, 1] and co.tolist() == [0, 96, 0]
    ci, co = mamba_mod.cake_ssd_chunk_metadata((128, 128), torch.device("cpu"))
    assert ci.tolist() == [0, 1] and co.tolist() == [0, 0]
    ci, co = mamba_mod.cake_ssd_chunk_metadata((40, 40, 48, 128), torch.device("cpu"))
    assert ci.tolist() == [0, 0, 0, 1] and co.tolist() == [0, 40, 80, 0]


def test_cake_chunk_metadata_exposes_track_boundaries():
    cpu = torch.device("cpu")
    # A boundary inside physical chunk 1 (the oracle case of the kernel test:
    # packed [0, 96) + [96, 256), checkpoint of sequence 1 at absolute 224).
    ci, co = mamba_mod.cake_ssd_chunk_metadata(LENGTHS, cpu, extra_boundaries=(224,))
    assert ci.tolist() == [0, 0, 1, 1] and co.tolist() == [0, 96, 0, 96]
    # On the 128 grid or at a sequence start: nothing added.
    ci, co = mamba_mod.cake_ssd_chunk_metadata(LENGTHS, cpu, extra_boundaries=(128, 96))
    assert ci.tolist() == [0, 0, 1] and co.tolist() == [0, 96, 0]


def test_cake_ssd_track_checkpoints_maps_unaligned_rows_only():
    cpu = torch.device("cpu")
    # rows: 0 tracked, 300 tokens -> chunk 1 -> boundary 256 (on the 128 grid);
    #       1 tracked, 520 tokens starting at 300 -> chunk 2 -> boundary 812;
    #       2 tracked, 512 tokens -> chunk-aligned -> engine slot copy, no checkpoint;
    #       3 not tracked.
    mapped = mamba_mod.cake_ssd_track_checkpoints(
        [True, True, True, False],
        [300, 520, 512, 100],
        [300, 520, 512, 100],
        [0, 0, 0, 0],
        256,
        torch.tensor([5, 6, 7, 8]),
        cpu,
    )
    assert mapped is not None
    assert mapped.boundaries == (256, 812)
    assert mapped.token_indices.dtype == torch.int32
    assert mapped.token_indices.tolist() == [256, 812, -1, -1]
    assert mapped.state_slots.dtype == torch.int32
    assert mapped.state_slots.tolist() == [5, 6, -1, -1]
    # Prefix lengths shift the tracked length, not the packed positions.
    mapped = mamba_mod.cake_ssd_track_checkpoints(
        [True], [1000], [400], [600], 256, torch.tensor([3]), cpu
    )
    assert mapped.boundaries == (256,) and mapped.token_indices.tolist() == [256]
    # A tracked row whose last boundary is its own start (chunk 0) is not
    # expressible as a Cake checkpoint: the batch stays on the stock path.
    assert (
        mamba_mod.cake_ssd_track_checkpoints(
            [True, True], [300, 200], [300, 200], [0, 0], 256, torch.tensor([1, 2]), cpu
        )
        is None
    )
    # Only chunk-aligned (or no) tracked rows: nothing to checkpoint, the
    # route admits without checkpoint arguments (the default-configuration
    # case: the track grid is a multiple of the model chunk size).
    for mask in ([False, False], [True, False]):
        mapped = mamba_mod.cake_ssd_track_checkpoints(
            mask, [512, 160], [512, 160], [0, 0], 256, torch.tensor([1, 2]), cpu
        )
        assert mapped == mamba_mod.CakeTrackCheckpoints((), None, None)


# ---------------------------------------------------------------------------
# SSD prefill
# ---------------------------------------------------------------------------


def _ssd_inputs(seqlen=S, lengths=LENGTHS, state_dtype=torch.bfloat16):
    conv_dim = H * HEADDIM + 2 * G * DSTATE
    hidden_B_C = torch.randn(seqlen, conv_dim).bfloat16()
    x_cols, b_cols, c_cols = torch.split(
        hidden_B_C, [H * HEADDIM, G * DSTATE, G * DSTATE], dim=-1
    )
    cu = [0]
    for n in lengths:
        cu.append(cu[-1] + n)
    seq_idx = torch.repeat_interleave(
        torch.arange(len(lengths), dtype=torch.int32),
        torch.tensor(lengths, dtype=torch.int64),
    ).unsqueeze(0)
    ci, co = mamba_mod.cake_ssd_chunk_metadata(lengths, torch.device("cpu"))
    return dict(
        x=x_cols.view(1, seqlen, H, HEADDIM),  # strided column view like the mixer
        dt=torch.randn(seqlen, H).bfloat16().unsqueeze(0),
        A=-torch.rand(H, dtype=torch.float32) - 1.0,
        B=b_cols.view(1, seqlen, G, DSTATE),
        C=c_cols.view(1, seqlen, G, DSTATE),
        chunk_size=256,
        D=torch.ones(H, dtype=torch.bfloat16),
        dt_bias=torch.ones(H, dtype=torch.bfloat16),
        seq_idx=seq_idx,
        chunk_indices=None,
        chunk_offsets=None,
        cu_seqlens=torch.tensor(cu, dtype=torch.int32),
        initial_states=None,
        track_seq_idx=None,
        track_end_locs=None,
        out=torch.zeros(1, seqlen, H, HEADDIM, dtype=torch.bfloat16),
        state_dtype=state_dtype,
        cake_chunk_indices=ci,
        cake_chunk_offsets=co,
    )


STOCK_SSD_KWARGS = {
    "chunk_size",
    "D",
    "z",
    "dt_bias",
    "seq_idx",
    "chunk_indices",
    "chunk_offsets",
    "cu_seqlens",
    "initial_states",
    "return_varlen_states",
    "return_final_states",
    "return_track_states",
    "track_seq_idx",
    "track_end_locs",
    "dt_softplus",
    "dt_limit",
    "out",
    "state_dtype",
}


def _stock_ssd(x, dt, A, B, C, **kw):
    kw["out"].fill_(1.0)
    return None, torch.full((2, H, HEADDIM, DSTATE), 1.0, dtype=torch.bfloat16), None


def _cake_ssd(x, dt, A, B, C, **kw):
    kw["out"].fill_(2.0)
    final = torch.full((2, H, HEADDIM, DSTATE), 2.0, dtype=torch.bfloat16)
    return kw["out"], final


def test_ssd_route_off_uses_stock_with_exact_kwargs():
    stock = mock.Mock(side_effect=_stock_ssd)
    supports, cake = mock.Mock(return_value=True), mock.Mock(side_effect=_cake_ssd)
    inputs = _ssd_inputs()
    with (
        _routes(mamba_mod),
        mock.patch.object(mamba_mod, "_cake_ssd_kernels", lambda: (supports, cake)),
    ):
        result = mamba_mod.ssd_prefill(stock, **inputs)
    stock.assert_called_once()
    args, kw = stock.call_args
    assert args[0] is inputs["x"] and args[1] is inputs["dt"] and args[2] is inputs["A"]
    assert args[3] is inputs["B"] and args[4] is inputs["C"]
    assert set(kw) == STOCK_SSD_KWARGS
    assert kw["chunk_size"] == 256 and kw["z"] is None and kw["out"] is inputs["out"]
    assert kw["return_varlen_states"] and not kw["return_final_states"]
    assert kw["return_track_states"] and kw["dt_softplus"]
    assert kw["dt_limit"] == (0.0, float("inf"))
    assert kw["initial_states"] is None and kw["chunk_indices"] is None
    supports.assert_not_called()
    cake.assert_not_called()
    assert result[0] is None and torch.all(result[1] == 1.0) and result[2] is None
    assert torch.all(inputs["out"] == 1.0)


def test_ssd_route_on_admitted_uses_cake_with_contract_args(caplog):
    caplog.set_level(logging.INFO, logger=mamba_mod.logger.name)
    stock = mock.Mock(side_effect=_stock_ssd)
    supports, cake = mock.Mock(return_value=True), mock.Mock(side_effect=_cake_ssd)
    inputs = _ssd_inputs()
    with (
        _routes(mamba_mod, "mamba_ssd_prefill"),
        mock.patch.object(mamba_mod, "_cake_ssd_kernels", lambda: (supports, cake)),
    ):
        result = mamba_mod.ssd_prefill(stock, **inputs)
    stock.assert_not_called()
    cake.assert_called_once()
    args, kw = cake.call_args
    assert args[0] is inputs["x"] and args[1] is inputs["dt"] and args[2] is inputs["A"]
    assert args[3] is inputs["B"] and args[4] is inputs["C"]
    assert kw["D"] is inputs["D"] and kw["dt_bias"] is inputs["dt_bias"]
    assert kw["z"] is None and kw["dt_softplus"] is True
    assert kw["dt_limit"] == (0.0, float("inf")) and kw["return_final_states"]
    assert kw["seq_idx"] is inputs["seq_idx"]
    # Chunk-128 metadata, not the engine's chunk-256 metadata.
    assert kw["chunk_indices"] is inputs["cake_chunk_indices"]
    assert kw["chunk_offsets"] is inputs["cake_chunk_offsets"]
    # Varlen without a prefix: ``None`` initial states plus the sequence count
    # (no zero buffer); the engine's token-major buffer is the kernel's out;
    # the engine's pool dtype is the kernel's state dtype (explicit).
    assert kw["initial_states"] is None and kw["num_seqs"] == 2
    assert kw["out"] is inputs["out"]
    assert kw["state_dtype"] is torch.bfloat16
    # Admission saw the same tensors at chunk 128.
    s_args, s_kw = supports.call_args
    assert s_args[0] is inputs["x"] and s_args[3] is inputs["B"]
    assert s_kw["chunk_size"] == 128 and s_kw["out"] is inputs["out"]
    assert s_kw["initial_states"] is None and s_kw["num_seqs"] == 2
    assert s_kw["state_dtype"] is torch.bfloat16
    assert s_kw["chunk_indices"] is inputs["cake_chunk_indices"]
    assert s_kw["seq_idx"] is inputs["seq_idx"] and s_kw["z"] is None
    # Stock return contract: (None, varlen_state, None) and the engine out filled.
    assert result[0] is None and result[2] is None
    assert tuple(result[1].shape) == (2, H, HEADDIM, DSTATE) and torch.all(
        result[1] == 2.0
    )
    assert torch.all(inputs["out"] == 2.0)
    assert "[cake-route] mamba_ssd_prefill: Cake kernel selected" in caplog.text


def _tracked_inputs(**overrides):
    inputs = _ssd_inputs()
    inputs["track_seq_idx"] = torch.zeros(0, dtype=torch.int64)
    inputs["track_end_locs"] = torch.zeros(0, dtype=torch.int64)
    inputs["cake_track_checkpoints"] = mamba_mod.CakeTrackCheckpoints(
        (224,),
        torch.tensor([-1, 224], dtype=torch.int32),
        torch.tensor([-1, 2], dtype=torch.int32),
    )
    inputs["track_states_out"] = torch.zeros(POOL, H, HEADDIM, DSTATE).bfloat16()
    inputs.update(overrides)
    return inputs


def test_ssd_route_tracked_batch_passes_checkpoints_and_returns_sentinel():
    stock = mock.Mock(side_effect=_stock_ssd)
    supports, cake = mock.Mock(return_value=True), mock.Mock(side_effect=_cake_ssd)
    inputs = _tracked_inputs()
    with (
        _routes(mamba_mod, "mamba_ssd_prefill"),
        mock.patch.object(mamba_mod, "_cake_ssd_kernels", lambda: (supports, cake)),
    ):
        result = mamba_mod.ssd_prefill(stock, **inputs)
    stock.assert_not_called()
    ckpt = inputs["cake_track_checkpoints"]
    for call in (supports.call_args, cake.call_args):
        kw = call.kwargs
        assert kw["checkpoint_token_indices"] is ckpt.token_indices
        assert kw["checkpoint_state_slots"] is ckpt.state_slots
        # Written straight into the layer's state pool (the track slots).
        assert kw["checkpoint_states"] is inputs["track_states_out"]
    assert result[0] is None and torch.all(result[1] == 2.0)
    assert result[2] is mamba_mod.SSD_TRACK_STATES_IN_PLACE
    # Tracked batch with only chunk-aligned rows: admitted without checkpoint
    # arguments (no pool needed), still the sentinel for the backend.
    cake.reset_mock()
    inputs = _tracked_inputs(
        cake_track_checkpoints=mamba_mod.CakeTrackCheckpoints((), None, None),
        track_states_out=None,
    )
    with (
        _routes(mamba_mod, "mamba_ssd_prefill"),
        mock.patch.object(mamba_mod, "_cake_ssd_kernels", lambda: (supports, cake)),
    ):
        result = mamba_mod.ssd_prefill(stock, **inputs)
    assert "checkpoint_states" not in cake.call_args.kwargs
    assert result[2] is mamba_mod.SSD_TRACK_STATES_IN_PLACE
    # An untracked batch never carries checkpoints.
    cake.reset_mock()
    with (
        _routes(mamba_mod, "mamba_ssd_prefill"),
        mock.patch.object(mamba_mod, "_cake_ssd_kernels", lambda: (supports, cake)),
    ):
        result = mamba_mod.ssd_prefill(stock, **_ssd_inputs())
    assert "checkpoint_states" not in cake.call_args.kwargs
    assert result[2] is None


@pytest.mark.parametrize("case", ["no_mapping", "no_pool", "strided_pool", "fp32_pool"])
def test_ssd_route_tracked_batch_without_usable_checkpoints_falls_back(case, caplog):
    caplog.set_level(logging.INFO, logger=mamba_mod.logger.name)
    stock = mock.Mock(side_effect=_stock_ssd)
    supports, cake = mock.Mock(return_value=True), mock.Mock(side_effect=_cake_ssd)
    if case == "no_mapping":
        inputs = _tracked_inputs(cake_track_checkpoints=None)
    elif case == "no_pool":
        inputs = _tracked_inputs(track_states_out=None)
    elif case == "strided_pool":
        inputs = _tracked_inputs(
            track_states_out=torch.zeros(POOL, H, HEADDIM, 2 * DSTATE).bfloat16()[
                ..., ::2
            ]
        )
    else:
        inputs = _tracked_inputs(
            track_states_out=torch.zeros(POOL, H, HEADDIM, DSTATE, dtype=torch.float32)
        )
    with (
        _routes(mamba_mod, "mamba_ssd_prefill"),
        mock.patch.object(mamba_mod, "_cake_ssd_kernels", lambda: (supports, cake)),
    ):
        result = mamba_mod.ssd_prefill(stock, **inputs)
    stock.assert_called_once()
    supports.assert_not_called()
    cake.assert_not_called()
    assert result[2] is None
    assert "[cake-route] mamba_ssd_prefill: fallback" in caplog.text


def test_backend_skips_unaligned_rows_the_kernel_already_wrote():
    from sglang.srt.layers.attention.hybrid_linear_attn_backend import (
        Mamba2AttnBackend,
    )

    be = Mamba2AttnBackend.__new__(Mamba2AttnBackend)
    md = SimpleNamespace(
        has_mamba_track_mask=True,
        track_ssm_h_src=torch.tensor([0]),
        track_ssm_h_dst=torch.tensor([3]),
        track_ssm_h_batch_src=torch.tensor([0]),
        track_ssm_recompute_dst=torch.tensor([4]),
        track_ssm_final_src=torch.tensor([1]),
        track_ssm_final_dst=torch.tensor([5]),
    )
    pool = torch.arange(6, dtype=torch.float32).view(6, 1, 1, 1).expand(6, H, 2, 2)
    pool = pool.contiguous().bfloat16()
    before = pool.clone()
    be._track_mamba_state_extend(
        None, None, pool, md, track_states=None, unaligned_rows_written=True
    )
    # Only the chunk-aligned row moved (slot 1 -> 5); 3 and 4 were the kernel's.
    assert torch.equal(pool[5], before[1])
    assert torch.equal(pool[3], before[3]) and torch.equal(pool[4], before[4])
    # Without the flag the stock contract still needs the chunk grid.
    with pytest.raises(AssertionError):
        be._track_mamba_state_extend(None, None, pool, md, track_states=None)


def test_ssd_route_passes_engine_initial_states_through():
    stock = mock.Mock(side_effect=_stock_ssd)
    supports, cake = mock.Mock(return_value=True), mock.Mock(side_effect=_cake_ssd)
    inputs = _ssd_inputs()
    inputs["initial_states"] = torch.randn(2, H, HEADDIM, DSTATE).bfloat16()
    with (
        _routes(mamba_mod, "mamba_ssd_prefill"),
        mock.patch.object(mamba_mod, "_cake_ssd_kernels", lambda: (supports, cake)),
    ):
        mamba_mod.ssd_prefill(stock, **inputs)
    assert cake.call_args.kwargs["initial_states"] is inputs["initial_states"]
    assert cake.call_args.kwargs["num_seqs"] == 2


def test_ssd_route_on_rejected_falls_back_and_logs_once(caplog):
    caplog.set_level(logging.INFO, logger=mamba_mod.logger.name)
    stock = mock.Mock(side_effect=_stock_ssd)
    supports, cake = mock.Mock(return_value=False), mock.Mock(side_effect=_cake_ssd)
    inputs = _ssd_inputs()
    with (
        _routes(mamba_mod, "mamba_ssd_prefill"),
        mock.patch.object(mamba_mod, "_cake_ssd_kernels", lambda: (supports, cake)),
    ):
        mamba_mod.ssd_prefill(stock, **inputs)
        mamba_mod.ssd_prefill(stock, **inputs)
    assert stock.call_count == 2 and supports.call_count == 2
    cake.assert_not_called()
    assert caplog.text.count("[cake-route] mamba_ssd_prefill: fallback") == 1
    assert "adapter admission rejected" in caplog.text


@pytest.mark.parametrize(
    "lengths",
    [(128,), (128, 128), (72, 128), (96, 160), (256, 128), (128, 896), (1000,)],
)
def test_ssd_route_single_chunk_and_partial_chunk_batches_are_routed(lengths):
    """Every packed geometry is routed: single-chunk sequences (the CAKE-950
    NaN geometry, fixed in the kernel) and token counts off the 128 grid (the
    kernel handles a partial last chunk); the Cake call names the sequence
    count and writes the engine buffer."""
    stock = mock.Mock(side_effect=_stock_ssd)
    supports, cake = mock.Mock(return_value=True), mock.Mock(side_effect=_cake_ssd)
    inputs = _ssd_inputs(seqlen=sum(lengths), lengths=lengths)
    with (
        _routes(mamba_mod, "mamba_ssd_prefill"),
        mock.patch.object(mamba_mod, "_cake_ssd_kernels", lambda: (supports, cake)),
    ):
        mamba_mod.ssd_prefill(stock, **inputs)
    cake.assert_called_once()
    stock.assert_not_called()
    assert cake.call_args.kwargs["num_seqs"] == len(lengths)
    assert cake.call_args.kwargs["out"] is inputs["out"]


@pytest.mark.parametrize("state_dtype", [torch.float32, torch.float16])
def test_ssd_route_names_the_engine_pool_state_dtype(state_dtype):
    """``--mamba-ssm-dtype float32`` (the engine default) and ``float16`` are
    routed with the pool dtype named explicitly, also for a prefix-less batch
    (``initial_states=None``), so the final states scattered back into the
    pool are computed in the pool dtype rather than FlashInfer's BF16
    inference."""
    stock = mock.Mock(side_effect=_stock_ssd)
    supports, cake = mock.Mock(return_value=True), mock.Mock(side_effect=_cake_ssd)
    inputs = _ssd_inputs(state_dtype=state_dtype)
    with (
        _routes(mamba_mod, "mamba_ssd_prefill"),
        mock.patch.object(mamba_mod, "_cake_ssd_kernels", lambda: (supports, cake)),
    ):
        mamba_mod.ssd_prefill(stock, **inputs)
    cake.assert_called_once()
    stock.assert_not_called()
    assert cake.call_args.kwargs["initial_states"] is None
    assert cake.call_args.kwargs["state_dtype"] is state_dtype
    assert supports.call_args.kwargs["state_dtype"] is state_dtype


def _fake_flashinfer_mamba():
    """A stand-in ``flashinfer.mamba`` recording which entry the adapter uses."""
    module = types.ModuleType("flashinfer.mamba")
    runners = []

    class SSDCombined:
        def __init__(self, *args, **kwargs):
            self.args, self.kwargs = args, kwargs
            self.run = mock.Mock(
                side_effect=lambda *a, **kw: (kw["out"], torch.zeros(1))
            )
            runners.append(self)

    module.SSDCombined = SSDCombined
    module.ssd_combined_fwd = mock.Mock(
        side_effect=lambda *a, **kw: (kw["out"], torch.zeros(1))
    )
    return module, runners


def test_adapter_explicit_state_dtype_uses_a_prepared_runner_once_per_config():
    """``state_dtype=`` cannot be expressed through FlashInfer's functional
    entry for a prefix-less call (it infers BF16), so the adapter serves it
    through ``SSDCombined(..., backend="cake", state_dtype=...)`` cached per
    device / stream / configuration; ``state_dtype=None`` stays functional."""
    from sglang.kernels.cake_kernels import mamba as cake_mamba

    cake_mamba._cached_ssd_runner.cache_clear()
    module, runners = _fake_flashinfer_mamba()
    inputs = _ssd_inputs(state_dtype=torch.float32)
    call = dict(
        D=inputs["D"],
        dt_bias=inputs["dt_bias"],
        dt_softplus=True,
        seq_idx=inputs["seq_idx"],
        chunk_indices=inputs["cake_chunk_indices"],
        chunk_offsets=inputs["cake_chunk_offsets"],
        num_seqs=2,
        out=inputs["out"],
    )
    positional = (inputs["x"], inputs["dt"], inputs["A"], inputs["B"], inputs["C"])
    with (
        mock.patch.dict(
            sys.modules,
            {
                "flashinfer": sys.modules.get(
                    "flashinfer", types.ModuleType("flashinfer")
                ),
                "flashinfer.mamba": module,
            },
        ),
        mock.patch.object(torch.cuda, "current_device", lambda: 0),
        mock.patch.object(
            torch.cuda, "current_stream", lambda *_: SimpleNamespace(cuda_stream=7)
        ),
        mock.patch.object(torch.cuda, "device", lambda *_: contextlib.nullcontext()),
    ):
        out, final = cake_mamba.ssd_combined_fwd(
            *positional, state_dtype=torch.float32, **call
        )
        cake_mamba.ssd_combined_fwd(*positional, state_dtype=torch.float32, **call)
        cake_mamba.ssd_combined_fwd(*positional, **call)
    try:
        assert out is inputs["out"] and final is not None
        # One prepared runner for the configuration, reused by the second call.
        assert len(runners) == 1
        runner = runners[0]
        assert runner.args == (128, H, HEADDIM, DSTATE, G)
        assert runner.kwargs["backend"] == "cake"
        assert runner.kwargs["state_dtype"] is torch.float32
        assert runner.kwargs["io_dtype"] is torch.bfloat16
        assert runner.kwargs["has_d"] and not runner.kwargs["d_has_hdim"]
        assert runner.kwargs["has_varlen"] and not runner.kwargs["has_initial_states"]
        assert not runner.kwargs["has_z"]
        assert runner.kwargs["seq_idx_dtype"] is torch.int32
        assert runner.run.call_count == 2
        run_kw = runner.run.call_args.kwargs
        assert run_kw["num_seqs"] == 2 and run_kw["out"] is inputs["out"]
        assert run_kw["initial_states"] is None and run_kw["dt_softplus"] is True
        assert run_kw["chunk_indices"] is inputs["cake_chunk_indices"]
        assert "state_dtype" not in run_kw  # a constructor argument, not a run one
        # Without an explicit dtype the functional entry is used unchanged.
        module.ssd_combined_fwd.assert_called_once()
        fn_kw = module.ssd_combined_fwd.call_args.kwargs
        assert fn_kw["num_seqs"] == 2 and fn_kw["out"] is inputs["out"]
        assert "state_dtype" not in fn_kw
    finally:
        cake_mamba._cached_ssd_runner.cache_clear()


@pytest.mark.parametrize("case", ["tracking", "no_metadata", "fp64_state"])
def test_ssd_route_static_fallbacks_skip_adapter(case):
    stock = mock.Mock(side_effect=_stock_ssd)
    supports, cake = mock.Mock(return_value=True), mock.Mock(side_effect=_cake_ssd)
    inputs = _ssd_inputs()
    if case == "tracking":
        # tracked batch whose rows were not mapped onto Cake checkpoints
        inputs["track_seq_idx"] = torch.zeros(1, S, dtype=torch.int32)
        inputs["track_end_locs"] = torch.tensor([96], dtype=torch.int32)
    elif case == "no_metadata":
        inputs["cake_chunk_indices"] = inputs["cake_chunk_offsets"] = None
    elif case == "fp64_state":
        inputs["state_dtype"] = torch.float64
    with (
        _routes(mamba_mod, "mamba_ssd_prefill"),
        mock.patch.object(mamba_mod, "_cake_ssd_kernels", lambda: (supports, cake)),
    ):
        mamba_mod.ssd_prefill(stock, **inputs)
    stock.assert_called_once()
    supports.assert_not_called()
    cake.assert_not_called()
    assert stock.call_args.kwargs["out"] is inputs["out"]


def test_ssd_flashinfer_refusal_falls_back_and_is_cached():
    stock = mock.Mock(side_effect=_stock_ssd)
    supports = mock.Mock(return_value=True)
    cake = mock.Mock(side_effect=NotImplementedError("no manifest row"))
    inputs = _ssd_inputs()
    with (
        _routes(mamba_mod, "mamba_ssd_prefill"),
        mock.patch.object(mamba_mod, "_cake_ssd_kernels", lambda: (supports, cake)),
    ):
        result = mamba_mod.ssd_prefill(stock, **inputs)
        mamba_mod.ssd_prefill(stock, **inputs)
    assert stock.call_count == 2 and cake.call_count == 1
    assert torch.all(result[1] == 1.0) and torch.all(inputs["out"] == 1.0)


# ---------------------------------------------------------------------------
# selective state update (decode / target verify)
# ---------------------------------------------------------------------------


def _ssu_verify_inputs(batch=2, steps=6, state_dtype=torch.bfloat16):
    """Mirror the mixer's target-verify call: BF16 broadcasts, int32 indices."""
    tokens = batch * steps
    dt_base = torch.rand(tokens, H).bfloat16()
    dt = dt_base[:, :, None].expand(-1, -1, HEADDIM).view(batch, steps, H, HEADDIM)
    A = (
        (-torch.rand(H, dtype=torch.float32) - 1.0)[:, None, None]
        .expand(-1, HEADDIM, DSTATE)
        .to(dtype=torch.float32)
    )
    D_param = torch.ones(H, dtype=torch.bfloat16)
    dt_bias_param = torch.full((H,), 0.5, dtype=torch.bfloat16)
    return dict(
        state=torch.zeros(POOL, H, HEADDIM, DSTATE, dtype=state_dtype),
        x=torch.randn(batch, steps, H, HEADDIM).bfloat16(),
        dt=dt,
        A=A,
        B=torch.randn(batch, steps, 1, DSTATE).bfloat16(),
        C=torch.randn(batch, steps, 1, DSTATE).bfloat16(),
        D=D_param[:, None].expand(-1, HEADDIM),
        kwargs=dict(
            z=None,
            dt_bias=dt_bias_param[:, None].expand(-1, HEADDIM),
            dt_softplus=True,
            state_batch_indices=(torch.arange(batch) % (POOL - 1) + 1).to(torch.int32),
            out=torch.zeros(batch, steps, H, HEADDIM, dtype=torch.bfloat16),
            disable_state_update=True,
            intermediate_states_buffer=torch.zeros(
                POOL, steps, H, HEADDIM, DSTATE, dtype=state_dtype
            ),
            cache_steps=steps,
            retrieve_parent_token=None,
            intermediate_state_indices=torch.arange(batch, dtype=torch.int32),
        ),
    )


def _ssu_decode_inputs(batch=2):
    """Mirror the mixer's plain decode call for a headdim-64 model."""
    dt = torch.rand(batch, H).bfloat16()[:, :, None].expand(-1, -1, HEADDIM)
    A = (
        (-torch.rand(H, dtype=torch.float32) - 1.0)[:, None, None]
        .expand(-1, HEADDIM, DSTATE)
        .to(dtype=torch.float32)
    )
    return dict(
        state=torch.zeros(POOL, H, HEADDIM, DSTATE, dtype=torch.bfloat16),
        x=torch.randn(batch, H, HEADDIM).bfloat16(),
        dt=dt,
        A=A,
        B=torch.randn(batch, 1, DSTATE).bfloat16(),
        C=torch.randn(batch, 1, DSTATE).bfloat16(),
        D=torch.ones(H, dtype=torch.bfloat16)[:, None].expand(-1, HEADDIM),
        kwargs=dict(
            z=None,
            dt_bias=torch.ones(H, dtype=torch.bfloat16)[:, None].expand(-1, HEADDIM),
            dt_softplus=True,
            state_batch_indices=torch.tensor([1, 2], dtype=torch.int32)[:batch],
            out=torch.zeros(batch, H, HEADDIM, dtype=torch.bfloat16),
        ),
    )


def _call_ssu(stock, inputs):
    return mamba_mod.selective_state_update(
        stock,
        inputs["state"],
        inputs["x"],
        inputs["dt"],
        inputs["A"],
        inputs["B"],
        inputs["C"],
        inputs["D"],
        **inputs["kwargs"],
    )


def _stock_ssu(state, x, dt, A, B, C, D, **kw):
    kw["out"].fill_(1.0)


def _cake_ssu(state, x, dt, A, B, C, D, **kw):
    kw["out"].fill_(2.0)
    return kw["out"]


def test_ssu_route_off_uses_stock_with_exact_kwargs():
    stock = mock.Mock(side_effect=_stock_ssu)
    supports, cake = mock.Mock(return_value=True), mock.Mock(side_effect=_cake_ssu)
    inputs = _ssu_verify_inputs()
    with (
        _routes(mamba_mod),
        _not_capturing(),
        mock.patch.object(mamba_mod, "_cake_ssu_kernels", lambda: (supports, cake)),
    ):
        _call_ssu(stock, inputs)
    stock.assert_called_once()
    args, kw = stock.call_args
    assert args[0] is inputs["state"] and args[1] is inputs["x"]
    assert args[2] is inputs["dt"] and args[6] is inputs["D"]
    assert set(kw) == set(inputs["kwargs"])
    assert all(kw[k] is inputs["kwargs"][k] for k in inputs["kwargs"])
    supports.assert_not_called()
    cake.assert_not_called()
    assert torch.all(inputs["kwargs"]["out"] == 1.0)


def test_ssu_verify_small_batch_passes_the_engine_storage_without_copies(caplog):
    """Below batch 32 the six-token cache row runs the ``mtp_cache_c4_t6``
    program, which reads the engine's BF16 coefficient broadcasts, int32 slot
    tables and projection views as they are: no per-call conversion."""
    caplog.set_level(logging.INFO, logger=mamba_mod.logger.name)
    stock = mock.Mock(side_effect=_stock_ssu)
    supports, cake = mock.Mock(return_value=True), mock.Mock(side_effect=_cake_ssu)
    inputs = _ssu_verify_inputs()
    with (
        _routes(mamba_mod, "mamba_ssu"),
        _not_capturing(),
        mock.patch.object(mamba_mod, "_cake_ssu_kernels", lambda: (supports, cake)),
    ):
        _call_ssu(stock, inputs)
        _call_ssu(stock, inputs)
    stock.assert_not_called()
    assert cake.call_count == 2
    assert supports.call_count == 1  # admission memoised per shape key
    args, kw = cake.call_args
    state, x, dt, A, B, C, D = args
    assert state is inputs["state"] and x is inputs["x"] and A is inputs["A"]
    assert B is inputs["B"] and C is inputs["C"]
    assert dt is inputs["dt"] and D is inputs["D"]
    assert kw["dt_bias"] is inputs["kwargs"]["dt_bias"]
    assert kw["state_batch_indices"] is inputs["kwargs"]["state_batch_indices"]
    assert kw["state_batch_indices"].dtype == torch.int32
    assert (
        kw["intermediate_state_indices"]
        is inputs["kwargs"]["intermediate_state_indices"]
    )
    assert kw["intermediate_state_indices"].dtype == torch.int32
    assert kw["out"] is inputs["kwargs"]["out"]
    assert kw["disable_state_update"] is True and kw["dt_softplus"] is True
    assert (
        kw["intermediate_states_buffer"]
        is (inputs["kwargs"]["intermediate_states_buffer"])
    )
    assert kw["cache_steps"] == 6 and kw["algorithm"] == "auto" and kw["z"] is None
    # Admission saw the engine's tensors.
    s_args, s_kw = supports.call_args
    assert s_args[0] is state and s_args[2] is inputs["dt"]
    assert s_kw["state_batch_indices"].dtype == torch.int32
    assert s_kw["algorithm"] == "auto" and s_kw["cache_steps"] == 6
    assert torch.all(inputs["kwargs"]["out"] == 2.0)
    assert "[cake-route] mamba_ssu: Cake kernel selected" in caplog.text


def test_ssu_verify_large_batch_converts_to_the_canonical_abi_and_horizontal():
    """From batch 32 on only the canonical-ABI ``mtp_horizontal`` program
    serves the cache row: FP32 per-head broadcasts (stride-0 trailing axes
    preserved), int64 tables, ``algorithm="horizontal"``."""
    stock = mock.Mock(side_effect=_stock_ssu)
    supports, cake = mock.Mock(return_value=True), mock.Mock(side_effect=_cake_ssu)
    inputs = _ssu_verify_inputs(batch=32)
    with (
        _routes(mamba_mod, "mamba_ssu"),
        _not_capturing(),
        mock.patch.object(mamba_mod, "_cake_ssu_kernels", lambda: (supports, cake)),
    ):
        _call_ssu(stock, inputs)
    stock.assert_not_called()
    args, kw = cake.call_args
    state, x, dt, A, B, C, D = args
    assert state is inputs["state"] and x is inputs["x"] and A is inputs["A"]
    assert dt.dtype == torch.float32 and dt.stride(-1) == 0
    assert tuple(dt.shape) == tuple(inputs["dt"].shape)
    assert torch.equal(dt, inputs["dt"].to(torch.float32))
    assert D.dtype == torch.float32 and D.stride(1) == 0 and torch.all(D == 1.0)
    assert kw["dt_bias"].dtype == torch.float32 and kw["dt_bias"].stride(1) == 0
    assert torch.all(kw["dt_bias"] == 0.5)
    assert kw["state_batch_indices"].dtype == torch.int64
    assert torch.equal(
        kw["state_batch_indices"],
        inputs["kwargs"]["state_batch_indices"].to(torch.int64),
    )
    assert kw["intermediate_state_indices"].dtype == torch.int64
    assert kw["intermediate_state_indices"].tolist() == list(range(32))
    assert kw["algorithm"] == "horizontal"
    s_args, s_kw = supports.call_args
    assert s_args[2].dtype == torch.float32
    assert s_kw["state_batch_indices"].dtype == torch.int64
    assert s_kw["algorithm"] == "horizontal"


def test_ssu_route_on_rejected_falls_back_and_logs_once(caplog):
    caplog.set_level(logging.INFO, logger=mamba_mod.logger.name)
    stock = mock.Mock(side_effect=_stock_ssu)
    supports, cake = mock.Mock(return_value=False), mock.Mock(side_effect=_cake_ssu)
    inputs = _ssu_verify_inputs()
    with (
        _routes(mamba_mod, "mamba_ssu"),
        _not_capturing(),
        mock.patch.object(mamba_mod, "_cake_ssu_kernels", lambda: (supports, cake)),
    ):
        _call_ssu(stock, inputs)
        _call_ssu(stock, inputs)
    assert stock.call_count == 2
    cake.assert_not_called()
    assert caplog.text.count("[cake-route] mamba_ssu: fallback") == 1
    assert "adapter admission rejected" in caplog.text


def test_ssu_capture_without_warmup_falls_back_then_replays_after_warmup():
    stock = mock.Mock(side_effect=_stock_ssu)
    supports, cake = mock.Mock(return_value=True), mock.Mock(side_effect=_cake_ssu)
    inputs = _ssu_verify_inputs()
    with (
        _routes(mamba_mod, "mamba_ssu"),
        mock.patch.object(mamba_mod, "_cake_ssu_kernels", lambda: (supports, cake)),
    ):
        with _capturing():
            _call_ssu(stock, inputs)  # shape first seen inside capture -> stock
        stock.assert_called_once()
        cake.assert_not_called()
        with _not_capturing():
            _call_ssu(stock, inputs)  # eager warm-up admits and runs Cake
        with _capturing():
            _call_ssu(stock, inputs)  # warmed shape is routed under capture
    assert stock.call_count == 1 and cake.call_count == 2


def test_ssu_headdim64_decode_row_passes_the_engine_storage_without_copies(caplog):
    """The headdim-64 T=1 row runs on Cake with the engine's BF16 coefficient
    broadcasts, int32 slot table and fused-projection views passed as they are."""
    caplog.set_level(logging.INFO, logger=mamba_mod.logger.name)
    stock = mock.Mock(side_effect=_stock_ssu)
    supports, cake = mock.Mock(return_value=True), mock.Mock(side_effect=_cake_ssu)
    inputs = _ssu_decode_inputs()
    with (
        _routes(mamba_mod, "mamba_ssu"),
        _not_capturing(),
        mock.patch.object(mamba_mod, "_cake_ssu_kernels", lambda: (supports, cake)),
    ):
        _call_ssu(stock, inputs)
    stock.assert_not_called()
    cake.assert_called_once()
    args, kw = cake.call_args
    state, x, dt, A, B, C, D = args
    assert state is inputs["state"] and x is inputs["x"] and A is inputs["A"]
    assert dt is inputs["dt"] and D is inputs["D"]
    assert kw["dt_bias"] is inputs["kwargs"]["dt_bias"]
    assert kw["state_batch_indices"] is inputs["kwargs"]["state_batch_indices"]
    assert kw["state_batch_indices"].dtype == torch.int32
    assert kw["out"] is inputs["kwargs"]["out"]
    assert kw["cache_steps"] == 0 and kw["algorithm"] == "auto" and kw["z"] is None
    assert kw["dt_softplus"] is True and kw["disable_state_update"] is False
    assert kw["pad_slot_id"] == -1  # the engine's padding slot value is forwarded
    s_args, s_kw = supports.call_args
    assert s_kw["pad_slot_id"] == -1
    assert (
        s_args[2] is inputs["dt"] and s_kw["state_batch_indices"].dtype == torch.int32
    )
    assert torch.all(inputs["kwargs"]["out"] == 2.0)
    assert "[cake-route] mamba_ssu: Cake kernel selected" in caplog.text


def test_ssu_decode_forwards_the_engine_pad_slot_id():
    stock = mock.Mock(side_effect=_stock_ssu)
    supports, cake = mock.Mock(return_value=True), mock.Mock(side_effect=_cake_ssu)
    inputs = _ssu_decode_inputs()
    inputs["kwargs"]["pad_slot_id"] = -7
    with (
        _routes(mamba_mod, "mamba_ssu"),
        _not_capturing(),
        mock.patch.object(mamba_mod, "_cake_ssu_kernels", lambda: (supports, cake)),
    ):
        _call_ssu(stock, inputs)
    assert cake.call_args.kwargs["pad_slot_id"] == -7
    assert supports.call_args.kwargs["pad_slot_id"] == -7


def test_ssu_static_row_admits_both_decode_tiles_only():
    state = torch.zeros(POOL, H, HEADDIM, DSTATE, dtype=torch.bfloat16)
    x = torch.zeros(2, H, HEADDIM, dtype=torch.bfloat16)
    assert mamba_mod._ssu_static_row(state, x, False, False) is None
    assert mamba_mod._ssu_static_row(state.float(), x, False, False) is None
    wide = torch.zeros(POOL, H, 128, 128, dtype=torch.bfloat16)
    assert (
        mamba_mod._ssu_static_row(
            wide, torch.zeros(2, H, 128, dtype=torch.bfloat16), False, False
        )
        is None
    )
    narrow = torch.zeros(POOL, H, 64, 64, dtype=torch.bfloat16)
    reason = mamba_mod._ssu_static_row(narrow, x, False, False)
    assert reason == "no promoted T=1 row for (dim, dstate)=(64, 64)"


def test_ssu_decode_with_fp32_coefficients_takes_the_canonical_conversion():
    """A headdim-64 decode call whose coefficients are not the engine's BF16
    storage goes through the FP32 / int64 conversions like the other rows."""
    stock = mock.Mock(side_effect=_stock_ssu)
    supports, cake = mock.Mock(return_value=True), mock.Mock(side_effect=_cake_ssu)
    inputs = _ssu_decode_inputs()
    inputs["dt"] = inputs["dt"].float()
    with (
        _routes(mamba_mod, "mamba_ssu"),
        _not_capturing(),
        mock.patch.object(mamba_mod, "_cake_ssu_kernels", lambda: (supports, cake)),
    ):
        _call_ssu(stock, inputs)
    args, kw = cake.call_args
    assert args[2].dtype == torch.float32 and args[6].dtype == torch.float32
    assert kw["dt_bias"].dtype == torch.float32
    assert kw["state_batch_indices"].dtype == torch.int64


def test_ssu_tree_verify_falls_back():
    stock = mock.Mock(side_effect=_stock_ssu)
    supports, cake = mock.Mock(return_value=True), mock.Mock(side_effect=_cake_ssu)
    inputs = _ssu_verify_inputs()
    inputs["kwargs"]["retrieve_parent_token"] = torch.zeros(2, 6, dtype=torch.int32)
    with (
        _routes(mamba_mod, "mamba_ssu"),
        _not_capturing(),
        mock.patch.object(mamba_mod, "_cake_ssu_kernels", lambda: (supports, cake)),
    ):
        _call_ssu(stock, inputs)
    stock.assert_called_once()
    supports.assert_not_called()
    cake.assert_not_called()


def test_ssu_flashinfer_refusal_falls_back_and_is_cached():
    stock = mock.Mock(side_effect=_stock_ssu)
    supports = mock.Mock(return_value=True)
    cake = mock.Mock(side_effect=NotImplementedError("no manifest row"))
    inputs = _ssu_verify_inputs()
    with (
        _routes(mamba_mod, "mamba_ssu"),
        _not_capturing(),
        mock.patch.object(mamba_mod, "_cake_ssu_kernels", lambda: (supports, cake)),
    ):
        _call_ssu(stock, inputs)
        _call_ssu(stock, inputs)
    assert stock.call_count == 2 and cake.call_count == 1


# ---------------------------------------------------------------------------
# sequence-parallel all-gather + matmul
# ---------------------------------------------------------------------------

TP = 8
K, N = 32, 16
ROWS = 4  # local shard rows (M_pad / tp)
NUM_TOKENS = TP * ROWS - 3  # real tokens: the exit narrow drops the padding


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


# ---------------------------------------------------------------------------
# route table / import hygiene
# ---------------------------------------------------------------------------


def test_route_names_exist_in_route_table():
    from sglang.kernels.cake_kernels._routes import ROUTES

    assert mamba_mod.CAKE_ROUTE_SSD_PREFILL in ROUTES
    assert mamba_mod.CAKE_ROUTE_SSU in ROUTES
    assert sp_mod.CAKE_ROUTE_SP_ALL_GATHER_MATMUL in ROUTES


def test_route_modules_import_no_flashinfer():
    for name in (
        "sglang.srt.layers.attention.mamba.cake_routes",
        "sglang.srt.layers.layernorm_sp",
    ):
        source = open(importlib.util.find_spec(name).origin).read()
        assert "import flashinfer" not in source and "from flashinfer" not in source
    # Fresh interpreter: other tests in the session legitimately import
    # FlashInfer-backed sglang modules, so the check must not share sys.modules.
    code = (
        "import sys; "
        "import sglang.srt.layers.attention.mamba.cake_routes; "
        "import sglang.srt.layers.layernorm_sp; "
        "bad = [n for n in ('flashinfer.mamba', 'flashinfer.comm') if n in sys.modules]; "
        "assert not bad, bad"
    )
    subprocess.run([sys.executable, "-c", code], check=True, timeout=600)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
