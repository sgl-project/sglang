"""The experimental Hopper path must never widen the deferred MoE contract."""

from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from sglang.srt.layers.layer_boundary.layout import Layout, SumGroup, TokenAxis
from sglang.srt.layers.layer_boundary.output import UnreducedOutput
from sglang.srt.layers.layer_boundary.row_transport import TransportRows
from sglang.srt.layers.layernorm import RMSNorm
from sglang.srt.layers.quantization import mxfp4_flashinfer_trtllm_moe as ar
from sglang.srt.model_executor.forward_mode import ForwardMode
from sglang.srt.models import deepseek_v4
from sglang.srt.models import deepseek_v4_boundary as boundary
from sglang.srt.models import deepseek_v4_mhc as mhc
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


@pytest.fixture
def runtime(monkeypatch):
    group = SimpleNamespace(
        cpu_group=object(), all_reduce=Mock(side_effect=lambda x: x)
    )
    parallel = SimpleNamespace(
        tp_size=8,
        attn_tp_size=8,
        attn_dp_size=1,
        attn_cp_size=1,
        tp_group=group,
        attn_tp_group=group,
    )
    flags = SimpleNamespace(
        sp_active=False,
        attn_input_scattered=False,
        attn_tp_sequence_sharded=False,
        mlp_reduce_scatter=False,
    )
    platform = SimpleNamespace(is_sm90=True, is_blackwell=False)
    guards = SimpleNamespace(
        prefill=False, breakable=False, invariant=False, quant=False
    )
    for module in (ar, boundary, mhc):
        monkeypatch.setattr(module, "get_parallel", lambda: parallel)
        monkeypatch.setattr(module, "get_platform", lambda: platform, raising=False)
    monkeypatch.setattr(deepseek_v4, "get_platform", lambda: platform)
    monkeypatch.setattr(mhc, "get_forward", lambda: flags)
    monkeypatch.setattr(mhc, "is_in_prefill_graph_scope", lambda: guards.prefill)
    monkeypatch.setattr(mhc, "is_in_breakable_cuda_graph", lambda: guards.breakable)
    monkeypatch.setattr(
        mhc, "is_batch_invariant_mode_enabled", lambda: guards.invariant
    )
    monkeypatch.setattr(
        mhc,
        "get_exec",
        lambda: SimpleNamespace(
            comm=SimpleNamespace(enable_quant_communications=guards.quant)
        ),
    )
    lease = SimpleNamespace(
        world_size=8,
        group=group.cpu_group,
        request=SimpleNamespace(push_slot_bytes=131072),
    )
    monkeypatch.setattr(ar, "_fused_all_reduce_lease", lease)
    monkeypatch.setattr(ar, "_fused_all_reduce_registered", True)
    monkeypatch.setenv("SGLANG_OPT_DSV41_SM90_ATTN_MHC_AR", "1")
    monkeypatch.setattr(
        boundary,
        "get_cp_decode_attn_tp_ctx",
        lambda: SimpleNamespace(use_decode_attn_tp=False),
    )
    cfg = mhc.HcConfig(4, 20, 1e-5, 1e-6, 5120, True, False)
    layer = SimpleNamespace(
        config=SimpleNamespace(model_type="deepseek_v41"),
        self_attn=SimpleNamespace(attn_tp_size=8),
        post_attention_layernorm=RMSNorm(5120, weight_dtype=torch.bfloat16),
        ffn_hc=None,
        hc_cfg=cfg,
        boundary_pre_from_prev=True,
        boundary_mhc_fusions=True,
        _can_fuse_attn_mhc=False,
    )
    layer._can_fuse_sm90_attn_mhc = (
        deepseek_v4.DeepseekV4CudaDecoderLayer._can_fuse_sm90_attn_mhc.func(layer)
    )
    assert layer._can_fuse_sm90_attn_mhc
    return SimpleNamespace(
        group=group,
        parallel=parallel,
        flags=flags,
        guards=guards,
        platform=platform,
        lease=lease,
        layer=layer,
    )


def _transport(
    runtime,
    rows,
    *,
    dtype=torch.bfloat16,
    mode=ForwardMode.DECODE,
    sharded=False,
    tbo=False,
    decode_group=None,
):
    layout = Layout(frozenset({TokenAxis.ATTN_TP}) if sharded else frozenset())
    return boundary.AttentionTransport(
        runtime.layer,
        torch.empty((rows, 5120), dtype=dtype),
        SimpleNamespace(forward_mode=mode, can_run_tbo=tbo),
        boundary.LayerCall(tbo=tbo),
        TransportRows(
            layout, layout, layout, SumGroup.ATTN_TP, decode_group=decode_group
        ),
    )


@pytest.mark.parametrize("mode", [ForwardMode.DECODE, ForwardMode.TARGET_VERIFY])
@pytest.mark.parametrize(
    "rows,eligible", [(0, False), (1, True), (6, True), (12, True), (13, False)]
)
def test_physical_rows_and_slot_capacity(runtime, rows, eligible, mode):
    assert _transport(runtime, rows, mode=mode).defers_sum is eligible
    assert mhc.SM90_ATTN_MHC_MAX_ROWS == 12
    # The legacy TP2/TP4 Blackwell gate remains the only MoE gate.
    assert not mhc.fused_mhc_tp_ok(8)
    assert not mhc.can_fuse_post(runtime.layer.hc_cfg)


def test_opt_in_and_pre_capture_registration(runtime, monkeypatch):
    register = Mock()
    monkeypatch.setattr(ar, "register_fused_all_reduce_comm", register)
    model = SimpleNamespace(config=runtime.layer.config, modules=list)
    monkeypatch.delenv("SGLANG_OPT_DSV41_SM90_ATTN_MHC_AR")
    assert not _transport(runtime, 6).defers_sum
    deepseek_v4.DeepseekV4ForCausalLM.precompile_kernels_after_loading(model)
    register.assert_not_called()

    monkeypatch.setenv("SGLANG_OPT_DSV41_SM90_ATTN_MHC_AR", "1")
    deepseek_v4.DeepseekV4ForCausalLM.precompile_kernels_after_loading(model)
    register.assert_called_once_with()


@pytest.mark.parametrize(
    "gate",
    [
        "prefill",
        "breakable",
        "invariant",
        "quant",
        "sp_active",
        "attn_input_scattered",
        "attn_tp_sequence_sharded",
        "mlp_reduce_scatter",
        "attn_dp_size",
        "attn_cp_size",
        "attn_tp_group",
        "lease_group",
        "short_lease",
        "no_lease",
        "architecture",
        "norm",
    ],
)
def test_ineligible_transport_uses_ordinary_reduction(runtime, monkeypatch, gate):
    if hasattr(runtime.guards, gate):
        setattr(runtime.guards, gate, True)
    elif hasattr(runtime.flags, gate):
        setattr(runtime.flags, gate, True)
    elif gate in ("attn_dp_size", "attn_cp_size"):
        setattr(runtime.parallel, gate, 2)
    elif gate == "attn_tp_group":
        runtime.parallel.attn_tp_group = SimpleNamespace(cpu_group=object())
    elif gate == "lease_group":
        runtime.lease.group = object()
    elif gate == "short_lease":
        runtime.lease.request.push_slot_bytes = 32 * 1024
    elif gate == "no_lease":
        monkeypatch.setattr(ar, "_fused_all_reduce_lease", None)
    elif gate == "architecture":
        runtime.platform.is_sm90 = False
    elif gate == "norm":
        runtime.layer.post_attention_layernorm = RMSNorm(
            5120, weight_dtype=torch.float32
        )
        runtime.layer._can_fuse_sm90_attn_mhc = (
            deepseek_v4.DeepseekV4CudaDecoderLayer._can_fuse_sm90_attn_mhc.func(
                runtime.layer
            )
        )
    assert not _transport(runtime, 6).defers_sum


@pytest.mark.parametrize(
    "kwargs",
    [
        {"mode": ForwardMode.EXTEND},
        {"mode": ForwardMode.MIXED},
        {"mode": ForwardMode.IDLE},
        {"mode": ForwardMode.DRAFT_EXTEND_V2},
        {"sharded": True},
        {"tbo": True},
        {"decode_group": SumGroup.ATTN_CP},
        {"dtype": torch.float16},
    ],
)
def test_ineligible_row_layout_phase_or_dtype(runtime, kwargs):
    assert not _transport(runtime, 6, **kwargs).defers_sum


@pytest.mark.parametrize("bad", ["dtype", "residual_dtype", "strides", "group", "rows"])
def test_completing_read_declines_an_unsupported_partial(runtime, bad):
    rows = 13 if bad == "rows" else 6
    dtype = torch.float16 if bad == "dtype" else torch.bfloat16
    partial = torch.empty(rows, 5120, dtype=dtype)
    if bad == "strides":
        partial = torch.empty(5120, rows, dtype=dtype).t()
    residual = torch.empty(
        rows,
        4,
        5120,
        dtype=torch.float16 if bad == "residual_dtype" else torch.bfloat16,
    )
    group = object() if bad == "group" else runtime.group
    owed = UnreducedOutput(partial, group=group)
    update = SimpleNamespace(attention=True, post=Mock())
    reader = boundary.HcRead(runtime.layer, attention=False)
    read_state = boundary.ReadState(residual)
    assert reader._complete_read(owed, read_state, None, None, update=update) is None
    update.post.assert_not_called()


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
