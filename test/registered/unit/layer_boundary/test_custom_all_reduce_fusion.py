"""Push-plane provider selection, fallback, and terminal residual ownership."""

import sys
from functools import partial
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch
from msgspec.structs import replace

from sglang.srt.layers.aux_hidden_states import AuxHiddenStateList
from sglang.srt.layers.layer_boundary import (
    NORM_QUANT_READOUT,
    PLAIN_ADD,
    Layout,
    TokenAxis,
    UnreducedOutput,
)
from sglang.srt.layers.layer_boundary.fusions import custom_all_reduce_fusion as custom
from sglang.srt.layers.layer_boundary.fusions.moe_finalize import MoeDeferredFinalize
from sglang.srt.layers.layer_boundary.prepare import _run_entry, _update_read
from sglang.srt.layers.layer_boundary.residual import batch as residual_batch
from sglang.srt.layers.layer_boundary.residual.stream import ResidualStream
from sglang.srt.layers.layernorm import GemmaRMSNorm, RMSNorm
from sglang.srt.model_executor.forward_batch_info import ForwardMode
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


def _norm(dtype=torch.bfloat16):
    return GemmaRMSNorm(8, eps=1e-6).to(dtype=dtype)


def _handoff(finish=None):
    return MoeDeferredFinalize(
        routed_output=torch.ones(4, 8, dtype=torch.bfloat16),
        expert_weights=torch.ones(2, 2),
        permuted_indices=torch.arange(4, dtype=torch.int32).view(2, 2),
        gated_shared_output=torch.ones(2, 8, dtype=torch.bfloat16),
        m=2,
        finish=finish or Mock(return_value=torch.ones(2, 8, dtype=torch.bfloat16)),
    )


@pytest.fixture
def provider(monkeypatch):
    """Real provider decisions with GPU kernels and process state replaced."""
    from sglang.srt import batch_invariant_ops

    group = SimpleNamespace(ranks=[0, 1, 2, 3])
    gates = SimpleNamespace(invariant=False, scattered=False, quant_comm=False)
    parallel = SimpleNamespace(tp_size=4, attn_tp_size=4, tp_group=group)
    monkeypatch.setattr(custom, "get_parallel", lambda: parallel)
    monkeypatch.setattr(
        custom,
        "get_exec",
        lambda: SimpleNamespace(
            comm=SimpleNamespace(enable_quant_communications=gates.quant_comm)
        ),
    )
    monkeypatch.setattr(
        custom,
        "get_attn_tp_context",
        lambda: SimpleNamespace(input_scattered=gates.scattered),
    )
    monkeypatch.setattr(custom, "is_in_breakable_cuda_graph", lambda: False)
    monkeypatch.setattr(
        batch_invariant_ops, "is_batch_invariant_mode_enabled", lambda: gates.invariant
    )
    monkeypatch.setattr(custom, "can_fuse_all_reduce", lambda m, h: True)
    plain = Mock(return_value=(torch.ones(2, 8), torch.full((2, 8), 2.0)))
    finalize = Mock(return_value=(torch.ones(2, 8), torch.full((2, 8), 3.0)))
    monkeypatch.setattr(custom, "all_reduce_residual_norm", plain)
    monkeypatch.setattr(custom, "moe_finalize_all_reduce_residual_norm", finalize)
    rows = Layout(frozenset())
    plan = SimpleNamespace(
        norm=_norm(),
        terminal=False,
        incoming_residual_rows=rows,
        fused_input_rows=lambda batch: rows,
    )
    batch = SimpleNamespace(
        forward_mode=ForwardMode.DECODE,
        can_run_tbo=False,
        input_ids=torch.zeros(2),
    )
    return SimpleNamespace(
        fusion=custom.CustomAllReduceFusion(hidden_size=8, defers_finalize=True),
        plan=plan,
        batch=batch,
        group=group,
        parallel=parallel,
        gates=gates,
        plain=plain,
        finalize=finalize,
    )


@pytest.mark.parametrize("mode", [ForwardMode.DECODE, ForwardMode.TARGET_VERIFY])
def test_supported_batch_fuses_each_consumer(provider, mode):
    p = provider
    p.batch.forward_mode = mode
    x = torch.ones(2, 8, dtype=torch.bfloat16)
    residual = torch.zeros_like(x)
    assert p.fusion.can_defer_finalize(p.plan, p.batch)
    (ffn,) = p.fusion.ffn_input_fusions(p.plan)
    ffn.run(x, residual, p.batch)
    finalize, plain = p.fusion.attn_input_fusions(p.plan)
    plain(UnreducedOutput(x, group=p.group), residual, p.batch, None)
    finalize(_handoff(), residual, p.batch, None)
    assert p.plain.call_count == 2
    p.finalize.assert_called_once()
    for call in (*p.plain.call_args_list, *p.finalize.call_args_list):
        assert call.kwargs["world_size"] == 4
        assert call.kwargs["residual_out"].data_ptr() != residual.data_ptr()
    assert torch.count_nonzero(residual) == 0


# Each gate is a batch the push-plane kernel would reduce wrongly or nondeterministically.
@pytest.mark.parametrize(
    "gate", ["tbo", "invariant", "scattered", "quant_comm", "sharded"]
)
def test_ineligible_batch_neither_defers_nor_launches(provider, gate):
    p = provider
    if gate == "tbo":
        p.batch.can_run_tbo = True
    elif gate == "sharded":
        p.plan.fused_input_rows = lambda batch: Layout(frozenset({TokenAxis.ATTN_TP}))
    else:
        setattr(p.gates, gate, True)
    x = torch.ones(2, 8, dtype=torch.bfloat16)
    owed = UnreducedOutput(x, group=p.group)
    assert not p.fusion.can_defer_finalize(p.plan, p.batch)
    assert p.fusion._ffn_input_add_norm(p.plan, x, x, p.batch) is None
    assert p.fusion._all_reduce_add_norm(p.plan, owed, x, p.batch, None) is None
    p.plain.assert_not_called()
    p.finalize.assert_not_called()


@pytest.mark.parametrize(
    "unsupported", ["norm", "input", "residual", "shared", "group"]
)
def test_inputs_the_kernel_cannot_take_decline_fusion(provider, unsupported):
    p = provider

    def dtype_unless(name):
        return torch.float16 if unsupported == name else torch.bfloat16

    x = torch.ones(2, 8, dtype=dtype_unless("input"))
    residual = torch.ones(2, 8, dtype=dtype_unless("residual"))
    group = SimpleNamespace(ranks=[0, 1]) if unsupported == "group" else p.group
    handoff = _handoff()
    handoff = replace(
        handoff,
        routed_output=handoff.routed_output.to(dtype_unless("input")),
        gated_shared_output=handoff.gated_shared_output.to(dtype_unless("shared")),
    )
    if unsupported == "norm":
        p.plan.norm = _norm(torch.float16)
        assert not p.fusion.can_defer_finalize(p.plan, p.batch)
        assert p.fusion.ffn_input_fusions(p.plan) == ()
    if unsupported in ("norm", "input", "residual"):
        assert p.fusion._ffn_input_add_norm(p.plan, x, residual, p.batch) is None
    if unsupported != "shared":
        owed = UnreducedOutput(x, group=group)
        assert (
            p.fusion._all_reduce_add_norm(p.plan, owed, residual, p.batch, None) is None
        )
    if unsupported != "group":
        assert (
            p.fusion._finalize_add_norm(p.plan, handoff, residual, p.batch, None)
            is None
        )
    p.plain.assert_not_called()
    p.finalize.assert_not_called()


def test_attention_dp_declines_ffn_input_fusion(provider):
    """The FFN input sum stays inside one attention-TP group; under attention DP
    the TP plane spans groups whose rows hold different requests."""
    provider.parallel.attn_tp_size = 2
    assert provider.fusion.ffn_input_fusions(provider.plan) == ()


def test_handoff_without_shared_expert_fuses(provider):
    """gated_shared_output=None is a valid handoff, which the finalize kernel takes."""
    p = provider
    handoff = replace(_handoff(), gated_shared_output=None)
    residual = torch.zeros(2, 8, dtype=torch.bfloat16)
    finalize, _ = p.fusion.attn_input_fusions(p.plan)
    assert finalize(handoff, residual, p.batch, None) is not None
    p.finalize.assert_called_once()
    assert p.finalize.call_args.args[4] is None


def test_declined_sum_completes_once_before_normal_read(provider):
    p = provider
    p.batch.forward_mode = ForwardMode.EXTEND
    finish = Mock(return_value=torch.ones(2, 8, dtype=torch.bfloat16))
    owed = UnreducedOutput(
        torch.zeros(2, 8, dtype=torch.bfloat16),
        group=SimpleNamespace(all_reduce=finish, ranks=p.group.ranks),
    )
    residual = torch.full((2, 8), 2.0, dtype=torch.bfloat16)

    def read(x, residual=None, post_residual_addition=None):
        updated = x + residual
        return updated * 2, updated

    # Keep the provider and entry path, replacing only the ordinary norm.
    p.plan.norm.forward = read
    result, updated = _run_entry(
        owed,
        residual,
        p.batch,
        p.plan.norm,
        is_plain_add=True,
        step=partial(
            _update_read,
            pre_move=None,
            enters_stack=False,
            read=NORM_QUANT_READOUT,
            update=PLAIN_ADD,
        ),
        carried_fusions=p.fusion.attn_input_fusions(p.plan),
    )
    assert finish.call_count == 1
    torch.testing.assert_close(result, torch.full_like(residual, 6))
    torch.testing.assert_close(updated, torch.full_like(residual, 3))
    p.plain.assert_not_called()
    p.finalize.assert_not_called()


@pytest.mark.parametrize("norm", [_norm(), _norm(torch.float16), RMSNorm(8)])
def test_terminal_deferral_requires_a_supported_final_norm(provider, norm):
    p = provider
    p.plan.terminal = True
    assert not p.fusion.can_defer_finalize(p.plan, p.batch)
    layer = SimpleNamespace(ffn_boundary=SimpleNamespace(fusions=p.fusion))
    installed = custom.install_terminal_finalize([layer], norm)
    supported = (
        isinstance(norm, GemmaRMSNorm) and norm.gemma_weight.dtype == torch.bfloat16
    )
    assert (installed is p.fusion) == supported
    assert p.fusion.can_defer_finalize(p.plan, p.batch) == supported


def test_final_norm_capture_owns_the_updated_residual(provider):
    p = provider
    p.fusion.final_norm_eps = p.plan.norm.variance_epsilon
    finish = Mock()
    stream = ResidualStream(torch.zeros(2, 8, dtype=torch.bfloat16))
    output = stream.record(_handoff(finish), PLAIN_ADD)
    p.batch.residual_stream = stream
    captures = AuxHiddenStateList()
    result = residual_batch.final_norm(
        output,
        p.batch,
        p.plan.norm,
        finalize_norm=p.fusion,
        capture=captures.capture,
    )
    normed, updated = p.finalize.return_value
    assert result is normed
    assert p.batch.residual_stream is None
    finish.assert_not_called()
    assert len(captures) == 1
    torch.testing.assert_close(captures[0], updated)
    updated.zero_()
    torch.testing.assert_close(captures[0], torch.full_like(updated, 3))


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
