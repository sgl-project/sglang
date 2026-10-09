"""Small fixtures around production boundaries; no parallel layout recipes."""

from types import SimpleNamespace
from unittest import mock

from sglang.srt.layers.layer_boundary import factories
from sglang.srt.layers.layer_boundary.construction import StagePlan
from sglang.srt.layers.layer_boundary.factories import (
    append_stages,
    declare_attn,
    declare_ffn,
    layer_stack,
)
from sglang.srt.layers.layer_boundary.ops import keep_output
from sglang.srt.layers.layer_boundary.residual.add_norm import PLAIN_RESIDUAL_OPS
from sglang.srt.layers.layer_boundary.stage import StageBoundary


def build_stages(*stages, previous=None, terminal=False):
    """Bind one sequence of stages in a layer stack of its own.

    Args:
        *stages: Items of (declaration, norm) or (declaration, norm, options),
            as for append_stages.
        previous: Declaration of the stage before the first one, standing for
            a layer on the same rank that the test does not build; None starts
            the stack.
        terminal: Whether the last stage ends the model's layer stack;
            otherwise the next layer's attention follows it on the same rank.
    """
    # The stack's neighbours are layers another pipeline rank holds; here
    # they stand for this rank's, so nothing is handed off to them.
    with (
        mock.patch.object(factories, "_handed_off", lambda declaration: declaration),
        layer_stack(
            previous_layers=[lambda: (previous,)] if previous is not None else [],
            next_layers=[] if terminal else [lambda: (declare_attn(),)],
        ),
    ):
        return append_stages(*stages)


def make_test_stages(
    *,
    attention_norm,
    ffn_norm,
    first=False,
    last=False,
    sparse=False,
    previous_sparse=False,
    next_layer_sparse=False,
    residual=PLAIN_RESIDUAL_OPS,
    output=None,
    **options,
):
    previous = (
        None
        if first
        else declare_ffn(
            sparse=previous_sparse, next_layer_sparse=sparse, update=residual.ffn_update
        )
    )
    attention = declare_attn(read=residual.attn_readout, update=residual.attn_update)
    ffn = declare_ffn(
        sparse=sparse,
        next_layer_sparse=next_layer_sparse,
        read=residual.ffn_readout,
        update=residual.ffn_update,
        output_transform=output,
    )
    common = {key: options.pop(key) for key in ("fusions",) if key in options}
    attn, ffn = build_stages(
        (
            attention,
            attention_norm,
            dict(**common, **options),
        ),
        (ffn, ffn_norm, common),
        previous=previous,
        terminal=last,
    )
    return SimpleNamespace(attn=attn, ffn=ffn)


def stub_plan():
    """Supply only defaults for tests that inject preselected paths/kernels."""
    plan = StagePlan.__new__(StagePlan)
    plan.norm = None
    plan.fusions = None
    plan._publish_lora_layout = False
    plan._next_input_rows = None
    plan._unpadded_attn_tp_size = None
    plan.paths = {}
    plan.enters_stack = False
    plan.finishes_directly = False
    return plan


def stub_stage(plan, kind):
    """Keep the same concrete boundary when a test patches its methods."""
    stages = plan.__dict__.setdefault("_stub_stages", {})
    if kind not in stages:
        stages[kind] = StageBoundary(
            plan,
            declaration=(declare_attn() if kind.name == "ATTENTION" else declare_ffn()),
        )
    return stages[kind]


def sp_region_steps():
    """Local SP rows with no owed sum, for tests of activation and exits."""
    from sglang.srt.layers.layer_boundary import (
        NORM_QUANT_READOUT,
        EdgeContract,
        InputContract,
        Layout,
        OutputContract,
        StagePath,
        TokenAxis,
        bind_entry,
    )
    from sglang.srt.layers.layer_boundary.prepare import _attn_input_default

    rows = Layout(frozenset({TokenAxis.ATTN_TP}))
    output = OutputContract(rows)

    def entry(read, attn_input_adapter=None):
        return bind_entry(
            EdgeContract(output, InputContract(rows, read=read), rows, rows),
            attn_input_adapter=attn_input_adapter,
        )

    return StagePath(
        entry(NORM_QUANT_READOUT, _attn_input_default), output, keep_output
    )


def prepare_input(stage, hidden, residual, forward_batch, **call):
    return prepare_raw(stage, "_prepare_input", hidden, residual, forward_batch, **call)


def prepare_attention(stage, hidden, residual, forward_batch, *args, **call):
    return prepare_raw(
        stage, "_prepare_attention", hidden, residual, forward_batch, *args, **call
    )


def prepare_raw(stage, method, hidden, residual, forward_batch, *args, **call):
    from sglang.srt.layers.layer_boundary.residual.add_norm import PLAIN_ADD
    from sglang.srt.layers.layer_boundary.residual.stream import ResidualStream

    if not isinstance(residual, ResidualStream):
        stream = ResidualStream(residual)
        if residual is not None:
            hidden = stream.record(
                hidden,
                call.get("update", PLAIN_ADD),
                declared_sum=stage.entry(forward_batch).declared_sum,
            )
    else:
        stream = residual
    return getattr(stage, method)(hidden, stream, forward_batch, *args, **call)


def finish_exit(scope, hidden, residual):
    from sglang.srt.layers.layer_boundary.residual.stream import ResidualStream

    if isinstance(residual, ResidualStream):
        scope._stream = residual
        return scope.finish(hidden), residual
    scope._stream.residual = residual
    output = scope.finish(hidden)
    if scope._stream.pending is None:
        return output, None
    return scope._stream.input(output)


def postprocess_output(boundary, hidden, residual, forward_batch):
    from sglang.srt.layers.layer_boundary.residual.stream import ResidualStream

    stream = (
        residual if isinstance(residual, ResidualStream) else ResidualStream(residual)
    )
    hidden = boundary.complete_now(hidden, stream, forward_batch)
    if isinstance(residual, ResidualStream):
        return hidden, stream
    return (hidden, None) if stream.pending is None else stream.input(hidden)


def identity_input(hidden_states, forward_batch):
    return hidden_states
