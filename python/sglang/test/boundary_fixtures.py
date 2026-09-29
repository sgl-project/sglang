"""Small fixtures around production boundaries; no parallel layout recipes."""

from types import SimpleNamespace

from sglang.srt.layers.layer_boundary.construction import StagePlan
from sglang.srt.layers.layer_boundary.factories import (
    declare_attn,
    declare_ffn,
    make_stages,
)
from sglang.srt.layers.layer_boundary.ops import identity_output
from sglang.srt.layers.layer_boundary.residual.add_norm import PLAIN_RESIDUAL
from sglang.srt.layers.layer_boundary.stage import StageBoundary


def make_test_stages(
    *,
    attention_norm,
    ffn_norm,
    first=False,
    last=False,
    sparse=False,
    previous_sparse=False,
    next_sparse=False,
    residual=PLAIN_RESIDUAL,
    output=None,
    **options,
):
    previous = (
        None
        if first
        else declare_ffn(
            sparse=previous_sparse, next_sparse=sparse, update=residual.ffn_update
        )
    )
    attention = declare_attn(
        read=residual.attention_read, update=residual.attention_update
    )
    ffn = declare_ffn(
        sparse=sparse,
        next_sparse=next_sparse,
        read=residual.ffn_read,
        update=residual.ffn_update,
        output_transform=output,
    )
    common = {key: options.pop(key) for key in ("fusions",) if key in options}
    attn, ffn = make_stages(
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
    plan._fusion_rows = None
    plan._paths = {}
    plan.enters_stack = False
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
        NORM_QUANT_READ,
        EdgeDecl,
        Layout,
        StageEntry,
        StageInput,
        StageOutput,
        StageSteps,
        TokenAxis,
        make_boundary,
    )
    from sglang.srt.layers.layer_boundary.prepare import _hand_qkv_hook_its_input

    rows = Layout(frozenset({TokenAxis.ATTN_TP_SCATTER}))
    output = StageOutput(rows)

    def entry(read, handoff=None):
        selected = make_boundary(
            EdgeDecl(output, StageInput(rows, read=read), rows, rows)
        )
        return StageEntry(selected.prepare, rows, handoff=handoff)

    return StageSteps(
        entry(NORM_QUANT_READ, _hand_qkv_hook_its_input), output, identity_output
    )


def prepare_input(stage, hidden, residual, forward_batch, **call):
    return prepare_raw(stage, "_prepare_input", hidden, residual, forward_batch, **call)


def prepare_attention(stage, hidden, residual, forward_batch, *args, **call):
    return prepare_raw(
        stage, "_prepare_attention", hidden, residual, forward_batch, *args, **call
    )


def prepare_raw(stage, method, hidden, residual, forward_batch, *args, **call):
    from sglang.srt.layers.layer_boundary.residual.add_norm import ADD
    from sglang.srt.layers.layer_boundary.residual.stream import ResidualStream

    if not isinstance(residual, ResidualStream):
        stream = ResidualStream(residual)
        if residual is not None:
            hidden = stream.leave(
                hidden,
                call.get("update", ADD),
                declared_sum=stage.entry(forward_batch).input_sum,
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
    hidden = boundary.postprocess_layer(hidden, stream, forward_batch)
    if isinstance(residual, ResidualStream):
        return hidden, stream
    return (hidden, None) if stream.pending is None else stream.input(hidden)


def identity_input(hidden_states, forward_batch):
    return hidden_states
