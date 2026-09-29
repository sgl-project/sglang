"""Small fixtures around production boundaries; no parallel layout recipes."""

from types import SimpleNamespace

from sglang.srt.layers.communicator.construction import BatchVariant, StagePlan
from sglang.srt.layers.communicator.factories import (
    declare_attn,
    declare_ffn,
    make_stages,
)
from sglang.srt.layers.communicator.residual.add_norm import PLAIN_RESIDUAL
from sglang.srt.layers.communicator.stage import StageCommunicator


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
        output=output,
    )
    common = {
        key: options.pop(key)
        for key in ("force_layernorm_before_dp_gather", "fusions")
        if key in options
    }
    attn, ffn = make_stages(
        (
            attention,
            attention_norm,
            dict(
                residual_in_hidden=residual.ffn_update.at_producer, **common, **options
            ),
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
    plan._cp_steps = plan._sp_steps = plan._input_scattered_steps = None
    plan.is_first_layer = False
    plan.residual_in_hidden = False
    # Keep production variant selection while tests replace individual paths.
    plan._batch_steps = lambda batch: {
        BatchVariant.ORDINARY: plan._steps,
        BatchVariant.CONTEXT_PARALLEL: plan._cp_steps,
        BatchVariant.INPUT_SCATTERED: plan._input_scattered_steps,
        BatchVariant.SEQUENCE_PARALLEL: plan._sp_steps,
    }[plan._variant(batch)]
    return plan


def stub_stage(plan, kind):
    """Keep the same concrete boundary when a test patches its methods."""
    stages = plan.__dict__.setdefault("_stub_stages", {})
    if kind not in stages:
        stages[kind] = StageCommunicator(plan, kind, plan.norm)
    return stages[kind]


def sp_region_steps():
    """Local SP rows with no owed sum, for tests of activation and exits."""
    from sglang.srt.layers.communicator import (
        NORM_QUANT_READ,
        NORM_READ,
        BoundarySteps,
        CommunicateSummableTensorPairFn,
        EdgeDecl,
        Layout,
        StageEntry,
        StageInput,
        StageOutput,
        TokenAxis,
        make_boundary,
    )
    from sglang.srt.layers.communicator.ops import _hand_qkv_hook_its_input

    rows = Layout(frozenset({TokenAxis.ATTN_TP_SCATTER}))
    output = StageOutput(rows)

    def entry(read, handoff=None):
        selected = make_boundary(
            EdgeDecl(output, StageInput(rows, read=read), rows, rows)
        )
        return StageEntry(selected.prepare, rows, handoff=handoff)

    return BoundarySteps(
        entry(NORM_QUANT_READ, _hand_qkv_hook_its_input),
        entry(NORM_READ),
        output,
        CommunicateSummableTensorPairFn._trivial,
        False,
    )
