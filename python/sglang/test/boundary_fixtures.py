"""Small fixtures around production boundaries; no parallel layout recipes."""

from contextlib import contextmanager
from types import SimpleNamespace
from unittest import mock
from unittest.mock import patch

from sglang.srt.layers import layernorm_sp
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
from sglang.srt.layers.moe import utils as moe_utils
from sglang.test.communicator_patch import patch_communicator


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


def parallel_of(*, attn_dp, attn_tp, attn_cp=1, **overrides):
    fields = dict(
        tp_size=attn_dp * attn_cp * attn_tp,
        tp_rank=0,
        attn_dp_size=attn_dp,
        attn_tp_size=attn_tp,
        attn_tp_rank=0,
        attn_cp_size=attn_cp,
        attn_cp_rank=0,
        moe_dense_tp_size=None,
        enable_prefill_cp=False,
        enable_cp_tp_group_sharing=False,
        moe_ep_size=1,
        moe_tp_size=attn_dp * attn_cp * attn_tp,
        moe_dp_size=1,
        dwdp_size=1,
        attn_dp_enabled=attn_dp > 1,
        enable_attn_tp_input_scattered=False,
        disable_attn_tp_gather=False,
        tp_group=SimpleNamespace(
            name="tp", ranks=list(range(attn_dp * attn_cp * attn_tp))
        ),
        attn_tp_group=SimpleNamespace(name="attn_tp", ranks=list(range(attn_tp))),
        attn_cp_group=SimpleNamespace(name="attn_cp", ranks=list(range(attn_cp))),
    )
    fields.update(overrides)
    return SimpleNamespace(**fields)


@contextmanager
def planning(
    parallel,
    *,
    sp=False,
    a2a=False,
    dsa_cp=False,
    boundary_reduction="rs+rsv",
):
    """What layer planning and communicator construction read, without the
    process-wide parallel state. ``parallel`` may be a callable, for a
    per-thread parallel state. ``dsa_cp``: the prefill CP is DSA's (MLA's is
    the same to the communicator)."""
    get_parallel = parallel if callable(parallel) else (lambda: parallel)

    def moe_cp_gathers():
        return get_parallel().attn_cp_size > get_parallel().moe_dp_size

    with (
        patch_communicator("get_parallel", get_parallel),
        patch_communicator("is_dsa_enable_prefill_cp", lambda: dsa_cp),
        patch_communicator("is_mla_cp_enabled", lambda: False),
        patch_communicator(
            "get_moe_cp_size",
            lambda: get_parallel().attn_cp_size if moe_cp_gathers() else 1,
        ),
        patch.object(layernorm_sp, "layernorm_sp_enabled", lambda: sp),
        patch_communicator(
            "get_spec", lambda: SimpleNamespace(speculative_algorithm=None)
        ),
        patch_communicator(
            "get_moe_a2a_backend",
            lambda: SimpleNamespace(is_none=lambda: not a2a),
        ),
        patch_communicator("is_moe_input_scattered_across_dp_ranks", lambda: a2a),
        patch_communicator("is_enable_moe_cp_allgather", moe_cp_gathers),
        patch_communicator("get_lora", lambda: SimpleNamespace(enable_lora=False)),
        # A MoE whose EP and TP sums merge: its output's group is the TP group.
        patch_communicator(
            "post_experts_reduction_group", lambda: get_parallel().tp_group
        ),
        # Planning asks whether a dense layer gathers for two-batch overlap.
        patch_communicator(
            "get_exec",
            lambda: SimpleNamespace(
                comm=SimpleNamespace(
                    boundary_reduction=boundary_reduction,
                    enable_quant_communications=False,
                ),
                overlap=SimpleNamespace(enable_two_batch_overlap=False),
            ),
        ),
        # What the MoE predicates an exit's decisions read, in their module.
        patch.object(moe_utils, "get_parallel", get_parallel),
        patch.object(
            moe_utils,
            "get_moe_a2a_backend",
            lambda: SimpleNamespace(
                is_none=lambda: not a2a,
                is_flashinfer=lambda: False,
                is_pplx=lambda: False,
                is_flashinfer_megamoe=lambda: False,
            ),
        ),
        patch.object(moe_utils, "get_lora", lambda: SimpleNamespace(enable_lora=False)),
        patch.object(
            moe_utils,
            "get_exec",
            lambda: SimpleNamespace(
                comm=SimpleNamespace(enable_quant_communications=False)
            ),
        ),
        patch.object(
            moe_utils, "post_experts_reduction_group", lambda: get_parallel().tp_group
        ),
        patch.object(
            moe_utils, "should_use_flashinfer_cutlass_moe_fp4_allgather", lambda: False
        ),
        patch.object(
            moe_utils,
            "is_dp_attention_enabled",
            lambda: get_parallel().attn_dp_size > 1,
        ),
    ):
        yield


class Norm:
    """add + norm stand-in: (2 * (h + r), h + r); one input: 2 * h."""

    def __call__(self, x, residual=None, post_residual_addition=None):
        assert post_residual_addition is None
        if residual is None:
            return x * 2
        s = x + residual
        return s * 2, s
