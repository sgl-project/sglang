"""CPU regressions for the NANF cross-lane FFN row merge."""

from sglang.test.afd.graph_fixtures import make_shape as planned_shape
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=10, suite="base-a-test-cpu")

from types import SimpleNamespace

import pytest

from sglang.srt.afd import config as config_mod
from sglang.srt.afd import contracts as contracts
from sglang.srt.afd import pipeline as pipeline_mod
from sglang.srt.afd.model_adapters import base as adapter_mod
from sglang.srt.afd.model_adapters import glm5 as afd_glm5_adapter


def _ffn_adapter(*, layer):
    instance = object.__new__(adapter_mod.AFDDecoderAdapter)
    instance.role = contracts.AFDRole.FFN
    instance.inner = SimpleNamespace(layers=[layer])
    return instance


def _stage(*, rows, merge_sizes, merge_lane, group_widths=None):
    lane_rows = (rows,) if isinstance(rows, int) else rows
    return adapter_mod.AFDStage(
        index=0,
        request_start=0,
        request_stop=sum(lane_rows),
        token_start=0,
        token_stop=sum(lane_rows),
        hidden_states=None,
        residual=None,
        positions=None,
        forward_batch=None,
        merge_sizes=merge_sizes,
        merge_lane=merge_lane,
        group_widths=(
            (merge_sizes[merge_lane],) if group_widths is None else group_widths
        ),
    )


@pytest.mark.parametrize(
    "adapter_type", [adapter_mod.AFDDecoderAdapter, afd_glm5_adapter.Glm5AFDAdapter]
)
@pytest.mark.parametrize(
    "rows,widths,sizes,rank",
    [
        ((5,), (32,), (32,), 0),
        ((64,), (64,), (32, 64, 32), 1),
        ((9,), (32,), (32, 32), 1),
        ((0,), (32,), (32, 32), 0),
        ((20, 9), (32, 32), (64, 64, 64, 64), 1),
        ((20, 9), (32, 32), (64,), 0),
    ],
)
def test_merge_packs_live_rows_with_stable_padding_and_rank_local_returns(
    monkeypatch, adapter_type, rows, widths, sizes, rank
):
    import torch

    from sglang.srt.runtime_context import get_forward

    inputs = tuple(torch.full((count, 4), float(i + 1)) for i, count in enumerate(rows))
    gather = len(sizes) > 1
    strides = widths if gather else rows
    expected = torch.cat(
        tuple(
            torch.nn.functional.pad(x, (0, 0, 0, stride - x.shape[0]))
            for x, stride in zip(inputs, strides)
        )
    )
    calls = []

    class Group:
        rank_in_group = rank

        def all_gatherv(self, local, sizes):
            torch.testing.assert_close(local, expected)
            if len(inputs) == 1 and rows == widths:
                assert local is inputs[0]  # Captured full-width input needs no packing.
            calls.append("gather")
            return [
                torch.cat(
                    [
                        local if r == rank else torch.zeros(n, 4)
                        for r, n in enumerate(sizes)
                    ]
                )
            ]

        def reduce_scatterv(self, partial, sizes):
            calls.append("scatter")
            return partial.narrow(0, sum(sizes[:rank]), sizes[rank])

    def compute(hidden, batch):
        assert bool(get_forward().mlp_reduce_scatter) == gather
        if not gather:
            torch.testing.assert_close(hidden, expected)
        return hidden

    monkeypatch.setattr(
        adapter_mod, "get_parallel", lambda: SimpleNamespace(tp_group=Group())
    )
    adapter = object.__new__(adapter_type)
    adapter.role = contracts.AFDRole.FFN
    adapter.inner = SimpleNamespace(
        layers=[SimpleNamespace(compute_ffn_output=compute)]
    )
    outputs, residual = adapter.local_compute(
        layer=0,
        stage=_stage(
            rows=rows, merge_sizes=sizes, merge_lane=rank, group_widths=widths
        ),
        hidden_states=inputs,
        residual=None,
    )
    assert residual is None
    for output, source in zip(outputs, inputs):
        torch.testing.assert_close(output, source)
    assert calls == (["gather", "scatter"] if gather else [])
    if not gather and len(inputs) == 1:
        assert outputs[0] is inputs[0]


@pytest.mark.parametrize("rows,widths", [((40,), (32,)), ((8,), (32, 32))])
def test_invalid_lane_widths_fail_before_collective(monkeypatch, rows, widths):
    import torch

    monkeypatch.setattr(
        adapter_mod, "get_parallel", lambda: pytest.fail("unexpected collective")
    )
    with pytest.raises(contracts.AFDError, match="AFD_FFN_MERGE_WIDTH_EXCEEDED"):
        _ffn_adapter(layer=object()).local_compute(
            layer=0,
            stage=_stage(
                rows=rows, merge_sizes=(64, 64), merge_lane=0, group_widths=widths
            ),
            hidden_states=tuple(torch.ones(n, 4) for n in rows),
            residual=None,
        )


@pytest.mark.parametrize(
    "lane_stage_rows,expected",
    [
        (((4, 4),), (True, True)),
        (((4, 0),), (True, False)),
        (((0, 0),), (False, False)),
        (((1, 0), (4, 4)), (True, True)),
        (((1, 0), (1, 0)), (True, False)),
        (((0, 0), (0, 0), (2, 1), (0, 0)), (True, True)),
    ],
)
def test_stage_participation_is_a_global_decision(lane_stage_rows, expected):
    """Every lane must reach the same answer from the shared row matrix."""

    assert pipeline_mod._stage_participation(lane_stage_rows) == expected


def test_merge_width_matches_the_shape_bucket_rows():
    """The width the FFN pads to must be the same one the buffers are sized by."""

    cfg = config_mod.AFDConfig(
        lanes=2,
    )
    shape = planned_shape(
        lane=1,
        lane_rows=((9, 3), (40, 40)),
        hidden_size=64,
        dtype="bfloat16",
        config=cfg,
    )
    for stage in range(cfg.stages):
        sizes = shape.merge_plan(stage=stage)
        assert sizes[shape.lane] == shape.bucket_rows[stage]


def test_stating_the_symmetric_lane_count_leaves_the_shape_alone():
    """Fan-in may add a shape, not move the one the measured arms ran at.

    The digest is pinned by value because it is what both roles key their receive
    buffers and bucket programs on: changing its recipe silently would let a rank
    reuse another shape's backing.
    """

    rows = ((9, 3), (40, 40))
    implicit = planned_shape(
        lane=1,
        lane_rows=rows,
        hidden_size=64,
        dtype="bfloat16",
        config=config_mod.AFDConfig(
            lanes=2,
        ),
    )
    explicit = planned_shape(
        lane=1,
        lane_rows=rows,
        hidden_size=64,
        dtype="bfloat16",
        config=config_mod.AFDConfig(
            lanes=2,
            attention_lanes=2,
        ),
    )
    assert implicit.digest == explicit.digest == "64x64|64x64:64:bfloat16"
    assert implicit.lanes_per_ffn == explicit.lanes_per_ffn == 1
    for stage in range(2):
        assert implicit.merge_plan(stage=stage) == explicit.merge_plan(stage=stage)
    for ordinal in range(2):
        assert implicit.group_lanes(ffn_ordinal=ordinal) == (ordinal,)
        assert (
            implicit.group_bucket_rows(ffn_ordinal=ordinal)
            == implicit.lane_bucket_rows[ordinal]
        )


def test_a_fanin_shape_groups_lanes_without_moving_the_digest():
    """8A4F: each FFN rank owns two lanes, and both roles still see one digest."""

    cfg = config_mod.AFDConfig(
        lanes=4,
        attention_lanes=8,
    )
    # Spread across buckets, or every lane quantizes to the same width and the
    # group projections become indistinguishable.
    rows = tuple((16 * index + 1, 8) for index in range(8))
    shape = planned_shape(
        lane=1,
        lane_rows=rows,
        hidden_size=64,
        dtype="bfloat16",
        config=cfg,
    )
    assert (shape.attention_lanes, shape.ffn_size, shape.lanes_per_ffn) == (8, 4, 2)
    assert shape.group_lanes(ffn_ordinal=1) == (2, 3)
    # Stage-major, so the receive buffers land stage by stage in lane order.
    assert shape.group_bucket_rows(ffn_ordinal=1) == (128, 128, 128, 128)
    assert shape.group_total_bucket_rows(ffn_ordinal=1) == (256, 256)
    assert shape.group_total_stage_rows(ffn_ordinal=1) == (33 + 49, 16)
    # One entry per FFN rank, and the widths sum to the whole padded matrix.
    for stage in range(2):
        plan = shape.merge_plan(stage=stage)
        assert len(plan) == 4
        assert sum(plan) == sum(
            shape.lane_bucket_rows[lane][stage] for lane in range(8)
        )
        assert plan[1] == shape.group_total_bucket_rows(ffn_ordinal=1)[stage]
    assert shape.merge_plan(stage=0) == (256, 256, 256, 256)
    # Spans the whole matrix, so every rank of both roles computes the same one.
    assert shape.digest == ("|".join(["128x128"] * 8) + ":64:bfloat16")


@pytest.mark.parametrize("attention", [False, True])
@pytest.mark.parametrize("sparse", [False, True])
@pytest.mark.parametrize("layer_id", [0, 1])
def test_the_attention_role_declares_its_mlp_lane_local(
    monkeypatch, attention, sparse, layer_id
):
    import torch

    from sglang.srt.layers.communicator import layer as comm
    from sglang.srt.layers.communicator.layout import Layout, TokenAxis

    monkeypatch.setattr(
        comm,
        "get_disagg",
        lambda: SimpleNamespace(afd_execution_mode="attention" if attention else "off"),
    )
    monkeypatch.setattr(
        comm,
        "get_parallel",
        lambda: SimpleNamespace(attn_tp_size=1, attn_cp_size=1, attn_dp_size=8),
    )
    for name in (
        "is_moe_input_scattered_across_dp_ranks",
        "enable_moe_dense_fully_dp",
        "_generic_prefill_cp_shards_tokens",
    ):
        monkeypatch.setattr(comm, name, lambda: False)
    monkeypatch.setattr(
        comm,
        "get_exec",
        lambda: SimpleNamespace(
            overlap=SimpleNamespace(enable_two_batch_overlap=False)
        ),
    )
    monkeypatch.setattr(
        comm,
        "token_axis_sizes",
        lambda **kw: {
            TokenAxis.ATTN_DP: 8,
            TokenAxis.ATTN_CP: 1,
            TokenAxis.ATTN_TP_SCATTER: 1,
        },
    )
    instance = object.__new__(comm.LayerCommunicator)
    instance.layer_facts = comm.LayerFacts.init_new(
        layer_id=layer_id,
        num_layers=2,
        is_layer_sparse=sparse,
        is_previous_layer_sparse=not sparse,
        is_next_layer_sparse=not sparse,
    )
    instance.allow_deferred_ffn_reduction = True
    instance.allow_reduce_scatter = False
    sides = instance._declared_sides()
    lane = Layout(frozenset({TokenAxis.ATTN_DP}))
    assert sides.input_rows == sides.output_rows == sides.ffn_residual_rows == lane
    assert sides.ffn.layout == (lane if attention else Layout(frozenset()))
    assert (sides.ffn_output.group is None) == attention
    if attention:
        instance._attn_input_fusions = ()
        instance._steps = instance._steps_from_declarations(sides)
        instance._sp_steps = instance._cp_steps = instance._input_scattered_steps = None
        instance._context = None
        assert not instance._steps.returns_over_dp
        assert not instance._steps.ffn_sum_is_movable
        hidden, residual = torch.ones(3, 4), torch.full((3, 4), 7.0)
        result, carry = instance.postprocess_layer(hidden, residual, SimpleNamespace())
        assert result is hidden and carry is residual


@pytest.mark.parametrize("mode", ["off", "ffn"])
@pytest.mark.parametrize("ep_size", [1, 4])
@pytest.mark.parametrize("inplace", [False, True])
@pytest.mark.parametrize("fused", [False, True])
def test_triton_runner_filters_afd_padding_even_without_expert_parallel(
    monkeypatch, mode, ep_size, inplace, fused
):
    import torch

    hidden = torch.ones((2, 4))
    calls = []

    def run_inplace(*args, **kwargs):
        calls.append(args[-1])

    def run_outplace(*args, **kwargs):
        calls.append(kwargs["filter_expert"])
        return hidden

    from sglang.srt.layers.moe.moe_runner import triton as triton_runner
    from sglang.srt.layers.moe.moe_runner.triton_utils import fused_moe

    monkeypatch.setattr(
        fused_moe, "get_disagg", lambda: SimpleNamespace(afd_execution_mode=mode)
    )
    monkeypatch.setattr(
        triton_runner,
        "get_server_args",
        lambda: SimpleNamespace(afd_execution_mode=mode),
    )
    monkeypatch.setattr(fused_moe, "inplace_fused_experts", run_inplace)
    monkeypatch.setattr(fused_moe, "outplace_fused_experts", run_outplace)
    config = SimpleNamespace(
        num_experts=8,
        num_local_experts=8 // ep_size,
        inplace=inplace,
        no_combine=False,
        activation="silu",
        is_gated=True,
        apply_router_weight_on_input=False,
        routed_scaling_factor=1.0,
        gemm1_alpha=None,
        gemm1_clamp_limit=None,
        swiglu_limit=None,
        gate_up_interleaved=True,
    )
    ids = torch.tensor([[0], [-1]], dtype=torch.int32)
    weights = torch.tensor([[1.0], [0.0]])
    if fused:
        output = fused_moe.fused_experts(
            hidden, None, None, (weights, ids, None), config
        )
    else:
        monkeypatch.setattr(fused_moe, "_fused_moe_kernel_sequence", run_outplace)
        quant = SimpleNamespace(
            w13_weight=None,
            w2_weight=None,
            b13=None,
            b2=None,
            use_mxfp8=False,
            use_fp8_w8a8=False,
            use_int8_w8a8=False,
            use_int8_w8a16=False,
            use_int4_w4a16=False,
            per_channel_quant=False,
            w13_scale=None,
            w2_scale=None,
            w13_zp=None,
            w2_zp=None,
            a13_scale=None,
            a2_scale=None,
            block_shape=None,
            fuse_swiglu_interleaved=False,
        )
        runner_input = SimpleNamespace(
            hidden_states=hidden,
            topk_weights=weights,
            topk_ids=ids,
            sorted_token_ids=None,
            expert_ids=None,
            num_tokens_post_padded=None,
        )
        output = triton_runner.TritonRunnerCore.run(
            SimpleNamespace(config=config),
            runner_input,
            quant,
            {"config": {}},
        ).hidden_states
    assert output is hidden
    assert calls == [mode == "ffn" or ep_size > 1]


@pytest.mark.parametrize("rank", [0, 1])
@pytest.mark.parametrize("rows", [(3, 1), (0, 2)])
def test_reduce_scatter_sums_rank_partials_and_returns_own_live_rows(
    monkeypatch, rank, rows
):
    import torch

    from sglang.srt.runtime_context import get_forward

    sizes = (4, 8)
    inputs = tuple(
        torch.arange(n * 4, dtype=torch.float32).reshape(n, 4) + 10 * r
        for r, n in enumerate(rows)
    )
    padded = tuple(
        torch.nn.functional.pad(x, (0, 0, 0, width - x.shape[0]))
        for x, width in zip(inputs, sizes)
    )
    merged = torch.cat(padded)
    seen = []

    class Group:
        rank_in_group = rank

        def all_gatherv(self, local, sizes):
            torch.testing.assert_close(local, padded[rank])
            return [merged]

        def reduce_scatterv(self, partial, sizes):
            torch.testing.assert_close(partial, merged * (rank + 1))
            seen.append(tuple(sizes))
            # Peer owns the other expert shard; sum both actual CPU tensors.
            total = partial + merged * (2 - rank)
            return total.narrow(0, sum(sizes[:rank]), sizes[rank])

    monkeypatch.setattr(
        adapter_mod, "get_parallel", lambda: SimpleNamespace(tp_group=Group())
    )

    def compute(hidden, batch):
        from sglang.srt.layers.moe.utils import post_experts_all_reduce
        from sglang.srt.runtime_context import get_parallel

        assert get_forward().mlp_reduce_scatter
        partial = hidden * (rank + 1)
        # The native MoE helper must leave the sum for AFD's reduce-scatter.
        # No process group exists here: an unexpected all-reduce fails the test.
        with get_parallel().override(moe_ep_size=2, moe_tp_size=1):
            assert post_experts_all_reduce(partial) is partial
        return partial

    result, _ = _ffn_adapter(
        layer=SimpleNamespace(compute_ffn_output=compute)
    ).local_compute(
        layer=0,
        stage=_stage(rows=rows[rank], merge_sizes=sizes, merge_lane=rank),
        hidden_states=(inputs[rank],),
        residual=None,
    )
    torch.testing.assert_close(result[0], inputs[rank] * 3)
    assert seen == [sizes]
    assert not get_forward().mlp_reduce_scatter


def test_shared_tp1_guard_rejects_replicated_contribution():
    layer = SimpleNamespace(mlp=SimpleNamespace(_shared_expert_tp1=True))
    with pytest.raises(
        contracts.AFDError, match="AFD_FFN_MERGE_REDUCE_SCATTER_SHARED_TP1"
    ):
        _ffn_adapter(layer=layer).validate_merge_reduce_scatter()


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v", "-x"]))
