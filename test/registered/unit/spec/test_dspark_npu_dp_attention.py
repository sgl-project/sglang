"""CPU execution of GLM compact DP contracts, including real Gloo collectives.

These tests do not execute DSA kernels, HCCL/DeepEP, or capture NPU graphs.
"""

from datetime import timedelta
from types import SimpleNamespace
from unittest.mock import Mock, patch

import msgspec
import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from sglang.srt.environ import envs
from sglang.srt.model_executor import forward_batch_info as fbi
from sglang.srt.speculative.dspark_components import dspark_dp, dspark_planner
from sglang.srt.speculative.ragged_verify import RaggedVerifyLayout
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=20, suite="base-a-test-cpu")


def _runner(graph=True):
    graph_runner = SimpleNamespace(
        ragged_verify_mode=True,
        max_bs=8,
        capture_num_tokens=[4, 8, 16, 32, 64],
        _ragged_capture_slots=lambda tokens: min(tokens, 8),
    )
    return SimpleNamespace(
        device="cpu",
        decode_cuda_graph_runner=graph_runner if graph else None,
        attn_backend=SimpleNamespace(
            _dspark_dsa_ragged_verify=True,
            get_cpu_graph_seq_len_fill_value=lambda: 1,
        ),
        attn_tp_sequence_sharded=lambda tokens: False,
        is_draft_worker=False,
    )


def _planner(runner=None, *, coordinate=True):
    planner = dspark_planner.DSparkVerifyPlanner.__new__(
        dspark_planner.DSparkVerifyPlanner
    )
    planner.model_runner = runner if runner is not None else _runner()
    planner.device = "cpu"
    planner.verify_num_draft_tokens = 9
    planner._verify_hooks = dspark_planner.DSparkVerifyHooks(
        sync_budget=dspark_dp.sync_verify_budget,
        coordinate_dp_verify=coordinate,
    )
    return planner


@pytest.mark.parametrize(
    "contributions,attn_tp,expected",
    [
        ([(2, 5), (1, 7), (0, 0)], 1, 8),
        ([(2, 5), (2, 5), (0, 0), (0, 0)], 2, 8),
        ([(1, -1), (0, 0)], 1, None),
        ([(0, 0), (0, 0)], 1, None),
        ([(9, 9), (1, 1)], 1, None),
        ([(1, 65), (0, 0)], 1, None),
    ],
)
def test_common_tier_and_collective_fallback(contributions, attn_tp, expected):
    planner = _planner()
    with patch.object(
        dspark_planner,
        "get_parallel",
        return_value=SimpleNamespace(attn_tp_size=attn_tp),
    ):
        assert (
            planner._resolve_dp_verify_tokens(
                [(tokens, bs) for bs, tokens in contributions]
            )
            == expected
        )


def test_eager_alignment_does_not_infer_request_count_from_tokens():
    planner = _planner(_runner(False))
    with patch.object(
        dspark_planner, "get_parallel", return_value=SimpleNamespace(attn_tp_size=4)
    ):
        assert planner._resolve_dp_verify_tokens([(5, 2), (0, 0)]) == 8


def test_packed_tier_is_not_clamped_to_fixed_window_and_checks_capture_slots():
    planner = _planner()
    with patch.object(
        dspark_planner, "get_parallel", return_value=SimpleNamespace(attn_tp_size=4)
    ):
        # Already padded to 16 for a single request: do not clamp to W=9.
        assert planner._resolve_dp_verify_tokens([(16, 1), (0, 0)]) == 16
        planner.model_runner.decode_cuda_graph_runner._ragged_capture_slots = (
            lambda tokens: 1
        )
        assert planner._resolve_dp_verify_tokens([(8, 2), (0, 0)]) is None


@pytest.mark.parametrize("lens", [[1, 4], [0]])
def test_packed_dp_mlp_padding_preserves_local_request_rows(lens):
    runner = _runner()
    layout = RaggedVerifyLayout.from_verify_lens_device(
        verify_lens=torch.tensor(lens), graph_num_tokens=8
    )
    bs = len(lens)
    batch = fbi.ForwardBatch(
        forward_mode=fbi.ForwardMode.TARGET_VERIFY,
        batch_size=bs,
        input_ids=torch.zeros(8, dtype=torch.long),
        req_pool_indices=torch.arange(bs),
        seq_lens=torch.full((bs,), 10),
        seq_lens_cpu=torch.full((bs,), 10),
        out_cache_loc=torch.zeros(8, dtype=torch.long),
        seq_lens_sum=bs * 10,
        positions=torch.arange(8),
        global_num_token_non_padded=torch.tensor(sum(lens)),
        global_num_token_non_padded_cpu=sum(lens),
        spec_info=SimpleNamespace(
            ragged_verify_layout=layout,
            num_tokens_per_req=9,
            is_draft_input=lambda: False,
        ),
        global_num_tokens_cpu=[18, 0],
        global_num_tokens_for_logprob_cpu=[18, 0],
        global_num_tokens_gpu=torch.tensor([18, 0]),
        global_num_tokens_for_logprob_gpu=torch.tensor([18, 0]),
    )
    parallel = SimpleNamespace(attn_dp_enabled=True, num_dp_ranks=2, attn_tp_size=2)
    with patch.object(dspark_dp, "get_parallel", return_value=parallel):
        dspark_dp.prepare_compact_dp_forward_batch(batch)
        assert batch.global_num_tokens_cpu == [8, 8]
        assert batch.global_num_token_non_padded.item() == 8
        assert batch.global_num_token_non_padded_cpu == 8
        assert batch.spec_info.ragged_verify_layout.verify_lens.tolist() == lens
        assert batch.global_num_tokens_for_logprob_gpu.tolist() == [8, 8]
        with (
            patch.multiple(
                fbi,
                get_parallel=lambda: parallel,
                get_exec=lambda: SimpleNamespace(
                    graph=SimpleNamespace(
                        cuda_graph_config=SimpleNamespace(
                            prefill=SimpleNamespace(bs=[])
                        )
                    )
                ),
                _elastic_should_preserve_local_token_counts=lambda **kwargs: False,
                dp_slot_in=lambda counts: 0,
                set_dp_buffer_len_from_batch=lambda *args: None,
                set_is_extend_in_batch=lambda *args: None,
                _is_cpu=True,
                _is_npu=True,
            ),
            patch.object(
                fbi.DpPaddingMode,
                "get_dp_padding_mode",
                return_value=fbi.DpPaddingMode.MAX_LEN,
            ),
            patch(
                "sglang.srt.batch_overlap.two_batch_overlap.TboForwardBatchPreparer.prepare"
            ),
        ):
            batch.prepare_mlp_sync_batch(runner)
    # The old fixed-width inverse produced 8 // 9 == 0 request rows.
    assert batch.batch_size == bs
    assert batch.seq_lens.numel() == bs
    assert batch.input_ids.numel() == 8
    assert batch.global_dp_buffer_len == 16
    # Output slicing must also keep the packed tier, even when another DP
    # group's token count exceeds this rank's fixed-width request envelope.
    batch.spec_info.ragged_verify_layout = msgspec.structs.replace(
        layout, graph_num_tokens=32
    )
    output = SimpleNamespace(
        next_token_logits=torch.ones(32, 4),
        hidden_states=torch.ones(32, 8),
        last_hidden_states=None,
    )
    batch.post_forward_mlp_sync_batch(output)
    assert output.next_token_logits.shape[0] == 32
    assert output.hidden_states.shape[0] == 32


def _collective_worker(rank, init_path, graph):
    dist.init_process_group(
        "gloo",
        init_method="file://" + init_path,
        rank=rank,
        world_size=4,
        timeout=timedelta(seconds=45),
    )
    try:
        groups = [dist.new_group([0, 1]), dist.new_group([2, 3])]
        attention_group = groups[rank // 2]
        runner = _runner(graph)
        parallel = SimpleNamespace(
            tp_group=SimpleNamespace(cpu_group=dist.group.WORLD),
            attn_tp_size=2,
            attn_tp_rank=rank % 2,
            attn_dp_rank=rank // 2,
        )

        # Two unrelated request sets must keep DIFFERENT budgets. The real
        # broadcast is confined to each attention-TP group.
        def broadcast(value, src):
            values = [value]
            dist.broadcast_object_list(
                values, src=(rank // 2) * 2, group=attention_group
            )
            return values[0]

        planner = _planner(runner)
        planner.model_runner.req_to_token_pool = SimpleNamespace(
            req_generation=torch.zeros(2, dtype=torch.long)
        )
        planner._budget_planner = Mock()
        planner._budget_from_resolved = Mock(
            return_value=(2 if rank < 2 else 6) + rank % 2
        )
        with (
            patch.object(
                dspark_dp,
                "get_dp_tp_group",
                return_value=SimpleNamespace(broadcast_object=broadcast),
            ),
        ):
            budget = planner.compute_budget_sync(
                confidence=torch.ones(1, 8),
                prefix_lens=torch.tensor([10]),
                req_pool_indices=torch.tensor([0]),
            )
        assert budget == (2 if rank < 2 else 6)
        # Existing scheduler-time path retains its one-column protocol and
        # per-rank list, including a negative fallback hint and an idle zero.
        planner._dp_tier_gather_enabled = True
        hint_batch = SimpleNamespace(is_extend_in_batch=False)
        with patch.object(dspark_planner, "get_parallel", return_value=parallel):
            planner._maybe_gather_dp_verify_tier(
                batch=hint_batch, local_tier_num_tokens=[3, 3, -1, 0][rank]
            )
        assert hint_batch.global_spec_verify_tier_num_tokens == [3, 3, -1, 0]
        assert (
            dspark_planner.dp_global_verify_tier_num_tokens(
                global_tier_num_tokens=hint_batch.global_spec_verify_tier_num_tokens
            )
            is None
        )
        # Alternate busy/idle ownership across rounds, then force a uniform
        # fallback on one group. Every rank must finish the same collectives.
        for step in range(6):
            busy = step == 4 or (step < 4 and rank // 2 == step % 2)
            lens = (
                torch.tensor([1, 2] if step == 4 and rank >= 2 else [1, 4])
                if busy
                else None
            )
            layout = (
                RaggedVerifyLayout.from_verify_lens_device(
                    verify_lens=lens, graph_num_tokens=int(lens.sum())
                )
                if busy and step != 3
                else None
            )
            batch = SimpleNamespace(batch_size=lambda: 2 if busy else 0)
            with patch.object(dspark_planner, "get_parallel", return_value=parallel):
                result = planner.coordinate_verify_layout(batch=batch, layout=layout)
            if step in (3, 5):
                assert result is None
            else:
                assert result.graph_num_tokens == (8 if graph else 6)
                assert result.verify_lens.tolist() == (lens.tolist() if busy else [0])
    finally:
        dist.destroy_process_group()


@pytest.mark.parametrize("graph", [False, True])
def test_real_gloo_busy_idle_and_attention_tp_budget_isolation(tmp_path, graph):
    mp.spawn(
        _collective_worker,
        args=(str(tmp_path / "gloo_init"), graph),
        nprocs=4,
        join=True,
    )


def test_uniform_fallback_does_not_rewrite_dp_counts():
    batch = SimpleNamespace(spec_info=None, global_num_tokens_cpu=[9, 0])
    with patch.object(dspark_dp, "get_parallel") as parallel:
        dspark_dp.prepare_compact_dp_forward_batch(batch)
    assert batch.global_num_tokens_cpu == [9, 0]
    parallel.assert_not_called()


@pytest.mark.parametrize("graph", [False, True])
@pytest.mark.parametrize("invalid_option", [None, "lm_head", "tp", "pp"])
def test_production_glm_dspark_reuses_parallel_and_speculative_validation(
    graph, invalid_option
):
    from sglang.srt.arg_groups.parallel_hook import handle_data_parallelism
    from sglang.srt.arg_groups.speculative_hook import _handle_dspark
    from sglang.srt.model_executor.cuda_graph_config import (
        Backend,
        CudaGraphConfig,
        PhaseConfig,
    )
    from sglang.srt.server_args import ServerArgs

    args = ServerArgs(model_path="dummy")
    args.device = "npu"
    args.speculative_algorithm = "DSPARK"
    args.speculative_draft_model_path = "unused-draft"
    args.speculative_dspark_block_size = 8
    args.speculative_num_draft_tokens = 9
    args.tp_size = 15 if invalid_option == "tp" else 16
    args.pp_size = 2 if invalid_option == "pp" else 1
    args.attn_dp_size = 4
    # The generic parallel hook runs after prefill defaults are resolved.
    args.chunked_prefill_size = 2048
    args.enable_dp_lm_head = invalid_option != "lm_head"
    args.moe_a2a_backend = "deepep"
    args.cuda_graph_config = CudaGraphConfig(
        decode=PhaseConfig(
            backend=Backend.FULL if graph else Backend.DISABLED, bs=[4, 8]
        ),
        prefill=PhaseConfig(backend=Backend.DISABLED),
    )
    args._model_config = SimpleNamespace(
        hf_config=SimpleNamespace(architectures=["GlmMoeDsaForCausalLM"])
    )
    with (
        envs.SGLANG_RAGGED_VERIFY_MODE.override("compact"),
        patch(
            "sglang.srt.speculative.dspark_components.dspark_config.read_draft_checkpoint_config",
            return_value=None,
        ),
    ):
        if invalid_option == "tp":
            with pytest.raises(AssertionError):
                handle_data_parallelism(args)
            return
        handle_data_parallelism(args)
        if invalid_option is not None:
            message = (
                "requires --enable-dp-lm-head"
                if invalid_option == "lm_head"
                else "pipeline parallelism requires PD prefill"
            )
            with pytest.raises(ValueError, match=message):
                _handle_dspark(args)
        else:
            _handle_dspark(args)


@pytest.mark.parametrize("is_npu", [False, True])
@pytest.mark.parametrize(
    "architecture",
    ["GlmMoeDsaForCausalLM", "DeepseekV4ForCausalLM", "Qwen3ForCausalLM"],
)
@pytest.mark.parametrize("mode", ["static", "compact"])
@pytest.mark.parametrize("dp", [False, True])
def test_verify_hook_selection_before_backend_init(is_npu, architecture, mode, dp):
    from sglang.srt.speculative.dspark_components import dspark_worker_v2

    runner = SimpleNamespace(
        model_config=SimpleNamespace(
            hf_config=SimpleNamespace(architectures=[architecture])
        )
    )
    with (
        patch.object(dspark_worker_v2, "_is_npu", is_npu),
        patch.object(
            dspark_worker_v2,
            "get_parallel",
            return_value=SimpleNamespace(attn_dp_enabled=dp),
        ),
        envs.SGLANG_RAGGED_VERIFY_MODE.override(mode),
        patch.object(dspark_dp, "get_dp_tp_group") as group,
    ):
        hooks = dspark_worker_v2._make_verify_hooks(runner)
    group.assert_not_called()
    assert not hasattr(runner, "attn_backend")
    if not is_npu or architecture != "GlmMoeDsaForCausalLM" or mode == "static":
        assert hooks is None
        return
    assert hooks.dynamic_tiers_with_dp
    assert hooks.coordinate_dp_verify == dp
    assert (hooks.prepare_forward_batch is not None) == dp


@pytest.mark.parametrize("enabled", [False, True])
def test_selected_path_rejects_incompatible_backend_before_draft_init(enabled):
    from sglang.srt.speculative.dspark_components import dspark_worker_v2

    worker = dspark_worker_v2.DSparkWorkerV2.__new__(dspark_worker_v2.DSparkWorkerV2)
    worker._hosts_draft = True
    worker._verify_hooks = dspark_dp.make_dspark_verify_hooks(
        coordinate_dp_verify=enabled
    )
    worker.model_runner = SimpleNamespace(attn_backend=object())
    worker._draft_worker = Mock()
    with pytest.raises(ValueError, match="GLM DSA ragged metadata support"):
        worker.init_attention_backends()
    worker._draft_worker.init_attention_backends.assert_not_called()


def test_default_planner_preserves_inputs_without_collectives():
    planner = _planner()
    planner._verify_hooks = None
    batch, layout = object(), object()
    with (
        patch.object(dspark_dp, "get_dp_tp_group") as group,
        patch.object(dspark_planner, "gather_dp_verify_tiers") as gather,
    ):
        assert not planner.coordinates_dp_verify
        assert planner.sync_verify_budget(None) is None
        assert planner.sync_verify_budget(3) == 3
        assert planner.coordinate_verify_layout(batch=batch, layout=layout) is layout
        assert planner.coordinate_verify_layout(batch=batch, layout=None) is None
    group.assert_not_called()
    gather.assert_not_called()


@pytest.mark.parametrize("busy", [False, True])
def test_boundary_gather_ignores_synthetic_idle_layout(busy):
    planner = _planner()
    original = RaggedVerifyLayout.from_verify_lens_device(
        verify_lens=torch.tensor([3] if busy else [0]), graph_num_tokens=32
    )
    batch = SimpleNamespace(batch_size=lambda: int(busy))
    with (
        patch.object(
            dspark_planner, "get_parallel", return_value=SimpleNamespace(attn_tp_size=2)
        ),
        patch.object(
            dspark_planner, "gather_dp_verify_tiers", return_value=[[32, 1], [0, 0]]
        ) as gather,
    ):
        result = planner.coordinate_verify_layout(batch=batch, layout=original)
    gather.assert_called_once_with(
        local_tier_num_tokens=32 if busy else 0, local_num_reqs=int(busy)
    )
    assert result.graph_num_tokens == 32
    assert result.verify_lens.tolist() == ([3] if busy else [0])


def test_existing_tier_gather_skips_mixed_prefill_and_disabled_paths():
    planner = _planner(coordinate=False)
    batch = SimpleNamespace(
        is_extend_in_batch=True, global_spec_verify_tier_num_tokens=[9]
    )
    with patch.object(dspark_planner, "gather_dp_verify_tiers") as gather:
        planner._dp_tier_gather_enabled = True
        planner._maybe_gather_dp_verify_tier(batch=batch, local_tier_num_tokens=3)
        assert batch.global_spec_verify_tier_num_tokens is None
        batch.is_extend_in_batch = False
        planner._dp_tier_gather_enabled = False
        planner._maybe_gather_dp_verify_tier(batch=batch, local_tier_num_tokens=3)
    gather.assert_not_called()


@pytest.mark.parametrize("hook_kind", ["none", "glm"])
@pytest.mark.parametrize("mode", ["static", "compact"])
@pytest.mark.parametrize("attn_tp", [1, 2])
@pytest.mark.parametrize("overlap", [False, True])
def test_planner_selects_one_coordination_point(hook_kind, mode, attn_tp, overlap):
    # Exercise the real constructor: defaults must retain upstream DSV4/GPU
    # scheduler-time gathering, while GLM target-boundary gathering replaces it.
    dp = True
    hooks = None
    if hook_kind == "glm":
        hooks = dspark_planner.DSparkVerifyHooks(
            sync_budget=lambda budget: budget,
            dynamic_tiers_with_dp=True,
            coordinate_dp_verify=dp and mode == "compact",
        )
    with (
        envs.SGLANG_RAGGED_VERIFY_MODE.override(mode),
        envs.SGLANG_SCHEDULER_SKIP_ALL_GATHER.override(False),
        patch.object(
            dspark_planner,
            "get_parallel",
            return_value=SimpleNamespace(
                attn_tp_size=attn_tp, attn_cp_size=1, pp_size=1
            ),
        ),
        patch.object(
            dspark_planner,
            "get_disagg",
            return_value=SimpleNamespace(disaggregation_mode="null"),
        ),
        patch.object(
            dspark_planner,
            "get_schedule",
            return_value=SimpleNamespace(disable_overlap_schedule=not overlap),
        ),
        patch.object(dspark_planner, "is_dp_attention_enabled", return_value=dp),
        patch.object(dspark_planner, "require_mlp_tp_gather", return_value=True),
        patch.object(
            dspark_planner,
            "get_spec",
            return_value=SimpleNamespace(
                speculative_dspark_align_verify_tokens_to_graph_tier=False,
                speculative_dspark_confidence_sts_path=None,
            ),
        ),
        patch.object(
            dspark_planner,
            "build_sps_cost_table",
            return_value=dspark_planner.build_uninitialized_sps_table(
                max_batch_tokens=32
            ),
        ),
    ):
        planner = dspark_planner.DSparkVerifyPlanner(
            draft_model=SimpleNamespace(confidence_head=object()),
            gamma=8,
            model_runner=_runner(),
            device="cpu",
            tp_rank=1,
            verify_num_draft_tokens=9,
            tp_sync=Mock(),
            verify_hooks=hooks,
        )
    assert planner.coordinates_dp_verify == (
        hook_kind == "glm" and dp and mode == "compact"
    )
    assert planner._dp_tier_gather_enabled == (
        hook_kind == "none" and dp and attn_tp == 1 and overlap and mode == "compact"
    )
    assert not (planner.coordinates_dp_verify and planner._dp_tier_gather_enabled)
    assert planner._dynamic_graph_tier == (
        mode == "compact" and (not dp or hook_kind == "glm")
    )


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
