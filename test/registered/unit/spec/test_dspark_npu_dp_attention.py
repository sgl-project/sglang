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
from sglang.srt.hardware_backend.npu import dspark_dp
from sglang.srt.hardware_backend.npu.attention.dspark_compact import (
    NpuCompactGraphMetadata,
)
from sglang.srt.model_executor import forward_batch_info as fbi
from sglang.srt.speculative.dspark_components import dspark_planner
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
    assert (
        dspark_dp.compact_dp_graph_tokens(
            contributions,
            attn_tp_size=attn_tp,
            runner=_runner().decode_cuda_graph_runner,
        )
        == expected
    )


def test_eager_alignment_does_not_infer_request_count_from_tokens():
    assert (
        dspark_dp.compact_dp_graph_tokens([(2, 5), (0, 0)], attn_tp_size=4, runner=None)
        == 8
    )


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
        dspark_dp.prepare_compact_dp_forward_batch(batch, runner)
        assert batch.global_num_tokens_cpu == [8, 8]
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


def test_idle_glm_metadata_uses_only_sink_page():
    state = NpuCompactGraphMetadata(
        num_slots=4, num_tokens=8, table_width=8, device="cpu"
    )
    state.update(
        prefix_lens=torch.zeros(1, dtype=torch.long),
        req_pool_indices=torch.zeros(1, dtype=torch.long),
        req_to_token=torch.full((2, 8), 99, dtype=torch.long),
        verify_lens=torch.zeros(1, dtype=torch.int32),
        page_size=1,
    )
    assert state.query_ends.tolist() == [0, 0, 0, 0, 8]
    assert state.kv_lens.tolist() == [0, 0, 0, 0, 8]
    assert torch.count_nonzero(state.block_tables) == 0
    # Busy replay after idle must replace all previous metadata.
    state.update(
        prefix_lens=torch.tensor([3, 2]),
        req_pool_indices=torch.tensor([0, 1]),
        req_to_token=torch.arange(16).view(2, 8),
        verify_lens=torch.tensor([2, 1]),
        page_size=1,
    )
    assert state.query_ends.tolist() == [2, 3, 3, 3, 8]
    assert state.kv_lens.tolist() == [5, 3, 0, 0, 5]


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

        planner = dspark_planner.DSparkVerifyPlanner.__new__(
            dspark_planner.DSparkVerifyPlanner
        )
        planner.model_runner = runner
        planner.model_runner.req_to_token_pool = SimpleNamespace(
            req_generation=torch.zeros(2, dtype=torch.long)
        )
        planner._budget_planner = Mock()
        planner._budget_from_resolved = Mock(
            return_value=(2 if rank < 2 else 6) + rank % 2
        )
        with (
            patch.object(dspark_planner, "_is_npu", True),
            patch.object(
                dspark_planner,
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
            with patch.object(dspark_dp, "get_parallel", return_value=parallel):
                result = dspark_dp.coordinate_compact_dp_layout(
                    batch=batch, layout=layout, model_runner=runner
                )
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


def test_non_glm_backend_does_not_rewrite_dp_counts():
    runner = _runner()
    runner.attn_backend._dspark_dsa_ragged_verify = False
    batch = SimpleNamespace(global_num_tokens_cpu=[9, 0])
    with patch.object(
        dspark_dp, "get_parallel", return_value=SimpleNamespace(attn_dp_enabled=True)
    ):
        dspark_dp.prepare_compact_dp_forward_batch(batch, runner)
    assert batch.global_num_tokens_cpu == [9, 0]


@pytest.mark.parametrize("graph", [False, True])
def test_production_glm_dspark_argument_hook_accepts_attention_dp(graph):
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
    args.tp_size = 16
    args.attn_dp_size = 4
    args.enable_dp_lm_head = True
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
        _handle_dspark(args)
