"""Exercise GLM compact DP coordination using the scheduler's existing Gloo exchange."""

from datetime import timedelta
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from sglang.srt.environ import envs
from sglang.srt.managers.scheduler_components import dp_attn
from sglang.srt.model_executor import forward_batch_info as fbi
from sglang.srt.model_executor.runner.decode_cuda_graph_runner import (
    DecodeCudaGraphRunner,
)
from sglang.srt.speculative.dspark_components import dspark_planner, dspark_worker_v2
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=20, suite="base-a-test-cpu")


def _worker(*, graph=True):
    worker = dspark_worker_v2.DSparkWorkerV2.__new__(dspark_worker_v2.DSparkWorkerV2)
    worker.device = torch.device("cpu")
    worker.verify_num_draft_tokens = 9
    worker._compact_attention_dp = True
    worker.model_runner = SimpleNamespace(
        device=worker.device,
        attn_backend=SimpleNamespace(needs_cpu_seq_lens=True),
        decode_cuda_graph_runner=(
            SimpleNamespace(
                ragged_verify_mode=True,
                capture_hidden_mode=fbi.CaptureHiddenMode.FULL,
                attn_backend=SimpleNamespace(supports_ragged_verify_graph=True),
                max_bs=8,
                capture_num_tokens=[32, 64],
            )
            if graph
            else None
        ),
    )
    worker._verify_planner = planner = dspark_planner.DSparkVerifyPlanner.__new__(
        dspark_planner.DSparkVerifyPlanner
    )
    planner.model_runner = worker.model_runner
    planner.verify_num_draft_tokens = 9
    planner._ragged_verify_mode = dspark_planner.RaggedVerifyMode.COMPACT
    planner._dynamic_graph_tier = False
    planner._align_verify_tokens_to_graph_tier = False
    planner._is_verify_all = False
    return worker


def _sync_info(bs, *, counts=None, prefill=False):
    mode = (
        fbi.ForwardMode.EXTEND
        if prefill
        else (fbi.ForwardMode.DECODE if bs else fbi.ForwardMode.IDLE)
    )
    return dp_attn.MLPSyncBatchInfo(
        num_dp_ranks=2,
        tp_size=2,
        cp_size=1,
        num_tokens=bs,
        num_tokens_for_logprob=bs,
        can_run_decode_cuda_graph=not prefill,
        can_run_draft_cuda_graph=True,
        can_run_prefill_cuda_graph=False,
        is_extend_in_batch=prefill,
        local_can_run_tbo=False,
        local_forward_mode=mode.value,
        global_num_tokens=counts,
    )


def _sync_batch(info, *, require_mlp_tp_gather=True, skip_global_metadata=False):
    batch = SimpleNamespace(forward_mode=fbi.ForwardMode(info.local_forward_mode))
    with envs.SGLANG_ENABLE_DP_SPEC_PREFILL_COORDINATION.override(False):
        dp_attn._update_gather_batch(
            batch,
            info,
            require_mlp_tp_gather=require_mlp_tp_gather,
            skip_global_metadata=skip_global_metadata,
        )
    return batch


def _busy_idle_worker(rank, init_path):
    dist.init_process_group(
        "gloo",
        init_method="file://" + init_path,
        rank=rank,
        world_size=4,
        timeout=timedelta(seconds=30),
    )
    try:
        parallel = SimpleNamespace(
            tp_group=SimpleNamespace(active_ranks_cpu=torch.ones(4, dtype=torch.bool))
        )
        worker = _worker()
        # Same TP pair, independent request counts per DP group. Change which
        # side is idle, then exceed the captured grid and mix in a prefilling peer.
        for counts, prefill_group, expected in (
            ([5, 2], None, 64),
            ([5, 0], None, 64),
            ([0, 2], None, 32),
            ([8, 0], None, None),
            ([6, 2], 0, None),
        ):
            local_bs = counts[rank // 2]
            prefill = rank // 2 == prefill_group
            info = _sync_info(local_bs, prefill=prefill)
            with patch.object(dp_attn, "get_parallel", return_value=parallel):
                info.all_gather("cpu", dist.group.WORLD)
            batch = _sync_batch(info)
            assert batch.global_num_tokens == counts
            if prefill:
                continue
            if local_bs:
                num_reqs = worker._compact_dp_verify_num_reqs(batch)
                if expected is None:
                    assert num_reqs is None
                else:
                    lens = torch.tensor(([4, 2] * local_bs)[:local_bs])
                    planner = worker._verify_planner
                    with patch.object(
                        planner, "_schedule_verify_lens", return_value=lens
                    ):
                        layout = planner.schedule_layout(
                            req_pool_indices=torch.arange(local_bs),
                            prefix_lens=torch.full((local_bs,), 32),
                            device=worker.device,
                            confidence=None,
                            budget=int(lens.sum()),
                            global_num_reqs=num_reqs,
                        )
                    assert layout.graph_num_tokens == expected
                    assert layout.verify_lens.tolist() == lens.tolist()
            else:
                layout = worker._idle_verify_ragged_layout(batch)
                if expected is None:
                    assert layout is None
                else:
                    assert layout.graph_num_tokens == expected
                    assert layout.verify_lens.tolist() == [0]
            assert batch.can_run_decode_cuda_graph == (expected is not None)
            assert batch.can_run_dp_draft_cuda_graph

            batch = _sync_batch(info, require_mlp_tp_gather=False)
            assert batch.global_num_tokens == [local_bs]
            assert batch.can_run_decode_cuda_graph == (prefill_group is None)
            if local_bs:
                assert worker._compact_dp_verify_num_reqs(batch) is None
            else:
                assert worker._idle_verify_ragged_layout(batch) is None
            assert not batch.can_run_decode_cuda_graph
            assert batch.can_run_dp_draft_cuda_graph
    finally:
        dist.destroy_process_group()


def test_busy_and_idle_reuse_dp_scheduler_exchange(tmp_path):
    mp.spawn(
        _busy_idle_worker, args=(str(tmp_path / "gloo_init"),), nprocs=4, join=True
    )


@pytest.mark.parametrize("bs", [0, 2])
def test_no_common_tier_blocks_busy_and_idle_graph_replay(bs):
    worker = _worker()
    # This rank's scheduler was configured to skip the DP exchange.
    batch = _sync_batch(_sync_info(bs), skip_global_metadata=True)
    assert worker._compact_dp_verify_num_reqs(batch) is None
    runner = Mock(
        ragged_verify_mode=True,
        captured_req_width=9,
        require_mlp_tp_gather=False,
        max_bs=8,
        disable_padding=False,
        require_mlp_sync=True,
        is_encoder_decoder=False,
        enable_two_batch_overlap=False,
    )
    runner.model_runner.spec_algorithm.is_ngram.return_value = False
    fb = SimpleNamespace(
        forward_mode=fbi.ForwardMode.TARGET_VERIFY if bs else fbi.ForwardMode.IDLE,
        batch_size=bs,
        replace_embeds=None,
        ragged_verify_layout=None,
        beam=None,
        spec_info=SimpleNamespace(num_tokens_per_req=9),
        can_run_decode_cuda_graph=batch.can_run_decode_cuda_graph,
    )
    assert not DecodeCudaGraphRunner.can_run_graph(runner, fb)


@pytest.mark.parametrize("bs", [0, 2])
def test_no_ragged_graph_uses_uniform_verify_inputs(bs):
    from sglang.srt.speculative.dflash_info import DFlashVerifyInput
    from sglang.srt.speculative.dspark_components.dspark_verify import (
        TargetVerifyExecutor,
    )

    worker = _worker(graph=False)
    batch = _sync_batch(_sync_info(bs, counts=[2, 0]))
    batch.seq_lens_cpu = torch.arange(bs)
    batch.seq_lens_sum = int(batch.seq_lens_cpu.sum())
    layout = worker._idle_verify_ragged_layout(batch) if not bs else None
    if bs:
        assert worker._compact_dp_verify_num_reqs(batch) is None
    assert layout is None
    assert not batch.can_run_decode_cuda_graph
    executor = TargetVerifyExecutor.__new__(TargetVerifyExecutor)
    executor.model_runner = worker.model_runner
    executor.target_worker = Mock(model_runner=worker.model_runner)
    executor.verify_epilogue = None
    executor.verify_num_draft_tokens = 9
    executor._verify_backend_self_adds_seq_lens_cache = False
    ids = torch.arange(bs * 9).reshape(bs, 9)

    def prepare(spec, schedule_batch, target_worker):
        assert schedule_batch is batch
        assert spec.ragged_verify_layout is None
        assert spec.draft_token_num == 9
        torch.testing.assert_close(spec.draft_token, ids.flatten())
        return Mock(), False

    with patch.object(DFlashVerifyInput, "prepare_for_verify", prepare):
        if bs:
            executor.run_non_compact(
                batch=batch,
                draft_input=Mock(),
                verify_ids_2d=ids,
                verify_window=SimpleNamespace(
                    positions_2d=ids, verify_cache_loc=ids.flatten(), iteration_ids=None
                ),
            )
        else:
            executor.run_idle_participation(batch=batch, idle_layout=layout)
    executor.target_worker.forward_batch_generation.assert_called_once()
    assert batch.forward_mode == (
        fbi.ForwardMode.DECODE if bs else fbi.ForwardMode.IDLE
    )


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


def test_selected_path_rejects_incompatible_backend_before_draft_init():
    worker = dspark_worker_v2.DSparkWorkerV2.__new__(dspark_worker_v2.DSparkWorkerV2)
    worker._hosts_draft = True
    worker._uses_ascend_glm_dsa = True
    worker._verify_planner = SimpleNamespace(mode_value="compact")
    worker.model_runner = SimpleNamespace(attn_backend=object())
    worker._draft_worker = Mock()
    with pytest.raises(ValueError, match="GLM DSA ragged metadata support"):
        worker.init_attention_backends()
    worker._draft_worker.init_attention_backends.assert_not_called()
