"""Eager prefill and graph-eligible DSPARK decode within one mixed step."""

from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest
import torch

from sglang.srt.managers.schedule_batch import ScheduleBatch
from sglang.srt.managers.utils import GenerationBatchResult
from sglang.srt.model_executor.cuda_graph_config import Backend
from sglang.srt.model_executor.forward_batch_info import DSV4OutCacheLoc, ForwardBatch
from sglang.srt.model_executor.forward_mode import ForwardMode
from sglang.srt.model_executor.runner.decode_cuda_graph_runner import (
    DecodeCudaGraphRunner,
)
from sglang.srt.speculative.dp_spec_prefill_coordination import (
    DPSpecPrefillCoordinationPlan,
)
from sglang.srt.speculative.dspark_components.dspark_draft import make_next_draft_input
from sglang.srt.speculative.dspark_components.dspark_worker_v2 import DSparkWorkerV2
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=6, suite="base-a-test-cpu")

MODULE = "sglang.srt.speculative.dspark_components.dspark_worker_v2"


def _plan():
    # Mixed (5 prefill tokens + 2 * W), decode, prefill, idle.
    return DPSpecPrefillCoordinationPlan(
        [17, 3, 7, 0],
        [13, 3, 1, 0],
        torch.tensor([1, 0, 1, 0]),
        5,
        6,
        mixed_decode_counts=[2, 0, 0, 0],
    )


def _batch(rank, local_counts=False):
    mode, n_pre, n_dec, pre_tokens = (
        (ForwardMode.MIXED, 1, 2, 5),
        (ForwardMode.DECODE, 0, 3, 0),
        (ForwardMode.EXTEND, 1, 0, 7),
        (ForwardMode.IDLE, 0, 0, 0),
    )[rank]
    bs = n_pre + n_dec
    seq_lens = torch.arange(128, 128 + bs)
    reqs = [SimpleNamespace(grammar=None, return_logprob=False) for _ in range(bs)]
    batch = ScheduleBatch(
        reqs=reqs,
        device="cpu",
        forward_mode=mode,
        num_prefill_rows=n_pre if mode.is_mixed() else None,
        req_pool_indices=torch.arange(bs),
        req_pool_indices_cpu=torch.arange(bs),
        seq_lens=seq_lens,
        seq_lens_cpu=seq_lens.clone(),
        orig_seq_lens=seq_lens.to(torch.int32),
        input_ids=torch.arange(pre_tokens) if n_pre else None,
        out_cache_loc=torch.arange(pre_tokens) + 100 if n_pre else None,
        extend_lens=([pre_tokens] if n_pre else []) + [6] * n_dec,
        prefix_lens=([128 - pre_tokens] if n_pre else []) + seq_lens[n_pre:].tolist(),
        extend_logprob_start_lens=([pre_tokens - 1] if n_pre else []) + [0] * n_dec,
        extend_num_tokens=pre_tokens + 6 * n_dec,
        multimodal_inputs=list(range(bs)),
        top_logprobs_nums=list(range(bs)),
        global_num_tokens=[_plan().counts[rank]] if local_counts else _plan().counts,
        is_extend_in_batch=True,
        # Original mixed step rejects graphs; the split decode must replace it.
        can_run_decode_cuda_graph=False,
    )
    if n_dec:
        batch.spec_info = make_next_draft_input(
            bonus_tokens=torch.arange(n_dec) + 200,
            new_seq_lens=seq_lens[n_pre:],
        )
    return batch


@pytest.mark.parametrize("local_only", [False, True])
@pytest.mark.parametrize("graph_enabled", [False, True])
def test_phase_counts_and_graph_gate(local_only, graph_enabled):
    plan = _plan()
    assert plan.phase_counts("prefill") == ([5, 0, 7, 0], [1, 0, 1, 0])
    assert plan.phase_counts("decode") == ([2, 3, 0, 0], [2, 3, 0, 0])
    for rank in range(4):
        batch = SimpleNamespace()
        plan.apply(batch, "prefill", rank, local_only=local_only)
        assert batch.global_num_tokens == (
            [[5, 0, 7, 0][rank]] if local_only else [5, 0, 7, 0]
        )
        assert batch.is_extend_in_batch
        assert not batch.can_run_decode_cuda_graph
        plan.apply(
            batch,
            "decode",
            rank,
            local_only=local_only,
            decode_graph_allowed=graph_enabled,
        )
        assert batch.global_num_tokens == (
            [[2, 3, 0, 0][rank]] if local_only else [2, 3, 0, 0]
        )
        assert not batch.is_extend_in_batch
        assert not batch.dp_spec_prefill_coordination_applied
        assert batch.can_run_decode_cuda_graph is graph_enabled


@pytest.mark.parametrize("rank", range(4))
@pytest.mark.parametrize("local_counts", [False, True])
@pytest.mark.parametrize("graph_enabled", [False, True])
def test_all_ranks_run_prefill_then_normal_decode(rank, local_counts, graph_enabled):
    batch = _batch(rank, local_counts)
    original_reqs = list(batch.reqs)
    original_lens = batch.seq_lens.clone()
    worker = object.__new__(DSparkWorkerV2)
    worker.device = "cpu"
    worker.verify_num_draft_tokens = 6
    worker._decode_graph_allowed = graph_enabled
    worker._verify_planner = Mock()
    worker._observers = Mock()
    worker.model_runner = SimpleNamespace(shared_read_done_event=None)
    events = []
    pre_tokens = None
    pre_lens = None
    dec_tokens = None
    dec_lens = None

    def prefill(view, publish):
        nonlocal pre_tokens, pre_lens
        assert publish is None
        assert view.forward_mode in (ForwardMode.EXTEND, ForwardMode.IDLE)
        assert not view.can_run_decode_cuda_graph
        assert view.is_extend_in_batch
        assert view.input_ids.numel() == view.extend_num_tokens
        assert view.input_ids.numel() == ([5, 0, 7, 0][rank])
        assert view.out_cache_loc.numel() == view.input_ids.numel()
        pre_tokens = torch.arange(len(view.reqs)) + 10
        pre_lens = view.seq_lens.clone()
        worker.model_runner.shared_read_done_event = "prefill read-done"
        events.append("prefill")
        return GenerationBatchResult(
            next_token_ids=pre_tokens,
            new_seq_lens=pre_lens,
            can_run_cuda_graph=False,
        )

    def decode(view, publish, barrier):
        nonlocal dec_tokens, dec_lens
        assert publish is None
        assert barrier == "grammar barrier"
        assert view.forward_mode in (ForwardMode.DECODE, ForwardMode.IDLE)
        assert view.can_run_decode_cuda_graph is graph_enabled
        assert not view.is_extend_in_batch
        assert not view.dp_spec_prefill_coordination_applied
        assert worker.model_runner.shared_read_done_event is None
        assert len(view.reqs) == [2, 3, 0, 0][rank]
        assert view.seq_lens.numel() == len(view.reqs)
        assert view.multimodal_inputs == list(range(len(batch.reqs)))[len(pre_tokens) :]
        # Exercise the lifetime boundary: decode can overwrite target scratch.
        pre_tokens.fill_(-1)
        dec_tokens = torch.arange(len(view.reqs) * 6) + 20
        dec_lens = view.seq_lens + 2
        events.append("decode")
        return GenerationBatchResult(
            next_token_ids=dec_tokens,
            new_seq_lens=dec_lens,
            accept_lens=torch.full((len(view.reqs),), 2, dtype=torch.int32),
            block_accept_lens=torch.full((len(view.reqs),), 2, dtype=torch.int32),
            can_run_cuda_graph=graph_enabled and bool(view.reqs),
            next_draft_input=make_next_draft_input(
                bonus_tokens=torch.arange(len(view.reqs)) + 300,
                new_seq_lens=dec_lens,
            ),
        )

    worker._forward_prefill = prefill
    worker._forward_decode = decode
    published = Mock()
    with patch(
        f"{MODULE}.get_parallel", return_value=SimpleNamespace(attn_dp_rank=rank)
    ):
        result = worker._forward_split_mixed(
            batch, published, "grammar barrier", _plan()
        )
    assert events == ["prefill", "decode"]
    expected_pre = torch.arange(len(pre_tokens)) + 10
    expected_tokens = (
        torch.cat([expected_pre, dec_tokens])
        if rank == 0
        else expected_pre
        if rank == 2
        else dec_tokens
    )
    torch.testing.assert_close(result.next_token_ids, expected_tokens)
    torch.testing.assert_close(result.new_seq_lens, torch.cat([pre_lens, dec_lens]))
    torch.testing.assert_close(
        result.next_draft_input.bonus_tokens,
        torch.cat([expected_pre, torch.arange(len(dec_lens)) + 300]),
    )
    assert published.call_count == 1
    assert published.call_args.args[0] is result.new_seq_lens
    if rank == 0:
        assert result.can_run_cuda_graph is False
        assert result.mixed_decode_can_run_cuda_graph is graph_enabled
        assert result.accept_lens.tolist() == [2, 2]
    assert batch.reqs == original_reqs
    torch.testing.assert_close(batch.seq_lens, original_lens)
    assert batch.can_run_decode_cuda_graph is False


def test_no_dp_split_clears_mixed_global_counts_and_keeps_running_spec_state():
    batch = _batch(0)
    batch.ne_token_table = torch.zeros((16, 1024), dtype=torch.int32)
    worker = object.__new__(DSparkWorkerV2)
    worker.device = "cpu"
    worker.verify_num_draft_tokens = 6
    worker._decode_graph_allowed = True
    view = worker._split_mixed_view(batch, 1, 3, prefill=False)
    worker._apply_split_phase(view, "decode", None, batch)
    assert view.global_num_tokens is None
    assert view.spec_info is batch.spec_info
    assert view.extend_lens == [6, 6]
    assert view.req_pool_indices.tolist() == [1, 2]
    assert view.top_logprobs_nums == [1, 2]
    assert view.ne_token_table is batch.ne_token_table
    assert view.can_run_decode_cuda_graph
    assert view.global_forward_mode == ForwardMode.DECODE
    assert batch.global_num_tokens == [17, 3, 7, 0]


def test_prefill_compressed_writes_exclude_running_rows_without_d2h():
    batch = _batch(0)
    batch.prefix_lens = [127, 129, 130]
    batch.extend_lens = [129, 6, 6]
    batch.extend_logprob_start_lens = [128, 0, 0]
    batch.input_ids = torch.arange(129)
    batch.out_cache_loc = torch.arange(127, 256) + 1024
    full = batch.out_cache_loc
    batch.out_cache_loc_dsv4 = DSV4OutCacheLoc(full, full, full, full)
    batch.req_to_token_pool = SimpleNamespace(
        req_to_c128_sidecar=torch.tensor([[10, 11], [20, 21], [30, 31]]),
        c128_page_size=2,
    )
    batch.token_to_kv_pool_allocator = SimpleNamespace(
        translate_loc_from_full_to_swa=lambda loc: loc + 8192
    )
    worker = object.__new__(DSparkWorkerV2)
    worker.device = "cpu"
    worker.verify_num_draft_tokens = 6
    with patch.object(
        torch.Tensor, "cpu", side_effect=AssertionError("unexpected D2H")
    ):
        pre = worker._split_mixed_view(batch, 0, 1, prefill=True)
    torch.testing.assert_close(pre.out_cache_loc_dsv4.out_full_loc, full)
    assert pre.out_cache_loc_dsv4.out_c4_loc.tolist() == list(range(287, 320))
    assert pre.out_cache_loc_dsv4.out_c128_loc.tolist() == [20, 21]
    assert pre.extend_lens == [129]
    assert batch.extend_lens == [129, 6, 6]


@pytest.mark.parametrize("rank", range(4))
@pytest.mark.parametrize("backend", Backend.ALL)
@pytest.mark.parametrize("disagg_mode", ["null", "prefill"])
def test_decode_backend_controls_mixed_rank_and_all_its_peers(
    rank, backend, disagg_mode
):
    # Stop before loading draft weights; exercise the real configuration setup.
    parallel = SimpleNamespace(
        attn_dp_enabled=True, pp_group=SimpleNamespace(is_last_rank=False)
    )
    with (
        patch(f"{MODULE}.get_schedule", return_value=SimpleNamespace(page_size=128)),
        patch(f"{MODULE}.get_parallel", return_value=parallel),
        patch(
            f"{MODULE}.get_disagg",
            return_value=SimpleNamespace(disaggregation_mode=disagg_mode),
        ),
        patch(
            f"{MODULE}.get_exec",
            return_value=SimpleNamespace(
                graph=SimpleNamespace(
                    cuda_graph_config=SimpleNamespace(
                        decode=SimpleNamespace(backend=backend)
                    )
                )
            ),
        ),
        patch(f"{MODULE}.verify_merged_mixed_enabled", return_value=True),
    ):
        worker = DSparkWorkerV2(
            SimpleNamespace(),
            0,
            0,
            SimpleNamespace(device="cpu", model_runner=SimpleNamespace()),
        )
    worker._hosts_draft = True
    worker.verify_num_draft_tokens = 6
    worker._proposer = SimpleNamespace(query_token_num=5)
    worker._verify_planner = Mock()
    worker._observers = Mock()
    worker._forward_split_mixed = Mock(return_value="split")
    worker._forward_mixed = Mock(return_value="merged")
    worker._forward_dp_spec_prefill_coordination = Mock(return_value="coordinated")
    batch = _batch(rank)
    plan = _plan()
    batch.dp_spec_prefill_coordination_metadata = (
        plan.counts,
        plan.logprob_counts,
        plan.prefills,
    )
    batch.dp_spec_prefill_mixed_decode_counts = plan.mixed_decode_counts
    expected = (
        "split"
        if backend != Backend.DISABLED and disagg_mode != "prefill"
        else "merged"
        if rank == 0
        else "coordinated"
    )
    assert worker.forward_batch_generation(batch) == expected
    if expected == "split":
        actual = worker._forward_split_mixed.call_args.args[3]
    else:
        worker._forward_split_mixed.assert_not_called()
        actual = (
            worker._forward_mixed.call_args.kwargs["coordination_plan"]
            if rank == 0
            else worker._forward_dp_spec_prefill_coordination.call_args.args[1]
        )
    assert actual.phase_counts("decode") == ([2, 3, 0, 0],) * 2


@pytest.mark.parametrize("rank", range(4))
def test_decode_metadata_scales_once_and_passes_real_graph_admission(rank):
    plan = _plan()
    view = _batch(rank)
    plan.apply(view, "decode", rank, local_only=False, decode_graph_allowed=True)
    view.can_run_dp_draft_cuda_graph = True
    forward = ForwardBatch(
        forward_mode=ForwardMode.TARGET_VERIFY,
        batch_size=[2, 3, 0, 0][rank],
        input_ids=torch.empty(0, dtype=torch.int64),
        req_pool_indices=torch.empty(0, dtype=torch.int64),
        seq_lens=torch.empty(0, dtype=torch.int64),
        out_cache_loc=torch.empty(0, dtype=torch.int64),
        seq_lens_sum=0,
        spec_info=SimpleNamespace(
            num_tokens_per_req=6, num_tokens_for_logprob_per_req=6
        ),
    )
    forward.init_mlp_sync_metadata(view, "cpu")
    assert forward.global_num_tokens_cpu == [12, 18, 0, 0]
    assert forward.global_num_tokens_for_logprob_cpu == [12, 18, 0, 0]
    assert forward.original_global_num_tokens_cpu == [2, 3, 0, 0]

    runner = object.__new__(DecodeCudaGraphRunner)
    runner.ragged_verify_mode = False
    runner.captured_req_width = 6
    runner.require_mlp_tp_gather = True
    runner.require_mlp_sync = True
    runner.disable_padding = False
    runner.max_bs = 4
    runner.is_encoder_decoder = False
    runner.enable_two_batch_overlap = False
    runner.model_runner = SimpleNamespace(
        spec_algorithm=SimpleNamespace(is_ngram=lambda: False)
    )
    runner._with_logits_variant = Mock()
    runner._make_graph_key = Mock()
    runner._resolve_lora_variant = Mock(return_value=None)
    runner._resolve_logits_variant = Mock(return_value=None)
    # Includes idle ranks: all ranks choose the same bucket from raw requests.
    assert runner.can_run_graph(forward)
    runner.max_bs = 2
    assert not runner.can_run_graph(forward)
    runner.max_bs = 4
    forward.can_run_decode_cuda_graph = False
    assert not runner.can_run_graph(forward)
