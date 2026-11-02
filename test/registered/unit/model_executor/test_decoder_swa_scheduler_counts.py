"""Decoder tail sizes reuse the scheduler exchange on eager CUDA DP steps."""

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
import torch
from sglang.srt.managers.scheduler_components import dp_attn
from sglang.srt.managers.scheduler_components.dp_attn import _update_gather_batch
from sglang.srt.model_executor import forward_batch_info as fbi
from sglang.srt.model_executor.cuda_graph_config import Backend
from sglang.srt.model_executor.encoder_swa_replay import decoder_swa_trim_rows
from sglang.srt.model_executor.forward_batch_info import ForwardBatch, ForwardMode
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


@pytest.mark.parametrize("folds,expected", [(False, 3968), (True, 3998)])
def test_trim_includes_folded_encoder_rows(folds, expected):
    batch = SimpleNamespace(
        forward_mode=ForwardMode.EXTEND,
        extend_lens=[4096, 30, 40],
        prefix_lens=[0, 512, 64],
        encoder_swa_reset=[True, True, True],
        return_logprob=False,
        return_hidden_states=False,
    )
    assert decoder_swa_trim_rows(batch, folds_encoder=folds) == expected
    batch.return_logprob = True
    assert decoder_swa_trim_rows(batch, folds_encoder=folds) == -1
    batch.return_logprob = False
    batch.return_hidden_states = True
    assert decoder_swa_trim_rows(batch, folds_encoder=folds) == -1


@pytest.mark.parametrize(
    "mode", [ForwardMode.DECODE, ForwardMode.IDLE, ForwardMode.TARGET_VERIFY]
)
def test_non_extend_peers_do_not_trim(mode):
    assert (
        decoder_swa_trim_rows(SimpleNamespace(forward_mode=mode), folds_encoder=True)
        == 0
    )


@pytest.mark.parametrize(
    "trims,expected",
    [
        ([3968, 30, 0, 0], [3968, 30, 0, 0]),
        ([0, 0, 0, 0], None),
        ([3968, -1, 0, 0], None),
    ],
)
def test_whole_output_peer_falls_back_on_every_rank(trims, expected):
    info = MagicMock()
    info.global_decoder_trim_rows = trims
    batch = SimpleNamespace()
    _update_gather_batch(batch, info, True, skip_global_metadata=True)
    assert batch.global_decoder_trim_rows == expected


@pytest.mark.parametrize(
    "rank,mode",
    [
        (0, ForwardMode.EXTEND),
        (1, ForwardMode.EXTEND),
        (2, ForwardMode.IDLE),
        (3, ForwardMode.DECODE),
    ],
)
@pytest.mark.parametrize("fallback", [None, "graph", "hip", "whole_outputs"])
def test_preparation_uses_scheduler_counts_or_preserves_backend_exchange(
    rank, mode, fallback
):
    rows, trims, late = [4096, 282, 0, 130], [3968, 30, 0, 0], [128, 252, 0, 130]
    n = rows[rank]
    batch = ForwardBatch(
        forward_mode=mode,
        batch_size=1 if n else 0,
        input_ids=torch.zeros(n, dtype=torch.long),
        req_pool_indices=torch.zeros(1 if n else 0, dtype=torch.long),
        seq_lens=torch.tensor([n] if n else []),
        out_cache_loc=torch.arange(n),
        seq_lens_sum=n,
        positions=torch.arange(n),
        is_extend_in_batch=True,
        global_num_tokens_cpu=rows,
        global_num_tokens_for_logprob_cpu=rows,
        global_num_tokens_gpu=torch.zeros(4, dtype=torch.int64),
        global_decoder_trim_rows_cpu=None if fallback == "whole_outputs" else trims,
    )
    runner = MagicMock(is_draft_worker=False)
    runner.attn_tp_sequence_sharded.return_value = False
    runner.attn_backend.late_layer_num_tokens.return_value = late[rank]
    cfg = SimpleNamespace(
        graph=SimpleNamespace(
            cuda_graph_config=SimpleNamespace(
                prefill=SimpleNamespace(
                    bs=[],
                    backend=Backend.BREAKABLE
                    if fallback == "graph"
                    else Backend.DISABLED,
                )
            )
        )
    )

    def exchange(output, local, **kwargs):
        assert local.item() == late[rank]
        output.copy_(torch.tensor(late))

    with (
        patch.multiple(
            fbi,
            get_parallel=lambda: SimpleNamespace(
                attn_tp_size=1, tp_group=SimpleNamespace(cpu_group=None)
            ),
            get_exec=lambda: cfg,
            get_platform=lambda: SimpleNamespace(is_cuda=fallback != "hip"),
            _elastic_should_preserve_local_token_counts=lambda **kwargs: False,
            _mega_moe_materializes_idle_rank=lambda batch: False,
            dp_slot_in=lambda counts: rank,
            set_dp_buffer_len_from_batch=lambda *args: None,
            set_is_extend_in_batch=lambda *args: None,
            _is_cpu=True,
        ),
        patch.object(fbi, "mambaish_config", return_value=None, create=True),
        patch.object(
            fbi.DpPaddingMode,
            "get_dp_padding_mode",
            return_value=fbi.DpPaddingMode.SUM_LEN,
        ),
        patch.object(ForwardBatch, "_pad_inputs_to_size"),
        patch(
            "sglang.srt.batch_overlap.two_batch_overlap.TboForwardBatchPreparer.prepare"
        ),
        patch.object(fbi, "all_gather_single", side_effect=exchange) as gather,
    ):
        batch.prepare_mlp_sync_batch(runner)
    assert batch.global_late_layer_num_tokens_cpu == late
    assert gather.call_count == (0 if fallback is None else 1)


def test_scheduler_exchange_carries_peer_trims_and_inactive_slots():
    infos = [
        dp_attn.MLPSyncBatchInfo(
            num_dp_ranks=4,
            tp_size=1,
            cp_size=1,
            num_tokens=n,
            num_tokens_for_logprob=n,
            can_run_decode_cuda_graph=False,
            can_run_draft_cuda_graph=False,
            can_run_prefill_cuda_graph=False,
            is_extend_in_batch=True,
            local_can_run_tbo=False,
            local_forward_mode=ForwardMode.EXTEND.value,
            decoder_trim_rows=t,
        )
        for n, t in zip([4096, 282, 0, 130], [3968, 30, 0, 0])
    ]
    gathered = torch.cat([info._get_local_tensor("cpu") for info in infos])
    with (
        patch.object(
            dp_attn,
            "exchange_sync_rows",
            side_effect=lambda out, *a, **kw: out.copy_(gathered),
        ),
        patch.object(
            dp_attn,
            "get_parallel",
            return_value=SimpleNamespace(
                tp_group=SimpleNamespace(active_ranks_cpu=torch.tensor([1, 1, 0, 1]))
            ),
        ),
    ):
        for info in infos:
            info.all_gather(device="cpu", group=None)
            assert info.global_decoder_trim_rows == [3968, 30, 0, 0]
            assert info.global_num_tokens == [4096, 282, 0, 130]
