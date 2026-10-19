from types import SimpleNamespace
from unittest.mock import patch

import pytest
import sglang.srt.arg_groups.speculative_hook as speculative_hook
import torch
from sglang.srt.speculative.dlite_worker_v2 import DLiteWorkerV2, _commit_accept
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


def test_commit_accept_shifts_candidates_and_inserts_bonus():
    candidates = torch.tensor(
        [
            [10, 11, 12, 13],
            [20, 21, 22, 23],
            [30, 31, 32, 33],
        ]
    )
    accept_len = torch.tensor([0, 2, 3], dtype=torch.int32)
    bonus = torch.tensor([100, 101, 102])

    out_tokens, commit_lens = _commit_accept(candidates, accept_len, bonus)

    torch.testing.assert_close(
        out_tokens,
        torch.tensor(
            [
                [100, 12, 13, 0],
                [21, 22, 101, 0],
                [31, 32, 33, 102],
            ]
        ),
        rtol=0,
        atol=0,
    )
    torch.testing.assert_close(
        commit_lens,
        torch.tensor([1, 3, 4], dtype=torch.int32),
        rtol=0,
        atol=0,
    )


def test_relay_state_preserves_pivot_hidden_states():
    bonus = torch.tensor([7, 8])
    seq_lens = torch.tensor([10, 20], dtype=torch.int32)
    hidden_states = torch.randn(2, 12)

    state = DLiteWorkerV2._state(
        bonus_tokens=bonus,
        new_seq_lens=seq_lens,
        hidden_states=hidden_states,
    )

    assert state.hidden_states is hidden_states
    assert state.bonus_tokens.dtype == torch.int64
    assert state.new_seq_lens.dtype == torch.int64


def test_rejects_non_greedy_request():
    batch = SimpleNamespace(
        sampling_info=SimpleNamespace(is_all_greedy=False),
    )
    with pytest.raises(ValueError, match="greedy decoding only"):
        DLiteWorkerV2._validate_request_sampling(None, batch)


def test_server_args_reject_non_nvidia_cuda_platform():
    cfg = SimpleNamespace(device="cuda")
    platform = SimpleNamespace(is_cuda=lambda: False)
    with (
        patch.object(speculative_hook, "resolving_view", return_value=cfg),
        patch.object(speculative_hook, "current_platform", platform),
        pytest.raises(ValueError, match="supports CUDA only"),
    ):
        speculative_hook._handle_dlite(SimpleNamespace())


def test_server_args_reject_mixed_chunk():
    cfg = SimpleNamespace(
        device="cuda",
        speculative_draft_model_path="/draft",
        pp_size=1,
        attn_cp_size=1,
        disaggregation_mode="null",
        enable_two_batch_overlap=False,
        enable_mixed_chunk=True,
    )
    platform = SimpleNamespace(is_cuda=lambda: True)
    with (
        patch.object(speculative_hook, "resolving_view", return_value=cfg),
        patch.object(speculative_hook, "current_platform", platform),
        patch.object(speculative_hook, "attn_dp_enabled_of", return_value=False),
        pytest.raises(ValueError, match="mixed chunked prefill"),
    ):
        speculative_hook._handle_dlite(SimpleNamespace())
