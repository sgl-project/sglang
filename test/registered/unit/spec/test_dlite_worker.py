from array import array
from http import HTTPStatus
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
import sglang.srt.arg_groups.speculative_hook as speculative_hook
import torch
from sglang.srt.disaggregation.utils import DisaggregationMode
from sglang.srt.managers.io_struct import AbortReq
from sglang.srt.managers.schedule_batch import Req
from sglang.srt.managers.scheduler import Scheduler
from sglang.srt.runtime_context import get_context, get_parallel
from sglang.srt.sampling.sampling_params import SamplingParams
from sglang.srt.speculative.dflash_utils import validate_dflash_request
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


@pytest.mark.parametrize(
    ("top_k", "return_hidden_states", "enable_overlap", "expected"),
    [
        pytest.param(1, False, False, None, id="greedy"),
        pytest.param(128, False, False, "greedy decoding only", id="stochastic"),
        pytest.param(
            1,
            True,
            False,
            "return_hidden_states",
            id="hidden_states_without_overlap",
        ),
        pytest.param(
            1,
            True,
            True,
            "return_hidden_states",
            id="hidden_states_with_overlap",
        ),
    ],
)
def test_request_validation_enforces_dlite_contract(
    top_k, return_hidden_states, enable_overlap, expected
):
    req = SimpleNamespace(
        sampling_params=SimpleNamespace(top_k=top_k),
        return_hidden_states=return_hidden_states,
    )

    error = validate_dflash_request(
        req,
        enable_overlap=enable_overlap,
        algorithm="DLITE",
    )

    if expected is None:
        assert error is None
    else:
        assert expected in error


def test_scheduler_rejects_invalid_dlite_request_without_queueing():
    scheduler = Scheduler.__new__(Scheduler)
    scheduler.enable_session_radix_cache = False
    scheduler.model_config = SimpleNamespace(hf_eos_token_id={1}, vocab_size=128)
    scheduler.disaggregation_mode = DisaggregationMode.NULL
    scheduler.metrics_reporter = SimpleNamespace(enable_metrics=False)
    scheduler.tokenizer = None
    scheduler.dllm_config = None
    scheduler.enable_overlap = True
    scheduler.spec_algorithm = SimpleNamespace(
        name="DLITE",
        is_dflash_family=lambda: True,
        is_uno=lambda: False,
    )
    scheduler._maybe_namespace_elastic_radix_cache = MagicMock()
    scheduler._record_kv_event_lora_name = MagicMock()
    scheduler._add_request_to_queue = MagicMock()
    scheduler.init_req_max_new_tokens = MagicMock()
    scheduler.output_streamer = MagicMock()
    sender = MagicMock()
    scheduler.ipc_channels = SimpleNamespace(send_to_tokenizer=sender)
    recv_req = MagicMock(
        session_params=None,
        session_id=None,
        input_embeds=None,
        bootstrap_port=1,
    )
    req = Req(
        "stochastic-dlite",
        "Hello",
        array("q", [1]),
        SamplingParams(temperature=0.7, top_k=20),
    )

    with (
        get_context().override_server_args(weight_version="test"),
        get_parallel().override(tp_rank=0),
        patch("sglang.srt.managers.scheduler.Req", return_value=req),
        patch(
            "sglang.srt.managers.scheduler.BeamCoordinator.request_beam_width",
            return_value=1,
        ),
    ):
        scheduler.handle_generate_request(recv_req)

    sender.send_output.assert_called_once()
    abort, original = sender.send_output.call_args.args
    assert isinstance(abort, AbortReq)
    assert original is req
    assert abort.finished_reason["status_code"] == HTTPStatus.BAD_REQUEST
    assert "greedy decoding only" in abort.finished_reason["message"]
    scheduler._add_request_to_queue.assert_not_called()
    scheduler.output_streamer.stream_output.assert_not_called()


def test_server_args_reject_non_nvidia_cuda_platform():
    cfg = SimpleNamespace(device="cuda")
    platform = SimpleNamespace(is_cuda=lambda: False)
    with (
        patch.object(speculative_hook, "resolving_view", return_value=cfg),
        patch.object(speculative_hook, "current_platform", platform),
        pytest.raises(ValueError, match="supports CUDA only"),
    ):
        speculative_hook._handle_dlite(SimpleNamespace())


def test_server_args_reject_pdmux():
    cfg = SimpleNamespace(
        device="cuda",
        speculative_draft_model_path="/draft",
        pp_size=1,
        attn_cp_size=1,
        disaggregation_mode="null",
        enable_two_batch_overlap=False,
        enable_mixed_chunk=False,
        enable_pdmux=True,
        speculative_adaptive=False,
        speculative_use_rejection_sampling=False,
        speculative_use_block_verification=False,
        speculative_draft_model_quantization=None,
        speculative_num_steps=1,
        speculative_eagle_topk=1,
        json_model_override_args="{}",
        trust_remote_code=False,
        speculative_draft_model_revision="main",
        speculative_num_draft_tokens=8,
        max_running_requests=48,
    )
    platform = SimpleNamespace(is_cuda=lambda: True)
    with (
        patch.object(speculative_hook, "resolving_view", return_value=cfg),
        patch.object(speculative_hook, "current_platform", platform),
        patch.object(speculative_hook, "attn_dp_enabled_of", return_value=False),
        patch.object(speculative_hook, "declare_resolution"),
        patch.object(
            speculative_hook,
            "model_config_of",
            return_value=SimpleNamespace(hf_text_config=SimpleNamespace()),
        ),
        patch(
            "sglang.srt.utils.hf_transformers_utils.get_config",
            return_value=SimpleNamespace(),
        ),
        patch(
            "sglang.srt.speculative.dlite_utils.validate_dlite_target_compatibility",
            return_value=SimpleNamespace(block_size=8),
        ),
        pytest.raises(ValueError, match="PDMux"),
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
