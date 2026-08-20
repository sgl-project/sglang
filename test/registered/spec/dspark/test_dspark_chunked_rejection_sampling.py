import pytest
import torch

from sglang.srt.sampling.sampling_batch_info import SamplingBatchInfo
from sglang.srt.sampling.sampling_params import TOP_K_ALL
from sglang.srt.speculative.dflash_info_v2 import DFlashDraftInputV2
from sglang.srt.speculative.dspark_components.dspark_draft import DraftBlockResult
from sglang.srt.speculative.dspark_components.dspark_verify import (
    _dspark_rs_estimate_workspace_bytes,
    _dspark_rs_plan_chunk_size,
    accept_draft_tokens,
)
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=60, stage="base-b", runner_config="1-gpu-small")


def test_dspark_rs_planner_respects_workspace_budget(monkeypatch):
    bytes_per_request = _dspark_rs_estimate_workspace_bytes(
        bs=1, gamma_rows=7, verify_num_draft_tokens=8, vocab=151936
    )
    budget = 2 * 1024**3
    monkeypatch.setenv("SGLANG_DSPARK_RS_MAX_WORKSPACE_BYTES", str(budget))
    monkeypatch.setenv("SGLANG_DSPARK_RS_CHUNK_SIZE", "0")
    chunk_size, full_bytes, max_bytes = _dspark_rs_plan_chunk_size(
        bs=511, gamma_rows=7, verify_num_draft_tokens=8, vocab=151936
    )
    assert max_bytes == budget
    assert full_bytes > budget
    assert chunk_size == budget // bytes_per_request
    assert chunk_size < 511
    monkeypatch.setenv("SGLANG_DSPARK_RS_MAX_WORKSPACE_BYTES", "0")
    chunk_size, _, _ = _dspark_rs_plan_chunk_size(
        bs=511, gamma_rows=7, verify_num_draft_tokens=8, vocab=151936
    )
    assert chunk_size == 0


def test_dspark_rs_planner_keeps_full_path_when_free_memory_is_sufficient(
    monkeypatch,
):
    budget = 2 * 1024**3
    monkeypatch.setenv("SGLANG_DSPARK_RS_MAX_WORKSPACE_BYTES", str(budget))
    monkeypatch.setenv("SGLANG_DSPARK_RS_CHUNK_SIZE", "0")
    chunk_size, full_bytes, _ = _dspark_rs_plan_chunk_size(
        bs=255,
        gamma_rows=7,
        verify_num_draft_tokens=8,
        vocab=151936,
        available_workspace_bytes=3 * 1024**3,
    )
    assert full_bytes < 3 * 1024**3
    assert chunk_size == 0


def test_dspark_rs_planner_uses_free_memory_as_additional_limit(monkeypatch):
    bytes_per_request = _dspark_rs_estimate_workspace_bytes(
        bs=1, gamma_rows=7, verify_num_draft_tokens=8, vocab=151936
    )
    monkeypatch.setenv("SGLANG_DSPARK_RS_MAX_WORKSPACE_BYTES", str(2 * 1024**3))
    monkeypatch.setenv("SGLANG_DSPARK_RS_CHUNK_SIZE", "0")
    chunk_size, _, _ = _dspark_rs_plan_chunk_size(
        bs=511,
        gamma_rows=7,
        verify_num_draft_tokens=8,
        vocab=151936,
        available_workspace_bytes=512 * 1024**2,
    )
    assert chunk_size == (512 * 1024**2) // bytes_per_request


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_dspark_rs_full_and_chunked_sampling_match(monkeypatch):
    device = torch.device("cuda")
    batch_size, gamma, verify_tokens, vocab = 8, 3, 4, 128
    torch.manual_seed(20260819)
    draft_logits = torch.randn(
        (batch_size, gamma, vocab), device=device, dtype=torch.bfloat16
    )
    target_logits = torch.randn(
        (batch_size * verify_tokens, vocab), device=device, dtype=torch.bfloat16
    )
    candidates = torch.randint(
        0, vocab, (batch_size, verify_tokens), device=device, dtype=torch.int64
    )
    temperatures = torch.full((batch_size, 1), 0.8, device=device, dtype=torch.float32)
    sampling_info = SamplingBatchInfo(
        temperatures=temperatures,
        top_ps=torch.ones(batch_size, device=device),
        top_ks=torch.full((batch_size,), TOP_K_ALL, device=device, dtype=torch.int32),
        min_ps=torch.zeros(batch_size, device=device),
        is_all_greedy=False,
        is_any_greedy=False,
        need_top_p_sampling=False,
        need_top_k_sampling=False,
        need_min_p_sampling=False,
        vocab_size=vocab,
        device="cuda",
    )
    draft_input = DFlashDraftInputV2.create_idle_input(device)
    draft_block = DraftBlockResult(
        draft_tokens=candidates[:, :gamma],
        corrected_logits=draft_logits,
        greedy_mask=torch.zeros(batch_size, device=device, dtype=torch.bool),
        temperatures=temperatures,
    )

    monkeypatch.setenv("SGLANG_DSPARK_RS_MAX_WORKSPACE_BYTES", "0")
    torch.manual_seed(777)
    full = accept_draft_tokens(
        candidates=candidates,
        target_logits=target_logits,
        draft_block=draft_block,
        sampling_info=sampling_info,
        draft_input=draft_input,
        gamma=gamma,
        verify_num_draft_tokens=verify_tokens,
    )
    torch.cuda.synchronize()
    full = tuple(value.clone() for value in full)

    monkeypatch.setenv("SGLANG_DSPARK_RS_MAX_WORKSPACE_BYTES", "8192")
    torch.manual_seed(777)
    chunked = accept_draft_tokens(
        candidates=candidates,
        target_logits=target_logits,
        draft_block=draft_block,
        sampling_info=sampling_info,
        draft_input=draft_input,
        gamma=gamma,
        verify_num_draft_tokens=verify_tokens,
    )
    torch.cuda.synchronize()
    chunked = tuple(value.clone() for value in chunked)

    assert all(torch.equal(lhs, rhs) for lhs, rhs in zip(full, chunked))
