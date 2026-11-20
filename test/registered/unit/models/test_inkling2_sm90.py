"""Hopper expert GEMMs and decode attention, and the MTP verify commit."""

import sys
import types

import pytest
import torch

from sglang.srt.models.inkling2_mtp import Inkling2ForConditionalGenerationMTP
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=65, stage="base-b-kernel-unit", runner_config="1-gpu-large")


EXPERTS, LOCAL, TOP_K = 64, 8, 8


def _fake_model(req_pool_size, hidden):
    model = types.SimpleNamespace(
        _f_rnn_buf=torch.full((req_pool_size + 1, hidden), -1.0, device="cuda")
    )

    def store_f_rnn(req_pool_indices, rows):
        model._f_rnn_buf.index_copy_(0, req_pool_indices.view(-1).long(), rows)

    model.store_f_rnn = store_f_rnn
    return model


@pytest.mark.parametrize("valid_mask", [None, "gated"])
def test_store_verify_commit_routes_invalid_rows_to_sink(valid_mask):
    if not torch.cuda.is_available():
        pytest.skip("needs CUDA")
    req_pool_size, hidden, stride, bs, real_bs = 16, 8, 4, 6, 3
    model = _fake_model(req_pool_size, hidden)
    verify_hidden = torch.arange(
        bs * stride * hidden, dtype=torch.float32, device="cuda"
    ).view(bs * stride, hidden)
    req_pool_indices = torch.tensor([5, 2, 9, 0, 0, 0], device="cuda")
    commit_lens = torch.tensor([1, 4, 2, 0, 1, 3], dtype=torch.int32, device="cuda")
    valid = torch.arange(bs, device="cuda") < real_bs if valid_mask == "gated" else None
    num_rows = bs if valid is not None else real_bs

    Inkling2ForConditionalGenerationMTP.store_verify_commit(
        model,
        req_pool_indices=req_pool_indices[:num_rows],
        verify_hidden=verify_hidden[: num_rows * stride],
        commit_lens=commit_lens[:num_rows],
        stride=stride,
        valid=valid,
    )

    expected = torch.full((req_pool_size + 1, hidden), -1.0, device="cuda")
    rows = verify_hidden.view(bs, stride, hidden)
    for i in range(real_bs):
        expected[req_pool_indices[i]] = rows[i, commit_lens[i] - 1]
    buf = model._f_rnn_buf
    assert torch.equal(buf[:req_pool_size], expected[:req_pool_size])
    if valid is None:
        assert torch.equal(buf[req_pool_size], expected[req_pool_size])


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
