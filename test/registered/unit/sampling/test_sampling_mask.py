"""Unit tests for tensor-backed sampling-mask packing and result copying."""

from sglang.test.ci.ci_register import register_amd_ci, register_cuda_ci

register_cuda_ci(est_time=10, stage="base-b", runner_config="1-gpu-small")
register_amd_ci(est_time=10, suite="stage-b-test-1-gpu-small-amd")

from types import SimpleNamespace

import pytest
import torch

from sglang.srt.layers.logits_processor import LogitsProcessorOutput
from sglang.srt.layers.sampler import Sampler
from sglang.srt.managers.utils import GenerationBatchResult


def test_sampling_mask_is_packed_as_tensors():
    probs = torch.tensor(
        [
            [0.4, 0.3, 0.2, 0.1],
            [0.1, 0.2, 0.3, 0.4],
        ],
        dtype=torch.float32,
        device="cuda",
    )
    sampling_info = SimpleNamespace(
        sampling_mask_max_top_k=3,
        top_ks=torch.tensor([3, 2], dtype=torch.int32, device="cuda"),
        top_ps=torch.tensor([0.6, 1.0], dtype=torch.float32, device="cuda"),
        min_ps=torch.zeros(2, dtype=torch.float32, device="cuda"),
        need_min_p_sampling=False,
    )
    sampled_tokens = torch.tensor([1, 0], dtype=torch.int64, device="cuda")
    output = LogitsProcessorOutput(next_token_logits=None)

    sampling_mask_data = Sampler._compute_sampling_mask_from_probs(
        None, probs, sampling_info
    )
    Sampler._attach_sampling_mask_to_output(
        None,
        output,
        sampling_info,
        sampled_tokens,
        sampling_mask_data,
    )

    assert isinstance(output.next_token_sampling_mask_idx, torch.Tensor)
    assert isinstance(output.next_token_sampling_mask_len, torch.Tensor)
    assert isinstance(output.next_token_sampling_logprobs, torch.Tensor)
    assert output.next_token_sampling_mask_idx.is_cuda
    assert output.next_token_sampling_mask_len.is_cuda
    assert output.next_token_sampling_logprobs.is_cuda
    assert output.next_token_sampling_mask_len.tolist() == [2, 3]
    assert output.next_token_sampling_mask_idx[0, :2].tolist() == [0, 1]
    # Token 0 is outside the second row's reconstructed top-2 support, so it is
    # appended after the retained candidates without materializing on the host.
    assert output.next_token_sampling_mask_idx[1, :3].tolist() == [3, 2, 0]
    assert output.next_token_sampling_logprobs.tolist() == pytest.approx(
        [
            torch.log(torch.tensor(0.3 / 0.7)).item(),
            torch.log(torch.tensor(0.1 / 0.8)).item(),
        ]
    )


def test_greedy_sampling_mask_is_tensor_backed():
    sampled_tokens = torch.tensor([4, 7], dtype=torch.int64, device="cuda")
    output = LogitsProcessorOutput(next_token_logits=None)

    Sampler._attach_greedy_sampling_mask_to_output(
        None,
        output,
        SimpleNamespace(),
        sampled_tokens,
    )

    assert output.next_token_sampling_mask_idx.tolist() == [[4], [7]]
    assert output.next_token_sampling_mask_len.tolist() == [1, 1]
    assert output.next_token_sampling_logprobs.tolist() == [0.0, 0.0]


def test_sampling_mask_uses_generation_result_copy_stream():
    output = LogitsProcessorOutput(
        next_token_logits=None,
        next_token_sampling_mask_idx=torch.tensor(
            [[3, 2, 0]], dtype=torch.int32, device="cuda"
        ),
        next_token_sampling_mask_len=torch.tensor(
            [3], dtype=torch.int32, device="cuda"
        ),
        next_token_sampling_logprobs=torch.tensor(
            [-0.5], dtype=torch.float32, device="cuda"
        ),
    )
    result = GenerationBatchResult(
        logits_output=output,
        next_token_ids=torch.tensor([0], dtype=torch.int64, device="cuda"),
        copy_done=torch.cuda.Event(),
    )

    copy_stream = torch.cuda.Stream()
    copy_stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(copy_stream):
        result.copy_to_cpu(return_logprob=False, return_hidden_states=False)
    result.copy_done.synchronize()

    assert result.logits_output.next_token_sampling_mask_idx.device.type == "cpu"
    assert result.logits_output.next_token_sampling_mask_len.device.type == "cpu"
    assert result.logits_output.next_token_sampling_logprobs.device.type == "cpu"
    assert result.logits_output.next_token_sampling_mask_idx.tolist() == [[3, 2, 0]]
    assert result.logits_output.next_token_sampling_mask_len.tolist() == [3]
    assert result.logits_output.next_token_sampling_logprobs.tolist() == [-0.5]
