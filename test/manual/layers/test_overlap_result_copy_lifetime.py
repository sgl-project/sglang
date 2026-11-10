"""Exercise allocator reuse while output copies wait on a separate stream."""

import pytest
import torch

from sglang.srt.layers.logits_processor import LogitsProcessorOutput
from sglang.srt.managers.utils import GenerationBatchResult

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


@pytest.mark.parametrize("return_logprob", [False, True])
def test_result_copy_keeps_sources_alive_until_copy_finishes(return_logprob):
    torch.cuda.synchronize()
    forward_stream, copy_stream = torch.cuda.Stream(), torch.cuda.Stream()
    with torch.cuda.stream(forward_stream):
        logits = LogitsProcessorOutput(
            next_token_logits=None,
            next_token_logprobs=torch.full((4,), -0.25, device="cuda"),
            input_token_logprobs=torch.full((4,), -0.5, device="cuda"),
            next_token_top_logprobs_val=[torch.full((4,), -0.75, device="cuda")],
            next_token_top_logprobs_idx=[torch.full((4,), 42, device="cuda")],
            next_token_token_ids_logprobs_val=[torch.full((4,), -1.25, device="cuda")],
            next_token_token_ids_logprobs_idx=[torch.full((4,), 43, device="cuda")],
            hidden_states=torch.full((4,), 0.75, device="cuda"),
        )
        result = GenerationBatchResult(
            logits_output=logits,
            next_token_ids=torch.full((4,), 29108, dtype=torch.int32, device="cuda"),
            accept_lens=torch.full((1,), 3, dtype=torch.int32, device="cuda"),
            copy_done=torch.cuda.Event(),
        )
    copy_stream.wait_stream(forward_stream)
    with torch.cuda.stream(copy_stream):
        # Make the source lifetime issue deterministic: the next forward
        # allocates and writes small tensors before these D2H reads execute.
        torch.cuda._sleep(100_000_000)
        result.copy_to_cpu(return_logprob=return_logprob)
    with torch.cuda.stream(forward_stream):
        trash = [
            torch.full((4,), -1, dtype=torch.int32, device="cuda") for _ in range(256)
        ]
    result.copy_done.synchronize()
    assert result.next_token_ids.tolist() == [29108] * 4
    assert result.accept_lens.tolist() == [3]
    assert logits.hidden_states.tolist() == [0.75] * 4
    if return_logprob:
        assert logits.next_token_logprobs.tolist() == [-0.25] * 4
        assert logits.input_token_logprobs.tolist() == [-0.5] * 4
        assert logits.next_token_top_logprobs_val[0].tolist() == [-0.75] * 4
        assert logits.next_token_top_logprobs_idx[0].tolist() == [42] * 4
        assert logits.next_token_token_ids_logprobs_val[0].tolist() == [-1.25] * 4
        assert logits.next_token_token_ids_logprobs_idx[0].tolist() == [43] * 4
    del trash
    torch.cuda.synchronize()
