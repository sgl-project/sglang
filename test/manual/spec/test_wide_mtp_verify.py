"""Every rejection position in six/eight-token native MTP verification."""

from types import SimpleNamespace

import pytest
import torch
from sglang.srt.runtime_context import publish
from sglang.srt.server_args import ServerArgs
from sglang.srt.speculative.eagle_info import EagleVerifyInput
from sglang.srt.speculative.eagle_utils import (
    TreeMaskMode,
    build_tree_kernel_efficient,
    eagle_sample,
)

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")

# `eagle_sample` reads the published speculative config (rejection sampling,
# simulated accept length); publish a minimal process context for bare pytest.
publish(ServerArgs(model_path="dummy"), role="test")


@pytest.mark.parametrize("width,rejected", [(w, r) for w in (6, 8) for r in range(w)])
def test_wide_chain_rejection_and_committed_indices(width, rejected):
    seq = torch.tensor([25000], device="cuda", dtype=torch.int64)
    proposals = torch.arange(11, 10 + width, device="cuda").long()[None]
    mask, positions, indices, child, sibling, tokens = build_tree_kernel_efficient(
        torch.tensor([10], device="cuda", dtype=torch.int64),
        torch.arange(-1, width - 2, device="cuda").long()[None],
        torch.arange(width - 1, device="cuda").long()[None],
        proposals,
        seq,
        25000,
        1,
        width - 1,
        width,
        TreeMaskMode.QLEN_ONLY,
    )
    verify = EagleVerifyInput(
        draft_token=tokens,
        custom_mask=mask,
        positions=positions,
        retrieve_index=indices,
        retrieve_next_token=child,
        retrieve_next_sibling=sibling,
        retrieve_cum_len=None,
        spec_steps=width - 1,
        topk=1,
        draft_token_num=width,
        capture_hidden_mode=None,
        seq_lens_sum=None,
        seq_lens_cpu=None,
    )
    choices = list(range(11, 10 + width)) + [99]
    choices[rejected] = 99
    logits = torch.full((width, 128), -100.0, device="cuda")
    logits[torch.arange(width, device="cuda"), torch.tensor(choices, device="cuda")] = (
        100
    )
    batch = SimpleNamespace(
        device="cuda",
        seq_lens=seq,
        forward_mode=SimpleNamespace(is_idle=lambda: False),
        sampling_info=SimpleNamespace(
            acc_additive_penalties=None,
            acc_scaling_penalties=None,
            logit_bias=None,
            is_all_greedy=True,
        ),
    )
    predicted, lengths, accepted = eagle_sample(
        verify, batch, SimpleNamespace(next_token_logits=logits), None
    )
    assert lengths.item() == rejected + 1
    assert predicted[accepted[0, : rejected + 1]].tolist() == list(
        range(11, 11 + rejected)
    ) + [99]
    assert accepted[0, : rejected + 1].tolist() == list(range(rejected + 1))
    assert positions.tolist() == list(range(25000, 25000 + width))


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__, "-v"]))
