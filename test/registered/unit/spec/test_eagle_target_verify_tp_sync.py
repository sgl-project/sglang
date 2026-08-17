"""Two-rank regression test for fused EAGLE top-k=1 TP canonicalization."""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch
import torch.distributed as dist

from sglang.kernels.ops.speculative.topk1 import (
    TargetVerifyTopk1Output,
    target_verify_topk1_postprocess,
)
from sglang.srt.model_executor.forward_batch_info import ForwardMode
from sglang.srt.speculative.eagle_target_verify import (
    maybe_eagle_sample_target_verify_topk1,
)
from sglang.srt.speculative.spec_info import SpecInputType
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase, run_distributed_test

register_cuda_ci(est_time=40, stage="base-b", runner_config="2-gpu-large")

WORLD_SIZE = 2
NUM_DRAFT_TOKENS = 4
VOCAB_SIZE = 16


class _RealBroadcastGroup:
    """Minimal coordinator facade whose broadcast is a real NCCL collective."""

    def __init__(self):
        self.world_size = dist.get_world_size()
        self.broadcast_calls = 0

    def broadcast(self, tensor: torch.Tensor, src: int = 0) -> torch.Tensor:
        assert src == 0
        self.broadcast_calls += 1
        dist.broadcast(tensor, src=src)
        return tensor


def _make_verify_case(
    device_rank: int, decision_rank: int, seq_lens_dtype: torch.dtype
):
    device = torch.device("cuda", device_rank)

    # Rank 0 accepts the complete chain [3, 4, 5] and emits bonus token 6.
    # Rank 1 predicts 7 at the root, rejects immediately, and emits 7 as its
    # bonus. This makes every TargetVerifyTopk1Output field rank-distinct.
    target_ids = (
        torch.tensor([3, 4, 5, 6], device=device)
        if decision_rank == 0
        else torch.tensor([7, 8, 9, 10], device=device)
    )
    logits = torch.full(
        (NUM_DRAFT_TOKENS, VOCAB_SIZE),
        -1000.0,
        dtype=torch.float32,
        device=device,
    )
    logits.scatter_(1, target_ids[:, None], 1000.0)

    candidates = torch.tensor([[0, 3, 4, 5]], dtype=torch.long, device=device)
    retrieve_index = torch.arange(
        NUM_DRAFT_TOKENS, dtype=torch.long, device=device
    ).view(1, NUM_DRAFT_TOKENS)
    retrieve_next_token = torch.tensor([[1, 2, 3, -1]], dtype=torch.long, device=device)
    seq_lens = torch.tensor([100], dtype=seq_lens_dtype, device=device)

    verify_input = SimpleNamespace(
        spec_input_type=SpecInputType.EAGLE_VERIFY,
        tree_topk=1,
        draft_token_num=NUM_DRAFT_TOKENS,
        max_tree_depth=NUM_DRAFT_TOKENS,
        draft_token=candidates.flatten(),
        retrieve_index=retrieve_index,
        retrieve_next_token=retrieve_next_token,
    )
    batch = SimpleNamespace(
        forward_mode=ForwardMode.DECODE,
        sampling_info=SimpleNamespace(
            is_all_greedy=True,
            acc_additive_penalties=None,
            acc_scaling_penalties=None,
            logit_bias=None,
        ),
        seq_lens=seq_lens,
    )
    logits_output = SimpleNamespace(next_token_logits=logits)
    return verify_input, batch, logits_output, candidates


def _run_adversarial_rank(rank: int):
    for seq_lens_dtype in (torch.int32, torch.int64):
        scenarios = (
            ("natural", 0, 0, "fixed"),
            ("simulated-fixed", 2, 2 + rank, "fixed"),
            ("simulated-real", 2, 2 + rank, "real-draft-token"),
        )
        for scenario, rank0_num_accept, local_num_accept, token_mode in scenarios:
            verify_input, batch, logits_output, candidates = _make_verify_case(
                rank, rank, seq_lens_dtype
            )
            use_real_draft_tokens = token_mode == "real-draft-token"
            local_output = target_verify_topk1_postprocess(
                logits_output.next_token_logits,
                candidates,
                verify_input.retrieve_index,
                verify_input.retrieve_next_token,
                batch.seq_lens,
                num_simulated_accept_tokens=local_num_accept,
                use_real_draft_tokens=(local_num_accept > 0 and use_real_draft_tokens),
            )

            # Compute the rank-0 oracle independently on both ranks. Simulated
            # cases deliberately sample different lengths on rank 1; the one
            # packed broadcast must make rank 0's full result canonical.
            ref_input, ref_batch, ref_logits_output, ref_candidates = _make_verify_case(
                rank, 0, seq_lens_dtype
            )
            rank0_output = target_verify_topk1_postprocess(
                ref_logits_output.next_token_logits,
                ref_candidates,
                ref_input.retrieve_index,
                ref_input.retrieve_next_token,
                ref_batch.seq_lens,
                num_simulated_accept_tokens=rank0_num_accept,
                use_real_draft_tokens=(rank0_num_accept > 0 and use_real_draft_tokens),
            )
            divergent_fields = {
                field
                for field in TargetVerifyTopk1Output._fields
                if not torch.equal(
                    getattr(local_output, field), getattr(rank0_output, field)
                )
            }

            real_group = _RealBroadcastGroup()
            with (
                patch(
                    "sglang.srt.speculative.eagle_target_verify.get_eagle_verify_tp_group",
                    return_value=real_group,
                ),
                patch(
                    "sglang.srt.speculative.spec_utils.SIMULATE_ACC_LEN",
                    1 if local_num_accept > 0 else -1,
                ),
                patch(
                    "sglang.srt.speculative.spec_utils.SIMULATE_ACC_TOKEN_MODE",
                    token_mode,
                ),
                patch(
                    "sglang.srt.speculative.spec_utils.sample_simulated_acc_len",
                    return_value=local_num_accept,
                ),
            ):
                synchronized = maybe_eagle_sample_target_verify_topk1(
                    verify_input, batch, logits_output
                )

            assert synchronized is not None
            assert real_group.broadcast_calls == 1

            # Check adversarial preconditions only after both ranks have
            # completed the collective, so a fixture failure cannot strand the
            # peer inside NCCL.
            if rank == 0:
                assert not divergent_fields
            else:
                expected_divergent_fields = (
                    set(TargetVerifyTopk1Output._fields)
                    if scenario != "simulated-fixed"
                    else {
                        "num_correct_drafts",
                        "accept_lens",
                        "accept_index",
                        "new_seq_lens",
                        "select_index",
                    }
                )
                assert divergent_fields == expected_divergent_fields, (
                    f"unexpected adversarial divergence for {scenario} with "
                    f"seq_lens dtype {seq_lens_dtype}: {divergent_fields}"
                )
            for field in TargetVerifyTopk1Output._fields:
                torch.testing.assert_close(
                    getattr(synchronized, field),
                    getattr(rank0_output, field),
                    rtol=0,
                    atol=0,
                )


class TestEagleTargetVerifyTpSync(CustomTestCase):
    @unittest.skipUnless(
        torch.cuda.is_available() and torch.cuda.device_count() >= WORLD_SIZE,
        "This test requires two CUDA GPUs.",
    )
    def test_adversarial_rank_local_decisions_are_canonicalized(self):
        run_distributed_test(
            _run_adversarial_rank,
            world_size=WORLD_SIZE,
            backend="nccl",
        )


if __name__ == "__main__":
    unittest.main()
