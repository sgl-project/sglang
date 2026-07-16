from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=10, stage="base-b", runner_config="1-gpu-small")

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.kernels.ops.speculative.topk1 import (
    draft_topk1_postprocess,
    target_verify_topk1_postprocess,
)
from sglang.srt.model_executor.forward_batch_info import ForwardMode
from sglang.srt.speculative.eagle_utils import (
    maybe_eagle_sample_target_verify_topk1,
)
from sglang.srt.speculative.spec_info import SpecInputType
from sglang.test.test_utils import CustomTestCase


def _make_logits_with_unique_argmax(
    batch_size: int,
    vocab_size: int,
    *,
    dtype: torch.dtype,
    device: torch.device,
    seed: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    g = torch.Generator(device=device).manual_seed(seed)
    logits = torch.randn(
        (batch_size, vocab_size), dtype=dtype, device=device, generator=g
    )
    expected_index = (
        torch.arange(batch_size, dtype=torch.long, device=device) * 9973 + 17
    ) % vocab_size
    logits.scatter_(1, expected_index[:, None], 1000.0)
    return logits, expected_index[:, None]


def _make_topk1_chain(batch_size: int, num_tokens: int, device: torch.device):
    retrieve_index = torch.arange(
        batch_size * num_tokens, dtype=torch.long, device=device
    ).view(batch_size, num_tokens)
    retrieve_next_token = torch.arange(
        1, num_tokens + 1, dtype=torch.long, device=device
    ).repeat(batch_size, 1)
    retrieve_next_token[:, -1] = -1
    return retrieve_index, retrieve_next_token


def _assert_target_verify_outputs(
    outputs,
    target_ids: torch.Tensor,
    accepted_drafts: list[int],
    seq_lens: torch.Tensor,
):
    (
        predict,
        num_correct_drafts,
        accept_lens,
        accept_index,
        bonus_tokens,
        new_seq_lens,
        select_index,
        draft_input_ids,
    ) = outputs
    batch_size, num_tokens = target_ids.shape
    expected_predict = torch.zeros_like(target_ids, dtype=torch.int32)
    expected_accept_index = torch.full(
        (batch_size, num_tokens), -1, dtype=torch.int32, device=target_ids.device
    )
    for batch_idx, num_accepted in enumerate(accepted_drafts):
        accept_len = num_accepted + 1
        expected_predict[batch_idx, :accept_len] = target_ids[batch_idx, :accept_len]
        expected_accept_index[batch_idx, :accept_len] = torch.arange(
            batch_idx * num_tokens,
            batch_idx * num_tokens + accept_len,
            dtype=torch.int32,
            device=target_ids.device,
        )

    expected_num_correct = torch.tensor(
        accepted_drafts, dtype=torch.int32, device=target_ids.device
    )
    expected_accept_lens = expected_num_correct + 1
    expected_bonus = target_ids[
        torch.arange(batch_size, device=target_ids.device), expected_num_correct.long()
    ].to(torch.int32)
    expected_select = (
        torch.arange(batch_size, device=target_ids.device) * num_tokens
        + expected_num_correct
    ).long()

    torch.testing.assert_close(predict, expected_predict.flatten(), rtol=0, atol=0)
    torch.testing.assert_close(num_correct_drafts, expected_num_correct, rtol=0, atol=0)
    torch.testing.assert_close(accept_lens, expected_accept_lens, rtol=0, atol=0)
    torch.testing.assert_close(accept_index, expected_accept_index, rtol=0, atol=0)
    torch.testing.assert_close(bonus_tokens, expected_bonus, rtol=0, atol=0)
    torch.testing.assert_close(
        new_seq_lens, seq_lens + expected_accept_lens, rtol=0, atol=0
    )
    torch.testing.assert_close(select_index, expected_select, rtol=0, atol=0)
    torch.testing.assert_close(
        draft_input_ids, expected_predict.flatten().long(), rtol=0, atol=0
    )


def _make_target_verify_selection_case(device: torch.device):
    batch_size, num_tokens, vocab_size = 1, 2, 16
    logits = torch.zeros(
        (batch_size * num_tokens, vocab_size), device=device, dtype=torch.float32
    )
    additive_penalty = torch.zeros((batch_size, vocab_size), device=device)
    additive_penalty[:, 3] = 10.0
    sampling_info = SimpleNamespace(
        is_all_greedy=True,
        acc_additive_penalties=additive_penalty,
        acc_scaling_penalties=None,
        logit_bias=None,
    )
    retrieve_index, retrieve_next_token = _make_topk1_chain(
        batch_size, num_tokens, device
    )
    verify_input = SimpleNamespace(
        spec_input_type=SpecInputType.EAGLE_VERIFY,
        tree_topk=1,
        draft_token_num=num_tokens,
        max_tree_depth=num_tokens,
        draft_token=torch.tensor([0, 3], device=device),
        retrieve_index=retrieve_index,
        retrieve_next_token=retrieve_next_token,
    )
    batch = SimpleNamespace(
        forward_mode=ForwardMode.DECODE,
        sampling_info=sampling_info,
        seq_lens=torch.tensor([32], dtype=torch.int32, device=device),
    )
    logits_output = SimpleNamespace(next_token_logits=logits)
    return verify_input, batch, logits_output


@unittest.skipUnless(torch.cuda.is_available(), "CUDA is required for this test.")
class TestSpecTopk1Triton(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.device = torch.device("cuda")

    def test_draft_topk1_postprocess_matches_argmax_and_position_add(self):
        configs = [
            (1, 127, torch.float32),
            (4, 8192, torch.float16),
            (7, 8193, torch.bfloat16),
            (3, 50000, torch.float32),
        ]
        for batch_size, vocab_size, dtype in configs:
            with self.subTest(
                batch_size=batch_size, vocab_size=vocab_size, dtype=dtype
            ):
                logits, expected_index = _make_logits_with_unique_argmax(
                    batch_size,
                    vocab_size,
                    dtype=dtype,
                    device=self.device,
                    seed=vocab_size,
                )
                positions = torch.arange(
                    batch_size, dtype=torch.long, device=self.device
                )
                expected_positions = positions + 1

                topk_p, topk_index = draft_topk1_postprocess(logits, positions)

                torch.testing.assert_close(topk_index, expected_index, rtol=0, atol=0)
                torch.testing.assert_close(
                    topk_p,
                    torch.ones(
                        (batch_size, 1), dtype=torch.float32, device=self.device
                    ),
                    rtol=0,
                    atol=0,
                )
                torch.testing.assert_close(
                    positions, expected_positions, rtol=0, atol=0
                )

    def test_draft_topk1_postprocess_can_write_draft_token_column(self):
        batch_size = 17
        # Multi-split vocab so the fused write composes with the split reduction.
        vocab_size = 50000
        logits, expected_index = _make_logits_with_unique_argmax(
            batch_size,
            vocab_size,
            dtype=torch.float32,
            device=self.device,
            seed=0,
        )
        positions = torch.zeros(batch_size, dtype=torch.long, device=self.device)
        backing = torch.full((batch_size, 5), -1, dtype=torch.long, device=self.device)
        draft_tokens = backing[:, 1:4]

        topk_p, topk_index = draft_topk1_postprocess(
            logits, positions, draft_tokens, draft_token_column=2
        )

        torch.testing.assert_close(topk_index, expected_index, rtol=0, atol=0)
        torch.testing.assert_close(topk_p, torch.ones_like(topk_p), rtol=0, atol=0)
        # Exactly one backing column is written; both neighbors stay untouched.
        expected_backing = torch.full_like(backing, -1)
        expected_backing[:, 3] = expected_index[:, 0]
        torch.testing.assert_close(backing, expected_backing, rtol=0, atol=0)
        torch.testing.assert_close(
            positions, torch.ones_like(positions), rtol=0, atol=0
        )

    def test_row_strided_logits_view_matches_argmax(self):
        batch_size = 5
        vocab_size = 8193
        # Poison the padding columns: if the kernel used the dense vocab width
        # as the row stride it would read them and pick the wrong index.
        backing = torch.full(
            (batch_size, vocab_size + 64),
            2000.0,
            dtype=torch.float32,
            device=self.device,
        )
        logits, expected_index = _make_logits_with_unique_argmax(
            batch_size,
            vocab_size,
            dtype=torch.float32,
            device=self.device,
            seed=1,
        )
        backing[:, :vocab_size] = logits
        strided_logits = backing[:, :vocab_size]
        self.assertFalse(strided_logits.is_contiguous())
        positions = torch.zeros(batch_size, dtype=torch.long, device=self.device)

        topk_p, topk_index = draft_topk1_postprocess(strided_logits, positions)

        torch.testing.assert_close(topk_index, expected_index, rtol=0, atol=0)
        torch.testing.assert_close(topk_p, torch.ones_like(topk_p), rtol=0, atol=0)
        torch.testing.assert_close(
            positions, torch.ones_like(positions), rtol=0, atol=0
        )

    def test_empty_batch(self):
        logits = torch.empty((0, 1024), dtype=torch.float32, device=self.device)
        positions = torch.empty((0,), dtype=torch.long, device=self.device)
        draft_tokens = torch.empty((0, 3), dtype=torch.long, device=self.device)

        topk_p, topk_index = draft_topk1_postprocess(
            logits, positions, draft_tokens, draft_token_column=1
        )

        self.assertEqual(topk_p.shape, (0, 1))
        self.assertEqual(topk_index.shape, (0, 1))
        self.assertEqual(draft_tokens.numel(), 0)

    def test_non_contiguous_inputs_raise(self):
        logits = torch.empty((16, 4), dtype=torch.float32, device=self.device).t()
        positions = torch.arange(8, dtype=torch.long, device=self.device)[::2]

        with self.assertRaises(AssertionError):
            draft_topk1_postprocess(
                logits, torch.empty(4, dtype=torch.long, device=self.device)
            )
        with self.assertRaises(AssertionError):
            draft_topk1_postprocess(torch.empty((4, 16), device=self.device), positions)

    def test_target_verify_mixed_acceptance_and_row_stride(self):
        batch_size, num_tokens, vocab_size = 4, 6, 8193
        accepted_drafts = [0, 1, num_tokens - 2, num_tokens - 1]
        backing = torch.full(
            (batch_size * num_tokens, vocab_size + 64),
            2000.0,
            dtype=torch.float32,
            device=self.device,
        )
        logits = backing[:, :vocab_size]
        target_ids = (
            torch.arange(batch_size * num_tokens, device=self.device) * 9973 + 17
        ) % vocab_size
        logits.fill_(-1000.0)
        logits.scatter_(1, target_ids[:, None], 1000.0)
        target_ids = target_ids.view(batch_size, num_tokens)

        candidates = torch.zeros(
            (batch_size, num_tokens), dtype=torch.long, device=self.device
        )
        candidates[:, 1:] = (target_ids[:, :-1] + 1) % vocab_size
        for batch_idx, num_accepted in enumerate(accepted_drafts):
            candidates[batch_idx, 1 : num_accepted + 1] = target_ids[
                batch_idx, :num_accepted
            ]
        retrieve_index, retrieve_next_token = _make_topk1_chain(
            batch_size, num_tokens, self.device
        )
        seq_lens = torch.arange(
            100, 100 + batch_size, dtype=torch.long, device=self.device
        )

        outputs = target_verify_topk1_postprocess(
            logits,
            candidates,
            retrieve_index,
            retrieve_next_token,
            seq_lens,
        )

        self.assertFalse(logits.is_contiguous())
        _assert_target_verify_outputs(outputs, target_ids, accepted_drafts, seq_lens)

    def test_target_verify_leftmost_ties_and_all_negative_infinity(self):
        batch_size, num_tokens, vocab_size = 1, 3, 8193
        logits = torch.full(
            (num_tokens, vocab_size),
            -float("inf"),
            dtype=torch.float32,
            device=self.device,
        )
        logits[0, 7] = 1.0
        logits[0, 8192] = 1.0
        # Row 1 remains all -inf, whose leftmost argmax is zero.
        logits[2, 8191] = 1.0
        logits[2, 8192] = 1.0
        target_ids = torch.tensor([[7, 0, 8191]], device=self.device)
        candidates = torch.tensor([[0, 7, 0]], device=self.device)
        retrieve_index, retrieve_next_token = _make_topk1_chain(
            batch_size, num_tokens, self.device
        )
        seq_lens = torch.tensor([41], dtype=torch.int32, device=self.device)

        outputs = target_verify_topk1_postprocess(
            logits,
            candidates,
            retrieve_index,
            retrieve_next_token,
            seq_lens,
        )

        _assert_target_verify_outputs(outputs, target_ids, [2], seq_lens)

    def test_target_verify_trivial_chain(self):
        logits = torch.tensor([[0.0, 3.0, 1.0]], device=self.device)
        candidates = torch.zeros((1, 1), dtype=torch.long, device=self.device)
        retrieve_index, retrieve_next_token = _make_topk1_chain(1, 1, self.device)
        seq_lens = torch.tensor([9], dtype=torch.long, device=self.device)

        outputs = target_verify_topk1_postprocess(
            logits,
            candidates,
            retrieve_index,
            retrieve_next_token,
            seq_lens,
        )

        _assert_target_verify_outputs(
            outputs, torch.tensor([[1]], device=self.device), [0], seq_lens
        )

    def test_target_verify_trace_shape(self):
        batch_size, num_tokens, vocab_size = 1, 6, 154880
        logits, target_ids = _make_logits_with_unique_argmax(
            batch_size * num_tokens,
            vocab_size,
            dtype=torch.float32,
            device=self.device,
            seed=2,
        )
        target_ids = target_ids.view(batch_size, num_tokens)
        candidates = torch.zeros(
            (batch_size, num_tokens), dtype=torch.long, device=self.device
        )
        candidates[:, 1:] = target_ids[:, :-1]
        retrieve_index, retrieve_next_token = _make_topk1_chain(
            batch_size, num_tokens, self.device
        )
        seq_lens = torch.tensor([2048], dtype=torch.long, device=self.device)

        outputs = target_verify_topk1_postprocess(
            logits,
            candidates,
            retrieve_index,
            retrieve_next_token,
            seq_lens,
        )

        _assert_target_verify_outputs(outputs, target_ids, [num_tokens - 1], seq_lens)

    def test_target_verify_selection_applies_logits_transform_once(self):
        verify_input, batch, logits_output = _make_target_verify_selection_case(
            self.device
        )

        result = maybe_eagle_sample_target_verify_topk1(
            verify_input, batch, logits_output, enabled=True
        )

        self.assertIsNotNone(result)
        torch.testing.assert_close(
            result.predict,
            torch.tensor([3, 3], dtype=torch.int32, device=self.device),
            rtol=0,
            atol=0,
        )
        # The in-place penalty application is the same one used by the legacy
        # sampler and must run exactly once before kernel selection.
        torch.testing.assert_close(
            logits_output.next_token_logits[:, 3],
            torch.full((2,), 10.0, device=self.device),
            rtol=0,
            atol=0,
        )

    def test_target_verify_selection_fallbacks(self):
        cases = (
            "disabled",
            "non_greedy",
            "input_type",
            "topk",
            "idle",
            "simulation",
            "stride",
        )
        for case in cases:
            with self.subTest(case=case):
                verify_input, batch, logits_output = _make_target_verify_selection_case(
                    self.device
                )
                enabled = True
                simulation = 0
                if case == "disabled":
                    enabled = False
                elif case == "non_greedy":
                    batch.sampling_info.is_all_greedy = False
                elif case == "input_type":
                    verify_input.spec_input_type = SpecInputType.FROZEN_KV_MTP_VERIFY
                elif case == "topk":
                    verify_input.tree_topk = 2
                elif case == "idle":
                    batch.forward_mode = ForwardMode.IDLE
                elif case == "simulation":
                    simulation = 1
                elif case == "stride":
                    logits_output.next_token_logits = torch.empty(
                        (16, 2), device=self.device
                    ).t()

                with patch(
                    "sglang.srt.speculative.spec_utils.SIMULATE_ACC_LEN", simulation
                ):
                    result = maybe_eagle_sample_target_verify_topk1(
                        verify_input,
                        batch,
                        logits_output,
                        enabled=enabled,
                    )
                self.assertIsNone(result)


if __name__ == "__main__":
    unittest.main()
