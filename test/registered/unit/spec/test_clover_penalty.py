# SPDX-License-Identifier: Apache-2.0
"""Clover penalty history, row alignment and request validation contracts."""

import unittest
from types import SimpleNamespace

import torch

from sglang.srt.sampling.sampling_params import SamplingParams
from sglang.srt.speculative.clover_config import validate_clover_request
from sglang.srt.speculative.clover_penalty import (
    CloverPenaltyState,
)
from sglang.srt.speculative.clover_penalty import (
    apply_clover_penalties as _apply_clover_penalties,
)
from sglang.srt.speculative.clover_penalty import (
    get_clover_penalty_state,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class NativeBatch:
    def __init__(self, reqs, device="cpu", vocab_size=8):
        from sglang.srt.sampling import penaltylib

        self.reqs = reqs
        self.device = device
        self.orchestrator = penaltylib.BatchedPenalizerOrchestrator(
            vocab_size=vocab_size,
            batch=self,
            penalizers={
                penaltylib.BatchedFrequencyPenalizer,
                penaltylib.BatchedPresencePenalizer,
                penaltylib.BatchedRepetitionPenalizer,
            },
        )

    def sampling_info(self):
        from sglang.srt.sampling.sampling_batch_info import SamplingBatchInfo

        values = torch.ones(len(self.reqs), device=self.device)
        return SamplingBatchInfo(
            temperatures=values[:, None],
            top_ps=values,
            top_ks=values.int(),
            min_ps=values * 0,
            is_all_greedy=True,
            is_any_greedy=True,
            need_top_p_sampling=False,
            need_top_k_sampling=False,
            need_min_p_sampling=False,
            vocab_size=self.orchestrator.vocab_size,
            penalizer_orchestrator=self.orchestrator,
        )


def apply_clover_penalties(logits, reqs, visible_tokens, *, state=None, **kwargs):
    state = state or CloverPenaltyState()
    batch = NativeBatch(reqs, logits.device, logits.shape[-1])
    state.sync_native(batch.orchestrator)
    info = batch.sampling_info().copy_for_forward()
    penalties = (info.acc_additive_penalties, info.acc_scaling_penalties)
    _apply_clover_penalties(
        logits, reqs, visible_tokens, state=state, penalties=penalties, **kwargs
    )


def request(history=(), **params):
    return SimpleNamespace(
        sampling_params=SamplingParams(**params),
        output_ids=list(history),
        origin_input_ids=[5, 5, 5],
        eos_token_ids={7},
        tokenizer=None,
        vocab_size=None,
        return_logprob=False,
        return_sampling_mask=False,
        custom_logit_processor=None,
        input_embeds=None,
        positional_embed_overrides=None,
        return_hidden_states=False,
    )


class TestCloverPenalties(CustomTestCase):
    def test_default_bias_resolution_partial_override_and_explicit_disable(self):
        class Tokenizer:
            unk_token_id = None

            def encode(self, text, add_special_tokens=False):
                return [6]

            def __len__(self):
                return 8

        req = request([1] * 4096, custom_params={"unrelated": True})
        req.tokenizer = Tokenizer()
        self.assertIsNone(validate_clover_request(req))
        custom = req.sampling_params.custom_params
        self.assertEqual(custom["dllm_length_penalty"], 10)
        self.assertEqual(custom["dllm_think_end_bias"], 10)
        self.assertEqual(custom["dllm_think_end_start"], 4096)
        self.assertEqual(custom["dllm_think_end_token_id"], 6)
        self.assertTrue(custom["unrelated"])
        logits = torch.zeros(1, 4, 8)
        apply_clover_penalties(logits, [req], torch.tensor([[1, 2, 3, 4]]))
        self.assertEqual(logits[0, 0, 6].item(), 10 / 4096)
        self.assertEqual(logits[0, :, 7].count_nonzero().item(), 0)
        disabled = request(
            [1], custom_params={"dllm_length_penalty": 0, "dllm_think_end_bias": 0}
        )
        self.assertIsNone(validate_clover_request(disabled))
        scores = torch.randn(1, 4, 8)
        before = scores.clone()
        apply_clover_penalties(scores, [disabled], torch.ones(1, 4))
        torch.testing.assert_close(scores, before)

    def test_tokenizerless_think_end_resolution_and_vocab_bound(self):
        """Without a tokenizer, the checkpoint id resolves and out-of-vocab ids are rejected."""
        generation_config = SimpleNamespace(
            to_dict=lambda: {"dllm_think_end_token_id": 6}
        )
        req = request([1])
        req.vocab_size = 8
        self.assertIsNone(
            validate_clover_request(req, generation_config=generation_config)
        )
        self.assertEqual(
            req.sampling_params.custom_params["dllm_think_end_token_id"], 6
        )
        out_of_vocab = request([1], custom_params={"dllm_think_end_token_id": 8})
        out_of_vocab.vocab_size = 8
        self.assertIn("valid vocabulary ID", validate_clover_request(out_of_vocab))

    def test_native_history_increment_filter_merge_and_rebuild(self):
        a = request(
            [1, 1], frequency_penalty=0.7, presence_penalty=0.9, repetition_penalty=1.05
        )
        b = request(
            [2], frequency_penalty=-0.3, presence_penalty=-0.2, repetition_penalty=0.8
        )
        batch = NativeBatch([a, b])
        state = CloverPenaltyState()

        def check():
            buffers = state.sync_native(batch.orchestrator)
            for row, req in enumerate(batch.reqs):
                counts = torch.tensor(
                    [req.output_ids.count(t) for t in range(8)], dtype=torch.float32
                )
                sp = req.sampling_params
                from sglang.srt.sampling import penaltylib

                reference = NativeBatch([req])
                native_frequency = reference.orchestrator.penalizers[
                    penaltylib.BatchedFrequencyPenalizer
                ]
                for token in req.output_ids:
                    native_frequency.cumulate_output_tokens(torch.tensor([token]))
                expected = (
                    native_frequency.cumulated_frequency_penalties[0],
                    (counts > 0).float() * sp.presence_penalty,
                    torch.where(counts > 0, sp.repetition_penalty, 1.0),
                )
                for buf, ref in zip(buffers, expected):
                    torch.testing.assert_close(buf[row], ref, atol=0, rtol=0)
            return buffers

        original = check()
        a.output_ids.extend([1, 3, 3])
        updated = check()
        self.assertTrue(all(x is y for x, y in zip(original, updated)))
        check()  # Preview/verify must not count the same tokens twice.
        batch.reqs = [b, a]
        batch.orchestrator.filter(torch.tensor([1, 0]))
        check()
        other = NativeBatch(
            [
                request(
                    [4, 4],
                    frequency_penalty=0.2,
                    presence_penalty=0.4,
                    repetition_penalty=2,
                )
            ]
        )
        batch.orchestrator.merge(other.orchestrator)
        batch.reqs += other.reqs
        check()
        a.output_ids = [5]  # Retraction/replacement on unchanged storage.
        check()
        del b.output_ids[:]
        check()
        # Same requests, newly allocated native storage.
        batch = NativeBatch(batch.reqs)
        check()

    def test_block_frequency_matches_native_token_updates(self):
        from sglang.srt.sampling import penaltylib

        for coefficient in (0.1, -0.3):
            with self.subTest(coefficient=coefficient):
                req = request([], frequency_penalty=coefficient)
                batch = NativeBatch([req])
                reference = NativeBatch([req])
                native = reference.orchestrator.penalizers[
                    penaltylib.BatchedFrequencyPenalizer
                ]
                state = CloverPenaltyState()
                for block in ([1] * 37 + [2, 2], [1, 3, 1] * 20, []):
                    req.output_ids.extend(block)
                    for token in block:
                        native.cumulate_output_tokens(torch.tensor([token]))
                    frequency, _, _ = state.sync_native(batch.orchestrator)
                    torch.testing.assert_close(
                        frequency, native.cumulated_frequency_penalties, atol=0, rtol=0
                    )

    def test_presence_and_repetition_only_track_committed_first_occurrences(self):
        req = request([1, 2], presence_penalty=0.9, repetition_penalty=1.05)
        batch = NativeBatch([req])
        state = CloverPenaltyState()

        def check():
            frequency, presence, repetition = state.sync_native(batch.orchestrator)
            self.assertIsNone(frequency)
            seen = torch.tensor([t in req.output_ids for t in range(8)])
            torch.testing.assert_close(presence[0], seen.float() * 0.9, atol=0, rtol=0)
            torch.testing.assert_close(
                repetition[0], torch.where(seen, 1.05, 1.0), atol=0, rtol=0
            )

        check()
        req.output_ids.extend([2, 3, 3])  # Only 3 becomes newly present.
        check()
        req.output_ids.extend([1, 2, 3])  # Entirely repeated committed block.
        check()
        req.output_ids = [4]  # Same native storage, different history.
        check()

    def test_explicit_history_tracks_committed_close_across_rounds(self):
        # Resolving native buffers must not hide new tokens from think-bias
        # history, including when all standard penalties are disabled.
        for presence in (0.0, 0.9):
            with self.subTest(presence=presence):
                req = request(
                    [1],
                    presence_penalty=presence,
                    custom_params={
                        "dllm_think_end_bias": 4.0,
                        "dllm_think_end_start": 0,
                        "dllm_think_end_window": 1,
                        "dllm_think_end_token_id": 6,
                    },
                )
                batch = NativeBatch([req])
                visible = torch.tensor([[1, 2]])

                state = get_clover_penalty_state(batch.orchestrator)

                def apply(buffers):
                    logits = torch.zeros(1, 2, 8)
                    _apply_clover_penalties(
                        logits, [req], visible, state=state, penalties=buffers
                    )
                    return logits

                def prepare():
                    state.sync_native(batch.orchestrator)
                    info = batch.sampling_info().copy_for_forward()
                    return info.acc_additive_penalties, info.acc_scaling_penalties

                first = prepare()
                initial = apply(first)
                self.assertEqual(initial[0, 0, 6].item(), 4.0)
                torch.testing.assert_close(apply(first), initial, atol=0, rtol=0)
                req.output_ids.extend([2, 6])
                closed = apply(prepare())
                self.assertAlmostEqual(closed[0, 0, 6].item(), -presence)
                req.output_ids = [1]
                reset = apply(prepare())
                torch.testing.assert_close(reset, initial, atol=0, rtol=0)

    def test_native_release_does_not_leave_a_clover_gpu_buffer_owner(self):
        import weakref

        batch = NativeBatch([request([1], frequency_penalty=0.5)])
        state = CloverPenaltyState()
        buffer_ref = weakref.ref(state.sync_native(batch.orchestrator)[0])
        batch.orchestrator.release()
        self.assertIsNone(buffer_ref())

    def test_min_new_tokens_rejected(self):
        error = validate_clover_request(request(min_new_tokens=1))
        self.assertIn("min_new_tokens", error)
        self.assertIsNone(
            validate_clover_request(
                request(min_new_tokens=0, custom_params={"dllm_think_end_bias": 0})
            )
        )

    def test_committed_counts_only_and_additive_before_scaling(self):
        req = request(
            [1, 1, 2], frequency_penalty=0.5, presence_penalty=1, repetition_penalty=2
        )
        logits = torch.tensor([[[4.0, 4.0, -2.0, 4.0, 4.0, 4.0, 4.0, 4.0]]]).repeat(
            1, 4, 1
        )
        apply_clover_penalties(logits, [req], torch.tensor([[2, 3, 3, 0]]))
        expected = torch.tensor([4.0, 1.0, -7.0, 4.0, 4.0, 4.0, 4.0, 4.0])
        torch.testing.assert_close(logits[0], expected.expand(4, -1))
        self.assertEqual(req.output_ids, [1, 1, 2])

    def test_length_ramp_alignment_stop_union(self):
        req = request(
            [1, 2, 3],
            stop_token_ids=[6],
            custom_params={
                "dllm_length_penalty": 8.0,
                "dllm_length_penalty_start": 3,
                "dllm_length_penalty_window": 4,
            },
        )
        preview = torch.zeros(1, 4, 8)
        verify = torch.zeros_like(preview)
        visible = torch.tensor([[3, 0, 0, 0]])
        apply_clover_penalties(preview, [req], visible, preview=True)
        apply_clover_penalties(verify, [req], visible)
        torch.testing.assert_close(preview[:, 1:], verify[:, :-1])
        torch.testing.assert_close(verify[0, :, 7], torch.tensor([2.0, 4.0, 6.0, 8.0]))
        torch.testing.assert_close(verify[0, :, 6], verify[0, :, 7])
        self.assertFalse(torch.isnan(verify).any())
        free = request(
            [1, 2, 3],
            ignore_eos=True,
            stop_token_ids=[6],
            custom_params=req.sampling_params.custom_params,
        )
        unbiased = torch.zeros_like(verify)
        apply_clover_penalties(unbiased, [free], visible)
        self.assertFalse(unbiased[0, :, 6:].any())

    def test_think_bias_first_position_and_committed_close(self):
        req = request(
            [1, 2],
            custom_params={
                "dllm_think_end_bias": 4.0,
                "dllm_think_end_start": 2,
                "dllm_think_end_window": 4,
                "dllm_think_end_token_id": 6,
            },
        )
        for preview, visible, expected in (
            (True, [2, 0, 0, 0], [0.0, 1.0, 0.0, 0.0]),
            (True, [2, 6, 0, 0], [0.0, 0.0, 0.0, 0.0]),
            (False, [2, 6, 3, 4], [1.0, 0.0, 0.0, 0.0]),
        ):
            logits = torch.zeros(1, 4, 8)
            apply_clover_penalties(
                logits, [req], torch.tensor([visible]), preview=preview
            )
            torch.testing.assert_close(logits[0, :, 6], torch.tensor(expected))
        req.output_ids.append(6)
        logits = torch.zeros(1, 4, 8)
        apply_clover_penalties(logits, [req], torch.tensor([[6, 1, 2, 3]]))
        self.assertEqual(logits.count_nonzero().item(), 0)

    def test_invalid_custom_controls_rejected(self):
        for key, value in (
            ("dllm_length_penalty", float("nan")),
            ("dllm_think_end_bias", -1),
            ("dllm_length_penalty_start", 1.5),
            ("dllm_length_penalty_window", 0),
            ("dllm_think_end_token_id", -1),
            ("dllm_think_end_bias", 1),  # No tokenizer or explicit token.
        ):
            with self.subTest(key=key, value=value):
                self.assertIsNotNone(
                    validate_clover_request(
                        request(custom_params={"dllm_think_end_bias": 0, key: value})
                    )
                )


if __name__ == "__main__":
    unittest.main()
