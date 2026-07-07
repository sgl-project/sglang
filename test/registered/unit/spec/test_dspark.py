"""Unit tests for DSpark speculative-decoding registration and arg handling."""

import json
import os
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

from sglang.srt.arg_groups.speculative_hook import _handle_dspark
from sglang.srt.speculative.spec_info import SpecInputType, SpeculativeAlgorithm
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class TestDSparkPredicates(CustomTestCase):
    """Lock the DSpark predicate truth table that the scheduler / model_runner /
    attention backends dispatch on."""

    def setUp(self):
        self.algo = SpeculativeAlgorithm.DSPARK

    def test_identity_predicates(self):
        self.assertTrue(self.algo.is_dspark())
        self.assertTrue(self.algo.is_some())
        self.assertTrue(self.algo.is_speculative())
        for other in (
            self.algo.is_none,
            self.algo.is_eagle,
            self.algo.is_eagle3,
            self.algo.is_frozen_kv_mtp,
            self.algo.is_dflash,
            self.algo.is_standalone,
            self.algo.is_ngram,
        ):
            self.assertFalse(other(), other.__name__)

    def test_capability_predicates(self):
        # Block drafter verified by the target (like DFLASH), with its own draft
        # KV chains and target hidden states carried in, and no topk tree.
        self.assertTrue(self.algo.supports_target_verify_for_draft())
        self.assertTrue(self.algo.has_draft_kv())
        self.assertTrue(self.algo.carries_draft_hidden_states())
        self.assertFalse(self.algo.need_topk())

    def test_from_string_resolves(self):
        self.assertIs(
            SpeculativeAlgorithm.from_string("DSPARK"), SpeculativeAlgorithm.DSPARK
        )
        self.assertIs(
            SpeculativeAlgorithm.from_string("dspark"), SpeculativeAlgorithm.DSPARK
        )

    def test_spec_input_types_classified(self):
        from sglang.srt.speculative.spec_info import SpecInput

        draft = SimpleNamespace(spec_input_type=SpecInputType.DSPARK_DRAFT)
        verify = SimpleNamespace(spec_input_type=SpecInputType.DSPARK_VERIFY)
        self.assertTrue(SpecInput.is_draft_input(draft))
        self.assertTrue(SpecInput.is_verify_input(verify))
        self.assertFalse(SpecInput.is_verify_input(draft))
        self.assertFalse(SpecInput.is_draft_input(verify))


def _make_server_args(**overrides):
    base = dict(
        enable_dp_attention=False,
        enable_dp_lm_head=False,
        pp_size=1,
        speculative_draft_model_path="deepseek-ai/DeepSeek-V4-Flash-DSpark",
        speculative_draft_model_revision="main",
        model_path="deepseek-ai/DeepSeek-V4-Flash-DSpark",
        revision="main",
        speculative_num_steps=None,
        speculative_eagle_topk=None,
        speculative_dspark_block_size=5,
        speculative_num_draft_tokens=None,
        speculative_dspark_confidence_threshold=0.0,
        max_running_requests=None,
        enable_mixed_chunk=False,
        trust_remote_code=False,
        json_model_override_args="{}",
    )
    base.update(overrides)
    return SimpleNamespace(**base)


class TestHandleDspark(CustomTestCase):
    def test_pins_steps_and_topk_and_block_size(self):
        args = _make_server_args()
        _handle_dspark(args)
        self.assertEqual(args.speculative_num_steps, 1)
        self.assertEqual(args.speculative_eagle_topk, 1)
        self.assertEqual(args.speculative_num_draft_tokens, 5)
        self.assertEqual(args.max_running_requests, 48)

    def test_block_size_alias_conflict_raises(self):
        args = _make_server_args(
            speculative_dspark_block_size=5, speculative_num_draft_tokens=8
        )
        with self.assertRaisesRegex(ValueError, "must match"):
            _handle_dspark(args)

    def test_nonpositive_block_size_raises(self):
        args = _make_server_args(speculative_dspark_block_size=0)
        with self.assertRaisesRegex(ValueError, "positive"):
            _handle_dspark(args)

    def test_confidence_threshold_bounds(self):
        args = _make_server_args(speculative_dspark_confidence_threshold=1.5)
        with self.assertRaisesRegex(ValueError, "confidence-threshold"):
            _handle_dspark(args)

    def test_draft_path_defaults_to_target(self):
        args = _make_server_args(
            speculative_draft_model_path=None,
            speculative_draft_model_revision=None,
            model_path="my/target",
            revision="abc",
        )
        _handle_dspark(args)
        self.assertEqual(args.speculative_draft_model_path, "my/target")
        self.assertEqual(args.speculative_draft_model_revision, "abc")

    def test_dp_attention_enables_dp_lm_head(self):
        args = _make_server_args(enable_dp_attention=True)
        _handle_dspark(args)
        self.assertTrue(args.enable_dp_lm_head)

    def test_pipeline_parallel_rejected(self):
        args = _make_server_args(pp_size=2)
        with self.assertRaisesRegex(ValueError, "pp_size"):
            _handle_dspark(args)

    def test_mixed_chunk_disabled(self):
        args = _make_server_args(enable_mixed_chunk=True)
        _handle_dspark(args)
        self.assertFalse(args.enable_mixed_chunk)


class TestDSparkDraftInputBatch(CustomTestCase):
    def setUp(self):
        try:
            import torch

            from sglang.srt.speculative.dspark_info import DSparkDraftInputV2
        except Exception as e:  # pragma: no cover - GPU-only deps on some runners
            self.skipTest(f"dspark_info unavailable on this runner: {e}")
        self.torch = torch
        self.cls = DSparkDraftInputV2

    def _make(self, bs, with_cur=True):
        t = self.torch
        return self.cls(
            bonus_tokens=t.arange(bs, dtype=t.int64),
            new_seq_lens=t.full((bs,), 10, dtype=t.int64),
            cur_allocated_seq_lens_cpu=(
                t.full((bs,), 12, dtype=t.int32) if with_cur else None
            ),
        )

    def test_merge_then_filter_asymmetric_cur_allocated(self):
        # Regression: merging an input without cur_allocated_seq_lens_cpu into one
        # that has it must drop it to None rather than leave a stale-sized array that
        # filter_batch then indexes out of bounds (the serving crash in merge_batch).
        a = self._make(2, with_cur=True)
        b = self._make(1, with_cur=False)
        a.merge_batch(b)
        self.assertEqual(len(a.bonus_tokens), 3)
        self.assertIsNone(a.cur_allocated_seq_lens_cpu)
        a.filter_batch(self.torch.tensor([0, 2], dtype=self.torch.int64))
        self.assertEqual(len(a.bonus_tokens), 2)

    def test_filter_batch_indexes_cur_allocated(self):
        a = self._make(4, with_cur=True)
        a.filter_batch(self.torch.tensor([1, 3], dtype=self.torch.int64))
        self.assertEqual(a.bonus_tokens.tolist(), [1, 3])
        self.assertEqual(len(a.cur_allocated_seq_lens_cpu), 2)

    def test_overlap_placeholders_inert_without_future_indices(self):
        a = self._make(2)
        self.assertTrue(a.direct_carry_valid)
        self.assertIsNone(a.future_indices)
        self.assertEqual(a.topk_p.numel(), 0)
        a.filter_batch(self.torch.tensor([1], dtype=self.torch.int64))
        self.assertEqual(len(a.bonus_tokens), 1)

    def test_transfer_warmup_rounds_survive_batch_ops(self):
        t = self.torch
        a = self.cls(
            bonus_tokens=t.tensor([10, 11], dtype=t.int64),
            new_seq_lens=t.tensor([100, 101], dtype=t.int64),
            transfer_warmup_rounds=t.tensor([2, 1], dtype=t.int32),
        )
        b = self.cls(
            bonus_tokens=t.tensor([12], dtype=t.int64),
            new_seq_lens=t.tensor([102], dtype=t.int64),
            transfer_warmup_rounds=t.tensor([0], dtype=t.int32),
        )
        a.merge_batch(b)
        self.assertEqual(a.transfer_warmup_rounds.tolist(), [2, 1, 0])

        a.filter_batch(t.tensor([0, 2], dtype=t.int64))
        self.assertEqual(a.bonus_tokens.tolist(), [10, 12])
        self.assertEqual(a.transfer_warmup_rounds.tolist(), [2, 0])


class TestDSparkPrefillHandoff(CustomTestCase):
    def setUp(self):
        try:
            import torch

            from sglang.srt.speculative.dspark_worker_v2 import DSparkWorkerV2
        except Exception as e:  # pragma: no cover - GPU-only deps on some runners
            self.skipTest(f"dspark worker unavailable on this runner: {e}")
        self.torch = torch
        self.worker_cls = DSparkWorkerV2

    def _worker(self, prefill_transfer_warmup_rounds):
        worker = self.worker_cls.__new__(self.worker_cls)
        worker._prefill_transfer_warmup_rounds = prefill_transfer_warmup_rounds
        return worker

    def _accept_worker(self, *, use_confidence_gate=False, threshold=0.5):
        worker = self._worker(0)
        worker._use_confidence_gate = use_confidence_gate
        worker.confidence_threshold = threshold
        return worker

    def _make_prefill_input(self, rounds):
        t = self.torch
        worker = self._worker(rounds)
        return self.worker_cls._make_next_draft_input_prefill(
            worker,
            bonus_tokens=t.tensor([1262, 2808], dtype=t.int32),
            seq_lens=t.tensor([3676, 42], dtype=t.int32),
        )

    def test_prefill_handoff_matches_deepspec_reference_no_warmup(self):
        # DeepSpec first proposal uses prompt context hidden plus the
        # prefill-sampled anchor immediately. In SGLang that contract is encoded
        # as transfer_warmup_rounds == 0 for the handoff draft input.
        draft_input = self._make_prefill_input(rounds=0)
        self.assertEqual(draft_input.bonus_tokens.tolist(), [1262, 2808])
        self.assertEqual(draft_input.new_seq_lens.tolist(), [3676, 42])
        self.assertEqual(draft_input.transfer_warmup_rounds.tolist(), [0, 0])
        self.assertEqual(
            tuple(draft_input.prefill_tail_hidden_states.shape), (0, 0, 0)
        )
        self.assertEqual(tuple(draft_input.prefill_tail_valid_mask.shape), (0, 0))

    def test_service_warmup_prefill_handoff_uses_configured_rounds(self):
        draft_input = self._make_prefill_input(rounds=2)
        self.assertEqual(draft_input.transfer_warmup_rounds.tolist(), [2, 2])

        recovered = self.worker_cls._get_transfer_warmup_rounds(
            self._worker(0),
            draft_input,
            bs=2,
            device=self.torch.device("cpu"),
        )
        self.assertEqual(recovered.tolist(), [2, 2])

    def test_mismatched_warmup_shape_falls_back_to_zero(self):
        draft_input = self._make_prefill_input(rounds=1)
        draft_input.transfer_warmup_rounds = self.torch.tensor(
            [1], dtype=self.torch.int32
        )
        recovered = self.worker_cls._get_transfer_warmup_rounds(
            self._worker(0),
            draft_input,
            bs=2,
            device=self.torch.device("cpu"),
        )
        self.assertEqual(recovered.tolist(), [0, 0])

    def test_prefill_tail_hidden_packs_variable_length_batch(self):
        t = self.torch
        hidden = t.arange(10 * 2, dtype=t.float32).view(10, 2)
        tail_hidden, tail_mask = self.worker_cls._pack_prefill_tail_hidden(
            self._worker(0),
            hidden=hidden,
            extend_lens=[3, 0, 7],
        )

        self.assertEqual(tuple(tail_hidden.shape), (3, 7, 2))
        self.assertEqual(tuple(tail_mask.shape), (3, 7))
        self.assertEqual(
            tail_mask[0].tolist(), [False, False, False, False, True, True, True]
        )
        self.assertEqual(tail_mask[1].tolist(), [False] * 7)
        self.assertEqual(tail_mask[2].tolist(), [True] * 7)
        self.assertTrue(t.equal(tail_hidden[0, 4:], hidden[:3]))
        self.assertTrue(t.equal(tail_hidden[1], t.zeros_like(tail_hidden[1])))
        self.assertTrue(t.equal(tail_hidden[2], hidden[3:10]))

    def test_prefill_tail_hidden_caps_to_last_128_tokens(self):
        t = self.torch
        hidden = t.arange(130, dtype=t.float32).view(130, 1)
        tail_hidden, tail_mask = self.worker_cls._pack_prefill_tail_hidden(
            self._worker(0),
            hidden=hidden,
            extend_lens=[130],
        )

        self.assertEqual(tuple(tail_hidden.shape), (1, 128, 1))
        self.assertTrue(tail_mask.all())
        self.assertEqual(tail_hidden[:, 0, 0].item(), 2.0)
        self.assertEqual(tail_hidden[:, -1, 0].item(), 129.0)

    def test_decode_anchor_tokens_mix_prompt_output_and_fallback(self):
        t = self.torch
        worker = self._worker(0)
        worker.noise_token_id = 999
        batch = SimpleNamespace(
            reqs=[
                SimpleNamespace(origin_input_ids=[10, 11, 12], output_ids=[]),
                SimpleNamespace(origin_input_ids=[20, 21], output_ids=[200, 201]),
                SimpleNamespace(origin_input_ids=[30, 31], output_ids=[]),
                SimpleNamespace(origin_input_ids=[], output_ids=[]),
            ],
            seq_lens_cpu=None,
        )
        anchors = self.worker_cls._get_decode_anchor_tokens(
            worker,
            batch=batch,
            prefix_lens=t.tensor([3, 4, 5, 0], dtype=t.int64),
            fallback_tokens=t.tensor([1000, 1001, 1002, 1003], dtype=t.int64),
            prefer_fallback_tokens=t.tensor([False, False, False, True]),
            bs=4,
            device=t.device("cpu"),
        )

        self.assertEqual(anchors.tolist(), [12, 201, 1002, 1003])

    def test_decode_next_input_carries_warmup_rounds(self):
        t = self.torch
        draft_input = self.worker_cls._make_next_draft_input_decode(
            self._worker(0),
            bonus_tokens=t.tensor([301, 302], dtype=t.int32),
            new_seq_lens=t.tensor([3681, 9], dtype=t.int32),
            transfer_warmup_rounds=t.tensor([1, 0], dtype=t.int64),
        )

        self.assertEqual(draft_input.bonus_tokens.dtype, t.int64)
        self.assertEqual(draft_input.new_seq_lens.dtype, t.int64)
        self.assertEqual(draft_input.transfer_warmup_rounds.dtype, t.int32)
        self.assertEqual(draft_input.bonus_tokens.tolist(), [301, 302])
        self.assertEqual(draft_input.new_seq_lens.tolist(), [3681, 9])
        self.assertEqual(draft_input.transfer_warmup_rounds.tolist(), [1, 0])

    def test_accept_bonus_uses_longest_contiguous_draft_match(self):
        t = self.torch
        candidates = t.tensor(
            [
                [10, 20, 30, 40],
                [11, 21, 31, 41],
            ],
            dtype=t.int64,
        )
        target_predict = t.tensor(
            [
                [20, 30, 99, 77],
                [21, 99, 31, 88],
            ],
            dtype=t.int64,
        )
        confidence = t.zeros((2, 3), dtype=t.float32)

        commit_lens, bonus_tokens, out_tokens = (
            self.worker_cls._compute_accept_bonus_eager(
                self._accept_worker(),
                candidates=candidates,
                target_predict=target_predict,
                confidence=confidence,
            )
        )

        self.assertEqual(commit_lens.tolist(), [3, 2])
        self.assertEqual(bonus_tokens.tolist(), [99, 99])
        self.assertEqual(out_tokens.tolist(), [[20, 30, 99, 0], [21, 99, 31, 0]])
        if os.getenv("SGLANG_DSPARK_TEST_VERBOSE", "0") == "1":
            print(
                "DSpark accept parity debug: "
                + json.dumps(
                    {
                        "case": "longest_contiguous_match",
                        "candidates": candidates.tolist(),
                        "target_predict": target_predict.tolist(),
                        "commit_lens": commit_lens.tolist(),
                        "bonus_tokens": bonus_tokens.tolist(),
                        "out_tokens": out_tokens.tolist(),
                    },
                    sort_keys=True,
                ),
                flush=True,
            )

    def test_accept_bonus_confidence_gate_truncates_match_prefix(self):
        t = self.torch
        candidates = t.tensor([[10, 20, 30, 40]], dtype=t.int64)
        target_predict = t.tensor([[20, 30, 40, 99]], dtype=t.int64)
        confidence = t.tensor([[10.0, -10.0, 10.0]], dtype=t.float32)

        commit_lens, bonus_tokens, out_tokens = (
            self.worker_cls._compute_accept_bonus_eager(
                self._accept_worker(use_confidence_gate=True, threshold=0.5),
                candidates=candidates,
                target_predict=target_predict,
                confidence=confidence,
            )
        )

        self.assertEqual(commit_lens.tolist(), [2])
        self.assertEqual(bonus_tokens.tolist(), [30])
        self.assertEqual(out_tokens.tolist(), [[20, 30, 40, 0]])

    def test_accept_bonus_rejects_mismatched_shapes(self):
        t = self.torch
        worker = self._accept_worker()
        candidates = t.tensor([[10, 20, 30]], dtype=t.int64)
        target_predict = t.tensor([[20, 30]], dtype=t.int64)
        confidence = t.zeros((1, 2), dtype=t.float32)

        with self.assertRaisesRegex(ValueError, "target_predict must match"):
            self.worker_cls._compute_accept_bonus_eager(
                worker,
                candidates=candidates,
                target_predict=target_predict,
                confidence=confidence,
            )

        with self.assertRaisesRegex(ValueError, "confidence must have"):
            self.worker_cls._compute_accept_bonus_eager(
                worker,
                candidates=candidates,
                target_predict=candidates,
                confidence=t.zeros((1, 1), dtype=t.float32),
            )


class TestDSparkDeepSpecSemanticReference(CustomTestCase):
    def setUp(self):
        try:
            import torch
        except Exception as e:  # pragma: no cover - torch unavailable on some runners
            self.skipTest(f"torch unavailable on this runner: {e}")
        self.torch = torch

    def test_materialized_context_matches_deepspec_concat_when_kv_is_equivalent(self):
        # This is the algebra DSpark relies on:
        #   DeepSpec: K/V = K/V(context hidden) + K/V(anchor/noise block)
        #   SGLang:   write K/V(context hidden), then run anchor/noise block
        # If the same projections, positions, masks and cache cleanup are used,
        # the block output should be identical.
        t = self.torch
        t.manual_seed(0)
        batch_size = 1
        context_len = 4
        block_size = 3
        hidden_size = 8
        head_dim = 8

        context_hidden = t.randn(batch_size, context_len, hidden_size)
        block_hidden = t.randn(batch_size, block_size, hidden_size)
        wq = t.randn(hidden_size, head_dim)
        wk = t.randn(hidden_size, head_dim)
        wv = t.randn(hidden_size, head_dim)

        def attention(q, k, v):
            scores = q @ k.transpose(-1, -2) / (head_dim**0.5)
            probs = t.softmax(scores, dim=-1)
            return probs @ v

        q_block = block_hidden @ wq
        k_context = context_hidden @ wk
        v_context = context_hidden @ wv
        k_block = block_hidden @ wk
        v_block = block_hidden @ wv

        deepspec_k = t.cat([k_context, k_block], dim=1)
        deepspec_v = t.cat([v_context, v_block], dim=1)
        deepspec_out = attention(q_block, deepspec_k, deepspec_v)

        materialized_k_cache = k_context.clone()
        materialized_v_cache = v_context.clone()
        sglang_k = t.cat([materialized_k_cache, k_block], dim=1)
        sglang_v = t.cat([materialized_v_cache, v_block], dim=1)
        sglang_out = attention(q_block, sglang_k, sglang_v)

        self.assertTrue(t.equal(deepspec_out, sglang_out))

    def test_markov_refine_matches_deepspec_vanilla_markov_reference(self):
        repo_root = Path(__file__).resolve().parents[5]
        deepspec_root = repo_root / "DeepSpec"
        if not deepspec_root.exists():
            self.skipTest(f"DeepSpec checkout not found: {deepspec_root}")
        sys.path.insert(0, str(deepspec_root))
        try:
            from deepspec.modeling.dspark.markov_head import VanillaMarkov
            from sglang.srt.speculative.dspark_worker_v2 import DSparkWorkerV2
        except Exception as e:
            self.skipTest(f"DeepSpec/SGLang DSpark imports unavailable: {e}")

        t = self.torch
        t.manual_seed(0)
        bs = 2
        block_size = 4
        verify_stride = block_size + 1
        hidden_size = 6
        vocab_size = 19
        markov_rank = 5

        block_hidden = t.randn(bs, block_size, hidden_size)
        anchor_tokens = t.tensor([3, 7], dtype=t.int64)
        lm_head = t.nn.Linear(hidden_size, vocab_size, bias=False)
        markov_head = VanillaMarkov(vocab_size=vocab_size, markov_rank=markov_rank)

        class _LogitsProcessor:
            def _compute_lm_head(self, hidden_states, head):
                return head(hidden_states)

        class _ConfidenceHead:
            def __call__(self, hidden_states, markov_embeds):
                return hidden_states.new_zeros(hidden_states.shape[:2])

        worker = DSparkWorkerV2.__new__(DSparkWorkerV2)
        worker.block_size = block_size
        worker.verify_stride = verify_stride
        worker.markov_rank = markov_rank
        worker.noise_token_id = 0
        worker._markov_refine_buffer_cap = 0
        worker._markov_candidates_buf = None
        worker._markov_embeds_buf = None
        worker._vocab_shard_mapping_cache = {}
        worker._accept_anomaly_enabled = False
        worker._parity_dump_enabled = False
        worker._last_markov_refine_debug = None
        worker._draft_inner = SimpleNamespace(
            vocab_size=vocab_size,
            markov_head=markov_head,
            confidence_head=_ConfidenceHead(),
        )
        worker.draft_model = SimpleNamespace(
            lm_head=lm_head,
            logits_processor=_LogitsProcessor(),
        )

        candidates, confidence = DSparkWorkerV2._refine_block_markov(
            worker,
            block_hidden=block_hidden,
            anchor_tokens=anchor_tokens,
        )

        base_logits = lm_head(block_hidden)
        ref_candidates = t.empty(bs, verify_stride, dtype=t.int64)
        ref_candidates[:, 0] = anchor_tokens
        prev_tokens = anchor_tokens
        for step in range(block_size):
            bias = markov_head.markov_w2(markov_head.get_prev_embeddings(prev_tokens))
            next_tokens = t.argmax(base_logits[:, step, :] + bias, dim=-1)
            ref_candidates[:, step + 1] = next_tokens
            prev_tokens = next_tokens

        parity_match = t.equal(candidates, ref_candidates)
        if os.getenv("SGLANG_DSPARK_TEST_VERBOSE", "0") == "1":
            print(
                "DSpark Markov parity debug: "
                + json.dumps(
                    {
                        "bs": bs,
                        "block_size": block_size,
                        "verify_stride": verify_stride,
                        "hidden_size": hidden_size,
                        "vocab_size": vocab_size,
                        "markov_rank": markov_rank,
                        "anchor_tokens": anchor_tokens.tolist(),
                        "sglang_candidates": candidates.tolist(),
                        "deepspec_reference_candidates": ref_candidates.tolist(),
                        "candidate_match": bool(parity_match),
                        "confidence_shape": list(confidence.shape),
                        "confidence_abs_sum": float(confidence.abs().sum().item()),
                        "base_top1": t.argmax(base_logits, dim=-1).tolist(),
                    },
                    sort_keys=True,
                ),
                flush=True,
            )

        self.assertEqual(candidates.tolist(), ref_candidates.tolist())
        self.assertEqual(tuple(confidence.shape), (bs, block_size))
        self.assertTrue(t.equal(confidence, t.zeros_like(confidence)))


if __name__ == "__main__":
    unittest.main(verbosity=3)
