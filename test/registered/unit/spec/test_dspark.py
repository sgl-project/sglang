"""Unit tests for DSpark speculative-decoding registration and arg handling."""

import unittest
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
        self.assertTrue(self.algo.supports_target_verify_for_draft())
        self.assertTrue(self.algo.has_draft_kv())
        self.assertFalse(self.algo.carries_draft_hidden_states())
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
        disable_overlap_schedule=False,
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

    def test_dp_attention_rejected(self):
        args = _make_server_args(enable_dp_attention=True)
        with self.assertRaisesRegex(ValueError, "dp attention"):
            _handle_dspark(args)

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

    def _make(self, bs):
        t = self.torch
        return self.cls(
            bonus_tokens=t.arange(bs, dtype=t.int64),
            new_seq_lens=t.full((bs,), 10, dtype=t.int64),
        )

    def test_merge_then_filter_concatenates_and_indexes(self):
        a = self._make(2)
        b = self._make(1)
        a.merge_batch(b)
        self.assertEqual(len(a.bonus_tokens), 3)
        self.assertEqual(len(a.new_seq_lens), 3)
        a.filter_batch(self.torch.tensor([0, 2], dtype=self.torch.int64))
        self.assertEqual(len(a.bonus_tokens), 2)

    def test_filter_batch_indexes_tensors(self):
        a = self._make(4)
        a.filter_batch(self.torch.tensor([1, 3], dtype=self.torch.int64))
        self.assertEqual(a.bonus_tokens.tolist(), [1, 3])
        self.assertEqual(a.new_seq_lens.tolist(), [10, 10])

    def test_overlap_placeholders_inert_without_future_indices(self):
        a = self._make(2)
        self.assertTrue(a.direct_carry_valid)
        self.assertIsNone(a.future_indices)
        self.assertEqual(a.topk_p.numel(), 0)
        a.filter_batch(self.torch.tensor([1], dtype=self.torch.int64))
        self.assertEqual(len(a.bonus_tokens), 1)


class TestDSparkRequestValidation(CustomTestCase):
    """validate_dspark_request rejects features DSpark cannot serve and accepts a
    plain request. Kept import-light: requests are duck-typed namespaces."""

    def setUp(self):
        try:
            from sglang.srt.speculative.dspark_info import validate_dspark_request
        except Exception as e:  # pragma: no cover - GPU-only deps on some runners
            self.skipTest(f"dspark_info unavailable on this runner: {e}")
        self.validate = validate_dspark_request

    @staticmethod
    def _make_req(**overrides):
        sampling = SimpleNamespace(
            json_schema=None, regex=None, ebnf=None, structural_tag=None
        )
        base = dict(
            return_logprob=False,
            return_hidden_states=False,
            sampling_params=sampling,
        )
        base.update(overrides)
        return SimpleNamespace(**base)

    def test_accepts_plain_request(self):
        self.assertIsNone(self.validate(self._make_req()))

    def test_rejects_return_logprob(self):
        msg = self.validate(self._make_req(return_logprob=True))
        self.assertIsNotNone(msg)
        self.assertIn("return_logprob", msg)

    def test_rejects_return_hidden_states(self):
        msg = self.validate(self._make_req(return_hidden_states=True))
        self.assertIsNotNone(msg)
        self.assertIn("return_hidden_states", msg)

    def test_rejects_grammar_constrained(self):
        for field_name in ("json_schema", "regex", "ebnf", "structural_tag"):
            sampling = SimpleNamespace(
                json_schema=None, regex=None, ebnf=None, structural_tag=None
            )
            setattr(sampling, field_name, "x")
            msg = self.validate(self._make_req(sampling_params=sampling))
            self.assertIsNotNone(msg, field_name)
            self.assertIn("grammar", msg)


class _DSparkMathBase(CustomTestCase):
    """Base for the DSpark block-verify commit-math tests.

    Imports torch lazily so the file collects on torch-less runners, and holds
    python replicas of the pure commit helpers from dspark_worker_v2.
    """

    def setUp(self):
        try:
            import torch
        except Exception as e:  # pragma: no cover - torch-less runner
            self.skipTest(f"torch unavailable on this runner: {e}")
        self.torch = torch

    def _confident_prefix(self, confidence, threshold):
        # Replica of DSparkWorkerV2._confident_prefix: leading run of sigmoid >= threshold
        # (cumprod zeroes everything after the first fail).
        keep = self.torch.sigmoid(confidence) >= threshold
        return keep.to(self.torch.int32).cumprod(dim=1).sum(dim=1)

    def _assemble_out_tokens(self, candidates, correct_len, bonus_tokens):
        # Replica of the out_tokens assembly in _forward_decode: drafts candidates[:, 1:]
        # then a 0 pad, with the bonus scattered at correct_len -> prefix [drafts.., bonus].
        t = self.torch
        bs, block_size = candidates.shape
        out_tokens = t.empty((bs, block_size), dtype=t.int64)
        if block_size > 1:
            out_tokens[:, : block_size - 1].copy_(candidates[:, 1:])
        out_tokens[:, block_size - 1].fill_(0)
        out_tokens.scatter_(
            1, correct_len.unsqueeze(1), bonus_tokens.unsqueeze(1).to(t.int64)
        )
        return out_tokens


class TestDSparkGreedyVerifyMath(_DSparkMathBase):
    """Lock in the greedy verify branch of _forward_decode: correct_len from
    compute_dflash_correct_drafts_and_bonus, the confidence min-cap, the bonus
    gather, out_tokens assembly, and commit_lens. Every value is cross-checked
    against a brute-force per-row Python reference."""

    def setUp(self):
        super().setUp()
        try:
            from sglang.srt.speculative.dflash_utils import (
                compute_dflash_correct_drafts_and_bonus,
            )
        except Exception as e:  # pragma: no cover - GPU-only deps on some runners
            self.skipTest(f"dflash_utils unavailable on this runner: {e}")
        self._compute_correct = compute_dflash_correct_drafts_and_bonus

    # sigmoid(x) >= 0.5  <=>  x >= 0, so PASS/FAIL confidence values drive the
    # confident-prefix deterministically at threshold 0.5.
    THRESH = 0.5
    PASS = 10.0
    FAIL = -10.0

    def _greedy_pipeline(self, candidates, target_predict, confidence, threshold):
        t = self.torch
        raw_correct, _ = self._compute_correct(
            candidates=candidates, target_predict=target_predict
        )
        confident_prefix = self._confident_prefix(confidence, threshold)
        correct_len = t.minimum(raw_correct.to(t.int64), confident_prefix.to(t.int64))
        bonus_tokens = target_predict.gather(1, correct_len.unsqueeze(1)).squeeze(1)
        commit_lens = correct_len.to(t.int32) + 1
        out_tokens = self._assemble_out_tokens(candidates, correct_len, bonus_tokens)
        return correct_len, bonus_tokens, commit_lens, out_tokens

    @staticmethod
    def _bruteforce(candidates, target_predict, confidence, threshold):
        import math

        bs, block_size = candidates.shape
        correct_len, bonus, commit, out = [], [], [], []
        for r in range(bs):
            cand = candidates[r].tolist()
            tgt = target_predict[r].tolist()
            conf = confidence[r].tolist()
            match_run = 0
            for i in range(block_size - 1):
                if cand[i + 1] == tgt[i]:
                    match_run += 1
                else:
                    break
            conf_run = 0
            for x in conf:
                if 1.0 / (1.0 + math.exp(-float(x))) >= threshold:
                    conf_run += 1
                else:
                    break
            cl = min(match_run, conf_run)
            b = tgt[cl]
            row = [0] * block_size
            for k in range(block_size - 1):
                row[k] = cand[k + 1]
            row[cl] = b
            correct_len.append(cl)
            bonus.append(b)
            commit.append(cl + 1)
            out.append(row)
        return correct_len, bonus, commit, out

    def test_mixed_batch_matches_bruteforce(self):
        t = self.torch
        P, F = self.PASS, self.FAIL
        # row0 full accept (drafts 1,2,3 all match)          -> correct_len 3
        # row1 first-token reject (draft 5 != target 6)      -> correct_len 0
        # row2 mid-block reject (draft 7 != target 8 at t=1) -> correct_len 1
        # row3 confidence truncation BELOW match (raw 3, cp1)-> correct_len 1
        # row4 confidence "truncation" ABOVE match (raw1,cp4)-> correct_len 1 (no-op)
        candidates = t.tensor(
            [
                [50, 1, 2, 3],
                [50, 5, 20, 21],
                [50, 1, 7, 22],
                [50, 1, 2, 3],
                [50, 1, 30, 31],
            ],
            dtype=t.int64,
        )
        target_predict = t.tensor(
            [
                [1, 2, 3, 99],
                [6, 60, 61, 62],
                [1, 8, 70, 71],
                [1, 2, 3, 88],
                [1, 40, 41, 42],
            ],
            dtype=t.int64,
        )
        confidence = t.tensor(
            [
                [P, P, P, P],
                [P, P, P, P],
                [P, P, P, P],
                [P, F, F, F],
                [P, P, P, P],
            ],
            dtype=t.float32,
        )
        correct_len, bonus, commit, out = self._greedy_pipeline(
            candidates, target_predict, confidence, self.THRESH
        )
        bf_cl, bf_bonus, bf_commit, bf_out = self._bruteforce(
            candidates, target_predict, confidence, self.THRESH
        )
        self.assertEqual(correct_len.tolist(), bf_cl)
        self.assertEqual(bonus.tolist(), bf_bonus)
        self.assertEqual(commit.tolist(), bf_commit)
        self.assertEqual(out.tolist(), bf_out)
        # Explicit expected values (validated numerically offline).
        self.assertEqual(correct_len.tolist(), [3, 0, 1, 1, 1])
        self.assertEqual(bonus.tolist(), [99, 6, 8, 2, 40])
        self.assertEqual(commit.tolist(), [4, 1, 2, 2, 2])
        # Committed prefix of every row is exactly [drafts.., bonus].
        for r in range(candidates.shape[0]):
            cl = int(correct_len[r])
            committed = out[r, : cl + 1].tolist()
            expected = candidates[r, 1 : cl + 1].tolist() + [int(bonus[r])]
            self.assertEqual(committed, expected)

    def test_block_size_one_edge(self):
        t = self.torch
        # block_size == 1: there are no drafts, so correct_len is always 0 and
        # the single committed token is the target's bonus at column 0.
        candidates = t.tensor([[50], [51]], dtype=t.int64)
        target_predict = t.tensor([[7], [8]], dtype=t.int64)
        confidence = t.tensor([[self.PASS], [self.FAIL]], dtype=t.float32)
        correct_len, bonus, commit, out = self._greedy_pipeline(
            candidates, target_predict, confidence, self.THRESH
        )
        self.assertEqual(correct_len.tolist(), [0, 0])
        self.assertEqual(bonus.tolist(), [7, 8])
        self.assertEqual(commit.tolist(), [1, 1])
        self.assertEqual(out.tolist(), [[7], [8]])

    def test_threshold_zero_disables_truncation(self):
        t = self.torch
        # With threshold 0, sigmoid(confidence) >= 0 always holds, so the
        # confident prefix never shortens the accepted run: correct_len equals
        # the raw greedy match length even when confidence is all-negative.
        candidates = t.tensor([[50, 1, 2, 3]], dtype=t.int64)
        target_predict = t.tensor([[1, 2, 3, 99]], dtype=t.int64)
        confidence = t.full((1, 4), self.FAIL, dtype=t.float32)
        correct_len, bonus, commit, _ = self._greedy_pipeline(
            candidates, target_predict, confidence, 0.0
        )
        self.assertEqual(correct_len.tolist(), [3])
        self.assertEqual(bonus.tolist(), [99])
        self.assertEqual(commit.tolist(), [4])


class TestDSparkSampledCommitMath(_DSparkMathBase):
    """Post-kernel logic of the SAMPLED verify branch of _forward_decode. The
    kernel result (accept_len, sampled_bonus) is supplied synthetically.

    Losslessness invariant: confidence only shortens the committed block. A
    truncated row commits candidates[correct_len+1], and correct_len+1 <=
    accept_len, so that token is one the kernel already accepted; the output
    distribution is preserved.
    """

    def _sampled_pipeline(
        self, candidates, accept_len, sampled_bonus, confident_prefix, block_size
    ):
        t = self.torch
        accept_len = accept_len.to(t.int64)
        correct_len = t.minimum(accept_len, confident_prefix.to(t.int64))
        truncated = correct_len < accept_len
        idx = (correct_len + 1).clamp(max=block_size - 1)
        next_draft = candidates.gather(1, idx.unsqueeze(1)).squeeze(1).to(t.int64)
        bonus_tokens = t.where(truncated, next_draft, sampled_bonus.to(t.int64))
        commit_lens = correct_len.to(t.int32) + 1
        out_tokens = self._assemble_out_tokens(candidates, correct_len, bonus_tokens)
        return correct_len, truncated, idx, bonus_tokens, commit_lens, out_tokens

    def test_sampled_commit_and_invariants(self):
        t = self.torch
        block_size = 4  # columns: [current, draft_1, draft_2, draft_3]
        candidates = t.tensor(
            [
                [50, 11, 12, 13],  # r0 not truncated (accept 2, cp 4)
                [50, 21, 22, 23],  # r1 truncated (accept 3, cp 1) -> bonus=cand[2]
                [50, 31, 32, 33],  # r2 accept 0, not truncated -> bonus=sampled
                [50, 41, 42, 43],  # r3 truncated (accept 3, cp 2) -> bonus=cand[3]
                [50, 51, 52, 53],  # r4 full accept 3, cp 3 -> bonus=sampled
            ],
            dtype=t.int64,
        )
        accept_len = t.tensor([2, 3, 0, 3, 3], dtype=t.int64)
        sampled_bonus = t.tensor([900, 901, 902, 903, 904], dtype=t.int64)
        confident_prefix = t.tensor([4, 1, 4, 2, 3], dtype=t.int64)

        correct_len, truncated, idx, bonus, commit, out = self._sampled_pipeline(
            candidates, accept_len, sampled_bonus, confident_prefix, block_size
        )

        # Explicit expected values (validated numerically offline).
        self.assertEqual(correct_len.tolist(), [2, 1, 0, 2, 3])
        self.assertEqual(truncated.tolist(), [False, True, False, True, False])
        self.assertEqual(bonus.tolist(), [900, 22, 902, 43, 904])
        self.assertEqual(commit.tolist(), [3, 2, 1, 3, 4])

        for r in range(candidates.shape[0]):
            cl = int(correct_len[r])
            if bool(truncated[r]):
                # (a) the committed bonus is a kernel-accepted token.
                self.assertLessEqual(cl + 1, int(accept_len[r]))
                # (d) the clamp at block_size-1 never changes a truncated index.
                self.assertEqual(int(idx[r]), cl + 1)
                self.assertEqual(int(bonus[r]), int(candidates[r, cl + 1]))
            else:
                # (b) an untruncated row commits the kernel bonus.
                self.assertEqual(int(bonus[r]), int(sampled_bonus[r]))
            # (c) committed prefix equals [drafts.., bonus].
            committed = out[r, : cl + 1].tolist()
            expected = candidates[r, 1 : cl + 1].tolist() + [int(bonus[r])]
            self.assertEqual(committed, expected)

    def test_accept_len_zero_commits_sampled_bonus(self):
        t = self.torch
        block_size = 4
        candidates = t.tensor([[50, 11, 12, 13]], dtype=t.int64)
        accept_len = t.tensor([0], dtype=t.int64)
        sampled_bonus = t.tensor([777], dtype=t.int64)
        confident_prefix = t.tensor([4], dtype=t.int64)
        correct_len, truncated, _, bonus, commit, out = self._sampled_pipeline(
            candidates, accept_len, sampled_bonus, confident_prefix, block_size
        )
        self.assertEqual(correct_len.tolist(), [0])
        self.assertEqual(truncated.tolist(), [False])
        self.assertEqual(bonus.tolist(), [777])
        self.assertEqual(commit.tolist(), [1])
        self.assertEqual(out[0, :1].tolist(), [777])


class TestDSparkConfidentPrefix(_DSparkMathBase):
    """Lock in the leading-run semantics of DSparkWorkerV2._confident_prefix."""

    THRESH = 0.5
    PASS = 10.0
    FAIL = -10.0

    def test_all_above_returns_block_size(self):
        t = self.torch
        conf = t.full((1, 5), self.PASS, dtype=t.float32)
        self.assertEqual(self._confident_prefix(conf, self.THRESH).tolist(), [5])

    def test_all_below_returns_zero(self):
        t = self.torch
        conf = t.full((1, 5), self.FAIL, dtype=t.float32)
        self.assertEqual(self._confident_prefix(conf, self.THRESH).tolist(), [0])

    def test_gap_in_middle_stops_the_run(self):
        t = self.torch
        # A below-threshold position ends the run; later above-threshold
        # positions must NOT be counted.
        conf = t.tensor(
            [[self.PASS, self.PASS, self.FAIL, self.PASS, self.PASS]],
            dtype=t.float32,
        )
        self.assertEqual(self._confident_prefix(conf, self.THRESH).tolist(), [2])

    def test_threshold_zero_returns_block_size(self):
        t = self.torch
        # threshold 0 disables truncation: sigmoid(confidence) >= 0 always holds.
        conf = t.full((1, 5), self.FAIL, dtype=t.float32)
        self.assertEqual(self._confident_prefix(conf, 0.0).tolist(), [5])

    def test_multi_row_batch(self):
        t = self.torch
        conf = t.tensor(
            [
                [self.PASS, self.PASS, self.PASS],
                [self.FAIL, self.PASS, self.PASS],
                [self.PASS, self.FAIL, self.PASS],
            ],
            dtype=t.float32,
        )
        self.assertEqual(self._confident_prefix(conf, self.THRESH).tolist(), [3, 0, 1])


class TestDSparkShardArgmaxPack(_DSparkMathBase):
    """Exactness of the shard-local argmax packing used by the collective-free
    Markov refine: pack (bf16 value, global index) into one int64 whose MAX
    across vocab shards reproduces torch.argmax's first-index tie-break exactly.
    Tests the real helpers from dspark_worker_v2."""

    def setUp(self):
        super().setUp()
        try:
            from sglang.srt.speculative.dspark_worker_v2 import (
                _dspark_decode_index,
                _dspark_pack_value_index,
                _dspark_shard_argmax_pack,
            )
        except Exception as e:  # pragma: no cover - GPU-only deps on some runners
            self.skipTest(f"dspark_worker_v2 unavailable on this runner: {e}")
        self.pack = _dspark_pack_value_index
        self.decode = _dspark_decode_index
        self.shard_pack = _dspark_shard_argmax_pack

    def _sharded_argmax(self, full_padded, real_vocab, tp):
        t = self.torch
        width = full_padded.shape[1] // tp
        merged = None
        for r in range(tp):
            shard = full_padded[:, r * width : (r + 1) * width]
            cols = r * width + t.arange(width)
            mask = cols >= real_vocab
            packed = self.shard_pack(
                shard, r * width, mask if bool(mask.any()) else None
            )
            # Elementwise max across shards simulates the NCCL MAX all-reduce.
            merged = packed if merged is None else t.maximum(merged, packed)
        return self.decode(merged)

    def _check(self, full_padded, real_vocab, tp):
        t = self.torch
        ref = t.argmax(full_padded[:, :real_vocab].float(), dim=-1)
        got = self._sharded_argmax(full_padded, real_vocab, tp)
        self.assertTrue(bool((got == ref).all()), f"tp={tp}: {got} vs {ref}")

    def test_padded_vocab_with_forced_ties(self):
        t = self.torch
        t.manual_seed(7)
        vocab, padded, tp = 1000, 1024, 4
        for _ in range(20):
            x = t.randn(256, padded).to(t.bfloat16)
            # Hostile garbage in the padding columns must never win.
            x[:, vocab:] = t.randn(256, padded - vocab).to(t.bfloat16) * 100
            row_max = x[:, :vocab].max(dim=-1).values
            for _ in range(3):
                idx = t.randint(0, vocab, (256,))
                x[t.arange(256), idx] = row_max
            self._check(x, vocab, tp)

    def test_all_negative_and_signed_zero_ties(self):
        t = self.torch
        t.manual_seed(11)
        vocab, padded = 1000, 1024
        x = (-t.rand(512, padded) - 0.5).to(t.bfloat16)
        zi = t.randint(0, vocab, (512, 2))
        x[t.arange(512), zi[:, 0]] = t.tensor(-0.0, dtype=t.bfloat16)
        x[t.arange(512), zi[:, 1]] = t.tensor(0.0, dtype=t.bfloat16)
        self._check(x, vocab, 4)

    def test_real_vocab_shape_tp8(self):
        t = self.torch
        t.manual_seed(13)
        x = (t.randn(64, 129280) * 4).to(t.bfloat16)
        row_max = x.max(dim=-1).values
        idx = t.randint(0, 129280, (64,))
        x[t.arange(64), idx] = row_max
        self._check(x, 129280, 8)
        self._check(x, 129280, 1)

    def test_key_order_monotone_across_bf16_range(self):
        t = self.torch
        vals = t.tensor(
            [
                float("-inf"),
                -3e38,
                -1.0,
                -1e-38,
                -0.0,
                0.0,
                1e-38,
                1.0,
                3e38,
                float("inf"),
            ],
            dtype=t.bfloat16,
        )
        vals = vals + 0.0  # the -0.0 normalize applied by _dspark_shard_argmax_pack
        keys = self.pack(vals, t.zeros(len(vals), dtype=t.int64))
        self.assertTrue(bool((keys[1:] >= keys[:-1]).all()), keys.tolist())
        self.assertEqual(int(keys[4]), int(keys[5]))


if __name__ == "__main__":
    unittest.main(verbosity=3)
