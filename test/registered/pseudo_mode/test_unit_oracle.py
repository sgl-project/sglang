"""Unit tests for :class:`PseudoOracle`.

Covers prefill / decode / chunked-prefill / preempt-resume / overlap /
spec-draft-step / EOS / cross-rank determinism cases plus the
regression-protection test for the
``expected_position == plan.write_positions`` paradox: the oracle must
recompute expected_position from per-req committed state, not echo
back the planner's value.
"""

from __future__ import annotations

import unittest
from dataclasses import dataclass
from typing import List, Optional

import torch

from sglang.srt.pseudo_mode.oracle import PseudoOracle
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=2)


_DEFAULT_VOCAB = 1024
_DEFAULT_EOS = 2


@dataclass
class _StubForwardMode:
    extend: bool
    decode: bool

    def is_extend(self, include_draft_extend_v2: bool = False) -> bool:
        return self.extend

    def is_decode(self) -> bool:
        return self.decode


@dataclass
class _StubForwardBatch:
    forward_mode: _StubForwardMode
    req_pool_indices: torch.Tensor


@dataclass
class _StubBatchPlan:
    write_req_pool_indices: List[int]
    write_positions: List[int]
    write_req_entry_starts: List[int]
    write_req_entry_counts: List[int]
    num_write: int
    num_write_reqs: int


def _make_oracle(seed: int = 0xC0FFEE) -> PseudoOracle:
    return PseudoOracle(seed=seed, vocab_size=_DEFAULT_VOCAB, eos_id=_DEFAULT_EOS)


def _admit_simple(
    oracle: PseudoOracle,
    *,
    req_id: str,
    prompt: List[int],
    max_new_tokens: int = 8,
    eos_at: Optional[int] = None,
    req_pool_idx: Optional[int] = None,
) -> None:
    oracle.admit(
        req_id=req_id,
        origin_input_ids=prompt,
        max_new_tokens=max_new_tokens,
        eos_at=eos_at,
    )
    if req_pool_idx is not None:
        oracle.register_req_pool_mapping(req_pool_idx=req_pool_idx, req_id=req_id)


def _decode_mode() -> _StubForwardMode:
    return _StubForwardMode(extend=False, decode=True)


def _extend_mode() -> _StubForwardMode:
    return _StubForwardMode(extend=True, decode=False)


class TestCase1PrefillPromptLookup(unittest.TestCase):
    """Case 1: pure prefill — predict_input_token returns origin_input_ids[position]."""

    def test_prompt_lookup_returns_origin_token(self) -> None:
        oracle = _make_oracle()
        prompt = [10, 20, 30, 40]
        _admit_simple(oracle, req_id="r0", prompt=prompt)
        for position, token in enumerate(prompt):
            self.assertEqual(
                oracle.predict_input_token(req_id="r0", position=position),
                token,
            )


class TestCase2DecodeHistoryLookup(unittest.TestCase):
    """Case 2: decode — predict_input_token at position P+k returns output_history[k]."""

    def test_history_lookup_after_commit(self) -> None:
        oracle = _make_oracle()
        prompt = [1, 2, 3]
        _admit_simple(oracle, req_id="r0", prompt=prompt)
        oracle.register_chunk_commit(req_id="r0", chunk_size=len(prompt))
        oracle.commit_step(req_id="r0", output_token=77)
        oracle.commit_step(req_id="r0", output_token=88)

        prefill_len = len(prompt)
        self.assertEqual(
            oracle.predict_input_token(req_id="r0", position=prefill_len),
            77,
        )
        self.assertEqual(
            oracle.predict_input_token(req_id="r0", position=prefill_len + 1),
            88,
        )

    def test_history_lookup_out_of_range_raises(self) -> None:
        oracle = _make_oracle()
        _admit_simple(oracle, req_id="r0", prompt=[1, 2])
        oracle.register_chunk_commit(req_id="r0", chunk_size=2)
        with self.assertRaises(IndexError):
            oracle.predict_input_token(req_id="r0", position=5)


class TestCase3ChunkedPrefill(unittest.TestCase):
    """Case 3: chunked prefill — commit_step NOT called between chunks; prompt lookup still works."""

    def test_prompt_lookup_unchanged_across_chunks(self) -> None:
        oracle = _make_oracle()
        prompt = list(range(100, 110))
        _admit_simple(oracle, req_id="r0", prompt=prompt)
        oracle.register_chunk_commit(req_id="r0", chunk_size=4)
        for position, token in enumerate(prompt):
            self.assertEqual(
                oracle.predict_input_token(req_id="r0", position=position),
                token,
            )
        oracle.register_chunk_commit(req_id="r0", chunk_size=6)
        for position, token in enumerate(prompt):
            self.assertEqual(
                oracle.predict_input_token(req_id="r0", position=position),
                token,
            )

    def test_commit_step_before_chunks_done_raises(self) -> None:
        """Cannot commit a decoded token while prompt chunks remain pending."""
        oracle = _make_oracle()
        _admit_simple(oracle, req_id="r0", prompt=[1, 2, 3])
        oracle.register_chunk_commit(req_id="r0", chunk_size=2)
        with self.assertRaises(RuntimeError):
            oracle.commit_step(req_id="r0", output_token=99)


class TestCase4PrefixCacheOOS(unittest.TestCase):
    """Case 4: prefix cache — out of scope for v1; install.py enforces --disable-radix-cache.

    Oracle itself stays prefix-cache-agnostic: queries for any in-range
    position return the same answer regardless of which slots sglang
    actually reuses, since the oracle has no knowledge of slot layout.
    """

    def test_oracle_blind_to_slot_reuse(self) -> None:
        oracle = _make_oracle()
        prompt_a = [10, 11, 12, 13, 14]
        prompt_b = [10, 11, 12, 99, 88]
        _admit_simple(oracle, req_id="rA", prompt=prompt_a)
        _admit_simple(oracle, req_id="rB", prompt=prompt_b)
        for position in range(5):
            self.assertEqual(
                oracle.predict_input_token(req_id="rA", position=position),
                prompt_a[position],
            )
            self.assertEqual(
                oracle.predict_input_token(req_id="rB", position=position),
                prompt_b[position],
            )


class TestCase5PreemptResume(unittest.TestCase):
    """Case 5: preempt + resume — oracle state retained, no commit_step on preempt."""

    def test_history_preserved_across_simulated_preempt(self) -> None:
        oracle = _make_oracle()
        prompt = [5, 6, 7]
        _admit_simple(oracle, req_id="r0", prompt=prompt, req_pool_idx=4)
        oracle.register_chunk_commit(req_id="r0", chunk_size=3)
        oracle.commit_step(req_id="r0", output_token=111)
        oracle.commit_step(req_id="r0", output_token=222)
        oracle.commit_step(req_id="r0", output_token=333)

        # Simulate sglang evicting r0: install.py does nothing to oracle.
        # On resume, scheduler re-prefills the same prompt + history; the
        # oracle answers unchanged.
        for k, expected in enumerate([111, 222, 333]):
            self.assertEqual(
                oracle.predict_input_token(req_id="r0", position=len(prompt) + k),
                expected,
            )

    def test_pool_idx_remap_after_resume(self) -> None:
        oracle = _make_oracle()
        _admit_simple(oracle, req_id="r0", prompt=[1, 2], req_pool_idx=3)
        # Same pool index remapping to the same req is idempotent.
        oracle.register_req_pool_mapping(req_pool_idx=3, req_id="r0")
        # Resume scheduler may pick a different pool slot.
        oracle.register_req_pool_mapping(req_pool_idx=5, req_id="r0")


class TestCase6OverlapDeterminism(unittest.TestCase):
    """Case 6: overlap scheduler — oracle is deterministic given (req_id, position) snapshot."""

    def test_same_snapshot_yields_same_answer(self) -> None:
        oracle = _make_oracle()
        _admit_simple(oracle, req_id="r0", prompt=[9, 8, 7])
        # Multiple deferred queries against the same frozen state.
        first = oracle.predict_input_token(req_id="r0", position=1)
        second = oracle.predict_input_token(req_id="r0", position=1)
        self.assertEqual(first, second)
        self.assertEqual(first, 8)


class TestCase7SpecDecodingDraftV1(unittest.TestCase):
    """Case 7: spec decoding draft — v1 only mirrors token sequence, not accept logic."""

    def test_draft_step_uses_same_predict_output(self) -> None:
        oracle = _make_oracle()
        _admit_simple(oracle, req_id="r0", prompt=[1])
        # Draft worker and target worker query the same oracle; predict_output_token
        # is a pure function of (seed, req_id, step), so both see identical answers.
        steps = [oracle.predict_output_token(req_id="r0", step=k) for k in range(4)]
        steps_again = [oracle.predict_output_token(req_id="r0", step=k) for k in range(4)]
        self.assertEqual(steps, steps_again)


class TestCase8Eos(unittest.TestCase):
    """Case 8: EOS — admit(eos_at=K) makes predict_output_token return eos_id at step K."""

    def test_eos_at_returns_eos_id_at_configured_step(self) -> None:
        oracle = _make_oracle()
        _admit_simple(oracle, req_id="r0", prompt=[1, 2], max_new_tokens=10, eos_at=3)
        for step in range(3):
            self.assertNotEqual(
                oracle.predict_output_token(req_id="r0", step=step),
                _DEFAULT_EOS,
            )
        self.assertEqual(
            oracle.predict_output_token(req_id="r0", step=3),
            _DEFAULT_EOS,
        )
        self.assertEqual(
            oracle.predict_output_token(req_id="r0", step=99),
            _DEFAULT_EOS,
        )

    def test_default_eos_at_is_max_new_tokens(self) -> None:
        oracle = _make_oracle()
        _admit_simple(oracle, req_id="r0", prompt=[1], max_new_tokens=4)
        self.assertNotEqual(
            oracle.predict_output_token(req_id="r0", step=3),
            _DEFAULT_EOS,
        )
        self.assertEqual(
            oracle.predict_output_token(req_id="r0", step=4),
            _DEFAULT_EOS,
        )


class TestCase9DpCrossRankIdentity(unittest.TestCase):
    """Case 9: DP — two oracle instances with the same seed agree token-by-token."""

    def test_two_instances_same_seed_same_tokens(self) -> None:
        oracle_a = _make_oracle(seed=42)
        oracle_b = _make_oracle(seed=42)
        prompt = [3, 4, 5]
        _admit_simple(oracle_a, req_id="rX", prompt=prompt)
        _admit_simple(oracle_b, req_id="rX", prompt=prompt)
        for step in range(16):
            self.assertEqual(
                oracle_a.predict_output_token(req_id="rX", step=step),
                oracle_b.predict_output_token(req_id="rX", step=step),
            )

    def test_different_seed_diverges(self) -> None:
        oracle_a = _make_oracle(seed=1)
        oracle_b = _make_oracle(seed=2)
        _admit_simple(oracle_a, req_id="rX", prompt=[1])
        _admit_simple(oracle_b, req_id="rX", prompt=[1])
        diverged = any(
            oracle_a.predict_output_token(req_id="rX", step=k)
            != oracle_b.predict_output_token(req_id="rX", step=k)
            for k in range(16)
        )
        self.assertTrue(diverged)


class TestCase10MultimodalOOS(unittest.TestCase):
    """Case 10: multimodal — OOS for v1; placeholder tokens treated as ordinary ints."""

    def test_placeholder_token_is_plain_int(self) -> None:
        oracle = _make_oracle()
        placeholder = 151643
        _admit_simple(oracle, req_id="r0", prompt=[1, placeholder, 2])
        self.assertEqual(
            oracle.predict_input_token(req_id="r0", position=1),
            placeholder,
        )


class TestPredictOutputTokenDeterminism(unittest.TestCase):
    """predict_output_token is a pure function of (seed, req_id, step)."""

    def test_same_inputs_same_token_within_instance(self) -> None:
        oracle = _make_oracle()
        _admit_simple(oracle, req_id="r0", prompt=[1])
        first = [oracle.predict_output_token(req_id="r0", step=k) for k in range(8)]
        second = [oracle.predict_output_token(req_id="r0", step=k) for k in range(8)]
        self.assertEqual(first, second)

    def test_output_in_vocab_range(self) -> None:
        oracle = _make_oracle()
        _admit_simple(oracle, req_id="r0", prompt=[1], max_new_tokens=100)
        for step in range(100):
            token = oracle.predict_output_token(req_id="r0", step=step)
            self.assertGreaterEqual(token, 0)
            self.assertLess(token, _DEFAULT_VOCAB)


class TestLifecycleRoundTrip(unittest.TestCase):
    """admit + commit_step + finish lifecycle round-trip."""

    def test_full_lifecycle(self) -> None:
        oracle = _make_oracle()
        _admit_simple(oracle, req_id="r0", prompt=[1, 2, 3], req_pool_idx=7)
        oracle.register_chunk_commit(req_id="r0", chunk_size=3)
        oracle.commit_step(req_id="r0", output_token=42)
        self.assertEqual(
            oracle.predict_input_token(req_id="r0", position=3),
            42,
        )
        oracle.finish(req_id="r0")
        # Both the pool mapping and the per-req state are dropped on
        # finish so long-running servers don't accumulate dead reqs.
        with self.assertRaises(KeyError):
            oracle.predict_next_tokens_for_active_batch(
                forward_batch=_StubForwardBatch(
                    forward_mode=_decode_mode(),
                    req_pool_indices=torch.tensor([7], dtype=torch.int64),
                ),
                device=torch.device("cpu"),
            )
        with self.assertRaises(KeyError):
            oracle.predict_input_token(req_id="r0", position=3)

    def test_readmit_after_finish_works(self) -> None:
        oracle = _make_oracle()
        _admit_simple(oracle, req_id="r0", prompt=[1, 2, 3], req_pool_idx=7)
        oracle.register_chunk_commit(req_id="r0", chunk_size=3)
        oracle.commit_step(req_id="r0", output_token=42)
        oracle.finish(req_id="r0")
        _admit_simple(oracle, req_id="r0", prompt=[9, 8, 7], req_pool_idx=11)
        oracle.register_chunk_commit(req_id="r0", chunk_size=3)
        self.assertEqual(
            oracle.predict_input_token(req_id="r0", position=0),
            9,
        )

    def test_double_admit_without_finish_raises(self) -> None:
        oracle = _make_oracle()
        _admit_simple(oracle, req_id="r0", prompt=[1])
        with self.assertRaises(ValueError):
            _admit_simple(oracle, req_id="r0", prompt=[2])

    def test_finish_unknown_raises(self) -> None:
        oracle = _make_oracle()
        with self.assertRaises(KeyError):
            oracle.finish(req_id="nope")


class TestPredictInputTokensForPlanIndependentPosition(unittest.TestCase):
    """The crucial regression test: oracle's expected_position is *independently* recomputed.

    Build a plan where the (simulated) sglang planner produced
    write_positions = [5, 6, 7] but oracle's committed state implies the
    real positions must be [3, 4, 5] (e.g. only 3 prefill chunks
    committed). The oracle helper must return [3, 4, 5], NOT the bogus
    planner values. Without this, INPUT_POSITION_MISMATCH would
    compare two equal-but-both-wrong numbers and miss #25015-class
    bugs.
    """

    def test_extend_independent_position_recompute(self) -> None:
        oracle = _make_oracle()
        prompt = list(range(8))
        _admit_simple(oracle, req_id="r0", prompt=prompt, req_pool_idx=2)
        # Pretend sglang scheduler thinks we're 5 chunks in already (buggy planner).
        # Oracle ignores that lie: only 3 chunks were ever register_chunk_commit'd.
        oracle.register_chunk_commit(req_id="r0", chunk_size=3)

        plan = _StubBatchPlan(
            write_req_pool_indices=[2],
            # Planner-claimed positions are wrong (this is the #25015 simulation).
            write_positions=[5, 6, 7],
            write_req_entry_starts=[0],
            write_req_entry_counts=[3],
            num_write=3,
            num_write_reqs=1,
        )
        forward_batch = _StubForwardBatch(
            forward_mode=_extend_mode(),
            req_pool_indices=torch.tensor([2], dtype=torch.int64),
        )

        tokens, positions = oracle.predict_input_tokens_for_plan(
            plan=plan, forward_batch=forward_batch
        )
        self.assertEqual(positions, [3, 4, 5])
        self.assertEqual(tokens, [prompt[3], prompt[4], prompt[5]])

    def test_decode_independent_position_recompute(self) -> None:
        oracle = _make_oracle()
        prompt = [10, 20]
        _admit_simple(oracle, req_id="r0", prompt=prompt, req_pool_idx=1)
        oracle.register_chunk_commit(req_id="r0", chunk_size=2)
        oracle.commit_step(req_id="r0", output_token=500)
        oracle.commit_step(req_id="r0", output_token=600)

        # In decode the entry feeds last-committed token; position = seq_len - 1
        # = prefill_len + len(output_history) - 1 = 2 + 2 - 1 = 3.
        plan = _StubBatchPlan(
            write_req_pool_indices=[1],
            write_positions=[99],  # planner lie
            write_req_entry_starts=[0],
            write_req_entry_counts=[1],
            num_write=1,
            num_write_reqs=1,
        )
        forward_batch = _StubForwardBatch(
            forward_mode=_decode_mode(),
            req_pool_indices=torch.tensor([1], dtype=torch.int64),
        )
        tokens, positions = oracle.predict_input_tokens_for_plan(
            plan=plan, forward_batch=forward_batch
        )
        self.assertEqual(positions, [3])
        self.assertEqual(tokens, [600])


class TestPredictNextTokensForActiveBatch(unittest.TestCase):
    """Vectorised next-token oracle for the sampler override path."""

    def test_returns_per_req_predictions(self) -> None:
        oracle = _make_oracle()
        _admit_simple(oracle, req_id="rA", prompt=[1], req_pool_idx=0)
        _admit_simple(oracle, req_id="rB", prompt=[1], req_pool_idx=1)
        oracle.register_chunk_commit(req_id="rA", chunk_size=1)
        oracle.register_chunk_commit(req_id="rB", chunk_size=1)
        oracle.commit_step(req_id="rA", output_token=100)

        forward_batch = _StubForwardBatch(
            forward_mode=_decode_mode(),
            req_pool_indices=torch.tensor([0, 1], dtype=torch.int64),
        )
        out = oracle.predict_next_tokens_for_active_batch(
            forward_batch=forward_batch, device=torch.device("cpu")
        )
        self.assertEqual(out.shape, (2,))
        self.assertEqual(out.dtype, torch.int64)
        # rA has 1 committed step → next step is index 1.
        self.assertEqual(
            int(out[0]),
            oracle.predict_output_token(req_id="rA", step=1),
        )
        # rB has 0 committed steps → next step is index 0.
        self.assertEqual(
            int(out[1]),
            oracle.predict_output_token(req_id="rB", step=0),
        )


if __name__ == "__main__":
    unittest.main(verbosity=3)
