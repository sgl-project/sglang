"""HuggingFace parity tests for setwise (multi-position) scoring.

Setwise scoring pools the SequenceClassification head at every occurrence of a
``score_extraction_token`` (one per candidate) instead of the last token. These
tests verify the *numerical* correctness of that readout against a HuggingFace
reference: run the same checkpoint through HF, gather the final hidden state at
the anchor positions, apply the model's own classification head, and compare to
the engine's setwise ``scores``.

The head is the real (pretrained) head, so these tests assert the engine pools
at the correct positions and applies the head correctly — a wrong pooling
position or head application changes the numbers, not just the shape.

Uses a public ``Qwen3ForSequenceClassification`` checkpoint. Like the MIS and
other scoring tests, the engine runs ``float16`` on the ``flashinfer`` attention
backend; the HF golden is computed in ``float32`` and compared with an
fp16-appropriate tolerance. The setwise anchor token ``<|object_ref_start|>`` is
a Qwen3 special token, so it tokenizes to a single dedicated id regardless of
whether the checkpoint was trained on setwise.
"""

import asyncio
import os
import unittest

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from sglang.srt.entrypoints.engine import Engine
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=200, stage="base-b", runner_config="1-gpu-small")

_SEQCLS_MODEL = os.environ.get(
    "TEST_CLASSIFICATION_BASE_MODEL",
    "tomaarsen/Qwen3-Reranker-0.6B-seq-cls",
)
_ANCHOR_TOKEN = os.environ.get("TEST_SCORE_EXTRACTION_TOKEN", "<|object_ref_start|>")
# Match MIS / other scoring tests: float16 on the flashinfer backend (flashinfer
# prefill has no float32 kernel). The HF golden is float32; the tolerance below
# accounts for the fp16-vs-fp32 gap. All three are overridable via env.
_DTYPE = os.environ.get("TEST_SEQCLS_DTYPE", "float16")
_ATOL = float(os.environ.get("TEST_SETWISE_ATOL", "0.2"))
_RTOL = float(os.environ.get("TEST_SETWISE_RTOL", "0.05"))


class TestSetwiseScoringHFParity(CustomTestCase):
    """Setwise engine scores must match an HF head-at-anchor reference."""

    @classmethod
    def setUpClass(cls):
        cls.tokenizer = AutoTokenizer.from_pretrained(_SEQCLS_MODEL)
        cls.anchor_id = cls.tokenizer.convert_tokens_to_ids(_ANCHOR_TOKEN)
        assert (
            cls.anchor_id is not None and cls.anchor_id != cls.tokenizer.unk_token_id
        ), f"{_ANCHOR_TOKEN!r} did not resolve to a dedicated token id"

        # float16 on the flashinfer backend, matching MIS and the other scoring
        # tests (flashinfer prefill has no float32 kernel). Setwise uses standard
        # prefill — no MIS mask — but flashinfer is the scoring backend of record.
        cls.engine = Engine(
            model_path=_SEQCLS_MODEL,
            disable_radix_cache=True,
            chunked_prefill_size=-1,
            attention_backend="flashinfer",
            dtype=_DTYPE,
            mem_fraction_static=0.15,
        )

    @classmethod
    def tearDownClass(cls):
        if getattr(cls, "engine", None) is not None:
            cls.engine.shutdown()
        torch.cuda.empty_cache()

    # ------------------------------------------------------------------
    # HF reference
    # ------------------------------------------------------------------

    def _hf_setwise_reference(self, prompt: str):
        """Reference: HF hidden state at each anchor -> classification head.

        Returns (ref_scores [N, num_labels] as list-of-lists, anchor_positions).
        Tokenized exactly like the engine (``tokenizer.encode`` with default
        special tokens) so the anchor positions line up.
        """
        input_ids = self.tokenizer.encode(prompt)
        anchor_positions = [i for i, t in enumerate(input_ids) if t == self.anchor_id]
        self.assertGreater(len(anchor_positions), 0, "prompt has no anchor tokens")

        model = AutoModelForSequenceClassification.from_pretrained(
            _SEQCLS_MODEL, torch_dtype=torch.float32
        ).eval()
        try:
            ids = torch.tensor([input_ids], dtype=torch.long)
            with torch.no_grad():
                outputs = model(input_ids=ids, output_hidden_states=True)
            last_hidden = outputs.hidden_states[-1][0]  # [seq, hidden]

            # The classification head is `score` on Qwen/Llama SeqCls models.
            head = getattr(model, "score", None) or getattr(model, "classifier")
            anchor_hidden = last_hidden[anchor_positions].to(head.weight.dtype)
            ref = head(anchor_hidden)  # [N, num_labels]
            return ref.tolist(), anchor_positions
        finally:
            model.cpu()
            del model
            torch.cuda.empty_cache()

    def _hf_anchor_hidden_states(self, prompt: str):
        """Reference: HF final hidden state at each anchor position (pre-head).

        Returns (hidden [N, hidden] as list-of-lists, anchor_positions).
        """
        input_ids = self.tokenizer.encode(prompt)
        anchor_positions = [i for i, t in enumerate(input_ids) if t == self.anchor_id]
        self.assertGreater(len(anchor_positions), 0, "prompt has no anchor tokens")

        model = AutoModelForSequenceClassification.from_pretrained(
            _SEQCLS_MODEL, torch_dtype=torch.float32
        ).eval()
        try:
            ids = torch.tensor([input_ids], dtype=torch.long)
            with torch.no_grad():
                outputs = model(input_ids=ids, output_hidden_states=True)
            last_hidden = outputs.hidden_states[-1][0]  # [seq, hidden]
            return last_hidden[anchor_positions].tolist(), anchor_positions
        finally:
            model.cpu()
            del model
            torch.cuda.empty_cache()

    def _assert_close(self, ref, sgl, atol=_ATOL, rtol=_RTOL):
        self.assertEqual(len(ref), len(sgl), "row count mismatch")
        for i, (rrow, srow) in enumerate(zip(ref, sgl)):
            self.assertEqual(len(rrow), len(srow), f"row {i} width mismatch")
            for j, (r, s) in enumerate(zip(rrow, srow)):
                self.assertLessEqual(
                    abs(r - s),
                    atol + rtol * abs(r),
                    f"row {i} label {j}: HF={r:.6f} SGLang={s:.6f}",
                )

    # ------------------------------------------------------------------
    # Tests
    # ------------------------------------------------------------------

    def _build_prompt(self, n_anchors: int) -> str:
        candidates = " ".join(f"Candidate {i}." for i in range(n_anchors))
        return f"Rank the candidates. {candidates} Scores:" + (
            _ANCHOR_TOKEN * n_anchors
        )

    def test_setwise_scores_match_hf_reference(self):
        """Engine setwise logits == HF head applied at each anchor position."""
        prompt = self._build_prompt(3)
        ref, anchor_positions = self._hf_setwise_reference(prompt)

        # `scores` is nested per item (one [N x num_labels] matrix per item); a
        # single item here yields one matrix with one row per anchor.
        sgl = self.engine.score(
            query="",
            items=[prompt],
            apply_softmax=False,
            score_extraction_token_id=self.anchor_id,
        ).scores

        self.assertEqual(len(sgl), 1)  # one item
        item = sgl[0]
        self.assertEqual(len(item), len(anchor_positions))  # one row per anchor
        self._assert_close(ref, item)

    def test_setwise_multiple_items_match_hf(self):
        """Multiple items (candidate sets) -> one HF-matching matrix per item.

        Without --enable-mis each item is scored as an independent ``query+item``
        sequence, so ``scores`` is nested per item and each item's matrix must
        match its own standalone HF reference.
        """
        prompt0 = self._build_prompt(3)
        prompt1 = self._build_prompt(2)
        ref0, anchors0 = self._hf_setwise_reference(prompt0)
        ref1, anchors1 = self._hf_setwise_reference(prompt1)

        sgl = self.engine.score(
            query="",
            items=[prompt0, prompt1],
            apply_softmax=False,
            score_extraction_token_id=self.anchor_id,
        ).scores

        self.assertEqual(len(sgl), 2)  # one matrix per item
        self.assertEqual(len(sgl[0]), len(anchors0))
        self.assertEqual(len(sgl[1]), len(anchors1))
        self._assert_close(ref0, sgl[0])
        self._assert_close(ref1, sgl[1])

    def test_setwise_single_anchor_matches_hf(self):
        """A single anchor is the degenerate case and must still match HF."""
        prompt = self._build_prompt(1)
        ref, _ = self._hf_setwise_reference(prompt)

        sgl = self.engine.score(
            query="",
            items=[prompt],
            apply_softmax=False,
            score_extraction_token_id=self.anchor_id,
        ).scores

        self.assertEqual(len(sgl), 1)  # one item
        self.assertEqual(len(sgl[0]), 1)  # one anchor row
        self._assert_close(ref, sgl[0])

    def test_setwise_candidate_ranking_matches_hf(self):
        """Ranking of candidates (argsort over label 0) agrees with HF.

        Ranking is the property the setwise ranker actually consumes, and it is
        robust to small floating-point differences.
        """
        prompt = self._build_prompt(5)
        ref, _ = self._hf_setwise_reference(prompt)

        # Single item -> unwrap its [N x num_labels] matrix.
        sgl = self.engine.score(
            query="",
            items=[prompt],
            apply_softmax=False,
            score_extraction_token_id=self.anchor_id,
        ).scores[0]

        ref_order = sorted(range(len(ref)), key=lambda i: ref[i][0])
        sgl_order = sorted(range(len(sgl)), key=lambda i: sgl[i][0])
        self.assertEqual(ref_order, sgl_order)

    def test_setwise_pooled_hidden_states_match_hf(self):
        """return_pooled_hidden_states -> per-anchor pre-head vectors == HF.

        Verifies the PHS path returns the raw hidden state at each anchor
        position (one vector per candidate), not just the head logits.
        """
        prompt = self._build_prompt(3)
        ref_hidden, anchor_positions = self._hf_anchor_hidden_states(prompt)

        result = self.engine.score(
            query="",
            items=[prompt],
            apply_softmax=False,
            score_extraction_token_id=self.anchor_id,
            return_pooled_hidden_states=True,
        )

        phs = result.pooled_hidden_states
        self.assertIsNotNone(phs)
        # Nested per item, parallel to scores: a single item -> one group of
        # pre-head vectors, one per anchor, aligned 1:1 with the score rows.
        self.assertEqual(len(phs), 1)  # one item
        self.assertEqual(len(result.scores), 1)
        item_phs = phs[0]
        self.assertEqual(len(item_phs), len(anchor_positions))
        self.assertEqual(len(item_phs), len(result.scores[0]))
        sgl_hidden = [vec.tolist() for vec in item_phs]
        self._assert_close(ref_hidden, sgl_hidden)

    def test_setwise_tokenized_items_omitted_query_match_hf(self):
        """Token-ID items with the query omitted take the token-ID path.

        ``engine.score(items=[[token_ids]], score_extraction_token_id=...)`` with
        no ``query`` (None) must normalize to an empty token-ID prefix and score
        via the token-ID path, matching HF — instead of being routed to the text
        builder and rejected.
        """
        prompt = self._build_prompt(3)
        ref, anchor_positions = self._hf_setwise_reference(prompt)
        token_ids = self.tokenizer.encode(prompt)

        sgl = self.engine.score(
            items=[token_ids],  # query omitted -> normalized to []
            apply_softmax=False,
            score_extraction_token_id=self.anchor_id,
        ).scores

        self.assertEqual(len(sgl), 1)
        self.assertEqual(len(sgl[0]), len(anchor_positions))
        self._assert_close(ref, sgl[0])

    def test_mixed_setwise_and_pointwise_concurrent_batch(self):
        """Setwise and pointwise requests batched together must both succeed.

        Both use the same ``/v1/score`` endpoint and scheduler queue, so a client
        cannot ensure homogeneous batches. The scheduler partitions prefill
        batches by pooling mode, so concurrent mixed traffic must not abort the
        setwise requests (pre-fix, a mixed batch returned HTTP 400 for them).
        """
        setwise_prompt = self._build_prompt(3)

        async def _run():
            coros = []
            for _ in range(6):
                coros.append(
                    self.engine.async_score(
                        query="",
                        items=[setwise_prompt],
                        apply_softmax=False,
                        score_extraction_token_id=self.anchor_id,
                    )
                )
                coros.append(
                    self.engine.async_score(
                        query="",
                        items=["Candidate A.", "Candidate B."],
                        apply_softmax=False,
                    )
                )
            return await asyncio.gather(*coros, return_exceptions=True)

        results = self.engine.loop.run_until_complete(_run())

        errors = [r for r in results if isinstance(r, BaseException)]
        self.assertEqual(errors, [], f"mixed batch produced errors: {errors}")
        for i, r in enumerate(results):
            if i % 2 == 0:  # setwise: one item, 3 anchor rows (nested)
                self.assertEqual(len(r.scores), 1)
                self.assertEqual(len(r.scores[0]), 3)
            else:  # pointwise: one score row per item (flat)
                self.assertEqual(len(r.scores), 2)


class TestSetwiseMultiItemMISScoring(CustomTestCase):
    """Multi-item setwise scoring under ``--enable-mis`` (fused sequence).

    The fused path returns one score matrix per item (nested ``scores``), isolates
    each set with the block-diagonal mask, and shares the query-prefix KV. It
    carries the standard MIS delimiter tokens, so it is **not** bit-identical to
    independent HF scoring — that trade-off is inherent to MIS, not to score
    extraction. These tests therefore assert the nested per-item shape and set
    isolation (a set's scores are unchanged by the presence of other sets) rather
    than HF parity, which is covered for the batched path in
    ``TestSetwiseScoringHFParity``.
    """

    @classmethod
    def setUpClass(cls):
        cls.tokenizer = AutoTokenizer.from_pretrained(_SEQCLS_MODEL)
        cls.anchor_id = cls.tokenizer.convert_tokens_to_ids(_ANCHOR_TOKEN)
        cls.engine = Engine(
            model_path=_SEQCLS_MODEL,
            disable_radix_cache=True,
            chunked_prefill_size=-1,
            enable_mis=True,
            attention_backend="flashinfer",
            dtype=_DTYPE,
            mem_fraction_static=0.15,
        )

    @classmethod
    def tearDownClass(cls):
        if getattr(cls, "engine", None) is not None:
            cls.engine.shutdown()
        torch.cuda.empty_cache()

    @staticmethod
    def _set_prompt(n_anchors: int) -> str:
        candidates = " ".join(f"Candidate {i}." for i in range(n_anchors))
        return f"Rank the candidates. {candidates} Scores:" + (
            _ANCHOR_TOKEN * n_anchors
        )

    # atol=5e-2 is a floor set by fp16, not slack. With a 10-bit mantissa one ULP
    # on these logits is already ~0.002 at magnitude ~2 and ~0.004 at magnitude ~7,
    # so a single rounding step exceeds 1e-3. Fusing set0 with another set changes
    # the batch shape (padding/tiling), which changes reduction order and stacks
    # several such steps; the observed drift is ~0.012. 5e-2 clears that with margin
    # while still catching real cross-set leakage, which would diverge far more.
    # A tighter bound (e.g. 1e-3) is below fp16 resolution and cannot pass.
    def _assert_matrix_close(self, a, b, atol=5e-2):
        self.assertEqual(len(a), len(b), "row count mismatch")
        for i, (ra, rb) in enumerate(zip(a, b)):
            self.assertEqual(len(ra), len(rb), f"row {i} width mismatch")
            for x, y in zip(ra, rb):
                self.assertLessEqual(abs(x - y), atol, f"{x} vs {y}")

    def test_mis_multi_item_nested_shape(self):
        """Fused multi-item -> one [Nᵢ x num_labels] matrix per item."""
        scores = self.engine.score(
            query="Rank:",
            items=[self._set_prompt(3), self._set_prompt(2)],
            apply_softmax=False,
            score_extraction_token_id=self.anchor_id,
        ).scores

        self.assertEqual(len(scores), 2)  # one matrix per item
        self.assertEqual(len(scores[0]), 3)  # 3 candidates
        self.assertEqual(len(scores[1]), 2)  # 2 candidates
        num_labels = len(scores[0][0])
        self.assertTrue(
            all(len(row) == num_labels for matrix in scores for row in matrix)
        )

    def test_mis_set_isolation(self):
        """Block-diagonal mask: a set's scores are unchanged by other sets.

        The first set is scored alone and again fused with a second set; the
        block-diagonal mask means the first set attends only over the shared query
        prefix and itself, so its matrix must be identical either way.
        """
        set0 = self._set_prompt(3)
        alone = self.engine.score(
            query="Rank:",
            items=[set0],
            apply_softmax=False,
            score_extraction_token_id=self.anchor_id,
        ).scores
        fused = self.engine.score(
            query="Rank:",
            items=[set0, self._set_prompt(2)],
            apply_softmax=False,
            score_extraction_token_id=self.anchor_id,
        ).scores

        self.assertEqual(len(alone), 1)
        self.assertEqual(len(fused), 2)
        # set0 is the first item in both requests -> identical matrix.
        self._assert_matrix_close(alone[0], fused[0])

    def test_mis_rejects_item_without_anchor(self):
        """An item carrying no extraction token is rejected under --enable-mis."""
        with self.assertRaisesRegex(ValueError, "at least one"):
            self.engine.score(
                query="Rank:",
                items=[self._set_prompt(2), "no anchors here"],
                apply_softmax=False,
                score_extraction_token_id=self.anchor_id,
            )


if __name__ == "__main__":
    unittest.main()
