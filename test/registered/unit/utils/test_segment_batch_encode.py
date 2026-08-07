"""Unit tests for segment_batch_encode."""

import unittest

from transformers import AutoTokenizer

from sglang.srt.utils.hf_transformers.segment_batch_encode import (
    DEFAULT_PASSAGE_DELIMITER,
    make_segment_batch_encode_tokenizer,
    resolve_split_delimiter,
    segment_batch_encode_ids,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=30, suite="stage-a-test-cpu", nightly=True)


class TestSegmentBatchEncode(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = "Qwen/Qwen2.5-0.5B-Instruct"
        cls.tok = AutoTokenizer.from_pretrained(cls.model, trust_remote_code=True)

    def _rag_text(self, n_passages: int = 5) -> str:
        parts = [f"{DEFAULT_PASSAGE_DELIMITER}{i}\nContent block {i}.\n" for i in range(n_passages)]
        return "".join(parts)

    def test_resolve_delimiter_passage(self):
        text = self._rag_text(3)
        self.assertEqual(
            resolve_split_delimiter(self.tok, None, sample_text=text),
            DEFAULT_PASSAGE_DELIMITER,
        )

    def test_lossless_passage_split(self):
        if not getattr(self.tok, "is_fast", False):
            self.skipTest("fast tokenizer required")
        text = self._rag_text(8)
        plain = self.tok.encode(text, add_special_tokens=False)
        seg = segment_batch_encode_ids(
            self.tok,
            text,
            split_delimiter=DEFAULT_PASSAGE_DELIMITER,
            add_special_tokens=False,
            interleave_delimiter=False,
        )
        self.assertEqual(plain, seg)

    def test_plan_workers_scale_with_length(self):
        if not getattr(self.tok, "is_fast", False):
            self.skipTest("fast tokenizer required")
        from sglang.srt.utils.hf_transformers.segment_batch_encode import (
            compute_max_workers,
            _plan_segment_items,
            _estimate_tokens,
        )

        text = self._rag_text(40)
        est = _estimate_tokens(text)
        self.assertEqual(compute_max_workers(est), max(1, (est * 2) // 1000))
        raw = text.split(DEFAULT_PASSAGE_DELIMITER)
        if raw and raw[-1] == "":
            raw.pop()
        planned = _plan_segment_items(raw, DEFAULT_PASSAGE_DELIMITER, est_tokens=est)
        self.assertLessEqual(len(planned), compute_max_workers(est))

    def test_lossless_adaptive_merge(self):
        if not getattr(self.tok, "is_fast", False):
            self.skipTest("fast tokenizer required")
        text = self._rag_text(40)
        plain = self.tok.encode(text, add_special_tokens=False)
        seg = segment_batch_encode_ids(
            self.tok,
            text,
            split_delimiter=DEFAULT_PASSAGE_DELIMITER,
            add_special_tokens=False,
        )
        self.assertEqual(plain, seg)

    def test_wrapper_short_text_unchanged(self):
        if not getattr(self.tok, "is_fast", False):
            self.skipTest("fast tokenizer required")
        wrapped = make_segment_batch_encode_tokenizer(
            AutoTokenizer.from_pretrained(self.model, trust_remote_code=True),
            split_delimiter=DEFAULT_PASSAGE_DELIMITER,
            min_chars=10_000,
        )
        short = "hello world"
        self.assertEqual(
            wrapped.encode(short, add_special_tokens=False),
            self.tok.encode(short, add_special_tokens=False),
        )

    def test_wrapper_call_single_batch(self):
        if not getattr(self.tok, "is_fast", False):
            self.skipTest("fast tokenizer required")
        text = self._rag_text(6)
        wrapped = make_segment_batch_encode_tokenizer(
            AutoTokenizer.from_pretrained(self.model, trust_remote_code=True),
            split_delimiter=DEFAULT_PASSAGE_DELIMITER,
            min_chars=1,
        )
        out = wrapped([text], add_special_tokens=False)
        self.assertEqual(out["input_ids"][0], self.tok.encode(text, add_special_tokens=False))


if __name__ == "__main__":
    unittest.main()
