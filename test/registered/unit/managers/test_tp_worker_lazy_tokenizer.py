"""TpModelWorker loads its tokenizer (and a multimodal model's processor) on
first read, not in __init__: the scheduler has its own, and only the speculative
workers that need one (ngram, standalone) read the worker's."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import Mock, patch

from sglang.srt.managers import tp_worker
from sglang.srt.managers.tp_worker import TpModelWorker
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

SERVING = SimpleNamespace(
    skip_tokenizer_init=False,
    tokenizer_path="/models/m",
    tokenizer_mode="auto",
    tokenizer_backend="auto",
)
MODEL = SimpleNamespace(trust_remote_code=True, revision=None, model_path="/models/m")


def bare_worker(*, multimodal=False, draft=False):
    # The loads read only these two attributes, so a worker built without
    # __init__ (no model runner, no GPU) exercises them as a real one would.
    w = object.__new__(TpModelWorker)
    w.is_draft_worker = draft
    w.model_config = SimpleNamespace(is_multimodal=multimodal)
    return w


class TestTpWorkerLazyTokenizer(CustomTestCase):
    def setUp(self):
        self.get_tokenizer = Mock(return_value="tokenizer")
        self.get_processor = Mock(return_value="processor")
        self.from_processor = Mock(return_value="tokenizer-from-processor")
        for name, value in (
            ("get_serving", lambda: SERVING),
            ("get_model", lambda: MODEL),
            ("get_tokenizer", self.get_tokenizer),
            ("get_processor", self.get_processor),
            ("get_tokenizer_from_processor", self.from_processor),
        ):
            p = patch.object(tp_worker, name, value)
            p.start()
            self.addCleanup(p.stop)

    def test_text_model_loads_tokenizer_once_on_first_read(self):
        w = bare_worker()
        self.get_tokenizer.assert_not_called()
        self.assertEqual(w.tokenizer, "tokenizer")
        self.assertEqual(w.tokenizer, "tokenizer")
        self.get_tokenizer.assert_called_once_with(
            "/models/m",
            tokenizer_mode="auto",
            trust_remote_code=True,
            revision=None,
            tokenizer_backend="auto",
        )
        self.assertIsNone(w.processor)
        self.get_processor.assert_not_called()

    def test_multimodal_model_derives_tokenizer_from_one_processor_load(self):
        w = bare_worker(multimodal=True)
        self.assertEqual(w.tokenizer, "tokenizer-from-processor")
        self.assertEqual(w.processor, "processor")
        self.get_processor.assert_called_once()
        self.from_processor.assert_called_once_with("processor")
        self.get_tokenizer.assert_not_called()

    def test_draft_worker_and_skip_tokenizer_init_load_nothing(self):
        for draft, skip in ((True, False), (False, True)):
            with (
                self.subTest(draft=draft, skip_tokenizer_init=skip),
                patch.object(
                    tp_worker,
                    "get_serving",
                    lambda: SimpleNamespace(
                        **{**vars(SERVING), "skip_tokenizer_init": skip}
                    ),
                ),
            ):
                w = bare_worker(multimodal=True, draft=draft)
                self.assertIsNone(w.tokenizer)
                self.assertIsNone(w.processor)
        self.get_tokenizer.assert_not_called()
        self.get_processor.assert_not_called()


if __name__ == "__main__":
    import unittest

    unittest.main()
