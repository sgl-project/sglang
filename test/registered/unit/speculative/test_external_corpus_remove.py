"""Unit tests for ExternalCorpusManager corpus removal during pending load (#36500).

Validates that removing a corpus while it is being asynchronously loaded is
rejected, and that removal after load completes works as expected.
"""
import time
import unittest

from sglang.srt.managers.io_struct import (
    AddExternalCorpusReqInput,
    ListExternalCorporaReqInput,
    RemoveExternalCorpusReqInput,
)
from sglang.srt.speculative.external_corpus_manager import ExternalCorpusManager


class _FakeWorker:
    def __init__(self):
        self._corpora = {}

    def add_external_corpus(self, corpus_id, token_chunks):
        time.sleep(0.5)
        return len(token_chunks)

    def remove_external_corpus(self, corpus_id):
        self._corpora.pop(corpus_id, None)

    def list_external_corpora(self):
        return dict(self._corpora)

    def commit_corpus_load(self, corpus_id, count):
        self._corpora[corpus_id] = count


class _FakeResponseCapture:
    def __init__(self):
        self.responses = []

    def __call__(self, output, recv_req):
        self.responses.append((output, recv_req))


class TestRemoveDuringPendingLoad(unittest.TestCase):
    def test_remove_during_pending_load_rejected(self):
        worker = _FakeWorker()
        mgr = ExternalCorpusManager(worker, _FakeResponseCapture())
        mgr.add(AddExternalCorpusReqInput(
            corpus_id="test_corpus",
            token_chunks=[[1, 2, 3, 4, 5]],
        ))

        result = mgr.remove(RemoveExternalCorpusReqInput(corpus_id="test_corpus"))
        self.assertFalse(result.success)
        self.assertIn("still being loaded", result.message)

    def test_corpus_committed_after_load(self):
        worker = _FakeWorker()
        mgr = ExternalCorpusManager(worker, _FakeResponseCapture())
        mgr.add(AddExternalCorpusReqInput(
            corpus_id="test_corpus",
            token_chunks=[[1, 2, 3, 4, 5]],
        ))
        time.sleep(1.0)
        mgr.check_pending_load()
        result = mgr.list(ListExternalCorporaReqInput())
        self.assertIn("test_corpus", result.corpus_token_counts)

    def test_remove_after_load_succeeds(self):
        worker = _FakeWorker()
        mgr = ExternalCorpusManager(worker, _FakeResponseCapture())
        mgr.add(AddExternalCorpusReqInput(
            corpus_id="test_corpus",
            token_chunks=[[1, 2, 3, 4, 5]],
        ))
        time.sleep(1.0)
        mgr.check_pending_load()
        result = mgr.remove(RemoveExternalCorpusReqInput(corpus_id="test_corpus"))
        self.assertTrue(result.success)
        listed = mgr.list(ListExternalCorporaReqInput())
        self.assertNotIn("test_corpus", listed.corpus_token_counts)

    def test_remove_different_corpus_during_load_succeeds(self):
        worker = _FakeWorker()
        mgr = ExternalCorpusManager(worker, _FakeResponseCapture())
        mgr.add(AddExternalCorpusReqInput(
            corpus_id="corpus_a",
            token_chunks=[[10, 20, 30]],
        ))
        result = mgr.remove(RemoveExternalCorpusReqInput(corpus_id="corpus_b"))
        self.assertTrue(result.success)


if __name__ == "__main__":
    unittest.main()
