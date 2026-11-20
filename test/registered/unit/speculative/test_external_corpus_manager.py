"""Regression test for removing an external corpus before load bookkeeping commits."""

import unittest
from unittest.mock import Mock

from sglang.srt.managers.io_struct import (
    AddExternalCorpusReqInput,
    RemoveExternalCorpusReqInput,
)
from sglang.srt.speculative.external_corpus_manager import ExternalCorpusManager
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


class TestExternalCorpusManager(CustomTestCase):
    def test_remove_before_load_commit(self):
        """A completed load must block removal until its bookkeeping is committed."""
        worker = Mock()
        worker.add_external_corpus.return_value = 2
        manager = ExternalCorpusManager(worker, Mock())
        manager.add(AddExternalCorpusReqInput(corpus_id="test", token_chunks=[[1, 2]]))

        # Finish the background load without processing its result in the scheduler.
        thread = manager._pending_load[1]
        thread.join(timeout=5)
        self.assertFalse(thread.is_alive())
        request = RemoveExternalCorpusReqInput(corpus_id="test")
        self.assertFalse(manager.remove(request).success)
        worker.remove_external_corpus.assert_not_called()

        manager.check_pending_load()
        worker.commit_corpus_load.assert_called_once_with("test", 2)
        self.assertTrue(manager.remove(request).success)


if __name__ == "__main__":
    unittest.main()
