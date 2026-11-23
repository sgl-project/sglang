"""Regression test for removing an external corpus before load bookkeeping commits."""

import unittest
from unittest.mock import Mock, patch

from sglang.srt.managers.io_struct import (
    AddExternalCorpusReqInput,
    RemoveExternalCorpusReqInput,
)
from sglang.srt.speculative.external_corpus_manager import ExternalCorpusManager
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


_MODULE = "sglang.srt.speculative.external_corpus_manager"


class TestExternalCorpusManager(CustomTestCase):
    def setUp(self):
        # Single TP rank: no collective is issued.
        parallel = Mock()
        parallel.tp_group.world_size = 1
        patcher = patch(f"{_MODULE}.get_parallel", return_value=parallel)
        patcher.start()
        self.addCleanup(patcher.stop)

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

    def test_pending_until_all_tp_ranks_finish(self):
        """A load finished locally stays pending while another TP rank is still loading."""
        worker = Mock()
        worker.add_external_corpus.return_value = 2
        manager = ExternalCorpusManager(worker, Mock())
        manager.add(AddExternalCorpusReqInput(corpus_id="b", token_chunks=[[1, 2]]))
        manager._pending_load[1].join(timeout=5)

        request = RemoveExternalCorpusReqInput(corpus_id="a")
        with patch.object(
            ExternalCorpusManager, "_all_tp_ranks_agree", return_value=False
        ):
            manager.check_pending_load()
        worker.commit_corpus_load.assert_not_called()
        self.assertFalse(manager.remove(request).success)
        worker.remove_external_corpus.assert_not_called()

        with patch.object(
            ExternalCorpusManager, "_all_tp_ranks_agree", return_value=True
        ):
            manager.check_pending_load()
        worker.commit_corpus_load.assert_called_once_with("b", 2)
        self.assertTrue(manager.remove(request).success)


if __name__ == "__main__":
    unittest.main()
