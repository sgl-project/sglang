"""Regression test for removing an external corpus before load bookkeeping commits."""

import socket
import threading
import unittest
from unittest.mock import Mock, patch

import torch.distributed as dist
import torch.multiprocessing as mp

from sglang.srt.managers.io_struct import (
    AddExternalCorpusReqInput,
    RemoveExternalCorpusReqInput,
)
from sglang.srt.speculative.external_corpus_manager import ExternalCorpusManager
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


_MODULE = "sglang.srt.speculative.external_corpus_manager"
_TP_SIZE = 2


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def _run_tp_rank(rank: int, port: int):
    """Run one TP rank. Rank 1 keeps loading corpus "b" until it is released."""
    dist.init_process_group(
        "gloo",
        init_method=f"tcp://127.0.0.1:{port}",
        rank=rank,
        world_size=_TP_SIZE,
    )
    try:
        parallel = Mock()
        parallel.tp_group.world_size = _TP_SIZE
        parallel.tp_group.cpu_group = dist.group.WORLD

        release_load = threading.Event()
        if rank != 1:
            release_load.set()

        def _load(corpus_id, token_chunks):
            assert release_load.wait(timeout=30)
            return 2

        worker = Mock()
        worker.add_external_corpus.side_effect = _load
        manager = ExternalCorpusManager(worker, Mock())
        remove_a = RemoveExternalCorpusReqInput(corpus_id="a")

        with patch(f"{_MODULE}.get_parallel", return_value=parallel):
            manager.add(
                AddExternalCorpusReqInput(corpus_id="b", token_chunks=[[1, 2]])
            )
            if rank == 0:
                # Rank 0 finishes its load. Rank 1 still loads.
                manager._pending_load[1].join(timeout=30)
                assert not manager._pending_load[1].is_alive()

            # One rank still loads, so no rank may commit "b" or remove "a".
            manager.check_pending_load()
            worker.commit_corpus_load.assert_not_called()
            assert not manager.remove(remove_a).success
            worker.remove_external_corpus.assert_not_called()

            release_load.set()
            manager._pending_load[1].join(timeout=30)

            # All ranks finished, so all ranks commit "b" and remove "a".
            manager.check_pending_load()
            worker.commit_corpus_load.assert_called_once_with("b", 2)
            assert manager.remove(remove_a).success
            worker.remove_external_corpus.assert_called_once_with("a")
    finally:
        dist.destroy_process_group()


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


class TestExternalCorpusManagerMultiRank(CustomTestCase):
    def test_tp_ranks_agree_on_pending_load(self):
        """With TP=2, no rank removes a corpus while another rank still loads."""
        mp.spawn(_run_tp_rank, args=(_free_port(),), nprocs=_TP_SIZE, join=True)


if __name__ == "__main__":
    unittest.main()
