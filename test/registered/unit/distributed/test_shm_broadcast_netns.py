"""MessageQueue reader selection across network namespaces.

Regression test for https://github.com/sgl-project/sglang/issues/53309

Two processes can share /dev/shm while living in different network
namespaces, for example two Pods on one host mounting the host's
/dev/shm. in_the_same_node_as reports such ranks as co-located because
it only tests shared memory visibility, but a local reader also connects
to the writer over 127.0.0.1, which can never reach the writer from
another network namespace, so create_from_process_group hung forever in
wait_until_ready. A rank that does not share the writer's network
namespace must be classified as a remote reader instead.
"""

import os
import tempfile
import unittest
from unittest.mock import patch

import torch.distributed as dist
import torch.multiprocessing as mp

import sglang.srt.distributed.device_communicators.shm_broadcast as shm_broadcast
from sglang.srt.distributed.device_communicators.shm_broadcast import MessageQueue
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=60, suite="base-a-test-cpu")

_WORLD_SIZE = 2


def _run_rank(rank, store_file, netns_ids, result_queue):
    # All ranks run on one host in this test, so loopback stands in for the
    # address a remote reader would use to reach the writer in a deployment.
    os.environ.setdefault("GLOO_SOCKET_IFNAME", "lo")
    dist.init_process_group(
        "gloo",
        init_method=f"file://{store_file}",
        rank=rank,
        world_size=_WORLD_SIZE,
    )
    try:
        with (
            patch.object(
                shm_broadcast,
                "_get_network_namespace_id",
                return_value=netns_ids[rank],
                create=True,
            ),
            patch.object(shm_broadcast, "get_local_ip_auto", return_value="127.0.0.1"),
        ):
            mq = MessageQueue.create_from_process_group(
                dist.group.WORLD, max_chunk_bytes=4096, max_chunks=2
            )
        received = mq.broadcast_object("payload" if rank == 0 else None)
        result_queue.put(
            (
                rank,
                list(mq.handle.local_reader_ranks),
                mq._is_local_reader,
                mq._is_remote_reader,
                received,
            )
        )
        dist.barrier()
    finally:
        dist.destroy_process_group()


class TestMessageQueueNetworkNamespace(CustomTestCase):
    def _create_queue(self, netns_ids):
        with tempfile.TemporaryDirectory() as tmpdir:
            store_file = os.path.join(tmpdir, "rendezvous")
            ctx = mp.get_context("spawn")
            result_queue = ctx.Queue()
            processes = [
                ctx.Process(
                    target=_run_rank,
                    args=(rank, store_file, netns_ids, result_queue),
                )
                for rank in range(_WORLD_SIZE)
            ]
            for process in processes:
                process.start()
            results = sorted(result_queue.get(timeout=180) for _ in range(_WORLD_SIZE))
            for process in processes:
                process.join(timeout=60)
                self.assertEqual(process.exitcode, 0)
            return results

    def test_reader_in_another_network_namespace_is_remote(self):
        results = self._create_queue(netns_ids=[111, 222])
        for _, local_reader_ranks, _, _, received in results:
            self.assertEqual(local_reader_ranks, [])
            self.assertEqual(received, "payload")
        _, _, rank1_is_local, rank1_is_remote, _ = results[1]
        self.assertTrue(rank1_is_remote)
        self.assertFalse(rank1_is_local)

    def test_reader_in_same_network_namespace_is_local(self):
        results = self._create_queue(netns_ids=[111, 111])
        for _, local_reader_ranks, _, _, received in results:
            self.assertEqual(local_reader_ranks, [1])
            self.assertEqual(received, "payload")
        _, _, rank1_is_local, rank1_is_remote, _ = results[1]
        self.assertTrue(rank1_is_local)
        self.assertFalse(rank1_is_remote)

    def test_network_namespace_id_matches_proc(self):
        if not os.path.exists("/proc/self/ns/net"):
            self.skipTest("requires /proc")
        self.assertEqual(
            shm_broadcast._get_network_namespace_id(),
            os.stat("/proc/self/ns/net").st_ino,
        )


if __name__ == "__main__":
    unittest.main()
