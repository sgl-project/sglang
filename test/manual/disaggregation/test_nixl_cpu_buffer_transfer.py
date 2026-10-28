"""Real UCX host transfer with CUDA KV pages.

Run with UCX_TLS=tcp,cuda_copy,self UCX_ACC_DEVICES=''.

This transport smoke test needs NIXL/UCX and one CUDA device. It does not launch
an inference server. NIXL_TEST_DEVICE=cpu also exercises the real host transport
without allocating GPU memory.
"""

import os
import threading
import time
import unittest
import uuid
from types import SimpleNamespace

import numpy as np
import torch

from sglang.srt.disaggregation.base.conn import StateType
from sglang.srt.disaggregation.nixl.conn import NixlKVManager
from sglang.srt.disaggregation.nixl.cpu_buffer import (
    CPU_BUFFER_VERSION,
    CpuBufferPool,
    page_views,
    scatter_pages,
)
from sglang.test.test_utils import CustomTestCase


class TestNixlCpuBufferTransfer(CustomTestCase):
    def test_real_chunked_transfer(self):
        self._roundtrip(with_qsa=False)

    def test_real_qsa_chunked_transfer(self):
        self._roundtrip(with_qsa=True)

    def _roundtrip(self, *, with_qsa):
        from nixl._api import nixl_agent, nixl_agent_config, nixl_thread_sync_t

        device = os.environ.get("NIXL_TEST_DEVICE", "cuda")
        source = [
            torch.arange(512, device=device).to(torch.bfloat16).reshape(16, 32),
            torch.arange(1024, device=device).to(torch.bfloat16).reshape(16, 64),
        ]
        target = [torch.zeros_like(t) for t in source]
        widths = [t[0].numel() * t.element_size() for t in source]
        state_source = (
            [
                [
                    torch.arange(16 * 12, device=device)
                    .to(torch.bfloat16)
                    .reshape(16, 12)
                ],
                [
                    torch.arange(16 * 4 * 8, device=device)
                    .to(torch.bfloat16)
                    .reshape(16, 32),
                    torch.arange(16 * 4, device=device, dtype=torch.int64).reshape(
                        16, 4
                    ),
                ],
                [
                    torch.arange(16 * 16 * 8, device=device)
                    .to(torch.bfloat16)
                    .reshape(16, 128)
                ],
            ]
            if with_qsa
            else []
        )
        state_target = [[torch.zeros_like(t) for t in group] for group in state_source]
        state_types = (
            [StateType.MAMBA, StateType.QSA_PENDING, StateType.QSA_COMPRESSED]
            if with_qsa
            else []
        )
        state_widths = [[t[0].nbytes for t in group] for group in state_source]
        src_state_ids = (
            [np.array([7]), np.array([3]), np.array([12, 5, 1, 9])] if with_qsa else []
        )
        dst_state_ids = (
            [np.array([1]), np.array([8]), np.array([3, 10, 2, 6])] if with_qsa else []
        )
        managers = []
        registrations = []
        for tensors, state_tensors in ((source, state_source), (target, state_target)):
            mgr = NixlKVManager.__new__(NixlKVManager)
            mgr._cpu_pool = CpuBufferPool(1 << 20)
            mgr._cpu_receives = {}
            mgr._cpu_receive_lock = threading.RLock()
            mgr.attn_tp_size = 1
            mgr.kv_args = SimpleNamespace(
                kv_item_lens=widths,
                kv_layer_ids=[0, 1],
                state_types=state_types,
                state_item_lens=state_widths,
            )
            mgr._cpu_state_views = [
                page_views(group, [t.data_ptr() for t in group], sw)
                for group, sw in zip(state_tensors, state_widths)
            ]
            mgr._cpu_views = page_views(
                tensors, [t.data_ptr() for t in tensors], widths
            )
            cfg = nixl_agent_config(
                backends=[],
                num_threads=1,
                sync_mode=nixl_thread_sync_t.NIXL_THREAD_SYNC_STRICT,
            )
            mgr.agent = nixl_agent("cpu-buffer-test-" + uuid.uuid4().hex, cfg)
            mgr.agent.create_backend("UCX", {"num_threads": "1"})
            registrations.append(mgr._cpu_pool.register(mgr.agent))
            managers.append(mgr)
        sender, receiver = managers
        receiver.agent.add_remote_agent(sender.agent.get_agent_metadata())
        remote_name = sender.agent.add_remote_agent(receiver.agent.get_agent_metadata())
        src_ids, dst_ids = np.array([9, 3, 14, 1, 7]), np.array([2, 10, 0, 5, 15])
        region = receiver._allocate_cpu_receive(7, dst_ids, 1, dst_state_ids)
        peer = SimpleNamespace(
            cpu_buffer_version=CPU_BUFFER_VERSION,
            decode_tp_size=1,
            dst_dcp_size=1,
            dst_kv_item_lens=widths,
            dst_kv_layer_ids=[0, 1],
            dst_state_data_ptrs=[],
            dst_state_types=state_types,
            dst_state_item_lens=state_widths,
        )
        req = SimpleNamespace(
            cpu_buffer_pages=len(dst_ids),
            dst_kv_indices=dst_ids,
            cpu_buffer_ptr=region.buffer.data_ptr(),
            agent_name=remote_name,
            dst_state_indices=dst_state_ids,
        )
        handles = []
        try:
            for chunk_id, (start, end) in enumerate(((0, 2), (2, 5))):
                chunk = SimpleNamespace(
                    index_slice=slice(start, end), prefill_kv_indices=src_ids[start:end]
                )
                sender._send_cpu_chunk(req, peer, chunk, f"chunk-{chunk_id}", handles)
                deadline = time.monotonic() + 30
                while True:
                    state = sender.agent.check_xfer_state(handles[-1])
                    self.assertNotEqual(state, "ERR")
                    if state == "DONE":
                        break
                    self.assertLess(
                        time.monotonic(), deadline, "NIXL transfer timed out"
                    )
                    time.sleep(0.001)
            if with_qsa:
                sender._send_cpu_state(
                    req,
                    peer,
                    SimpleNamespace(state_indices=src_state_ids),
                    "7_state_0",
                    handles,
                )
                deadline = time.monotonic() + 30
                while sender.agent.check_xfer_state(handles[-1]) != "DONE":
                    self.assertNotEqual(
                        sender.agent.check_xfer_state(handles[-1]), "ERR"
                    )
                    self.assertLess(
                        time.monotonic(), deadline, "QSA transfer timed out"
                    )
                    time.sleep(0.001)
            if with_qsa and device != "cpu":
                # Model a redundant forward still writing a recently freed slot.
                # Warm copy kernels and temporary allocations so first-use
                # initialization cannot accidentally drain the producer.
                scatter_pages(
                    receiver._cpu_views,
                    dst_ids,
                    torch.zeros(len(dst_ids) * sum(widths), dtype=torch.uint8),
                )
                for views, ids in zip(receiver._cpu_state_views, dst_state_ids):
                    scatter_pages(
                        views,
                        ids,
                        torch.zeros(
                            len(ids) * sum(v.shape[1] for v in views), dtype=torch.uint8
                        ),
                    )
                producer = torch.cuda.Stream(device=device)
                state_ids = torch.as_tensor(dst_state_ids[0], device=device)
                with torch.cuda.stream(producer):
                    state_target[0][0].index_fill_(0, state_ids, 0)
                producer.synchronize()
                with torch.cuda.stream(producer):
                    torch.cuda._sleep(500_000_000)
                    state_target[0][0].index_fill_(0, state_ids, 7)
                receiver._cpu_restore_stream = producer
            receiver._finish_cpu_receive(7)
            if with_qsa and device != "cpu":
                producer.synchronize()
            for sg, tg, si, di in zip(
                state_source, state_target, src_state_ids, dst_state_ids
            ):
                for before, after in zip(sg, tg):
                    self.assertTrue(torch.equal(before[si], after[di]))
                    self.assertEqual(
                        after[[i for i in range(16) if i not in di]].count_nonzero(), 0
                    )
            self.assertFalse(receiver._cpu_pool._ranges)
            for before, after in zip(source, target):
                self.assertTrue(torch.equal(before[src_ids], after[dst_ids]))
                untouched = [i for i in range(16) if i not in dst_ids]
                self.assertEqual(torch.count_nonzero(after[untouched]).item(), 0)
            sender._cpu_pool.reap(sender.agent)
            self.assertFalse(sender._cpu_pool._ranges)
            print(
                f"NIXL CPU-buffer roundtrip passed: device={device}, qsa={with_qsa}, chunks=2, kv_bytes={len(src_ids) * sum(widths)}"
            )
        finally:
            # Keep backing tensors and registration descriptors alive until
            # agents are destroyed, including on a failed transfer.
            for mgr in managers:
                del mgr.agent


if __name__ == "__main__":
    unittest.main()
