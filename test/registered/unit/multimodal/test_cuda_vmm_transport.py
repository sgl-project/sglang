"""CUDA VMM multimodal feature transport regression tests."""

from __future__ import annotations

import gc
import multiprocessing as mp
import os
import pickle
import queue
import threading
import unittest
from unittest.mock import MagicMock, patch

import torch

from sglang.srt.runtime_context import get_parallel
from sglang.srt.utils.cuda_vmm_transport_utils import (
    CudaVmmMemoryPool,
    CudaVmmPackedTensorTransportProxy,
    _imported_pool_cache_clear,
    _PosixFdBroker,
)
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=33, stage="base-c", runner_config="4-gpu-gb300")


class _FabricUnavailableCudaVmmMemoryPool(CudaVmmMemoryPool):
    def _allocate(self, memory_size: int) -> None:
        if self.use_fabric:
            raise RuntimeError("forced FABRIC allocation failure")
        super()._allocate(memory_size)


def _produce_vmm_tensor(proxy_queue, consumer_done, result_queue, mode):
    pool = source = proxy = None
    try:
        torch.cuda.set_device(0)
        pool_cls = (
            _FabricUnavailableCudaVmmMemoryPool
            if mode == "posix_fallback"
            else CudaVmmMemoryPool
        )
        pool = pool_cls(
            memory_size=4 << 20,
            recycle_interval=60,
            base_gpu_id=0,
            consumer_count=2,
            allow_posix_fallback=True,
        )
        source = torch.arange(35, dtype=torch.float32, device="cuda").reshape(5, 7)
        expected = source.cpu().tolist()
        proxy = pool.wrap_tensor(source)
        proxy_queue.put((proxy, expected))
        if not consumer_done.wait(timeout=60):
            raise TimeoutError("consumers did not release the CUDA VMM tensor")
        with pool._lock:
            pool._recycle_chunks()
            pool._merge_chunks()
            if pool.occupied_chunks:
                raise RuntimeError(
                    "consumer acknowledgements did not recycle the slice"
                )
    except Exception as exc:  # noqa: BLE001  # pragma: no cover
        result_queue.put(("error", repr(exc)))
        return
    finally:
        del proxy, source
        if pool is not None:
            pool.shutdown()
            del pool
        gc.collect()
    result_queue.put(("ok", None))


class TestCudaVmmTransport(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        if (
            not torch.cuda.is_available()
            or torch.version.cuda is None
            or torch.cuda.device_count() < 3
        ):
            raise unittest.SkipTest("At least three NVIDIA CUDA GPUs are required")

    def _run_round_trip(self, mode: str):
        consumer_devices = (1, 2)
        torch.cuda.set_device(consumer_devices[0])
        ctx = mp.get_context("spawn")
        proxy_queue = ctx.Queue()
        producer_results = ctx.Queue()
        consumer_done = ctx.Event()
        producer = ctx.Process(
            target=_produce_vmm_tensor,
            args=(proxy_queue, consumer_done, producer_results, mode),
        )
        producer.start()
        proxy = second_proxy = None
        reconstructed = []
        producer_result = None
        try:
            try:
                proxy, expected = proxy_queue.get(timeout=60)
            except queue.Empty:
                producer_result = producer_results.get(timeout=5)
                _status, payload = producer_result
                self.fail(
                    f"CUDA VMM producer failed before sending its proxy: {payload}"
                )

            if mode == "posix_fallback":
                self.assertIsNone(proxy.fabric_handle)
                self.assertIsNotNone(proxy.posix_socket_path)
            else:
                self.assertIsNotNone(proxy.fabric_handle)
                self.assertIsNone(proxy.posix_socket_path)
            second_proxy = pickle.loads(pickle.dumps(proxy))
            for tp_rank, (consumer_proxy, device) in enumerate(
                zip((proxy, second_proxy), consumer_devices)
            ):
                torch.cuda.set_device(device)
                with get_parallel().override(
                    attn_tp_size=2,
                    attn_tp_rank=tp_rank,
                    attn_cp_size=1,
                    attn_cp_rank=0,
                ):
                    tensor = consumer_proxy.reconstruct_on_target_device(
                        device, consumer_count=1
                    )
                torch.cuda.synchronize(device)
                self.assertEqual(tensor.cpu().tolist(), expected)
                reconstructed.append(tensor)
        finally:
            del reconstructed, second_proxy, proxy
            _imported_pool_cache_clear()
            gc.collect()
            consumer_done.set()
            producer.join(timeout=60)
            try:
                if producer_result is None:
                    producer_result = producer_results.get(timeout=5)
                status, payload = producer_result
                self.assertEqual(status, "ok", payload)
            finally:
                if producer.is_alive():
                    producer.terminate()
                    producer.join(timeout=10)
                torch.cuda.set_device(0)
            self.assertEqual(producer.exitcode, 0)

    def test_posix_fd_fallback_tensor_round_trip_and_recycling(self):
        self._run_round_trip(mode="posix_fallback")

    def test_auto_prefers_fabric_tensor_round_trip_and_recycling(self):
        self._run_round_trip(mode="auto")

    def test_reused_chunk_clears_acknowledgements(self):
        pool = CudaVmmMemoryPool(4 << 20, 60, 0, 2, allow_posix_fallback=True)
        try:
            old = pool.wrap_tensor(torch.ones(1024, dtype=torch.uint8, device="cuda:0"))
            pool.memory_pool[old.control_offset : old.control_offset + 8].view(
                torch.int32
            ).fill_(1)
            torch.cuda.synchronize(0)
            with pool._lock:
                pool._recycle_chunks()
                pool._merge_chunks()

            pool.wrap_tensor(torch.ones(100, dtype=torch.uint8, device="cuda:0"))
            live = pool.wrap_tensor(torch.ones(256, dtype=torch.uint8, device="cuda:0"))
            control = pool.memory_pool[
                live.control_offset : live.control_offset + 8
            ].view(torch.int32)
            self.assertTrue(torch.equal(control, torch.zeros_like(control)))

            with pool._lock:
                pool._recycle_chunks()
            self.assertIn(
                live.control_offset,
                [chunk.start for chunk in pool.occupied_chunks],
            )
        finally:
            pool.shutdown()

    def test_packed_tensors_round_trip_through_one_shared_buffer(self):
        pool = CudaVmmMemoryPool(4 << 20, 60, 0, 1, allow_posix_fallback=True)
        sources = [
            torch.arange(24, dtype=torch.float32, device="cuda:0")
            .reshape(4, 6)
            .transpose(0, 1),
            torch.arange(7, dtype=torch.bfloat16),
            torch.arange(5, dtype=torch.int64, device="cuda:0"),
        ]
        expected = [source.contiguous().cpu() for source in sources]
        proxies = reconstructed = None
        try:
            stream = MagicMock(wraps=torch.cuda.current_stream(0))
            with patch("torch.cuda.current_stream", return_value=stream):
                proxies = pool.wrap_tensors(sources)

            self.assertIsNotNone(proxies)
            self.assertEqual(stream.synchronize.call_count, 1)
            self.assertEqual(len(pool.occupied_chunks), 1)
            self.assertTrue(
                all(
                    isinstance(proxy, CudaVmmPackedTensorTransportProxy)
                    for proxy in proxies
                )
            )
            self.assertEqual(len({proxy.control_offset for proxy in proxies}), 1)

            proxies = pickle.loads(pickle.dumps(proxies))
            self.assertIs(proxies[0]._packed_owner, proxies[-1]._packed_owner)
            with get_parallel().override(
                attn_tp_size=1,
                attn_tp_rank=0,
                attn_cp_size=1,
                attn_cp_rank=0,
            ):
                reconstructed = [
                    proxy.reconstruct_on_target_device(0, consumer_count=1)
                    for proxy in proxies
                ]
            torch.cuda.synchronize(0)

            for actual, wanted in zip(reconstructed, expected):
                self.assertTrue(torch.equal(actual.cpu(), wanted))
            packed_storage = proxies[
                0
            ]._packed_owner.reconstruct_tensor.untyped_storage()
            self.assertTrue(
                all(
                    tensor.untyped_storage().data_ptr() == packed_storage.data_ptr()
                    for tensor in reconstructed
                )
            )
            with pool._lock:
                pool._recycle_chunks()
            self.assertFalse(pool.occupied_chunks)
        finally:
            del reconstructed, proxies, expected, sources
            _imported_pool_cache_clear()
            pool.shutdown()

    def test_packed_cancel_is_shared_and_idempotent(self):
        pool = CudaVmmMemoryPool(4 << 20, 60, 0, 1, allow_posix_fallback=True)
        try:
            proxies = pool.wrap_tensors(
                [
                    torch.ones(8, dtype=torch.float32, device="cuda:0"),
                    torch.ones(8, dtype=torch.float32, device="cuda:0"),
                ]
            )
            self.assertIsNotNone(proxies)

            pool.cancel_proxy(proxies[0])
            pool.cancel_proxy(proxies[1])

            self.assertFalse(pool.occupied_chunks)
            self.assertEqual(
                sum(chunk.size for chunk in pool.available_chunks),
                pool.allocation_size,
            )
        finally:
            pool.shutdown()

    def test_packed_reservation_failure_returns_fallback_signal(self):
        pool = CudaVmmMemoryPool(4 << 20, 60, 0, 1, allow_posix_fallback=True)
        try:
            source = torch.empty(
                pool.allocation_size, dtype=torch.uint8, device="cuda:0"
            )

            self.assertIsNone(pool.wrap_tensors([source]))
            self.assertFalse(pool.occupied_chunks)
            self.assertEqual(
                sum(chunk.size for chunk in pool.available_chunks),
                pool.allocation_size,
            )
        finally:
            pool.shutdown()

    def test_oversized_tensor_falls_back_to_cpu(self):
        pool = CudaVmmMemoryPool(4 << 20, 60, 0, 1, allow_posix_fallback=True)
        try:
            source = torch.empty(
                pool.allocation_size, dtype=torch.uint8, device="cuda:0"
            )

            fallback = pool.wrap_tensor(source)

            self.assertTrue(fallback.is_cpu)
            self.assertFalse(pool.occupied_chunks)
            self.assertEqual(
                sum(chunk.size for chunk in pool.available_chunks),
                pool.allocation_size,
            )
        finally:
            pool.shutdown()

    def test_failed_copy_rolls_back_reservation(self):
        pool = CudaVmmMemoryPool(4 << 20, 60, 0, 1, allow_posix_fallback=True)
        try:
            with self.assertRaisesRegex(NotImplementedError, "meta tensor"):
                pool.wrap_tensor(torch.ones(16, device="meta"))
            with self.assertRaisesRegex(NotImplementedError, "meta tensor"):
                pool.wrap_tensors(
                    [
                        torch.ones(16, device="cuda:0"),
                        torch.ones(16, device="meta"),
                    ]
                )
            self.assertFalse(pool.occupied_chunks)
            self.assertEqual(
                sum(chunk.size for chunk in pool.available_chunks),
                pool.allocation_size,
            )
        finally:
            pool.shutdown()

    def test_undispatched_proxy_can_be_cancelled_immediately(self):
        pool = CudaVmmMemoryPool(4 << 20, 60, 0, 1, allow_posix_fallback=True)
        try:
            proxy = pool.wrap_tensor(torch.ones(16, device="cuda:0"))

            pool.cancel_proxy(proxy)

            self.assertFalse(pool.occupied_chunks)
            self.assertEqual(
                sum(chunk.size for chunk in pool.available_chunks),
                pool.allocation_size,
            )
        finally:
            pool.shutdown()

    def test_failed_cleanup_sync_quarantines_pool(self):
        pool = CudaVmmMemoryPool(4 << 20, 60, 0, 1, allow_posix_fallback=True)
        stream = MagicMock()
        stream.synchronize.side_effect = RuntimeError("forced sync failure")
        try:
            with (
                patch("torch.cuda.current_stream", return_value=stream),
                self.assertRaisesRegex(RuntimeError, "forced sync failure"),
            ):
                pool.wrap_tensor(torch.ones(16, device="cuda:0"))
            torch.cuda.synchronize(0)
            self.assertIsNotNone(pool._pool_error)
            with self.assertRaisesRegex(RuntimeError, "pool failed"):
                pool.wrap_tensor(torch.ones(16, device="cuda:0"))
        finally:
            pool.shutdown()

    def test_shutdown_waits_for_active_publisher(self):
        pool = CudaVmmMemoryPool(4 << 20, 60, 0, 1, allow_posix_fallback=True)
        publisher_entered = threading.Event()
        allow_publisher_to_finish = threading.Event()
        shutdown_entered = threading.Event()
        shutdown_finished = threading.Event()
        errors = []
        real_stream = torch.cuda.current_stream(0)
        stream = MagicMock(wraps=real_stream)

        def synchronize():
            publisher_entered.set()
            if not allow_publisher_to_finish.wait(timeout=10):
                raise TimeoutError("publisher was not released")
            real_stream.synchronize()

        stream.synchronize.side_effect = synchronize

        def publish():
            try:
                with patch("torch.cuda.current_stream", return_value=stream):
                    pool.wrap_tensor(torch.ones(16, device="cuda:0"))
            except Exception as error:  # pragma: no cover
                errors.append(error)

        def shutdown():
            shutdown_entered.set()
            try:
                pool.shutdown()
            except Exception as error:  # pragma: no cover
                errors.append(error)
            finally:
                shutdown_finished.set()

        publisher = threading.Thread(target=publish)
        shutdown_thread = threading.Thread(target=shutdown)
        try:
            publisher.start()
            self.assertTrue(publisher_entered.wait(timeout=10))
            shutdown_thread.start()
            self.assertTrue(shutdown_entered.wait(timeout=10))
            self.assertFalse(shutdown_finished.wait(timeout=0.1))
        finally:
            allow_publisher_to_finish.set()
            publisher.join(timeout=10)
            shutdown_thread.join(timeout=10)
            if not pool._closed:
                pool.shutdown()

        self.assertFalse(publisher.is_alive())
        self.assertFalse(shutdown_thread.is_alive())
        self.assertFalse(errors)

    def test_consumer_copy_failure_releases_slice_without_allowing_retry(self):
        pool = CudaVmmMemoryPool(4 << 20, 60, 0, 1, allow_posix_fallback=True)
        try:
            proxy = pool.wrap_tensor(torch.ones(1, device="cuda:0"))
            proxy.shape = (2,)
            with (
                get_parallel().override(
                    attn_tp_size=1,
                    attn_tp_rank=0,
                    attn_cp_size=1,
                    attn_cp_rank=0,
                ),
                self.assertRaises(RuntimeError),
            ):
                proxy.reconstruct_on_target_device(0, consumer_count=1)
            torch.cuda.synchronize(0)
            with pool._lock:
                pool._recycle_chunks()
            self.assertFalse(pool.occupied_chunks)

            with (
                get_parallel().override(
                    attn_tp_size=1,
                    attn_tp_rank=0,
                    attn_cp_size=1,
                    attn_cp_rank=0,
                ),
                self.assertRaisesRegex(RuntimeError, "already released"),
            ):
                proxy.reconstruct_on_target_device(0, consumer_count=1)
        finally:
            _imported_pool_cache_clear()
            pool.shutdown()

    def test_posix_export_fd_closes_when_allocation_setup_fails(self):
        with (
            patch(
                "sglang.srt.utils.cuda_vmm_transport_utils.tensor_from_pointer",
                side_effect=RuntimeError("forced storage failure"),
            ),
            patch(
                "sglang.srt.utils.cuda_vmm_transport_utils.os.close",
                wraps=os.close,
            ) as close_fd,
            self.assertRaisesRegex(RuntimeError, "forced storage failure"),
        ):
            _FabricUnavailableCudaVmmMemoryPool(
                4 << 20, 60, 0, 1, allow_posix_fallback=True
            )
        close_fd.assert_called_once()

    def test_stream_setup_failure_releases_pool_and_posix_broker(self):
        release_allocation = CudaVmmMemoryPool._release_allocation
        close_broker = _PosixFdBroker.close
        with (
            patch(
                "sglang.srt.utils.cuda_vmm_transport_utils.torch.cuda.Stream",
                side_effect=RuntimeError("forced stream failure"),
            ),
            patch.object(
                CudaVmmMemoryPool,
                "_release_allocation",
                autospec=True,
                side_effect=release_allocation,
            ) as release_pool,
            patch.object(
                _PosixFdBroker,
                "close",
                autospec=True,
                side_effect=close_broker,
            ) as close_fd_broker,
            self.assertRaisesRegex(RuntimeError, "forced stream failure"),
        ):
            _FabricUnavailableCudaVmmMemoryPool(
                4 << 20, 60, 0, 1, allow_posix_fallback=True
            )

        close_fd_broker.assert_called_once()
        release_pool.assert_called_once()


if __name__ == "__main__":
    unittest.main(verbosity=2)
