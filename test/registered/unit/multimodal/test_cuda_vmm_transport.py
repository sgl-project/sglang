"""CUDA VMM multimodal feature transport regression tests."""

from __future__ import annotations

import asyncio
import concurrent.futures
import gc
import multiprocessing as mp
import os
import pickle
import queue
import threading
import unittest
from unittest.mock import patch

import torch

from sglang.srt.environ import envs
from sglang.srt.managers.req_types import MultimodalDataItem, MultimodalProcessorOutput
from sglang.srt.multimodal.modality import Modality
from sglang.srt.multimodal.transport import receive_budget
from sglang.srt.multimodal.transport.cuda_vmm import (
    CudaVmmFeatureTransport,
    CudaVmmMemoryPool,
    CudaVmmPackedTensorTransportProxy,
    _imported_pool_cache_clear,
    _PosixFdBroker,
)
from sglang.srt.multimodal.transport.receive_budget import ReceiveFeatureBudget
from sglang.srt.runtime_context import get_parallel
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=35, stage="base-c", runner_config="4-gpu-gb300")


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
        pool._recycle_chunks()
        if pool.occupied_chunks:
            raise RuntimeError("consumer acknowledgements did not recycle the slice")
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
                    tp_rank=tp_rank,
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

    def test_publication_waits_for_source_stream(self):
        torch.cuda.set_device(0)
        for mode in ("single", "packed", "async", "async_fallback"):
            with self.subTest(mode=mode):
                pool = CudaVmmMemoryPool(4 << 20, 60, 0, 1, allow_posix_fallback=True)
                transport = object.__new__(CudaVmmFeatureTransport)
                transport.pool = pool
                try:
                    # Make a wrong-stream copy observably read the previous value.
                    size = 8 << 20 if mode == "async_fallback" else 1024
                    source = torch.zeros(size, dtype=torch.uint8, device="cuda")
                    torch.cuda.synchronize()
                    stream = torch.cuda.Stream()
                    with concurrent.futures.ThreadPoolExecutor(1) as executor:
                        transport._publisher_executor = executor
                        with torch.cuda.stream(stream):
                            torch.cuda._sleep(200_000_000)
                            source.fill_(73)
                            if mode == "single":
                                proxy = pool.wrap_tensor(source)
                            elif mode == "packed":
                                proxy = pool.wrap_tensors([source, source])[0]
                            else:
                                item = MultimodalDataItem(
                                    modality=Modality.IMAGE, feature=source
                                )
                                asyncio.run(
                                    transport.prepare_for_dispatch_async(
                                        [MultimodalProcessorOutput(mm_items=[item])]
                                    )
                                )
                                proxy = item.feature
                    result = (
                        proxy
                        if isinstance(proxy, torch.Tensor)
                        else proxy.reconstruct_on_target_device(0)
                    )
                    self.assertTrue(
                        torch.equal(
                            result.cpu(), torch.full((size,), 73, dtype=torch.uint8)
                        )
                    )
                finally:
                    torch.cuda.synchronize()
                    _imported_pool_cache_clear()
                    pool.shutdown()

    def test_reused_chunk_clears_acknowledgements(self):
        pool = CudaVmmMemoryPool(4 << 20, 60, 0, 2, allow_posix_fallback=True)
        try:
            old = pool.wrap_tensor(torch.ones(1024, dtype=torch.uint8, device="cuda:0"))
            # VMM must bypass the CUDA IPC zero-copy borrow protocol.
            self.assertIsNone(old.borrow_on_target_device(0))
            pool.memory_pool[old.control_offset : old.control_offset + 8].view(
                torch.int32
            ).fill_(1)
            torch.cuda.synchronize(0)
            pool._recycle_chunks()

            pool.wrap_tensor(torch.ones(100, dtype=torch.uint8, device="cuda:0"))
            live = pool.wrap_tensor(torch.ones(256, dtype=torch.uint8, device="cuda:0"))
            control = pool.memory_pool[
                live.control_offset : live.control_offset + 8
            ].view(torch.int32)
            self.assertTrue(torch.equal(control, torch.zeros_like(control)))

            pool._recycle_chunks()
            self.assertIn(
                live.control_offset,
                [chunk.start for chunk in pool.occupied_chunks],
            )
        finally:
            pool.shutdown()

    def test_packed_cpu_tensors_round_trip_through_one_shared_buffer(self):
        pool = CudaVmmMemoryPool(4 << 20, 60, 0, 1, allow_posix_fallback=True)
        sources = [
            torch.arange(24, dtype=torch.float32).reshape(4, 6).transpose(0, 1),
            torch.arange(7, dtype=torch.bfloat16),
            torch.arange(5, dtype=torch.int64),
        ]
        expected = [source.contiguous().cpu() for source in sources]
        proxies = reconstructed = None
        try:
            with patch.object(
                pool._publish_stream,
                "synchronize",
                wraps=pool._publish_stream.synchronize,
            ) as synchronize:
                proxies = pool.wrap_tensors(sources)

            self.assertIsNotNone(proxies)
            self.assertEqual(synchronize.call_count, 1)
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
        try:
            with (
                patch.object(
                    pool._publish_stream,
                    "synchronize",
                    side_effect=RuntimeError("forced sync failure"),
                ),
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
        real_synchronize = pool._publish_stream.synchronize

        def synchronize():
            publisher_entered.set()
            if not allow_publisher_to_finish.wait(timeout=10):
                raise TimeoutError("publisher was not released")
            real_synchronize()

        def publish():
            try:
                with patch.object(
                    pool._publish_stream,
                    "synchronize",
                    side_effect=synchronize,
                ):
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
                "sglang.srt.multimodal.transport.cuda_vmm.tensor_from_pointer",
                side_effect=RuntimeError("forced storage failure"),
            ),
            patch(
                "sglang.srt.multimodal.transport.cuda_vmm.os.close",
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
                "sglang.srt.multimodal.transport.cuda_vmm.torch.cuda.Stream",
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


class TestReceiveFeatureBudget(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        if not torch.cuda.is_available() or torch.version.cuda is None:
            raise unittest.SkipTest("An NVIDIA CUDA GPU is required")

    def setUp(self):
        torch.cuda.set_device(0)
        self.pool = CudaVmmMemoryPool(32 << 20, 60, 0, 1, allow_posix_fallback=True)
        self.addCleanup(self.pool.shutdown)
        self.addCleanup(_imported_pool_cache_clear)
        host_room = patch.object(
            receive_budget, "_host_allowance", return_value=1 << 40
        )
        host_room.start()
        self.addCleanup(host_room.stop)

    def _request(self, *sources, field="feature", hashed=True):
        """A request as the scheduler receives it: unpickled VMM proxies."""
        if len(sources) > 1:
            proxies = self.pool.wrap_tensors(sources)
        else:
            proxies = [self.pool.wrap_tensor(sources[0])]
        return MultimodalProcessorOutput(
            mm_items=[
                MultimodalDataItem(
                    modality=Modality.IMAGE,
                    hash=index if hashed else None,
                    **{field: proxy},
                )
                for index, proxy in enumerate(pickle.loads(pickle.dumps(proxies)), 1)
            ]
        )

    @staticmethod
    def _materialize(budget, request):
        try:
            return budget.materialize(request, requires_mm_token_modalities=False)
        except RuntimeError:
            for item in request.mm_items:
                item.release_transport_proxies()
            raise

    def _assert_pool_drained(self):
        torch.cuda.synchronize(0)
        self.pool._recycle_chunks()
        self.assertFalse(self.pool.occupied_chunks)

    def test_copies_past_the_gpu_budget_are_staged_on_host(self):
        """Queued requests beyond the per-rank reservation must not hold GPU copies."""
        budget = ReceiveFeatureBudget(device_index=0, gpu_budget_bytes=1 << 20)
        halves = [torch.full((256, 1024), i, dtype=torch.bfloat16) for i in range(2)]

        queued = self._materialize(budget, self._request(*halves))
        # The packed pair shares one copy.
        self.assertEqual(budget.live_gpu_bytes, 1 << 20)
        for item, source in zip(
            self._materialize(budget, self._request(*halves)).mm_items, halves
        ):
            self.assertTrue(item.feature.is_pinned())
            self.assertTrue(torch.equal(item.feature, source))
        # The first prefill forward drops the GPU copy (offload to host).
        del queued
        torch.cuda.synchronize(0)
        self.assertTrue(
            self._materialize(budget, self._request(*halves))
            .mm_items[0]
            .feature.is_cuda
        )
        self._assert_pool_drained()

    def test_host_staged_features_are_complete_when_materialize_returns(self):
        budget = ReceiveFeatureBudget(device_index=0, gpu_budget_bytes=0)
        budget._staging_stream = torch.cuda.Stream(device=0)
        # A first cudaHostAlloc syncs the device; warm the pinned cache past it.
        self._materialize(budget, self._request(torch.randn(1024, 1024, device="cuda")))
        # A host source would leave its bytes in the pinned block staging reuses.
        source = torch.randn(1024, 1024, device="cuda")
        expected = source.cpu()
        request = self._request(source, field="precomputed_embeddings")
        with torch.cuda.stream(budget._staging_stream):
            torch.cuda._sleep(100_000_000)

        mm_inputs = self._materialize(budget, request)

        staged = mm_inputs.mm_items[0].precomputed_embeddings
        self.assertTrue(torch.equal(staged, expected))
        self._assert_pool_drained()

    def test_unhashed_item_gets_the_same_pad_value_on_gpu_and_host(self):
        """Ranks may place one request differently; its pad values must agree."""
        source = torch.arange(4096, dtype=torch.bfloat16).reshape(64, 64)
        on_gpu, on_host = (
            self._materialize(
                ReceiveFeatureBudget(device_index=0, gpu_budget_bytes=gpu_budget),
                self._request(source, hashed=False),
            ).mm_items[0]
            for gpu_budget in (1 << 30, 0)
        )

        self.assertTrue(on_gpu.feature.is_cuda)
        self.assertTrue(on_host.feature.is_pinned())
        self.assertEqual(on_host.pad_values, on_gpu.pad_values)
        # GPU ranks would hash the feature buffer's float32 copy instead.
        with envs.SGLANG_MM_BUFFER_SIZE_MB.override(1), self.assertRaises(RuntimeError):
            self._materialize(
                ReceiveFeatureBudget(device_index=0, gpu_budget_bytes=0),
                self._request(source, hashed=False),
            )
        self._assert_pool_drained()

    def test_freed_copy_stays_charged_until_its_stream_can_reuse_it(self):
        budget = ReceiveFeatureBudget(device_index=0, gpu_budget_bytes=1 << 30)
        mm_inputs = self._materialize(
            budget, self._request(torch.ones(1 << 20, dtype=torch.uint8))
        )
        side = torch.cuda.Stream(device=0)

        with torch.cuda.stream(side):
            torch.cuda._sleep(100_000_000)
            # The first prefill forward's offload: record_stream, then drop the copy.
            mm_inputs.mm_items[0].feature.record_stream(side)
            del mm_inputs
            self.assertEqual(budget.live_gpu_bytes, 1 << 20)
        side.synchronize()
        self.assertEqual(budget.live_gpu_bytes, 0)
        self._assert_pool_drained()

    def test_host_guard_releases_cached_pinned_blocks_before_refusing(self):
        budget = ReceiveFeatureBudget(device_index=0, gpu_budget_bytes=0)
        torch.accelerator.empty_host_cache()
        held = torch.cuda.host_memory_stats()["allocated_bytes.current"]

        def allowance():
            # The OS counts pinned blocks torch keeps cached as used memory.
            pinned = torch.cuda.host_memory_stats()["allocated_bytes.current"]
            return (2 << 20) - (pinned - held)

        # CUDA sources: publishing them pins nothing.
        small, large = (
            torch.ones(n << 20, dtype=torch.uint8, device="cuda") for n in (1, 2)
        )
        with (
            patch.object(receive_budget, "_HOST_READ_INTERVAL_S", -1.0),
            patch.object(receive_budget, "_host_allowance", side_effect=allowance),
        ):
            # Dropped at once; its pinned block stays cached until released.
            self._materialize(budget, self._request(small))
            staged = self._materialize(budget, self._request(small))
            with self.assertRaises(RuntimeError):
                self._materialize(budget, self._request(large))

        self.assertTrue(staged.mm_items[0].feature.is_pinned())
        self._assert_pool_drained()


if __name__ == "__main__":
    unittest.main(verbosity=2)
