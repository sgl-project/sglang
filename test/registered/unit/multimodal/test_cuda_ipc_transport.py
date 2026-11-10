"""CUDA IPC multimodal feature transport regression tests.

This covers the production path where a tokenizer worker places a feature in
the bounded pool and the scheduler process opens the shared CUDA allocation.
CPU-only policy tests intentionally cannot exercise this cross-process handle.
"""

import asyncio
import gc
import multiprocessing as mp
import queue
import threading
import time
import unittest
import weakref
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch

from sglang.srt.managers.req_types import (
    MultimodalDataItem,
    MultimodalInputs,
    MultimodalProcessorOutput,
)
from sglang.srt.multimodal.modality import Modality
from sglang.srt.multimodal.processors.base_processor import (
    BaseMultimodalProcessor,
    BaseMultiModalProcessorOutput,
    MultimodalSpecialTokens,
)
from sglang.srt.multimodal.processors.executor import MultimodalProcessorExecutor
from sglang.srt.multimodal.transport.admission import MmMemoryOverloaded
from sglang.srt.multimodal.transport.cuda_ipc import (
    CudaIpcTensorTransportProxy,
    MmItemMemoryPool,
    _pool_handle_cache_clear,
)
from sglang.srt.multimodal.transport.publisher import BoundedFeaturePublisher
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=54, stage="base-b", runner_config="1-gpu-small")


class _PausedHFProcessor:
    """pause the external processor call while the real async caller cancels"""

    def __init__(self, before_return):
        self.before_return = before_return
        self.tokenizer = SimpleNamespace(bos_token=None)

    def __call__(self, **kwargs):
        self.before_return()
        return {
            "input_ids": torch.tensor([[7, 99, 8]]),
            "pixel_values": torch.ones(1, 4, device="cuda"),
        }


def _produce_pooled_tensor(proxy_queue, consumer_done, result_queue, guarded=False):
    """Create a tokenizer-worker-like CUDA IPC pool in a spawned producer."""
    pool = source = proxy = publisher = None
    try:
        torch.cuda.set_device(0)
        pool = MmItemMemoryPool(
            memory_size=1 << 20,
            recycle_interval=0.01,
            base_gpu_id=0,
            consumer_count=1,
        )
        source = torch.arange(35, dtype=torch.float32, device="cuda").reshape(5, 7)
        expected = torch.arange(35, dtype=torch.float32).reshape(5, 7).tolist()
        if guarded:
            publisher = BoundedFeaturePublisher(pool, 1000, cpu_tensors=False)
            output = MultimodalProcessorOutput(
                input_ids=[1],
                mm_items=[MultimodalDataItem(modality=Modality.IMAGE, feature=source)],
            )
            asyncio.run(publisher.prepare([output]))
            proxy = output.mm_items[0].feature
            del output
        else:
            proxy = pool.wrap_tensor(source, use_pool_handle_cache=True)
        if proxy is None:
            raise RuntimeError("test tensor did not fit in the CUDA IPC pool")
        # Intentionally do not synchronize the producer. The consumer stream
        # wait must order its copy after the producer-ready write.
        proxy_queue.put((proxy, expected))
        if not consumer_done.wait(timeout=60):
            raise TimeoutError("consumer did not release the CUDA IPC tensor")
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline:
            if pool.active_lease_count == 0:
                break
            time.sleep(0.01)
        else:
            raise TimeoutError("pool did not observe the stream-ordered consumer ack")
    except Exception as exc:  # pragma: no cover - returned to the parent
        result_queue.put(("error", repr(exc)))
        return
    finally:
        del proxy, source
        if publisher is not None:
            publisher.shutdown()
        if pool is not None:
            pool.shutdown()
            del pool
        gc.collect()
        torch.cuda.ipc_collect()
    result_queue.put(("ok", None))


def _consume_pooled_tensors(proxies, expected, result_queue):
    """CUDA IPC event handles must be opened by another process, not locally."""
    try:
        torch.cuda.set_device(0)
        for proxy in proxies:
            copied = proxy.reconstruct_on_target_device(0)
            torch.testing.assert_close(copied.cpu(), expected, rtol=0, atol=0)
            del copied
        torch.cuda.synchronize()
        _pool_handle_cache_clear()
        result_queue.put(None)
    except Exception as error:
        result_queue.put(repr(error))


class TestCudaIpcTransport(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        if not torch.cuda.is_available():
            raise unittest.SkipTest("CUDA is required")

    def test_publication_retains_source_without_waiting_for_copy(self):
        """A blocked producer copy must not block publication or lose its source."""
        for noncontiguous in (False, True):
            with self.subTest(noncontiguous=noncontiguous):
                self._check_publication_source_lifetime(noncontiguous)

    def _check_publication_source_lifetime(self, noncontiguous):
        pool = MmItemMemoryPool(1 << 20, 0.01, 0, 1)
        data = torch.arange(1024, device="cuda", dtype=torch.float32).reshape(32, 32)
        source = torch.from_dlpack(data.T if noncontiguous else data)
        del data
        expected = source.cpu()
        copy_stream = torch.cuda.Stream()
        delay_done = torch.cuda.Event()
        released_before_ready = []
        source_ref = weakref.ref(
            source,
            lambda _: released_before_ready.append(not delay_done.query()),
        )
        torch.cuda.synchronize()
        leases = pool.try_reserve_batch([source.numel() * source.element_size()])
        published = threading.Event()
        proxies, failures = [], []

        def publish(tensor):
            try:
                with torch.cuda.stream(copy_stream):
                    # finite device work avoids a test-induced circular CUDA wait
                    torch.cuda._sleep(500_000_000)
                    delay_done.record()
                    proxies.extend(pool.publish_reserved_batch([[tensor]], leases))
            except BaseException as error:
                failures.append(error)
            finally:
                published.set()

        worker = threading.Thread(target=publish, args=(source,))
        del source
        worker.start()
        try:
            try:
                self.assertTrue(
                    published.wait(5), "publication waited for the GPU copy"
                )
                worker.join(5)
                self.assertFalse(worker.is_alive())
                self.assertEqual(failures, [])
                self.assertNotIn(
                    True,
                    released_before_ready,
                    "source owner released before copy readiness",
                )
                self.assertFalse(
                    delay_done.query(), "publication waited for device work"
                )
                self.assertIsNotNone(
                    source_ref(), "source was released before its copy"
                )
            finally:
                worker.join(10)
                copy_stream.synchronize()
            lease = leases[0]
            actual = (
                pool.memory_pool[lease.start : lease.start + lease.nbytes]
                .view(torch.float32)
                .reshape(expected.shape)
            )
            torch.testing.assert_close(actual.cpu(), expected, rtol=0, atol=0)
            deadline = time.monotonic() + 5
            while source_ref() is not None and time.monotonic() < deadline:
                time.sleep(0.01)
            self.assertIsNone(source_ref(), "finished copies retained their sources")
            self.assertEqual(released_before_ready, [False])
            self.assertEqual(pool.active_lease_count, 1)
        finally:
            for proxy in proxies:
                pool.cancel_proxy(proxy)
            pool.shutdown()

    def test_failed_reservation_does_not_make_a_contiguous_copy(self):
        pool = MmItemMemoryPool(1 << 20, 0.01, 0, 1)
        source = torch.empty((1024, 1024), device="cuda").T
        try:
            with patch.object(
                torch.Tensor,
                "contiguous",
                side_effect=AssertionError("copy started before pool reservation"),
            ):
                self.assertEqual(pool._pool.copy_tensor(source), (None, None))
            self.assertEqual(pool.active_lease_count, 0)
            self.assertEqual(pool._pool._copy_sources, {})
        finally:
            pool.shutdown()

    def test_failed_ready_write_keeps_inflight_copy_reserved(self):
        """A failed publication must not reuse bytes an enqueued copy still writes."""
        pool = MmItemMemoryPool(1 << 20, 0.01, 0, 1, max_inflight_slices=1)
        source = torch.full((64 << 10,), 73, dtype=torch.uint8, device="cuda")
        torch.cuda.synchronize()
        stream = torch.cuda.Stream()
        try:
            with torch.cuda.stream(stream):
                # finite work makes an outstanding copy observable without a wait cycle
                torch.cuda._sleep(500_000_000)
                with (
                    patch(
                        "sglang.srt.multimodal.transport.memory_pool.stream_write_value32",
                        side_effect=RuntimeError("producer-ready write failed"),
                    ),
                    self.assertRaisesRegex(RuntimeError, "producer-ready write failed"),
                ):
                    pool.wrap_tensor(source, use_pool_handle_cache=True)
            self.assertTrue(
                stream.query() or pool.active_lease_count == 1,
                "failed publication released a slice before its copy completed",
            )
            stream.synchronize()
            self.assertEqual(pool.active_lease_count, 0)
            self.assertEqual(pool._pool._copy_sources, {})
        finally:
            stream.synchronize()
            pool.shutdown()

    def test_guarded_batch_never_spills_and_recovers_after_ack(self):
        pool = MmItemMemoryPool(1 << 20, 0.01, 0, 1)
        publisher = BoundedFeaturePublisher(pool, 0, cpu_tensors=False)
        source = torch.arange(120000, dtype=torch.float32, device="cuda")
        publish = pool.publish_reserved_batch

        def isolated_publish(*args):
            self.assertNotEqual(
                torch.cuda.current_stream(0), torch.cuda.default_stream(0)
            )
            return publish(*args)

        def request():
            return MultimodalProcessorOutput(
                input_ids=[1],
                mm_items=[MultimodalDataItem(modality=Modality.IMAGE, feature=source)],
            )

        async def run():
            first, second, rejected = request(), request(), request()
            await publisher.prepare([first, second])
            self.assertEqual(pool.active_lease_count, 2)
            with self.assertRaises(MmMemoryOverloaded):
                await publisher.prepare([rejected])
            self.assertIs(rejected.mm_items[0].feature, source)
            proxies = [output.mm_items[0].feature for output in (first, second)]
            for proxy in proxies:
                self.assertIsInstance(proxy, CudaIpcTensorTransportProxy)
            ctx = mp.get_context("spawn")
            results = ctx.Queue()
            consumer = ctx.Process(
                target=_consume_pooled_tensors,
                args=(proxies, source.cpu(), results),
            )
            consumer.start()
            try:
                self.assertIsNone(await asyncio.to_thread(results.get, True, 60))
                await asyncio.to_thread(consumer.join, 10)
                self.assertEqual(consumer.exitcode, 0)
            finally:
                if consumer.is_alive():
                    consumer.terminate()
                    consumer.join(10)
                results.close()
                results.join_thread()
            deadline = time.monotonic() + 5
            while pool.active_lease_count and time.monotonic() < deadline:
                await asyncio.sleep(0.01)
            self.assertEqual(pool.active_lease_count, 0)
            publisher.admission.timeout_ms = 1000
            await publisher.prepare([rejected])
            pool.cancel_proxy(rejected.mm_items[0].feature)
            self.assertEqual(publisher.admission.pending_bytes, 0)

        try:
            with patch.object(pool, "publish_reserved_batch", isolated_publish):
                asyncio.run(run())
        finally:
            publisher.shutdown()
            pool.shutdown()
            _pool_handle_cache_clear()

    def test_proxy_construction_failure_reclaims_only_after_copy(self):
        pool = MmItemMemoryPool(1 << 20, 0.01, 0, 1)
        source = (
            torch.arange(1024, dtype=torch.float32, device="cuda").reshape(32, 32).T
        )
        leases = pool.try_reserve_batch([source.numel() * source.element_size()])
        try:
            with patch(
                "sglang.srt.multimodal.transport.cuda_ipc.CudaIpcTensorTransportProxy",
                side_effect=RuntimeError("proxy construction failed"),
            ):
                with self.assertRaisesRegex(RuntimeError, "proxy construction failed"):
                    pool.publish_reserved_batch([[source]], leases)
            deadline = time.monotonic() + 5
            while pool.active_lease_count and time.monotonic() < deadline:
                time.sleep(0.01)
            self.assertEqual(pool.active_lease_count, 0)
            self.assertEqual(pool._pool._copy_sources, {})
            self.assertEqual(pool.capacity_snapshot()["cpu_spills"], 0)
        finally:
            pool.shutdown()

    def test_cancelled_native_publication_retains_charge_until_completion(self):
        pool = MmItemMemoryPool(1 << 20, 0.01, 0, 1)
        publisher = BoundedFeaturePublisher(pool, 1000, cpu_tensors=False)
        started, complete = threading.Event(), threading.Event()
        publish = pool.publish_reserved_batch

        def delayed_publish(*args):
            started.set()
            if not complete.wait(10):
                raise TimeoutError("test did not release publisher")
            return publish(*args)

        async def run():
            output = MultimodalProcessorOutput(
                input_ids=[1],
                mm_items=[
                    MultimodalDataItem(
                        modality=Modality.IMAGE, feature=torch.ones(100, device="cuda")
                    )
                ],
            )
            with patch.object(pool, "publish_reserved_batch", delayed_publish):
                task = asyncio.create_task(publisher.prepare([output]))
                self.assertTrue(await asyncio.to_thread(started.wait, 5))
                task.cancel()
                with self.assertRaises(asyncio.CancelledError):
                    await task
                task.cancel()
                self.assertEqual(publisher.admission.pending_count, 1)
                self.assertEqual(pool.active_lease_count, 1)
                complete.set()
                deadline = time.monotonic() + 10
                while (
                    publisher.admission.pending_count or pool.active_lease_count
                ) and time.monotonic() < deadline:
                    await asyncio.sleep(0.01)
                self.assertEqual(
                    (publisher.admission.pending_count, pool.active_lease_count), (0, 0)
                )
                self.assertIsInstance(output.mm_items[0].feature, torch.Tensor)

        try:
            asyncio.run(run())
        finally:
            complete.set()
            publisher.shutdown()
            pool.shutdown()

    def test_cancelled_processor_worker_does_not_publish_an_orphaned_lease(self):
        """cancelling the await cannot stop a running HF call; publish after receipt"""

        async def run(cancel):
            loop = asyncio.get_running_loop()
            entered = asyncio.Event()
            resume = threading.Event()

            def pause():
                loop.call_soon_threadsafe(entered.set)
                if not resume.wait(timeout=10):
                    raise TimeoutError("processor was not resumed")

            hf_processor = _PausedHFProcessor(pause)
            with patch.object(BaseMultimodalProcessor, "__abstractmethods__", set()):
                processor = BaseMultimodalProcessor.__new__(BaseMultimodalProcessor)
            processor._processor = hf_processor
            processor._tokenizer = hf_processor.tokenizer
            processor._tokenizer_auto_adds_specials = False
            processor.image_config = {}
            processor.video_config = {}
            processor.audio_config = {}
            processor.ATTR_NAME_TO_MODALITY = {"pixel_values": Modality.IMAGE}
            processor.FEATURE_NAMES = ["pixel_values"]
            processor.mm_feature_transport = "cuda_ipc"
            processor.use_cuda_ipc = True
            processor.use_ipc_pool_handle_cache = True
            processor.processor_fingerprint = "sha256:" + "12" * 32
            pool = MmItemMemoryPool(1 << 20, 0.01, 0, consumer_count=1)
            processor.cudaipc_mmfeature_pool = pool
            executor = MultimodalProcessorExecutor(lambda: hf_processor, max_workers=1)
            processor.mm_processor_executor = executor
            feature_threads = []
            publish_threads = []
            prepare_features = processor._precompute_hashes_before_cpu_transfer
            publish_features = processor._prepare_mm_items_for_transport

            def record_prepare(*args, **kwargs):
                feature_threads.append(threading.get_ident())
                return prepare_features(*args, **kwargs)

            def record_publish(*args, **kwargs):
                publish_threads.append(threading.get_ident())
                return publish_features(*args, **kwargs)

            processor._precompute_hashes_before_cpu_transfer = record_prepare
            processor._prepare_mm_items_for_transport = record_publish
            task = asyncio.create_task(
                processor.process_and_combine_mm_data_async(
                    BaseMultiModalProcessorOutput(
                        input_text="image", images=[torch.zeros(3, 2, 2)]
                    ),
                    MultimodalSpecialTokens(image_token_id=99),
                )
            )
            try:
                await asyncio.wait_for(entered.wait(), timeout=10)
                if cancel:
                    task.cancel()
                    with self.assertRaises(asyncio.CancelledError):
                        await task
                resume.set()
                if not cancel:
                    items, input_ids, _ = await task
                    self.assertEqual(input_ids.tolist(), [7, 99, 8])
                    self.assertEqual(items[0].offsets, [(1, 1)])
                    self.assertIsInstance(items[0].feature, CudaIpcTensorTransportProxy)
                    self.assertEqual(pool.active_lease_count, 1)
                    pool.cancel_proxy(items[0].feature)
                await asyncio.to_thread(executor.shutdown)
                torch.cuda.synchronize()
                deadline = time.monotonic() + 5
                while pool.active_lease_count and time.monotonic() < deadline:
                    await asyncio.sleep(0.01)
                self.assertEqual(pool.active_lease_count, 0)
                self.assertEqual(len(feature_threads), 1)
                self.assertNotEqual(feature_threads[0], threading.get_ident())
                self.assertEqual(
                    publish_threads, [] if cancel else [threading.get_ident()]
                )
            finally:
                resume.set()
                await asyncio.to_thread(executor.shutdown)
                pool.shutdown()

        for cancel in (True, False):
            with self.subTest(cancel=cancel):
                asyncio.run(run(cancel))

    def test_pooled_tensor_reconstructs_in_spawned_process(self):
        self._run_spawned_round_trip(guarded=False)

    def test_guarded_tensor_reconstructs_in_spawned_process(self):
        self._run_spawned_round_trip(guarded=True)

    def _run_spawned_round_trip(self, *, guarded):
        """Consumer releases the pool mapping before the producer tears down."""
        ctx = mp.get_context("spawn")
        proxy_queue = ctx.Queue()
        producer_results = ctx.Queue()
        consumer_done = ctx.Event()
        producer = ctx.Process(
            target=_produce_pooled_tensor,
            args=(proxy_queue, consumer_done, producer_results, guarded),
        )
        producer.start()
        proxy = reconstructed = None
        producer_result = None
        try:
            try:
                proxy, expected = proxy_queue.get(timeout=60)
            except queue.Empty:
                producer_result = producer_results.get(timeout=5)
                _status, payload = producer_result
                self.fail(
                    f"CUDA IPC producer failed before sending its proxy: {payload}"
                )

            reconstructed = proxy.reconstruct_on_target_device(0)
            torch.cuda.synchronize()
            self.assertEqual(reconstructed.cpu().tolist(), expected)
        finally:
            # The scheduler retains this cache for its lifetime. The test's
            # consumer exits quickly, so it must close the mapping before the
            # producer destroys the shared allocation.
            del reconstructed, proxy
            _pool_handle_cache_clear()
            gc.collect()
            torch.cuda.ipc_collect()
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
            self.assertEqual(producer.exitcode, 0)

    def test_borrowed_tensor_keeps_lease_until_explicit_release(self):
        ctx = mp.get_context("spawn")
        proxy_queue = ctx.Queue()
        producer_results = ctx.Queue()
        consumer_done = ctx.Event()
        producer = ctx.Process(
            target=_produce_pooled_tensor,
            args=(proxy_queue, consumer_done, producer_results),
        )
        producer.start()
        proxy = borrowed = consumed = None
        producer_result = None
        try:
            try:
                proxy, expected = proxy_queue.get(timeout=60)
            except queue.Empty:
                producer_result = producer_results.get(timeout=5)
                _status, payload = producer_result
                self.fail(
                    f"CUDA IPC producer failed before sending its proxy: {payload}"
                )

            borrowed = proxy.borrow_on_target_device(0)
            self.assertIsNotNone(borrowed)
            consumed = borrowed + 1
            torch.cuda.synchronize()

            self.assertFalse(proxy._consumer_acknowledged)
            proxy.release_without_reconstruction()
            torch.cuda.synchronize()

            self.assertEqual(
                consumed.cpu().tolist(),
                (torch.tensor(expected) + 1).tolist(),
            )
            self.assertTrue(proxy._consumer_acknowledged)
            self.assertIsNone(proxy._borrowed_storage)
        finally:
            del consumed, borrowed, proxy
            _pool_handle_cache_clear()
            gc.collect()
            torch.cuda.ipc_collect()
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
            self.assertEqual(producer.exitcode, 0)

    def test_failed_reconstruction_releases_pooled_tensor(self):
        ctx = mp.get_context("spawn")
        proxy_queue = ctx.Queue()
        producer_results = ctx.Queue()
        consumer_done = ctx.Event()
        producer = ctx.Process(
            target=_produce_pooled_tensor,
            args=(proxy_queue, consumer_done, producer_results),
        )
        producer.start()
        proxy = item = output = None
        producer_result = None
        original_empty = torch.empty
        try:
            try:
                proxy, _expected = proxy_queue.get(timeout=60)
            except queue.Empty:
                producer_result = producer_results.get(timeout=5)
                _status, payload = producer_result
                self.fail(
                    f"CUDA IPC producer failed before sending its proxy: {payload}"
                )

            output_shape = proxy.proxy_state["ipc_extra"]["recons_shape"]

            def fail_destination_allocation(size, *args, **kwargs):
                if isinstance(size, (tuple, torch.Size)) and tuple(size) == tuple(
                    output_shape
                ):
                    raise RuntimeError("forced reconstruction failure")
                return original_empty(size, *args, **kwargs)

            item = MultimodalDataItem(
                modality=Modality.IMAGE,
                hash=1,
                pad_value=1,
                feature=proxy,
            )
            output = MultimodalProcessorOutput(input_ids=[1], mm_items=[item])
            with (
                patch(
                    "sglang.srt.multimodal.transport.cuda_ipc.torch.empty",
                    side_effect=fail_destination_allocation,
                ),
                self.assertRaisesRegex(RuntimeError, "forced reconstruction failure"),
            ):
                MultimodalInputs.from_processor_output(output)

            torch.cuda.synchronize()
            self.assertTrue(proxy._consumer_acknowledged)
        finally:
            # the processor output also owns the proxy and its imported mapping
            del output, item, proxy
            _pool_handle_cache_clear()
            gc.collect()
            torch.cuda.ipc_collect()
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
            self.assertEqual(producer.exitcode, 0)

    def test_uncached_mapping_waits_before_proxy_release(self):
        proxy = object.__new__(CudaIpcTensorTransportProxy)
        proxy.proxy_state = {"ipc_extra": {"use_pool_handle_cache": False}}
        proxy._pool_storage = None
        stream = Mock()

        with patch(
            "sglang.srt.multimodal.transport.cuda_ipc.torch.cuda.current_stream",
            return_value=stream,
        ):
            proxy._retain_storage_until_stream_completes(object(), 0)

        stream.synchronize.assert_called_once_with()
        self.assertIsNone(proxy._pool_storage)

    def test_failed_item_batch_releases_undispatched_pool_slice(self):
        from sglang.srt.managers.req_types import MultimodalDataItem
        from sglang.srt.multimodal.modality import Modality
        from sglang.srt.multimodal.processors.base_processor import (
            BaseMultimodalProcessor,
        )

        pool = MmItemMemoryPool(
            memory_size=1 << 20,
            recycle_interval=0.01,
            base_gpu_id=0,
            consumer_count=4,
        )
        with patch.object(BaseMultimodalProcessor, "__abstractmethods__", set()):
            processor = BaseMultimodalProcessor.__new__(BaseMultimodalProcessor)
        processor.use_cuda_ipc = True
        processor.use_ipc_pool_handle_cache = True
        processor.cudaipc_mmfeature_pool = pool
        processor.processor_fingerprint = "sha256:" + "12" * 32
        features = [
            torch.ones(16, device="cuda"),
            torch.empty(0, device="cuda"),
        ]
        items = [
            MultimodalDataItem(modality=Modality.IMAGE, feature=feature)
            for feature in features
        ]

        try:
            with self.assertRaisesRegex(ValueError, "empty tensor"):
                processor._prepare_mm_items_for_transport(items)

            deadline = time.monotonic() + 5
            while pool.active_lease_count and time.monotonic() < deadline:
                time.sleep(0.01)
            self.assertEqual(pool.active_lease_count, 0)
            self.assertIs(items[0].feature, features[0])
            self.assertIs(items[1].feature, features[1])
        finally:
            pool.shutdown()

    def test_rejected_request_releases_unconsumed_pool_slice(self):
        ctx = mp.get_context("spawn")
        proxy_queue = ctx.Queue()
        producer_results = ctx.Queue()
        consumer_done = ctx.Event()
        producer = ctx.Process(
            target=_produce_pooled_tensor,
            args=(proxy_queue, consumer_done, producer_results),
        )
        producer.start()
        proxy = None
        producer_result = None
        try:
            proxy, _ = proxy_queue.get(timeout=60)
            item = MultimodalDataItem(modality=Modality.IMAGE, feature=proxy)
            mm_inputs = MultimodalInputs(mm_items=[item])

            mm_inputs.release_features()
            torch.cuda.synchronize()

            self.assertIsNone(item.feature)
        finally:
            del proxy
            _pool_handle_cache_clear()
            gc.collect()
            torch.cuda.ipc_collect()
            consumer_done.set()
            producer.join(timeout=60)
            try:
                producer_result = producer_results.get(timeout=5)
                status, payload = producer_result
                self.assertEqual(status, "ok", payload)
            finally:
                if producer.is_alive():
                    producer.terminate()
                    producer.join(timeout=10)
            self.assertEqual(producer.exitcode, 0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
