"""Cross-process coverage of request-local CUDA IPC packing."""

import gc
import multiprocessing as mp
import pickle
import time
import unittest
from contextlib import contextmanager
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.srt.managers.schedule_batch import (
    Modality,
    MultimodalDataItem,
    MultimodalInputs,
    MultimodalProcessorOutput,
)
from sglang.srt.multimodal.processors.base_processor import BaseMultimodalProcessor
from sglang.srt.multimodal.transport.cuda_ipc import (
    DEFER_CUDA_IPC_FEATURE_RECONSTRUCTION_KEY,
    CudaIpcPackedTensorTransportProxy,
    MmItemMemoryPool,
    _pool_handle_cache_clear,
)
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=45, stage="base-b", runner_config="1-gpu-large")


def _features(device):
    # Odd byte counts, noncontiguous layouts and scalars exercise typed offsets.
    return [
        torch.arange(3, dtype=torch.uint8, device=device),
        torch.arange(12, dtype=torch.float16, device=device).reshape(3, 4).T,
        torch.tensor(7.5, dtype=torch.float32, device=device),
        torch.arange(14, dtype=torch.bfloat16, device=device).reshape(2, 7),
    ]


def _wait_for_recycle(pool):
    deadline = time.monotonic() + 5
    while pool.active_lease_count and time.monotonic() < deadline:
        time.sleep(0.01)
    if pool.active_lease_count:
        raise TimeoutError("CUDA IPC lease was not recycled")


def _publish_batch(commands, results, consumer_count, use_handle_cache):
    pool = None
    try:
        torch.cuda.set_device(0)
        pool = MmItemMemoryPool(1 << 20, 0.01, 0, consumer_count)
        sources = _features("cuda")
        proxies = pool.wrap_tensors(sources, use_pool_handle_cache=use_handle_cache)
        offset = proxies[0].owner.proxy_state["ipc_extra"]["pool_byte_offset"]
        # The production wire format must preserve the siblings' shared owner.
        results.put(pickle.dumps(proxies))
        while True:
            command = commands.get(timeout=60)
            if command == "stop":
                _wait_for_recycle(pool)
                break
            if command == "reuse":
                _wait_for_recycle(pool)
                replacements = pool.wrap_tensors(
                    [torch.zeros_like(source) for source in sources],
                    use_pool_handle_cache=use_handle_cache,
                )
                reused_offset = replacements[0].owner.proxy_state["ipc_extra"][
                    "pool_byte_offset"
                ]
                torch.cuda.synchronize()
                pool.cancel_proxy(replacements[0])
                _wait_for_recycle(pool)
                results.put(("reused", reused_offset == offset))
            else:
                results.put(("leases", pool.active_lease_count))
        results.put(("done", None))
    except BaseException as error:
        results.put(("error", repr(error)))
    finally:
        if pool is not None:
            pool.shutdown()
            del pool
        gc.collect()
        torch.cuda.ipc_collect()


def _consume_second_rank(wire, device, results):
    # Real TP ranks use separate processes, each with its own CUDA context.
    torch.cuda.set_device(device)
    proxies = pickle.loads(wire)
    try:
        for proxy, expected in zip(proxies, _features("cpu"), strict=True):
            actual = proxy.reconstruct_on_target_device(device, consumer_rank=1)
            torch.testing.assert_close(actual.cpu(), expected, rtol=0, atol=0)
        results.put(("ok", None))
    except BaseException as error:
        results.put(("error", repr(error)))
    finally:
        proxies[0].owner._pool_storage = None
        _pool_handle_cache_clear()
        gc.collect()
        torch.cuda.ipc_collect()


@contextmanager
def _published_batch(consumer_count=1, use_handle_cache=True):
    context = mp.get_context("spawn")
    commands, results = context.Queue(), context.Queue()
    producer = context.Process(
        target=_publish_batch,
        args=(commands, results, consumer_count, use_handle_cache),
    )
    producer.start()
    proxies = []
    try:
        wire = results.get(timeout=60)
        if not isinstance(wire, bytes):
            raise AssertionError(wire)
        proxies = pickle.loads(wire)
        yield proxies, wire, commands, results
    finally:
        if proxies:
            owner = proxies[0].owner
            owner.release_without_reconstruction(consumer_count)
            torch.cuda.synchronize()
            owner._pool_storage = None
        _pool_handle_cache_clear()
        gc.collect()
        torch.cuda.ipc_collect()
        commands.put("stop")
        producer.join(timeout=60)
        if producer.is_alive():
            producer.terminate()
            producer.join(timeout=10)
            raise AssertionError("CUDA IPC producer did not stop")
        assert producer.exitcode == 0
        assert results.get(timeout=5) == ("done", None)


class TestPackedCudaIpcTransport(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        if not torch.cuda.is_available():
            raise unittest.SkipTest("CUDA is required")
        torch.cuda.set_device(0)

    def test_shared_reconstruction_survives_pool_reuse(self):
        for use_cache in (False, True):
            with (
                self.subTest(handle_cache=use_cache),
                _published_batch(use_handle_cache=use_cache) as (
                    proxies,
                    _,
                    commands,
                    results,
                ),
            ):
                owner = proxies[0].owner
                self.assertTrue(all(proxy.owner is owner for proxy in proxies))
                with patch.object(
                    owner, "_open_pool_slice", wraps=owner._open_pool_slice
                ) as open_slice:
                    # Out-of-order materialization must still copy exactly once.
                    for proxy in reversed(proxies):
                        proxy.reconstruct_on_target_device(0)
                    self.assertEqual(open_slice.call_count, 1)
                commands.put("reuse")
                self.assertEqual(results.get(timeout=10), ("reused", True))
                for proxy, expected in zip(proxies, _features("cpu"), strict=True):
                    first = proxy.reconstruct_tensor
                    # A re-prefill reuses owned storage, never the overwritten pool.
                    self.assertIs(proxy.reconstruct_on_target_device(0), first)
                    torch.testing.assert_close(first.cpu(), expected, rtol=0, atol=0)
                    self.assertEqual(
                        first.untyped_storage().data_ptr(),
                        owner.reconstruct_tensor.untyped_storage().data_ptr(),
                    )

    def test_all_consumers_must_ack_before_reuse(self):
        with _published_batch(consumer_count=2) as (proxies, wire, commands, results):
            proxies[0].reconstruct_on_target_device(0, consumer_rank=0)
            torch.cuda.synchronize()
            commands.put("inspect")
            self.assertEqual(results.get(timeout=10), ("leases", 1))
            # One-GPU CI still checks independent processes and TP ack slots.
            device = min(1, torch.cuda.device_count() - 1)
            context = mp.get_context("spawn")
            consumer_results = context.Queue()
            consumer = context.Process(
                target=_consume_second_rank, args=(wire, device, consumer_results)
            )
            consumer.start()
            try:
                self.assertEqual(consumer_results.get(timeout=60), ("ok", None))
            finally:
                consumer.join(timeout=60)
                if consumer.is_alive():
                    consumer.terminate()
                    consumer.join(timeout=10)
            self.assertEqual(consumer.exitcode, 0)
            commands.put("reuse")
            self.assertEqual(results.get(timeout=10), ("reused", True))

    def test_abandoned_batch_releases_once_without_copying(self):
        with _published_batch() as (proxies, _, commands, results):
            for proxy in proxies:
                proxy.release_without_reconstruction()
            self.assertIsNone(proxies[0].owner.reconstruct_tensor)
            commands.put("reuse")
            self.assertEqual(results.get(timeout=10), ("reused", True))
            with self.assertRaisesRegex(RuntimeError, "already released its lease"):
                proxies[-1].reconstruct_on_target_device(0)

    def test_scheduler_failure_releases_shared_lease(self):
        with _published_batch() as (proxies, _, commands, results):
            items = [
                MultimodalDataItem(
                    modality=Modality.IMAGE, feature=proxy, hash=1, pad_value=1
                )
                for proxy in proxies
            ]
            owner = proxies[0].owner
            shape = owner.proxy_state["ipc_extra"]["recons_shape"]
            original_empty = torch.empty

            def fail_allocation(size, *args, **kwargs):
                if isinstance(size, (tuple, torch.Size)) and tuple(size) == tuple(
                    shape
                ):
                    raise RuntimeError("forced packed allocation failure")
                return original_empty(size, *args, **kwargs)

            with (
                patch(
                    "sglang.srt.multimodal.transport.cuda_ipc.torch.empty",
                    side_effect=fail_allocation,
                ),
                self.assertRaisesRegex(
                    RuntimeError, "forced packed allocation failure"
                ),
            ):
                MultimodalInputs.from_processor_output(
                    MultimodalProcessorOutput(input_ids=[1], mm_items=items)
                )
            commands.put("reuse")
            self.assertEqual(results.get(timeout=10), ("reused", True))

    def test_scheduler_materializes_before_embedding_cache_lookup(self):
        with _published_batch() as (proxies, _, commands, results):
            items = [
                MultimodalDataItem(
                    modality=Modality.IMAGE, feature=proxy, hash=index + 1, pad_value=1
                )
                for index, proxy in enumerate(proxies)
            ]
            for item in items:
                item.model_specific_data[DEFER_CUDA_IPC_FEATURE_RECONSTRUCTION_KEY] = (
                    True
                )
            inputs = MultimodalInputs.from_processor_output(
                MultimodalProcessorOutput(input_ids=[1], mm_items=items)
            )
            commands.put("reuse")
            self.assertEqual(results.get(timeout=10), ("reused", True))
            for index, (item, expected) in enumerate(
                zip(inputs.mm_items, _features("cpu"), strict=True)
            ):
                self.assertIsInstance(item.feature, torch.Tensor)
                self.assertEqual(item.hash, index + 1)
                torch.testing.assert_close(item.feature.cpu(), expected, rtol=0, atol=0)

    def test_pack_failure_and_distinct_request_ownership(self):
        pool = MmItemMemoryPool(1 << 20, 0.01, 0, 1)
        features = _features("cuda")
        try:
            with (
                patch(
                    "sglang.srt.multimodal.transport.cuda_ipc.CudaIpcPackedTensorTransportProxy",
                    side_effect=RuntimeError("forced proxy creation failure"),
                ),
                self.assertRaisesRegex(RuntimeError, "forced proxy creation failure"),
            ):
                pool.wrap_tensors(features, use_pool_handle_cache=True)
            _wait_for_recycle(pool)
            first = pool.wrap_tensors(features, use_pool_handle_cache=True)
            second = pool.wrap_tensors(features, use_pool_handle_cache=True)
            self.assertIsNot(first[0].owner, second[0].owner)
            self.assertEqual(pool.active_lease_count, 2)
            for batch in (first, second):
                for proxy in batch:
                    pool.cancel_proxy(proxy)
            _wait_for_recycle(pool)
        finally:
            pool.shutdown()

    def test_noncontiguous_inputs_do_not_accumulate_staging_buffers(self):
        pool = MmItemMemoryPool(16 << 20, 0.01, 0, 1)
        features = [
            torch.arange(512 * 512, device="cuda").reshape(512, 512).T for _ in range(4)
        ]
        try:
            torch.cuda.synchronize()
            allocated = torch.cuda.memory_allocated()
            torch.cuda.reset_peak_memory_stats()
            proxies = pool.wrap_tensors(features, use_pool_handle_cache=True)
            self.assertIsNotNone(proxies)
            staging_peak = torch.cuda.max_memory_allocated() - allocated
            image_bytes = features[0].numel() * features[0].element_size()
            self.assertLess(staging_peak, 2 * image_bytes)
            pool.cancel_proxy(proxies[0])
            _wait_for_recycle(pool)
        finally:
            pool.shutdown()

    def test_full_pool_falls_back_to_individual_transport(self):
        pool = MmItemMemoryPool(1 << 20, 0.01, 0, 1)
        with patch.object(BaseMultimodalProcessor, "__abstractmethods__", set()):
            processor = BaseMultimodalProcessor.__new__(BaseMultimodalProcessor)
        processor.use_cuda_ipc = True
        processor.use_ipc_pool_handle_cache = True
        processor.cudaipc_mmfeature_pool = pool
        # One image fits, but the packed request exceeds the existing pool budget.
        features = [torch.ones(160_000, device="cuda") for _ in range(2)]
        items = [
            MultimodalDataItem(modality=Modality.IMAGE, feature=feature)
            for feature in features
        ]
        try:
            with patch(
                "sglang.srt.multimodal.processors.base_processor.get_mm",
                return_value=SimpleNamespace(mm_enable_dp_encoder=False),
            ):
                processor._prepare_mm_items_for_transport(items)
            self.assertEqual(pool.active_lease_count, 1)
            self.assertNotIsInstance(
                items[0].feature, CudaIpcPackedTensorTransportProxy
            )
            self.assertEqual(items[1].feature.device.type, "cpu")
            torch.testing.assert_close(
                items[1].feature, features[1].cpu(), rtol=0, atol=0
            )
            pool.cancel_proxy(items[0].feature)
            _wait_for_recycle(pool)
        finally:
            pool.shutdown()

    def test_generic_processor_policy_and_rollback(self):
        pool = MmItemMemoryPool(1 << 20, 0.01, 0, 1)
        with patch.object(BaseMultimodalProcessor, "__abstractmethods__", set()):
            processor = BaseMultimodalProcessor.__new__(BaseMultimodalProcessor)
        processor.use_cuda_ipc = True
        processor.use_ipc_pool_handle_cache = True
        processor.cudaipc_mmfeature_pool = pool
        try:
            for encoder_dp in (False, True):
                for modality in (Modality.IMAGE, Modality.VIDEO, Modality.AUDIO):
                    with self.subTest(encoder_dp=encoder_dp, modality=modality):
                        items = [
                            MultimodalDataItem(
                                modality=modality, feature=feature, hash=1, pad_value=1
                            )
                            for feature in _features("cuda")
                        ]
                        with patch(
                            "sglang.srt.multimodal.processors.base_processor.get_mm",
                            return_value=SimpleNamespace(
                                mm_enable_dp_encoder=encoder_dp
                            ),
                        ):
                            processor._prepare_mm_items_for_transport(items)
                        self.assertEqual(
                            pool.active_lease_count, 4 if encoder_dp else 1
                        )
                        for item in items:
                            self.assertEqual(
                                isinstance(
                                    item.feature, CudaIpcPackedTensorTransportProxy
                                ),
                                not encoder_dp,
                            )
                            item.model_specific_data[
                                DEFER_CUDA_IPC_FEATURE_RECONSTRUCTION_KEY
                            ] = True
                            self.assertEqual(
                                item.can_defer_cuda_ipc_feature_reconstruction(),
                                encoder_dp,
                            )
                            pool.cancel_proxy(item.feature)
                        _wait_for_recycle(pool)

            features = _features("cuda") + [torch.empty(0, device="cuda")]
            items = [
                MultimodalDataItem(modality=Modality.IMAGE, feature=feature)
                for feature in features
            ]
            with (
                patch(
                    "sglang.srt.multimodal.processors.base_processor.get_mm",
                    return_value=SimpleNamespace(mm_enable_dp_encoder=False),
                ),
                self.assertRaisesRegex(ValueError, "empty tensor"),
            ):
                processor._prepare_mm_items_for_transport(items)
            _wait_for_recycle(pool)
            for item, original in zip(items, features, strict=True):
                self.assertIs(item.feature, original)
        finally:
            pool.shutdown()


if __name__ == "__main__":
    unittest.main()
