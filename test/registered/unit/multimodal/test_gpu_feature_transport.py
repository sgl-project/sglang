import asyncio
import concurrent.futures
import threading
import unittest
from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, call, patch

import torch

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=10, suite="base-a-test-cpu")


class TestCudaVmmFeatureTransport(unittest.TestCase):
    def test_failed_consumer_reconstruction_releases_remaining_proxies(self):
        from sglang.srt.managers.schedule_batch import (
            Modality,
            MultimodalDataItem,
            MultimodalInputs,
            MultimodalProcessorOutput,
        )
        from sglang.srt.multimodal.transport.cuda_ipc import (
            CudaIpcTensorTransportProxy,
        )

        class FakeProxy(CudaIpcTensorTransportProxy):
            def __init__(self, *, fail_reconstruct=False, fail_release=False):
                self.fail_reconstruct = fail_reconstruct
                self.fail_release = fail_release
                self.released = False

            def reconstruct_on_target_device(self, _device, consumer_count=1):
                if self.fail_reconstruct:
                    raise RuntimeError("reconstruct failed")
                return torch.ones(1)

            def release_without_reconstruction(self, consumer_count=1):
                self.released = True
                if self.fail_release:
                    raise RuntimeError("release failed")

        reconstructed = FakeProxy()
        failed = FakeProxy(fail_reconstruct=True, fail_release=True)
        remaining = FakeProxy()
        items = [
            MultimodalDataItem(
                modality=Modality.IMAGE,
                hash=1,
                pad_value=1,
                feature=reconstructed,
            ),
            MultimodalDataItem(
                modality=Modality.IMAGE,
                hash=2,
                pad_value=2,
                feature=failed,
            ),
            MultimodalDataItem(
                modality=Modality.IMAGE,
                hash=3,
                pad_value=3,
                feature=remaining,
            ),
        ]
        output = MultimodalProcessorOutput(input_ids=[1], mm_items=items)

        with (
            patch(
                "sglang.srt.managers.schedule_batch.torch.cuda.current_device",
                return_value=0,
            ),
            self.assertRaisesRegex(RuntimeError, "reconstruct failed"),
        ):
            MultimodalInputs.from_processor_output(output)

        self.assertIsInstance(items[0].feature, torch.Tensor)
        self.assertTrue(failed.released)
        self.assertTrue(remaining.released)

    def test_abandoned_packed_proxy_releases_shared_owner(self):
        from sglang.srt.utils.cuda_vmm_transport_utils import (
            CudaVmmPackedTensorTransportProxy,
        )

        owner = MagicMock()
        proxy = object.__new__(CudaVmmPackedTensorTransportProxy)
        proxy._packed_owner = owner
        proxy._consumer_acknowledged = False

        proxy.release_without_reconstruction(consumer_count=2)

        owner.acknowledge_consumption.assert_called_once_with(2)
        self.assertTrue(proxy._consumer_acknowledged)

    def test_partial_pool_release_can_be_retried(self):
        from sglang.srt.utils import cuda_vmm_transport_utils as vmm

        pool = object.__new__(vmm.CudaVmmMemoryPool)
        pool.memory_pool = object()
        pool.use_fabric = True
        pool.shareable_handle = b"handle"
        allocation = MagicMock()
        allocation.close.side_effect = [
            RuntimeError("forced allocation close failure"),
            None,
        ]
        pool._allocation = allocation
        pool.device_index = 0

        with (
            patch.object(vmm.torch.cuda, "device", return_value=nullcontext()),
            self.assertRaisesRegex(RuntimeError, "forced allocation close failure"),
        ):
            pool._release_allocation()

        self.assertIs(pool._allocation, allocation)

        with patch.object(vmm.torch.cuda, "device", return_value=nullcontext()):
            pool._release_allocation()

        self.assertIsNone(pool._allocation)
        self.assertEqual(allocation.close.call_count, 2)

    def test_model_class_controls_cuda_vmm_opt_in(self):
        from sglang.srt.managers.tokenizer_manager import TokenizerManager
        from sglang.srt.runtime_context import get_context

        class SupportedModel:
            supports_cuda_vmm_feature_transport = True

        class UnsupportedModel:
            pass

        override = get_context().override_server_args(mm_feature_transport="cuda_vmm")
        override.install()
        self.addCleanup(override.restore)
        manager = object.__new__(TokenizerManager)
        manager.model_config = object()

        with patch(
            "sglang.srt.model_loader.utils.get_model_architecture",
            return_value=(SupportedModel, "supported"),
        ):
            manager._validate_cuda_vmm_feature_transport_support()

        with (
            patch(
                "sglang.srt.model_loader.utils.get_model_architecture",
                return_value=(UnsupportedModel, "unsupported"),
            ),
            self.assertRaisesRegex(ValueError, "UnsupportedModel"),
        ):
            manager._validate_cuda_vmm_feature_transport_support()

    def test_cpu_transport_skips_model_opt_in_lookup(self):
        from sglang.srt.managers.tokenizer_manager import TokenizerManager
        from sglang.srt.runtime_context import get_context

        override = get_context().override_server_args(mm_feature_transport="cpu")
        override.install()
        self.addCleanup(override.restore)
        manager = object.__new__(TokenizerManager)
        manager.model_config = object()

        with patch(
            "sglang.srt.model_loader.utils.get_model_architecture"
        ) as get_model_architecture:
            manager._validate_cuda_vmm_feature_transport_support()

        get_model_architecture.assert_not_called()

    def test_vmm_transport_initializes_pool(self):
        from sglang.srt.runtime_context import get_context
        from sglang.srt.utils import cuda_vmm_transport_utils as vmm

        server_args = SimpleNamespace(
            mm_feature_transport="cuda_vmm",
            tokenizer_worker_num=2,
            base_gpu_id=3,
            tp_size=4,
            nnodes=1,
        )
        # The consumer count comes from the published topology.
        override = get_context().override_server_args(
            enable_dp_attention=False, tp_size=4, mm_feature_transport="cuda_vmm"
        )
        override.install()
        self.addCleanup(override.restore)
        pool = object()
        with (
            patch.object(vmm, "get_mm_feature_pool_size_per_worker", return_value=123),
            patch.object(vmm, "CudaVmmMemoryPool", return_value=pool) as pool_class,
        ):
            transport = vmm.CudaVmmFeatureTransport(server_args, SimpleNamespace())

        self.assertIs(transport.pool, pool)
        pool_class.assert_called_once_with(
            memory_size=123,
            recycle_interval=vmm.MM_ITEM_MEMORY_POOL_RECYCLE_INTERVAL,
            base_gpu_id=3,
            consumer_count=4,
            allow_posix_fallback=True,
        )

    def test_disabled_transport_is_a_noop(self):
        from sglang.srt.runtime_context import get_context
        from sglang.srt.utils.cuda_vmm_transport_utils import (
            CudaVmmFeatureTransport,
        )

        # The transport choice is a bag leaf.
        override = get_context().override_server_args(mm_feature_transport="cpu")
        override.install()
        self.addCleanup(override.restore)
        transport = CudaVmmFeatureTransport(SimpleNamespace(), None)

        self.assertEqual(transport.prepare_for_dispatch([None]), [])
        self.assertEqual(asyncio.run(transport.prepare_for_dispatch_async([None])), [])
        transport.cancel_for_dispatch([])
        transport.shutdown()
        self.assertIsNone(transport.pool)

    def test_vmm_transport_requires_processor(self):
        from sglang.srt.runtime_context import get_context
        from sglang.srt.utils.cuda_vmm_transport_utils import (
            CudaVmmFeatureTransport,
        )

        override = get_context().override_server_args(mm_feature_transport="cuda_vmm")
        override.install()
        self.addCleanup(override.restore)
        with self.assertRaisesRegex(RuntimeError, "multimodal processor"):
            CudaVmmFeatureTransport(SimpleNamespace(), None)

    def test_image_features_are_packed_per_request(self):
        from sglang.srt.managers.schedule_batch import Modality, MultimodalDataItem
        from sglang.srt.utils.cuda_vmm_transport_utils import (
            CudaVmmFeatureTransport,
        )

        transport = object.__new__(CudaVmmFeatureTransport)
        transport.pool = MagicMock()
        features = [torch.arange(4), torch.arange(4, 8)]
        proxies = [object(), object()]
        transport.pool.wrap_tensors.return_value = proxies
        items = [
            MultimodalDataItem(modality=Modality.IMAGE, feature=feature)
            for feature in features
        ]

        transport.wrap_items(items)

        transport.pool.wrap_tensors.assert_called_once_with(features)
        transport.pool.wrap_tensor.assert_not_called()
        self.assertEqual([item.feature for item in items], proxies)

    def test_video_clip_features_are_packed_per_request(self):
        from sglang.srt.managers.schedule_batch import Modality, MultimodalDataItem
        from sglang.srt.utils.cuda_vmm_transport_utils import (
            CudaVmmFeatureTransport,
        )

        transport = object.__new__(CudaVmmFeatureTransport)
        transport.pool = MagicMock()
        features = [torch.arange(4), torch.arange(4, 8)]
        proxies = [object(), object()]
        transport.pool.wrap_tensors.return_value = proxies
        items = [
            MultimodalDataItem(modality=Modality.VIDEO, feature=feature)
            for feature in features
        ]

        transport.wrap_items(items)

        transport.pool.wrap_tensors.assert_called_once_with(features)
        transport.pool.wrap_tensor.assert_not_called()
        self.assertEqual([item.feature for item in items], proxies)

    def test_deferred_features_are_not_packed(self):
        from sglang.srt.managers.schedule_batch import Modality, MultimodalDataItem
        from sglang.srt.utils.cuda_ipc_transport_utils import (
            DEFER_CUDA_IPC_FEATURE_RECONSTRUCTION_KEY,
        )
        from sglang.srt.utils.cuda_vmm_transport_utils import (
            CudaVmmFeatureTransport,
        )

        transport = object.__new__(CudaVmmFeatureTransport)
        transport.pool = MagicMock()
        features = [torch.arange(4), torch.arange(4, 8)]
        proxies = [object(), object()]
        transport.pool.wrap_tensor.side_effect = proxies
        items = [
            MultimodalDataItem(
                modality=Modality.IMAGE,
                feature=feature,
                model_specific_data={DEFER_CUDA_IPC_FEATURE_RECONSTRUCTION_KEY: True},
            )
            for feature in features
        ]

        transport.wrap_items(items)

        transport.pool.wrap_tensors.assert_not_called()
        self.assertEqual(
            transport.pool.wrap_tensor.call_args_list,
            [call(feature) for feature in features],
        )
        self.assertEqual([item.feature for item in items], proxies)

    def test_tensor_containers_fail_closed(self):
        from sglang.srt.managers.schedule_batch import Modality, MultimodalDataItem
        from sglang.srt.utils.cuda_vmm_transport_utils import (
            CudaVmmFeatureTransport,
        )

        transport = object.__new__(CudaVmmFeatureTransport)
        transport.pool = MagicMock()
        item = MultimodalDataItem(
            modality=Modality.IMAGE,
            feature=[torch.arange(4), torch.arange(4, 8)],
        )

        with self.assertRaisesRegex(TypeError, "single tensor"):
            transport.wrap_items([item])

        transport.pool.wrap_tensor.assert_not_called()
        transport.pool.wrap_tensors.assert_not_called()

    def test_partial_failure_restores_tensors_and_cancels_packed_chunk_once(self):
        from sglang.srt.managers.schedule_batch import Modality, MultimodalDataItem
        from sglang.srt.utils.cuda_vmm_transport_utils import (
            CudaVmmFeatureTransport,
            CudaVmmMemoryPool,
            CudaVmmPackedTensorTransportProxy,
            _CudaVmmPackedTransportOwner,
        )

        owner = object.__new__(_CudaVmmPackedTransportOwner)
        owner.control_offset = 64
        owner._producer_cancelled = False
        proxies = [object.__new__(CudaVmmPackedTensorTransportProxy) for _ in range(2)]
        for proxy in proxies:
            proxy._packed_owner = owner

        pool = object.__new__(CudaVmmMemoryPool)
        pool.wrap_tensors = MagicMock(return_value=proxies)
        pool.wrap_tensor = MagicMock(side_effect=RuntimeError("copy failed"))
        pool._cancel_control_offset = MagicMock()
        transport = object.__new__(CudaVmmFeatureTransport)
        transport.pool = pool

        features = [torch.arange(4), torch.arange(4, 8)]
        embedding = torch.arange(2)
        items = [
            MultimodalDataItem(
                modality=Modality.IMAGE,
                feature=features[0],
                precomputed_embeddings=embedding,
            ),
            MultimodalDataItem(modality=Modality.IMAGE, feature=features[1]),
        ]

        with self.assertRaisesRegex(RuntimeError, "copy failed"):
            transport.wrap_items(items)

        for item, feature in zip(items, features, strict=True):
            self.assertIs(item.feature, feature)
        self.assertIs(items[0].precomputed_embeddings, embedding)
        pool._cancel_control_offset.assert_called_once_with(owner.control_offset)

    def test_text_request_uses_base_send_path(self):
        from sglang.srt.managers import tokenizer_manager
        from sglang.srt.managers.tokenizer_manager import TokenizerManager

        manager = object.__new__(TokenizerManager)
        transport = MagicMock()
        transport.prepare_for_dispatch_async = AsyncMock(return_value=[])
        manager.cuda_vmm_feature_transport = transport
        manager._dispatch_to_scheduler = MagicMock()
        tokenized_obj = SimpleNamespace(
            rid="test-request",
            mm_inputs=None,
            time_stats=MagicMock(),
            wrap_pickle_fields=MagicMock(),
        )

        with patch.object(tokenizer_manager, "wrap_shm_features", lambda obj: obj):
            asyncio.run(manager._send_one_request(tokenized_obj))

        manager._dispatch_to_scheduler.assert_called_once_with(tokenized_obj)
        transport.prepare_for_dispatch_async.assert_awaited_once_with((None,))
        transport.cancel_for_dispatch.assert_not_called()

    def test_failed_dispatch_cancels_published_items(self):
        from sglang.srt.managers import tokenizer_manager
        from sglang.srt.managers.schedule_batch import (
            Modality,
            MultimodalDataItem,
            MultimodalProcessorOutput,
        )

        manager = object.__new__(tokenizer_manager.TokenizerManager)
        transport = MagicMock()
        manager._dispatch_to_scheduler = MagicMock(
            side_effect=RuntimeError("send failed")
        )
        items = [MultimodalDataItem(modality=Modality.IMAGE, feature=torch.arange(2))]
        tokenized_obj = SimpleNamespace(
            rid="test-request",
            mm_inputs=MultimodalProcessorOutput(input_ids=[1], mm_items=items),
            time_stats=MagicMock(),
            wrap_pickle_fields=MagicMock(),
        )
        transport.prepare_for_dispatch_async = AsyncMock(return_value=items)
        manager.cuda_vmm_feature_transport = transport

        with (
            patch.object(tokenizer_manager, "wrap_shm_features", lambda obj: obj),
            self.assertRaisesRegex(RuntimeError, "send failed"),
        ):
            asyncio.run(manager._send_one_request(tokenized_obj))

        transport.prepare_for_dispatch_async.assert_awaited_once_with(
            (tokenized_obj.mm_inputs,)
        )
        transport.cancel_for_dispatch.assert_called_once_with(items)

    def test_post_dispatch_failure_does_not_cancel_published_items(self):
        from sglang.srt.managers import tokenizer_manager
        from sglang.srt.managers.schedule_batch import (
            Modality,
            MultimodalDataItem,
            MultimodalProcessorOutput,
        )

        manager = object.__new__(tokenizer_manager.TokenizerManager)
        transport = MagicMock()
        manager._dispatch_to_scheduler = MagicMock()
        time_stats = MagicMock()
        time_stats.set_api_server_dispatch_finish_time.side_effect = RuntimeError(
            "bookkeeping failed"
        )
        items = [MultimodalDataItem(modality=Modality.IMAGE, feature=torch.arange(2))]
        tokenized_obj = SimpleNamespace(
            rid="test-request",
            mm_inputs=MultimodalProcessorOutput(input_ids=[1], mm_items=items),
            time_stats=time_stats,
            wrap_pickle_fields=MagicMock(),
        )
        transport.prepare_for_dispatch_async = AsyncMock(return_value=items)
        manager.cuda_vmm_feature_transport = transport

        with (
            patch.object(tokenizer_manager, "wrap_shm_features", lambda obj: obj),
            self.assertRaisesRegex(RuntimeError, "bookkeeping failed"),
        ):
            asyncio.run(manager._send_one_request(tokenized_obj))

        manager._dispatch_to_scheduler.assert_called_once_with(tokenized_obj)
        transport.cancel_for_dispatch.assert_not_called()

    def test_async_publication_keeps_event_loop_responsive(self):
        from sglang.srt.utils.cuda_vmm_transport_utils import (
            CudaVmmFeatureTransport,
        )

        transport = object.__new__(CudaVmmFeatureTransport)
        transport.pool = object()
        transport._publisher_executor = concurrent.futures.ThreadPoolExecutor(
            max_workers=1
        )
        started = threading.Event()
        event_loop_responsive = threading.Event()
        release = threading.Event()
        prepared = [object()]

        def block_publication(_):
            started.set()
            release.wait()
            return prepared

        def unblock_if_event_loop_stalls():
            started.wait()
            if not event_loop_responsive.wait(timeout=0.5):
                release.set()

        transport.prepare_for_dispatch = MagicMock(side_effect=block_publication)
        mm_inputs = SimpleNamespace(mm_items=[object()])

        async def run():
            watchdog = threading.Thread(target=unblock_if_event_loop_stalls)
            watchdog.start()
            try:
                task = asyncio.create_task(
                    transport.prepare_for_dispatch_async([mm_inputs])
                )
                while not started.is_set():
                    await asyncio.sleep(0)
                event_loop_responsive.set()
                self.assertFalse(task.done())
                release.set()
                self.assertEqual(await task, prepared)
            finally:
                started.set()
                event_loop_responsive.set()
                release.set()
                watchdog.join()

        try:
            asyncio.run(run())
        finally:
            transport._publisher_executor.shutdown(wait=True)

    def test_text_only_publication_bypasses_publisher_executor(self):
        from sglang.srt.utils.cuda_vmm_transport_utils import (
            CudaVmmFeatureTransport,
        )

        transport = object.__new__(CudaVmmFeatureTransport)
        transport.pool = object()
        transport._publisher_executor = MagicMock()

        result = asyncio.run(
            transport.prepare_for_dispatch_async([None, SimpleNamespace(mm_items=[])])
        )

        self.assertEqual(result, [])
        transport._publisher_executor.submit.assert_not_called()

    def test_pageable_copy_source_is_pinned(self):
        from sglang.srt.utils import cuda_vmm_transport_utils as vmm

        source = torch.arange(4)
        pinned = torch.arange(4)
        with patch.object(
            torch.Tensor, "pin_memory", autospec=True, return_value=pinned
        ) as pin_memory:
            result = vmm._prepare_pinned_copy_source(source)

        self.assertIs(result, pinned)
        pin_memory.assert_called_once_with()

    def test_packed_cpu_sources_use_one_pinned_staging_buffer(self):
        from sglang.srt.utils import cuda_vmm_transport_utils as vmm

        sources = [torch.arange(4, dtype=torch.int32), torch.arange(2)]
        layouts, packed_data_nbytes = vmm._build_packed_tensor_layout(sources)
        staging = torch.empty(packed_data_nbytes, dtype=torch.uint8)

        with patch.object(vmm.torch, "empty", return_value=staging) as empty:
            result = vmm._pack_pinned_copy_sources(sources, layouts, packed_data_nbytes)

        self.assertIs(result, staging)
        empty.assert_called_once_with(
            packed_data_nbytes,
            dtype=torch.uint8,
            device="cpu",
            pin_memory=True,
        )
        for source, layout in zip(sources, layouts, strict=True):
            actual = staging[
                layout.relative_offset : layout.relative_offset + layout.data_nbytes
            ]
            self.assertTrue(torch.equal(actual, source.reshape(-1).view(torch.uint8)))

    def test_recycler_polls_all_chunks_in_one_batch(self):
        from sglang.srt.utils import cuda_vmm_transport_utils as vmm

        pool = object.__new__(vmm.CudaVmmMemoryPool)
        pool.device_index = 0
        pool.consumer_count = 2
        pool._recycle_stream = object()
        pool.memory_pool = MagicMock()
        pool.available_chunks = []
        pool.occupied_chunks = [
            vmm._CudaVmmMemoryChunk(0, 64),
            vmm._CudaVmmMemoryChunk(64, 128),
        ]
        acknowledgement_words = object()
        acknowledgement_counts = MagicMock()
        acknowledgement_counts.cpu.return_value.tolist.return_value = [2, 1]

        with (
            patch.object(vmm.torch.cuda, "device", return_value=nullcontext()),
            patch.object(vmm.torch.cuda, "stream", return_value=nullcontext()),
            patch.object(
                vmm.torch, "stack", return_value=acknowledgement_words
            ) as stack,
            patch.object(
                vmm.torch,
                "count_nonzero",
                return_value=acknowledgement_counts,
            ) as count_nonzero,
        ):
            pool._recycle_chunks()

        stack.assert_called_once()
        count_nonzero.assert_called_once_with(acknowledgement_words, dim=1)
        self.assertEqual(pool.available_chunks, [vmm._CudaVmmMemoryChunk(0, 64)])
        self.assertEqual(pool.occupied_chunks, [vmm._CudaVmmMemoryChunk(64, 128)])

    def test_prepare_batch_cancels_prior_groups_on_failure(self):
        from sglang.srt.utils.cuda_vmm_transport_utils import (
            CudaVmmFeatureTransport,
        )

        transport = object.__new__(CudaVmmFeatureTransport)
        transport.pool = MagicMock()
        transport.wrap_items = MagicMock(
            side_effect=[None, RuntimeError("wrap failed")]
        )
        transport.cancel_for_dispatch = MagicMock()
        item_groups = [[object()], [object()]]
        mm_inputs_batch = [SimpleNamespace(mm_items=items) for items in item_groups]

        with self.assertRaisesRegex(RuntimeError, "wrap failed"):
            transport.prepare_for_dispatch(mm_inputs_batch)

        self.assertEqual(
            transport.wrap_items.call_args_list,
            [call(item_groups[0]), call(item_groups[1])],
        )
        transport.cancel_for_dispatch.assert_called_once_with(item_groups[0])

    def test_prepare_batch_returns_flattened_items(self):
        from sglang.srt.utils.cuda_vmm_transport_utils import (
            CudaVmmFeatureTransport,
        )

        transport = object.__new__(CudaVmmFeatureTransport)
        transport.pool = MagicMock()
        transport.wrap_items = MagicMock()
        item_groups = [[object()], [object(), object()]]

        prepared = transport.prepare_for_dispatch(
            [
                None,
                SimpleNamespace(mm_items=[]),
                *(SimpleNamespace(mm_items=items) for items in item_groups),
            ]
        )

        self.assertEqual(prepared, item_groups[0] + item_groups[1])
        self.assertEqual(
            transport.wrap_items.call_args_list,
            [call(items) for items in item_groups],
        )

    def test_engine_shutdown_is_idempotent(self):
        from sglang.srt.entrypoints import engine as engine_module
        from sglang.srt.entrypoints.engine import Engine
        from sglang.srt.managers.tokenizer_manager import TokenizerManager
        from sglang.srt.utils.cuda_vmm_transport_utils import (
            CudaVmmFeatureTransport,
        )

        manager = object.__new__(TokenizerManager)
        transport = object.__new__(CudaVmmFeatureTransport)
        pool = MagicMock()
        transport.pool = pool
        transport._publisher_executor = None
        manager.cuda_vmm_feature_transport = transport
        manager._subprocess_watchdog = None
        engine = object.__new__(Engine)
        engine.tokenizer_manager = manager

        with patch.object(
            engine_module,
            "kill_process_tree",
            side_effect=RuntimeError("base failed"),
        ):
            for _ in range(2):
                with self.assertRaisesRegex(RuntimeError, "base failed"):
                    engine.shutdown()

        self.assertEqual(pool.shutdown.call_count, 2)
        self.assertIs(transport.pool, pool)

    def test_engine_startup_failure_releases_parent_pool(self):
        from sglang.srt.entrypoints import engine as engine_module
        from sglang.srt.entrypoints.engine import Engine
        from sglang.srt.managers.tokenizer_manager import TokenizerManager
        from sglang.srt.utils.cuda_vmm_transport_utils import (
            CudaVmmFeatureTransport,
        )

        manager = object.__new__(TokenizerManager)
        transport = object.__new__(CudaVmmFeatureTransport)
        pool = MagicMock()
        transport.pool = pool
        transport._publisher_executor = None
        manager.cuda_vmm_feature_transport = transport
        # A real record: the launcher publishes it partway through, and what it
        # reads after that comes out of the bags, which only project from a
        # dataclass. The validation is stubbed so the dummy path still launches.
        from sglang.srt.server_args import ServerArgs

        server_args = ServerArgs(model_path="dummy", tokenizer_worker_num=1)
        server_args.check_server_args = MagicMock()
        from sglang.srt.runtime_context import reset_context

        self.addCleanup(reset_context)
        scheduler_init_result = SimpleNamespace(
            all_child_pids=[],
            scheduler_infos=[],
            wait_for_ready=MagicMock(side_effect=RuntimeError("startup failed")),
            engine_info_bootstrap_server=None,
        )

        with (
            patch.object(engine_module, "configure_logger"),
            patch.object(engine_module, "_set_envs_and_config"),
            patch.object(engine_module, "load_plugins"),
            patch.object(
                Engine,
                "_launch_scheduler_processes",
                return_value=(scheduler_init_result, []),
            ),
            patch.object(
                Engine, "_launch_detokenizer_subprocesses", return_value=([], [])
            ),
            self.assertRaisesRegex(RuntimeError, "startup failed"),
        ):
            Engine._launch_subprocesses(
                server_args=server_args,
                init_tokenizer_manager_func=MagicMock(return_value=(manager, object())),
                run_scheduler_process_func=MagicMock(),
                run_detokenizer_process_func=MagicMock(),
                port_args=SimpleNamespace(),
            )

        pool.shutdown.assert_called_once_with()
        self.assertIs(transport.pool, pool)

    def test_failed_pool_shutdown_remains_retryable(self):
        from sglang.srt.utils.cuda_vmm_transport_utils import (
            CudaVmmFeatureTransport,
        )

        transport = object.__new__(CudaVmmFeatureTransport)
        pool = MagicMock()
        pool.shutdown.side_effect = [RuntimeError("shutdown failed"), None]
        transport.pool = pool
        transport._publisher_executor = None

        with self.assertRaisesRegex(RuntimeError, "shutdown failed"):
            transport.shutdown()
        self.assertIs(transport.pool, pool)

        transport.shutdown()
        self.assertIs(transport.pool, pool)


class TestSchedulerMmTransportBoundary(unittest.TestCase):
    def _publish(self, **fields):
        from sglang.srt.runtime_context import get_context

        override = get_context().override_server_args(**fields)
        override.install()
        self.addCleanup(override.restore)

    @staticmethod
    def _prepare_scheduler(scheduler):
        scheduler.session_controller = SimpleNamespace(maybe_reap=MagicMock())
        scheduler._request_dispatcher = MagicMock(return_value=None)
        scheduler.flush_wrapper = SimpleNamespace(check_pending=MagicMock())
        scheduler.external_corpus_manager = None

    @staticmethod
    def _materialize_with_rank_errors(local_exception=None, remote_error=None):
        from sglang.srt.managers import scheduler as scheduler_module

        class TokenizedRequest:
            def __init__(self):
                self.mm_inputs = object()

        scheduler = object.__new__(scheduler_module.Scheduler)
        scheduler.dp_tp_cpu_group = object()
        request = TokenizedRequest()

        def gather_errors(errors, local_error, **_kwargs):
            errors[:] = [local_error, remote_error]

        materialize = MagicMock(
            side_effect=local_exception,
            return_value=object(),
        )
        with (
            patch.object(
                scheduler_module, "TokenizedGenerateReqInput", TokenizedRequest
            ),
            patch.object(
                scheduler_module, "TokenizedEmbeddingReqInput", TokenizedRequest
            ),
            patch.object(
                scheduler_module.MultimodalInputs,
                "from_processor_output",
                materialize,
            ),
            patch.object(
                scheduler_module.torch.distributed, "is_available", return_value=True
            ),
            patch.object(
                scheduler_module.torch.distributed,
                "is_initialized",
                return_value=True,
            ),
            patch.object(
                scheduler_module.torch.distributed, "get_world_size", return_value=2
            ),
            patch.object(
                scheduler_module.torch.distributed,
                "all_gather_object",
                side_effect=gather_errors,
            ),
        ):
            errors = scheduler._materialize_cuda_vmm_inputs(request)

        return request, errors

    def test_materializes_inputs_directly_before_base_dispatch(self):
        from sglang.srt.managers import scheduler as scheduler_module

        scheduler = object.__new__(scheduler_module.Scheduler)
        # The transport gate reads the published bags, so the case publishes
        # the configuration under test.
        self._publish(
            mm_feature_transport="cuda_vmm",
            enable_broadcast_mm_inputs_process=True,
        )
        self._prepare_scheduler(scheduler)
        raw_inputs = object()
        materialized = object()
        request = SimpleNamespace(mm_inputs=raw_inputs)

        with (
            patch.object(
                scheduler_module, "TokenizedGenerateReqInput", SimpleNamespace
            ),
            patch.object(
                scheduler_module.MultimodalInputs,
                "from_processor_output",
                return_value=materialized,
            ) as build_inputs,
            patch.object(
                scheduler, "_process_and_broadcast_mm_inputs"
            ) as cpu_broadcast,
            patch.object(
                scheduler_module, "is_health_check_generate_req", return_value=False
            ),
        ):
            scheduler.process_input_requests([request])

        build_inputs.assert_called_once_with(raw_inputs)
        self.assertIs(request.mm_inputs, materialized)
        scheduler._request_dispatcher.assert_called_once_with(request)
        cpu_broadcast.assert_not_called()

    def test_materializes_batched_inputs_before_dispatch(self):
        from sglang.srt.managers import scheduler as scheduler_module

        class TokenizedRequest:
            def __init__(self, mm_inputs):
                self.mm_inputs = mm_inputs

        class BatchRequest:
            def __init__(self, batch):
                self.batch = batch

            def __iter__(self):
                return iter(self.batch)

        scheduler = object.__new__(scheduler_module.Scheduler)
        self._publish(mm_feature_transport="cuda_vmm")
        self._prepare_scheduler(scheduler)
        raw_inputs = [object(), object()]
        materialized = [object(), object()]
        inner_requests = [TokenizedRequest(value) for value in raw_inputs]
        request = BatchRequest(inner_requests)

        with (
            patch.object(
                scheduler_module, "TokenizedGenerateReqInput", TokenizedRequest
            ),
            patch.object(
                scheduler_module, "TokenizedEmbeddingReqInput", TokenizedRequest
            ),
            patch.object(
                scheduler_module, "BatchTokenizedGenerateReqInput", BatchRequest
            ),
            patch.object(
                scheduler_module, "BatchTokenizedEmbeddingReqInput", BatchRequest
            ),
            patch.object(
                scheduler_module.MultimodalInputs,
                "from_processor_output",
                side_effect=materialized,
            ) as build_inputs,
            patch.object(
                scheduler_module, "is_health_check_generate_req", return_value=False
            ),
        ):
            scheduler.process_input_requests([request])

        self.assertEqual(
            build_inputs.call_args_list,
            [call(value) for value in raw_inputs],
        )
        self.assertEqual(
            [inner.mm_inputs for inner in inner_requests],
            materialized,
        )
        scheduler._request_dispatcher.assert_called_once_with(request)

    def test_already_materialized_inputs_are_reused(self):
        from sglang.srt.managers.schedule_batch import MultimodalInputs
        from sglang.srt.managers.scheduler import Scheduler

        scheduler = object.__new__(Scheduler)
        mm_inputs = MultimodalInputs(mm_items=[])

        with patch.object(
            scheduler, "_process_and_broadcast_mm_inputs"
        ) as process_and_broadcast:
            self.assertIs(scheduler._get_multimodal_inputs(mm_inputs), mm_inputs)

        process_and_broadcast.assert_not_called()

    def test_broadcast_mm_inputs_sends_entry_rank_processing_error(self):
        from sglang.srt.managers import scheduler as scheduler_module

        scheduler = object.__new__(scheduler_module.Scheduler)
        scheduler.dp_tp_group = SimpleNamespace(rank_in_group=0, first_rank=0)
        scheduler.dp_tp_cpu_group = object()

        with (
            patch.object(
                scheduler_module.MultimodalInputs,
                "from_processor_output",
                side_effect=ValueError("bad image"),
            ),
            patch.object(
                scheduler_module.torch.distributed, "is_available", return_value=True
            ),
            patch.object(
                scheduler_module.torch.distributed,
                "is_initialized",
                return_value=True,
            ),
            patch.object(
                scheduler_module.torch.distributed, "get_world_size", return_value=2
            ),
            patch.object(
                scheduler_module.torch.distributed, "broadcast_object_list"
            ) as broadcast,
            self.assertRaisesRegex(
                scheduler_module._MultimodalInputProcessingError,
                "ValueError: bad image",
            ),
        ):
            scheduler._process_and_broadcast_mm_inputs(object())

        payload = broadcast.call_args.args[0][0]
        self.assertIn("ValueError: bad image", payload.error)

    def test_broadcast_mm_inputs_peer_rank_receives_processing_error(self):
        from sglang.srt.managers import scheduler as scheduler_module

        scheduler = object.__new__(scheduler_module.Scheduler)
        scheduler.dp_tp_group = SimpleNamespace(rank_in_group=1, first_rank=0)
        scheduler.dp_tp_cpu_group = object()

        def receive_error(obj_list, **_kwargs):
            obj_list[0] = scheduler_module._MultimodalInputBroadcast(error="bad image")

        with (
            patch.object(
                scheduler_module.MultimodalInputs, "from_processor_output"
            ) as materialize,
            patch.object(
                scheduler_module.torch.distributed, "is_available", return_value=True
            ),
            patch.object(
                scheduler_module.torch.distributed,
                "is_initialized",
                return_value=True,
            ),
            patch.object(
                scheduler_module.torch.distributed, "get_world_size", return_value=2
            ),
            patch.object(
                scheduler_module.torch.distributed,
                "broadcast_object_list",
                side_effect=receive_error,
            ),
            self.assertRaisesRegex(
                scheduler_module._MultimodalInputProcessingError, "bad image"
            ),
        ):
            scheduler._process_and_broadcast_mm_inputs(object())

        materialize.assert_not_called()

    def test_embedding_request_aborts_broadcast_processing_error(self):
        from sglang.srt.managers import scheduler as scheduler_module

        scheduler = object.__new__(scheduler_module.Scheduler)
        scheduler.tokenizer = object()
        scheduler._maybe_namespace_elastic_radix_cache = MagicMock()
        scheduler._add_request_to_queue = MagicMock()
        scheduler._get_multimodal_inputs = MagicMock(
            side_effect=scheduler_module._MultimodalInputProcessingError("bad image")
        )
        req = MagicMock()
        recv_req = SimpleNamespace(
            rid="request-id",
            input_text="prompt",
            input_ids=[1],
            sampling_params=object(),
            positional_embed_overrides=None,
            token_type_ids=None,
            routed_dp_rank=None,
            priority=None,
            dimensions=None,
            lora_id=None,
            http_worker_ipc=None,
            time_stats=None,
            return_pooled_hidden_states=False,
            multi_item_delimiter_indices=None,
            mm_inputs=object(),
        )

        with patch.object(scheduler_module, "Req", return_value=req):
            scheduler.handle_embedding_request(recv_req)

        req.set_finish_with_abort.assert_called_once_with(
            "bad image",
            status_code=500,
            err_type="InternalServerError",
        )
        scheduler._add_request_to_queue.assert_called_once_with(req)

    def test_vmm_materialization_consensus_rejects_any_rank_failure(self):
        cases = (
            (None, "RuntimeError: remote failure", "rank 1: RuntimeError"),
            (ValueError("bad proxy"), None, "rank 0: ValueError: bad proxy"),
        )
        for local_exception, remote_error, expected in cases:
            with self.subTest(expected=expected):
                request, errors = self._materialize_with_rank_errors(
                    local_exception, remote_error
                )
                self.assertIn(expected, errors[0])
                self.assertIsNone(request.mm_inputs)

    def test_vmm_batch_dispatches_good_and_failed_requests_individually(self):
        from sglang.srt.managers import scheduler as scheduler_module

        class TokenizedRequest:
            pass

        class EmbeddingRequest:
            pass

        class BatchRequest:
            def __init__(self, requests):
                self.requests = requests

            def __iter__(self):
                return iter(self.requests)

        scheduler = object.__new__(scheduler_module.Scheduler)
        self._publish(mm_feature_transport="cuda_vmm")
        self._prepare_scheduler(scheduler)
        scheduler.is_fully_idle = MagicMock(return_value=True)
        scheduler.return_health_check_ipcs = []
        scheduler.handle_generate_request = MagicMock()
        scheduler.handle_embedding_request = MagicMock()
        scheduler._materialize_cuda_vmm_inputs = MagicMock(
            return_value=[None, "reconstruction failed"]
        )
        requests = [TokenizedRequest(), TokenizedRequest()]
        batch = BatchRequest(requests)

        with (
            patch.object(
                scheduler_module, "TokenizedGenerateReqInput", TokenizedRequest
            ),
            patch.object(
                scheduler_module, "TokenizedEmbeddingReqInput", EmbeddingRequest
            ),
            patch.object(
                scheduler_module, "BatchTokenizedGenerateReqInput", BatchRequest
            ),
            patch.object(scheduler_module, "BatchTokenizedEmbeddingReqInput", tuple),
            patch.object(
                scheduler_module, "is_health_check_generate_req", return_value=False
            ),
        ):
            scheduler.process_input_requests([batch])

        self.assertEqual(
            scheduler.handle_generate_request.call_args_list,
            [
                call(requests[0], mm_input_error=None),
                call(requests[1], mm_input_error="reconstruction failed"),
            ],
        )
        scheduler.handle_embedding_request.assert_not_called()
        scheduler._request_dispatcher.assert_not_called()

    def test_vmm_materialization_abort_reports_internal_error(self):
        from sglang.srt.managers import schedule_batch

        req = object.__new__(schedule_batch.Req)
        req.rid = "request-id"
        req.multimodal_inputs = schedule_batch.MultimodalInputs(mm_items=[])
        req.session = None
        req.grammar = object()
        req.origin_input_ids = [1, 2]
        req.return_logprob = True
        req.logprob_start_len = 0
        req.to_finish = None

        with patch.object(
            schedule_batch, "get_parallel", return_value=SimpleNamespace(tp_rank=1)
        ):
            req.set_finish_with_abort(
                "reconstruction failed",
                status_code=500,
                err_type="InternalServerError",
            )

        self.assertEqual(
            req.to_finish.to_json(),
            {
                "type": "abort",
                "message": "reconstruction failed",
                "status_code": 500,
                "err_type": "InternalServerError",
            },
        )
        self.assertIsNone(req.multimodal_inputs)


class TestVmmConsumerCount(unittest.TestCase):
    def test_proxy_defaults_to_one_consumer(self):
        from sglang.srt.utils import cuda_vmm_transport_utils as vmm

        proxy = object.__new__(vmm.CudaVmmTensorTransportProxy)
        proxy.consumer_count = 4
        self.assertEqual(proxy._resolve_consumer_count(None), 1)
        self.assertEqual(proxy._resolve_consumer_count(2), 2)

    def test_acknowledgement_ranges_include_cp_rank(self):
        from sglang.srt.runtime_context import get_parallel
        from sglang.srt.utils.cuda_vmm_transport_utils import (
            CudaVmmTensorTransportProxy,
        )

        proxy = object.__new__(CudaVmmTensorTransportProxy)
        proxy.consumer_count = 4
        with get_parallel().override(
            attn_tp_size=2,
            attn_tp_rank=1,
            attn_cp_size=2,
            attn_cp_rank=1,
        ):
            self.assertEqual(proxy._acknowledgement_range(1), (3, 4))
            self.assertEqual(proxy._acknowledgement_range(2), (2, 4))
            self.assertEqual(proxy._acknowledgement_range(4), (0, 4))


if __name__ == "__main__":
    unittest.main()
