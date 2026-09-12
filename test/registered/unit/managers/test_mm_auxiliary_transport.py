"""CPU-only coverage for auxiliary multimodal feature transport."""

import asyncio
import copy
import pickle
import unittest
from multiprocessing import shared_memory
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, Mock, call, patch

import torch

from sglang.test.test_utils import CustomTestCase, maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()

from sglang.srt.managers import mm_utils  # noqa: E402
from sglang.srt.managers.io_struct import (  # noqa: E402
    BatchTokenizedGenerateReqInput,
    TokenizedGenerateReqInput,
)
from sglang.srt.managers.schedule_batch import (  # noqa: E402
    Modality,
    MultimodalDataItem,
    MultimodalInputs,
)
from sglang.srt.managers.tokenizer_manager import TokenizerManager  # noqa: E402
from sglang.srt.multimodal.processors.base_processor import (  # noqa: E402
    BaseMultimodalProcessor,
)
from sglang.srt.sampling.sampling_params import SamplingParams  # noqa: E402
from sglang.test.ci.ci_register import register_cpu_ci  # noqa: E402

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


def _tokenized_request(item, rid="request"):
    return TokenizedGenerateReqInput(
        rid=rid,
        input_text="",
        input_ids=[1],
        input_embeds=None,
        mm_inputs=MultimodalInputs(mm_items=[item]),
        token_type_ids=None,
        sampling_params=SamplingParams(),
        return_logprob=False,
        logprob_start_len=0,
        top_logprobs_num=0,
        token_ids_logprob=None,
        stream=False,
        time_stats=MagicMock(),
    )


class _TestMultimodalProcessor(BaseMultimodalProcessor):
    async def process_mm_data_async(
        self,
        image_data,
        audio_data,
        input_text,
        request_obj,
        **kwargs,
    ):
        raise NotImplementedError


class TestAuxiliaryFeatureSharedMemory(CustomTestCase):
    def setUp(self):
        self.transport_patches = (
            patch.object(mm_utils, "_get_is_default_transport", return_value=False),
            patch.object(
                mm_utils,
                "get_serving",
                return_value=SimpleNamespace(skip_tokenizer_init=False),
            ),
        )
        for transport_patch in self.transport_patches:
            transport_patch.start()
            self.addCleanup(transport_patch.stop)

    def test_round_trip_preserves_nested_repeated_auxiliary_feature(self):
        tensor = torch.arange(256 * 1024, dtype=torch.float32)
        metadata = object()
        original_model_specific_data = {
            "patch_pixel_values": tensor,
            "metadata": metadata,
        }
        item = MultimodalDataItem(
            modality=Modality.IMAGE,
            hash=123,
            model_specific_data=original_model_specific_data,
        )
        request = _tokenized_request(item)
        shm_name = None

        try:
            mm_utils.wrap_shm_features(request)

            producer_proxy = item.model_specific_data["patch_pixel_values"]
            self.assertIsInstance(producer_proxy, mm_utils.ShmPointerMMData)
            self.assertIsNot(item.model_specific_data, original_model_specific_data)
            self.assertIs(original_model_specific_data["patch_pixel_values"], tensor)
            self.assertIs(item.model_specific_data["metadata"], metadata)
            shm_name = producer_proxy.shm_name

            # Pickle/unpickle models the tokenizer-to-scheduler process boundary.
            consumer_proxy = pickle.loads(pickle.dumps(producer_proxy))
            item.model_specific_data["patch_pixel_values"] = {
                "first": consumer_proxy,
                "again": [consumer_proxy],
            }
            self.assertTrue(mm_utils.has_shm_features([request]))

            with patch.object(
                consumer_proxy,
                "materialize",
                wraps=consumer_proxy.materialize,
            ) as materialize:
                mm_utils.unwrap_shm_features(request)

            restored = item.model_specific_data["patch_pixel_values"]
            materialize.assert_called_once_with()
            self.assertIs(restored["first"], restored["again"][0])
            self.assertTrue(torch.equal(restored["first"], tensor))
            self.assertFalse(mm_utils.has_shm_features([request]))

            # A second scheduler-side scan is harmless after materialization.
            self.assertIs(mm_utils.unwrap_shm_features(request), request)
        finally:
            if shm_name is not None:
                try:
                    shm = shared_memory.SharedMemory(name=shm_name)
                except FileNotFoundError:
                    pass
                else:
                    shm.close()
                    shm.unlink()

    def test_zero_sized_and_allocation_failure_fall_back_inline(self):
        empty = torch.empty(0)
        with patch.object(mm_utils, "ShmPointerMMData") as shm_pointer:
            self.assertIs(mm_utils._wrap_shm_or_inline(empty), empty)
        shm_pointer.assert_not_called()

        tensor = torch.ones(8)
        with (
            patch.object(
                mm_utils,
                "ShmPointerMMData",
                side_effect=OSError("shared memory full"),
            ),
            patch.object(mm_utils, "print_warning_once") as warning,
        ):
            self.assertIs(mm_utils._wrap_shm_or_inline(tensor), tensor)
        warning.assert_called_once()


class TestAuxiliaryFeatureProducerTransport(CustomTestCase):
    def test_processor_wraps_only_registered_auxiliary_tensor_copy_on_write(self):
        feature = torch.tensor([[1.0]])
        patch_pixels = torch.tensor([[2.0]])
        unrelated_tensor = torch.tensor([3.0])
        feature_proxy = object()
        patch_proxy = object()
        original_model_specific_data = {
            "patch_pixel_values": patch_pixels,
            "unrelated_tensor": unrelated_tensor,
        }
        source_item = MultimodalDataItem(
            modality=Modality.IMAGE,
            hash=11,
            pad_value=12,
            offsets=[(0, 0)],
            feature=feature,
            model_specific_data=original_model_specific_data,
        )
        processor = _TestMultimodalProcessor.__new__(_TestMultimodalProcessor)
        processor.use_cuda_ipc = True
        processor._wrap_tensor_for_cuda_ipc = Mock(
            side_effect=[feature_proxy, patch_proxy]
        )
        items = processor._prepare_mm_items_for_transport([source_item])

        self.assertEqual(len(items), 1)
        item = items[0]
        self.assertIs(item.feature, feature_proxy)
        self.assertIsNot(item.model_specific_data, original_model_specific_data)
        self.assertIs(item.model_specific_data["patch_pixel_values"], patch_proxy)
        self.assertIs(item.model_specific_data["unrelated_tensor"], unrelated_tensor)
        processor._wrap_tensor_for_cuda_ipc.assert_has_calls(
            [call(feature), call(patch_pixels)]
        )

        # The shallow-copied source dictionary remains unchanged.
        self.assertIs(original_model_specific_data["patch_pixel_values"], patch_pixels)


class TestBatchedTokenizerAuxiliaryTransport(CustomTestCase):
    def test_each_subrequest_is_wrapped_before_pickle_and_dispatch(self):
        original_requests = [
            _tokenized_request(
                MultimodalDataItem(modality=Modality.IMAGE), rid=f"request-{index}"
            )
            for index in range(2)
        ]
        wrapped_requests = [
            _tokenized_request(
                MultimodalDataItem(modality=Modality.IMAGE),
                rid=f"wrapped-request-{index}",
            )
            for index in range(2)
        ]
        requests = list(original_requests)
        replacement_by_rid = {
            original.rid: wrapped
            for original, wrapped in zip(original_requests, wrapped_requests)
        }
        manager = TokenizerManager.__new__(TokenizerManager)
        manager._dispatch_to_scheduler = Mock()
        manager._mark_state_dispatched = Mock()
        manager.cuda_vmm_feature_transport = SimpleNamespace(
            prepare_for_dispatch_async=AsyncMock(return_value=[]),
            cancel_for_dispatch=Mock(),
        )
        events = []

        def record_wrap(request):
            events.append(("wrap", request.rid))
            return replacement_by_rid[request.rid]

        def record_pickle(request):
            events.append(("pickle", request.rid))

        with (
            patch(
                "sglang.srt.managers.tokenizer_manager.wrap_shm_features",
                side_effect=record_wrap,
            ) as wrap,
            patch.object(
                TokenizedGenerateReqInput,
                "wrap_pickle_fields",
                autospec=True,
                side_effect=record_pickle,
            ),
        ):
            asyncio.run(manager._send_batch_request(requests))

        self.assertEqual(
            events,
            [
                ("wrap", "request-0"),
                ("pickle", "wrapped-request-0"),
                ("wrap", "request-1"),
                ("pickle", "wrapped-request-1"),
            ],
        )
        self.assertEqual(
            wrap.call_args_list,
            [call(original_requests[0]), call(original_requests[1])],
        )
        batch = manager._dispatch_to_scheduler.call_args.args[0]
        self.assertEqual(list(batch), wrapped_requests)
        self.assertEqual(requests, wrapped_requests)


class TestAuxiliaryLifecycle(CustomTestCase):
    def setUp(self):
        for context in (
            patch.object(mm_utils, "_get_is_default_transport", return_value=False),
            patch.object(
                mm_utils,
                "get_serving",
                return_value=SimpleNamespace(skip_tokenizer_init=False),
            ),
        ):
            context.start()
            self.addCleanup(context.stop)

    def make_request(self):
        request = _tokenized_request(
            MultimodalDataItem(
                modality=Modality.IMAGE,
                model_specific_data={"patch_pixel_values": torch.ones(256 * 1024)},
            )
        )
        self.addCleanup(mm_utils.discard_shm_features, request)
        return request

    def assert_segment_removed(self, name):
        with self.assertRaises(FileNotFoundError):
            shared_memory.SharedMemory(name=name)

    def test_discard_releases_nested_auxiliary_segments(self):
        request = self.make_request()
        mm_utils.wrap_shm_features(request)
        item = request.mm_inputs.mm_items[0]
        proxy = item.model_specific_data["patch_pixel_values"]
        item.model_specific_data["patch_pixel_values"] = {
            "first": proxy,
            "again": (proxy,),
        }
        batch = BatchTokenizedGenerateReqInput(batch=[request])
        mm_utils.discard_shm_features(batch)
        mm_utils.discard_shm_features(batch)
        self.assert_segment_removed(proxy.shm_name)

    def test_repeated_proxy_across_requests_materializes_once(self):
        request = self.make_request()
        mm_utils.wrap_shm_features(request)
        producer = request.mm_inputs.mm_items[0].model_specific_data[
            "patch_pixel_values"
        ]
        consumer = pickle.loads(pickle.dumps(producer))
        first = request.mm_inputs.mm_items[0]
        first.model_specific_data = {"patch_pixel_values": consumer}
        second = copy.copy(first)
        second.model_specific_data = dict(first.model_specific_data)
        other = _tokenized_request(second)
        with patch.object(
            consumer, "materialize", wraps=consumer.materialize
        ) as materialize:
            mm_utils.unwrap_shm_features(
                BatchTokenizedGenerateReqInput(batch=[request, other])
            )
        materialize.assert_called_once()
        self.assertIs(
            first.model_specific_data["patch_pixel_values"],
            second.model_specific_data["patch_pixel_values"],
        )
        self.assert_segment_removed(producer.shm_name)

    def test_nested_wrap_failure_releases_already_created_segment(self):
        proxy = mm_utils.ShmPointerMMData(torch.ones(256 * 1024))
        self.addCleanup(proxy.close_and_unlink)
        with patch.object(
            mm_utils,
            "_wrap_shm_or_inline",
            side_effect=[proxy, RuntimeError("copy failed")],
        ):
            with self.assertRaisesRegex(RuntimeError, "copy failed"):
                mm_utils._wrap_auxiliary_transport_value(
                    [torch.ones(256 * 1024), torch.ones(256 * 1024)]
                )
        self.assert_segment_removed(proxy.shm_name)

    def test_dispatch_failure_releases_single_and_batch_shm(self):
        for batched in (False, True):
            with self.subTest(batched=batched):
                requests = [self.make_request() for _ in range(2 if batched else 1)]
                names = []

                def dispatch(_request):
                    names.extend(
                        r.mm_inputs.mm_items[0]
                        .model_specific_data["patch_pixel_values"]
                        .shm_name
                        for r in requests
                    )
                    raise RuntimeError("send failed")

                manager = TokenizerManager.__new__(TokenizerManager)
                manager.cuda_vmm_feature_transport = SimpleNamespace(
                    prepare_for_dispatch_async=AsyncMock(return_value=[]),
                    cancel_for_dispatch=Mock(),
                )
                manager._dispatch_to_scheduler = dispatch
                with self.assertRaisesRegex(RuntimeError, "send failed"):
                    asyncio.run(
                        manager._send_batch_request(requests)
                        if batched
                        else manager._send_one_request(requests[0])
                    )
                self.assertEqual(len(names), len(requests))
                for name in names:
                    self.assert_segment_removed(name)
                manager.cuda_vmm_feature_transport.cancel_for_dispatch.assert_called_once_with(
                    []
                )

    def test_post_dispatch_failure_keeps_scheduler_owned_shm(self):
        for batched in (False, True):
            with self.subTest(batched=batched):
                request = self.make_request()
                manager = TokenizerManager.__new__(TokenizerManager)
                manager.cuda_vmm_feature_transport = SimpleNamespace(
                    prepare_for_dispatch_async=AsyncMock(return_value=[]),
                    cancel_for_dispatch=Mock(),
                )
                manager._dispatch_to_scheduler = Mock()
                manager._mark_state_dispatched = Mock(
                    side_effect=RuntimeError("state failed")
                )
                with self.assertRaisesRegex(RuntimeError, "state failed"):
                    asyncio.run(
                        manager._send_batch_request([request])
                        if batched
                        else manager._send_one_request(request)
                    )
                manager._dispatch_to_scheduler.assert_called_once()
                manager.cuda_vmm_feature_transport.cancel_for_dispatch.assert_not_called()
                proxy = request.mm_inputs.mm_items[0].model_specific_data[
                    "patch_pixel_values"
                ]
                segment = shared_memory.SharedMemory(name=proxy.shm_name)
                segment.close()

    def test_offload_preserves_parallel_sampling_metadata(self):
        gpu = Mock(spec=torch.Tensor)
        gpu.is_cuda = True
        cpu = torch.ones(4)
        gpu.to.return_value = cpu
        metadata = {"patch_pixel_values": gpu, "unrelated": gpu}
        item = MultimodalDataItem(modality=Modality.IMAGE, model_specific_data=metadata)
        sibling = copy.copy(item)
        mm_utils._offload_auxiliary_features(item)
        self.assertIs(item.model_specific_data["patch_pixel_values"], cpu)
        self.assertIs(item.model_specific_data["unrelated"], gpu)
        self.assertIs(sibling.model_specific_data, metadata)
        self.assertIs(metadata["patch_pixel_values"], gpu)
        gpu.to.assert_called_once_with("cpu", non_blocking=True)

    def test_auxiliary_reconstruction_is_copy_on_write(self):
        proxy = Mock(spec=mm_utils.CudaIpcTensorTransportProxy)
        tensor = torch.ones(4)
        proxy.reconstruct_on_target_device.return_value = tensor
        metadata = {"patch_pixel_values": proxy}
        item = MultimodalDataItem(modality=Modality.IMAGE, model_specific_data=metadata)
        sibling = copy.copy(item)
        item.reconstruct(0)
        self.assertIs(item.model_specific_data["patch_pixel_values"], tensor)
        self.assertIs(sibling.model_specific_data["patch_pixel_values"], proxy)
        proxy.reconstruct_on_target_device.assert_called_once_with(0)

    def test_cuda_wrap_failure_rolls_back_auxiliary_and_primary_features(self):
        for fail_cancel in (False, True):
            with self.subTest(fail_cancel=fail_cancel):
                feature, auxiliary, later = torch.ones(4), torch.ones(4), torch.ones(4)
                metadata = {"patch_pixel_values": auxiliary}
                items = [
                    MultimodalDataItem(
                        modality=Modality.IMAGE,
                        feature=feature,
                        model_specific_data=metadata,
                    ),
                    MultimodalDataItem(modality=Modality.IMAGE, feature=later),
                ]
                proxies = [
                    Mock(spec=mm_utils.CudaIpcTensorTransportProxy) for _ in range(2)
                ]
                processor = _TestMultimodalProcessor.__new__(_TestMultimodalProcessor)
                processor.use_cuda_ipc = True
                processor._wrap_tensor_for_cuda_ipc = Mock(
                    side_effect=[*proxies, RuntimeError("wrap failed")]
                )
                processor.cudaipc_mmfeature_pool = SimpleNamespace(
                    cancel_proxy=Mock(
                        side_effect=[RuntimeError("cancel failed"), None]
                        if fail_cancel
                        else None
                    )
                )
                with self.assertRaisesRegex(RuntimeError, "wrap failed"):
                    processor._prepare_mm_items_for_transport(items)
                self.assertEqual(
                    processor.cudaipc_mmfeature_pool.cancel_proxy.call_args_list,
                    [call(proxies[1]), call(proxies[0])],
                )
                self.assertIs(items[0].feature, feature)
                self.assertIs(items[0].model_specific_data, metadata)
                self.assertIs(metadata["patch_pixel_values"], auxiliary)
                self.assertIs(items[1].feature, later)

    def test_empty_auxiliary_tensor_does_not_allocate_ipc_pool_slice(self):
        item = MultimodalDataItem(
            modality=Modality.IMAGE,
            model_specific_data={"patch_pixel_values": torch.empty(0)},
        )
        processor = _TestMultimodalProcessor.__new__(_TestMultimodalProcessor)
        processor.use_cuda_ipc = True
        processor._wrap_tensor_for_cuda_ipc = Mock()
        processor._prepare_mm_items_for_transport([item])
        processor._wrap_tensor_for_cuda_ipc.assert_not_called()
        self.assertEqual(item.model_specific_data["patch_pixel_values"].numel(), 0)


if __name__ == "__main__":
    unittest.main()
