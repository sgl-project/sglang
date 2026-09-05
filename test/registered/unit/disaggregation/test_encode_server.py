import asyncio
import pickle
import threading
import unittest
from http import HTTPStatus
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock, patch

import numpy as np
import torch

import sglang.srt.disaggregation.encoder.server as encoder_server
from sglang.srt.disaggregation.encoder import http_server
from sglang.srt.disaggregation.encoder import runtime as encoder_runtime
from sglang.srt.disaggregation.encoder.preprocessor import EncoderPreprocessor
from sglang.srt.disaggregation.encoder.receiver import EmbeddingData
from sglang.srt.disaggregation.encoder.runtime import (
    _DP_RELEASE_AFTER_ENCODE,
    DPDispatcher,
    _retire_abandoned_encode,
    execute_encode_pipeline,
    send_staged_embedding,
)
from sglang.srt.disaggregation.encoder.server import (
    BadRequestError,
    EncodeContext,
    EncoderDelivery,
    EncoderMetaRegistry,
    InternalError,
    MMEncoder,
    MMError,
    MooncakeDelivery,
    ReqState,
    SendDestination,
    ZmqDelivery,
    _await_transfer_completion,
    meta_registry,
    rid_to_cond,
    rid_to_receive_count,
    rid_to_receive_endpoint,
)
from sglang.srt.distributed.device_communicators.mooncake_transfer_engine import (
    MooncakeTransferEngine,
)
from sglang.srt.managers.io_struct import unwrap_from_pickle
from sglang.srt.managers.schedule_batch import Modality
from sglang.srt.mem_cache.multimodal_cache import (
    EmbeddingResult,
    MultiModalStaticCache,
)
from sglang.srt.utils.common import safe_pickle_loads
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


class TestEncoderDPErrorHandling(CustomTestCase):
    @staticmethod
    async def _run_registration_error(error):
        encoder = SimpleNamespace(
            register_embedding_destinations=AsyncMock(side_effect=error)
        )
        send = AsyncMock()
        request = {
            "req_id": "req",
            "receive_count": 1,
            "receive_url": "tcp://127.0.0.1:1",
        }
        with patch.object(encoder_runtime, "async_sock_send", send):
            await encoder_runtime._dp_worker_handle_request(
                encoder,
                None,
                object(),
                asyncio.Lock(),
                0,
                request,
                "register_destinations",
            )
        return unwrap_from_pickle(send.await_args.args[1])

    def test_worker_reports_third_party_exception_with_callable_code(self):
        class RpcLikeError(Exception):
            def code(self):
                return "INTERNAL"

        envelope = asyncio.run(
            self._run_registration_error(RpcLikeError("registration failed"))
        )
        self.assertEqual(envelope["_error"], "registration failed")
        self.assertEqual(envelope["_error_code"], 500)

    def test_worker_preserves_mm_error_status(self):
        envelope = asyncio.run(
            self._run_registration_error(MMError("bad destination", code=400))
        )
        self.assertEqual(envelope["_error_code"], 400)

    def test_dispatcher_drops_malformed_result_without_stopping_listener(self):
        async def run():
            dispatcher = encoder_runtime.DPDispatcher(
                dp_size=1,
                dispatch_sockets=[object()],
                release_sockets=[object()],
                result_socket=object(),
                worker_processes=[],
            )
            future = asyncio.get_running_loop().create_future()
            dispatcher.pending_futures[0]["req"] = future
            dispatcher.req_id_to_rank["req"] = 0
            valid = {"req_id": "req", "_dp_type": "encode", "content": None}
            recv = AsyncMock(
                side_effect=[
                    ["not", "an", "envelope"],
                    valid,
                    asyncio.CancelledError(),
                ]
            )

            with patch.object(encoder_runtime, "async_sock_recv", recv):
                listener = asyncio.create_task(dispatcher._result_listener())
                await asyncio.wait_for(future, timeout=1)
                listener.cancel()
                with self.assertRaises(asyncio.CancelledError):
                    await listener

            self.assertEqual(future.result(), valid)

        asyncio.run(run())


class TestEncoderMetaRegistry(CustomTestCase):
    def test_stale_releases_do_not_block_each_other(self):
        async def run():
            registry = EncoderMetaRegistry(wait_timeout=1, sweep_timeout=1)
            blocked_started = asyncio.Event()
            unblock = asyncio.Event()
            fast_released = asyncio.Event()

            async def release(req_id):
                if req_id == "blocked":
                    blocked_started.set()
                    await unblock.wait()
                else:
                    fast_released.set()

            registry.on_release = release
            registry._pending_at.update(blocked=0, fast=0)

            blocked_task = registry._schedule_stale_release("blocked")
            await asyncio.wait_for(blocked_started.wait(), timeout=1)
            fast_task = registry._schedule_stale_release("fast")
            await asyncio.wait_for(fast_released.wait(), timeout=1)
            await fast_task

            self.assertIn("blocked", registry._pending_at)
            self.assertNotIn("fast", registry._pending_at)

            unblock.set()
            await blocked_task
            self.assertNotIn("blocked", registry._pending_at)

        asyncio.run(run())

    def test_failed_stale_release_is_retried(self):
        async def run():
            registry = EncoderMetaRegistry(wait_timeout=1, sweep_timeout=1)
            attempts = 0

            async def release(_req_id):
                nonlocal attempts
                attempts += 1
                if attempts == 1:
                    raise RuntimeError("transient cleanup failure")

            registry.on_release = release
            registry._pending_at["req"] = 0

            await registry._release_stale("req")
            self.assertIn("req", registry._pending_at)
            retry_at = registry._pending_at["req"]
            self.assertGreater(retry_at, 0)

            await registry._release_stale("req")
            self.assertNotIn("req", registry._pending_at)
            self.assertEqual(attempts, 2)

        asyncio.run(run())

    def test_send_retries_do_not_release_before_all_destinations_finish(self):
        async def run():
            registry = EncoderMetaRegistry(wait_timeout=1, sweep_timeout=1)
            released = AsyncMock()
            registry.on_release = released

            await registry.note_send_done("req", 2, "10.0.0.1:5000")
            await registry.note_send_done("req", 2, "10.0.0.1:5000")
            released.assert_not_awaited()

            await registry.note_send_done("req", 2, "10.0.0.2:5000")
            released.assert_awaited_once_with("req")

        asyncio.run(run())

    def test_http_send_counts_the_normalized_destination(self):
        async def run():
            send = AsyncMock(return_value=True)
            note_send_done = AsyncMock()
            request = {
                "req_id": "req",
                "prefill_host": "127.0.0.1",
                "embedding_port": 5000,
                "session_id": "session",
                "buffer_address": 1234,
                "receive_count": 2,
            }
            with (
                patch.object(http_server, "dp_dispatcher", None),
                patch.object(http_server, "encoder", SimpleNamespace(send=send)),
                patch.object(
                    encoder_server.meta_registry,
                    "note_send_done",
                    note_send_done,
                ),
            ):
                response = await http_server.handle_send_request(request)

            self.assertEqual(response.status_code, 200)
            note_send_done.assert_awaited_once_with("req", 2, "127.0.0.1:5000")

        asyncio.run(run())

    def test_dp_send_counts_the_normalized_destination(self):
        async def run():
            encoder = SimpleNamespace(send=AsyncMock(return_value=True))
            note_send_done = AsyncMock()
            request = {
                "req_id": "req",
                "prefill_host": "127.0.0.1",
                "embedding_port": 5000,
                "session_id": "session",
                "buffer_address": 1234,
                "receive_count": 2,
            }
            with (
                patch.object(encoder_runtime, "async_sock_send", AsyncMock()),
                patch.object(
                    encoder_server.meta_registry,
                    "note_send_done",
                    note_send_done,
                ),
            ):
                await encoder_runtime._dp_worker_handle_request(
                    encoder,
                    None,
                    object(),
                    asyncio.Lock(),
                    0,
                    request,
                    "send",
                )

            note_send_done.assert_awaited_once_with("req", 2, "127.0.0.1:5000")

        asyncio.run(run())


class TestEncoderPreprocessorKimiGrid(CustomTestCase):
    @staticmethod
    def _make_preprocessor(model_type="kimi_vl"):
        preprocessor = EncoderPreprocessor.__new__(EncoderPreprocessor)
        preprocessor.model_type = model_type
        preprocessor.model_config = SimpleNamespace(
            hf_config=SimpleNamespace(
                vision_config=SimpleNamespace(merge_kernel_size=(2, 2))
            )
        )
        preprocessor.image_processor = SimpleNamespace(merge_size=2)
        preprocessor._model_preprocessor = None
        return preprocessor

    @staticmethod
    def _make_encoder():
        return MMEncoder.__new__(MMEncoder)

    def test_kimi_vl_prefers_and_normalizes_hw_grid(self):
        mm_inputs = {
            "image_grid_hws": np.array([[40, 60]], dtype=np.int64),
            "image_grid_thw": torch.tensor([[1, 20, 30]]),
            "grid_thws": torch.tensor([[1, 10, 15]]),
        }

        grid = self._make_preprocessor()._get_mm_grid_dim(mm_inputs, Modality.IMAGE)

        self.assertIsInstance(grid, torch.Tensor)
        torch.testing.assert_close(grid, torch.tensor([[40, 60]]))

    def test_kimi_k25_keeps_thw_grid_preference(self):
        mm_inputs = {
            "image_grid_hws": np.array([[40, 60]], dtype=np.int64),
            "grid_thws": np.array([[1, 10, 15]], dtype=np.int64),
        }

        grid = self._make_preprocessor("kimi_k25")._get_mm_grid_dim(
            mm_inputs, Modality.IMAGE
        )

        torch.testing.assert_close(grid, torch.tensor([[1, 10, 15]]))

    def test_kimi_vl_2d_grid_counting_and_slicing(self):
        preprocessor = self._make_preprocessor()
        encoder = self._make_encoder()
        grids = torch.tensor([[40, 60], [20, 40]])
        embedding = torch.arange(800 * 2).reshape(800, 2)

        self.assertEqual(
            preprocessor.get_num_patches(grids[0], Modality.IMAGE),
            2400,
        )
        self.assertEqual(
            preprocessor.get_num_tokens(grids[0], Modality.IMAGE),
            600,
        )

        slices = encoder.slice_embedding(embedding, [600, 200])

        self.assertEqual([item.shape for item in slices], [(600, 2), (200, 2)])
        torch.testing.assert_close(slices[0], embedding[:600])
        torch.testing.assert_close(slices[1], embedding[600:])

    def test_kimi_3d_grid_remains_supported(self):
        preprocessor = self._make_preprocessor()
        grid = torch.tensor([1, 40, 60])

        self.assertEqual(preprocessor.get_num_patches(grid, Modality.IMAGE), 2400)
        self.assertEqual(preprocessor.get_num_tokens(grid, Modality.IMAGE), 600)

    def test_kimi_k25_3d_patch_counting_is_unchanged(self):
        preprocessor = self._make_preprocessor("kimi_k25")
        grid = torch.tensor([2, 12, 16])

        self.assertEqual(preprocessor.get_num_patches(grid, Modality.IMAGE), 384)
        self.assertEqual(preprocessor.get_num_tokens(grid, Modality.IMAGE), 48)

    def test_grid_metadata_is_safe_to_deserialize(self):
        grid = self._make_preprocessor()._get_mm_grid_dim(
            {"image_grid_hws": np.array([[40, 60]], dtype=np.int64)},
            Modality.IMAGE,
        )
        embedding_data = EmbeddingData(
            req_id="test-request",
            num_parts=1,
            part_idx=0,
            grid_dim=grid,
            modality=Modality.IMAGE,
            embedding=torch.zeros((600, 4)),
        )

        restored = safe_pickle_loads(
            pickle.dumps(embedding_data.copy_without_embedding())
        )

        torch.testing.assert_close(restored.grid_dim, torch.tensor([[40, 60]]))


class TestEncoderDelivery(CustomTestCase):
    def test_cancelled_zero_copy_transfer_drains_before_return(self):
        async def run():
            transfer_started = threading.Event()
            finish_transfer = threading.Event()

            def transfer():
                transfer_started.set()
                finish_transfer.wait()

            task = asyncio.create_task(
                _await_transfer_completion(asyncio.to_thread(transfer), "test transfer")
            )
            while not transfer_started.is_set():
                await asyncio.sleep(0)

            task.cancel()
            await asyncio.sleep(0)
            self.assertFalse(task.done())

            task.cancel()
            await asyncio.sleep(0)
            self.assertFalse(task.done())

            finish_transfer.set()
            with self.assertRaises(asyncio.CancelledError):
                await task

        asyncio.run(run())

    def test_cancelled_mooncake_send_keeps_embedding_until_transfer_stops(self):
        async def run():
            transfer_started = threading.Event()
            finish_transfer = threading.Event()

            def transfer_sync(*_args):
                transfer_started.set()
                finish_transfer.wait()
                return 0

            encoder = MMEncoder.__new__(MMEncoder)
            encoder.req_states = {}
            encoder._element_size = 2
            encoder.transfer_backend = "mooncake"
            encoder.engine = SimpleNamespace(
                register=unittest.mock.Mock(),
                transfer_sync=unittest.mock.Mock(side_effect=transfer_sync),
                deregister=unittest.mock.Mock(),
            )
            encoder.delivery = MooncakeDelivery(encoder)

            embedding = torch.ones((2, 4), dtype=torch.float16)
            state = ReqState(
                "cancelled-transfer",
                EmbeddingData(
                    "cancelled-transfer",
                    1,
                    0,
                    None,
                    Modality.IMAGE,
                    embedding=embedding,
                ),
            )
            state.embedding_ready.set()
            encoder.req_states[state.req_id] = state

            with (
                patch(
                    "sglang.srt.disaggregation.encoder.server.get_disagg",
                    return_value=SimpleNamespace(encoder_transfer_backend="mooncake"),
                ),
                patch.object(meta_registry, "discard", AsyncMock()),
            ):
                send_task = asyncio.create_task(
                    encoder.send_to_destination(
                        state,
                        SendDestination(
                            "127.0.0.1:1", session_id="session", buffer_address=1
                        ),
                    )
                )
                while not transfer_started.is_set():
                    await asyncio.sleep(0)

                send_task.cancel()
                release_task = asyncio.create_task(
                    encoder.release_request(state.req_id)
                )
                await asyncio.sleep(0)

                self.assertFalse(send_task.done())
                self.assertFalse(release_task.done())
                self.assertIs(state.embedding_data.embedding, embedding)
                encoder.engine.deregister.assert_not_called()

                finish_transfer.set()
                with self.assertRaises(asyncio.CancelledError):
                    await send_task
                await release_task

            encoder.engine.register.assert_called_once_with(
                embedding.data_ptr(), embedding.nbytes
            )
            encoder.engine.deregister.assert_called_once_with(embedding.data_ptr())
            self.assertIsNone(state.embedding_data)
            self.assertNotIn(state.req_id, encoder.req_states)

        asyncio.run(run())

    def test_failed_mooncake_transfer_releases_per_send_registration(self):
        async def run():
            encoder = MMEncoder.__new__(MMEncoder)
            encoder._element_size = 2
            encoder.transfer_backend = "mooncake"
            encoder.engine = SimpleNamespace(
                register=unittest.mock.Mock(),
                transfer_sync=unittest.mock.Mock(
                    side_effect=RuntimeError("transfer failed")
                ),
                deregister=unittest.mock.Mock(),
            )
            embedding = torch.ones((2, 4), dtype=torch.float16)
            mm_data = EmbeddingData(
                "failed-transfer",
                1,
                0,
                None,
                Modality.IMAGE,
                embedding=embedding,
            )

            with patch(
                "sglang.srt.disaggregation.encoder.server.get_disagg",
                return_value=SimpleNamespace(encoder_transfer_backend="mooncake"),
            ):
                with self.assertRaisesRegex(RuntimeError, "transfer failed"):
                    await encoder._send(
                        embedding,
                        mm_data,
                        session_id="session",
                        buffer_address=1,
                    )

            encoder.engine.register.assert_called_once_with(
                embedding.data_ptr(), embedding.nbytes
            )
            encoder.engine.deregister.assert_called_once_with(embedding.data_ptr())

        asyncio.run(run())

    @staticmethod
    def _global_cache_context(num_items=2):
        return SimpleNamespace(
            req_id="req",
            num_items=num_items,
            str_mm_hashes=[f"hash-{i}" for i in range(num_items)],
            modality=Modality.IMAGE,
            preprocess_result=SimpleNamespace(token_counts=[2] * num_items),
        )

    @staticmethod
    def _make_prefix_cache_encoder_and_context(get_feature_fn):
        encoder = MMEncoder.__new__(MMEncoder)
        encoder.mm_cache = MultiModalStaticCache(1024 * 1024)
        encoder.mm_cache_lock = asyncio.Lock()
        item = SimpleNamespace(hash=123, set_pad_value=lambda: None)
        encoder._build_model_mm_items = Mock(return_value=[item])
        ctx = SimpleNamespace(
            req_id="req",
            modality=Modality.IMAGE,
            num_items=1,
            mm_feature=None,
            preprocess_result=SimpleNamespace(token_counts=[2], mm_inputs={}),
            get_feature_fn=get_feature_fn,
            is_health_check=False,
            items_per_req=[1],
            aux_data={},
        )
        return encoder, ctx

    def test_invalid_fresh_embedding_is_not_cached(self):
        async def run():
            for actual_tokens in (1, 3):
                with self.subTest(actual_tokens=actual_tokens):
                    get_feature_fn = Mock(return_value=torch.zeros((actual_tokens, 4)))
                    encoder, ctx = self._make_prefix_cache_encoder_and_context(
                        get_feature_fn
                    )

                    with (
                        patch(
                            "sglang.srt.disaggregation.encoder.server.get_mm",
                            return_value=SimpleNamespace(enable_prefix_mm_cache=True),
                        ),
                        self.assertRaisesRegex(
                            InternalError,
                            f"Encoder produced {actual_tokens} tokens, but "
                            "preprocessor metadata expected 2",
                        ),
                    ):
                        await encoder._compute_direct_embedding(ctx, keep_on_gpu=False)

                    self.assertEqual(len(encoder.mm_cache), 0)
                    get_feature_fn.assert_called_once()

        asyncio.run(run())

    def test_valid_fresh_embedding_is_cached_and_reused(self):
        async def run():
            get_feature_fn = Mock(return_value=torch.zeros((2, 4)))
            encoder, ctx = self._make_prefix_cache_encoder_and_context(get_feature_fn)

            with patch(
                "sglang.srt.disaggregation.encoder.server.get_mm",
                return_value=SimpleNamespace(enable_prefix_mm_cache=True),
            ):
                first = await encoder._compute_direct_embedding(ctx, keep_on_gpu=False)
                second = await encoder._compute_direct_embedding(ctx, keep_on_gpu=False)

            torch.testing.assert_close(first, second)
            self.assertEqual(len(encoder.mm_cache), 1)
            get_feature_fn.assert_called_once()

        asyncio.run(run())

    def test_invalid_cached_embedding_is_evicted(self):
        async def run():
            get_feature_fn = Mock()
            encoder, ctx = self._make_prefix_cache_encoder_and_context(get_feature_fn)
            mm_hash = MultiModalStaticCache.combine_hashes([123])
            encoder.mm_cache.set(
                mm_hash,
                EmbeddingResult(embedding=torch.zeros((1, 4))),
            )

            with (
                patch(
                    "sglang.srt.disaggregation.encoder.server.get_mm",
                    return_value=SimpleNamespace(enable_prefix_mm_cache=True),
                ),
                self.assertRaisesRegex(
                    InternalError,
                    "Encoder produced 1 tokens, but preprocessor metadata expected 2",
                ),
            ):
                await encoder._compute_direct_embedding(ctx, keep_on_gpu=False)

            self.assertEqual(len(encoder.mm_cache), 0)
            get_feature_fn.assert_not_called()

        asyncio.run(run())

    def test_background_task_failure_is_observed(self):
        async def run():
            encoder = MMEncoder.__new__(MMEncoder)
            encoder.background_tasks = set()

            async def fail():
                raise RuntimeError("background failure")

            with patch(
                "sglang.srt.disaggregation.encoder.server.logger.exception"
            ) as log_exception:
                task = encoder._create_background_task(fail())
                await asyncio.sleep(0)
                await asyncio.sleep(0)

            self.assertTrue(task.done())
            self.assertNotIn(task, encoder.background_tasks)
            log_exception.assert_called_once_with("MMEncoder background task failed")

        asyncio.run(run())

    def test_contract_has_two_direct_implementations(self):
        self.assertEqual(EncoderDelivery.__abstractmethods__, {"send", "release"})
        self.assertEqual(
            set(EncoderDelivery.__subclasses__()),
            {
                MooncakeDelivery,
                ZmqDelivery,
            },
        )

    def test_failed_staged_send_releases_request(self):
        async def run():
            encoder = SimpleNamespace(
                send=AsyncMock(side_effect=RuntimeError("transfer failed")),
                release_request=AsyncMock(),
            )
            request = {
                "req_id": "req",
                "prefill_host": "127.0.0.1",
                "embedding_port": 1,
                "session_id": "session",
                "buffer_address": 2,
            }

            with self.assertRaisesRegex(RuntimeError, "transfer failed"):
                await send_staged_embedding(
                    encoder, request, release_without_count=False
                )

            encoder.release_request.assert_awaited_once_with("req")

        asyncio.run(run())

    def test_cleanup_failure_preserves_send_error_on_python_310(self):
        async def run():
            encoder = SimpleNamespace(
                send=AsyncMock(side_effect=ValueError("transfer failed")),
                release_request=AsyncMock(side_effect=RuntimeError("cleanup failed")),
            )
            request = {
                "req_id": "req",
                "prefill_host": "127.0.0.1",
                "embedding_port": 1,
                "session_id": "session",
                "buffer_address": 2,
            }

            with (
                patch.object(encoder_runtime.sys, "version_info", (3, 10)),
                self.assertLogs(encoder_runtime.logger, level="ERROR"),
                self.assertRaisesRegex(ValueError, "transfer failed"),
            ):
                await send_staged_embedding(
                    encoder, request, release_without_count=False
                )

        asyncio.run(run())

    def test_cancelled_staged_send_releases_request(self):
        async def run():
            encoder = SimpleNamespace(
                send=AsyncMock(side_effect=asyncio.CancelledError()),
                release_request=AsyncMock(),
            )
            request = {
                "req_id": "req",
                "prefill_host": "127.0.0.1",
                "embedding_port": 1,
                "session_id": "session",
                "buffer_address": 2,
            }

            with self.assertRaises(asyncio.CancelledError):
                await send_staged_embedding(
                    encoder, request, release_without_count=False
                )

            encoder.release_request.assert_awaited_once_with("req")

        asyncio.run(run())

    def test_staged_send_uses_refcount_or_legacy_release_policy(self):
        async def run():
            request = {
                "req_id": "req",
                "prefill_host": "127.0.0.1",
                "embedding_port": 1,
                "session_id": "session",
                "buffer_address": 2,
                "receive_count": 2,
            }
            encoder = SimpleNamespace(
                send=AsyncMock(return_value=True),
                release_request=AsyncMock(),
            )

            note_send_done = AsyncMock()
            with patch.object(meta_registry, "note_send_done", note_send_done):
                self.assertTrue(
                    await send_staged_embedding(
                        encoder, request, release_without_count=True
                    )
                )
            note_send_done.assert_awaited_once_with("req", 2, "127.0.0.1:1")
            encoder.release_request.assert_not_awaited()

            request.pop("receive_count")
            self.assertTrue(
                await send_staged_embedding(
                    encoder, request, release_without_count=True
                )
            )
            encoder.release_request.assert_awaited_once_with("req")

        asyncio.run(run())

    @staticmethod
    def _make_mooncake_send(engine):
        embedding = torch.zeros((2, 4), dtype=torch.float32)
        mm_data = EmbeddingData(
            "req",
            1,
            0,
            None,
            Modality.IMAGE,
            embedding=embedding,
        )
        encoder = MMEncoder.__new__(MMEncoder)
        encoder._element_size = embedding.element_size()
        encoder.engine = engine
        return encoder, embedding, mm_data

    def test_mooncake_fallback_registration_is_released_after_transfer_error(self):
        async def run():
            events = []

            def register(*_):
                events.append("register")

            def transfer_sync(*_):
                events.append("transfer")
                raise RuntimeError("transfer failed")

            def deregister(*_):
                events.append("deregister")

            engine = SimpleNamespace(
                register=register,
                transfer_sync=transfer_sync,
                deregister=deregister,
            )
            encoder, embedding, mm_data = self._make_mooncake_send(engine)

            with (
                patch(
                    "sglang.srt.disaggregation.encoder.server.get_disagg",
                    return_value=SimpleNamespace(encoder_transfer_backend="mooncake"),
                ),
                self.assertRaisesRegex(RuntimeError, "transfer failed"),
            ):
                await encoder._send(
                    embedding,
                    mm_data,
                    session_id="session",
                    buffer_address=1,
                )

            self.assertEqual(events, ["register", "transfer", "deregister"])

        asyncio.run(run())

    def test_mooncake_cancel_waits_before_releasing_fallback_registration(self):
        async def run():
            events = []
            transfer_started = threading.Event()
            finish_transfer = threading.Event()

            def register(*_):
                events.append("register")

            def transfer_sync(*_):
                events.append("transfer-start")
                transfer_started.set()
                finish_transfer.wait(timeout=2)
                events.append("transfer-finish")
                return 0

            def deregister(*_):
                events.append("deregister")

            engine = SimpleNamespace(
                register=register,
                transfer_sync=transfer_sync,
                deregister=deregister,
            )
            encoder, embedding, mm_data = self._make_mooncake_send(engine)

            with patch(
                "sglang.srt.disaggregation.encoder.server.get_disagg",
                return_value=SimpleNamespace(encoder_transfer_backend="mooncake"),
            ):
                send_task = asyncio.create_task(
                    encoder._send(
                        embedding,
                        mm_data,
                        session_id="session",
                        buffer_address=1,
                    )
                )
                self.assertTrue(await asyncio.to_thread(transfer_started.wait, 1))
                send_task.cancel()
                await asyncio.sleep(0)
                self.assertFalse(send_task.done())
                self.assertNotIn("deregister", events)

                send_task.cancel()
                await asyncio.sleep(0)
                self.assertFalse(send_task.done())
                self.assertNotIn("deregister", events)

                finish_transfer.set()
                with self.assertRaises(asyncio.CancelledError):
                    await send_task

            self.assertEqual(
                events,
                ["register", "transfer-start", "transfer-finish", "deregister"],
            )

        asyncio.run(run())

    def test_zmq_delivery_cleanup_is_configurable(self):
        async def run():
            req_id = "test-zmq-delivery-cleanup"
            rid_to_receive_endpoint[req_id] = {"127.0.0.1:1"}
            rid_to_receive_count[req_id] = 1
            rid_to_cond[req_id] = asyncio.Condition()
            state = ReqState(req_id)
            encoder = SimpleNamespace()

            await ZmqDelivery(encoder, cleanup_receive_state=False).release(state)
            self.assertIn(req_id, rid_to_receive_endpoint)
            self.assertIn(req_id, rid_to_receive_count)
            self.assertIn(req_id, rid_to_cond)

            await ZmqDelivery(encoder, cleanup_receive_state=True).release(state)
            self.assertNotIn(req_id, rid_to_receive_endpoint)
            self.assertNotIn(req_id, rid_to_receive_count)
            self.assertNotIn(req_id, rid_to_cond)

        asyncio.run(run())

    def test_preprocess_metadata_precedes_embedding(self):
        async def run():
            encoder = MMEncoder.__new__(MMEncoder)
            encoder.rank = 0
            encoder.req_states = {}
            encoder.abandoned_req_ids = set()
            encoder._embedding_dims = {Modality.IMAGE: 8}
            encoder._embedding_dtype = torch.float16
            encoder._element_size = 2
            first_state = encoder._acquire_encode_ref("req-0")
            second_state = encoder._acquire_encode_ref("req-1")
            ctx = SimpleNamespace(
                req_id="req-0",
                modality=Modality.IMAGE,
                items_per_req=[1, 2],
                preprocess_result=SimpleNamespace(
                    token_counts=[2, 3, 4],
                    grid_thw=[[1, 2, 3], [1, 4, 5], [1, 6, 7]],
                ),
            )
            requests = [
                {"req_id": "req-0", "num_parts": 2, "part_idx": 0},
                {"req_id": "req-1", "num_parts": 2, "part_idx": 1},
            ]

            publish = AsyncMock()
            with patch.object(meta_registry, "publish", publish):
                await encoder._publish_preprocess_metadata(ctx, requests)

            self.assertIs(encoder.req_states["req-0"], first_state)
            self.assertIs(encoder.req_states["req-1"], second_state)
            self.assertEqual(first_state.embedding_data.shape, [2, 8])
            self.assertEqual(second_state.embedding_data.shape, [7, 8])
            self.assertEqual(first_state.embedding_data.grid_dim, [[1, 2, 3]])
            self.assertEqual(
                second_state.embedding_data.grid_dim,
                [[1, 4, 5], [1, 6, 7]],
            )
            self.assertEqual(first_state.embedding_data.dtype, torch.float16)
            self.assertEqual(second_state.embedding_data.dtype, torch.float16)
            self.assertFalse(first_state.embedding_ready.is_set())
            self.assertFalse(second_state.embedding_ready.is_set())
            self.assertEqual(
                publish.await_args_list,
                [
                    unittest.mock.call("req-0", 32, 2, 8),
                    unittest.mock.call("req-1", 112, 7, 8),
                ],
            )
            await encoder._release_encode_ref(first_state)
            await encoder._release_encode_ref(second_state)

        asyncio.run(run())

    @staticmethod
    def _encode_context():
        return EncodeContext(
            req_id="req",
            modality=Modality.IMAGE,
            preprocess_result=SimpleNamespace(
                token_counts=[2],
                grid_thw=torch.tensor([[1, 2, 4]]),
            ),
            get_feature_fn=None,
            mm_feature=torch.zeros((8, 3)),
            num_items=1,
            items_per_req=[1],
            aux_data={},
            str_mm_hashes=None,
            use_global_cache=False,
            is_health_check=False,
        )

    @staticmethod
    def _load_grpc_server():
        try:
            import grpc
            from grpc_health.v1 import health_pb2
            from smg_grpc_proto import sglang_encoder_pb2

            health_pb2.HealthCheckRequest()
        except ImportError as e:
            raise unittest.SkipTest(f"gRPC test dependencies unavailable: {e}") from e
        except Exception as e:
            # Generated protobuf modules raise VersionError when the runner's
            # protobuf runtime is older than the code generator.
            if not (
                type(e).__module__ == "google.protobuf.runtime_version"
                and type(e).__name__ == "VersionError"
            ):
                raise
            raise unittest.SkipTest(f"gRPC test dependencies unavailable: {e}") from e

        # Import SGLang outside the dependency guard. Product-code import
        # failures are regressions and must fail the test instead of skipping.
        from sglang.srt.disaggregation.encoder.grpc_server import (
            SGLangEncoderServer,
        )

        return grpc, sglang_encoder_pb2, SGLangEncoderServer

    def test_remote_preprocess_failure_stops_all_tp_ranks_before_forward(self):
        async def run():
            encoder = MMEncoder.__new__(MMEncoder)
            encoder._prepare_encode_context = AsyncMock(
                return_value=self._encode_context()
            )
            encoder._publish_preprocess_metadata = AsyncMock()

            class TPGroup:
                world_size = 2
                cpu_group = object()

                @staticmethod
                def all_gather_object(local_error):
                    return [local_error, "bad image"]

            def all_gather(statuses, local_status, group):
                self.assertIs(group, TPGroup.cpu_group)
                statuses[0].copy_(local_status)
                statuses[1].copy_(torch.tensor([400, 1, 0, 0]))

            with (
                patch(
                    "sglang.srt.disaggregation.encoder.server.get_tp_group",
                    return_value=TPGroup(),
                ),
                patch(
                    "sglang.srt.disaggregation.encoder.server.torch.distributed.all_gather",
                    side_effect=all_gather,
                ),
            ):
                with self.assertRaisesRegex(
                    BadRequestError,
                    "failed on TP rank 1: bad image",
                ):
                    await encoder._prepare_encode_context_on_all_ranks(
                        [{"req_id": "req"}],
                        Modality.IMAGE,
                        use_global_cache=False,
                    )

        asyncio.run(run())

    def test_tp_preprocess_layout_mismatch_fails_before_forward(self):
        async def run():
            encoder = MMEncoder.__new__(MMEncoder)
            encoder._prepare_encode_context = AsyncMock(
                return_value=self._encode_context()
            )
            encoder._publish_preprocess_metadata = AsyncMock()

            class TPGroup:
                world_size = 2
                cpu_group = object()

            def all_gather(statuses, local_status, group):
                self.assertIs(group, TPGroup.cpu_group)
                statuses[0].copy_(local_status)
                statuses[1].copy_(local_status)
                statuses[1][2] += 1

            with (
                patch(
                    "sglang.srt.disaggregation.encoder.server.get_tp_group",
                    return_value=TPGroup(),
                ),
                patch(
                    "sglang.srt.disaggregation.encoder.server.torch.distributed.all_gather",
                    side_effect=all_gather,
                ),
            ):
                with self.assertRaisesRegex(
                    InternalError,
                    "inconsistent layouts across TP ranks 0 and 1",
                ):
                    await encoder._prepare_encode_context_on_all_ranks(
                        [{"req_id": "req"}],
                        Modality.IMAGE,
                        use_global_cache=False,
                    )

        asyncio.run(run())

    def test_remote_metadata_failure_stops_tp_peer_before_forward(self):
        async def run():
            encoder = MMEncoder.__new__(MMEncoder)
            encoder._prepare_encode_context = AsyncMock(
                return_value=self._encode_context()
            )
            encoder._publish_preprocess_metadata = AsyncMock()

            class TPGroup:
                world_size = 2
                cpu_group = object()

                @staticmethod
                def all_gather_object(local_error):
                    self.assertIsNone(local_error)
                    return ["registry down", None]

            def all_gather(statuses, local_status, group):
                self.assertIs(group, TPGroup.cpu_group)
                statuses[0].copy_(torch.tensor([500, 2, 0, 0]))
                statuses[1].copy_(local_status)

            with (
                patch(
                    "sglang.srt.disaggregation.encoder.server.get_tp_group",
                    return_value=TPGroup(),
                ),
                patch(
                    "sglang.srt.disaggregation.encoder.server.torch.distributed.all_gather",
                    side_effect=all_gather,
                ),
            ):
                with self.assertRaisesRegex(
                    InternalError,
                    "metadata publication failed on TP rank 0: registry down",
                ):
                    await encoder._prepare_encode_context_on_all_ranks(
                        [{"req_id": "req"}],
                        Modality.IMAGE,
                        use_global_cache=False,
                    )

        asyncio.run(run())

    def test_unexpected_preprocess_failure_is_internal(self):
        async def run():
            encoder = MMEncoder.__new__(MMEncoder)
            encoder.preprocessor = SimpleNamespace(
                process_batch_mm_items=AsyncMock(side_effect=RuntimeError("boom"))
            )
            with self.assertRaisesRegex(InternalError, "boom"):
                await encoder._prepare_encode_context(
                    [{"req_id": "req"}],
                    Modality.IMAGE,
                    use_global_cache=False,
                )

        asyncio.run(run())

    def test_grpc_rejects_invalid_request_before_tp_dispatch(self):
        async def run():
            grpc, sglang_encoder_pb2, SGLangEncoderServer = self._load_grpc_server()

            context = SimpleNamespace(
                set_code=unittest.mock.Mock(),
                set_details=unittest.mock.Mock(),
            )
            server = SGLangEncoderServer(
                encoder=SimpleNamespace(),
                send_sockets=[object()],
                server_args=SimpleNamespace(),
            )
            request = sglang_encoder_pb2.EncodeRequest(
                mm_items=["image"],
                req_id="invalid",
                part_idx=0,
            )

            with patch(
                "sglang.srt.disaggregation.encoder.grpc_server.async_sock_send",
                new_callable=AsyncMock,
            ) as send:
                await server.Encode(request, context)

            send.assert_not_awaited()
            context.set_code.assert_called_once_with(grpc.StatusCode.INVALID_ARGUMENT)
            self.assertIn("num_parts", context.set_details.call_args.args[0])

        asyncio.run(run())

    def test_grpc_maps_processor_bad_request_to_invalid_argument(self):
        async def run():
            grpc, sglang_encoder_pb2, SGLangEncoderServer = self._load_grpc_server()

            encoder = SimpleNamespace(
                encode_dispatch_lock=asyncio.Lock(),
                encode_request=AsyncMock(
                    return_value=(
                        0,
                        0,
                        0,
                        "invalid image",
                        HTTPStatus.BAD_REQUEST,
                    )
                ),
                release_request=AsyncMock(),
            )
            context = SimpleNamespace(
                set_code=unittest.mock.Mock(),
                set_details=unittest.mock.Mock(),
            )
            server = SGLangEncoderServer(
                encoder=encoder,
                send_sockets=[],
                server_args=SimpleNamespace(),
            )
            request = sglang_encoder_pb2.EncodeRequest(
                mm_items=["bad-image"],
                req_id="bad-image",
                num_parts=1,
                part_idx=0,
            )

            await server.Encode(request, context)

            context.set_code.assert_called_once_with(grpc.StatusCode.INVALID_ARGUMENT)
            context.set_details.assert_called_once_with("invalid image")
            encoder.release_request.assert_awaited_once_with("bad-image")

        asyncio.run(run())

    def test_grpc_serializes_tp_dispatch_with_rank_zero_encode(self):
        async def run():
            _, sglang_encoder_pb2, SGLangEncoderServer = self._load_grpc_server()
            from sglang.srt.managers.io_struct import unwrap_from_pickle

            first_started = asyncio.Event()
            release_first = asyncio.Event()
            events = []

            class Encoder:
                def __init__(self):
                    self.encode_dispatch_lock = asyncio.Lock()

                async def encode_request(self, request, _modality):
                    req_id = request["req_id"]
                    events.append(("encode-start", req_id))
                    if req_id == "first":
                        first_started.set()
                        await release_first.wait()
                    events.append(("encode-end", req_id))
                    return 8, 1, 8, None, None

            async def send(_socket, payload):
                request = unwrap_from_pickle(payload)
                events.append(("send", request["req_id"]))

            server = SGLangEncoderServer(
                encoder=Encoder(),
                send_sockets=[object()],
                server_args=SimpleNamespace(),
            )
            requests = [
                sglang_encoder_pb2.EncodeRequest(
                    mm_items=["image"],
                    req_id=req_id,
                    num_parts=1,
                    part_idx=0,
                )
                for req_id in ("first", "second")
            ]
            contexts = [
                SimpleNamespace(
                    set_code=unittest.mock.Mock(),
                    set_details=unittest.mock.Mock(),
                )
                for _ in requests
            ]

            with (
                patch(
                    "sglang.srt.disaggregation.encoder.grpc_server.async_sock_send",
                    side_effect=send,
                ),
                patch(
                    "sglang.srt.disaggregation.encoder.grpc_server.get_disagg",
                    return_value=SimpleNamespace(encoder_transfer_backend="mooncake"),
                ),
            ):
                first = asyncio.create_task(server.Encode(requests[0], contexts[0]))
                await first_started.wait()
                second = asyncio.create_task(server.Encode(requests[1], contexts[1]))
                await asyncio.sleep(0)
                self.assertEqual(
                    events,
                    [("send", "first"), ("encode-start", "first")],
                )
                release_first.set()
                await asyncio.gather(first, second)

            self.assertEqual(
                events,
                [
                    ("send", "first"),
                    ("encode-start", "first"),
                    ("encode-end", "first"),
                    ("send", "second"),
                    ("encode-start", "second"),
                    ("encode-end", "second"),
                ],
            )

        asyncio.run(run())

    def test_grpc_encode_cancellation_drains_tp_collective_before_release(self):
        async def run():
            _, sglang_encoder_pb2, SGLangEncoderServer = self._load_grpc_server()

            encode_started = asyncio.Event()
            finish_encode = asyncio.Event()

            async def encode_request(*_args):
                encode_started.set()
                await finish_encode.wait()
                return 8, 1, 8, None, None

            encoder = SimpleNamespace(
                encode_dispatch_lock=asyncio.Lock(),
                encode_request=AsyncMock(side_effect=encode_request),
                release_request=AsyncMock(),
            )
            server = SGLangEncoderServer(
                encoder=encoder,
                send_sockets=[],
                server_args=SimpleNamespace(),
            )
            request = sglang_encoder_pb2.EncodeRequest(
                mm_items=["image"],
                req_id="cancelled-encode",
                num_parts=1,
                part_idx=0,
            )
            context = SimpleNamespace(
                set_code=unittest.mock.Mock(),
                set_details=unittest.mock.Mock(),
            )

            with patch(
                "sglang.srt.disaggregation.encoder.grpc_server.get_disagg",
                return_value=SimpleNamespace(encoder_transfer_backend="mooncake"),
            ):
                task = asyncio.create_task(server.Encode(request, context))
                await encode_started.wait()
                task.cancel()
                await asyncio.sleep(0)

                # Cancellation cannot interrupt an in-flight TP collective.
                self.assertFalse(task.done())
                encoder.release_request.assert_not_awaited()

                task.cancel()
                await asyncio.sleep(0)
                self.assertFalse(task.done())
                encoder.release_request.assert_not_awaited()

                finish_encode.set()
                with self.assertRaises(asyncio.CancelledError):
                    await task

            encoder.release_request.assert_awaited_once_with("cancelled-encode")

        asyncio.run(run())

    def test_grpc_encode_cancellation_during_tp_dispatch_completes_encode(self):
        async def run():
            _, sglang_encoder_pb2, SGLangEncoderServer = self._load_grpc_server()

            send_started = asyncio.Event()
            finish_send = asyncio.Event()

            async def send_to_tp(*_args):
                send_started.set()
                await finish_send.wait()

            encoder = SimpleNamespace(
                encode_dispatch_lock=asyncio.Lock(),
                encode_request=AsyncMock(return_value=(8, 1, 8, None, None)),
                release_request=AsyncMock(),
            )
            server = SGLangEncoderServer(
                encoder=encoder,
                send_sockets=[object()],
                server_args=SimpleNamespace(),
            )
            request = sglang_encoder_pb2.EncodeRequest(
                mm_items=["image"],
                req_id="cancelled-dispatch",
                num_parts=1,
                part_idx=0,
            )
            context = SimpleNamespace(
                set_code=unittest.mock.Mock(),
                set_details=unittest.mock.Mock(),
            )

            with (
                patch(
                    "sglang.srt.disaggregation.encoder.grpc_server.async_sock_send",
                    side_effect=send_to_tp,
                ),
                patch(
                    "sglang.srt.disaggregation.encoder.grpc_server.get_disagg",
                    return_value=SimpleNamespace(encoder_transfer_backend="mooncake"),
                ),
            ):
                task = asyncio.create_task(server.Encode(request, context))
                await send_started.wait()
                task.cancel()
                await asyncio.sleep(0)

                self.assertFalse(task.done())
                encoder.encode_request.assert_not_awaited()

                finish_send.set()
                with self.assertRaises(asyncio.CancelledError):
                    await task

            encoder.encode_request.assert_awaited_once()
            encoder.release_request.assert_awaited_once_with("cancelled-dispatch")

        asyncio.run(run())

    def test_grpc_send_cancellation_releases_request(self):
        async def run():
            _, sglang_encoder_pb2, SGLangEncoderServer = self._load_grpc_server()

            send_started = asyncio.Event()

            async def send(*_args, **_kwargs):
                send_started.set()
                await asyncio.Event().wait()

            encoder = SimpleNamespace(
                send=AsyncMock(side_effect=send),
                release_request=AsyncMock(),
            )
            server = SGLangEncoderServer(
                encoder=encoder,
                send_sockets=[],
                server_args=SimpleNamespace(),
            )
            request = sglang_encoder_pb2.SendRequest(
                req_id="cancelled-send",
                prefill_host="127.0.0.1",
                embedding_port=30001,
            )
            context = SimpleNamespace()

            task = asyncio.create_task(server.Send(request, context))
            await send_started.wait()
            task.cancel()
            with self.assertRaises(asyncio.CancelledError):
                await task

            encoder.release_request.assert_awaited_once_with("cancelled-send")

        asyncio.run(run())

    def test_global_cache_lookup_failure_falls_back_to_all_misses(self):
        async def run():
            encoder = MMEncoder.__new__(MMEncoder)
            encoder.rank = 0
            encoder.mm_global_cache = SimpleNamespace(
                batch_is_exist=AsyncMock(side_effect=RuntimeError("store down"))
            )
            encoder._broadcast_global_cache_mask = unittest.mock.Mock()

            missing_indices, hit_indices = await encoder._lookup_global_cache(
                self._global_cache_context()
            )

            self.assertEqual(missing_indices, [0, 1])
            self.assertEqual(hit_indices, [])
            torch.testing.assert_close(
                encoder._broadcast_global_cache_mask.call_args.args[0],
                torch.zeros(2, dtype=torch.int32),
            )

        asyncio.run(run())

    def test_global_cache_prefetch_failure_immediately_falls_back(self):
        async def run():
            encoder = MMEncoder.__new__(MMEncoder)
            encoder.rank = 0
            encoder.mm_global_cache = SimpleNamespace(
                prefetch=unittest.mock.Mock(side_effect=RuntimeError("store down"))
            )
            encoder._broadcast_global_cache_mask = unittest.mock.Mock()
            ctx = self._global_cache_context()

            hit_hashes, failed = encoder._prefetch_global_cache_hits(ctx, [0, 1])
            fallback_indices = await encoder._wait_global_cache_prefetch(
                ctx, [0, 1], hit_hashes, failed
            )

            self.assertTrue(failed)
            self.assertEqual(hit_hashes, [])
            self.assertEqual(fallback_indices, [0, 1])
            torch.testing.assert_close(
                encoder._broadcast_global_cache_mask.call_args.args[0],
                torch.ones(2, dtype=torch.int32),
            )

        asyncio.run(run())

    def test_global_cache_staging_failure_skips_insert(self):
        encoder = MMEncoder.__new__(MMEncoder)
        encoder.mm_global_cache = SimpleNamespace(
            store_to_pool_async=unittest.mock.Mock(
                side_effect=RuntimeError("pool full")
            )
        )

        hashes, handles = encoder._stage_global_cache_slices(
            self._global_cache_context(num_items=1),
            [0],
            [torch.ones((2, 4))],
        )

        self.assertEqual(hashes, [])
        self.assertEqual(handles, [])

    def test_global_cache_insert_failure_is_contained(self):
        async def run():
            encoder = MMEncoder.__new__(MMEncoder)
            encoder.background_tasks = set()
            encoder.mm_global_cache = SimpleNamespace(
                wait_store_to_pool=unittest.mock.Mock(
                    side_effect=RuntimeError("store down")
                ),
                insert_batch=unittest.mock.Mock(),
            )

            encoder._launch_global_cache_insert(
                self._global_cache_context(num_items=1), ["hash-0"], [object()]
            )
            await asyncio.gather(*encoder.background_tasks)

            encoder.mm_global_cache.insert_batch.assert_not_called()

        asyncio.run(run())

    def test_mooncake_embedding_is_ready_only_after_cuda_sync(self):
        class FakeCudaEmbedding:
            shape = (2, 4)
            dtype = torch.float16
            nbytes = 16
            is_cuda = True
            device = "cuda:0"

            def __getitem__(self, key):
                return self

        encoder = MMEncoder.__new__(MMEncoder)
        encoder.rank = 0
        events = []
        state = ReqState("req", active_encodes=1)
        state.embedding_ready = SimpleNamespace(set=lambda: events.append("ready"))
        encoder.req_states = {"req": state}
        ctx = SimpleNamespace(
            req_id="req",
            modality=Modality.IMAGE,
            items_per_req=[1],
            preprocess_result=SimpleNamespace(
                token_counts=[2],
                grid_thw=[[1, 2, 3]],
            ),
            aux_data={},
            use_global_cache=True,
        )
        stream = SimpleNamespace(synchronize=lambda: events.append("sync"))

        with patch.object(torch.cuda, "current_stream", return_value=stream):
            encoder._stage_embeddings(
                ctx,
                [{"req_id": "req", "num_parts": 1, "part_idx": 0}],
                FakeCudaEmbedding(),
                keep_on_gpu=True,
            )

        self.assertEqual(events, ["sync", "ready"])

    def test_shared_mr_registration_failure_keeps_send_fallback_enabled(self):
        encoder = MMEncoder.__new__(MMEncoder)
        encoder.engine = unittest.mock.Mock()
        encoder.engine.register.side_effect = RuntimeError("register failed")
        embedding = torch.ones((2, 4))
        mm_data = EmbeddingData(
            "req", 1, 0, [[1, 1, 1]], Modality.IMAGE, embedding=embedding
        )

        encoder._register_shared_mr(mm_data, embedding)

        self.assertIsNone(mm_data._mr_ptr)

    def test_fused_staging_rolls_back_mrs_before_any_result_is_published(self):
        encoder = MMEncoder.__new__(MMEncoder)
        encoder.rank = 0
        encoder.engine = Mock()
        first_state = ReqState("req-0", active_encodes=1)
        second_metadata = EmbeddingData(
            "req-1",
            1,
            0,
            [[1, 1, 1]],
            Modality.IMAGE,
            embedding_shape=[2, 4],
            dtype=torch.float32,
        )
        second_state = ReqState(
            "req-1", embedding_data=second_metadata, active_encodes=1
        )
        encoder.req_states = {"req-0": first_state, "req-1": second_state}
        ctx = SimpleNamespace(
            req_id="req-0",
            modality=Modality.IMAGE,
            items_per_req=[1, 1],
            preprocess_result=SimpleNamespace(
                token_counts=[1, 1],
                grid_thw=[[1, 1, 1], [1, 1, 1]],
            ),
            aux_data={},
            use_global_cache=False,
        )
        requests = [
            {"req_id": "req-0", "num_parts": 1, "part_idx": 0},
            {"req_id": "req-1", "num_parts": 1, "part_idx": 0},
        ]

        with self.assertRaisesRegex(InternalError, "Embedding metadata mismatch"):
            encoder._stage_embeddings(
                ctx, requests, torch.ones((2, 4)), keep_on_gpu=True
            )

        registered_ptrs = [
            call.args[0] for call in encoder.engine.register.call_args_list
        ]
        deregistered_ptrs = [
            call.args[0] for call in encoder.engine.deregister.call_args_list
        ]
        self.assertEqual(len(registered_ptrs), 2)
        self.assertCountEqual(deregistered_ptrs, registered_ptrs)
        self.assertIsNone(first_state.embedding_data)
        self.assertIs(second_state.embedding_data, second_metadata)
        self.assertFalse(first_state.embedding_ready.is_set())
        self.assertFalse(second_state.embedding_ready.is_set())

    def test_stage_embedding_does_not_resurrect_missing_state(self):
        encoder = MMEncoder.__new__(MMEncoder)
        encoder.req_states = {}

        with self.assertRaisesRegex(
            InternalError, "No request state exists while encoding request: req"
        ):
            encoder._stage_embedding(
                EmbeddingData(
                    "req",
                    1,
                    0,
                    None,
                    Modality.IMAGE,
                    embedding=torch.ones((1, 1)),
                )
            )

        self.assertNotIn("req", encoder.req_states)

    def test_stage_embedding_requires_active_encode(self):
        encoder = MMEncoder.__new__(MMEncoder)
        state = ReqState("req")
        encoder.req_states = {"req": state}

        with self.assertRaisesRegex(
            InternalError, "Request state has no active encode work: req"
        ):
            encoder._stage_embedding(
                EmbeddingData(
                    "req",
                    1,
                    0,
                    None,
                    Modality.IMAGE,
                    embedding=torch.ones((1, 1)),
                )
            )

        self.assertIs(encoder.req_states["req"], state)
        self.assertIsNone(state.embedding_data)
        self.assertFalse(state.embedding_ready.is_set())

    def test_release_during_encode_is_deferred_without_resurrecting_state(self):
        async def run():
            encoder = MMEncoder.__new__(MMEncoder)
            encoder.rank = 0
            encoder.req_states = {}
            encoder.abandoned_req_ids = set()
            encoder.delivery = SimpleNamespace(release=AsyncMock())

            state = encoder._acquire_encode_ref("req")
            state.embedding_data = EmbeddingData(
                "req",
                1,
                0,
                None,
                Modality.IMAGE,
                embedding_shape=[1, 1],
                dtype=torch.float32,
            )

            discard = AsyncMock()
            with patch.object(meta_registry, "discard", discard):
                await encoder.release_request("req")
                self.assertTrue(state.release_requested)
                self.assertIn("req", encoder.req_states)
                encoder.delivery.release.assert_not_awaited()

                embedding = torch.ones((1, 1))
                encoder._stage_embedding(
                    EmbeddingData(
                        "req",
                        1,
                        0,
                        None,
                        Modality.IMAGE,
                        embedding=embedding,
                    )
                )
                await encoder._release_encode_ref(state)

            encoder.delivery.release.assert_awaited_once_with(state)
            discard.assert_awaited_once_with("req")
            self.assertIsNone(state.embedding_data)
            self.assertNotIn("req", encoder.req_states)

        asyncio.run(run())

    def test_abandon_before_encode_is_applied_when_state_is_created(self):
        async def run():
            encoder = MMEncoder.__new__(MMEncoder)
            encoder.rank = 0
            encoder.req_states = {}
            encoder.abandoned_req_ids = set()
            encoder.delivery = SimpleNamespace(release=AsyncMock())

            await encoder.abandon_request("req")
            self.assertIn("req", encoder.abandoned_req_ids)

            state = encoder._acquire_encode_ref("req")
            self.assertTrue(state.release_requested)
            self.assertNotIn("req", encoder.abandoned_req_ids)
            encoder._stage_embedding(
                EmbeddingData(
                    "req",
                    1,
                    0,
                    None,
                    Modality.IMAGE,
                    embedding=torch.ones((1, 1)),
                )
            )
            with patch.object(meta_registry, "discard", AsyncMock()):
                await encoder._release_encode_ref(state)

            encoder.delivery.release.assert_awaited_once_with(state)
            self.assertNotIn("req", encoder.req_states)

        asyncio.run(run())

    def test_error_metadata_survives_buffer_release_for_waiter(self):
        async def run():
            req_id = "test-error-metadata-waiter"
            await meta_registry.discard(req_id)

            encoder = MMEncoder.__new__(MMEncoder)
            encoder.req_states = {}
            encoder.delivery = SimpleNamespace(release=AsyncMock())
            state = ReqState(
                req_id,
                EmbeddingData(
                    req_id,
                    1,
                    0,
                    None,
                    Modality.IMAGE,
                    error_msg="encode failed",
                ),
            )
            state.embedding_ready.set()
            encoder.req_states[req_id] = state

            waiter = asyncio.create_task(meta_registry.wait(req_id))
            await asyncio.sleep(0)
            try:
                await meta_registry.publish(req_id, 0, 0, 0, error="encode failed")
                await encoder.release_request(req_id, preserve_metadata=True)
                meta = await asyncio.wait_for(waiter, timeout=1)
                self.assertEqual(meta, {"error": "encode failed"})
                self.assertNotIn(req_id, encoder.req_states)
            finally:
                if not waiter.done():
                    waiter.cancel()
                await meta_registry.discard(req_id)

        asyncio.run(run())

    def test_zmq_pipeline_sends_only_after_encode_completes(self):
        async def run():
            events = []
            finish_encode = asyncio.Event()
            encoder = MMEncoder.__new__(MMEncoder)
            encoder.transfer_backend = "zmq_to_tokenizer"

            async def encode(**kwargs):
                events.append("metadata_published")
                await finish_encode.wait()
                events.append("encode_completed")
                return 16, 2, 4, None, None

            async def send(**kwargs):
                events.append("send")
                return True

            async def release_request(req_id, **kwargs):
                events.append("release")

            encoder.encode = AsyncMock(side_effect=encode)
            encoder.send = AsyncMock(side_effect=send)
            encoder.release_request = AsyncMock(side_effect=release_request)

            publish = AsyncMock()
            request = {
                "req_id": "req",
                "mm_items": ["item"],
                "modality": "image",
                "num_parts": 1,
                "part_idx": 0,
                "prefill_host": "127.0.0.1",
                "embedding_port": 1234,
            }
            with patch.object(meta_registry, "publish", publish):
                task = asyncio.create_task(
                    execute_encode_pipeline(encoder, None, request)
                )
                await asyncio.sleep(0)
                self.assertEqual(events, ["metadata_published"])
                encoder.send.assert_not_awaited()

                finish_encode.set()
                self.assertIsNone(await task)

            self.assertEqual(
                events,
                ["metadata_published", "encode_completed", "send", "release"],
            )
            publish.assert_awaited_once_with("req", 16, 2, 4)

        asyncio.run(run())

    def test_cancelled_tp_pipeline_drains_encode_before_release(self):
        async def run():
            encode_started = asyncio.Event()
            finish_encode = asyncio.Event()
            encoder = MMEncoder.__new__(MMEncoder)
            encoder.transfer_backend = "zmq_to_tokenizer"
            encoder.encode_dispatch_lock = asyncio.Lock()

            async def encode(**_kwargs):
                encode_started.set()
                await finish_encode.wait()
                self.assertTrue(encoder.encode_dispatch_lock.locked())
                return 16, 2, 4, None, None

            encoder.encode = AsyncMock(side_effect=encode)
            encoder.release_request = AsyncMock()
            request = {
                "req_id": "cancelled",
                "mm_items": ["item"],
                "modality": "video",
                "num_parts": 1,
                "part_idx": 0,
            }

            with patch("sglang.srt.disaggregation.encoder.runtime.sock_send") as send:
                task = asyncio.create_task(
                    execute_encode_pipeline(
                        encoder, None, request, send_sockets=[object()]
                    )
                )
                await encode_started.wait()
                task.cancel()
                await asyncio.sleep(0)

                self.assertFalse(task.done())
                self.assertTrue(encoder.encode_dispatch_lock.locked())
                encoder.release_request.assert_not_awaited()
                send.assert_called_once()

                task.cancel()
                await asyncio.sleep(0)
                self.assertFalse(task.done())
                self.assertTrue(encoder.encode_dispatch_lock.locked())

                finish_encode.set()
                with self.assertRaises(asyncio.CancelledError):
                    await task

            encoder.release_request.assert_awaited_once_with("cancelled")
            self.assertFalse(encoder.encode_dispatch_lock.locked())

        asyncio.run(run())

    def test_pipeline_releases_request_when_error_publish_fails(self):
        async def run():
            encoder = MMEncoder.__new__(MMEncoder)
            encoder.transfer_backend = "mooncake"
            encoder.encode = AsyncMock(side_effect=RuntimeError("encode failed"))
            encoder.release_request = AsyncMock()
            request = {
                "req_id": "req",
                "mm_items": ["item"],
                "modality": "image",
                "num_parts": 1,
                "part_idx": 0,
            }

            with patch.object(
                meta_registry,
                "publish",
                AsyncMock(side_effect=RuntimeError("registry failed")),
            ):
                with self.assertRaisesRegex(RuntimeError, "encode failed"):
                    await execute_encode_pipeline(encoder, None, request)

            encoder.release_request.assert_awaited_once_with(
                "req", preserve_metadata=False
            )

        asyncio.run(run())

    def test_pipeline_releases_error_result_when_error_send_fails(self):
        async def run():
            encoder = MMEncoder.__new__(MMEncoder)
            encoder.transfer_backend = "zmq_to_scheduler"
            encoder.encode = AsyncMock(return_value=(0, 0, 0, "bad image", 400))
            encoder.release_request = AsyncMock()
            request = {
                "req_id": "req",
                "mm_items": ["item"],
                "modality": "image",
                "num_parts": 1,
                "part_idx": 0,
            }

            with (
                patch.object(meta_registry, "publish", AsyncMock()),
                patch(
                    "sglang.srt.disaggregation.encoder.runtime._push_embedding_to_prefill",
                    AsyncMock(side_effect=RuntimeError("send failed")),
                ),
            ):
                with self.assertRaisesRegex(MMError, "bad image"):
                    await execute_encode_pipeline(encoder, None, request)

            encoder.release_request.assert_awaited_once_with(
                "req", preserve_metadata=False
            )

        asyncio.run(run())

    def test_send_waits_for_embedding_published_by_encode(self):
        async def run():
            encoder = MMEncoder.__new__(MMEncoder)
            encoder.rank = 0
            encoder.req_states = {}
            encoder.abandoned_req_ids = set()
            state = encoder._acquire_encode_ref("req")
            state.embedding_data = EmbeddingData(
                "req",
                1,
                0,
                None,
                Modality.IMAGE,
                embedding_shape=[1, 1],
                dtype=torch.float32,
            )

            delivered = []

            async def send(current_state, destination):
                delivered.append(await encoder._wait_for_embedding(current_state))

            encoder.delivery = SimpleNamespace(
                send=AsyncMock(side_effect=send),
                release=AsyncMock(),
            )
            send_task = asyncio.create_task(
                encoder.send_to_destination(state, SendDestination("127.0.0.1:1"))
            )
            await asyncio.sleep(0)
            self.assertFalse(send_task.done())

            embedding = torch.ones((1, 1))
            encoder._stage_embedding(
                EmbeddingData(
                    "req",
                    1,
                    0,
                    None,
                    Modality.IMAGE,
                    embedding=embedding,
                )
            )
            await send_task
            await encoder._release_encode_ref(state)

            with patch.object(meta_registry, "discard", AsyncMock()):
                await encoder.release_request("req")

            self.assertEqual(len(delivered), 1)
            self.assertIs(delivered[0].embedding, embedding)

        asyncio.run(run())

    def test_release_waits_for_send_then_clears_embedding(self):
        async def run():
            encoder = MMEncoder.__new__(MMEncoder)
            encoder.req_states = {}

            send_started = asyncio.Event()
            finish_send = asyncio.Event()

            async def send(state, destination):
                send_started.set()
                await finish_send.wait()

            embedding_seen_by_release = []

            async def release(state):
                embedding_seen_by_release.append(state.embedding_data.embedding)

            encoder.delivery = SimpleNamespace(
                send=AsyncMock(side_effect=send),
                release=AsyncMock(side_effect=release),
            )
            embedding = torch.ones((1, 1))
            state = ReqState(
                "req",
                EmbeddingData("req", 1, 0, None, Modality.IMAGE, embedding=embedding),
            )
            state.embedding_ready.set()
            encoder.req_states[state.req_id] = state

            send_task = asyncio.create_task(
                encoder.send_to_destination(state, SendDestination("127.0.0.1:1"))
            )
            await send_started.wait()
            release_task = asyncio.create_task(encoder.release_request("req"))
            await asyncio.sleep(0)

            encoder.delivery.release.assert_not_awaited()
            self.assertIs(state.embedding_data.embedding, embedding)

            finish_send.set()
            await send_task
            await release_task

            encoder.delivery.release.assert_awaited_once_with(state)
            self.assertEqual(len(embedding_seen_by_release), 1)
            self.assertIs(embedding_seen_by_release[0], embedding)
            self.assertIsNone(state.embedding_data)
            self.assertNotIn("req", encoder.req_states)

        asyncio.run(run())

    def test_release_wakes_destination_waiter(self):
        async def run():
            req_id = "test-release-wakes-destination-waiter"
            await meta_registry.discard(req_id)

            encoder = MMEncoder.__new__(MMEncoder)
            encoder.req_states = {req_id: ReqState(req_id)}
            encoder.send_timeout = 60
            encoder.delivery = ZmqDelivery(encoder, cleanup_receive_state=True)

            send_task = asyncio.create_task(encoder.send_with_url(req_id))
            await asyncio.sleep(0)
            self.assertFalse(send_task.done())

            await asyncio.wait_for(encoder.release_request(req_id), timeout=1)
            await asyncio.wait_for(send_task, timeout=1)

            self.assertNotIn(req_id, encoder.req_states)
            self.assertNotIn(req_id, rid_to_cond)
            self.assertNotIn(req_id, rid_to_receive_endpoint)
            self.assertNotIn(req_id, rid_to_receive_count)

        asyncio.run(run())

    def test_destination_registration_respects_request_lifecycle(self):
        async def run():
            req_id = "registration-lifecycle"
            await meta_registry.discard(req_id)

            encoder = MMEncoder.__new__(MMEncoder)
            state = ReqState(req_id)
            state.active_encodes = 1
            encoder.req_states = {req_id: state}
            encoder.delivery = ZmqDelivery(encoder, cleanup_receive_state=True)

            await encoder.register_embedding_destinations(
                req_id, 1, ["tcp://127.0.0.1:1"]
            )
            self.assertIn(req_id, rid_to_receive_endpoint)

            with patch.object(meta_registry, "discard", AsyncMock()):
                await encoder.release_request(req_id)

            self.assertTrue(state.release_requested)
            self.assertIn(req_id, encoder.req_states)
            with self.assertRaisesRegex(BadRequestError, "not active"):
                await encoder.register_embedding_destinations(
                    req_id, 1, ["tcp://127.0.0.1:2"]
                )
            self.assertEqual(rid_to_receive_endpoint[req_id], {"tcp://127.0.0.1:1"})

            with patch.object(meta_registry, "discard", AsyncMock()):
                await encoder._release_encode_ref(state)

            self.assertNotIn(req_id, rid_to_receive_endpoint)
            self.assertNotIn(req_id, rid_to_receive_count)
            self.assertNotIn(req_id, rid_to_cond)

            # a reused ID starts a new request lifecycle
            encoder.req_states[req_id] = ReqState(req_id)
            await encoder.register_embedding_destinations(
                req_id, 1, ["tcp://127.0.0.1:3"]
            )
            self.assertEqual(rid_to_receive_endpoint[req_id], {"tcp://127.0.0.1:3"})

            with patch.object(meta_registry, "discard", AsyncMock()):
                await encoder.release_request(req_id)

        asyncio.run(run())


class TestEncoderDPAbandonedRequest(CustomTestCase):
    @staticmethod
    def _make_dispatcher():
        return DPDispatcher(
            dp_size=1,
            dispatch_sockets=[object()],
            release_sockets=[object()],
            result_socket=object(),
            worker_processes=[],
        )

    def test_dispatch_timeout_notifies_worker_to_release(self):
        async def run():
            dispatcher = self._make_dispatcher()
            sent = []
            release_sent = asyncio.Event()

            async def send(socket, payload):
                message = unwrap_from_pickle(payload)
                sent.append((socket, message))
                if message.get("_dp_type") == _DP_RELEASE_AFTER_ENCODE:
                    release_sent.set()

            request = {"req_id": "timed-out", "modality": "image"}
            with (
                patch(
                    "sglang.srt.disaggregation.encoder.runtime.async_sock_send",
                    side_effect=send,
                ),
                patch(
                    "sglang.srt.disaggregation.encoder.runtime.server_module.ENCODER_REQ_TIMEOUT",
                    0.01,
                ),
            ):
                result = await dispatcher.dispatch(request)
                await asyncio.wait_for(release_sent.wait(), timeout=1)

            self.assertEqual(result["_error_type"], "TimeoutError")
            self.assertIs(sent[0][0], dispatcher.dispatch_sockets[0])
            self.assertEqual(sent[0][1], request)
            self.assertIs(sent[1][0], dispatcher.release_sockets[0])
            self.assertEqual(
                sent[1][1],
                {
                    "_dp_type": _DP_RELEASE_AFTER_ENCODE,
                    "req_id": "timed-out",
                },
            )
            self.assertEqual(dispatcher.pending_counts, [0])
            self.assertNotIn("timed-out", dispatcher.req_id_to_rank)

        asyncio.run(run())

    def test_dispatch_cancellation_notifies_worker_to_release(self):
        async def run():
            dispatcher = self._make_dispatcher()
            encode_sent = asyncio.Event()
            release_sent = asyncio.Event()

            async def send(_socket, payload):
                message = unwrap_from_pickle(payload)
                if message.get("_dp_type") == _DP_RELEASE_AFTER_ENCODE:
                    release_sent.set()
                else:
                    encode_sent.set()

            with patch(
                "sglang.srt.disaggregation.encoder.runtime.async_sock_send",
                side_effect=send,
            ):
                task = asyncio.create_task(
                    dispatcher.dispatch({"req_id": "cancelled", "modality": "image"})
                )
                await encode_sent.wait()
                task.cancel()
                with self.assertRaises(asyncio.CancelledError):
                    await task
                await asyncio.wait_for(release_sent.wait(), timeout=1)

            self.assertEqual(dispatcher.pending_counts, [0])
            self.assertNotIn("cancelled", dispatcher.req_id_to_rank)

        asyncio.run(run())

    def test_worker_marks_running_encode_abandoned(self):
        async def run():
            async def encode():
                await asyncio.Event().wait()

            encode_task = asyncio.create_task(encode())
            encoder = SimpleNamespace(
                abandon_request=AsyncMock(),
                release_request=AsyncMock(),
            )
            await _retire_abandoned_encode(encoder, encode_task, "abandoned")

            encoder.abandon_request.assert_awaited_once_with("abandoned")
            encoder.release_request.assert_not_awaited()
            encode_task.cancel()
            await asyncio.gather(encode_task, return_exceptions=True)

        asyncio.run(run())

    def test_worker_preserves_release_before_encode_task_exists(self):
        async def run():
            encoder = MMEncoder.__new__(MMEncoder)
            encoder.rank = 0
            encoder.req_states = {}
            encoder.abandoned_req_ids = set()
            encoder.delivery = SimpleNamespace(release=AsyncMock())

            await _retire_abandoned_encode(encoder, None, "abandoned")
            self.assertIn("abandoned", encoder.abandoned_req_ids)

            state = encoder._acquire_encode_ref("abandoned")
            self.assertTrue(state.release_requested)
            with patch.object(meta_registry, "discard", AsyncMock()):
                await encoder._release_encode_ref(state)

            encoder.delivery.release.assert_awaited_once_with(state)
            self.assertNotIn("abandoned", encoder.req_states)

        asyncio.run(run())

    def test_worker_release_survives_encode_failure(self):
        async def run():
            async def encode():
                raise RuntimeError("bad image")

            encode_task = asyncio.create_task(encode())
            await asyncio.sleep(0)
            encoder = SimpleNamespace(
                abandon_request=AsyncMock(),
                release_request=AsyncMock(),
            )
            await _retire_abandoned_encode(
                encoder,
                encode_task,
                "failed",
            )

            encoder.release_request.assert_awaited_once_with("failed")
            encoder.abandon_request.assert_not_awaited()
            await asyncio.gather(encode_task, return_exceptions=True)

        asyncio.run(run())


class TestMooncakeRegistration(CustomTestCase):
    def setUp(self):
        self.engine = MooncakeTransferEngine.__new__(MooncakeTransferEngine)
        self.engine.engine = unittest.mock.Mock()

    def test_register_raises_on_nonzero_status(self):
        self.engine.engine.register_memory.return_value = -1

        with self.assertRaisesRegex(RuntimeError, "registration failed.*ret=-1"):
            self.engine.register(1234, 4096)

    def test_deregister_preserves_backend_failure(self):
        backend_error = OSError("backend failed")
        self.engine.engine.unregister_memory.side_effect = backend_error

        with self.assertRaisesRegex(RuntimeError, "deregistration failed") as ctx:
            self.engine.deregister(1234)

        self.assertIs(ctx.exception.__cause__, backend_error)


if __name__ == "__main__":
    unittest.main()
