import asyncio
import pickle
import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

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
)
from sglang.srt.disaggregation.encoder.server import (
    EncoderMetaRegistry,
    MMEncoder,
    MMError,
    meta_registry,
)
from sglang.srt.distributed.device_communicators.mooncake_transfer_engine import (
    MooncakeTransferEngine,
)
from sglang.srt.managers.io_struct import unwrap_from_pickle
from sglang.srt.managers.schedule_batch import Modality
from sglang.srt.utils.common import safe_pickle_loads
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=3, suite="base-a-test-cpu")


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
