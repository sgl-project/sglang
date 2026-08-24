import asyncio
import pickle
import threading
import time
import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import numpy as np
import torch

from sglang.srt.disaggregation.encoder.preprocessor import EncoderPreprocessor
from sglang.srt.disaggregation.encoder.receiver import EmbeddingData
from sglang.srt.disaggregation.encoder.runtime import execute_encode_pipeline
from sglang.srt.disaggregation.encoder.server import (
    EncoderDelivery,
    InternalError,
    MMEncoder,
    MMError,
    MooncakeDelivery,
    PreprocessHandle,
    PreprocessWorker,
    ReqState,
    SendDestination,
    ZmqDelivery,
    meta_registry,
    rid_to_cond,
    rid_to_receive_count,
    rid_to_receive_endpoint,
)
from sglang.srt.managers.schedule_batch import Modality
from sglang.srt.utils.common import safe_pickle_loads
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


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
    def test_contract_has_two_direct_implementations(self):
        self.assertEqual(EncoderDelivery.__abstractmethods__, {"send", "release"})
        self.assertEqual(
            set(EncoderDelivery.__subclasses__()),
            {
                MooncakeDelivery,
                ZmqDelivery,
            },
        )

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

    def test_prepare_lane_advances_while_main_event_loop_is_blocked(self):
        async def run():
            encoder = MMEncoder.__new__(MMEncoder)
            encoder.rank = 1
            encoder.use_mooncake = False
            encoder.mm_global_cache = None
            encoder.preprocess_worker = PreprocessWorker(encoder)
            advanced = threading.Event()

            async def prepare_context(*_args, **_kwargs):
                await asyncio.sleep(0.01)
                advanced.set()
                return "prepared-context"

            encoder._prepare_encode_context = prepare_context
            try:
                await encoder.preprocess_worker.submit(
                    7, [{"req_id": "req"}], Modality.IMAGE
                )
                # Simulate the synchronous ViT forward occupying the main
                # encoder loop. The independent prepare loop must still run.
                time.sleep(0.05)
                self.assertTrue(advanced.is_set())
                await encoder.preprocess_worker.wait_ready(7)
                requests, modality, handle = encoder.preprocess_worker.take(7)
                self.assertEqual(await handle.context_future, "prepared-context")
                self.assertEqual(requests, [{"req_id": "req"}])
                self.assertEqual(modality, Modality.IMAGE)
                self.assertEqual(handle.states, [None])
            finally:
                encoder.preprocess_worker.shutdown()

        asyncio.run(run())

    def test_remote_preprocess_failure_is_raised_before_model_forward(self):
        async def run():
            encoder = MMEncoder.__new__(MMEncoder)
            encoder.rank = 0
            future = asyncio.get_running_loop().create_future()
            future.set_result("local-context")
            encoder.preprocess_worker = PreprocessWorker(encoder)
            handle = PreprocessHandle(states=[], context_future=future)
            tp_group = SimpleNamespace(
                world_size=2,
                all_gather_object=lambda _local: [
                    None,
                    (1, "owner preprocessing failed", 400),
                ],
            )

            with patch(
                "sglang.srt.disaggregation.encoder.server.get_tp_group",
                return_value=tp_group,
            ):
                with self.assertRaisesRegex(
                    MMError, "Rank 1 failed to prepare encoder inputs"
                ) as error:
                    await encoder.preprocess_worker.resolve(handle)

            self.assertEqual(error.exception.code, 400)
            encoder.preprocess_worker.shutdown()

        asyncio.run(run())

    def test_batch_encode_reuses_prepared_context(self):
        async def run():
            encoder = MMEncoder.__new__(MMEncoder)
            encoder.rank = 1
            encoder.use_mooncake = False
            encoder.mm_global_cache = None
            encoder.profiler = None
            encoder._prepare_encode_context = AsyncMock(
                side_effect=AssertionError("preprocess must not run twice")
            )
            encoder._publish_preprocess_metadata = AsyncMock()
            encoder._compute_embedding = AsyncMock(return_value="embedding")
            encoder._stage_embeddings = lambda *args, **kwargs: ["result"]
            encoder.preprocess_worker = PreprocessWorker(encoder)
            future = asyncio.get_running_loop().create_future()
            future.set_result("prepared-context")
            handle = PreprocessHandle(states=[], context_future=future)

            with patch(
                "sglang.srt.disaggregation.encoder.server.get_tp_group",
                return_value=SimpleNamespace(world_size=1),
            ):
                result = await encoder.batch_encode(
                    [{"req_id": "req"}],
                    Modality.IMAGE,
                    preprocess_handle=handle,
                )

            self.assertEqual(result, ["result"])
            encoder._prepare_encode_context.assert_not_awaited()
            encoder._publish_preprocess_metadata.assert_awaited_once_with(
                "prepared-context", [{"req_id": "req"}]
            )
            encoder._compute_embedding.assert_awaited_once_with(
                "prepared-context", keep_on_gpu=False
            )
            self.assertTrue(handle.released)
            encoder.preprocess_worker.shutdown()

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
        encoder._stage_embedding = lambda mm_data: events.append("ready")
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

    def test_send_waits_for_embedding_published_by_encode(self):
        async def run():
            encoder = MMEncoder.__new__(MMEncoder)
            encoder.rank = 0
            encoder.req_states = {}
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

    def test_delivery_fence_waits_for_late_decoder_tp_sends(self):
        async def run():
            encoder = MMEncoder.__new__(MMEncoder)
            encoder.use_mooncake = True
            encoder.send_timeout = 1
            encoder.req_states = {}
            encoder.delivery = SimpleNamespace(
                send=AsyncMock(),
                release=AsyncMock(),
            )
            state = ReqState(
                "req",
                EmbeddingData(
                    "req",
                    1,
                    0,
                    None,
                    Modality.IMAGE,
                    embedding=torch.ones((1, 1)),
                ),
            )
            state.embedding_ready.set()
            encoder.req_states[state.req_id] = state

            fence = asyncio.create_task(encoder.wait_for_batch_delivery([state.req_id]))
            await asyncio.sleep(0)
            self.assertFalse(fence.done())

            await encoder.send_to_destination(
                state,
                SendDestination("127.0.0.1:1"),
                expected_send_count=2,
            )
            self.assertEqual(state.finished_sends, 1)
            self.assertFalse(fence.done())

            await encoder.send_to_destination(
                state,
                SendDestination("127.0.0.1:2"),
                expected_send_count=2,
            )
            await asyncio.wait_for(fence, timeout=1)
            self.assertEqual(state.expected_sends, 2)
            self.assertEqual(state.finished_sends, 2)

        asyncio.run(run())

    def test_send_fence_tracks_real_transfer_completion(self):
        async def run():
            encoder = MMEncoder.__new__(MMEncoder)
            encoder.use_mooncake = True
            encoder.send_timeout = 1
            encoder.req_states = {}
            transfer_started = asyncio.Event()
            finish_transfer = asyncio.Event()

            async def send(_state, _destination):
                transfer_started.set()
                await finish_transfer.wait()

            encoder.delivery = SimpleNamespace(send=send, release=AsyncMock())
            state = ReqState("req")
            encoder.req_states[state.req_id] = state

            send_task = asyncio.create_task(
                encoder.send_to_destination(
                    state,
                    SendDestination("127.0.0.1:1"),
                )
            )
            await transfer_started.wait()
            fence = asyncio.create_task(encoder.wait_for_batch_delivery([state.req_id]))
            send_task.cancel()
            await asyncio.sleep(0)

            self.assertFalse(send_task.done())
            self.assertFalse(fence.done())
            self.assertEqual(state.active_sends, 1)

            finish_transfer.set()
            with self.assertRaises(asyncio.CancelledError):
                await send_task
            await asyncio.wait_for(fence, timeout=1)
            self.assertEqual(state.active_sends, 0)
            self.assertEqual(state.finished_sends, 1)

            async def fail_send(_state, _destination):
                raise RuntimeError("transfer failed")

            encoder.delivery.send = fail_send
            failed_state = ReqState("failed")
            encoder.req_states[failed_state.req_id] = failed_state
            with self.assertRaisesRegex(RuntimeError, "transfer failed"):
                await encoder.send_to_destination(
                    failed_state, SendDestination("127.0.0.1:2")
                )
            self.assertEqual(failed_state.active_sends, 0)
            self.assertEqual(failed_state.finished_sends, 0)

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


if __name__ == "__main__":
    unittest.main()
