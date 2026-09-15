import asyncio
import multiprocessing
import os
import unittest
from functools import partial
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock, patch

import numpy as np
import torch

from sglang.srt.managers.schedule_batch import (
    Modality,
    MultimodalDataItem,
    MultimodalProcessorOutput,
)
from sglang.srt.observability.mm_frontend_metrics import MultimodalFrontendMetrics
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=15, suite="base-a-test-cpu")


def _metrics(registry=None):
    from prometheus_client import Counter, Gauge, Histogram

    return MultimodalFrontendMetrics(
        labels={"model_name": "test"},
        counter_cls=partial(Counter, registry=registry),
        gauge_cls=partial(Gauge, registry=registry),
        histogram_cls=partial(Histogram, registry=registry),
    )


def _worker_metrics(connection):
    metrics = _metrics()
    with metrics.record("hash"):
        connection.send("ready")
        if not connection.poll(30):
            raise TimeoutError("parent did not release metrics worker")
        connection.recv()


class TestMultimodalFrontendMetrics(unittest.IsolatedAsyncioTestCase, CustomTestCase):
    def setUp(self):
        from prometheus_client import CollectorRegistry

        self.registry = CollectorRegistry()
        self.metrics = _metrics(self.registry)

    def sample(self, suffix, **labels):
        return self.registry.get_sample_value(
            "sglang:mm_frontend_" + suffix, {"model_name": "test", **labels}
        )

    async def test_concurrent_cancellation_and_error_balance_inflight(self):
        """One request ending must not clear another request's active stage."""
        entered = asyncio.Event()
        release = asyncio.Event()

        async def request():
            with self.metrics.record("preprocess"):
                entered.set()
                await release.wait()

        first = asyncio.create_task(request())
        await entered.wait()
        entered.clear()
        second = asyncio.create_task(request())
        await entered.wait()
        self.assertEqual(self.sample("inflight", stage="preprocess"), 2)
        first.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await first
        self.assertEqual(self.sample("inflight", stage="preprocess"), 1)
        release.set()
        await second
        with self.assertRaisesRegex(ValueError, "invalid feature"):
            with self.metrics.record("preprocess"):
                raise ValueError("invalid feature")
        self.assertEqual(self.sample("inflight", stage="preprocess"), 0)
        for outcome in ("success", "error", "cancelled"):
            self.assertEqual(
                self.sample("stage_seconds_count", stage="preprocess", outcome=outcome),
                1,
            )

    async def test_elapsed_time_includes_awaited_work(self):
        with patch(
            "sglang.srt.observability.mm_frontend_metrics.time.perf_counter",
            side_effect=[10, 13.5],
        ):
            with self.metrics.record("hash"):
                await asyncio.sleep(0)
        self.assertEqual(
            self.sample("stage_seconds_sum", stage="hash", outcome="success"), 3.5
        )

    async def test_workload_counts_logical_bytes_without_reading_device_data(self):
        inputs = MultimodalProcessorOutput(
            mm_items=[
                MultimodalDataItem(
                    modality=Modality.IMAGE,
                    feature=[
                        torch.empty(2, 3, device="meta"),
                        np.zeros((2, 4), dtype=np.uint8),
                    ],
                ),
                MultimodalDataItem(
                    modality=Modality.AUDIO,
                    precomputed_embeddings=torch.ones(5, dtype=torch.bfloat16),
                ),
            ]
        )
        self.metrics.observe_inputs(inputs)
        self.assertEqual(self.sample("items_total", modality="image"), 1)
        self.assertEqual(self.sample("feature_bytes_total", modality="image"), 32)
        self.assertEqual(self.sample("feature_bytes_total", modality="audio"), 10)

    async def test_dispatch_counts_each_multimodal_request_and_skips_text(self):
        from sglang.srt.managers.tokenizer_manager import TokenizerManager

        manager = TokenizerManager.__new__(TokenizerManager)
        manager.enable_metrics = True
        manager.metrics_collector = SimpleNamespace(mm_frontend=self.metrics)
        manager.cuda_vmm_feature_transport = SimpleNamespace(
            prepare_for_dispatch_async=AsyncMock(return_value=[]),
            cancel_for_dispatch=Mock(),
        )
        manager._dispatch_to_scheduler = Mock()
        manager._mark_state_dispatched = Mock()
        manager.encoder_dispatch_ready = {}

        def request(mm_inputs):
            return SimpleNamespace(
                rid="test",
                mm_inputs=mm_inputs,
                time_stats=Mock(),
                wrap_pickle_fields=Mock(),
            )

        with patch(
            "sglang.srt.managers.tokenizer_manager.wrap_shm_features", lambda obj: obj
        ):
            await manager._send_one_request(request(None))
            self.assertIsNone(
                self.sample("stage_seconds_count", stage="dispatch", outcome="success")
            )
            await manager._send_one_request(request(SimpleNamespace(mm_items=[])))
        await manager._send_batch_request(
            [
                request(None),
                request(SimpleNamespace(mm_items=[])),
                request(SimpleNamespace(mm_items=[])),
            ]
        )
        self.assertEqual(
            self.sample("stage_seconds_count", stage="dispatch", outcome="success"), 3
        )
        manager.enable_metrics = False
        with patch(
            "sglang.srt.observability.mm_frontend_metrics.time.perf_counter",
            side_effect=AssertionError("disabled metrics read the clock"),
        ):
            with manager._mm_frontend_stage("preprocess"):
                pass

    async def test_tokenization_observes_preprocessing_hash_and_failures(self):
        from sglang.srt.environ import envs
        from sglang.srt.managers.io_struct import GenerateReqInput
        from sglang.srt.managers.tokenizer_manager import TokenizerManager
        from sglang.srt.runtime_context import get_context

        manager = TokenizerManager.__new__(TokenizerManager)
        manager.enable_metrics = True
        manager.metrics_collector = SimpleNamespace(mm_frontend=self.metrics)
        manager.model_config = SimpleNamespace(
            hf_config=SimpleNamespace(architectures=[])
        )
        output = MultimodalProcessorOutput(
            mm_items=[
                MultimodalDataItem(modality=Modality.IMAGE, feature=torch.ones(8))
            ]
        )
        manager.mm_processor = AsyncMock(
            prefer_tokenized_input=True,
            process_mm_data_async=AsyncMock(return_value=output),
        )
        manager.max_req_input_len = 1024
        manager._validate_mm_limits = Mock()
        manager._normalize_mm_content_hashes = Mock()
        manager._validate_one_request = Mock()
        manager._create_tokenized_object = lambda *args: args[4]
        request = GenerateReqInput(input_ids=[1], image_data=["image"])
        with (
            get_context().override_server_args(language_only=False),
            envs.SGLANG_MM_PRECOMPUTE_HASH.override(True),
        ):
            result = await manager._tokenize_one_request(request)
            self.assertIs(result, output)
            self.assertEqual(
                self.sample(
                    "stage_seconds_count", stage="preprocess", outcome="success"
                ),
                1,
            )
            self.assertEqual(
                self.sample("stage_seconds_count", stage="hash", outcome="success"), 1
            )
            self.assertEqual(self.sample("feature_bytes_total", modality="image"), 32)
            manager.mm_processor.process_mm_data_async.side_effect = ValueError(
                "bad image"
            )
            with self.assertRaisesRegex(ValueError, "bad image"):
                await manager._tokenize_one_request(request)
        self.assertEqual(
            self.sample("stage_seconds_count", stage="preprocess", outcome="error"), 1
        )
        self.assertEqual(self.sample("inflight", stage="preprocess"), 0)


class TestMultimodalFrontendMultiprocess(CustomTestCase):
    def test_live_workers_sum_and_dead_worker_gauge_is_removed(self):
        """Tokenizer workers share counters but their live gauges must add up."""
        from prometheus_client import CollectorRegistry, multiprocess

        context = multiprocessing.get_context("spawn")
        with (
            TemporaryDirectory() as directory,
            patch.dict(os.environ, {"PROMETHEUS_MULTIPROC_DIR": directory}),
        ):
            pipes = [context.Pipe(), context.Pipe()]
            workers = [
                context.Process(target=_worker_metrics, args=(child,))
                for _, child in pipes
            ]
            try:
                for worker in workers:
                    worker.start()
                for parent, child in pipes:
                    child.close()
                    self.assertTrue(parent.poll(20))
                    self.assertEqual(parent.recv(), "ready")
                registry = CollectorRegistry()
                multiprocess.MultiProcessCollector(registry, path=directory)
                labels = {"model_name": "test", "stage": "hash"}
                self.assertEqual(
                    registry.get_sample_value("sglang:mm_frontend_inflight", labels), 2
                )
                workers[0].terminate()
                workers[0].join(5)
                multiprocess.mark_process_dead(workers[0].pid, path=directory)
                self.assertEqual(
                    registry.get_sample_value("sglang:mm_frontend_inflight", labels), 1
                )
                pipes[1][0].send("finish")
                workers[1].join(10)
                self.assertEqual(workers[1].exitcode, 0)
                self.assertEqual(
                    registry.get_sample_value("sglang:mm_frontend_inflight", labels), 0
                )
                self.assertEqual(
                    registry.get_sample_value(
                        "sglang:mm_frontend_stage_seconds_count",
                        {**labels, "outcome": "success"},
                    ),
                    1,
                )
            finally:
                for parent, child in pipes:
                    parent.close()
                    child.close()
                for worker in workers:
                    if worker.pid is not None:
                        worker.join(5)
                        if worker.is_alive():
                            worker.terminate()
                            worker.join(5)


if __name__ == "__main__":
    unittest.main()
