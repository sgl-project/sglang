import json
import multiprocessing as mp
import queue
import tempfile
import time
import unittest
from types import SimpleNamespace

import zmq
from prometheus_client import CollectorRegistry, Histogram

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase, maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()

from sglang.srt.managers import io_struct
from sglang.srt.managers.io_struct import (
    AbortReq,
    BatchEmbeddingOutput,
    ConfigureLoggingReq,
    sock_recv,
)
from sglang.srt.managers.multi_tokenizer_mixin import MultiDetokenizerRouter
from sglang.srt.managers.scheduler_components.output_sender import SenderWrapper

register_cpu_ci(est_time=35, suite="base-a-test-cpu")


class _RecordingHistogram:
    def __init__(self, result_queue):
        self.result_queue = result_queue
        self.registry = CollectorRegistry()
        self.histogram = Histogram(
            "sglang:detokenizer_router_delivery_seconds",
            "delivery test histogram",
            buckets=(
                0.0001,
                0.0005,
                0.001,
                0.005,
                0.01,
                0.05,
                0.1,
                0.5,
                1.0,
                5.0,
                10.0,
                30.0,
                60.0,
                120.0,
            ),
            registry=self.registry,
        )

    def observe(self, value):
        self.histogram.observe(value)
        self.result_queue.put(
            {
                "value": float(value),
                "count": self.registry.get_sample_value(
                    "sglang:detokenizer_router_delivery_seconds_count"
                ),
                "sum": self.registry.get_sample_value(
                    "sglang:detokenizer_router_delivery_seconds_sum"
                ),
            }
        )


def _worker(endpoint, ready, result_queue):
    context = zmq.Context()
    pull = context.socket(zmq.PULL)
    pull.bind(endpoint)
    ready.set()
    poller = zmq.Poller()
    poller.register(pull, zmq.POLLIN)
    objects = []
    deadline = time.monotonic() + 5.0
    while time.monotonic() < deadline:
        if not poller.poll(100):
            continue
        objects.append(sock_recv(pull))
    result_queue.put(
        {
            "count": len(objects),
            "types": [type(obj).__name__ for obj in objects],
        }
    )
    pull.close(0)
    context.term()


def _router(endpoint, worker_endpoints, ready, start_event, observations):
    router = MultiDetokenizerRouter(
        worker_endpoints,
        SimpleNamespace(detokenizer_ipc_name=endpoint),
        enable_metrics=True,
    )
    router.delivery_seconds = _RecordingHistogram(observations)
    ready.set()
    assert start_event.wait(5), "producer did not send"
    # Hold the receive side closed after the producer has sent all messages.
    time.sleep(0.20)
    router.event_loop()


def _frame_sink(endpoint, count, ready, result_queue):
    context = zmq.Context()
    pull = context.socket(zmq.PULL)
    pull.bind(endpoint)
    ready.set()
    result_queue.put([len(pull.recv_multipart()) for _ in range(count)])
    pull.close(0)
    context.term()


def _delivery_case(pickle_ipc):
    io_struct._USE_PICKLE_IPC = pickle_ipc
    ctx = mp.get_context("fork")
    with tempfile.TemporaryDirectory(prefix="sglang-router-delivery-") as tmp:
        router_endpoint = f"ipc://{tmp}/router.sock"
        worker_endpoints = [
            f"ipc://{tmp}/worker-a.sock",
            f"ipc://{tmp}/worker-b.sock",
        ]
        result_queue = ctx.Queue()
        observations = ctx.Queue()
        worker_ready = [ctx.Event(), ctx.Event()]
        workers = [
            ctx.Process(target=_worker, args=(endpoint, ready, result_queue))
            for endpoint, ready in zip(worker_endpoints, worker_ready)
        ]
        router_ready = ctx.Event()
        router_start = ctx.Event()
        router_process = ctx.Process(
            target=_router,
            args=(
                router_endpoint,
                worker_endpoints,
                router_ready,
                router_start,
                observations,
            ),
        )
        context = zmq.Context()
        push = None
        try:
            for worker in workers:
                worker.start()
            for ready in worker_ready:
                assert ready.wait(5), "worker did not bind"
            router_process.start()
            assert router_ready.wait(5), "router did not bind"

            push = context.socket(zmq.PUSH)
            push.connect(router_endpoint)
            time.sleep(0.10)
            sender = SenderWrapper(push, timestamped=True)
            messages = [
                AbortReq(rid="delivery-a", http_worker_ipc="worker-a"),
                ConfigureLoggingReq(log_level="debug"),
                BatchEmbeddingOutput(
                    rids=[],
                    http_worker_ipcs=[],
                    finished_reasons=[],
                    embeddings=[],
                    prompt_tokens=[],
                    cached_tokens=[],
                    placeholder_tokens_idx=[],
                    placeholder_tokens_val=[],
                ),
                AbortReq(rid="delivery-b", http_worker_ipc="worker-b"),
            ]
            for message in messages:
                sender.send_output(message)
            SenderWrapper(push).send_output(
                AbortReq(rid="legacy", http_worker_ipc="worker-a")
            )
            router_start.set()

            worker_results = [result_queue.get(timeout=8) for _ in workers]
            samples = [observations.get(timeout=8) for _ in messages]
            try:
                observations.get(timeout=0.5)
            except queue.Empty:
                pass
            else:
                raise AssertionError("legacy single-frame message was observed")
            return {"samples": samples, "worker_results": worker_results}
        finally:
            if push is not None:
                push.close(0)
            context.term()
            if router_process.is_alive():
                router_process.terminate()
            router_process.join(timeout=5)
            for worker in workers:
                worker.join(timeout=5)
                if worker.is_alive():
                    worker.terminate()
                    worker.join()


def _overhead_case(timestamped):
    ctx = mp.get_context("fork")
    with tempfile.TemporaryDirectory(prefix="sglang-router-overhead-") as tmp:
        endpoint = f"ipc://{tmp}/sink.sock"
        count = 128
        result_queue = ctx.Queue()
        ready = ctx.Event()
        sink = ctx.Process(
            target=_frame_sink, args=(endpoint, count, ready, result_queue)
        )
        sink.start()
        assert ready.wait(5), "sink did not bind"
        context = zmq.Context()
        push = context.socket(zmq.PUSH)
        push.connect(endpoint)
        time.sleep(0.05)
        sender = SenderWrapper(push, timestamped=timestamped)
        messages = [AbortReq(rid=f"overhead-{i}") for i in range(count)]
        original_monotonic_ns = io_struct.time.monotonic_ns
        if not timestamped:

            def fail_if_clock_read():
                raise AssertionError("metrics-off path read the timestamp clock")

            io_struct.time.monotonic_ns = fail_if_clock_read
        samples_us = []
        try:
            for message in messages:
                start = time.perf_counter_ns()
                sender.send_output(message)
                samples_us.append((time.perf_counter_ns() - start) / 1e3)
            frame_counts = result_queue.get(timeout=8)
        finally:
            io_struct.time.monotonic_ns = original_monotonic_ns
            push.close(0)
            context.term()
            sink.join(timeout=5)
            if sink.is_alive():
                sink.terminate()
                sink.join()
        return {
            "frame_counts": frame_counts,
            "samples_us": samples_us,
            "mean_us": sum(samples_us) / len(samples_us),
        }


class TestRouterDeliveryIntegration(CustomTestCase):
    def test_real_zmq_delivery_backlog_and_both_codecs(self):
        original_codec = io_struct._USE_PICKLE_IPC
        try:
            for pickle_ipc in (False, True):
                with self.subTest(pickle_ipc=pickle_ipc):
                    delivery = _delivery_case(pickle_ipc)
                    overhead_off = _overhead_case(False)
                    overhead_on = _overhead_case(True)
                    self.assertEqual(len(delivery["samples"]), 4)
                    self.assertEqual(delivery["samples"][-1]["count"], 4.0)
                    self.assertGreater(delivery["samples"][-1]["sum"], 0.15)
                    self.assertGreater(
                        max(sample["value"] for sample in delivery["samples"]),
                        0.15,
                    )
                    self.assertEqual(
                        sum(item["count"] for item in delivery["worker_results"]),
                        7,
                    )
                    self.assertEqual(
                        sorted(
                            typ
                            for item in delivery["worker_results"]
                            for typ in item["types"]
                        ),
                        sorted(
                            [
                                "AbortReq",
                                "AbortReq",
                                "AbortReq",
                                "ConfigureLoggingReq",
                                "ConfigureLoggingReq",
                                "BatchEmbeddingOutput",
                                "BatchEmbeddingOutput",
                            ]
                        ),
                    )
                    self.assertTrue(all(x == 1 for x in overhead_off["frame_counts"]))
                    self.assertTrue(all(x == 2 for x in overhead_on["frame_counts"]))
                    print(
                        json.dumps(
                            {
                                "pickle_ipc": pickle_ipc,
                                "delivery": delivery,
                                "overhead_off": overhead_off,
                                "overhead_on": overhead_on,
                            },
                            sort_keys=True,
                        )
                    )
        finally:
            io_struct._USE_PICKLE_IPC = original_codec


if __name__ == "__main__":
    unittest.main()
