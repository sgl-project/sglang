import json
import multiprocessing as mp
import pickle
import queue
import tempfile
import time
import traceback
import unittest
from contextlib import contextmanager
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

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


def _run_child(errors, target, args):
    try:
        target(*args)
    except BaseException:
        errors.put(traceback.format_exc())
        raise


def _child_diagnostics(errors, processes):
    details = []
    while True:
        try:
            details.append(errors.get(timeout=0.1))
        except queue.Empty:
            break
    details.extend(
        f"{process.name}: pid={process.pid}, exitcode={process.exitcode}"
        for process in processes
    )
    return "\n".join(details)


@contextmanager
def _socket_pair():
    with zmq.Context() as context:
        with context.socket(zmq.PAIR) as sender, context.socket(zmq.PAIR) as receiver:
            for socket in (sender, receiver):
                socket.setsockopt(zmq.LINGER, 0)
                socket.setsockopt(zmq.RCVTIMEO, 1000)
                socket.setsockopt(zmq.SNDTIMEO, 1000)
            sender.bind("inproc://timestamp-framing")
            receiver.connect("inproc://timestamp-framing")
            yield sender, receiver


class TestTimestampFraming(CustomTestCase):
    def setUp(self):
        self.original_codec = io_struct._USE_PICKLE_IPC
        self.addCleanup(setattr, io_struct, "_USE_PICKLE_IPC", self.original_codec)

    def test_legacy_and_exact_timestamp_round_trip(self):
        message = AbortReq(rid="framing")
        timestamp = 1234567890123456
        for pickle_ipc in (False, True):
            with (
                self.subTest(pickle_ipc=pickle_ipc),
                _socket_pair() as (sender, receiver),
            ):
                io_struct._USE_PICKLE_IPC = pickle_ipc
                io_struct.sock_send(sender, message)
                self.assertEqual(
                    io_struct.sock_recv_with_timestamp(receiver), (None, message)
                )
                with patch.object(
                    io_struct.time, "monotonic_ns", return_value=timestamp
                ):
                    io_struct.sock_send_with_timestamp(sender, message)
                self.assertEqual(
                    io_struct.sock_recv_with_timestamp(receiver), (timestamp, message)
                )
                self.assertFalse(receiver.getsockopt(zmq.RCVMORE))

    def test_rejects_malformed_timestamp_frames(self):
        message = AbortReq(rid="malformed")
        for pickle_ipc in (False, True):
            io_struct._USE_PICKLE_IPC = pickle_ipc
            payload = (
                pickle.dumps(message)
                if pickle_ipc
                else io_struct.msgpack_encode(message)
            )
            cases = [
                ([bytes(size)], "Invalid timestamp frame length") for size in (0, 7, 9)
            ]
            cases.append(([bytes(8), b"extra"], "Expected at most two frames"))
            for frames, error in cases:
                with (
                    self.subTest(pickle_ipc=pickle_ipc, lengths=list(map(len, frames))),
                    _socket_pair() as (sender, receiver),
                ):
                    sender.send_multipart([payload, *frames])
                    with self.assertRaisesRegex(ValueError, error):
                        io_struct.sock_recv_with_timestamp(receiver)

    def test_encoding_failure_sends_no_partial_message(self):
        for pickle_ipc in (False, True):
            io_struct._USE_PICKLE_IPC = pickle_ipc
            encoder = "pickle.dumps" if pickle_ipc else "msgpack_encode"
            socket = MagicMock()
            with (
                self.subTest(pickle_ipc=pickle_ipc),
                patch(
                    f"sglang.srt.managers.io_struct.{encoder}",
                    side_effect=ValueError("encoding failed"),
                ),
                self.assertRaisesRegex(ValueError, "encoding failed"),
            ):
                io_struct.sock_send_with_timestamp(socket, AbortReq(rid="encoding"))
            self.assertEqual(socket.mock_calls, [])


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
        errors = ctx.Queue()
        worker_ready = [ctx.Event(), ctx.Event()]
        workers = [
            ctx.Process(
                target=_run_child,
                args=(errors, _worker, (endpoint, ready, result_queue)),
            )
            for endpoint, ready in zip(worker_endpoints, worker_ready)
        ]
        router_ready = ctx.Event()
        router_start = ctx.Event()
        router_process = ctx.Process(
            target=_run_child,
            args=(
                errors,
                _router,
                (
                    router_endpoint,
                    worker_endpoints,
                    router_ready,
                    router_start,
                    observations,
                ),
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
        except Exception as exc:
            raise AssertionError(
                f"Delivery subprocess failure: {exc}\n"
                + _child_diagnostics(errors, [*workers, router_process])
            ) from exc
        finally:
            if push is not None:
                push.close(0)
            context.term()
            if router_process.is_alive():
                router_process.terminate()
            if router_process.pid is not None:
                router_process.join(timeout=5)
            for worker in workers:
                if worker.pid is None:
                    continue
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
        errors = ctx.Queue()
        ready = ctx.Event()
        sink = ctx.Process(
            target=_run_child,
            args=(errors, _frame_sink, (endpoint, count, ready, result_queue)),
        )
        context = None
        push = None
        original_monotonic_ns = io_struct.time.monotonic_ns
        samples_us = []
        try:
            sink.start()
            assert ready.wait(5), "sink did not bind"
            context = zmq.Context()
            push = context.socket(zmq.PUSH)
            push.connect(endpoint)
            time.sleep(0.05)
            sender = SenderWrapper(push, timestamped=timestamped)
            messages = [AbortReq(rid=f"overhead-{i}") for i in range(count)]
            if not timestamped:

                def fail_if_clock_read():
                    raise AssertionError("metrics-off path read the timestamp clock")

                io_struct.time.monotonic_ns = fail_if_clock_read
            for message in messages:
                start = time.perf_counter_ns()
                sender.send_output(message)
                samples_us.append((time.perf_counter_ns() - start) / 1e3)
            frame_counts = result_queue.get(timeout=8)
        except Exception as exc:
            raise AssertionError(
                f"Frame sink subprocess failure: {exc}\n"
                + _child_diagnostics(errors, [sink])
            ) from exc
        finally:
            io_struct.time.monotonic_ns = original_monotonic_ns
            if push is not None:
                push.close(0)
            if context is not None:
                context.term()
            if sink.pid is not None:
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
    def test_worker_startup_failure_preserves_traceback(self):
        with patch(
            f"{__name__}._worker", side_effect=RuntimeError("worker startup failed")
        ):
            with self.assertRaisesRegex(
                AssertionError, "RuntimeError: worker startup failed"
            ) as error:
                _delivery_case(io_struct._USE_PICKLE_IPC)
        self.assertIn("Traceback", str(error.exception))
        self.assertIn("exitcode=1", str(error.exception))

    def test_sink_startup_failure_preserves_traceback(self):
        with patch(
            f"{__name__}._frame_sink", side_effect=RuntimeError("sink startup failed")
        ):
            with self.assertRaisesRegex(
                AssertionError, "RuntimeError: sink startup failed"
            ) as error:
                _overhead_case(False)
        self.assertIn("Traceback", str(error.exception))
        self.assertIn("exitcode=1", str(error.exception))

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
