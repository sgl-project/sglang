import multiprocessing as mp
import queue
import tempfile
import time
import traceback
import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import zmq

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase, maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()

from sglang.srt.managers import delivery_clock, io_struct
from sglang.srt.managers.io_struct import (
    AbortReq,
    BatchEmbeddingOutput,
    ConfigureLoggingReq,
    sock_recv,
)
from sglang.srt.managers.multi_tokenizer_mixin import MultiDetokenizerRouter
from sglang.srt.managers.scheduler_components.output_sender import SenderWrapper
from sglang.test.router_delivery_test_utils import router_for_socket, socket_pair

register_cpu_ci(est_time=35, suite="base-a-test-cpu")

_TEST_CLOCK_DOMAIN = bytes(range(24))
_WORKER_DONE = "router-delivery-test-done"


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


def _join_successful_child(process):
    process.join(timeout=5)
    if process.is_alive():
        raise TimeoutError(f"{process.name} did not exit within 5 seconds")
    if process.exitcode != 0:
        raise RuntimeError(f"{process.name} failed with exitcode={process.exitcode}")


def _stop_processes(processes):
    for process in processes:
        if process.pid is not None and process.is_alive():
            process.terminate()
    for process in processes:
        if process.pid is None:
            continue
        process.join(timeout=5)
        if process.is_alive():
            process.kill()
            process.join(timeout=5)
        if process.is_alive():
            raise RuntimeError(f"Failed to stop {process.name}")


class TestProductionHistogram(CustomTestCase):
    def test_registered_histogram_includes_decode_time(self):
        original_codec = io_struct._USE_PICKLE_IPC
        self.addCleanup(setattr, io_struct, "_USE_PICKLE_IPC", original_codec)
        io_struct._USE_PICKLE_IPC = False
        clock_ns = [10_000_000_000]
        original_decode = io_struct.msgpack_decode

        def decode_with_delay(data):
            decoded = original_decode(data)
            clock_ns[0] += 2_000_000_000
            return decoded

        with (
            socket_pair() as (sender, receiver),
            router_for_socket(receiver, _TEST_CLOCK_DOMAIN) as (router, registry),
        ):
            router._send = MagicMock(side_effect=StopIteration)
            with patch.object(
                delivery_clock.time, "monotonic_ns", side_effect=lambda: clock_ns[0]
            ):
                SenderWrapper(sender, clock_domain=_TEST_CLOCK_DOMAIN).send_output(
                    AbortReq(rid="timed")
                )
            clock_ns[0] += 1_000_000_000
            with (
                patch.object(
                    io_struct, "msgpack_decode", side_effect=decode_with_delay
                ),
                patch(
                    "sglang.srt.managers.multi_tokenizer_mixin.monotonic_ns",
                    side_effect=lambda: clock_ns[0],
                ),
                self.assertRaises(StopIteration),
            ):
                router.event_loop()
            router._send.assert_called_once()
        self.assertEqual(
            registry.get_sample_value(
                "sglang:detokenizer_router_delivery_seconds_count"
            ),
            1.0,
        )
        self.assertEqual(
            registry.get_sample_value("sglang:detokenizer_router_delivery_seconds_sum"),
            3.0,
        )


class _RecordingHistogram:
    def __init__(self, result_queue):
        self.result_queue = result_queue
        self.count = 0
        self.total = 0.0

    def observe(self, value):
        self.count += 1
        self.total += value
        self.result_queue.put({"value": value, "count": self.count, "sum": self.total})


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
        obj = sock_recv(pull)
        if isinstance(obj, ConfigureLoggingReq) and obj.log_level == _WORKER_DONE:
            break
        objects.append(obj)
    else:
        raise TimeoutError("worker did not receive the completion marker")
    result_queue.put(
        {
            "endpoint": endpoint,
            "count": len(objects),
            "types": [type(obj).__name__ for obj in objects],
            "rids": [obj.rid for obj in objects if isinstance(obj, AbortReq)],
        }
    )
    pull.close(0)
    context.term()


def _make_router(endpoint, worker_endpoints, observations):
    with patch(
        "sglang.srt.managers.multi_tokenizer_mixin.read_clock_domain",
        return_value=_TEST_CLOCK_DOMAIN,
    ):
        router = MultiDetokenizerRouter(
            worker_endpoints,
            SimpleNamespace(detokenizer_ipc_name=endpoint),
            enable_metrics=True,
        )
    router.delivery_seconds = _RecordingHistogram(observations)
    return router


def _router(endpoint, worker_endpoints, ready, start_event, observations):
    router = _make_router(endpoint, worker_endpoints, observations)
    ready.set()
    assert start_event.wait(5), "producer did not send"
    # Hold the receive side closed after the producer has sent all messages.
    time.sleep(0.20)
    router.event_loop()


def _frame_sink(endpoint, ready, result_queue):
    context = zmq.Context()
    pull = context.socket(zmq.PULL)
    pull.bind(endpoint)
    ready.set()
    result_queue.put(len(pull.recv_multipart()))
    pull.close(0)
    context.term()


def _delivery_case(pickle_ipc, expected_router_failure=None):
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
            sender = SenderWrapper(push, clock_domain=_TEST_CLOCK_DOMAIN)
            messages = [
                AbortReq(rid="delivery-a", http_worker_ipc="worker-a"),
                ConfigureLoggingReq(log_level="debug"),
                BatchEmbeddingOutput(
                    rids=[],
                    http_worker_ipcs=[],
                    finished_reasons=[],
                    embeddings=[],
                    request_stats=[],
                    placeholder_tokens_idx=[],
                    placeholder_tokens_val=[],
                ),
                AbortReq(rid="delivery-b", http_worker_ipc="worker-d"),
            ]
            for message in messages:
                sender.send_output(message)
            SenderWrapper(push).send_output(
                AbortReq(rid="legacy", http_worker_ipc="worker-a")
            )
            # The broadcast marker follows every payload on each router pipe,
            # so workers can stop without a fixed five-second collection window.
            SenderWrapper(push).send_output(ConfigureLoggingReq(log_level=_WORKER_DONE))
            router_start.set()

            worker_results = [result_queue.get(timeout=8) for _ in workers]
            for worker in workers:
                _join_successful_child(worker)
            samples = [observations.get(timeout=8) for _ in messages]
            try:
                observations.get(timeout=0.5)
            except queue.Empty:
                pass
            else:
                raise AssertionError("legacy single-frame message was observed")
            # A failed child may still be alive while its Queue feeder flushes.
            # Drain a reported error before terminating the long-running router.
            if expected_router_failure is not None and not expected_router_failure.wait(
                5
            ):
                raise TimeoutError("router did not reach the injected failure")
            try:
                timeout = 5 if expected_router_failure is not None else 0.1
                child_error = errors.get(timeout=timeout)
            except queue.Empty:
                if expected_router_failure is not None:
                    raise RuntimeError("router failure was not reported")
            else:
                raise RuntimeError(f"Child process reported an error:\n{child_error}")
            if not router_process.is_alive():
                raise RuntimeError(
                    f"router exited unexpectedly with exitcode={router_process.exitcode}"
                )
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
            _stop_processes([router_process, *workers])
            for result in (result_queue, observations, errors):
                result.close()
                result.join_thread()


def _frame_count_case(timestamped):
    ctx = mp.get_context("fork")
    with tempfile.TemporaryDirectory(prefix="sglang-router-frames-") as tmp:
        endpoint = f"ipc://{tmp}/sink.sock"
        result_queue = ctx.Queue()
        errors = ctx.Queue()
        ready = ctx.Event()
        sink = ctx.Process(
            target=_run_child,
            args=(errors, _frame_sink, (endpoint, ready, result_queue)),
        )
        context = None
        push = None
        original_monotonic_ns = delivery_clock.time.monotonic_ns
        try:
            sink.start()
            assert ready.wait(5), "sink did not bind"
            context = zmq.Context()
            push = context.socket(zmq.PUSH)
            push.connect(endpoint)
            sender = SenderWrapper(
                push,
                clock_domain=_TEST_CLOCK_DOMAIN if timestamped else None,
            )
            if not timestamped:

                def fail_if_clock_read():
                    raise AssertionError("metrics-off path read the timestamp clock")

                delivery_clock.time.monotonic_ns = fail_if_clock_read
            sender.send_output(AbortReq(rid="frame-count"))
            frame_counts = result_queue.get(timeout=8)
            _join_successful_child(sink)
            return frame_counts
        except Exception as exc:
            raise AssertionError(
                f"Frame sink subprocess failure: {exc}\n"
                + _child_diagnostics(errors, [sink])
            ) from exc
        finally:
            delivery_clock.time.monotonic_ns = original_monotonic_ns
            if push is not None:
                push.close(0)
            if context is not None:
                context.term()
            _stop_processes([sink])
            for result in (result_queue, errors):
                result.close()
                result.join_thread()


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
                _frame_count_case(False)
        self.assertIn("Traceback", str(error.exception))
        self.assertIn("exitcode=1", str(error.exception))

    def test_router_late_failure_preserves_traceback(self):
        failure_started = mp.get_context("fork").Event()

        def fail_after_fanout(endpoint, worker_endpoints, ready, start, observations):
            router = _make_router(endpoint, worker_endpoints, observations)
            original_send = router._send
            marker_count = [0]

            def send_then_fail(target, message):
                original_send(target, message)
                if (
                    isinstance(message, ConfigureLoggingReq)
                    and message.log_level == _WORKER_DONE
                ):
                    marker_count[0] += 1
                    if marker_count[0] == len(worker_endpoints):
                        failure_started.set()
                        raise RuntimeError(
                            "router failed after fanout: " + "X" * 262144
                        )

            router._send = send_then_fail
            ready.set()
            assert start.wait(5), "producer did not send"
            time.sleep(0.20)
            router.event_loop()

        with patch(f"{__name__}._router", side_effect=fail_after_fanout):
            with self.assertRaisesRegex(
                AssertionError, "RuntimeError: router failed after fanout"
            ) as error:
                _delivery_case(io_struct._USE_PICKLE_IPC, failure_started)
            self.assertIn("Child process reported an error", str(error.exception))

    def test_worker_late_failure_preserves_traceback(self):
        original_worker = _worker

        def fail_after_report(*args):
            original_worker(*args)
            raise RuntimeError("worker failed after report")

        with patch(f"{__name__}._worker", side_effect=fail_after_report):
            with self.assertRaisesRegex(
                AssertionError, "RuntimeError: worker failed after report"
            ) as error:
                _delivery_case(io_struct._USE_PICKLE_IPC)
        self.assertIn("exitcode=1", str(error.exception))

    def test_sink_late_failure_preserves_traceback(self):
        original_sink = _frame_sink

        def fail_after_report(*args):
            original_sink(*args)
            raise RuntimeError("sink failed after report")

        with patch(f"{__name__}._frame_sink", side_effect=fail_after_report):
            with self.assertRaisesRegex(
                AssertionError, "RuntimeError: sink failed after report"
            ) as error:
                _frame_count_case(False)
        self.assertIn("exitcode=1", str(error.exception))

    def test_real_zmq_delivery_backlog_and_both_codecs(self):
        original_codec = io_struct._USE_PICKLE_IPC
        try:
            for pickle_ipc in (False, True):
                with self.subTest(pickle_ipc=pickle_ipc):
                    delivery = _delivery_case(pickle_ipc)
                    frames_off = _frame_count_case(False)
                    frames_on = _frame_count_case(True)
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
                    self.assertEqual(
                        {
                            item["endpoint"].rsplit("/", 1)[-1]: item["rids"]
                            for item in delivery["worker_results"]
                        },
                        {
                            "worker-a.sock": ["legacy"],
                            "worker-b.sock": ["delivery-a", "delivery-b"],
                        },
                    )
                    self.assertEqual(frames_off, 1)
                    self.assertEqual(frames_on, 2)
        finally:
            io_struct._USE_PICKLE_IPC = original_codec


if __name__ == "__main__":
    unittest.main()
