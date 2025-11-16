import multiprocessing as mp
import os
import subprocess
import time
import unittest
from dataclasses import dataclass
from typing import Any, Dict, Optional

import requests
import zmq

from sglang import Engine
from sglang.srt.tracing.trace import *
from sglang.srt.utils import get_zmq_socket, kill_process_tree
from sglang.test.test_utils import (
    DEFAULT_SMALL_MODEL_NAME_FOR_TEST,
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
)


@dataclass
class Req:
    rid: int
    trace_context: Optional[Dict[str, Any]] = None


class TestTrace(CustomTestCase):
    def __launch_otel_jaeger(self):
        cmd = [
            "docker",
            "compose",
            "-f",
            "../../examples/monitoring/tracing_compose.yaml",
            "up",
            "-d",
        ]
        proc = subprocess.run(cmd)

        if proc.returncode != 0:
            print("launch opentelemetry collector and jaeger docker err")
            return False
        return True

    def __stop_otel_jaeger(self):
        cmd = [
            "docker",
            "compose",
            "-f",
            "../../examples/monitoring/tracing_compose.yaml",
            "down",
        ]
        proc = subprocess.run(cmd)

        if proc.returncode != 0:
            print("stop opentelemetry collector and jaeger docker err")
            return False
        return True

    def __clear_trace_file(self):
        try:
            os.remove("/tmp/otel_trace.json")
        except:
            pass

    def test_trace_enable(self):
        self.__clear_trace_file()
        assert self.__launch_otel_jaeger()

        process = popen_launch_server(
            DEFAULT_SMALL_MODEL_NAME_FOR_TEST,
            DEFAULT_URL_FOR_TEST,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=["--enable-trace", "--otlp-traces-endpoint", "0.0.0.0:4317"],
        )

        try:
            # Make some requests to generate trace data
            response = requests.get(f"{DEFAULT_URL_FOR_TEST}/health_generate")
            self.assertEqual(response.status_code, 200)

            response = requests.post(
                f"{DEFAULT_URL_FOR_TEST}/generate",
                json={
                    "text": "The capital of France is",
                    "sampling_params": {
                        "temperature": 0,
                        "max_new_tokens": 32,
                    },
                    "stream": True,
                },
                stream=True,
            )
            for _ in response.iter_lines(decode_unicode=False):
                pass

            # sleep for a few seconds to wait for opentelemetry collector to asynchronously export data to file.
            time.sleep(10)

            # check trace file
            assert os.path.isfile("/tmp/otel_trace.json"), "trace file not exist"
            assert os.path.getsize("/tmp/otel_trace.json") > 0, "trace file is empty"

        finally:
            kill_process_tree(process.pid)
            assert self.__stop_otel_jaeger()

    def test_trace_engine_enable(self):
        self.__clear_trace_file()
        assert self.__launch_otel_jaeger()

        prompt = "Today is a sunny day and I like"
        model_path = DEFAULT_SMALL_MODEL_NAME_FOR_TEST

        sampling_params = {"temperature": 0, "max_new_tokens": 8}

        engine = Engine(
            model_path=model_path,
            random_seed=42,
            enable_trace=True,
            otlp_traces_endpoint="localhost:4317",
        )

        try:
            engine.generate(prompt, sampling_params)

            # sleep for a few seconds to wait for opentelemetry collector to asynchronously export data to file.
            time.sleep(10)

            # check trace file
            assert os.path.isfile("/tmp/otel_trace.json"), "trace file not exist"
            assert os.path.getsize("/tmp/otel_trace.json") > 0, "trace file is empty"
        finally:
            engine.shutdown()
            assert self.__stop_otel_jaeger()

    def test_trace_engine_encode(self):
        self.__clear_trace_file()
        assert self.__launch_otel_jaeger()

        prompt = "Today is a sunny day and I like"
        model_path = "Qwen/Qwen2-7B"

        engine = Engine(
            model_path=model_path,
            random_seed=42,
            enable_trace=True,
            otlp_traces_endpoint="localhost:4317",
            is_embedding=True,
        )

        try:
            engine.encode(prompt)

            # sleep for a few seconds to wait for opentelemetry collector to asynchronously export data to file.
            time.sleep(10)

            # check trace file
            assert os.path.isfile("/tmp/otel_trace.json"), "trace file not exist"
            assert os.path.getsize("/tmp/otel_trace.json") > 0, "trace file is empty"
        finally:
            engine.shutdown()
            assert self.__stop_otel_jaeger()

    def test_slice_trace_simple(self):
        self.__clear_trace_file()
        assert self.__launch_otel_jaeger()
        try:
            process_tracing_init("0.0.0.0:4317", "test")
            trace_set_thread_info("Test")
            trace_req_start(0)
            trace_slice_start("test slice", 0)
            time.sleep(1)
            trace_slice_end("test slice", 0)
            trace_req_finish(0)

            # sleep for a few seconds to wait for opentelemetry collector to asynchronously export data to file.
            time.sleep(10)
            # check trace file
            assert os.path.isfile("/tmp/otel_trace.json"), "trace file not exist"
            assert os.path.getsize("/tmp/otel_trace.json") > 0, "trace file is empty"
        finally:
            assert self.__stop_otel_jaeger()

    def test_slice_trace_complex(self):
        self.__clear_trace_file()
        assert self.__launch_otel_jaeger()
        try:
            process_tracing_init("0.0.0.0:4317", "test")
            trace_set_thread_info("Test")
            trace_req_start(0)
            trace_slice_start("", 0, anonymous=True)
            time.sleep(1)
            trace_slice_end("slice A", 0, auto_next_anon=True)
            time.sleep(1)
            trace_slice_end("slice B", 0, auto_next_anon=True)
            time.sleep(1)
            trace_slice_end("slice C", 0, thread_finish_flag=True)
            trace_req_finish(0)

            # sleep for a few seconds to wait for opentelemetry collector to asynchronously export data to file.
            time.sleep(10)
            # check trace file
            assert os.path.isfile("/tmp/otel_trace.json"), "trace file not exist"
            assert os.path.getsize("/tmp/otel_trace.json") > 0, "trace file is empty"
        finally:
            assert self.__stop_otel_jaeger()

    def test_trace_context_propagete(self):
        def __process_work():
            process_tracing_init("0.0.0.0:4317", "test")
            trace_set_thread_info("Sub Process")

            context = zmq.Context(2)
            recv_from_main = get_zmq_socket(
                context, zmq.PULL, "ipc:///tmp/zmq_test.ipc", True
            )

            try:
                req = recv_from_main.recv_pyobj()
                trace_set_proc_propagate_context(req.rid, req.trace_context)
                trace_slice_start("work", req.rid)
                time.sleep(1)
                trace_slice_end("work", req.rid, thread_finish_flag=True)
            finally:
                recv_from_main.close()
                context.term()

        self.__clear_trace_file()
        assert self.__launch_otel_jaeger()

        context = zmq.Context(2)
        send_to_subproc = get_zmq_socket(
            context, zmq.PUSH, "ipc:///tmp/zmq_test.ipc", False
        )
        try:
            process_tracing_init("0.0.0.0:4317", "test")
            trace_set_thread_info("Main Process")

            subproc = mp.Process(target=__process_work)
            subproc.start()

            # sleep for a few second to ensure subprocess init
            time.sleep(1)

            req = Req(rid=0)
            trace_req_start(req.rid)
            trace_slice_start("dispatch", req.rid)
            time.sleep(1)
            req.trace_context = trace_get_proc_propagate_context(req.rid)
            send_to_subproc.send_pyobj(req)
            trace_slice_end("dispatch", req.rid)

            subproc.join()
            trace_req_finish(req.rid)

            # sleep for a few seconds to wait for opentelemetry collector to asynchronously export data to file.
            time.sleep(10)
            # check trace file
            assert os.path.isfile("/tmp/otel_trace.json"), "trace file not exist"
            assert os.path.getsize("/tmp/otel_trace.json") > 0, "trace file is empty"

        finally:
            send_to_subproc.close()
            context.term()
            assert self.__stop_otel_jaeger()


if __name__ == "__main__":
    unittest.main()
