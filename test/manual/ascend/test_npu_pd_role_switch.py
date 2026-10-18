"""Ascend PD role-switch service regression (requires four idle NPU devices).

Run inside the configured NPU environment:
  python3 test/manual/ascend/test_npu_pd_role_switch.py > pd-role-switch.log 2>&1

The defaults below match the existing NPU PD tests: Qwen3-32B, TP2 per side.
Set ASCEND_RT_VISIBLE_DEVICES before running to select four idle devices.
Environment overrides remain available for other setups.
BASELINE_ONLY=1 runs fixed roles without --enable-pd-role-switch, including
its disabled-flag rejection. Worker logs and JSON evidence are printed to
standard output, so shell redirection captures the complete test log.
This test does not establish PLE, hybrid/Mamba, RDMA, or other topology support.
"""

import concurrent.futures
import io
import json
import os
import re
import shlex
import signal
import socket
import subprocess
import sys
import threading
import time
import unittest
import uuid
from pathlib import Path

import requests

from sglang.test.ascend.test_ascend_utils import QWEN3_32B_WEIGHTS_PATH

# Default launch configuration for direct manual runs.
MODEL_PATH = QWEN3_32B_WEIGHTS_PATH
TP_SIZE = 2
MEM_FRACTION_STATIC = 0.9


class TestNpuPdRoleSwitch(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = os.environ.get("SGLANG_TEST_PD_ROLE_SWITCH_MODEL", MODEL_PATH)
        cls.tp = int(os.environ.get("SGLANG_TEST_PD_ROLE_SWITCH_TP_SIZE", str(TP_SIZE)))
        cls.baseline_only = (
            os.environ.get("SGLANG_TEST_PD_ROLE_SWITCH_BASELINE_ONLY") == "1"
        )
        cls.port = int(os.environ.get("SGLANG_TEST_PD_ROLE_SWITCH_PORT", "28000"))
        cls.processes, cls.log_threads, cls.records = [], [], []
        cls.urls = [f"http://127.0.0.1:{cls.port + i}" for i in range(2)]
        cls.bootstrap_ports = [cls.port + 10, cls.port + 11]
        cls.logs = [io.StringIO() for _ in range(2)]
        cls.log_output_lock = threading.Lock()
        # Refuse to take a port held by another service.
        visible = os.environ.get("ASCEND_RT_VISIBLE_DEVICES", "").split(",")
        if (
            cls.tp < 1
            or len(visible) != 2 * cls.tp
            or any(not x.isdigit() for x in visible)
        ):
            raise ValueError(
                "ASCEND_RT_VISIBLE_DEVICES must list exactly 2 * TP_SIZE NPU devices"
            )
        print(
            f"Model: {cls.model}; TP per side: {cls.tp}; "
            f"ASCEND_RT_VISIBLE_DEVICES={','.join(visible)}",
            flush=True,
        )
        ports = [
            cls.port,
            cls.port + 1,
            *cls.bootstrap_ports,
            cls.port + 20,
            cls.port + 21,
            cls.port + 30,
        ]
        for port in ports:
            with socket.socket() as sock:
                sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                sock.bind(("0.0.0.0", port))
        try:
            for i, role in enumerate(("prefill", "decode")):
                args = [
                    sys.executable,
                    "-m",
                    "sglang.launch_server",
                    "--model-path",
                    cls.model,
                    "--host",
                    "127.0.0.1",
                    "--port",
                    str(cls.port + i),
                    "--device",
                    "npu",
                    "--attention-backend",
                    "ascend",
                    "--dtype",
                    "bfloat16",
                    "--tp-size",
                    str(cls.tp),
                    "--base-gpu-id",
                    str(i * cls.tp),
                    "--disaggregation-mode",
                    role,
                    "--disaggregation-transfer-backend",
                    "ascend",
                    "--disaggregation-bootstrap-port",
                    str(cls.bootstrap_ports[i]),
                    "--nccl-port",
                    str(cls.port + 20 + i),
                    "--page-size",
                    "128",
                    "--chunked-prefill-size",
                    "512",
                    "--mem-fraction-static",
                    str(MEM_FRACTION_STATIC),
                    "--max-total-tokens",
                    "4096",
                    "--max-running-requests",
                    "2",
                    "--cuda-graph-bs-decode",
                    "1",
                    "2",
                    "--decode-log-interval",
                    "1",
                ]
                if not cls.baseline_only:
                    args.append("--enable-pd-role-switch")
                args.extend(
                    shlex.split(
                        os.environ.get("SGLANG_TEST_PD_ROLE_SWITCH_EXTRA_ARGS", "")
                    )
                )
                env = dict(
                    os.environ,
                    ASCEND_MF_STORE_URL=f"tcp://127.0.0.1:{cls.port + 30}",
                    PYTHONUNBUFFERED="1",
                )
                print(f"Launching {role}: {shlex.join(args)}", flush=True)
                cls.records.append({"launch": args, "visible_dies": visible})
                cls.processes.append(
                    subprocess.Popen(
                        args,
                        env=env,
                        stdout=subprocess.PIPE,
                        stderr=subprocess.STDOUT,
                        text=True,
                        encoding="utf-8",
                        errors="replace",
                        bufsize=1,
                        start_new_session=True,
                    )
                )
                thread = threading.Thread(
                    target=cls._forward_log, args=(i,), daemon=True
                )
                cls.log_threads.append(thread)
                thread.start()
                cls._wait_ready(i)
        except BaseException:
            cls.tearDownClass()
            raise

    @classmethod
    def _forward_log(cls, i):
        # Keep a copy for graph assertions while streaming everything to the
        # caller's stdout; `> test.log 2>&1` captures both worker logs.
        with cls.processes[i].stdout as output:
            for line in output:
                cls.logs[i].write(line)
                with cls.log_output_lock:
                    sys.stdout.write(f"[worker {i}] {line}")
                    sys.stdout.flush()

    @classmethod
    def _wait_ready(cls, i):
        deadline = time.monotonic() + 600
        while time.monotonic() < deadline:
            if cls.processes[i].poll() is not None:
                raise RuntimeError(f"Worker {i} exited; see the worker logs above")
            try:
                if requests.get(cls.urls[i] + "/health", timeout=5).status_code == 200:
                    return
            except requests.RequestException:
                pass
            time.sleep(1)
        raise TimeoutError(
            f"Worker {i} did not become ready; see the worker logs above"
        )

    @classmethod
    def tearDownClass(cls):
        # Each worker has its own process group; never kill other launch_server
        # processes or rely on a container-wide pkill during this test.
        for process in cls.processes:
            try:
                os.killpg(process.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
        for process in cls.processes:
            try:
                process.wait(timeout=120)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait(timeout=30)
        for thread in cls.log_threads:
            thread.join(timeout=30)
        print("PD role-switch evidence:", json.dumps(cls.records, indent=2), flush=True)
        # Preserve explicit artifact export for existing launch wrappers.
        log_dir = os.environ.get("SGLANG_TEST_PD_ROLE_SWITCH_LOG_DIR")
        if log_dir:
            root = Path(log_dir)
            root.mkdir(parents=True, exist_ok=True)
            for i, log in enumerate(cls.logs):
                (root / f"worker_{i}.log").write_text(log.getvalue())
            (root / "evidence.json").write_text(json.dumps(cls.records, indent=2))

    def _state(self, i):
        response = requests.get(self.urls[i] + "/server_info", timeout=30)
        response.raise_for_status()
        states = response.json()["internal_states"]
        self.assertEqual(len(states), 1)
        return states

    def _switch(self, i, role, expected=200, **extra):
        start = time.monotonic()
        response = requests.post(
            self.urls[i] + "/pd_role_switch",
            json={"new_role": role, **extra},
            timeout=180,
        )
        self.records.append(
            {
                "switch_worker": i,
                "body": {"new_role": role, **extra},
                "status": response.status_code,
                "result": response.json(),
                "seconds": time.monotonic() - start,
            }
        )
        self.assertEqual(response.status_code, expected, response.text)
        self.assertEqual(response.json()["success"], expected == 200)
        return response.json()

    def _generate_pair(self, prefill, length):
        decode = 1 - prefill
        for url in self.urls:
            response = requests.post(url + "/flush_cache", timeout=30)
            self.assertEqual(response.status_code, 200, response.text)
        room = uuid.uuid4().int & ((1 << 63) - 1)
        request = {
            "input_ids": [100 + i % 200 for i in range(length)],
            "sampling_params": {
                "temperature": 0,
                "max_new_tokens": 16,
                "ignore_eos": True,
            },
            "return_logprob": True,
            "logprob_start_len": -1,
            "top_logprobs_num": 5,
            "bootstrap_host": "127.0.0.1",
            "bootstrap_port": self.bootstrap_ports[prefill],
            "bootstrap_room": room,
        }
        offset = self.logs[decode].tell()
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
            futures = [
                executor.submit(
                    requests.post, url + "/generate", json=request, timeout=180
                )
                for url in self.urls
            ]
            responses = [future.result() for future in futures]
        for response in responses:
            self.assertEqual(response.status_code, 200, response.text)
        output = responses[decode].json()
        metadata = output["meta_info"]
        self.assertEqual(metadata["prompt_tokens"], length)
        self.assertEqual(metadata["completion_tokens"], 16)
        tokens = [item[1] for item in metadata["output_token_logprobs"]]
        self.assertEqual(len(tokens), 16)
        # Require graph use in this request's new decode log lines, rather than
        # accepting a graph captured at startup but never replayed after a flip.
        graph_pattern = re.compile(
            r"Decode batch.*(?:NPU|CUDA) graph: True", re.IGNORECASE
        )
        deadline = time.monotonic() + 5
        while True:
            messages = self.logs[decode].getvalue()[offset:]
            if graph_pattern.search(messages) or time.monotonic() >= deadline:
                break
            time.sleep(0.01)  # Allow the log-forwarding thread to drain the pipe.
        self.assertRegex(messages, graph_pattern)
        self.records.append(
            {"prefill": prefill, "input_length": length, "decode_response": output}
        )
        return tokens

    def test_service_role_switch(self):
        lengths = (16, 129, 513)
        self.assertEqual(
            {s["disaggregation_mode"] for s in self._state(0)}, {"prefill"}
        )
        peer = self._state(1)
        self.assertEqual({s["disaggregation_mode"] for s in peer}, {"decode"})
        self.assertTrue(all(s["decode_cuda_graph_bs"] for s in peer))
        baseline = {n: self._generate_pair(0, n) for n in lengths}
        # Calibrate same-role determinism before using exact tokens as an oracle.
        for n in lengths:
            self.assertEqual(self._generate_pair(0, n), baseline[n])
        if self.baseline_only:
            result = self._switch(0, "decode", expected=400)
            self.assertIn("enable-pd-role-switch", result["message"])
            self.assertTrue(result["safe_to_restore"])
            return
        self._switch(0, "prefill")
        self._switch(0, "", expected=400)
        missing = self._switch(0, "decode", expected=400)
        self.assertIn("decode_cuda_graph_memory_gb", missing["message"])
        insufficient = self._switch(
            0, "decode", expected=400, decode_cuda_graph_memory_gb=1e6
        )
        self.assertIn("insufficient", insufficient["message"])
        self.assertTrue(insufficient["safe_to_restore"])
        graph_args = {
            "decode_cuda_graph_bs": peer[0]["decode_cuda_graph_bs"],
            "decode_cuda_graph_memory_gb": max(
                s["decode_cuda_graph_memory_gb"] for s in peer
            ),
        }
        self.assertGreater(graph_args["decode_cuda_graph_memory_gb"], 0)
        prefill = 0
        for _ in range(4):
            new_prefill = 1 - prefill
            self._switch(prefill, "decode", **graph_args)
            self._switch(new_prefill, "prefill")
            prefill = new_prefill
            states = [self._state(i) for i in range(2)]
            self.records.append({"states_after_switch": states})
            self.assertTrue(
                all(s["disaggregation_mode"] == "prefill" for s in states[prefill])
            )
            self.assertTrue(
                all(
                    s["disaggregation_mode"] == "decode"
                    and s["decode_cuda_graph_bs"] == graph_args["decode_cuda_graph_bs"]
                    for s in states[1 - prefill]
                )
            )
            for n in lengths:
                self.assertEqual(
                    self._generate_pair(prefill, n),
                    baseline[n],
                    f"token mismatch after flip for input length {n}",
                )
        # Only one decode capture per rank/worker, despite repeated flips.
        for log in self.logs:
            messages = log.getvalue()
            self.assertEqual(
                messages.count("Capture target decode NPU graph begin."), self.tp
            )
            for failure in (
                "Decode CUDA graph capture on role switch failed",
                "instance unhealthy",
                "Failed to deregister buffers",
            ):
                self.assertNotIn(failure, messages)


if __name__ == "__main__":
    unittest.main()
