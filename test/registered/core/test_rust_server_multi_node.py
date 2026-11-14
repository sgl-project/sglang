"""The embedded Rust server in a two-node engine, both nodes on one host.

One TP2 engine spans two "nodes" (``--nnodes 2``), each a separate launcher
process with one GPU: node 0 on GPU 0 and node 1 on GPU 1, sharing a
``--dist-init-addr`` on loopback. With ``SGLANG_RUST_SERVER=1`` the Rust
listener runs inside the global rank-0 scheduler on node 0. Node 1 hosts no
listener: it runs only its scheduler and the launcher's placeholder health
server on its own ``--port``, as in Python mode.

Every request crosses nodes: the rank-0 scheduler broadcasts it to the node-1
rank, and every forward runs collectives across both nodes.

Usage:
python3 -m unittest test_rust_server_multi_node
"""

import json
import os
import subprocess
import unittest
from urllib.parse import urlparse

import requests

from sglang.srt.utils import kill_process_tree
from sglang.srt.utils.network import get_free_port_below_ephemeral
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import (
    DEFAULT_SMALL_MODEL_NAME_FOR_TEST,
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    collect_process_tree_pids,
    find_available_port,
    is_rust_server_built,
    popen_launch_server,
    wait_for_gpu_release,
)

register_cuda_ci(est_time=300, stage="base-b", runner_config="2-gpu-large")

# Maximum time this test allows a node to exit after its peer is killed.
PEER_EXIT_TIMEOUT_SECS = 120


class TwoNodeRustServerGang:
    """One TP2 engine as two single-GPU launcher processes in Rust mode."""

    def __init__(self, model: str):
        parsed = urlparse(DEFAULT_URL_FOR_TEST)
        self.host = parsed.hostname
        self.node0_url = DEFAULT_URL_FOR_TEST
        node1_port = find_available_port(parsed.port + 100)
        self.node1_url = f"http://{self.host}:{node1_port}"
        self.model = model
        # Node 0's TCPStore binds this port; pick it below the ephemeral range
        # so no outgoing connection can take it first.
        dist_init_addr = f"{self.host}:{get_free_port_below_ephemeral()}"
        self.common_args = [
            "--tp-size",
            "2",
            "--nnodes",
            "2",
            "--dist-init-addr",
            dist_init_addr,
            # What is under test is the frontend and the cross-node plumbing,
            # not kernels: portable backends and no CUDA graphs keep the two
            # launches short on any runner GPU.
            "--attention-backend",
            "triton",
            "--sampling-backend",
            "pytorch",
            "--cuda-graph-backend-decode",
            "disabled",
            "--cuda-graph-backend-prefill",
            "disabled",
        ]
        self.env = {**os.environ, "SGLANG_RUST_SERVER": "1"}
        self.node0 = None
        self.node1 = None

    def launch(self) -> None:
        # Node 0 serves only once node 1 has joined the process groups, so
        # start node 1 first and without waiting for it.
        self.node1 = subprocess.Popen(
            [
                "sglang",
                "serve",
                "--model-path",
                self.model,
                *self.common_args,
                "--node-rank",
                "1",
                "--base-gpu-id",
                "1",
                "--host",
                self.host,
                "--port",
                str(urlparse(self.node1_url).port),
            ],
            env=self.env,
        )
        try:
            self.node0 = popen_launch_server(
                self.model,
                self.node0_url,
                timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
                env={"SGLANG_RUST_SERVER": "1"},
                other_args=[
                    *self.common_args,
                    "--node-rank",
                    "0",
                    "--base-gpu-id",
                    "0",
                ],
            )
        except BaseException:
            kill_process_tree(self.node1.pid)
            raise

    def shutdown(self) -> None:
        processes = [p for p in (self.node0, self.node1) if p is not None]
        # Snapshot both trees first: once one node is gone the other follows,
        # and an exited tree can no longer be walked.
        pids = [pid for p in processes for pid in collect_process_tree_pids(p.pid)]
        # SIGTERM node 0 first; node 1 leaves it to node 0 to stop the gang.
        for process in processes:
            process.terminate()
        for process in processes:
            try:
                process.wait(timeout=PEER_EXIT_TIMEOUT_SECS)
            except subprocess.TimeoutExpired:
                pass
            kill_process_tree(process.pid)
        wait_for_gpu_release(pids)


def _stream_generate(url: str, body: dict, on_first_chunk=None) -> list:
    """POST a streaming /generate; return every SSE payload before [DONE].
    ``on_first_chunk`` runs once the first chunk arrived, the request being
    admitted and running by then."""
    chunks = []
    with requests.post(
        f"{url}/generate", json={**body, "stream": True}, stream=True, timeout=300
    ) as response:
        response.raise_for_status()
        for line in response.iter_lines():
            if not line.startswith(b"data:"):
                continue
            payload = line[len(b"data:") :].strip()
            if payload == b"[DONE]":
                return chunks
            chunks.append(json.loads(payload))
            if len(chunks) == 1 and on_first_chunk is not None:
                on_first_chunk()
    raise AssertionError(f"stream ended without [DONE] after {len(chunks)} chunks")


@unittest.skipUnless(is_rust_server_built(), "embedded rust server extension not built")
class TestRustServerMultiNode(CustomTestCase):
    """Serving through a two-node gang, which every test leaves running."""

    @classmethod
    def setUpClass(cls):
        cls.gang = TwoNodeRustServerGang(DEFAULT_SMALL_MODEL_NAME_FOR_TEST)
        cls.gang.launch()

    @classmethod
    def tearDownClass(cls):
        cls.gang.shutdown()

    def test_node1_serves_only_the_placeholder_health_server(self):
        # Node 1 has no listener rank: the launcher's placeholder owns its port.
        for path in ("/health", "/health_generate"):
            response = requests.get(f"{self.gang.node1_url}{path}", timeout=10)
            self.assertEqual(response.status_code, 200, path)
        response = requests.post(
            f"{self.gang.node1_url}/generate", json={"text": "Hi"}, timeout=10
        )
        self.assertEqual(response.status_code, 404, response.text)

    def test_generate(self):
        response = requests.post(
            f"{self.gang.node0_url}/generate",
            json={
                "text": "The capital of France is",
                "sampling_params": {
                    "max_new_tokens": 16,
                    "temperature": 0,
                    "ignore_eos": True,
                },
            },
            timeout=300,
        )
        self.assertEqual(response.status_code, 200, response.text)
        result = response.json()
        self.assertTrue(result["text"])
        self.assertEqual(result["meta_info"]["completion_tokens"], 16)

    def test_generate_stream(self):
        chunks = _stream_generate(
            self.gang.node0_url,
            {
                "text": "Count from one to ten:",
                "sampling_params": {
                    "max_new_tokens": 16,
                    "temperature": 0,
                    "ignore_eos": True,
                },
            },
        )
        # A short generation may arrive coalesced into a single payload; the
        # final metadata and [DONE] are what show the stream completed.
        self.assertTrue(chunks)
        meta_info = chunks[-1]["meta_info"]
        self.assertEqual(meta_info["completion_tokens"], 16)
        self.assertEqual(meta_info["finish_reason"]["type"], "length")

    def test_abort_request(self):
        rid = "multi-node-abort"
        max_new_tokens = 4096

        def abort():
            response = requests.post(
                f"{self.gang.node0_url}/abort_request", json={"rid": rid}, timeout=30
            )
            self.assertEqual(response.status_code, 200, response.text)

        chunks = _stream_generate(
            self.gang.node0_url,
            {
                "rid": rid,
                "text": "Write a very long story.",
                "sampling_params": {
                    "max_new_tokens": max_new_tokens,
                    "temperature": 0,
                    "ignore_eos": True,
                },
            },
            on_first_chunk=abort,
        )
        meta_info = chunks[-1]["meta_info"]
        self.assertEqual(meta_info["finish_reason"]["type"], "abort")
        self.assertLess(meta_info["completion_tokens"], max_new_tokens)


@unittest.skipUnless(is_rust_server_built(), "embedded rust server extension not built")
class TestRustServerMultiNodeLifecycle(CustomTestCase):
    """Each test gets a fresh gang, since each ends it."""

    def setUp(self):
        self.gang = TwoNodeRustServerGang(DEFAULT_SMALL_MODEL_NAME_FOR_TEST)
        self.gang.launch()

    def tearDown(self):
        self.gang.shutdown()

    def _callTestMethod(self, method):
        # No method-only retry: a retry would not rerun setUp, so it would run
        # against the gang the first attempt already ended.
        unittest.TestCase._callTestMethod(self, method)

    def test_node0_exits_when_node1_dies(self):
        kill_process_tree(self.gang.node1.pid)
        try:
            self.gang.node0.wait(timeout=PEER_EXIT_TIMEOUT_SECS)
        except subprocess.TimeoutExpired:
            self.fail(
                f"node 0 still running {PEER_EXIT_TIMEOUT_SECS}s after node 1 died"
            )


if __name__ == "__main__":
    unittest.main()
