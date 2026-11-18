"""Server fixture for the Rust server's multimodal feature hand-off matrix.

Launches a Qwen-VL server under ``SGLANG_RUST_SERVER=1`` for one cell of the
(one node, two nodes) x ``--mm-feature-transport`` (cpu, cuda_ipc, cuda_vmm)
matrix, drives image requests through ``/generate``, and reads back the
hand-off accounting the scheduler exposes on ``/server_info`` as
``rust_mm_transport`` (see ``sglang.srt.rust_server.multimodal.MmTransportStats``).

The Rust MM workers produce every feature tensor in host memory. It reaches the
scheduler in one of two shapes, chosen by topology rather than by the transport
flag:

- ``inline``: a numpy array that owns the Rust vector, viewed by
  ``torch.from_numpy``. Single rank, and every multi-node layout (POSIX shm
  cannot cross a node boundary).
- ``shm``: a POSIX segment the worker wrote, named in the request so the TP
  broadcast carries a stub and every rank maps it. TP > 1 on one node.

``--mm-feature-transport`` governs the Python processor's GPU pools and the KV
budget reservation; the Rust path has to stay zero-copy under every value, on
every topology. Subclasses pick the cell and say which shape they expect.

Two nodes are emulated on one host: node 1 is started first as a plain
subprocess on its own port and GPU (it only joins the process group and serves
a health stub), then node 0 is launched through ``popen_launch_server`` on the
test URL, which cannot become healthy until node 1 has joined.
"""

import base64
import importlib.util
import io
import os
import subprocess
import unittest

import numpy as np
import requests
from PIL import Image

from sglang.test.test_utils import (
    DEFAULT_SMALL_VLM_MODEL_NAME_FOR_TEST,
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    find_available_port,
    popen_launch_server,
    terminate_and_kill_process_tree,
)

IMAGE_URL = "https://raw.githubusercontent.com/sgl-project/sgl-test-files/refs/heads/main/images/man_ironing_on_back_of_suv.png"
IMAGE_KEYWORDS = ("iron", "man", "taxi", "cab", "car", "suv", "street")
VISION_BLOCK = "<|vision_start|><|image_pad|><|vision_end|>"

# The smallest model that declares ``supports_cuda_vmm_feature_transport``,
# which the ``cuda_vmm`` cells use so they exercise a model the transport is
# defined for. Its image processor is the same Qwen-VL family the Rust pipeline
# serves.
CUDA_VMM_MODEL = "Qwen/Qwen3-VL-2B-Instruct"


def chat_prompt(question: str, image_count: int = 1) -> str:
    return (
        f"<|im_start|>user\n{VISION_BLOCK * image_count}{question}<|im_end|>\n"
        "<|im_start|>assistant\n"
    )


def solid_image_data_url(fmt: str = "PNG", rgb=(255, 0, 0)) -> str:
    buffer = io.BytesIO()
    Image.fromarray(np.full((64, 64, 3), rgb, dtype=np.uint8)).save(buffer, format=fmt)
    encoded = base64.b64encode(buffer.getvalue()).decode()
    return f"data:image/{fmt.lower()};base64,{encoded}"


def rust_server_extension_installed() -> bool:
    return importlib.util.find_spec("sglang.srt.rust_extensions._server") is not None


class RustMmTransportServerBase(CustomTestCase):
    """One (nodes x transport) cell. Subclasses set the class attributes below."""

    model = DEFAULT_SMALL_VLM_MODEL_NAME_FOR_TEST
    # ``--mm-feature-transport``: "cpu", "cuda_ipc" or "cuda_vmm".
    transport = "cpu"
    nnodes = 1
    tp = 1
    # The hand-off shape this topology must produce: "inline" or "shm".
    expected_handoff = "inline"
    mem_fraction_static = 0.8
    extra_args: list = []

    base_url = DEFAULT_URL_FOR_TEST
    timeout = DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH

    # Always present so tearDownClass can run after a setUpClass that failed
    # before launching anything.
    process = None
    peer_procs: list = []

    @classmethod
    def setUpClass(cls):
        if cls is RustMmTransportServerBase:
            raise unittest.SkipTest("fixture base class, not a test")
        if not rust_server_extension_installed():
            raise unittest.SkipTest(
                "sglang-server rust extension not installed (e.g. AMD suite)"
            )
        cls.peer_procs = []
        cls.process = None

        server_args = [
            "--enable-multimodal",
            "--mem-fraction-static",
            str(cls.mem_fraction_static),
            "--mm-feature-transport",
            cls.transport,
            "--tp",
            str(cls.tp),
            *cls.extra_args,
        ]
        env = {"SGLANG_RUST_SERVER": "1"}
        _, host, port = cls.base_url.split(":")
        host = host[2:]

        if cls.nnodes > 1:
            assert cls.tp % cls.nnodes == 0, "tp must split evenly across nodes"
            gpus_per_node = cls.tp // cls.nnodes
            dist_args = [
                "--nnodes",
                str(cls.nnodes),
                "--dist-init-addr",
                f"127.0.0.1:{find_available_port(20000)}",
            ]
            # Peers first: each joins the process group, takes its own GPU
            # range, and serves only a health stub on a port of its own. None
            # of them can become ready before node 0 joins, so node 0 is the
            # one to wait on.
            for node_rank in range(1, cls.nnodes):
                command = [
                    "sglang",
                    "serve",
                    "--model-path",
                    cls.model,
                    *server_args,
                    *dist_args,
                    "--node-rank",
                    str(node_rank),
                    "--base-gpu-id",
                    str(node_rank * gpus_per_node),
                    "--host",
                    host,
                    "--port",
                    str(find_available_port(int(port) + 1000)),
                ]
                print(f"[node {node_rank}] {' '.join(command)}")
                cls.peer_procs.append(
                    subprocess.Popen(command, env={**os.environ, **env})
                )
            server_args += [*dist_args, "--node-rank", "0"]

        try:
            cls.process = popen_launch_server(
                cls.model,
                cls.base_url,
                timeout=cls.timeout,
                other_args=server_args,
                env=env,
            )
        except Exception:
            cls.tearDownClass()
            raise

    @classmethod
    def tearDownClass(cls):
        # Node 0 first: it owns the listener the peers rendezvoused with.
        if cls.process is not None:
            terminate_and_kill_process_tree(cls.process)
            cls.process = None
        for proc in cls.peer_procs:
            terminate_and_kill_process_tree(proc)
        cls.peer_procs = []

    # -- helpers ---------------------------------------------------------

    def generate(self, prompt: str, image_data: list, max_new_tokens: int = 48):
        response = requests.post(
            self.base_url + "/generate",
            json={
                "text": prompt,
                "image_data": image_data,
                "sampling_params": {"temperature": 0, "max_new_tokens": max_new_tokens},
            },
        )
        self.assertEqual(response.status_code, 200, response.text)
        return response.json()["text"].lower()

    def transport_stats(self) -> dict:
        response = requests.get(self.base_url + "/server_info")
        self.assertEqual(response.status_code, 200, response.text)
        # The Rust `/server_info` nests the scheduler's allowlisted metrics
        # under `internal_states`, as the Python server does.
        state = response.json()["internal_states"][0]
        self.assertIn("rust_mm_transport", state, state.keys())
        return state["rust_mm_transport"]

    # -- tests -----------------------------------------------------------

    def test_image_request_answers(self):
        text = self.generate(
            prompt=chat_prompt("Describe this image in one sentence."),
            image_data=[IMAGE_URL],
        )
        self.assertTrue(any(w in text for w in IMAGE_KEYWORDS), text)

    def test_handoff_is_zero_copy(self):
        """Every feature crosses in the shape this topology dictates, and
        none of them is copied on the Python side of the hand-off."""
        before = self.transport_stats()
        text = self.generate(
            prompt=chat_prompt("What color is the second image?", image_count=2),
            image_data=[IMAGE_URL, solid_image_data_url("PNG")],
        )
        self.assertIn("red", text)
        after = self.transport_stats()

        expected = f"{self.expected_handoff}_features"
        other = (
            "shm_features" if self.expected_handoff == "inline" else "inline_features"
        )
        self.assertEqual(
            after[expected] - before[expected], 2, f"before={before} after={after}"
        )
        self.assertEqual(after[other], before[other], f"unexpected {other}: {after}")
        self.assertGreater(
            after[f"{self.expected_handoff}_bytes"],
            before[f"{self.expected_handoff}_bytes"],
            after,
        )
        self.assertEqual(after["copies"], 0, f"hand-off copied a feature: {after}")
