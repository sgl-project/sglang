"""
EPD (encoder / language split) integration tests for the Mooncake encoder
transfer backend on Intel XPU.

Launches an encoder-only server and a language-only server and sends an image
chat request through the language server, so the image embedding crosses the
Mooncake transfer engine into a device landing buffer on the language side.
Mooncake runs on the TENT engine, which copies card to card through Level Zero
IPC on one node, so no RDMA NIC is needed; both servers must see both cards.

Requirements:
    A mooncake-transfer-engine built with -DUSE_XPU=ON -DUSE_TENT=ON (installed
    by docker/xpu.Dockerfile).
"""

import importlib
import os
import tempfile
import time
import unittest

import requests
import torch

from sglang.test.ci.ci_register import register_xpu_ci
from sglang.test.test_utils import (
    DEFAULT_IMAGE_URL,
    DEFAULT_SMALL_VLM_MODEL_NAME_FOR_TEST,
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    is_in_ci,
    popen_launch_server,
    terminate_and_kill_process_tree,
)

register_xpu_ci(est_time=300, suite="nightly-xpu-kernel-main-2-gpu", nightly=True)

_XPU_DEVICE_COUNT = torch.xpu.device_count() if torch.xpu.is_available() else 0
_SKIP_REASON = f"EPD on XPU needs 2 XPUs, found {_XPU_DEVICE_COUNT}"
# Outside CI, skip when fewer than 2 XPUs are visible; in CI the class runs and
# setUpClass fails, so a runner with a missing card is not silently green.
_requires_two_xpus = unittest.skipUnless(
    _XPU_DEVICE_COUNT >= 2 or is_in_ci(), _SKIP_REASON
)

_SERVER_ENV = {
    # XPU support lives in Mooncake's TENT engine; with MOONCAKE_PROTOCOL left at
    # its default it copies card to card through Level Zero IPC.
    "MC_USE_TENT": "1",
    # Workaround: on torch 2.14+xpu inductor's static Triton launcher segfaults in
    # load_kernel on the compiled vision attention; likely pytorch/pytorch#199511,
    # drop once the XPU torch pin includes that fix.
    "TORCHINDUCTOR_USE_STATIC_CUDA_LAUNCHER": "0",
}


# Mooncake logs this on the encoder when it maps the language server's buffer for
# a card-to-card copy, and the second line when it falls back to TCP instead.
_IPC_MAPPED_LOG = "XpuTransport: mapped peer buffer"
_IPC_UNAVAILABLE_LOG = "is on a device not visible here"


class _EPDMooncakeXpuTestMixin:
    """Subclasses pick which language-side landing buffer receives the embedding."""

    language_env: dict

    @classmethod
    def setUpClass(cls):
        cls.process_encode = None
        cls.process_language = None
        # The encoder's (stdout, stderr), copied to files so the test can read
        # which transport Mooncake used.
        cls.encode_logs = []
        if _XPU_DEVICE_COUNT < 2:
            raise RuntimeError(_SKIP_REASON)
        # Fail rather than skip, so a CI image missing the Mooncake build from
        # docker/xpu.Dockerfile shows up red instead of silently green.
        importlib.import_module("mooncake.engine")
        cls.model = DEFAULT_SMALL_VLM_MODEL_NAME_FOR_TEST
        host, port = DEFAULT_URL_FOR_TEST.rsplit(":", 1)
        cls.base_url = f"{host}:{int(port) + 400}"
        encode_url = f"{host}:{int(port) + 401}"
        transfer_args = ["--encoder-transfer-backend", "mooncake"]

        cls.encode_logs = [
            tempfile.NamedTemporaryFile("w", suffix=".log", delete=False)
            for _ in range(2)
        ]
        cls.process_encode = popen_launch_server(
            cls.model,
            encode_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=["--trust-remote-code", "--encoder-only", "--tp", "1"]
            + transfer_args,
            env=_SERVER_ENV,
            return_stdout_stderr=tuple(cls.encode_logs),
        )
        cls.process_language = popen_launch_server(
            cls.model,
            cls.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=[
                "--trust-remote-code",
                "--language-only",
                "--encoder-urls",
                encode_url,
                "--tp",
                "1",
                "--base-gpu-id",
                "1",
            ]
            + transfer_args,
            env={**_SERVER_ENV, **cls.language_env},
        )

    @classmethod
    def tearDownClass(cls):
        for process in (cls.process_language, cls.process_encode):
            if process is not None:
                terminate_and_kill_process_tree(process)
        for log in cls.encode_logs:
            log.close()
            os.remove(log.name)

    def test_image_embedding_reaches_language_model(self):
        """An image answer needs the encoder's embedding to land intact in the
        language server's XPU buffer, copied card to card over Level Zero IPC."""
        response = requests.post(
            self.base_url + "/v1/chat/completions",
            json={
                "model": self.model,
                "temperature": 0,
                "max_tokens": 48,
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image_url",
                                "image_url": {"url": DEFAULT_IMAGE_URL},
                            },
                            {
                                "type": "text",
                                "text": "Describe this image in one sentence.",
                            },
                        ],
                    }
                ],
            },
            timeout=300,
        )
        self.assertEqual(response.status_code, 200, response.text)
        answer = response.json()["choices"][0]["message"]["content"]
        # The image shows a man ironing on the back of a taxi.
        self.assertTrue(
            any(word in answer.lower() for word in ("iron", "taxi")),
            f"Unexpected description: {answer!r}",
        )
        self._assert_transfer_used_level_zero_ipc()

    def _assert_transfer_used_level_zero_ipc(self):
        # The server writes logs from a pipe-reader thread, so allow a moment.
        deadline = time.monotonic() + 10
        while True:
            encode_log = ""
            for log in self.encode_logs:
                with open(log.name) as f:
                    encode_log += f.read()
            if _IPC_MAPPED_LOG in encode_log or time.monotonic() > deadline:
                break
            time.sleep(0.5)
        # assertTrue/False keep the encoder log out of the failure message.
        self.assertFalse(
            _IPC_UNAVAILABLE_LOG in encode_log,
            "Mooncake fell back to TCP: both servers must see both cards",
        )
        self.assertTrue(
            _IPC_MAPPED_LOG in encode_log,
            "Mooncake did not copy the embedding card to card over Level Zero IPC",
        )


@_requires_two_xpus
class TestEPDMooncakeXpuEmbeddingPool(_EPDMooncakeXpuTestMixin, CustomTestCase):
    """Default: the embedding lands in the pre-registered EmbeddingPool."""

    language_env = {}


@_requires_two_xpus
class TestEPDMooncakeXpuPerRequestBuffer(_EPDMooncakeXpuTestMixin, CustomTestCase):
    """Pool disabled: each request allocates and registers its own landing buffer."""

    language_env = {"SGLANG_EMBEDDING_POOL_SIZE_MB": "0"}


if __name__ == "__main__":
    unittest.main()
