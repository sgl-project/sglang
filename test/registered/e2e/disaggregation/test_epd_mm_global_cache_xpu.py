"""--enable-mm-global-cache over the Mooncake store on Intel XPU.

The global embedding cache stages ViT output from the card into a host paged
pool and publishes it to a Mooncake cluster, so a repeated image skips ViT.
With --encoder-transfer-backend mooncake the encoder keeps that output on the
card, so the staging call receives an XPU tensor.

Guards the regression where EmbeddingCacheController reached the accelerator
through torch.cuda: the copy streams and events could not be created off CUDA
and store_to_pool_async rejected any non-CUDA tensor. The encoder catches that
and recomputes, so serving kept working and the flag silently did nothing --
hence the assertions on the encoder log rather than on the answer alone.

Requirements:
    A mooncake-transfer-engine built with -DUSE_XPU=ON -DUSE_TENT=ON and
    -DWITH_STORE=ON, plus mooncake_master on PATH (both from
    docker/xpu.Dockerfile).
"""

import importlib
import os
import subprocess
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
    find_available_port,
    is_in_ci,
    popen_launch_server,
    terminate_and_kill_process_tree,
)

register_xpu_ci(est_time=360, suite="nightly-kernel-main-test-2-gpu-xpu", nightly=True)

_XPU_DEVICE_COUNT = torch.xpu.device_count() if torch.xpu.is_available() else 0
_SKIP_REASON = f"mm global cache on XPU needs 2 XPUs, found {_XPU_DEVICE_COUNT}"
# Outside CI, skip when fewer than 2 XPUs are visible; in CI the class runs and
# setUpClass fails, so a runner with a missing card is not silently green.
_requires_two_xpus = unittest.skipUnless(
    _XPU_DEVICE_COUNT >= 2 or is_in_ci(), _SKIP_REASON
)

# The refusal the fix removed, and the encoder's own staging accounting.
_STORE_REFUSAL_LOG = "store_to_pool_async expects"
_INSERT_LOG = "embeddings into Mooncake cluster"


@_requires_two_xpus
class TestEPDMooncakeXpuGlobalCache(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        cls.process_encode = None
        cls.process_language = None
        cls.metadata_process = None
        cls.master_process = None
        # The encoder's (stdout, stderr), copied to files so the test can read
        # whether staging was refused.
        cls.encode_logs = []
        if _XPU_DEVICE_COUNT < 2:
            raise RuntimeError(_SKIP_REASON)
        # Fail rather than skip, so a CI image missing the Mooncake build from
        # docker/xpu.Dockerfile shows up red instead of silently green.
        importlib.import_module("mooncake.engine")
        importlib.import_module("mooncake.store")

        cls.model = DEFAULT_SMALL_VLM_MODEL_NAME_FOR_TEST
        host, port = DEFAULT_URL_FOR_TEST.rsplit(":", 1)
        cls.base_url = f"{host}:{int(port) + 420}"
        encode_url = f"{host}:{int(port) + 421}"
        master_port = find_available_port(50061)
        metadata_port = find_available_port(18090)

        try:
            cls._start_mooncake_services(master_port, metadata_port)
            server_env = {
                "MC_USE_TENT": "1",
                # Workaround: on torch 2.14+xpu inductor's static Triton launcher
                # segfaults in load_kernel on the compiled vision attention;
                # likely pytorch/pytorch#199511, drop once the XPU torch pin
                # includes that fix.
                "TORCHINDUCTOR_USE_STATIC_CUDA_LAUNCHER": "0",
                "MC_MS_AUTO_DISC": "0",
                "MOONCAKE_DEVICE": "",
                "MOONCAKE_MASTER": f"127.0.0.1:{master_port}",
                "MOONCAKE_TE_META_DATA_SERVER": f"http://127.0.0.1:{metadata_port}/metadata",
                "MOONCAKE_GLOBAL_SEGMENT_SIZE": "4294967296",  # 4 GiB
                "MOONCAKE_LOCAL_HOSTNAME": "127.0.0.1",
            }
            transfer_args = ["--encoder-transfer-backend", "mooncake"]

            cls.encode_logs = [
                tempfile.NamedTemporaryFile("w", suffix=".log", delete=False)
                for _ in range(2)
            ]
            cls.process_encode = popen_launch_server(
                cls.model,
                encode_url,
                timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
                other_args=[
                    "--trust-remote-code",
                    "--encoder-only",
                    "--tp",
                    "1",
                    "--enable-mm-global-cache",
                ]
                + transfer_args,
                env=server_env,
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
                env=server_env,
            )
        except Exception:
            cls.tearDownClass()
            raise

    @classmethod
    def _start_mooncake_services(cls, master_port, metadata_port):
        cls.metadata_process = subprocess.Popen(
            [
                "python3",
                "-m",
                "mooncake.http_metadata_server",
                "--host",
                "127.0.0.1",
                "--port",
                str(metadata_port),
            ],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            preexec_fn=os.setsid,
        )
        cls.master_process = subprocess.Popen(
            ["mooncake_master", "--port", str(master_port)],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            preexec_fn=os.setsid,
        )
        # Neither service reports readiness on stdout; both bind within a second.
        time.sleep(8)
        for process, name in (
            (cls.metadata_process, "metadata server"),
            (cls.master_process, "mooncake_master"),
        ):
            if process.poll() is not None:
                raise RuntimeError(f"Mooncake {name} exited during startup")

    @classmethod
    def tearDownClass(cls):
        for process in (cls.process_language, cls.process_encode):
            if process is not None:
                terminate_and_kill_process_tree(process)
        for process in (cls.master_process, cls.metadata_process):
            if process is not None:
                terminate_and_kill_process_tree(process)
        for log in cls.encode_logs:
            log.close()
            os.remove(log.name)

    def _read_encode_log(self):
        text = ""
        for log in self.encode_logs:
            with open(log.name) as f:
                text += f.read()
        return text

    def _ask_about_the_image(self):
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
        return response.json()["choices"][0]["message"]["content"]

    def test_global_cache_stages_embeddings_from_xpu(self):
        """The encoder must stage its XPU-resident ViT output into the host pool
        and publish it, not fall back to recomputing every request."""
        answer = self._ask_about_the_image()
        # The image shows a man ironing on the back of a taxi.
        self.assertTrue(
            any(word in answer.lower() for word in ("iron", "taxi")),
            f"Unexpected description: {answer!r}",
        )

        # The server writes logs from a pipe-reader thread, so allow a moment.
        deadline = time.monotonic() + 15
        while True:
            encode_log = self._read_encode_log()
            if _INSERT_LOG in encode_log or time.monotonic() > deadline:
                break
            time.sleep(0.5)

        # assertTrue/False keep the encoder log out of the failure message.
        self.assertFalse(
            _STORE_REFUSAL_LOG in encode_log,
            "The global cache refused the encoder's XPU tensor, so it staged "
            "nothing and the ViT ran again",
        )
        self.assertTrue(
            _INSERT_LOG in encode_log,
            "The encoder never published an embedding to the Mooncake cluster",
        )


if __name__ == "__main__":
    unittest.main()
