"""End-to-end tests for diffusion models with AR stage.

Launches AR model instances plus a DiffusionServer,
sends a generation request through the HTTP front-end, and verifies
that a non-empty output comes back.

Run directly:

    pytest -v python/sglang/multimodal_gen/test/server/test_ar_models.py
    pytest -v ... -k GLMImage              # one class
"""

from __future__ import annotations

import os
import unittest
from pathlib import Path

from sglang.multimodal_gen.runtime.utils.hf_diffusers_utils import maybe_download_model
from sglang.multimodal_gen.test.test_utils import (
    DEFAULT_AR_MODEL_NAME_FOR_TEST,
    find_free_port,
    wait_for_server_health,
)
from sglang.test.test_utils import CustomTestCase

HOST = "127.0.0.1"
_LOG_DIR = Path(os.environ.get("SGLANG_TEST_LOG_DIR", "/tmp"))

from sglang.multimodal_gen.test.single_test_file.test_disagg_server import (
    DisaggCluster,
    _DisaggTestBase,
    _generate_image,
    _require_gpus,
    _tail_log,
)

# ---------------------------------------------------------------------------
# AR cluster helper
# ---------------------------------------------------------------------------


class ARCluster(DisaggCluster):
    """Launch AR stage / main Diffusion stage server as separate processes."""

    def _alloc_ports(self) -> None:
        self.api_port = find_free_port(HOST)
        self.ar_port = find_free_port(HOST)

    # -- internals -----------------------------------------------------------

    def _launch_roles(self) -> None:
        gpus = self.gpu_layout["ar"]
        log = _LOG_DIR / "ar.log"
        self._logs["ar"] = log

        # The AR sub-encoder and processor live in subfolders of the model repo
        # (``vision_language_encoder/`` and ``processor/``).  A HuggingFace repo
        # id cannot carry a subfolder, so passing e.g.
        # ``zai-org/GLM-Image/vision_language_encoder/`` as --model-path fails
        # with "Repo id must be in the form 'repo_name' or 'namespace/repo_name'".
        # Resolve the repo to a local snapshot first and pass the local
        # subfolder directories, which are valid filesystem paths.
        local_model = maybe_download_model(self.model)

        cmd = [
            "sglang",
            "serve",
            "--model-path",
            os.path.join(local_model, "vision_language_encoder"),
            "--tokenizer-path",
            os.path.join(local_model, "processor"),
            "--enable-multimodal",
            "--cuda-graph-bs",
            "1",
            "--disable-fast-image-processor",
            "--tp-size",
            str(len(gpus)),
            "--port",
            str(self.ar_port),
            "--base-gpu-id",
            str(gpus[0]),
            "--mem-fraction-static",
            "0.4",
        ]
        self._start_proc(cmd, log)

        try:
            wait_for_server_health(
                f"http://{HOST}:{self.ar_port}",
                path="/v1/models",
                timeout=self.startup_timeout,
            )
        except Exception as e:
            raise RuntimeError(
                f"AR model failed to start for {self.name}. Log tail:\n"
                f"{_tail_log(log)}"
            ) from e

    def _launch_server_head(self) -> None:
        gpus = self.gpu_layout["ar"]
        num_gpus = str(len(gpus))
        log = _LOG_DIR / f"diffusion_server.log"
        self._logs["server"] = log

        cmd = [
            "sglang",
            "serve",
            "--model-path",
            self.model,
            "--srt-encoder-url",
            f"http://{HOST}:{self.ar_port}",
            "--port",
            str(self.api_port),
            "--host",
            HOST,
            "--num-gpus",
            num_gpus,
            "--sp-degree",
            num_gpus,
            "--base-gpu-id",
            str(gpus[0]),
            "--warmup-mode",
            "off",
        ]
        self._start_proc(cmd, log)
        try:
            wait_for_server_health(
                f"http://{HOST}:{self.api_port}",
                path="/v1/models",
                timeout=self.startup_timeout,
            )
        except Exception as e:
            raise RuntimeError(
                f"server head failed to become healthy for {self.name}: {e}\n"
                f"Server log tail:\n{_tail_log(log)}"
            ) from e


# ---------------------------------------------------------------------------
# Test classes
# ---------------------------------------------------------------------------


class _ARTestBase(_DisaggTestBase):

    @classmethod
    def setUpClass(cls) -> None:
        super(CustomTestCase, cls).setUpClass()
        _require_gpus(cls.required_gpus)
        cls.cluster = ARCluster(
            model=cls.model,
            name=cls.cluster_name,
            gpu_layout=cls.gpu_layout,
            extra_role_args=cls.extra_role_args,
        )
        cls.cluster.__enter__()


class TestGLMImage(_ARTestBase):
    """Baseline: 2 devices for ar, 1 for diffusion, 2 physical GPUs."""

    model = DEFAULT_AR_MODEL_NAME_FOR_TEST
    cluster_name = "glmimage"
    required_gpus = 2
    gpu_layout = {
        "ar": [0, 1],
        "diffusion": [0],
    }

    def test_generates_image(self) -> None:
        assert self.cluster is not None
        img = _generate_image(self.cluster.api_port, self.model)
        # A real PNG is well above 1 KB; catches empty / error responses.
        self.assertGreater(len(img), 1_000, f"image too small: {len(img)} bytes")


if __name__ == "__main__":
    unittest.main()
