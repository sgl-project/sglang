"""R-Fork (--load-format remote_instance) on Intel XPU.

A seed server on XPU 0 serves its weights to a client server on XPU 1, which
loads them over XCCL (backend ``nccl``) or Mooncake TransferEngine (backend
``transfer_engine``). The client must hold the seed's exact weights and produce
the seed's greedy output.

Requirements: 2 XPUs; Mooncake built with XPU/TENT support for the
transfer_engine class (installed by docker/xpu.Dockerfile).

Usage:
    python3 -m pytest test/registered/model_loading/test_load_weights_from_remote_instance_xpu.py -v
"""

import importlib.util
import unittest
from urllib.parse import urlparse

import numpy as np
import requests
import torch

from sglang.test.ci.ci_register import register_xpu_ci
from sglang.test.test_utils import (
    ADMIN_HEADERS_FOR_TEST,
    ADMIN_LAUNCH_ARGS_FOR_TEST,
    DEFAULT_SMALL_MODEL_NAME_FOR_TEST,
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    is_in_ci,
    popen_launch_server,
    terminate_and_kill_process_tree,
)

register_xpu_ci(est_time=400, suite="nightly-kernel-main-test-2-gpu-xpu", nightly=True)

_XPU_DEVICE_COUNT = torch.xpu.device_count() if torch.xpu.is_available() else 0
_SKIP_REASON = f"R-Fork needs 2 XPUs, found {_XPU_DEVICE_COUNT}"
# In CI the class runs and fails, so a runner missing a card is not silently green.
_requires_two_xpus = unittest.skipUnless(
    _XPU_DEVICE_COUNT >= 2 or is_in_ci(), _SKIP_REASON
)

# Both servers share the admin key: the client sends it to the seed's weight-send endpoints.
_XPU_ARGS = (
    "--device",
    "xpu",
    "--attention-backend",
    "intel_xpu",
    *ADMIN_LAUNCH_ARGS_FOR_TEST,
)
# XPU support in Mooncake lives in its TENT engine.
_MOONCAKE_XPU_ENV = {"MC_USE_TENT": "1"}
_CHECKED_PARAMS = (
    "model.embed_tokens.weight",
    "model.layers.0.input_layernorm.weight",
    "model.layers.1.self_attn.q_proj.weight",
    "model.layers.3.self_attn.v_proj.weight",
    "model.layers.5.mlp.gate_proj.weight",
    "model.layers.7.mlp.down_proj.weight",
    "model.norm.weight",
)
_PROMPT = "The capital of France is"


def _weights(url):
    return [
        requests.get(
            f"{url}/get_weights_by_name",
            json={"name": name, "truncate_size": 64},
            headers=ADMIN_HEADERS_FOR_TEST,
            timeout=60,
        ).json()
        for name in _CHECKED_PARAMS
    ]


def _greedy(url):
    response = requests.post(
        f"{url}/generate",
        json={
            "text": _PROMPT,
            "sampling_params": {"temperature": 0, "max_new_tokens": 16},
        },
        timeout=120,
    )
    response.raise_for_status()
    return response.json()["text"]


class _RemoteInstanceXpuTestMixin:
    loader_backend: str
    seed_args: tuple = ()
    client_args: tuple = ()
    server_env: dict = {}

    @classmethod
    def setUpClass(cls):
        if _XPU_DEVICE_COUNT < 2:
            raise RuntimeError(_SKIP_REASON)
        cls.seed_process = cls.client_process = None
        seed = urlparse(DEFAULT_URL_FOR_TEST)
        cls.seed_url = DEFAULT_URL_FOR_TEST
        cls.client_url = f"http://{seed.hostname}:{seed.port + 100}"
        env = dict(cls.server_env) or None
        cls.seed_process = popen_launch_server(
            DEFAULT_SMALL_MODEL_NAME_FOR_TEST,
            cls.seed_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=(*_XPU_ARGS, "--base-gpu-id", "0", *cls.seed_args),
            env=env,
        )
        cls.client_process = popen_launch_server(
            DEFAULT_SMALL_MODEL_NAME_FOR_TEST,
            cls.client_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=(
                *_XPU_ARGS,
                "--base-gpu-id",
                "1",
                "--load-format",
                "remote_instance",
                "--remote-instance-weight-loader-backend",
                cls.loader_backend,
                "--remote-instance-weight-loader-seed-instance-ip",
                seed.hostname,
                "--remote-instance-weight-loader-seed-instance-service-port",
                str(seed.port),
                *cls.client_args,
            ),
            env=env,
        )

    @classmethod
    def tearDownClass(cls):
        for process in (cls.client_process, cls.seed_process):
            if process is not None:
                terminate_and_kill_process_tree(process)

    def test_client_holds_the_seed_weights(self):
        for name, seed, client in zip(
            _CHECKED_PARAMS, _weights(self.seed_url), _weights(self.client_url)
        ):
            with self.subTest(param=name):
                # A device-to-device copy moves bits; nothing is recomputed.
                self.assertTrue(np.array_equal(np.array(seed), np.array(client)))

    def test_client_generates_like_the_seed(self):
        self.assertEqual(_greedy(self.client_url), _greedy(self.seed_url))


@_requires_two_xpus
class TestRemoteInstanceXccl(_RemoteInstanceXpuTestMixin, CustomTestCase):
    """Backend ``nccl``: seed and client build an XCCL group on XPU."""

    loader_backend = "nccl"
    client_args = (
        "--remote-instance-weight-loader-send-weights-group-ports",
        "[60000]",
    )


@unittest.skipUnless(
    is_in_ci() or importlib.util.find_spec("mooncake") is not None,
    "Mooncake is required for the transfer_engine backend",
)
@_requires_two_xpus
class TestRemoteInstanceTransferEngine(_RemoteInstanceXpuTestMixin, CustomTestCase):
    """Backend ``transfer_engine``: the client reads the seed's registered weights."""

    loader_backend = "transfer_engine"
    seed_args = ("--remote-instance-weight-loader-start-seed-via-transfer-engine",)
    client_args = (
        "--remote-instance-weight-loader-start-seed-via-transfer-engine",
        "--engine-info-bootstrap-port",
        "6790",
    )
    server_env = _MOONCAKE_XPU_ENV


if __name__ == "__main__":
    unittest.main()
