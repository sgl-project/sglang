"""R-Fork (--load-format remote_instance) on Intel XPU at TP2.

A TP2 seed on XPUs 0-1 serves a TP2 client on XPUs 2-3; each seed/client rank
pair builds its own group. Same checks as the TP1 file.

Requirements: 4 XPUs; Mooncake built with XPU/TENT support for the
transfer_engine class (installed by docker/xpu.Dockerfile).

Usage:
    python3 -m pytest test/registered/model_loading/test_load_weights_from_remote_instance_xpu_tp2.py -v
"""

import unittest

from test_load_weights_from_remote_instance_xpu import (
    MOONCAKE_XPU_ENV,
    TRANSFER_ENGINE_CLIENT_ARGS,
    TRANSFER_ENGINE_SEED_ARGS,
    RemoteInstanceXpuTestMixin,
    requires_mooncake,
    requires_xpus,
)

from sglang.test.ci.ci_register import register_xpu_ci
from sglang.test.test_utils import CustomTestCase

register_xpu_ci(est_time=500, suite="nightly-kernel-main-test-4-gpu-xpu", nightly=True)


@requires_xpus(4)
class TestRemoteInstanceXcclTP2(RemoteInstanceXpuTestMixin, CustomTestCase):
    loader_backend = "nccl"
    tp_size = 2
    # One send-group port per TP rank pair.
    client_args = (
        "--remote-instance-weight-loader-send-weights-group-ports",
        "[60000, 60001]",
    )


@requires_mooncake
@requires_xpus(4)
class TestRemoteInstanceTransferEngineTP2(RemoteInstanceXpuTestMixin, CustomTestCase):
    loader_backend = "transfer_engine"
    tp_size = 2
    seed_args = TRANSFER_ENGINE_SEED_ARGS
    client_args = TRANSFER_ENGINE_CLIENT_ARGS
    server_env = MOONCAKE_XPU_ENV


if __name__ == "__main__":
    unittest.main()
