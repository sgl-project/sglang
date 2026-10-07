"""A speculative draft publishes its weights under its own role, after the target's embed and head are bound."""

import socket
import time
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import requests
import torch

import sglang.srt.managers.scheduler as scheduler_mod
import sglang.srt.model_executor.model_runner_components.remote_instance_weight_transporter as transporter_mod
from sglang.srt.entrypoints.engine_info_bootstrap_server import (
    EngineInfoBootstrapServer,
)
from sglang.srt.managers.scheduler import Scheduler
from sglang.srt.model_executor.model_runner_components.remote_instance_weight_transporter import (
    RemoteInstanceWeightTransporter,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


def _free_port():
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def _start_bootstrap_server():
    port = _free_port()
    server = EngineInfoBootstrapServer("127.0.0.1", port)
    url = f"http://127.0.0.1:{port}"
    for _ in range(100):
        try:
            if requests.get(f"{url}/health", timeout=1).status_code == 200:
                return server, url
        except requests.exceptions.ConnectionError:
            time.sleep(0.05)
    raise TimeoutError(f"the bootstrap server on {url} did not start")


def _publish(role, session_id, port):
    model_config = SimpleNamespace(
        remote_instance_weight_loader_backend=None, engine_info_bootstrap_port=port
    )
    with (
        patch.object(
            transporter_mod,
            "get_parallel",
            lambda: SimpleNamespace(tp_rank=0, dist_init_addr=None),
        ),
        patch.object(transporter_mod, "get_model", lambda: model_config),
        patch.object(
            transporter_mod, "remote_instance_transfer_engine_enabled", lambda: True
        ),
        patch.object(
            transporter_mod,
            "register_memory_region",
            lambda model, engine: {f"{role}.weight": (1, 4, 2)},
        ),
    ):
        transporter = RemoteInstanceWeightTransporter(
            server_args=SimpleNamespace(registers_parallelism_config=lambda: True),
            get_model=lambda: torch.nn.Module(),
            gpu_id=0,
        )
        transporter.engine = object()
        transporter.session_id = session_id
        transporter.parallelism_config = SimpleNamespace(
            to_dict=lambda: {"published_by": role}
        )
        transporter.maybe_register_and_publish_weight_info(role=role)


class TestPublicationRoles(CustomTestCase):
    def test_a_draft_publishing_for_the_same_rank_keeps_its_target(self):
        """Keyed by rank alone, the draft replaced the target, and p2p wrote target weights by the draft's table."""
        server, url = _start_bootstrap_server()
        self.addCleanup(server.close)
        _publish("target", "target-session", server.port)
        _publish("draft", "draft-session", server.port)

        def get(path, **params):
            return requests.get(f"{url}/{path}", params={"rank": 0, **params}).json()

        self.assertEqual(
            get("get_transfer_engine_info")["remote_instance_transfer_engine_info"],
            ["target-session", {"target.weight": [1, 4, 2]}],
        )
        self.assertEqual(
            get("get_transfer_engine_info", role="draft")[
                "remote_instance_transfer_engine_info"
            ],
            ["draft-session", {"draft.weight": [1, 4, 2]}],
        )
        self.assertEqual(get("get_parallelism_config"), {"published_by": "target"})
        self.assertEqual(
            get("get_parallelism_config", role="draft"), {"published_by": "draft"}
        )

    def test_drafts_publish_after_the_draft_worker_is_built(self):
        """Building the draft worker replaces the draft's embed and head with the target's; a table published
        before that holds freed addresses."""
        events = []

        class _Transporter:
            def __init__(self):
                self.weight_info = None

            def maybe_register_and_publish_weight_info(self, role):
                events.append(("published", role))

        runner = SimpleNamespace(remote_instance_weight_transporter=_Transporter())

        class _DraftWorker:
            def __init__(self, **kwargs):
                events.append("built")

            def weight_update_runners(self):
                return [("draft", runner)]

        scheduler = Scheduler.__new__(Scheduler)
        scheduler.server_args = None
        scheduler.nccl_port = 0
        scheduler.tp_worker = None
        scheduler.spec_algorithm = SimpleNamespace(
            is_none=lambda: False,
            is_ngram=lambda: False,
            create_worker=lambda server_args: _DraftWorker,
        )
        with (
            patch.object(
                scheduler_mod, "get_device", lambda: SimpleNamespace(gpu_id=0)
            ),
            patch.object(
                scheduler_mod, "get_parallel", lambda: SimpleNamespace(pp_size=1)
            ),
        ):
            scheduler.maybe_init_draft_worker()

        self.assertEqual(events, ["built", ("published", "draft")])


if __name__ == "__main__":
    unittest.main()
