"""Unit tests for EAGLEWorkerV2's draft-worker factory."""

import contextlib
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from sglang.srt.runtime_context import get_context
from sglang.srt.speculative import eagle_worker_v2
from sglang.srt.speculative.eagle_worker_v2 import EAGLEWorkerV2
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class _DraftWorker:
    def __init__(self, *args):
        self.args = args


class _Worker(EAGLEWorkerV2):
    def create_draft_worker(self, server_args, gpu_id, nccl_port, target_worker):
        return _DraftWorker(server_args, gpu_id, nccl_port, target_worker)


class TestCreateDraftWorker(CustomTestCase):
    def _build(self, is_last_rank):
        override = get_context().override_server_args(
            speculative_algorithm="EAGLE",
            speculative_eagle_topk=1,
            speculative_num_steps=3,
            speculative_num_draft_tokens=4,
        )
        override.install()
        self.addCleanup(override.restore)
        parallel = SimpleNamespace(pp_group=SimpleNamespace(is_last_rank=is_last_rank))
        with (
            patch.object(eagle_worker_v2, "get_parallel", return_value=parallel),
            patch.object(
                eagle_worker_v2,
                "get_device",
                return_value=SimpleNamespace(device="cpu"),
            ),
            patch.object(
                eagle_worker_v2,
                "get_plan_stream",
                return_value=(None, contextlib.nullcontext),
            ),
        ):
            return _Worker("server_args", 0, 1234, "target_worker")

    def test_subclass_supplies_the_draft_worker(self):
        worker = self._build(is_last_rank=True)
        self.assertIsInstance(worker.draft_worker, _DraftWorker)
        self.assertEqual(
            worker.draft_worker.args, ("server_args", 0, 1234, "target_worker")
        )
        # The rest of __init__ still ran.
        self.assertEqual(worker.target_worker, "target_worker")
        self.assertEqual(worker.speculative_num_draft_tokens, 4)

    def test_only_the_last_pp_stage_builds_a_draft_worker(self):
        self.assertIsNone(self._build(is_last_rank=False).draft_worker)


if __name__ == "__main__":
    unittest.main()
