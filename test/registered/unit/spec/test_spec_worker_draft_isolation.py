"""The v2 spec workers hand their draft a ServerArgs copy.

Regression: they used to write the draft's context_length onto the instance
they share with the target worker, so the target's config carried draft values
for the rest of the process.
"""

import importlib
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from sglang.srt.runtime_context import get_context
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

WORKERS = {
    "eagle": (
        "sglang.srt.speculative.eagle_worker_v2",
        "EAGLEWorkerV2",
        "EagleDraftWorker",
    ),
    "standalone": (
        "sglang.srt.speculative.standalone_worker_v2",
        "StandaloneWorkerV2",
        "StandaloneDraftWorker",
    ),
    "multi_layer_eagle": (
        "sglang.srt.speculative.multi_layer_eagle_worker_v2",
        "MultiLayerEagleWorkerV2",
        "MultiLayerEagleDraftWorker",
    ),
    "frozen_kv_mtp": (
        "sglang.srt.speculative.frozen_kv_mtp_worker_v2",
        "FrozenKVMTPWorkerV2",
        "FrozenKVMTPDraftWorker",
    ),
}


def _target_worker():
    model_config = SimpleNamespace(context_len=4096)
    return SimpleNamespace(
        model_config=model_config,
        model_runner=SimpleNamespace(model_config=model_config),
        get_memory_pool=lambda: (None, None),
    )


class TestSpecWorkerDraftIsolation(CustomTestCase):
    def _seed(self):
        override = get_context().override_server_args(
            context_length=None,
            speculative_algorithm="EAGLE",
            speculative_num_steps=1,
            speculative_eagle_topk=1,
            speculative_num_draft_tokens=2,
        )
        server_args = override.install()
        self.addCleanup(override.restore)
        return server_args

    def test_the_draft_gets_a_copy_and_the_target_keeps_its_own(self):
        for label, (module_path, worker_name, draft_name) in WORKERS.items():
            with self.subTest(worker=label):
                module = importlib.import_module(module_path)
                worker_cls = getattr(module, worker_name)
                target_worker = _target_worker()
                server_args = self._seed()
                seen = {}

                def record(draft_server_args, *args, **kwargs):
                    seen["server_args"] = draft_server_args
                    raise _StopConstruction

                with patch.object(module, draft_name, record):
                    with self.assertRaises(_StopConstruction):
                        worker_cls.__init__(
                            worker_cls.__new__(worker_cls),
                            server_args=server_args,
                            gpu_id=0,
                            ps=SimpleNamespace(pp_rank=0),
                            nccl_port=0,
                            target_worker=target_worker,
                        )

                draft = seen["server_args"]
                self.assertIsNot(draft, server_args)
                self.assertEqual(draft.context_length, 4096)
                self.assertIsNone(server_args.context_length)


class _StopConstruction(Exception):
    """Cuts each worker's __init__ off once the draft args are captured."""


if __name__ == "__main__":
    unittest.main()
