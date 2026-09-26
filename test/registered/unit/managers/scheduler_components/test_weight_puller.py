import tempfile
import threading
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch.distributed

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase, maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()

from sglang.srt.managers.io_struct import PullWeightsReqInput  # noqa: E402
from sglang.srt.managers.scheduler_components import weight_puller  # noqa: E402
from sglang.srt.weight_sync import local_checkpoint  # noqa: E402

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class TestSchedulerWeightPuller(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        init_file = tempfile.NamedTemporaryFile(delete=False).name
        torch.distributed.init_process_group(
            backend="gloo", init_method=Path(init_file).as_uri(), rank=0, world_size=1
        )

    @classmethod
    def tearDownClass(cls):
        torch.distributed.destroy_process_group()

    def test_pull_runs_off_the_event_loop_and_replies_once_done(self):
        """The event loop must keep iterating while a pull copies weights, and reply only once it finishes."""
        release = threading.Event()
        sent = []
        puller = weight_puller.SchedulerWeightPuller(
            tp_cpu_group=None,
            ipc_channels=SimpleNamespace(
                send_to_tokenizer=SimpleNamespace(
                    send_output=lambda output, recv_req: sent.append(output)
                )
            ),
        )
        req = PullWeightsReqInput(
            local_checkpoint_dir="local", source_dir="published", target_version=1
        )
        model = SimpleNamespace(
            model_path="base",
            download_dir=None,
            revision=None,
            custom_pull_weights_pre_read_hook=None,
        )
        with (
            patch.object(weight_puller, "get_model", return_value=model),
            patch.object(
                weight_puller,
                "get_parallel",
                return_value=SimpleNamespace(
                    enable_dp_attention_local_control_broadcast=False
                ),
            ),
            patch.object(
                weight_puller,
                "get_exec",
                return_value=SimpleNamespace(
                    moe=SimpleNamespace(is_ep_scale_joiner=False)
                ),
            ),
            patch.object(
                weight_puller, "download_weights_from_hf", side_effect=lambda p, **_: p
            ),
            patch.object(
                local_checkpoint, "pull", side_effect=lambda **_: release.wait()
            ),
        ):
            self.assertIsNone(puller.handle(req))
            puller.check_pending()
            self.assertEqual(sent, [])
            self.assertIn("already in progress", puller.handle(req).message)

            release.set()
            deadline = time.monotonic() + 10
            while not sent and time.monotonic() < deadline:
                puller.check_pending()
                time.sleep(0.01)

        self.assertEqual(len(sent), 1)
        self.assertTrue(sent[0].success)


if __name__ == "__main__":
    unittest.main()
