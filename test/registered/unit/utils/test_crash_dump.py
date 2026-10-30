"""The `--crash-dump-folder` pickle the Python TokenizerManager writes, which
the dump readers load."""

import glob
import os
import pickle
import sys
import tempfile
import unittest
from collections import deque
from types import SimpleNamespace
from unittest.mock import patch

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase, maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()

from sglang.srt.environ import envs  # noqa: E402
from sglang.srt.managers.io_struct import GenerateReqInput  # noqa: E402
from sglang.srt.managers.tokenizer_manager import TokenizerManager  # noqa: E402

register_cpu_ci(est_time=10, suite="base-a-test-cpu")


def _request(rid: str) -> GenerateReqInput:
    obj = GenerateReqInput(
        rid=rid, text="Hello", sampling_params={"max_new_tokens": 4, "stop": ["\n"]}
    )
    obj.normalize_batch_and_arguments()
    return obj


def _out(rid: str) -> dict:
    return {
        "text": " world",
        "output_ids": [11, 12],
        "meta_info": {"id": rid, "prompt_tokens": 3, "completion_tokens": 2},
    }


def _load_only_dump(folder: str) -> tuple:
    (path,) = glob.glob(os.path.join(folder, "*", "crash_dump_*.pkl"))
    with open(path, "rb") as f:
        return path, pickle.load(f)


class TestTokenizerManagerCrashDump(CustomTestCase):
    def test_file_name_and_contents(self):
        """The TokenizerManager's dump: one file per crash under the host's
        folder, holding the finished and unfinished requests with the launch
        state."""
        with tempfile.TemporaryDirectory() as folder:
            finished = (_request("done"), _out("done"), 100.0, 101.5)
            unfinished_state = SimpleNamespace(
                finished=False,
                obj=_request("running"),
                out_list=[],
                get_crash_dump_output=lambda: {"text": " par", "output_ids": [7]},
                time_stats=SimpleNamespace(
                    created_time=1.0,
                    set_finished_time=lambda: None,
                    finished_time=2.0,
                ),
            )
            tm = SimpleNamespace(
                crash_dump_folder=folder,
                crash_dump_performed=False,
                crash_dump_request_list=deque([finished]),
                rid_to_state={"running": unfinished_state},
                server_args=SimpleNamespace(watermark_config=None),
                _server_args_for_dump=lambda: {"model_path": "m"},
                _dump_config_snapshot=lambda: {"resolved": True},
            )
            with (
                patch(
                    "sglang.srt.managers.tokenizer_manager.convert_time_to_realtime",
                    side_effect=lambda t: t + 1000.0,
                ),
                patch.object(sys, "argv", ["launch_server", "--model-path", "m"]),
                # Only the pickle: no py-spy, CUDA coredump or their sleep.
                envs.SGLANG_PYSPY_DUMP_BEFORE_CRASH.override(False),
                envs.SGLANG_CUDA_COREDUMP_BEFORE_CRASH.override(False),
            ):
                TokenizerManager.dump_requests_before_crash(tm, hostname="host-a")
                # Once per process: a second dump would add a host-b file.
                TokenizerManager.dump_requests_before_crash(tm, hostname="host-b")

            path, dump = _load_only_dump(folder)
            self.assertEqual(os.path.dirname(path), os.path.join(folder, "host-a"))
            self.assertRegex(
                os.path.basename(path),
                r"^crash_dump_\d{4}-\d\d-\d\d_\d\d-\d\d-\d\d\.pkl$",
            )
            # Preserve the file mode produced by ordinary open() under the current umask.
            umask = os.umask(0)
            os.umask(umask)
            self.assertEqual(os.stat(path).st_mode & 0o777, 0o666 & ~umask)
            self.assertEqual(
                list(dump),
                [
                    "server_args",
                    "config_updates",
                    "resolved_config",
                    "requests",
                    "launch_command",
                ],
            )
            self.assertEqual(dump["server_args"], {"model_path": "m"})
            self.assertEqual(dump["resolved_config"], {"resolved": True})
            self.assertEqual(dump["launch_command"], "launch_server --model-path m")
            self.assertIsInstance(dump["config_updates"], list)
            (done, running) = dump["requests"]
            self.assertEqual(done[0], finished[0])
            self.assertEqual(done[1:], finished[1:])
            self.assertEqual(running[0], unfinished_state.obj)
            self.assertEqual(
                running[1:], ({"text": " par", "output_ids": [7]}, 1001.0, 1002.0)
            )


if __name__ == "__main__":
    unittest.main()
