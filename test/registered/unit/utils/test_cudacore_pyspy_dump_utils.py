# Copyright 2023-2024 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""CPU-only tests for CUDA coredump pipe resolution."""

import os
import stat
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

from sglang.srt.utils import cudacore_pyspy_dump_utils as dump_utils
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


class TestCudaCoreDumpTrigger(CustomTestCase):
    def setUp(self):
        self.proc = MagicMock()
        self.proc.pid = 1234
        self.proc.cwd.return_value = "/srv/sglang"

    def test_preserves_documented_default_when_present(self):
        with (
            patch.dict(
                os.environ,
                {"CUDA_ENABLE_USER_TRIGGERED_COREDUMP": "1"},
                clear=True,
            ),
            patch.object(dump_utils, "platform") as mock_platform,
            patch.object(dump_utils.psutil, "Process", return_value=self.proc),
            patch.object(dump_utils.os, "open", return_value=7) as mock_open,
            patch.object(dump_utils.os, "write") as mock_write,
            patch.object(dump_utils.os, "close") as mock_close,
        ):
            mock_platform.node.return_value = "gpu-host"

            dump_utils.trigger_cuda_user_coredump()

        mock_open.assert_called_once_with(
            Path("/srv/sglang/corepipe.cuda.gpu-host.1234"),
            os.O_WRONLY | os.O_NONBLOCK,
        )
        mock_write.assert_called_once_with(7, b"1")
        mock_close.assert_called_once_with(7)

    def test_uses_driver_default_name_when_documented_default_is_missing(self):
        fifo_stat = MagicMock(st_mode=stat.S_IFIFO | 0o600)
        with (
            patch.dict(
                os.environ,
                {"CUDA_ENABLE_USER_TRIGGERED_COREDUMP": "1"},
                clear=True,
            ),
            patch.object(dump_utils, "platform") as mock_platform,
            patch.object(dump_utils.psutil, "Process", return_value=self.proc),
            patch.object(
                dump_utils.os,
                "open",
                side_effect=[FileNotFoundError, 7],
            ) as mock_open,
            patch.object(dump_utils.os, "fstat", return_value=fifo_stat),
            patch.object(dump_utils.os, "write") as mock_write,
            patch.object(dump_utils.os, "close") as mock_close,
        ):
            mock_platform.node.return_value = "gpu-host"

            dump_utils.trigger_cuda_user_coredump()

        self.assertEqual(
            [call.args[0] for call in mock_open.call_args_list],
            [
                Path("/srv/sglang/corepipe.cuda.gpu-host.1234"),
                Path("/srv/sglang/corepipe_gpu-host_1234"),
            ],
        )
        mock_write.assert_called_once_with(7, b"1")
        mock_close.assert_called_once_with(7)

    def test_does_not_probe_driver_default_for_explicit_pipe_template(self):
        with (
            patch.dict(
                os.environ,
                {
                    "CUDA_ENABLE_USER_TRIGGERED_COREDUMP": "1",
                    "CUDA_COREDUMP_PIPE": "/var/run/cuda/%h-%p",
                },
                clear=True,
            ),
            patch.object(dump_utils, "platform") as mock_platform,
            patch.object(dump_utils.psutil, "Process", return_value=self.proc),
            patch.object(
                dump_utils.os, "open", side_effect=FileNotFoundError
            ) as mock_open,
            patch.object(dump_utils.os, "write") as mock_write,
        ):
            mock_platform.node.return_value = "gpu-host"

            dump_utils.trigger_cuda_user_coredump()

        mock_open.assert_called_once_with(
            Path("/var/run/cuda/gpu-host-1234"), os.O_WRONLY | os.O_NONBLOCK
        )
        mock_write.assert_not_called()

    def test_does_not_write_to_non_fifo_driver_default_path(self):
        regular_file_stat = MagicMock(st_mode=stat.S_IFREG | 0o600)
        with (
            patch.dict(
                os.environ,
                {"CUDA_ENABLE_USER_TRIGGERED_COREDUMP": "1"},
                clear=True,
            ),
            patch.object(dump_utils, "platform") as mock_platform,
            patch.object(dump_utils.psutil, "Process", return_value=self.proc),
            patch.object(dump_utils.os, "open", side_effect=[FileNotFoundError, 7]),
            patch.object(dump_utils.os, "fstat", return_value=regular_file_stat),
            patch.object(dump_utils.os, "write") as mock_write,
            patch.object(dump_utils.os, "close") as mock_close,
        ):
            mock_platform.node.return_value = "gpu-host"

            dump_utils.trigger_cuda_user_coredump()

        mock_write.assert_not_called()
        mock_close.assert_called_once_with(7)


if __name__ == "__main__":
    unittest.main()
