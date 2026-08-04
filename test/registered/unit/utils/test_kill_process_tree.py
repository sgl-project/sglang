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
"""Unit tests for process tree cleanup helpers."""

import os
import unittest
from unittest.mock import MagicMock, patch

from sglang.srt.utils.common import kill_process_tree
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=2, suite="stage-a-test-cpu")


class TestKillProcessTree(CustomTestCase):
    def test_current_process_exit_code_defaults_to_zero(self):
        proc = MagicMock()
        proc.children.return_value = []

        with patch("sglang.srt.utils.common.psutil.Process", return_value=proc):
            with self.assertRaises(SystemExit) as ctx:
                kill_process_tree(os.getpid())

        proc.kill.assert_called_once()
        self.assertEqual(ctx.exception.code, 0)

    def test_current_process_exit_code_can_be_overridden(self):
        proc = MagicMock()
        proc.children.return_value = []

        with patch("sglang.srt.utils.common.psutil.Process", return_value=proc):
            with self.assertRaises(SystemExit) as ctx:
                kill_process_tree(os.getpid(), exit_code=1)

        proc.kill.assert_called_once()
        self.assertEqual(ctx.exception.code, 1)


if __name__ == "__main__":
    unittest.main()
