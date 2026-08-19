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
"""Unit tests for sglang/srt/utils/torch_memory_saver_adapter.py — no server, no GPU."""

import unittest
from unittest.mock import MagicMock, patch

from sglang.srt.utils.torch_memory_saver_adapter import (
    TorchMemorySaverAdapter,
    _TorchMemorySaverAdapterNoop,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


class TestTorchMemorySaverAdapterCreate(CustomTestCase):
    """TorchMemorySaverAdapter.create() factory method tests."""

    def test_create_disabled_returns_noop(self):
        """When enable=False, create() returns a _TorchMemorySaverAdapterNoop."""
        adapter = TorchMemorySaverAdapter.create(enable=False)
        self.assertIsInstance(adapter, _TorchMemorySaverAdapterNoop)

    @patch(
        "sglang.srt.utils.torch_memory_saver_adapter.import_error",
        new=ImportError("mock missing package"),
    )
    def test_create_enabled_without_package_raises(self):
        """When enable=True but torch_memory_saver is not installed, raises ImportError."""
        with self.assertRaises(ImportError):
            TorchMemorySaverAdapter.create(enable=True)


class TestNoopAdapter(CustomTestCase):
    """Tests for _TorchMemorySaverAdapterNoop (always available, no GPU required)."""

    def setUp(self):
        self.adapter = _TorchMemorySaverAdapterNoop()

    def test_enabled_is_false(self):
        self.assertFalse(self.adapter.enabled)

    def test_configure_subprocess_yields(self):
        """configure_subprocess is a context manager that yields."""
        with self.adapter.configure_subprocess():
            pass  # Should not raise

    def test_region_yields(self):
        """region is a context manager that yields."""
        with self.adapter.region(tag="test_tag"):
            pass
        with self.adapter.region(tag="test_tag", enable_cpu_backup=True):
            pass

    def test_cuda_graph_yields(self):
        """cuda_graph is a context manager that yields."""
        with self.adapter.cuda_graph():
            pass
        with self.adapter.cuda_graph(foo="bar"):
            pass

    def test_disable_yields(self):
        """disable is a context manager that yields."""
        with self.adapter.disable():
            pass

    def test_pause_resume_noop(self):
        """pause and resume are no-ops that don't raise."""
        self.adapter.pause(tag="test_tag")
        self.adapter.resume(tag="test_tag")

    def test_check_validity_logs_warning(self):
        """check_validity logs a warning when adapter is not enabled."""
        with self.assertLogs(
            "sglang.srt.utils.torch_memory_saver_adapter", level="WARNING"
        ) as cm:
            self.adapter.check_validity("test_caller")
        self.assertIn("test_caller", cm.output[0])
        self.assertIn("not save memory", cm.output[0])

    def test_context_managers_reentrant(self):
        """Context managers can be nested and reused."""
        with self.adapter.region(tag="outer"):
            with self.adapter.region(tag="inner"):
                with self.adapter.disable():
                    pass

        # Reuse after exit
        with self.adapter.region(tag="reuse"):
            pass


class TestNoopAdapterMultipleCalls(CustomTestCase):
    """Ensure Noop adapter methods can be called multiple times without side effects."""

    def setUp(self):
        self.adapter = _TorchMemorySaverAdapterNoop()

    def test_multiple_pause_resume_cycles(self):
        for i in range(10):
            self.adapter.pause(tag=f"tag_{i}")
            self.adapter.resume(tag=f"tag_{i}")

    def test_multiple_region_exits_cleanly(self):
        for i in range(10):
            with self.adapter.region(tag=f"tag_{i}"):
                pass
        self.assertFalse(self.adapter.enabled)


if __name__ == "__main__":
    unittest.main()
