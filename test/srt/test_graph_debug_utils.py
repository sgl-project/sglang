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
"""Unit tests for graph_debug_utils."""

import os
import shutil
import tempfile
import unittest
from pathlib import Path

import torch

from sglang.srt.utils.graph_debug_utils import DumpConfig, GraphDebugger


class TestGraphDebugUtils(unittest.TestCase):
    """Test CUDA Graph debugging utilities."""

    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        # Save original env vars
        self.orig_env = {}
        for key in [
            "SGLANG_GRAPH_DEBUG",
            "SGLANG_GRAPH_DEBUG_DIR",
            "SGLANG_GRAPH_DEBUG_LAYERS",
        ]:
            self.orig_env[key] = os.environ.get(key)

    def tearDown(self):
        """Clean up test artifacts."""
        # Restore env vars
        for key, value in self.orig_env.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value

        # Clean up temp dir
        if Path(self.temp_dir).exists():
            shutil.rmtree(self.temp_dir)

    def test_dump_config_from_env_disabled(self):
        """Test DumpConfig when debugging is disabled."""
        os.environ.pop("SGLANG_GRAPH_DEBUG", None)

        config = DumpConfig.from_env()

        self.assertFalse(config.enabled)

    def test_dump_config_from_env_enabled(self):
        """Test DumpConfig creation from environment variables."""
        os.environ["SGLANG_GRAPH_DEBUG"] = "1"
        os.environ["SGLANG_GRAPH_DEBUG_DIR"] = self.temp_dir
        os.environ["SGLANG_GRAPH_DEBUG_LAYERS"] = "0,1,2"

        config = DumpConfig.from_env()

        self.assertTrue(config.enabled)
        self.assertEqual(config.dump_dir, Path(self.temp_dir))
        self.assertEqual(config.debug_layers, [0, 1, 2])

    def test_debugger_disabled_by_default(self):
        """Test that debugger is disabled when env var not set."""
        os.environ.pop("SGLANG_GRAPH_DEBUG", None)

        debugger = GraphDebugger()
        self.assertFalse(debugger.config.enabled)

        # Should not crash when disabled
        device = "cuda" if torch.cuda.is_available() else "cpu"
        tensor = torch.randn(2, 3, device=device)
        debugger.capture_tensor("test", tensor)
        debugger.flush()

    def test_set_phase(self):
        """Test phase setting."""
        os.environ["SGLANG_GRAPH_DEBUG"] = "1"
        os.environ["SGLANG_GRAPH_DEBUG_DIR"] = self.temp_dir

        debugger = GraphDebugger()

        debugger.set_phase("capture", batch_size=4, token_step=0)
        self.assertEqual(debugger.phase, "capture")
        self.assertEqual(debugger.batch_size, 4)
        self.assertEqual(debugger.token_step, 0)

        debugger.set_phase("replay", batch_size=4, token_step=1)
        self.assertEqual(debugger.phase, "replay")
        self.assertEqual(debugger.token_step, 1)

    def test_capture_tensor_basic(self):
        """Test basic tensor capture."""
        os.environ["SGLANG_GRAPH_DEBUG"] = "1"
        os.environ["SGLANG_GRAPH_DEBUG_DIR"] = self.temp_dir

        debugger = GraphDebugger()
        device = "cuda" if torch.cuda.is_available() else "cpu"

        # Simulate capture phase
        debugger.set_phase("capture", batch_size=2, token_step=0)
        tensor = torch.randn(2, 3, device=device)
        debugger.capture_tensor("test_tensor", tensor)

        # Check buffer was allocated
        self.assertIn(2, debugger.buffers_by_bs)
        self.assertIn("test_tensor", debugger.buffers_by_bs[2])

    def test_capture_with_layer_filter(self):
        """Test tensor capture with layer filtering."""
        os.environ["SGLANG_GRAPH_DEBUG"] = "1"
        os.environ["SGLANG_GRAPH_DEBUG_DIR"] = self.temp_dir
        os.environ["SGLANG_GRAPH_DEBUG_LAYERS"] = "0,1"

        debugger = GraphDebugger()
        device = "cuda" if torch.cuda.is_available() else "cpu"

        debugger.set_phase("capture", batch_size=2, token_step=0)

        # Layer 0 should be captured
        tensor0 = torch.randn(2, 3, device=device)
        debugger.capture_tensor("tensor0", tensor0, layer_id=0)
        self.assertIn("L0_tensor0", debugger.buffers_by_bs.get(2, {}))

        # Layer 5 should be skipped
        tensor5 = torch.randn(2, 3, device=device)
        debugger.capture_tensor("tensor5", tensor5, layer_id=5)
        self.assertNotIn("L5_tensor5", debugger.buffers_by_bs.get(2, {}))

    def test_flush_creates_output(self):
        """Test that flush creates output files."""
        os.environ["SGLANG_GRAPH_DEBUG"] = "1"
        os.environ["SGLANG_GRAPH_DEBUG_DIR"] = self.temp_dir

        debugger = GraphDebugger()
        device = "cuda" if torch.cuda.is_available() else "cpu"

        # Capture phase
        debugger.set_phase("capture", batch_size=2, token_step=0)
        tensor = torch.randn(2, 3, device=device)
        debugger.capture_tensor("test_tensor", tensor)

        # Replay phase
        debugger.set_phase("replay", batch_size=2, token_step=1)
        debugger.flush()

        # Check output directory was created
        output_dir = Path(self.temp_dir) / "replay_bs2_token1"
        self.assertTrue(output_dir.exists())

        # Check tensor file exists
        tensor_files = list(output_dir.glob("*.pt"))
        self.assertEqual(len(tensor_files), 1)

        # Verify saved data
        saved_data = torch.load(tensor_files[0])
        self.assertIn("tensor", saved_data)
        self.assertIn("shape", saved_data)
        self.assertIn("dtype", saved_data)
        self.assertEqual(saved_data["batch_size"], 2)
        self.assertEqual(saved_data["phase"], "replay")

    def test_capture_dict(self):
        """Test capturing multiple tensors at once."""
        os.environ["SGLANG_GRAPH_DEBUG"] = "1"
        os.environ["SGLANG_GRAPH_DEBUG_DIR"] = self.temp_dir

        debugger = GraphDebugger()
        device = "cuda" if torch.cuda.is_available() else "cpu"

        debugger.set_phase("capture", batch_size=2, token_step=0)

        tensors = {
            "tensor_a": torch.randn(2, 3, device=device),
            "tensor_b": torch.randn(2, 4, device=device),
        }

        debugger.capture_dict(tensors, prefix="test")

        self.assertIn("test_tensor_a", debugger.buffers_by_bs[2])
        self.assertIn("test_tensor_b", debugger.buffers_by_bs[2])

    def test_max_buffers_limit(self):
        """Test that buffer count is limited."""
        os.environ["SGLANG_GRAPH_DEBUG"] = "1"
        os.environ["SGLANG_GRAPH_DEBUG_DIR"] = self.temp_dir
        os.environ["SGLANG_GRAPH_DEBUG_MAX_BUFFERS"] = "3"

        debugger = GraphDebugger()
        device = "cuda" if torch.cuda.is_available() else "cpu"

        debugger.set_phase("capture", batch_size=2, token_step=0)

        # Try to capture more tensors than max_buffers
        for i in range(5):
            tensor = torch.randn(2, 3, device=device)
            debugger.capture_tensor(f"tensor_{i}", tensor)

        # Only first 3 should be captured
        self.assertEqual(len(debugger.buffers_by_bs[2]), 3)


if __name__ == "__main__":
    unittest.main()
