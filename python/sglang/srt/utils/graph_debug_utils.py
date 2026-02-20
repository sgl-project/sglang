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
"""CUDA Graph debugging utilities for tensor dumping.

This module provides tools to dump tensors during CUDA Graph replay phase,
which is challenging due to Python code not executing during graph replay.

Core principle:
- Capture phase: Pre-allocate GPU buffers and record copy operations into the graph
- Replay phase: Graph executes recorded copy operations automatically
- Post-replay: Save buffer contents to files in Python layer

Usage:
    1. Set environment variables:
        export SGLANG_GRAPH_DEBUG=1                    # Enable debugging
        export SGLANG_GRAPH_DEBUG_DIR=/tmp/graph_debug # Dump directory
        export SGLANG_GRAPH_DEBUG_LAYERS=0,1,2         # Layers to dump (optional)

    2. Insert dump points in model code:
        from sglang.srt.utils.graph_debug_utils import gdebug

        # In model forward pass
        gdebug.capture_tensor("rope_q", q, layer_id=self.layer_id)

        # After graph replay
        gdebug.flush()
"""

from __future__ import annotations

import os
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch


@dataclass
class DumpConfig:
    """Configuration for tensor dumping."""

    enabled: bool = False
    dump_dir: Path = Path("./dump/graph_debug")
    debug_points: List[str] = field(default_factory=list)
    debug_layers: List[int] = field(default_factory=list)
    max_buffers: int = 50  # Maximum number of buffers to pre-allocate

    @classmethod
    def from_env(cls) -> "DumpConfig":
        enabled = os.environ.get("SGLANG_GRAPH_DEBUG", "0") == "1"
        dump_dir = Path(os.environ.get("SGLANG_GRAPH_DEBUG_DIR", "./dump/graph_debug"))

        layers_str = os.environ.get("SGLANG_GRAPH_DEBUG_LAYERS", "")
        debug_layers = [int(l.strip()) for l in layers_str.split(",") if l.strip()]

        max_buffers = int(os.environ.get("SGLANG_GRAPH_DEBUG_MAX_BUFFERS", "50"))

        return cls(
            enabled=enabled,
            dump_dir=dump_dir,
            debug_layers=debug_layers,
            max_buffers=max_buffers,
        )


class GraphDebugger:
    """CUDA Graph compatible tensor dump tool.

    This debugger works around the limitation that Python code doesn't execute
    during CUDA Graph replay by pre-allocating GPU buffers during capture phase
    and recording copy operations into the graph.

    Working principle:
        1. Capture phase: Allocate GPU buffer for each dump point and execute
           buffer.copy_(tensor). This copy operation is recorded into CUDA Graph.
        2. Replay phase: graph.replay() executes pre-recorded copy operations
           automatically, updating buffer contents.
        3. Post-replay: Call flush() to save buffer contents to files.

    Note:
        - Each batch size has independent buffer sets because tensor shapes differ.
        - This class is designed for debugging only. Thread safety is not guaranteed
          for concurrent capture/flush operations across different threads.
    """

    _instance = None
    _lock = threading.Lock()

    def __init__(self):
        self.config = DumpConfig.from_env()
        self.phase = "unknown"  # "capture", "replay", "non_graph"
        self.batch_size = 0
        self.token_step = 0

        # Buffers organized by batch_size: {bs: {name: (buffer_tensor, shape, dtype)}}
        self.buffers_by_bs: Dict[
            int, Dict[str, Tuple[torch.Tensor, List[int], torch.dtype]]
        ] = {}
        # Buffer registration order by batch_size
        self.buffer_names_by_bs: Dict[int, List[str]] = {}

        # Current dump step (for non-graph mode)
        self.step = 0
        self.dump_count = 0

        if self.config.enabled:
            self.config.dump_dir.mkdir(parents=True, exist_ok=True)
            self._log(f"GraphDebugger enabled, dump directory: {self.config.dump_dir}")

    @classmethod
    def get_instance(cls) -> "GraphDebugger":
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = cls()
        return cls._instance

    def _log(self, msg: str):
        print(f"[GraphDebug] {msg}")

    def _should_capture(self, name: str, layer_id: Optional[int] = None) -> bool:
        """Check if this tensor should be captured."""
        if not self.config.enabled:
            return False

        # Check buffer count for current batch size
        bs = self.batch_size
        if (
            bs in self.buffers_by_bs
            and len(self.buffers_by_bs[bs]) >= self.config.max_buffers
        ):
            return False

        if self.config.debug_layers and layer_id is not None:
            if layer_id not in self.config.debug_layers:
                return False

        return True

    def set_phase(self, phase: str, batch_size: int = 0, token_step: int = 0):
        """Set current execution phase.

        Args:
            phase: One of "capture", "replay", or "non_graph"
            batch_size: Current batch size
            token_step: Current token generation step
        """
        if not self.config.enabled:
            return

        self.phase = phase
        self.batch_size = batch_size
        self.token_step = token_step
        self.step = 0

    def capture_tensor(
        self,
        name: str,
        tensor: torch.Tensor,
        layer_id: Optional[int] = None,
    ):
        """Capture a tensor in CUDA Graph.

        Behavior depends on current phase:
        - Capture phase: Allocate buffer and execute copy (recorded into graph)
        - Replay phase: Graph executes pre-recorded copy, data auto-updates to buffer
        - Non-graph phase: Directly dump to file

        Args:
            name: Tensor identifier
            tensor: Tensor to capture
            layer_id: Optional layer ID for filtering
        """
        full_name = f"L{layer_id}_{name}" if layer_id is not None else name

        if not self._should_capture(full_name, layer_id):
            return

        bs = self.batch_size

        # Check if we're actually in CUDA graph capture
        # If phase="capture" but tensor's first dim != bs, not real capture
        actual_bs = tensor.shape[0] if tensor.dim() > 0 else 1
        if self.phase == "capture" and actual_bs != bs:
            # Not a real capture, possibly warmup forward
            # Skip without dumping
            return

        if self.phase == "capture":
            # Ensure buffer dict exists for current batch size
            if bs not in self.buffers_by_bs:
                self.buffers_by_bs[bs] = {}
                self.buffer_names_by_bs[bs] = []

            buffers = self.buffers_by_bs[bs]
            buffer_names = self.buffer_names_by_bs[bs]

            # Capture phase: Allocate buffer and execute copy
            if full_name not in buffers:
                # First encounter, allocate buffer
                buffer = torch.empty_like(tensor)
                buffers[full_name] = (buffer, list(tensor.shape), tensor.dtype)
                buffer_names.append(full_name)
                self._log(
                    f"Allocated buffer (bs={bs}): {full_name}, shape={tensor.shape}"
                )

            # Execute copy - this operation will be recorded into CUDA Graph!
            buffer, _, _ = buffers[full_name]
            buffer.copy_(tensor)

        elif self.phase == "replay":
            # Replay phase: Graph has already executed copy, nothing to do
            # Data will be saved in flush()
            pass

        else:
            # Non-graph phase (e.g., prefill): Directly dump
            self._dump_immediate(full_name, tensor)

    def capture_dict(
        self,
        tensors: Dict[str, torch.Tensor],
        prefix: str = "",
        layer_id: Optional[int] = None,
    ):
        """Capture multiple tensors at once.

        Args:
            tensors: Dictionary of name -> tensor
            prefix: Optional prefix for tensor names
            layer_id: Optional layer ID for filtering
        """
        for name, tensor in tensors.items():
            full_name = f"{prefix}_{name}" if prefix else name
            self.capture_tensor(full_name, tensor, layer_id=layer_id)

    def flush(self):
        """Save all buffers to files.

        Should be called after graph.replay().
        """
        if not self.config.enabled or self.phase != "replay":
            return

        bs = self.batch_size
        if bs not in self.buffers_by_bs or not self.buffers_by_bs[bs]:
            self._log(f"Warning: bs={bs} has no pre-allocated buffers")
            return

        buffers = self.buffers_by_bs[bs]
        buffer_names = self.buffer_names_by_bs.get(bs, [])

        phase_dir = (
            self.config.dump_dir
            / f"{self.phase}_bs{self.batch_size}_token{self.token_step}"
        )
        phase_dir.mkdir(parents=True, exist_ok=True)

        for idx, name in enumerate(buffer_names):
            if name not in buffers:
                continue

            buffer, shape, dtype = buffers[name]

            try:
                tensor_cpu = buffer.detach().clone().cpu()
            except Exception as e:
                self._log(f"Warning: Cannot save {name}: {e}")
                continue

            filepath = phase_dir / f"step{idx:04d}_{name}.pt"

            save_data = {
                "tensor": tensor_cpu,
                "shape": shape,
                "dtype": str(dtype),
                "phase": self.phase,
                "batch_size": self.batch_size,
                "token_step": self.token_step,
                "step": idx,
                "name": name,
            }

            torch.save(save_data, filepath)

        self._log(f"Saved {len(buffer_names)} tensors to {phase_dir}")

    def _dump_immediate(self, name: str, tensor: torch.Tensor):
        """Immediately dump tensor in non-graph mode."""
        phase_dir = (
            self.config.dump_dir
            / f"{self.phase}_bs{self.batch_size}_token{self.token_step}"
        )
        phase_dir.mkdir(parents=True, exist_ok=True)

        try:
            tensor_cpu = tensor.detach().clone().cpu()
        except Exception as e:
            self._log(f"Warning: Cannot dump {name}: {e}")
            return

        filepath = phase_dir / f"step{self.step:04d}_{name}.pt"

        save_data = {
            "tensor": tensor_cpu,
            "shape": list(tensor.shape),
            "dtype": str(tensor.dtype),
            "phase": self.phase,
            "batch_size": self.batch_size,
            "token_step": self.token_step,
            "step": self.step,
            "name": name,
        }

        torch.save(save_data, filepath)
        self.step += 1
        self.dump_count += 1

    def dump(
        self,
        name: str,
        tensor: torch.Tensor,
        layer_id: Optional[int] = None,
        condition: bool = True,
        extra_info: Optional[Dict[str, Any]] = None,  # noqa: ARG002
    ):
        """Directly dump tensor in non-graph scenarios.

        Args:
            name: Tensor identifier
            tensor: Tensor to dump
            layer_id: Optional layer ID
            condition: Only dump if True
            extra_info: Reserved for future use (e.g., custom metadata)
        """
        # extra_info is reserved for future extensions
        del extra_info
        if not condition or not self.config.enabled:
            return

        full_name = f"L{layer_id}_{name}" if layer_id is not None else name
        self._dump_immediate(full_name, tensor)

    def dump_dict(
        self,
        tensors: Dict[str, torch.Tensor],
        prefix: str = "",
        layer_id: Optional[int] = None,
        condition: bool = True,
    ):
        for name, tensor in tensors.items():
            full_name = f"{prefix}_{name}" if prefix else name
            self.dump(full_name, tensor, layer_id=layer_id, condition=condition)


# Global singleton instance
gdebug = GraphDebugger.get_instance()


if __name__ == "__main__":
    print("Graph Debug Utils - CUDA Graph Compatible Tensor Dump")
    print("Usage: Set environment variable SGLANG_GRAPH_DEBUG=1 to enable")
    print("In model code: gdebug.capture_tensor('name', tensor)")
    print("After replay: gdebug.flush()")
