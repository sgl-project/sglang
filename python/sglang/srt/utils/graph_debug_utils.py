"""
CUDA Graph compatible tensor dump tool.

Unlike --debug-tensor-dump-output-folder which disables CUDA graphs,
this tool works WITH graphs by pre-allocating GPU buffers whose
copy ops are recorded into the graph during capture.

Activated via --debug-graph-tensor-dump-output-folder CLI flag.
Hooks are registered automatically on the model (same as the non-graph
tensor dumper). During graph capture the hooks record buffer.copy_(tensor)
into the graph; on replay the copies execute inside the graph; flush()
persists to disk as Pass{N}.pt files.

Phase flow:
  1. "capture" (warmup, outside graph ctx): hooks allocate buffers + copy
  2. "capture" (inside graph ctx): hooks only copy into existing buffers
     (recorded into graph)
  3. "idle": hooks are no-ops
  4. "replay": graph executes recorded copies, flush() saves to disk
"""

import logging
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch

logger = logging.getLogger(__name__)


class GraphDebugger:
    _instance: Optional["GraphDebugger"] = None

    def __init__(self):
        self.enabled = False
        self._dump_dir: Optional[Path] = None
        self._max_buffers = 200
        self._capturing_graph = False

        self._phase = "idle"
        self._batch_size = 0
        self._flush_count = 0

        # {batch_size: {name: (buffer, shape, dtype)}}
        self._buffers: Dict[
            int, Dict[str, Tuple[torch.Tensor, List[int], torch.dtype]]
        ] = {}
        self._buffer_order: Dict[int, List[str]] = {}

    @classmethod
    def get_instance(cls) -> "GraphDebugger":
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def configure(
        self,
        dump_dir: str,
        tp_size: int,
        tp_rank: int,
        pp_rank: int,
        max_buffers: int = 200,
    ):
        self.enabled = True
        rank = tp_size * pp_rank + tp_rank
        self._dump_dir = (
            Path(dump_dir) / f"TP{tp_rank}_PP{pp_rank}_Rank{rank}_pid{os.getpid()}"
        )
        self._dump_dir.mkdir(parents=True, exist_ok=True)
        self._max_buffers = max_buffers
        logger.info("GraphDebugger enabled, dump_dir=%s", self._dump_dir)

    def register_hooks(self, model: torch.nn.Module):
        top_level_name = os.getenv("TENSOR_DUMP_TOP_LEVEL_MODULE_NAME", "model")
        self._add_hooks_recursive(model, "", top_level_name)
        logger.info("GraphDebugger hooks registered on model")

    def _add_hooks_recursive(self, module, prefix, top_level_name):
        for name, child in module._modules.items():
            if child is None:
                continue
            cur_name = name if not prefix else f"{prefix}.{name}"
            sub_count = self._add_hooks_recursive(child, cur_name, top_level_name)
            is_top_level = (not prefix) and (name == top_level_name)
            if sub_count == 0 or is_top_level:
                child.register_forward_hook(self._make_hook(cur_name))
        return len(module._modules)

    def _make_hook(self, name: str):
        def hook(module, input, output):
            if not self.enabled or self._phase != "capture":
                return
            if isinstance(output, torch.Tensor):
                self.capture_tensor(name, output)
            elif isinstance(output, (tuple, list)):
                for i, t in enumerate(output):
                    if isinstance(t, torch.Tensor):
                        self.capture_tensor(f"{name}.{i}", t)

        return hook

    def set_phase(self, phase: str, batch_size: int = 0):
        if not self.enabled:
            return
        self._phase = phase
        self._batch_size = batch_size

    def set_capturing_graph(self, capturing: bool):
        self._capturing_graph = capturing

    def capture_tensor(self, name: str, tensor: torch.Tensor):
        if not self.enabled or self._phase != "capture":
            return

        bs = self._batch_size
        buf_dict = self._buffers.setdefault(bs, {})
        if name not in buf_dict:
            if self._capturing_graph:
                # Inside graph capture context: allocation is forbidden.
                # Skip tensors not seen during warmup.
                return
            if len(buf_dict) >= self._max_buffers:
                return
            buf_dict[name] = (
                torch.empty_like(tensor),
                list(tensor.shape),
                tensor.dtype,
            )
            self._buffer_order.setdefault(bs, []).append(name)
        buf_dict[name][0].copy_(tensor)

    def flush(self):
        if not self.enabled or self._phase != "replay":
            return

        bs = self._batch_size
        buf_dict = self._buffers.get(bs)
        if not buf_dict:
            return

        tensors = {}
        for name in self._buffer_order.get(bs, []):
            entry = buf_dict.get(name)
            if entry is None:
                continue
            buffer, shape, dtype = entry
            try:
                tensors[name] = buffer.detach().cpu()
            except RuntimeError as e:
                logger.warning("GraphDebugger: failed to copy tensor %s: %s", name, e)
                continue

        if tensors:
            out_file = self._dump_dir / f"Pass{self._flush_count:05d}.pt"
            torch.save(tensors, out_file)
            logger.info(
                "GraphDebugger flushed %d tensors to %s", len(tensors), out_file
            )
            self._flush_count += 1


gdebug = GraphDebugger.get_instance()
