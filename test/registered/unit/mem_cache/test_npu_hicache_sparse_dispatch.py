"""Exercise real MLA PF/LF routing and layer ownership without an NPU runtime."""

from __future__ import annotations

import ast
import os
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=1, suite="base-a-test-cpu")

ROOT = Path(__file__).resolve().parents[4]


class Buffer:
    def __init__(self, shape, dtype="bf16"):
        self.shape, self.dtype = tuple(shape), dtype
        self.ndim = len(self.shape)
        self.selection = None

    def narrow(self, axis, start, length):
        if start < 0 or start + length > self.shape[axis]:
            raise IndexError("invalid layer range")
        shape = list(self.shape)
        shape[axis] = length
        view = Buffer(shape, self.dtype)
        view.selection = (self, axis, start, length)
        return view

    def unsqueeze(self, axis):
        shape = list(self.shape)
        shape.insert(axis if axis >= 0 else axis + self.ndim + 1, 1)
        view = Buffer(shape, self.dtype)
        view.selection = self.selection
        return view

    def transpose(self, a, b):
        shape = list(self.shape)
        shape[a], shape[b] = shape[b], shape[a]
        return Buffer(shape, self.dtype)

    def numel(self):
        result = 1
        for dim in self.shape:
            result *= dim
        return result


class TestRegisteredNPUHostAllocation(unittest.TestCase):
    def test_pin_memory_allocator_selects_configured_backend(self):
        path = ROOT / "python/sglang/srt/mem_cache/pool_host/common.py"
        tree = ast.parse(path.read_text())
        functions = [
            node
            for node in tree.body
            if isinstance(node, ast.FunctionDef)
            and node.name in ("alloc_with_pin_memory", "alloc_with_pin_memory_npu")
        ]
        module = ast.Module(body=functions, type_ignores=[])
        native = Mock(return_value="native")
        memfabric = Mock(return_value="memfabric")
        acl = Mock(return_value="acl")
        get_backend = Mock(return_value=None)
        backend = SimpleNamespace(get_host_memory_backend=get_backend)
        namespace = {
            "torch": SimpleNamespace(Tensor=object, dtype=object, empty=native)
        }
        with patch.dict(
            sys.modules,
            {"sglang.srt.mem_cache.pool_host.npu_host_memory": backend},
        ):
            exec(compile(module, str(path), "exec"), namespace)
            allocate = namespace["alloc_with_pin_memory_npu"]
            native_allocate = namespace["alloc_with_pin_memory"]
            kwargs = dict(
                dims=(2, 3),
                dtype="uint8",
                device="cpu",
                pin_memory=True,
                allocator=None,
            )

            self.assertEqual(allocate(**kwargs), "native")
            native.assert_called_once_with(
                (2, 3), dtype="uint8", device="cpu", pin_memory=True
            )

            get_backend.return_value = SimpleNamespace(alloc=memfabric)
            self.assertEqual(allocate(**kwargs), "memfabric")
            memfabric.assert_called_once_with((2, 3), "uint8", "cpu", True, None)
            self.assertEqual(native_allocate(**kwargs), "native")

            get_backend.return_value = SimpleNamespace(alloc=acl)
            self.assertEqual(allocate(**kwargs), "acl")
            acl.assert_called_once_with((2, 3), "uint8", "cpu", True, None)


