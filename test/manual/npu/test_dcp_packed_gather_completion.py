"""Verify packed KV consumers observe completed chunked collectives."""

import ast
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch

ROOT = Path(__file__).resolve().parents[3]


def gather():
    path = (
        ROOT
        / "python/sglang/srt/hardware_backend/npu/modules/deepseek_v2_attention_mla_npu.py"
    )
    tree = ast.parse(path.read_text())
    function = next(
        n
        for n in tree.body
        if isinstance(n, ast.FunctionDef) and n.name == "_dcp_issue_gather_chunks"
    )
    future = ast.ImportFrom(
        module="__future__", names=[ast.alias(name="annotations")], level=0
    )
    namespace = {}
    exec(
        compile(
            ast.fix_missing_locations(
                ast.Module(body=[future, function], type_ignores=[])
            ),
            str(path),
            "exec",
        ),
        namespace,
    )
    return namespace[function.name]


class DelayedGroup:
    def __init__(self):
        self.waited = 0

    def all_gather_into_tensor(self, out, send):
        out.copy_(torch.cat([send, send + 100]))

    def all_gather_into_tensor_async(self, out, send):
        def wait():
            self.waited += 1
            self.all_gather_into_tensor(out, send)

        return SimpleNamespace(wait=wait)


class TestPackedGatherCompletion(unittest.TestCase):
    def test_chunked_output_is_ready_for_inline_read_or_ready_event(self):
        group = DelayedGroup()
        out = torch.full((10, 1), -777)
        send = torch.arange(5).view(-1, 1)
        plan = SimpleNamespace(dcp_size=2, send_rows=5, gathered_rows=10, chunks=3)
        gather()(group, out, send, plan)
        expected = torch.tensor([0, 100, 1, 101, 2, 3, 4, 102, 103, 104]).view(-1, 1)
        self.assertTrue(torch.equal(out, expected))
        self.assertEqual(group.waited, 3)

    def test_single_chunk_preserves_synchronous_path(self):
        group = DelayedGroup()
        out = torch.empty((10, 1), dtype=torch.int64)
        send = torch.arange(5).view(-1, 1)
        plan = SimpleNamespace(dcp_size=2, send_rows=5, gathered_rows=10, chunks=1)
        gather()(group, out, send, plan)
        self.assertTrue(torch.equal(out, torch.cat([send, send + 100])))
        self.assertEqual(group.waited, 0)


if __name__ == "__main__":
    unittest.main()
