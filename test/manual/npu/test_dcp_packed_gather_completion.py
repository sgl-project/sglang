"""Verify packed KV consumers observe completed chunked collectives."""

import ast
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch

ROOT = Path(__file__).resolve().parents[3]
SOURCE = (
    ROOT
    / "python/sglang/srt/hardware_backend/npu/modules/deepseek_v2_attention_mla_npu.py"
)


def load_gather(async_impl):
    """``_dcp_issue_gather_chunks`` exec'd without importing torch_npu.

    The function calls the module-global ``_dcp_all_gather_async``, so both are
    extracted from the AST and the async collective is faked by binding ``torch``
    in the exec'd namespace: the real helper only touches
    ``torch.distributed.all_gather_into_tensor``.
    """
    tree = ast.parse(SOURCE.read_text())
    functions = [
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef)
        and node.name in ("_dcp_all_gather_async", "_dcp_issue_gather_chunks")
    ]
    future = ast.ImportFrom(
        module="__future__", names=[ast.alias(name="annotations")], level=0
    )
    namespace = {}
    exec(
        compile(
            ast.fix_missing_locations(
                ast.Module(body=[future, *functions], type_ignores=[])
            ),
            str(SOURCE),
            "exec",
        ),
        namespace,
    )
    fn = namespace["_dcp_issue_gather_chunks"]
    fn.__globals__["torch"] = SimpleNamespace(
        distributed=SimpleNamespace(all_gather_into_tensor=async_impl)
    )
    return fn


class DelayedGroup:
    """Two fake ranks; ``all_gather_into_tensor`` concatenates the send twice."""

    device_group = None

    def __init__(self):
        self.waited = 0

    def all_gather_into_tensor(self, out, send):
        out.copy_(torch.cat([send, send + 100]))


def delayed_async(group):
    """The bytes land only on wait(), standing in for the async collective."""

    def impl(out, send, **kwargs):
        def wait():
            group.waited += 1
            group.all_gather_into_tensor(out, send)

        return SimpleNamespace(wait=wait)

    return impl


class TestPackedGatherCompletion(unittest.TestCase):
    def test_chunked_output_is_ready_for_inline_read_or_ready_event(self):
        group = DelayedGroup()
        out = torch.full((10, 1), -777)
        send = torch.arange(5).view(-1, 1)
        plan = SimpleNamespace(dcp_size=2, send_rows=5, gathered_rows=10, chunks=3)
        load_gather(delayed_async(group))(group, out, send, plan)
        expected = torch.tensor([0, 100, 1, 101, 2, 3, 4, 102, 103, 104]).view(-1, 1)
        self.assertTrue(torch.equal(out, expected))
        self.assertEqual(group.waited, 3)

    def test_single_chunk_preserves_synchronous_path(self):
        group = DelayedGroup()
        out = torch.empty((10, 1), dtype=torch.int64)
        send = torch.arange(5).view(-1, 1)
        plan = SimpleNamespace(dcp_size=2, send_rows=5, gathered_rows=10, chunks=1)
        load_gather(delayed_async(group))(group, out, send, plan)
        self.assertTrue(torch.equal(out, torch.cat([send, send + 100])))
        self.assertEqual(group.waited, 0)


if __name__ == "__main__":
    unittest.main()
