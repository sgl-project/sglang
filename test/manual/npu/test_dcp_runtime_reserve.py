"""CPU checks for the bounded DCP runtime headroom."""

import ast
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch

ROOT = Path(__file__).resolve().parents[3]
SOURCE = ROOT / "python/sglang/srt/hardware_backend/npu/attention/dcp_memory.py"


class TestReserve(unittest.TestCase):
    def reserve(self, *, ep=1, streamed=False, prefill_requests=1):
        projection = torch.empty(1, dtype=torch.bfloat16)
        layer = SimpleNamespace(
            use_dsa=True,
            o_proj=SimpleNamespace(output_size=7168),
            kv_lora_rank=512,
            qk_rope_head_dim=64,
        )
        setattr(layer, "w_vc_stream_shard" if streamed else "w_vc_qrep", projection)
        namespace = dict(
            get_parallel=lambda: SimpleNamespace(attn_tp_size=16, moe_ep_size=ep)
        )
        tree = ast.parse(SOURCE.read_text())
        functions = [node for node in tree.body if isinstance(node, ast.FunctionDef)]
        exec(
            compile(ast.Module(body=functions, type_ignores=[]), str(SOURCE), "exec"),
            namespace,
        )
        return namespace["runtime_gather_reserve_bytes"](
            SimpleNamespace(modules=lambda: [layer]),
            8192,
            context_tokens=1048576,
            prefill_requests=prefill_requests,
        )

    def test_streamed_projection_still_reserves_runtime_storage(self):
        self.assertGreater(self.reserve(streamed=True), 1 << 30)
        self.assertEqual(self.reserve(streamed=True), self.reserve())

    def test_ep_native_headroom(self):
        self.assertEqual(self.reserve(ep=16) - self.reserve(), 1 << 30)

    def test_multi_request_extends_hold_two_prefix_copies(self):
        # Single-request extends admit the packed read (one prefix copy); with
        # more requests the generic layout needs the contiguous output plus the
        # rank-major scratch, so the reserve more than doubles.
        self.assertGreater(self.reserve(prefill_requests=2), 2 * self.reserve())


if __name__ == "__main__":
    unittest.main()
