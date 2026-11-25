"""CPU checks for the bounded DCP runtime headroom."""

import ast
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch

from sglang.srt.hardware_backend.npu.dcp_extend_gather import dcp_packed_gather_capacity

ROOT = Path(__file__).resolve().parents[3]
SOURCE = ROOT / "python/sglang/srt/hardware_backend/npu/attention/dcp_memory.py"


class TestReserve(unittest.TestCase):
    def reserve(self, *, prefetch=False, ep=1, streamed=False):
        flags = {
            "SGLANG_NPU_ENABLE_DCP_PACKED_GATHER_PREFETCH": prefetch,
            "SGLANG_NPU_ENABLE_DCP_PACKED_READ": True,
            "SGLANG_NPU_SFA_KV_8BIT": True,
            "SGLANG_NPU_ENABLE_DCP_SFA_GATHER_OVERLAP": False,
            "SGLANG_NPU_DCP_EXTEND_GATHER_PIECE_ROWS": 262144,
        }
        envs = SimpleNamespace(
            **{k: SimpleNamespace(get=lambda v=v: v) for k, v in flags.items()}
        )
        projection = torch.empty(1, dtype=torch.bfloat16)
        layer = SimpleNamespace(
            use_dsa=True,
            o_proj=SimpleNamespace(output_size=7168),
            kv_lora_rank=512,
            qk_rope_head_dim=64,
        )
        setattr(layer, "w_vc_stream_shard" if streamed else "w_vc_qrep", projection)
        namespace = dict(
            envs=envs,
            dcp_packed_gather_capacity=dcp_packed_gather_capacity,
            get_exec=lambda: SimpleNamespace(moe=SimpleNamespace(fuseep_mode=0)),
            get_parallel=lambda: SimpleNamespace(attn_tp_size=16, moe_ep_size=ep),
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
            prefill_requests=1,
        )

    def test_streamed_projection_still_reserves_runtime_storage(self):
        self.assertGreater(self.reserve(streamed=True), 1 << 30)
        self.assertEqual(self.reserve(streamed=True), self.reserve())

    def test_ep_native_headroom_and_prefetch_lifetimes(self):
        self.assertEqual(self.reserve(ep=16) - self.reserve(), 1 << 30)
        self.assertGreater(self.reserve(prefetch=True), 1 << 30)
        self.assertNotEqual(self.reserve(prefetch=True), self.reserve())


if __name__ == "__main__":
    unittest.main()
