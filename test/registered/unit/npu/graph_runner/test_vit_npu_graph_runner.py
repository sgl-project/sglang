"""
Unit tests for sglang.srt.hardware_backend.npu.graph_runner.vit_npu_graph_runner.
"""

import importlib.util
import sys
import types
import unittest
from pathlib import Path
from unittest.mock import MagicMock

import torch

if not hasattr(torch, "npu"):
    torch.npu = MagicMock()

for _pkg in (
    "sglang",
    "sglang.test",
    "sglang.test.ci",
    "sglang.srt",
    "sglang.srt.configs",
    "sglang.srt.distributed",
    "sglang.srt.distributed.device_communicators",
    "sglang.srt.layers",
    "sglang.srt.layers.attention",
    "sglang.srt.multimodal",
    "sglang.srt.hardware_backend",
    "sglang.srt.hardware_backend.npu",
    "sglang.srt.hardware_backend.npu.graph_runner",
):
    sys.modules.setdefault(_pkg, MagicMock())

_ci_mod = types.ModuleType("sglang.test.ci.ci_register")
_ci_mod.register_npu_ci = lambda *a, **kw: None
sys.modules.setdefault("sglang.test.ci.ci_register", _ci_mod)

from sglang.test.ci.ci_register import register_npu_ci  # noqa: E402

register_npu_ci(est_time=5, suite="base-a-test-1-npu-a2")

sys.modules.setdefault("torch_npu", MagicMock())

_pynccl_mod = types.ModuleType(
    "sglang.srt.distributed.device_communicators.pynccl_allocator"
)


def set_graph_pool_id(pool_id):
    pass


_pynccl_mod.set_graph_pool_id = set_graph_pool_id
sys.modules["sglang.srt.distributed.device_communicators.pynccl_allocator"] = (
    _pynccl_mod
)

_va_mod = types.ModuleType("sglang.srt.layers.attention.vision")


class VisionAttention:
    pass


_va_mod.VisionAttention = VisionAttention
sys.modules["sglang.srt.layers.attention.vision"] = _va_mod

_parent_mod = types.ModuleType("sglang.srt.multimodal.vit_cuda_graph_runner")


class ViTCudaGraphRunner:
    _graph_memory_pool = None

    def __init__(self, vit):
        self.vit = vit
        self.block_graphs = {}
        self.block_output = {}
        self.block_input = {}
        self.block_ws = {}
        self._attn_backend = "ascend_attn"
        self._deepstack_visual_indexes = None
        self._deepstack_merger_list = None

    def _get_graph_key(self, x_3d):
        return x_3d.shape[0]


_parent_mod.ViTCudaGraphRunner = ViTCudaGraphRunner
sys.modules["sglang.srt.multimodal.vit_cuda_graph_runner"] = _parent_mod

_TARGET_FILE = (
    Path(__file__).resolve().parents[5]
    / "python"
    / "sglang"
    / "srt"
    / "hardware_backend"
    / "npu"
    / "graph_runner"
    / "vit_npu_graph_runner.py"
)

_spec = importlib.util.spec_from_file_location(
    "sglang.srt.hardware_backend.npu.graph_runner.vit_npu_graph_runner",
    str(_TARGET_FILE),
)
_target_mod = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = _target_mod
_spec.loader.exec_module(_target_mod)

ViTNpuGraphRunner = _target_mod.ViTNpuGraphRunner


def _make_vit(num_blocks=3, num_heads=8, head_dim=64):
    vit = MagicMock()
    vit.device = torch.device("cpu")
    vit.dtype = torch.float32
    blocks = []
    for _ in range(num_blocks):
        blk = MagicMock(return_value=torch.randn(10, 1, 64))
        blk.attn.num_attention_heads_per_partition = num_heads
        blk.attn.head_size = head_dim
        blocks.append(blk)
    vit.blocks = blocks
    vit.merger = MagicMock(return_value=torch.randn(10, 64))
    return vit


def _make_runner(vit=None):
    if vit is None:
        vit = _make_vit()
    r = object.__new__(ViTNpuGraphRunner)
    r.vit = vit
    r.device_module = MagicMock()
    r.cu_seq_lens = {}
    r.sin_cos_ws = {}
    r.block_graphs = {}
    r.block_output = {}
    r.block_input = {}
    r.block_ws = {}
    r._attn_backend = "ascend_attn"
    r._deepstack_visual_indexes = None
    r._deepstack_merger_list = None
    return r


def _setup_capture(runner, graph_key=10):
    """Pre-set all buffers _create_graph reads."""
    runner.block_input[graph_key] = torch.randn(graph_key, 1, 64)
    runner.block_ws[graph_key] = torch.empty(graph_key, 8, 64)
    runner.cu_seq_lens[graph_key] = torch.tensor([5, graph_key], dtype=torch.int32)
    runner.sin_cos_ws[graph_key] = (
        torch.randn(graph_key, 32),
        torch.randn(graph_key, 32),
    )
    return runner


class TestViTNpuGraphRunner(unittest.TestCase):
    def setUp(self):
        ViTNpuGraphRunner._graph_memory_pool = None

    def test_init(self):
        vit = _make_vit()
        r = ViTNpuGraphRunner(vit)
        self.assertIsNotNone(r.device_module)
        self.assertEqual(r.cu_seq_lens, {})
        self.assertEqual(r.sin_cos_ws, {})

    def test_device_and_dtype(self):
        r = _make_runner()
        self.assertEqual(r.device, torch.device("cpu"))
        self.assertEqual(r.dtype, torch.float32)

    # --- _create_graph (source order: before create_graph) ---

    def test_create_graph_capture_basic(self):
        """Blocks called in order, merger called, graph and output stored."""
        r = _setup_capture(_make_runner(vit=_make_vit(num_blocks=3)))
        r._create_graph(graph_key=10)
        for blk in r.vit.blocks:
            blk.assert_called_once()
        r.vit.merger.assert_called_once()
        self.assertIn(10, r.block_graphs)
        self.assertIs(r.block_output[10], r.vit.merger.return_value)

    def test_create_graph_first_block_uses_block_input(self):
        """Layer 0 gets block_input, layer 1+ gets previous block's output."""
        r = _setup_capture(_make_runner(vit=_make_vit(num_blocks=2)))
        r._create_graph(graph_key=10)
        self.assertIs(r.vit.blocks[0].call_args.args[0], r.block_input[10])
        self.assertIs(r.vit.blocks[1].call_args.args[0], r.vit.blocks[0].return_value)

    def test_create_graph_unsupported_backend(self):
        """Non-ascend_attn backend raises RuntimeError."""
        r = _setup_capture(_make_runner())
        r._attn_backend = "flashinfer"
        with self.assertRaises(RuntimeError):
            r._create_graph(graph_key=10)

    def test_create_graph_deepstack(self):
        """Deepstack: merger called at specified layers, output = cat([main, deepstack])."""
        r = _setup_capture(_make_runner(vit=_make_vit(num_blocks=3)))
        r._deepstack_visual_indexes = [1]
        deepstack_merger = MagicMock(return_value=torch.randn(10, 32))
        r._deepstack_merger_list = [deepstack_merger]
        r._create_graph(graph_key=10)
        deepstack_merger.assert_called_once()
        self.assertEqual(r.block_output[10].shape, (10, 96))

    def test_create_graph_deepstack_missing_merger(self):
        """deepstack_visual_indexes set but merger_list is None -> RuntimeError."""
        r = _setup_capture(_make_runner(vit=_make_vit(num_blocks=3)))
        r._deepstack_visual_indexes = [1]
        r._deepstack_merger_list = None
        with self.assertRaises(RuntimeError):
            r._create_graph(graph_key=10)

    # --- create_graph ---

    def test_create_graph_workspace(self):
        r = _make_runner()
        x_3d = torch.randn(10, 1, 64)
        cu_seqlens = torch.tensor([0, 5, 10])
        graph_key = r.create_graph(x_3d, cu_seqlens)
        self.assertEqual(graph_key, 10)
        self.assertIs(r.block_input[10], x_3d)
        self.assertEqual(r.block_ws[10].shape, (10, 8, 64))
        self.assertTrue(
            torch.equal(
                r.cu_seq_lens[10],
                cu_seqlens[1:].to("cpu").to(torch.int32),
            )
        )
        self.assertNotIn(10, r.block_graphs)

    def test_create_graph_skips_existing(self):
        r = _make_runner()
        r.block_graphs[10] = MagicMock()
        result = r.create_graph(
            torch.randn(10, 1, 64),
            torch.tensor([0, 5, 10]),
            rotary_pos_emb_cos=torch.randn(10),
            rotary_pos_emb_sin=torch.randn(10),
        )
        self.assertEqual(result, 10)
        self.assertNotIn(10, r.block_input)

    # --- replay ---

    def test_replay(self):
        r = _make_runner()
        graph = MagicMock()
        input_buf = torch.randn(10, 1, 64)
        output_buf = torch.randn(10, 64)
        r.block_graphs[10] = graph
        r.block_input[10] = input_buf.clone()
        r.block_output[10] = output_buf
        result = r.replay(10, torch.randn(10, 1, 64))
        graph.replay.assert_called_once()
        self.assertTrue(torch.equal(result, output_buf))

    def test_replay_output_indices(self):
        r = _make_runner()
        output_buf = torch.randn(10, 64)
        r.block_graphs[10] = MagicMock()
        r.block_input[10] = torch.randn(10, 1, 64)
        r.block_output[10] = output_buf
        indices = torch.tensor([3, 1, 7])
        result = r.replay(10, torch.randn(10, 1, 64), output_indices=indices)
        self.assertTrue(torch.equal(result, output_buf.index_select(0, indices)))

    def test_replay_updates_rotary(self):
        r = _make_runner()
        r.block_graphs[10] = MagicMock()
        r.block_input[10] = torch.randn(10, 1, 64)
        r.block_output[10] = torch.randn(10, 64)
        r.sin_cos_ws[10] = (torch.randn(10, 32), torch.randn(10, 32))
        new_cos, new_sin = torch.randn(10, 32), torch.randn(10, 32)
        r.replay(
            10,
            torch.randn(10, 1, 64),
            rotary_pos_emb_cos=new_cos,
            rotary_pos_emb_sin=new_sin,
        )
        self.assertTrue(torch.equal(r.sin_cos_ws[10][0], new_cos))
        self.assertTrue(torch.equal(r.sin_cos_ws[10][1], new_sin))

    # --- run ---

    def test_run_first_call_creates(self):
        r = _make_runner()
        r._get_graph_key = lambda x_3d: x_3d.shape[0]
        r.create_graph = MagicMock(return_value=10)
        r.replay = MagicMock(return_value="output")
        result = r.run(
            torch.randn(10, 64),
            torch.tensor([0, 5, 10]),
            rotary_pos_emb_cos=torch.randn(10, 32),
            rotary_pos_emb_sin=torch.randn(10, 32),
        )
        r.create_graph.assert_called_once()
        r.replay.assert_called_once()
        self.assertEqual(result, "output")

    def test_run_second_call_replays(self):
        r = _make_runner()
        r._get_graph_key = lambda x_3d: x_3d.shape[0]
        r.block_graphs[10] = MagicMock()
        r.create_graph = MagicMock()
        r.replay = MagicMock(return_value="output")
        result = r.run(torch.randn(10, 64), torch.tensor([0, 5, 10]))
        r.create_graph.assert_not_called()
        r.replay.assert_called_once()
        self.assertEqual(result, "output")


if __name__ == "__main__":
    unittest.main()
