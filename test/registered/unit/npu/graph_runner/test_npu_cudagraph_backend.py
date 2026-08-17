"""
Unit tests for sglang.srt.hardware_backend.npu.graph_runner.npu_cudagraph_backend.
"""

import importlib.util
import sys
import types
import unittest
from contextlib import nullcontext
from pathlib import Path
from unittest.mock import MagicMock

import torch

# Ensure torch.npu is mockable on CPU hosts.
if not hasattr(torch, "npu"):
    torch.npu = MagicMock()

# ---------------------------------------------------------------------------
# Mock the entire sglang package tree.
# ---------------------------------------------------------------------------
for _pkg in (
    "sglang",
    "sglang.test",
    "sglang.test.ci",
    "sglang.srt",
    "sglang.srt.configs",
    "sglang.srt.distributed",
    "sglang.srt.distributed.device_communicators",
    "sglang.srt.model_executor",
    "sglang.srt.model_executor.runner_backend",
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

# ---------------------------------------------------------------------------
# Stub dependencies.
# ---------------------------------------------------------------------------
_const_mod = types.ModuleType("sglang.srt.constants")
_const_mod.GPU_MEMORY_TYPE_CUDA_GRAPH = "gpu_memory_type_cuda_graph"
sys.modules["sglang.srt.constants"] = _const_mod

_pynccl_mod = types.ModuleType(
    "sglang.srt.distributed.device_communicators.pynccl_allocator"
)


def set_graph_pool_id(pool_id):
    pass


_pynccl_mod.set_graph_pool_id = set_graph_pool_id
sys.modules["sglang.srt.distributed.device_communicators.pynccl_allocator"] = (
    _pynccl_mod
)

_shapekey_mod = types.ModuleType("sglang.srt.model_executor.runner.shape_key")


class ShapeKey:
    def __init__(self, size=None):
        self.size = size

    def __hash__(self):
        return hash(self.size)

    def __eq__(self, other):
        return isinstance(other, ShapeKey) and self.size == other.size


_shapekey_mod.ShapeKey = ShapeKey
sys.modules["sglang.srt.model_executor.runner.shape_key"] = _shapekey_mod

_base_mod = types.ModuleType(
    "sglang.srt.model_executor.runner_backend.base_cuda_graph_backend"
)


class BaseCudaGraphBackend:
    pass


_base_mod.BaseCudaGraphBackend = BaseCudaGraphBackend
sys.modules["sglang.srt.model_executor.runner_backend.base_cuda_graph_backend"] = (
    _base_mod
)

_utils_mod = types.ModuleType("sglang.srt.utils")


def empty_context():
    return nullcontext()


def get_bool_env_var(name, default="False"):
    return False


_utils_mod.empty_context = empty_context
_utils_mod.get_bool_env_var = get_bool_env_var
sys.modules["sglang.srt.utils"] = _utils_mod

_msaver_mod = types.ModuleType("sglang.srt.utils.torch_memory_saver_adapter")


class TorchMemorySaverAdapter:
    def __init__(self):
        self.enabled = False

    @classmethod
    def create(cls, enable=False):
        return cls()


_msaver_mod.TorchMemorySaverAdapter = TorchMemorySaverAdapter
sys.modules["sglang.srt.utils.torch_memory_saver_adapter"] = _msaver_mod

# Mock torch_npu for capture_one's `import torch_npu`.
sys.modules.setdefault("torch_npu", MagicMock())

# ---------------------------------------------------------------------------
# Load the target module directly from file.
# ---------------------------------------------------------------------------
_TARGET_FILE = (
    Path(__file__).resolve().parents[5]
    / "python"
    / "sglang"
    / "srt"
    / "hardware_backend"
    / "npu"
    / "graph_runner"
    / "npu_cudagraph_backend.py"
)

_spec = importlib.util.spec_from_file_location(
    "sglang.srt.hardware_backend.npu.graph_runner.npu_cudagraph_backend",
    str(_TARGET_FILE),
)
_target_mod = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = _target_mod
_spec.loader.exec_module(_target_mod)

NPUCudaGraphBackend = _target_mod.NPUCudaGraphBackend


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_backend(graphs=None, outputs=None):
    """Create an NPUCudaGraphBackend with a mocked cuda_graph_runner."""
    cgr = MagicMock()
    cgr.device_module.current_device.return_value = 0
    cgr.model_runner.tp_group = MagicMock()
    cgr.enable_torch_compile = False
    backend = NPUCudaGraphBackend(cgr, enable_memory_saver=False)
    if graphs is not None:
        backend._graphs = graphs
    if outputs is not None:
        backend._outputs = outputs
    return backend


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestNPUCudaGraphBackend(unittest.TestCase):
    def test_init(self):
        cgr = MagicMock()
        cgr.device_module.current_device.return_value = 5
        cgr.model_runner.tp_group = "tp"
        cgr.enable_torch_compile = True
        backend = NPUCudaGraphBackend(cgr, enable_memory_saver=False)
        self.assertEqual(backend._graphs, {})
        self.assertEqual(backend._outputs, {})
        self.assertIsNone(backend._pool)
        self.assertEqual(backend._device_id, 5)
        self.assertEqual(backend._tp_group, "tp")
        self.assertIsNone(backend._capture_stream)
        self.assertTrue(backend._enable_torch_compile)

    def test_capture_session(self):
        """Pool is created once and cached; stream set during, cleared after."""
        backend = _make_backend()
        stream = MagicMock(name="stream")
        with backend.capture_session(stream):
            self.assertIsNotNone(backend._pool)
            self.assertIs(backend._capture_stream, stream)
        self.assertIsNone(backend._capture_stream)
        pool = backend._pool
        with backend.capture_session(stream):
            self.assertIs(backend._pool, pool)
        self.assertIsNone(backend._capture_stream)

    def test_capture_one(self):
        """Two warmups + one capture; graph and output stored."""
        backend = _make_backend()
        forward_fn = MagicMock(return_value="output")
        post_warmup = MagicMock()
        backend.capture_one("key", forward_fn, post_warmup_hook=post_warmup)
        self.assertEqual(forward_fn.call_count, 3)
        self.assertEqual(post_warmup.call_count, 2)
        self.assertIn("key", backend._graphs)
        self.assertEqual(backend._outputs["key"], "output")

    def test_capture_one_no_warmup_hook(self):
        """capture_one works without post_warmup_hook (None branch)."""
        backend = _make_backend()
        forward_fn = MagicMock(return_value="output")
        backend.capture_one("key", forward_fn, post_warmup_hook=None)
        self.assertEqual(forward_fn.call_count, 3)
        self.assertIn("key", backend._graphs)

    def test_can_run(self):
        backend = _make_backend(graphs={"a": MagicMock()}, outputs={})
        self.assertTrue(backend.can_run(MagicMock(), "a"))
        self.assertFalse(backend.can_run(MagicMock(), "b"))

    def test_replay(self):
        graph = MagicMock()
        out = MagicMock(name="out")
        backend = _make_backend(graphs={"k": graph}, outputs={"k": out})
        result = backend.replay("k", MagicMock())
        graph.replay.assert_called_once()
        self.assertIs(result, out)

    def test_replay_with_input_update_legacy(self):
        """Legacy: seq_lens + attr_name + attr_type -> constructs cpu_update_input."""
        graph = MagicMock()
        out = MagicMock(name="out")
        backend = _make_backend(graphs={"k": graph}, outputs={"k": out})
        result = backend.replay_with_input_update(
            "k", seq_lens=[10, 20], attr_name="actual_seq_lengths_kv", attr_type=[]
        )
        update_kwargs = graph.update.call_args.kwargs
        self.assertEqual(
            update_kwargs["cpu_update_input"],
            [{"actual_seq_lengths_kv": [10, 20]}],
        )
        graph.replay.assert_called_once()
        self.assertIs(result, out)

    def test_replay_with_input_update_direct(self):
        """Direct: caller provides cpu_update_input list (EAGLE multi-step)."""
        graph = MagicMock()
        out = MagicMock(name="out")
        backend = _make_backend(graphs={"k": graph}, outputs={"k": out})
        cpu_update_input = [
            {"actual_seq_lengths_kv": [11, 21, 0, 0]},
            {"actual_seq_lengths_kv": [12, 22, 0, 0]},
        ]
        result = backend.replay_with_input_update(
            "k", seq_lens=None, cpu_update_input=cpu_update_input
        )
        update_kwargs = graph.update.call_args.kwargs
        self.assertIs(update_kwargs["cpu_update_input"], cpu_update_input)
        graph.replay.assert_called_once()
        self.assertIs(result, out)

    def test_replay_with_input_update_tensor_conversion(self):
        """attr_type is torch.Tensor -> seq_lens converted to int32 tensor."""
        graph = MagicMock()
        backend = _make_backend(graphs={"k": graph}, outputs={"k": "out"})
        backend.replay_with_input_update(
            "k",
            seq_lens=[10, 20],
            attr_name="ctx_lens",
            attr_type=torch.Tensor(),
        )
        converted = graph.update.call_args.kwargs["cpu_update_input"][0]["ctx_lens"]
        self.assertIsInstance(converted, torch.Tensor)
        self.assertEqual(converted.dtype, torch.int32)
        self.assertTrue(
            torch.equal(converted, torch.tensor([10, 20], dtype=torch.int32))
        )

    def test_cleanup(self):
        backend = _make_backend(graphs={"a": MagicMock()}, outputs={"a": "out"})
        backend._pool = MagicMock()
        backend.cleanup()
        self.assertEqual(backend._graphs, {})
        self.assertEqual(backend._outputs, {})
        self.assertIsNone(backend._pool)


if __name__ == "__main__":
    unittest.main()
