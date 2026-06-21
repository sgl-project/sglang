import importlib.util
import sys
import types
from contextlib import nullcontext
from pathlib import Path

import torch


class _FakeMemorySaverAdapter:
    def region(self, _memory_type):
        return nullcontext()


class _FakeKVCache:
    def __init__(
        self,
        size,
        page_size,
        dtype,
        layer_num,
        device,
        enable_memory_saver,
        start_layer=None,
        end_layer=None,
    ):
        self.size = size
        self.page_size = page_size
        self.dtype = dtype
        self.store_dtype = dtype
        self.layer_num = layer_num
        self.device = device
        self.enable_memory_saver = enable_memory_saver
        self.start_layer = start_layer or 0
        self.end_layer = end_layer or layer_num - 1
        self.memory_saver_adapter = _FakeMemorySaverAdapter()
        self.mem_usage = 0

    def _finalize_allocation_log(self, _num_tokens):
        pass


class _FakeMHATokenToKVPool(_FakeKVCache):
    def __init__(
        self,
        size,
        page_size,
        dtype,
        head_num,
        head_dim,
        layer_num,
        device,
        enable_memory_saver,
        v_head_dim=None,
        swa_head_num=None,
        swa_head_dim=None,
        swa_v_head_dim=None,
        start_layer=None,
        end_layer=None,
        **_kwargs,
    ):
        super().__init__(
            size,
            page_size,
            dtype,
            layer_num,
            device,
            enable_memory_saver,
            start_layer,
            end_layer,
        )
        self.head_num = swa_head_num if swa_head_num is not None else head_num
        self.head_dim = swa_head_dim if swa_head_dim is not None else head_dim
        self.v_head_dim = (
            swa_v_head_dim
            if swa_v_head_dim is not None
            else v_head_dim if v_head_dim is not None else head_dim
        )
        self._create_buffers()


class _FakeMHATokenToKOnlyPool(_FakeKVCache):
    pass


class _FakeMiniMaxSparseKVPool:
    def __init__(self, *args, **kwargs):
        pass


class _FakeMLATokenToKVPool(_FakeKVCache):
    pass


def _load_npu_memory_pool_module():
    for name in (
        "sglang",
        "sglang.srt",
        "sglang.srt.constants",
        "sglang.srt.mem_cache",
        "sglang.srt.mem_cache.memory_pool",
        "sglang.srt.utils",
        "sglang.srt.utils.common",
    ):
        sys.modules.setdefault(name, types.ModuleType(name))

    constants = sys.modules["sglang.srt.constants"]
    constants.GPU_MEMORY_TYPE_KV_CACHE = "kv_cache"

    memory_pool = sys.modules["sglang.srt.mem_cache.memory_pool"]
    memory_pool.MHATokenToKVPool = _FakeMHATokenToKVPool
    memory_pool.MHATokenToKOnlyPool = _FakeMHATokenToKOnlyPool
    memory_pool.MiniMaxSparseKVPool = _FakeMiniMaxSparseKVPool
    memory_pool.MLATokenToKVPool = _FakeMLATokenToKVPool
    memory_pool.get_tensor_size_bytes = lambda tensor: tensor.nbytes
    memory_pool.maybe_detect_oob = lambda *args, **kwargs: None
    memory_pool.unwrap_write_loc = lambda loc_info: (loc_info, None)

    utils = sys.modules["sglang.srt.utils"]
    utils.get_bool_env_var = lambda _name, default: default == "True"

    common = sys.modules["sglang.srt.utils.common"]
    common.is_npu = lambda: False

    module_path = (
        Path(__file__).resolve().parents[2]
        / "python/sglang/srt/hardware_backend/npu/memory_pool_npu.py"
    )
    spec = importlib.util.spec_from_file_location(
        "_npu_memory_pool_under_test", module_path
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_npu_minimax_k_only_index_cache_uses_scatter_writer():
    npu_memory_pool = _load_npu_memory_pool_module()

    calls = []

    class FakeTorchNpu:
        @staticmethod
        def npu_scatter_nd_update_(cache, indices, updates):
            assert cache.shape == (10, 1, 4)
            assert indices.shape == (2, 1)
            assert updates.shape == (2, 1, 4)
            calls.append((cache, indices, updates))

        @staticmethod
        def _npu_reshape_and_cache(*, key, value, key_cache, value_cache, slot_indices):
            raise AssertionError("K-only MiniMax index cache should use scatter")

    npu_memory_pool.torch_npu = FakeTorchNpu

    pool = npu_memory_pool.NPUMHATokenToKOnlyPool(
        size=8,
        page_size=2,
        dtype=torch.bfloat16,
        head_num=1,
        head_dim=4,
        layer_num=1,
        device="cpu",
        enable_memory_saver=False,
    )

    loc = torch.tensor([1, 3], dtype=torch.int64)
    cache_k = torch.randn((2, 1, 4), dtype=torch.bfloat16)

    pool.set_k_buffer(0, loc, cache_k)

    assert len(calls) == 1
    k_size, v_size = pool.get_kv_size_bytes()
    assert k_size > 0
    assert v_size == 0
