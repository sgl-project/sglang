"""Packed model payloads need no storage; native CPU allocation state does."""

import ast
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from sglang_simulator.simulation.sglang.model_runner import C_DSV4PayloadHook

SOURCE = (
    Path(__file__).resolve().parents[3]
    / "python/sglang/srt/mem_cache/deepseek_v4_memory_pool.py"
)


def native_method(class_name, method_name, namespace):
    tree = ast.parse(SOURCE.read_text(encoding="utf-8"))
    owner = next(
        n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == class_name
    )
    method = next(
        n
        for n in owner.body
        if isinstance(n, ast.FunctionDef) and n.name == method_name
    )
    module = ast.Module(body=[method], type_ignores=[])
    # Execute the checkout's unchanged allocation method without GPU imports.
    exec(compile(module, str(SOURCE), "exec"), namespace)  # noqa: S102
    return namespace[method_name]


@pytest.fixture
def cpu_simulator(monkeypatch):
    monkeypatch.setenv("SGLANG_SIMULATOR_BOOTSTRAP", "1")
    monkeypatch.setenv("SGLANG_USE_CPU_ENGINE", "1")


def test_real_uniform_payload_keeps_native_large_geometry_without_storage(
    cpu_simulator,
):
    method = native_method(
        "DeepSeekV4UniformFP8KVPool", "create_buffer", {"torch": torch}
    )
    pool_type = type("DeepSeekV4UniformFP8KVPool", (), {"create_buffer": method})
    C_DSV4PayloadHook.hook(pool_type)
    pool = pool_type()
    pool.device = torch.device("cpu")
    pool.page_size = 256
    pool.get_bytes_per_token = lambda: 512
    # This exact native method describes almost 30 GB of value payload.
    payload = pool.create_buffer(num_pages=58_032_128 // 256)
    assert payload.device.type == "meta"
    assert payload.shape == (226_688, 131_072)
    assert payload.dtype == torch.float8_e4m3fn
    assert payload.untyped_storage().data_ptr() == 0
    assert pool.device == torch.device("cpu")
    assert pool.kv_cache_total_dim == 512
    assert pool.bytes_per_page_padded == 131_072
    # A real integer allocator remains independently addressable on CPU.
    slots = torch.arange(5, dtype=torch.int64)
    assert slots.device.type == "cpu" and slots.tolist() == list(range(5))


@pytest.mark.parametrize("split", [False, True])
def test_real_indexer_payload_shapes_dtype_and_pool_identity(cpu_simulator, split):
    namespace = {
        "torch": torch,
        "nullcontext": nullcontext,
        "GPU_MEMORY_TYPE_KV_CACHE": "kv",
        "_num_dsv4_physical_kv_pages": lambda size, physical, logical: (
            (size + logical + physical - 1) // physical
        ),
    }
    method = native_method("DeepSeekV4IndexerPool", "_create_buffer", namespace)
    pool_type = type("DeepSeekV4IndexerPool", (), {"_create_buffer": method})
    C_DSV4PayloadHook.hook(pool_type)
    pool = pool_type()
    pool.device = "cpu"
    pool.size = 58_032_128
    pool.page_size = pool.global_page_size = 256
    pool.layer_num = 2
    pool.get_bytes_per_token = lambda: 132
    pool.memory_saver_adapter = SimpleNamespace(region=lambda _: nullcontext())
    pool.custom_mem_pool = None
    pool.uses_aiter_fp4_layout = split
    pool.index_k_with_scale_buffer_dtype = torch.uint8
    pool._create_buffer()
    buffers = (
        pool.index_k_payload_buffer + pool.index_k_scale_buffer
        if split
        else pool.index_k_with_scale_buffer
    )
    assert len(buffers) == (4 if split else 2)
    assert all(
        b.device.type == "meta" and b.untyped_storage().data_ptr() == 0 for b in buffers
    )
    assert pool.device == "cpu" and pool.size == 58_032_128
    assert buffers[0].shape[0] == 226_689
    if split:
        assert buffers[0].dtype == torch.float4_e2m1fn_x2
        assert pool.index_k_scale_buffer[0].shape == (226_689, 1, 4, 256)
    else:
        assert buffers[0].shape == (226_689, 33_792)


def test_pool_device_restored_and_original_called_once_on_failure(cpu_simulator):
    calls = []

    def create_buffer(self, *, num_pages):
        calls.append((self.device, num_pages))
        raise RuntimeError("native payload construction failed")

    target = type("DeepSeekV4SingleKVPool", (), {"create_buffer": create_buffer})
    C_DSV4PayloadHook.hook(target)
    pool = target()
    pool.device = "cpu"
    with pytest.raises(RuntimeError, match="native payload construction"):
        pool.create_buffer(num_pages=7)
    assert calls == [("meta", 7)] and pool.device == "cpu"


@pytest.mark.parametrize(
    "bootstrap,cpu,device", [("0", "1", "cpu"), ("1", "0", "cpu"), ("1", "1", "cuda:0")]
)
def test_non_simulator_and_cuda_calls_delegate_unchanged(
    monkeypatch, bootstrap, cpu, device
):
    monkeypatch.setenv("SGLANG_SIMULATOR_BOOTSTRAP", bootstrap)
    monkeypatch.setenv("SGLANG_USE_CPU_ENGINE", cpu)
    calls = []
    sentinel = object()

    def create_buffer(self, *, num_pages):
        calls.append((self.device, num_pages))
        return sentinel

    target = type("DeepSeekV4SingleKVPool", (), {"create_buffer": create_buffer})
    C_DSV4PayloadHook.hook(target)
    pool = target()
    pool.device = device
    assert pool.create_buffer(num_pages=3) is sentinel
    assert calls == [(device, 3)] and pool.device == device


@pytest.mark.parametrize("fp8", [False, True])
def test_real_unified_payload_preserves_native_geometry_and_dtype(cpu_simulator, fp8):
    namespace = {
        "torch": torch,
        "nullcontext": nullcontext,
        "List": list,
        "GPU_MEMORY_TYPE_KV_CACHE": "kv",
        "DSV4_FP8_NOPE_ROW_BYTES": 512,
    }
    initialize = native_method("DeepSeekV4UnifiedKVPool", "__init__", namespace)
    owner = next(
        n
        for n in ast.parse(SOURCE.read_text(encoding="utf-8")).body
        if isinstance(n, ast.ClassDef) and n.name == "DeepSeekV4UnifiedKVPool"
    )
    constants = next(
        ast.literal_eval(n.value)
        for n in owner.body
        if isinstance(n, ast.Assign)
        and any(isinstance(t, ast.Name) and t.id == "K_PER_BLOCK" for t in n.targets)
    )
    target = type(
        "DeepSeekV4UnifiedKVPool",
        (),
        {"__init__": initialize, "K_PER_BLOCK": constants},
    )
    C_DSV4PayloadHook.hook(target)
    pool = target(
        stage_ratios=[0, 4, 128],
        num_slots=3,
        num_blocks=226_688,
        page_size=256,
        qk_nope_head_dim=448,
        qk_rope_head_dim=64,
        device="cpu",
        memory_saver_adapter=SimpleNamespace(region=lambda _: nullcontext()),
        custom_mem_pool=None,
        swa_ring_size=512,
        fp8=fp8,
    )
    assert pool.swa_pages == 1536 and pool.num_blocks == 226_688
    assert pool.head_dim == 512 and pool.rope_head_dim == 64
    assert [b.shape for b in pool.kv_buffer] == [
        (1536, 512),
        (7_255_616, 512),
        (228_226, 512),
    ]
    assert all(b.device.type == "meta" for b in pool.kv_buffer)
    assert all(b.untyped_storage().data_ptr() == 0 for b in pool.kv_buffer)
    assert all(
        b.dtype == (torch.float8_e4m3fn if fp8 else torch.bfloat16)
        for b in pool.kv_buffer
    )
    if fp8:
        assert all(b.device.type == "meta" for b in pool.kv_buffer_rope)
        assert all(b.dtype == torch.bfloat16 for b in pool.kv_buffer_rope)
        assert [b.shape for b in pool.kv_buffer_rope] == [
            (1536, 64),
            (7_255_616, 64),
            (228_226, 64),
        ]
    else:
        assert pool.kv_buffer_rope == [None, None, None]
    # The payload-only hook never redirects subsequent native integer state.
    slots = torch.arange(3)
    assert slots.device.type == "cpu" and slots.tolist() == [0, 1, 2]
