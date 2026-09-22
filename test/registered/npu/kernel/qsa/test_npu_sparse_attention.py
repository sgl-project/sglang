"""Strict sparse-attention dispatch, cache adaptation and real backend calls."""
import importlib
from types import SimpleNamespace

import pytest
import torch

from sglang.srt.layers.attention import qwen_sparse_attn_backend as backend_module
from sglang.srt.layers.attention.qsa import kernel
from sglang.srt.layers.attention.qwen_sparse_attn_backend import QwenSparseAttnBackend
from sglang.srt.model_executor.forward_batch_info import ForwardMode
from sglang.srt.utils import is_npu
from sglang.test.ci.ci_register import register_npu_ci

register_npu_ci(est_time=60, suite="base-b-test-1-npu-a3")
pytestmark = pytest.mark.skipif(not is_npu(), reason="NPU is required")
impl = importlib.import_module("sgl_kernel_npu.qwen3_8_flash_next.sparse_attention")


def inputs(heads=3, kv_heads=1):
    torch.manual_seed(997)
    q = torch.randn(4, heads, 256, device="npu", dtype=torch.bfloat16)
    k = torch.randn(64, kv_heads, 256, device="npu", dtype=q.dtype)
    v = torch.randn_like(k)
    slots = torch.full((4, 2051), -1, device="npu", dtype=torch.int32)
    slots[1:, :17] = torch.arange(17, device="npu", dtype=torch.int32)
    return q, k, v, slots


@pytest.mark.parametrize("heads,kv_heads", [(3, 1), (6, 1), (12, 1), (24, 2)])
@pytest.mark.parametrize("layout", ["flat", "paged", "fia"])
def test_dispatch_graph(heads, kv_heads, layout, monkeypatch):
    q, k, v, s = inputs(heads, kv_heads)
    if layout == "paged":
        k, v = k.reshape(4, 16, kv_heads, 256), v.reshape(4, 16, kv_heads, 256)
    elif layout == "fia":
        k, v = k.unsqueeze(1), v.unsqueeze(1)
    raw = impl.sparse_attention
    cpu_reference = kernel.qsa_sparse_attention_reference
    calls = []
    def traced(*args):
        assert args[1].data_ptr() == k.data_ptr()
        assert args[2].data_ptr() == v.data_ptr()
        calls.append((args[1].shape, args[2].shape))
        return raw(*args)
    def forbidden(*args, **kwargs):
        raise AssertionError("NPU production must not call the Torch reference")
    monkeypatch.setattr(impl, "sparse_attention", traced)
    monkeypatch.setattr(kernel, "qsa_sparse_attention_reference", forbidden)
    def forward():
        return backend_module._npu_sparse_attention(q, k, v, s)
    for _ in range(2): forward()
    torch.npu.synchronize()
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph):
        out = forward()
    count = len(calls)
    pointers = [x.data_ptr() for x in (q, k, v, s)]
    for change in (None, q, k, v, s):
        if change is s:
            s.fill_(-1)
            s[1:, :5] = 2
        elif change is not None:
            change.neg_()
        graph.replay()
        torch.npu.synchronize()
        assert len(calls) == count
        expected = cpu_reference(q.cpu(), k.reshape(64, kv_heads, 256).cpu(),
                                 v.reshape(64, kv_heads, 256).cpu(), s.cpu())
        torch.testing.assert_close(out.cpu(), expected, atol=0.02, rtol=0.02)
        assert pointers == [x.data_ptr() for x in (q, k, v, s)]
    assert all(a == b == (64, kv_heads, 256) for a, b in calls)


@pytest.mark.parametrize("case", ["fp16", "fp32", "heads", "dimension", "slots_dtype", "width", "cpu"])
def test_invalid_metadata_never_falls_back(case, monkeypatch):
    q, k, v, s = inputs()
    if case in ("fp16", "fp32"):
        dtype = torch.float16 if case == "fp16" else torch.float32
        q, k, v = (x.to(dtype) for x in (q, k, v))
    elif case == "heads": q = q[:, :2].contiguous()
    elif case == "dimension": q, k, v = (x[..., :128].contiguous() for x in (q, k, v))
    elif case == "slots_dtype": s = s.long()
    elif case == "width": s = s[:, :33]
    elif case == "cpu": q = q.cpu()
    def forbidden(*args, **kwargs):
        raise AssertionError("unsupported metadata must not fall back")
    monkeypatch.setattr(kernel, "qsa_sparse_attention_reference", forbidden)
    with pytest.raises(ValueError):
        backend_module._npu_sparse_attention(q, k, v, s)


@pytest.mark.parametrize("cache_name", ["k", "v"])
def test_cache_adapter_rejects_implicit_copy(cache_name, monkeypatch):
    q, k, v, slots = inputs()
    # Same [pages, page_size, heads, dim] shape as a valid pool, but the
    # first two strides cannot be merged into a view. flatten would copy it.
    incompatible = torch.empty(
        (16, 4, 1, 256), device=q.device, dtype=q.dtype
    ).transpose(0, 1)
    assert incompatible.shape == (4, 16, 1, 256)
    copied = incompatible.flatten(0, 1)
    assert copied.data_ptr() != incompatible.data_ptr()

    def forbidden(*args, **kwargs):
        raise AssertionError("invalid KV layout must fail before computation")

    monkeypatch.setattr(impl, "sparse_attention", forbidden)
    monkeypatch.setattr(backend_module, "qsa_sparse_attention", forbidden)
    monkeypatch.setattr(kernel, "qsa_sparse_attention_reference", forbidden)
    if cache_name == "k":
        k = incompatible
    else:
        v = incompatible
    with pytest.raises(RuntimeError, match="view size is not compatible"):
        backend_module._npu_sparse_attention(q, k, v, slots)


def test_kernel_errors_propagate(monkeypatch):
    def fail(*args, **kwargs):
        raise RuntimeError("intentional sparse attention failure")
    monkeypatch.setattr(impl, "sparse_attention", fail)
    with pytest.raises(RuntimeError, match="intentional"):
        backend_module._npu_sparse_attention(*inputs())


@pytest.mark.parametrize("mode", [ForwardMode.EXTEND, ForwardMode.TARGET_VERIFY, ForwardMode.DRAFT_EXTEND_V2])
@pytest.mark.parametrize("width", [2051, 2054, 2055])
def test_backend_kv_write_mapping_and_output_padding(mode, width, monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("NPU production must not call the Torch reference")

    monkeypatch.setattr(backend_module, "qsa_sparse_attention", forbidden)
    q, k, v, _ = inputs()
    backend = QwenSparseAttnBackend.__new__(QwenSparseAttnBackend)
    table = torch.arange(64, device="npu", dtype=torch.int32)[None, :]
    backend.forward_metadata = SimpleNamespace(
        is_cuda_graph=False, token_to_batch_idx=torch.zeros(3, device="npu", dtype=torch.int32),
        sequence_lengths=torch.tensor([64], device="npu", dtype=torch.int32),
        token_slot_table=table)
    class Pool:
        def set_kv_buffer(self, layer, loc, keys, values):
            k[loc.long()] = keys
            v[loc.long()] = values
        def get_key_buffer(self, layer_id):
            return k.reshape(4, 16, 1, 256)
        def get_value_buffer(self, layer_id):
            return v.reshape(4, 16, 1, 256)
    backend.token_to_kv_pool = Pool()
    layer = SimpleNamespace(tp_q_head_num=3, head_dim=256, layer_id=0, scaling=1 / 16)
    batch = SimpleNamespace(forward_mode=mode, out_cache_loc=torch.arange(4, device="npu", dtype=torch.int32))
    tokens = torch.full((3, width), -1, device="npu", dtype=torch.int32)
    tokens[:2, :4] = torch.arange(4, device="npu", dtype=torch.int32)
    new_k, new_v = torch.zeros_like(k[:4]), torch.ones_like(v[:4])
    out = backend.forward_extend(q, new_k, new_v, layer, batch, topk_indices=tokens)
    assert out.shape == (4, 3 * 256)
    torch.testing.assert_close(k[:4], new_k, atol=0, rtol=0)
    torch.testing.assert_close(v[:4], new_v, atol=0, rtol=0)
    expected = torch.zeros_like(out)
    expected[:2] = 1
    torch.testing.assert_close(out, expected, atol=0, rtol=0)


@pytest.mark.parametrize("fia", [False, True])
def test_actual_npu_pool_write_and_graph(fia, monkeypatch):
    from sglang.srt.hardware_backend.npu.memory_pool_npu import NPUMHATokenToKVPool
    from sglang.srt.mem_cache.kv_cache_dtype import configure_kv_cache_dtype

    # External runtime setting: test both real paged and FIA storage views.
    monkeypatch.setenv("ASCEND_USE_FIA", "True" if fia else "False")
    _, dtype = configure_kv_cache_dtype(
        server_args_kv_cache_dtype="auto", model=SimpleNamespace(quant_config=None),
        model_dtype=torch.bfloat16, is_draft_worker=False, is_dflash=False,
        speculative_draft_attention_backend="ascend")
    pool = NPUMHATokenToKVPool(
        size=128, page_size=64, dtype=dtype, head_num=1, head_dim=256,
        layer_num=2, device="npu", enable_memory_saver=False, enable_alt_stream=False)
    layer = SimpleNamespace(layer_id=1)
    k, v = pool.get_key_buffer(1), pool.get_value_buffer(1)
    assert k.dtype == v.dtype == torch.bfloat16
    assert k.storage_offset() > 0 and v.storage_offset() > 0
    flat_k = backend_module._flatten_qsa_kv_cache(k, "k")
    flat_v = backend_module._flatten_qsa_kv_cache(v, "v")
    assert flat_k.data_ptr() == k.data_ptr() and flat_v.data_ptr() == v.data_ptr()
    assert flat_k.stride() == flat_v.stride() == (256, 256, 1)
    q = torch.ones(2, 3, 256, device="npu", dtype=dtype)
    new_k = torch.zeros(2, 1, 256, device="npu", dtype=dtype)
    new_v = torch.ones_like(new_k)
    loc = torch.tensor([1, 64], device="npu", dtype=torch.int32)
    slots = torch.full((2, 2051), -1, device="npu", dtype=torch.int32)
    slots[1, :2] = loc
    def forward():
        pool.set_kv_buffer(layer, loc, new_k, new_v)
        return backend_module._npu_sparse_attention(
            q, pool.get_key_buffer(1), pool.get_value_buffer(1), slots
        )
    for _ in range(2): forward()
    torch.npu.synchronize()
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph):
        out = forward()
    for value in (3, 5):
        new_v[1] = value
        graph.replay()
        torch.npu.synchronize()
        expected = torch.zeros_like(q)
        expected[1] = (1 + value) / 2
        torch.testing.assert_close(out, expected, atol=0, rtol=0)
