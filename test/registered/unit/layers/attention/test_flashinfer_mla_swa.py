# SPDX-License-Identifier: Apache-2.0
import sys
from types import SimpleNamespace

import pytest
import torch

from sglang.srt.layers.attention.swa_mla import FlashInferMLASWA
from sglang.test.ci.ci_register import register_cpu_ci, register_cuda_ci

register_cpu_ci(est_time=12, suite="base-a-test-cpu")
register_cuda_ci(est_time=30, stage="base-b-kernel-unit", runner_config="1-gpu-large")


def test_cpu_keeps_fallback():
    assert FlashInferMLASWA.create("cpu") is None


def test_older_flashinfer_keeps_fallback(monkeypatch):
    import flashinfer.mla as mla

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda device: (9, 0))
    monkeypatch.setattr(
        torch.cuda,
        "get_device_properties",
        lambda device: SimpleNamespace(shared_memory_per_block_optin=227328),
    )
    monkeypatch.setattr(mla, "batch_decode_with_kv_cache_mla_swa", None, raising=False)
    assert FlashInferMLASWA.create("cuda:0") is None


@pytest.mark.parametrize(
    "heads,swa_heads,cp,expected",
    [(128, 64, 1, True), (16, 8, 1, False), (128, 64, 2, False), (64, None, 1, False)],
)
def test_wrapper_prepares_only_qualified_domain(
    monkeypatch, heads, swa_heads, cp, expected
):
    from sglang.srt.layers.attention.dots_hybrid_backend import DotsSWAMLAAttnBackend

    marker = object()
    calls = []

    def create(device):
        calls.append(device)
        return marker

    monkeypatch.setattr(FlashInferMLASWA, "create", create)
    inner = SimpleNamespace(
        num_attention_heads=heads,
        attn_cp_size=cp,
        device="cuda:0",
        token_to_kv_pool=object(),
        req_to_token_pool=object(),
        kv_index_translator=object(),
    )
    wrapper = DotsSWAMLAAttnBackend(inner, num_swa_heads=swa_heads)
    assert (wrapper._mla_swa is marker) == expected
    assert bool(calls) == expected


@pytest.mark.parametrize("attn_tp_size,expected", [(1, 64), (8, 8)])
def test_target_uses_swa_heads_not_full_attention_heads(
    monkeypatch, attn_tp_size, expected
):
    from sglang.srt import runtime_context
    from sglang.srt.layers.attention import dots_hybrid_backend as dots

    monkeypatch.setattr(
        runtime_context,
        "get_parallel",
        lambda: SimpleNamespace(attn_tp_size=attn_tp_size),
    )
    calls = []
    monkeypatch.setattr(
        dots,
        "DotsSWAMLAAttnBackend",
        lambda backend, **kwargs: calls.append(kwargs["num_swa_heads"]),
    )
    runner = SimpleNamespace(
        model_config=SimpleNamespace(
            is_draft_model=False,
            hf_text_config=SimpleNamespace(
                num_attention_heads=128, swa_num_attention_heads=64, index_topk=None
            ),
        )
    )
    dots.wrap_dots_attention_backend(runner, object())
    assert calls == [expected]


@pytest.mark.parametrize("kind", ["direct", "dots", "tbo", "absent"])
def test_startup_autotune_detects_native_adapter(monkeypatch, kind):
    from sglang.srt.model_executor.runner import flashinfer_autotune as policy

    config = SimpleNamespace(
        kernel=SimpleNamespace(disable_flashinfer_autotune=False),
        deterministic=SimpleNamespace(enable_deterministic_inference=False),
        moe=SimpleNamespace(moe_runner_backend=None, moe_a2a_backend=None),
    )
    monkeypatch.setattr(policy, "get_exec", lambda: config)
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda *args: (9, 0))
    swa = SimpleNamespace(_mla_swa=object() if kind != "absent" else None)
    backend = swa if kind in ("direct", "absent") else SimpleNamespace(swa_backend=swa)
    if kind == "tbo":
        backend = SimpleNamespace(primary=backend)
    runner = SimpleNamespace(
        device="cuda",
        model=torch.nn.Module(),
        model_config=SimpleNamespace(quantization=None),
        attn_backend=backend,
        spec_algorithm=SimpleNamespace(is_speculative=lambda: False),
    )
    assert policy.should_run_flashinfer_autotune(runner) == (kind != "absent")


def test_missing_aot_is_recoverable_but_other_errors_propagate(monkeypatch):
    import flashinfer.mla as mla
    from flashinfer.jit import MissingJITCacheError

    if not hasattr(mla, "batch_decode_with_kv_cache_mla_swa"):
        pytest.skip("FlashInfer MLA+SWA API unavailable")
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda device: (9, 0))
    monkeypatch.setattr(
        torch.cuda,
        "get_device_properties",
        lambda device: SimpleNamespace(shared_memory_per_block_optin=227328),
    )
    for name in ("empty", "zeros", "ones"):
        monkeypatch.setattr(torch, name, lambda *args, **kwargs: object())

    def missing(*args, **kwargs):
        raise MissingJITCacheError("JIT disabled, artifact absent")

    monkeypatch.setattr(mla, "batch_decode_with_kv_cache_mla_swa", missing)
    assert FlashInferMLASWA.create("cuda:0") is None

    def broken(*args, **kwargs):
        raise RuntimeError("loader or CUDA execution error")

    monkeypatch.setattr(mla, "batch_decode_with_kv_cache_mla_swa", broken)
    with pytest.raises(RuntimeError, match="execution error"):
        FlashInferMLASWA.create("cuda:0")


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("sq", [1, 4])
def test_public_api_adapter_and_fallback(monkeypatch, tmp_path, sq):
    if torch.cuda.get_device_capability() != (9, 0):
        pytest.skip("SM90 required")
    adapter = FlashInferMLASWA.create(torch.device("cuda", torch.cuda.current_device()))
    if adapter is None:
        pytest.skip("FlashInfer MLA+SWA API unavailable")
    import flashinfer

    q = torch.randn(1, sq, 64, 1088, device="cuda", dtype=torch.bfloat16)
    kv = torch.randn(17 * 64, 1, 1088, device="cuda", dtype=torch.bfloat16)
    table = torch.arange(17, device="cuda", dtype=torch.int32)[None]
    lengths = torch.tensor([1055], device="cuda", dtype=torch.int32)
    layer = SimpleNamespace(tp_k_head_num=1, scaling=0.0625, is_cross_attention=False)
    cache = tmp_path / "mla_swa.json"
    with flashinfer.autotune(True, cache=str(cache)):
        adapter.forward(q, kv, table, lengths, layer, 1088, 1024, 513)
    assert cache.is_file()
    assert "mla_swa_sm90" in cache.read_text()
    expected = adapter.forward(q, kv, table, lengths, layer, 1088, 1024, 513)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        actual = adapter.forward(q, kv, table, lengths, layer, 1088, 1024, 513)
    graph.replay()
    torch.cuda.synchronize()
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    marker = torch.empty(0, device="cuda")
    calls = []

    def fallback(*args):
        calls.append(args)
        return marker

    monkeypatch.setattr(
        "sglang.srt.layers.attention.swa_mla.forward_dense_kvlora_swa_torch_fallback",
        fallback,
    )
    assert (
        adapter.forward(q[:, :, :8], kv, table, lengths, layer, 1088, 1024, 513)
        is marker
    )
    assert len(calls) == 1

    def fail(*args, **kwargs):
        raise RuntimeError("CUDA execution errors must propagate")

    adapter.run = fail
    with pytest.raises(RuntimeError, match="must propagate"):
        adapter.forward(q, kv, table, lengths, layer, 1088, 1024, 513)
    assert len(calls) == 1


if __name__ == "__main__":
    sys.exit(pytest.main([__file__]))
