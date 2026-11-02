import pytest
import torch

import sglang.kernels.ops.attention.extend_attention as attention
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=40, stage="base-b-kernel-unit", runner_config="1-gpu-large")

pytestmark = pytest.mark.skipif(
    not attention._is_rtx_pro_6000,
    reason="RTX PRO 6000 launch tuning requires its qualified hardware",
)


def inputs(prefix, queries, cache_dtype=torch.float8_e4m3fn, kv_heads=2):
    torch.manual_seed(78013 + prefix + queries)
    q = torch.randn((queries, 32, 192), device="cuda", dtype=torch.bfloat16)
    k = torch.randn((queries, kv_heads, 192), device="cuda", dtype=torch.bfloat16)
    v = torch.randn((queries, kv_heads, 128), device="cuda", dtype=torch.bfloat16)
    kb = torch.randn((prefix, kv_heads, 192), device="cuda", dtype=torch.bfloat16).to(
        cache_dtype
    )
    vb = torch.randn((prefix, kv_heads, 128), device="cuda", dtype=torch.bfloat16).to(
        cache_dtype
    )
    qo = torch.tensor([0, queries], device="cuda", dtype=torch.int32)
    ki = torch.tensor([0, prefix], device="cuda", dtype=torch.int32)
    indices = torch.arange(prefix, device="cuda", dtype=torch.int32)
    return q, k, v, kb, vb, qo, ki, indices


def execute(data, **kwargs):
    q, k, v, kb, vb, qo, ki, indices = data
    output = torch.empty((q.shape[0], 32, 128), device="cuda", dtype=torch.bfloat16)
    kwargs.setdefault("extend_prefix_lens_cpu", [kb.shape[0]])
    attention.extend_attention_fwd(
        q,
        k,
        v,
        output,
        kb,
        vb,
        qo,
        ki,
        indices,
        None,
        True,
        None,
        q.shape[0],
        1.0,
        1.0,
        sm_scale=192**-0.5,
        **kwargs,
    )
    return output


@pytest.mark.parametrize("prefix", [0, 257, 8191, 8192, 8193])
@pytest.mark.parametrize("queries", [512, 513])
def test_mimo_global_fp8_tiles_match_original_kernel(monkeypatch, prefix, queries):
    data = inputs(prefix, queries)
    with monkeypatch.context() as original:
        original.setattr(attention, "_is_rtx_pro_6000", False)
        reference = execute(data)
    result = execute(data)
    torch.cuda.synchronize()
    assert torch.isfinite(result).all()
    difference = result.float() - reference.float()
    relative_rmse = (
        difference.square().mean().sqrt() / reference.float().square().mean().sqrt()
    )
    assert relative_rmse.item() < 0.005
    torch.testing.assert_close(result, reference, atol=0.002, rtol=0.02)
    if prefix < 8192:
        assert torch.equal(result, reference)


@pytest.mark.parametrize(
    "kind", ["short", "bf16", "swa", "other-kv-heads", "missing-prefix-metadata"]
)
def test_other_attention_paths_keep_the_original_result(monkeypatch, kind):
    data = inputs(
        8192,
        127 if kind == "short" else 513,
        cache_dtype=torch.bfloat16 if kind == "bf16" else torch.float8_e4m3fn,
        kv_heads=4 if kind == "other-kv-heads" else 2,
    )
    kwargs = {}
    if kind == "swa":
        kwargs = {
            "sliding_window_size": 128,
            "sinks": torch.linspace(-2, 2, 32, device="cuda"),
        }
    elif kind == "missing-prefix-metadata":
        kwargs = {"extend_prefix_lens_cpu": None}
    with monkeypatch.context() as original:
        original.setattr(attention, "_is_rtx_pro_6000", False)
        reference = execute(data, **kwargs)
    result = execute(data, **kwargs)
    torch.cuda.synchronize()
    assert torch.equal(result, reference)


def test_captured_prefill_keeps_original_tiles_for_shorter_replay(monkeypatch):
    data = inputs(8192, 512)
    with monkeypatch.context() as original:
        original.setattr(attention, "_is_rtx_pro_6000", False)
        reference = execute(data)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured = execute(data)
    graph.replay()
    torch.cuda.synchronize()
    assert torch.equal(captured, reference)

    # Replay does not re-evaluate CPU prefix metadata. A graph must therefore
    # retain the original reduction partition when its device lengths shrink.
    data[6][1] = 257
    with monkeypatch.context() as original:
        original.setattr(attention, "_is_rtx_pro_6000", False)
        reference = execute(data)
    graph.replay()
    torch.cuda.synchronize()
    assert torch.equal(captured, reference)
