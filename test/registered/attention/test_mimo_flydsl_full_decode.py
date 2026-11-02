"""MiMo ordinary full decode must honor FlyDSL, including mixed DP eager paths."""

from types import SimpleNamespace

import pytest
import torch

from sglang.srt.layers.attention import aiter_utils
from sglang.srt.layers.quantization.fp8_kernel import fp8_dtype
from sglang.test.ci.ci_register import register_amd_ci

register_amd_ci(est_time=120, suite="stage-b-test-1-gpu-small-amd")
pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or not torch.version.hip,
    reason="MiMo FlyDSL full decode requires ROCm",
)


@pytest.mark.parametrize(
    "kv_dtype", [torch.bfloat16, fp8_dtype], ids=["bf16-kv", "fp8-kv"]
)
@pytest.mark.parametrize("batch_size", [1, 2])
@pytest.mark.parametrize(
    "configured_query_length", [1, 8], ids=["ordinary", "dflash-mixed"]
)
def test_full_decode_reference_and_graph_replay(
    monkeypatch, kv_dtype, batch_size, configured_query_length
):
    if not (aiter_utils.is_gfx942() or aiter_utils.is_gfx950()):
        pytest.skip("Validated MiMo FlyDSL architectures are gfx942 and gfx950")
    kernels = aiter_utils.load_flydsl_pa_decode_kernels()
    torch.manual_seed(5272)
    device = torch.device("cuda")
    page, heads, qdim, vdim, partitions = 64, 16, 192, 128, 16
    capacity = 65536 + page
    pages_per_seq = capacity // page
    key_scale = torch.tensor([0.75], device=device)
    value_scale = torch.tensor([0.5], device=device)
    k_linear = (torch.randn(batch_size, capacity, 1, qdim, device=device) * 0.2).to(
        kv_dtype
    )
    v_linear = (torch.randn(batch_size, capacity, 1, vdim, device=device) * 0.2).to(
        kv_dtype
    )
    vector = 8 if kv_dtype == torch.bfloat16 else 16
    k_cache = (
        k_linear.view(batch_size * pages_per_seq, page, 1, qdim // vector, vector)
        .permute(0, 2, 3, 1, 4)
        .contiguous()
    )
    v_cache = (
        v_linear.view(batch_size * pages_per_seq, page // vector, vector, 1, vdim)
        .permute(0, 3, 1, 4, 2)
        .contiguous()
    )
    table = torch.arange(
        batch_size * pages_per_seq, device=device, dtype=torch.int32
    ).view(batch_size, pages_per_seq)
    table = torch.cat((table, torch.zeros_like(table)))
    # Queries are views of fused QKV storage, so row stride exceeds Q width.
    fused = (
        torch.randn(
            batch_size, heads * qdim + qdim + vdim, device=device, dtype=torch.bfloat16
        )
        * 0.1
    )
    q = fused[:, : heads * qdim]
    output = torch.empty(
        (batch_size, heads * vdim), device=device, dtype=torch.bfloat16
    )
    scalar_numel = batch_size * partitions * configured_query_length * heads
    backend = SimpleNamespace(
        _use_flydsl_pa_decode=True,
        _flydsl_pa_decode_tile=kernels.pa_decode_tile,
        _flydsl_pa_decode_num_partitions=partitions,
        _flydsl_pa_decode_query_length=configured_query_length,
        _flydsl_pa_decode_pmax=torch.empty(scalar_numel, device=device),
        _flydsl_pa_decode_psum=torch.empty(scalar_numel, device=device),
        _flydsl_pa_decode_pout=torch.empty(
            scalar_numel * vdim, device=device, dtype=torch.bfloat16
        ),
        _flydsl_pa_decode_context_lengths=torch.empty(
            batch_size, device=device, dtype=torch.int32
        ),
        forward_metadata=SimpleNamespace(max_q_len=1, kv_indices=table),
        kv_cache_dtype=kv_dtype,
        page_size=page,
        k_scale=key_scale,
        v_scale=value_scale,
        device=device,
    )
    layer = SimpleNamespace(
        sliding_window_size=-1,
        tp_q_head_num=heads,
        tp_k_head_num=1,
        qk_head_dim=qdim,
        v_head_dim=vdim,
        scaling=qdim**-0.5,
        logit_cap=0.0,
        k_scale=key_scale.view(()),
        v_scale=value_scale.view(()),
    )
    batch = SimpleNamespace(
        batch_size=batch_size,
        seq_lens=torch.tensor([65, 129][:batch_size], device=device, dtype=torch.int64),
    )

    def unexpected_gluon(**kwargs):
        pytest.fail("FlyDSL ordinary full decode entered Gluon")

    monkeypatch.setattr(aiter_utils, "pa_decode_gluon", unexpected_gluon)

    def run():
        aiter_utils.forward_decode_vectorized_5d(
            backend,
            q,
            layer,
            batch,
            k_cache,
            v_cache,
            output,
            sinks=None,
        )

    def check(lengths):
        expected = []
        for index, length in enumerate(lengths):
            k = k_linear[index, :length, 0].float()
            v = v_linear[index, :length, 0].float()
            if kv_dtype != torch.bfloat16:
                k = k * key_scale
                v = v * value_scale
            scores = q[index].view(heads, qdim).float() @ k.T * layer.scaling
            expected.append(torch.softmax(scores, dim=-1) @ v)
        torch.testing.assert_close(
            output.view(batch_size, heads, vdim).float(),
            torch.stack(expected),
            rtol=0.005,
            atol=0.003,
        )
        assert backend._flydsl_pa_decode_context_lengths.tolist() == lengths

    aiter_utils.warmup_flydsl_full_decode(backend)
    # The startup warmup must materialize executables, rather than just factories.
    # Different runtime lengths, batch sizes and query strides must stay warm.
    import flydsl.compiler as flyc

    def unexpected_compile(*args, **kwargs):
        pytest.fail("Ordinary decode compiled after its startup warmup")

    monkeypatch.setattr(flyc, "compile", unexpected_compile)
    run()
    check([65, 129][:batch_size])
    eager = output.clone()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        run()
    graph.replay()
    torch.testing.assert_close(output, eager, rtol=0, atol=0)
    # Replay updates only device-resident lengths, crossing page/partition boundaries.
    for lengths in ([64, 128], [257, 8192], [65536, 65537]):
        lengths = lengths[:batch_size]
        batch.seq_lens.copy_(torch.tensor(lengths, device=device, dtype=torch.int64))
        output.fill_(float("nan"))
        graph.replay()
        check(lengths)
