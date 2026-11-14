# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

pytest.importorskip("aiter")
from aiter import dtypes  # noqa: E402

from sglang.kernels.ops.attention import mla_q_cache_aiter_hip  # noqa: E402
from sglang.srt.models.kimi_k3_rocm_mla import KimiK3RocmMlaQCacheWriter  # noqa: E402
from sglang.srt.runtime_context import get_resources  # noqa: E402
from sglang.test.ci.ci_register import register_amd_ci  # noqa: E402

register_amd_ci(est_time=35, stage="stage-b", runner_config="1-gpu-small-amd-mi35x")

pytestmark = pytest.mark.skipif(
    not mla_q_cache_aiter_hip.available(),
    reason="Kimi-K3 fused MLA Q/cache tests require AITER on gfx950",
)


def _inputs(tokens: int, cache_dtype=torch.bfloat16, output_dtype=None):
    torch.manual_seed(23)
    heads = 12
    q_nope = torch.randn(
        tokens, heads, 512, device="cuda", dtype=torch.bfloat16
    ).contiguous()
    q_pe = torch.randn(
        tokens, heads, 64, device="cuda", dtype=torch.bfloat16
    ).contiguous()
    k_nope = torch.randn(
        tokens, 1, 512, device="cuda", dtype=torch.bfloat16
    ).contiguous()
    k_pe = torch.randn(tokens, 1, 64, device="cuda", dtype=torch.bfloat16).contiguous()
    return {
        "q_nope": q_nope,
        "q_pe": q_pe,
        "k_nope": k_nope,
        "k_pe": k_pe,
        "kv_cache": torch.zeros(tokens + 4, 1, 576, device="cuda", dtype=cache_dtype),
        "slot_mapping": torch.arange(tokens, device="cuda", dtype=torch.int64),
        # The cos/sin tables have one identity row, so every position must be 0.
        "positions": torch.zeros(tokens, device="cuda", dtype=torch.int64),
        "k_scale": torch.ones(1, device="cuda", dtype=torch.float32),
        "q_scale": torch.ones(1, device="cuda", dtype=torch.float32),
        "cos_cache": torch.ones(1, 32, device="cuda", dtype=torch.bfloat16),
        "sin_cache": torch.zeros(1, 32, device="cuda", dtype=torch.bfloat16),
        "out": torch.empty(
            tokens,
            heads,
            576,
            device="cuda",
            dtype=cache_dtype if output_dtype is None else output_dtype,
        ),
    }


def _reference(values):
    return (
        torch.cat((values["q_nope"], values["q_pe"]), dim=-1).to(values["out"].dtype),
        torch.cat((values["k_nope"], values["k_pe"]), dim=-1).to(
            values["kv_cache"].dtype
        ),
    )


@pytest.mark.parametrize("tokens", [1, 4, 8, 64, 128, 256])
@pytest.mark.parametrize(
    ("cache_dtype", "output_dtype"),
    [
        (torch.bfloat16, torch.bfloat16),
        (dtypes.fp8, dtypes.fp8),
        (dtypes.fp8, torch.bfloat16),
    ],
)
def test_fused_mla_q_cache_identity_rope(tokens, cache_dtype, output_dtype):
    values = _inputs(tokens, cache_dtype=cache_dtype, output_dtype=output_dtype)
    expected_q, expected_k = _reference(values)
    actual = mla_q_cache_aiter_hip.run(**values)
    torch.cuda.synchronize()
    assert actual is values["out"]
    torch.testing.assert_close(actual.float(), expected_q.float(), atol=0.02, rtol=0.02)
    torch.testing.assert_close(
        values["kv_cache"][:tokens].float(),
        expected_k.float(),
        atol=0.02,
        rtol=0.02,
    )


def test_fused_mla_q_cache_skips_negative_slots():
    """A slot < 0 token skips only its KV cache write; its Q row must still be
    written, as the split chain and the Triton kernel write it."""
    values = _inputs(4)
    values["slot_mapping"][-2:] = -1
    values["out"].fill_(float("nan"))
    expected_q, expected_k = _reference(values)
    mla_q_cache_aiter_hip.run(**values)
    torch.cuda.synchronize()
    torch.testing.assert_close(values["out"], expected_q)
    torch.testing.assert_close(values["kv_cache"][:2], expected_k[:2])
    assert torch.count_nonzero(values["kv_cache"][2:]).item() == 0


def test_fused_mla_q_cache_uses_each_token_slot_across_page_boundaries():
    values = _inputs(8)
    slots = torch.tensor(
        [1, 127, 128, 129, 255, 256, 257, 299],
        dtype=torch.int64,
        device="cuda",
    )
    values["slot_mapping"] = slots
    values["kv_cache"] = torch.zeros(304, 1, 576, dtype=torch.bfloat16, device="cuda")
    expected_q, expected_k = _reference(values)

    actual = mla_q_cache_aiter_hip.run(**values)
    torch.cuda.synchronize()

    torch.testing.assert_close(actual, expected_q)
    torch.testing.assert_close(values["kv_cache"][slots], expected_k)


def test_fused_mla_q_cache_graph_replay_uses_changed_inputs():
    values = _inputs(64)
    mla_q_cache_aiter_hip.run(**values)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        mla_q_cache_aiter_hip.run(**values)
    graph.replay()
    torch.cuda.synchronize()
    first = values["out"].clone()
    values["q_nope"].copy_(torch.randn_like(values["q_nope"]))
    values["q_pe"].copy_(torch.randn_like(values["q_pe"]))
    values["k_nope"].copy_(torch.randn_like(values["k_nope"]))
    values["k_pe"].copy_(torch.randn_like(values["k_pe"]))
    values["kv_cache"].zero_()
    graph.replay()
    torch.cuda.synchronize()
    expected_q, expected_k = _reference(values)
    assert not torch.equal(first, values["out"])
    torch.testing.assert_close(values["out"], expected_q)
    torch.testing.assert_close(values["kv_cache"][:64], expected_k)


def test_fused_mla_q_cache_support_is_narrow():
    values = _inputs(1)
    assert mla_q_cache_aiter_hip.covered(**values)
    values["positions"] = values["positions"].to(torch.float32)
    assert not mla_q_cache_aiter_hip.covered(**values)
    with pytest.raises(ValueError, match="unsupported inputs"):
        mla_q_cache_aiter_hip.run(**values)


def test_fused_mla_q_cache_writes_into_padded_out():
    values = _inputs(4, cache_dtype=dtypes.fp8, output_dtype=dtypes.fp8)
    heads = values["q_nope"].shape[1]
    padded = torch.empty((4, 16, 576), device="cuda", dtype=dtypes.fp8)
    padded.view(torch.uint8).fill_(7)
    values["out"] = padded
    assert mla_q_cache_aiter_hip.covered(**values)
    actual = mla_q_cache_aiter_hip.run(**values)
    torch.cuda.synchronize()
    expected_q, _ = _reference(values)
    torch.testing.assert_close(
        actual[:, :heads].float(), expected_q.float(), atol=0.02, rtol=0.02
    )
    assert torch.equal(actual[:, heads:], padded[:, heads:])


def _writer_inputs(values):
    return dict(
        q_nope=values["q_nope"],
        q_pe=values["q_pe"],
        k_nope=values["k_nope"],
        k_pe=values["k_pe"],
        out_cache_loc=values["slot_mapping"],
        kv_cache=values["kv_cache"],
        k_scale_float=1.0,
        attention_backend="aiter",
        is_target_verify=False,
    )


@pytest.fixture(params=["aiter", "triton"])
def writer(request, monkeypatch):
    if request.param == "triton":
        # The writer falls back to the AITER Triton kernel off gfx950.
        monkeypatch.setattr(mla_q_cache_aiter_hip, "available", lambda: False)
    return KimiK3RocmMlaQCacheWriter(kv_lora_rank=512, qk_rope_head_dim=64).cuda()


@pytest.mark.parametrize("cache_dtype", [torch.bfloat16, dtypes.fp8])
def test_writer_matches_split_chain_bytes(writer, cache_dtype):
    """The split chain builds cat(q_nope, q_pe) and plainly casts cat(k_nope, k_pe);
    K3's KV scale is 1, so the fused write must store the same bytes."""
    values = _inputs(33, cache_dtype=cache_dtype)
    values["slot_mapping"] = torch.randperm(37, device="cuda")[:33]
    q, k = writer.write(**_writer_inputs(values))
    torch.cuda.synchronize()
    expected_q = torch.cat((values["q_nope"], values["q_pe"]), dim=-1)
    expected_k = torch.cat((values["k_nope"], values["k_pe"]), dim=-1).to(cache_dtype)
    assert torch.equal(q, expected_q)
    assert k.shape == expected_k.shape
    cache = values["kv_cache"].view(torch.uint8)
    assert torch.equal(cache[values["slot_mapping"]], expected_k.view(torch.uint8))
    untouched = torch.ones(cache.shape[0], dtype=torch.bool, device="cuda")
    untouched[values["slot_mapping"]] = False
    assert torch.count_nonzero(cache[untouched]).item() == 0


@pytest.mark.parametrize("tokens", [1, 8, 33])
def test_writer_fp8_q_out_matches_quant_then_pad(tokens):
    """With the backend's fp8 query buffer, the writer must store the bytes the asm
    verify used to build itself: static FP8 quant of cat(q_nope, q_pe), which
    saturates at +-448, copied into a zeroed 16-head buffer. The q_pe part
    saturates only with an AITER that clamps its rope output, so it stays in range."""
    from sglang.kernels.ops.quantization.fp8_kernel import scaled_fp8_quant

    writer = KimiK3RocmMlaQCacheWriter(kv_lora_rank=512, qk_rope_head_dim=64).cuda()
    values = _inputs(tokens, cache_dtype=dtypes.fp8)
    values["q_nope"][:, 0, :4] = 1.0e4
    scale = torch.ones(1, device="cuda", dtype=torch.float32)
    buf = torch.zeros(64, 16, 576, device="cuda", dtype=dtypes.fp8)[:tokens]

    q, _ = writer.write(**_writer_inputs(values), get_fp8_q_out=lambda: (buf, scale))
    torch.cuda.synchronize()

    bf16_q = torch.cat((values["q_nope"], values["q_pe"]), dim=-1)
    quant, _ = scaled_fp8_quant(bf16_q.reshape(tokens, -1), scale)
    expected = torch.zeros(tokens, 16, 576, device="cuda", dtype=dtypes.fp8)
    expected[:, :12] = quant.view(tokens, 12, 576)
    assert q.data_ptr() == buf.data_ptr()
    assert torch.equal(q.view(torch.uint8), expected.view(torch.uint8))


def test_writer_graph_survives_a_larger_eager_batch(writer):
    """A captured step keeps reading the zero positions it saw,
    even after a larger eager batch replaces the shared buffer."""
    # Start from no shared buffer so the larger batch below has to replace it.
    get_resources().buffers.pop("kimi_k3_mla_zero_positions", None)
    small = _inputs(4)
    writer.write(**_writer_inputs(small))
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        q, _ = writer.write(**_writer_inputs(small))
    big = _inputs(1000)
    writer.write(**_writer_inputs(big))
    torch.cuda.synchronize()
    # Refill freed small blocks so a dropped buffer would hold non-zero positions.
    junk = [  # noqa: F841
        torch.full((64,), 1, dtype=torch.int64, device="cuda") for _ in range(256)
    ]
    for name in ("q_nope", "q_pe", "k_nope", "k_pe"):
        small[name].copy_(torch.randn_like(small[name]))
    small["kv_cache"].zero_()
    graph.replay()
    torch.cuda.synchronize()
    expected_q, expected_k = _reference(dict(small, out=q))
    assert torch.equal(q, expected_q)
    assert torch.equal(small["kv_cache"][:4], expected_k)


def test_writer_reads_a_strided_out_cache_loc(writer):
    """A non-contiguous out_cache_loc must still land each token in its own slot;
    a kernel reading it at stride 1 would write the wrong rows."""
    values = _inputs(16)
    values["kv_cache"] = torch.zeros(32, 1, 576, device="cuda", dtype=torch.bfloat16)
    values["slot_mapping"] = torch.arange(32, device="cuda", dtype=torch.int64)[::2]
    q, _ = writer.write(**_writer_inputs(values))
    torch.cuda.synchronize()
    _, expected_k = _reference(dict(values, out=q))
    assert torch.equal(values["kv_cache"][values["slot_mapping"]], expected_k)


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__]))
