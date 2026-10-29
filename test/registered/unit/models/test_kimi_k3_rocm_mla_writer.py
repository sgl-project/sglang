# SPDX-License-Identifier: Apache-2.0
"""KimiK3RocmMlaQCacheWriter against the split rope/concat/cache chain on gfx950."""

import pytest
import torch

pytest.importorskip("aiter")
from aiter import dtypes  # noqa: E402

from sglang.kernels.ops.attention import mla_q_cache_aiter_hip  # noqa: E402
from sglang.srt.models.kimi_k3_rocm_mla import KimiK3RocmMlaQCacheWriter  # noqa: E402
from sglang.srt.runtime_context import get_resources  # noqa: E402
from sglang.test.ci.ci_register import register_amd_ci  # noqa: E402

register_amd_ci(est_time=15, stage="stage-b", runner_config="1-gpu-small-amd-mi35x")

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


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__]))
