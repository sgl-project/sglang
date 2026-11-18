# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

pytest.importorskip("aiter")
from aiter import dtypes  # noqa: E402

from sglang.kernels.ops.attention import mla_q_cache_aiter_hip  # noqa: E402
from sglang.srt.models.kimi_k3_rocm_mla import KimiK3RocmMlaQCacheWriter  # noqa: E402
from sglang.srt.runtime_context import get_resources  # noqa: E402
from sglang.test.ci.ci_register import register_amd_ci  # noqa: E402
from sglang.test.kernels.mla_q_cache import (  # noqa: E402
    mla_q_cache_inputs,
    mla_q_cache_reference,
)

register_amd_ci(est_time=10, stage="base-b", runner_config="1-gpu-small-amd-mi35x")

pytestmark = pytest.mark.skipif(
    not mla_q_cache_aiter_hip.available(),
    reason="Kimi-K3 fused MLA Q/cache tests require AITER on gfx950",
)


def _writer_inputs(values):
    return dict(
        q_nope=values["q_nope"],
        q_pe=values["q_pe"],
        k_nope=values["k_nope"],
        k_pe=values["k_pe"],
        slot_mapping=values["slot_mapping"],
        kv_cache=values["kv_cache"],
        k_scale_float=1.0,
        attention_backend="aiter",
        is_target_verify=False,
    )


@pytest.fixture
def writer():
    return KimiK3RocmMlaQCacheWriter(kv_lora_rank=512, qk_rope_head_dim=64)


@pytest.mark.parametrize("cache_dtype", [torch.bfloat16, dtypes.fp8])
def test_writer_matches_split_chain_bytes(writer, cache_dtype):
    """The split chain builds cat(q_nope, q_pe) and plainly casts cat(k_nope, k_pe);
    K3's KV scale is 1, so the fused write must store the same bytes."""
    values = mla_q_cache_inputs(33, cache_dtype=cache_dtype)
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
    get_resources().buffers.pop("kimi_k3_mla_identity_rope", None)
    small = mla_q_cache_inputs(4)
    writer.write(**_writer_inputs(small))
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        q, _ = writer.write(**_writer_inputs(small))
    big = mla_q_cache_inputs(1000)
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
    expected_q, expected_k = mla_q_cache_reference(dict(small, out=q))
    assert torch.equal(q, expected_q)
    assert torch.equal(small["kv_cache"][:4], expected_k)


def test_capture_never_allocates_the_zero_positions(writer):
    """A capture the eager warmup did not size must not allocate the shared buffers
    inside the graph; the writer declines and that step keeps the split chain."""
    get_resources().buffers.pop("kimi_k3_mla_identity_rope", None)
    values = mla_q_cache_inputs(4)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        result = writer.write(**_writer_inputs(values))
    assert result is None
    rope = get_resources().buffers["kimi_k3_mla_identity_rope"]
    assert rope.positions is None and rope.tables is None


def test_writer_raises_on_a_strided_slot_mapping(writer):
    """The kernel reads slot_mapping at stride 1, so a strided one must fail loudly
    rather than land tokens in the wrong slots; the caller makes it contiguous."""
    values = mla_q_cache_inputs(16)
    values["kv_cache"] = torch.zeros(32, 1, 576, device="cuda", dtype=torch.bfloat16)
    values["slot_mapping"] = torch.arange(32, device="cuda", dtype=torch.int64)[::2]
    with pytest.raises(ValueError):
        writer.write(**_writer_inputs(values))
    torch.cuda.synchronize()
    assert torch.count_nonzero(values["kv_cache"]).item() == 0


@pytest.mark.parametrize("cache", ["fp8_e5m2", "paged_4d"])
def test_writer_falls_back_on_caches_the_kernel_does_not_take(writer, cache):
    """Caches the HIP op cannot write must reach the split chain unlaunched,
    not fail the decode step or graph capture."""
    values = mla_q_cache_inputs(4)
    if cache == "fp8_e5m2":
        values["kv_cache"] = torch.zeros(
            8, 1, 576, device="cuda", dtype=torch.float8_e5m2
        )
    else:
        values["kv_cache"] = torch.zeros(
            8, 1, 1, 576, device="cuda", dtype=torch.bfloat16
        )
    assert writer.write(**_writer_inputs(values)) is None
    torch.cuda.synchronize()
    assert torch.count_nonzero(values["kv_cache"].view(torch.uint8)).item() == 0


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__]))
